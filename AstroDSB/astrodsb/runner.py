from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, lr_scheduler
from torch_ema import ExponentialMovingAverage
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

from datasets_mod import denormalize_sample, normalize_array

from . import util
from .diffusion import Diffusion
from .network import AstroNet


MAG_CHANNEL_SCHEMAS = {
    # Assumption from Xu et al. (2025) / AstroDSB appendix figure ordering:
    # [column density, angle dispersion, LOS velocity dispersion, projected B-field]
    "default_xu2025": {
        "column_density": 0,
        "angle_dispersion": 1,
        "los_velocity_dispersion": 2,
        "projected_b_field": 3,
    }
}

DEFAULT_UNIT_INTERVAL_STATS = type(
    "NormStats",
    (),
    {"mode": "strict_unit_interval", "min_value": 0.0, "max_value": 1.0},
)()


def crop_image(img, patch_size=128, step=8):
    h, w = img.shape[:2]
    if h < patch_size or w < patch_size:
        raise ValueError(f"Image must be at least patch-sized, got {(h, w)} for patch_size={patch_size}")

    y_positions = list(range(0, h - patch_size + 1, step))
    x_positions = list(range(0, w - patch_size + 1, step))
    if y_positions[-1] != h - patch_size:
        y_positions.append(h - patch_size)
    if x_positions[-1] != w - patch_size:
        x_positions.append(w - patch_size)

    patches = []
    positions = []
    for y in y_positions:
        for x in x_positions:
            patches.append(img[y : y + patch_size, x : x + patch_size])
            positions.append((x, y))
    return patches, positions, (h, w)


def merge_patches(patches, positions, img_size, patch_size=128):
    h, w = img_size
    merged_img = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for patch, (x, y) in zip(patches, positions):
        merged_img[y:y + patch_size, x:x + patch_size] += patch
        count_map[y:y + patch_size, x:x + patch_size] += 1

    count_map[count_map == 0] = 1
    return merged_img / count_map


def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = torch.linspace(
        linear_start ** 0.5,
        linear_end ** 0.5,
        n_timestep,
        dtype=torch.float64,
    ) ** 2
    return betas.numpy()


def build_optimizer_sched(opt, net, log, checkpoint=None):
    optimizer = AdamW(net.parameters(), lr=opt.lr, weight_decay=opt.l2_norm)
    log.info(f"[Opt] Built AdamW optimizer lr={opt.lr} weight_decay={opt.l2_norm}")

    sched = None
    if opt.lr_gamma < 1.0:
        sched = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)
        log.info(f"[Opt] Built StepLR step_size={opt.lr_step} gamma={opt.lr_gamma}")

    if checkpoint is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info("[Opt] Loaded optimizer state from checkpoint")
        if sched is not None and checkpoint.get("sched") is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info("[Opt] Loaded scheduler state from checkpoint")

    return optimizer, sched


def _format_duration(seconds):
    seconds = max(float(seconds), 0.0)
    minutes, secs = divmod(seconds, 60.0)
    hours, minutes = divmod(minutes, 60.0)
    if hours >= 1:
        return f"{int(hours)}h{int(minutes):02d}m{int(secs):02d}s"
    if minutes >= 1:
        return f"{int(minutes)}m{int(secs):02d}s"
    return f"{secs:.1f}s"


def _format_memory_bytes(num_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)}{unit}"
    return f"{value:.1f}{unit}"


def _tensor_stats(tensor):
    detached = tensor.detach()
    return (
        f"shape={tuple(detached.shape)} "
        f"min={detached.min().item():+.4f} "
        f"max={detached.max().item():+.4f} "
        f"mean={detached.mean().item():+.4f}"
    )


class Runner:
    def __init__(self, opt, log, save_opt=True):
        super().__init__()
        self.opt = opt
        self.log = log

        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as handle:
                pickle.dump(opt, handle)
            log.info(f"Saved options pickle to {opt_pkl_path}")

        betas = make_beta_schedule(
            n_timestep=opt.interval,
            linear_end=opt.beta_max / opt.interval,
        )
        betas = np.concatenate([betas[: opt.interval // 2], np.flip(betas[: opt.interval // 2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built AstroDSB diffusion with steps={len(betas)}")

        noise_levels = (
            torch.linspace(opt.t0, opt.T, opt.interval, dtype=torch.float32, device=opt.device)
            * opt.interval
        )
        self.net = AstroNet(
            log,
            noise_levels,
            input_channels=opt.target_channels,
            cond_channels=opt.observation_channels,
            image_size=opt.image_size,
            model_channels=opt.model_channels,
            num_res_blocks=opt.num_res_blocks,
        )
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)

        checkpoint = None
        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint["net"])
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ckpt] Loaded network and EMA from {opt.load}")

        self.net.to(opt.device)
        self.ema.to(opt.device)
        self.checkpoint = checkpoint

    def _effective_eval_nfe(self, opt):
        return opt.eval_nfe if opt.eval_nfe is not None else opt.interval - 1

    def _effective_eval_batch_size(self, opt, *, training_eval):
        if getattr(opt, "eval_batch_size", None) is not None:
            return opt.eval_batch_size
        return 4 if training_eval else opt.microbatch

    def _effective_eval_max_samples(self, opt, *, training_eval):
        configured = getattr(opt, "eval_max_samples", None)
        if configured is not None:
            return configured
        return 512 if training_eval else None

    def _resolve_eval_nfe(self, opt, *, training_eval):
        configured = getattr(opt, "eval_nfe", None)
        if configured is not None:
            return configured
        return 49 if training_eval else opt.interval - 1

    def _prepare_eval_dataset(self, opt, dataset, *, training_eval):
        full_size = len(dataset)
        max_samples = self._effective_eval_max_samples(opt, training_eval=training_eval)
        subset_seed = getattr(opt, "eval_subset_seed", 42)
        if not training_eval or max_samples is None or max_samples <= 0 or full_size <= max_samples:
            return dataset, {
                "effective_samples": full_size,
                "full_samples": full_size,
                "subset_enabled": False,
                "subset_seed": subset_seed,
            }

        rng = np.random.default_rng(subset_seed)
        indices = rng.permutation(full_size)[:max_samples].tolist()
        return Subset(dataset, indices), {
            "effective_samples": len(indices),
            "full_samples": full_size,
            "subset_enabled": True,
            "subset_seed": subset_seed,
        }

    def _gpu_memory_summary(self):
        if not torch.cuda.is_available():
            return "cuda=off"
        device = torch.device(self.opt.device)
        if device.type != "cuda":
            return "cuda=off"
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_allocated = torch.cuda.max_memory_allocated(device)
        return (
            f"cuda_alloc={_format_memory_bytes(allocated)} "
            f"cuda_reserved={_format_memory_bytes(reserved)} "
            f"cuda_peak={_format_memory_bytes(max_allocated)}"
        )

    def _log_bridge_tensor_health(self, prefix, tensor):
        if torch.isnan(tensor).any():
            self.log.warning(f"{prefix} detected NaN values")
        if torch.isinf(tensor).any():
            self.log.warning(f"{prefix} detected Inf values")

    def _assert_batch(self, y, x1):
        if y.ndim != 4 or x1.ndim != 4:
            raise ValueError(f"Expected y/x1 to be rank-4, got {y.shape} and {x1.shape}")
        if x1.shape[1] != self.opt.target_channels:
            raise ValueError(
                f"x1 channel mismatch: expected {self.opt.target_channels}, got {x1.shape[1]}"
            )
        if y.shape[1] != self.opt.observation_channels:
            raise ValueError(
                f"y channel mismatch: expected {self.opt.observation_channels}, got {y.shape[1]}"
            )
        if y.shape[-2:] != x1.shape[-2:]:
            raise ValueError(f"Spatial mismatch between y {y.shape} and x1 {x1.shape}")

    def _project_observation(self, y):
        if self.opt.task == "density":
            if y.shape[1] != self.opt.target_channels:
                raise ValueError(
                    f"Density task requires observation and target channel counts to match, got {y.shape[1]} and {self.opt.target_channels}"
                )
            return y

        if self.opt.task != "mag":
            raise ValueError(f"Unsupported AstroDSB task for bridge endpoint projection: {self.opt.task}")
        if self.opt.target_channels != 1:
            raise ValueError(
                f"Magnetic bridge endpoint adapter expects a scalar target, got {self.opt.target_channels} channels"
            )
        if y.shape[1] < 4:
            raise ValueError(
                f"Magnetic task expects at least 4 observation channels, got {y.shape[1]}"
            )

        if self.opt.mag_channel_schema not in MAG_CHANNEL_SCHEMAS:
            raise ValueError(f"Unsupported magnetic observation schema: {self.opt.mag_channel_schema}")
        channel_map = MAG_CHANNEL_SCHEMAS[self.opt.mag_channel_schema]
        if self.opt.mag_bridge_mode not in channel_map:
            raise ValueError(f"Unsupported magnetic bridge mode: {self.opt.mag_bridge_mode}")

        return y[:, channel_map[self.opt.mag_bridge_mode] : channel_map[self.opt.mag_bridge_mode] + 1]

    def _sample_noisy_observation(self, y):
        y_bridge = self._project_observation(y)
        eps = torch.randn_like(y_bridge) * self.opt.obs_noise_scale
        y_noisy = y_bridge - eps
        return y_bridge, y_noisy, eps

    def compute_target(self, step, y_noisy, x1):
        sigma_t = self.diffusion.get_std_fwd(step, xdim=x1.shape[1:])
        return ((y_noisy - x1) / sigma_t).detach()

    def compute_pred_x1(self, step, y_noisy, net_out, clip_denoise=False):
        sigma_t = self.diffusion.get_std_fwd(step, xdim=net_out.shape[1:])
        pred_x1 = y_noisy - sigma_t * net_out
        if clip_denoise:
            pred_x1 = pred_x1.clamp(-1.0, 1.0)
        return pred_x1

    def _create_train_model(self):
        if not self.opt.distributed:
            return self.net
        return DDP(self.net, device_ids=[self.opt.local_rank], output_device=self.opt.local_rank)

    def _get_target_denorm_stats(self):
        stats = getattr(self.opt, "target_normalization", None)
        if stats is None:
            return None
        return type("NormStats", (), stats)()

    def _get_observation_norm_stats(self):
        stats = getattr(self.opt, "observation_normalization", None)
        if stats is None:
            return DEFAULT_UNIT_INTERVAL_STATS
        return type("NormStats", (), stats)()

    def _build_train_dataloader(self, dataset):
        if self.opt.distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.opt.global_size,
                rank=self.opt.global_rank,
                shuffle=True,
                drop_last=True,
            )
            return util.build_dataloader_with_sampler(
                dataset,
                batch_size=self.opt.microbatch,
                sampler=sampler,
                num_workers=self.opt.num_workers,
                drop_last=True,
            ), sampler
        return util.build_dataloader(
            dataset,
            batch_size=self.opt.microbatch,
            num_workers=self.opt.num_workers,
            shuffle=True,
            drop_last=True,
        ), None

    def _build_eval_dataloader(self, dataset, *, batch_size):
        return util.build_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=self.opt.num_workers,
            shuffle=False,
            drop_last=False,
        )

    def _next_batch(self, loader):
        batch = next(loader)
        y = batch["y"].to(self.opt.device).float()
        x1 = batch["x1"].to(self.opt.device).float()
        self._assert_batch(y, x1)
        return y, x1

    def _save_checkpoint(self, optimizer, sched):
        state = {
            "net": self.net.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sched": sched.state_dict() if sched is not None else None,
        }
        torch.save(state, self.opt.ckpt_path / "latest.pt")
        self.log.info(f"[Ckpt] Saved checkpoint to {self.opt.ckpt_path / 'latest.pt'}")

    def _prediction_to_save(self, pred_x1s):
        return pred_x1s[:, -1, ...]

    def _reduce_metric_sums(self, total_squared_error, total_elements, total_samples):
        if not self.opt.distributed:
            return total_squared_error, total_elements, total_samples

        stats = torch.tensor(
            [total_squared_error, total_elements, total_samples],
            dtype=torch.float64,
            device=self.opt.device,
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        return float(stats[0].item()), float(stats[1].item()), int(stats[2].item())

    def train(self, opt, train_dataset, val_dataset):
        self.writer = util.build_log_writer(opt)
        net = self._create_train_model()
        optimizer, sched = build_optimizer_sched(opt, self.net, self.log, checkpoint=self.checkpoint)
        train_loader, train_sampler = self._build_train_dataloader(train_dataset)
        train_iter = iter(train_loader)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            if train_sampler is not None:
                train_sampler.set_epoch(it)
            optimizer.zero_grad()
            loss_value = None

            for _ in range(n_inner_loop):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                y = batch["y"].to(opt.device).float()
                x1 = batch["x1"].to(opt.device).float()
                self._assert_batch(y, x1)
                _y_bridge, y_noisy, _eps = self._sample_noisy_observation(y)
                step = torch.randint(0, opt.interval, (x1.shape[0],), device=opt.device)
                xt = self.diffusion.q_sample(step, x1=x1, y_noisy=y_noisy, ot_ode=opt.ot_ode)
                target = self.compute_target(step, y_noisy, x1)
                pred = net(xt, step, cond=y)
                loss = F.mse_loss(pred, target)
                loss.backward()
                loss_value = loss.detach()

            optimizer.step()
            self.ema.update()
            if sched is not None:
                sched.step()

            self.log.info(
                "train_it {}/{} | lr:{} | loss:{}".format(
                    it + 1,
                    opt.num_itr,
                    "{:.2e}".format(optimizer.param_groups[0]["lr"]),
                    "{:+.4f}".format(loss_value.item()),
                )
            )
            if it % 10 == 0:
                self.writer.add_scalar(it, "loss", loss_value)

            if it % opt.save_interval == 0 and opt.global_rank == 0:
                self._save_checkpoint(optimizer, sched)
            if opt.distributed and it % opt.save_interval == 0:
                torch.distributed.barrier()

            if it == opt.eval_first_iter or (it > 0 and it % opt.eval_interval == 0):
                self.evaluation(opt, it, val_dataset)
                net.train()

        self.writer.close()

    @torch.no_grad()
    def bridge_sampling(
        self,
        opt,
        y,
        *,
        y_noisy=None,
        clip_denoise=False,
        nfe=None,
        log_count=10,
        verbose=True,
        keep_history=True,
    ):
        self._assert_batch(y, torch.zeros((y.shape[0], opt.target_channels, *y.shape[-2:]), device=y.device))
        if y_noisy is None:
            _y_bridge, y_noisy, _eps = self._sample_noisy_observation(y)

        if y_noisy.shape[1] != opt.target_channels:
            raise ValueError(f"Start state must have {opt.target_channels} channels, got {y_noisy.shape[1]}")

        nfe = nfe or opt.interval - 1
        if not (0 < nfe < opt.interval == len(self.diffusion.betas)):
            raise ValueError(f"Invalid nfe={nfe} for interval={opt.interval}")

        steps = util.space_indices(opt.interval, nfe + 1)
        log_count = min(len(steps), log_count)
        if len(steps) <= 2:
            log_steps = steps[:]
        else:
            log_positions = util.space_indices(len(steps) - 1, min(len(steps) - 1, log_count))
            log_steps = [steps[i] for i in log_positions]
        if 0 not in log_steps:
            log_steps = [0] + log_steps
        if steps[-1] not in log_steps:
            log_steps = log_steps + [steps[-1]]
        log_steps = sorted(set(log_steps))

        xt = y_noisy.to(opt.device).float()
        cond = y.to(opt.device).float()
        if not torch.allclose(xt, y_noisy):
            raise ValueError("Bridge sampling must start from y_noisy")

        xs = [] if keep_history else None
        pred_x1s = [] if keep_history else None
        final_xt = None
        final_pred_x1 = None
        rollout_start = time.time()
        progress_points = sorted(
            set(
                [
                    0,
                    len(steps) - 1,
                    round((len(steps) - 1) * 0.25),
                    round((len(steps) - 1) * 0.50),
                    round((len(steps) - 1) * 0.75),
                ]
            )
        )

        if verbose:
            self.log.info(
                f"[BridgeDebug] rollout_start total_steps={len(steps)} nfe={nfe} "
                f"log_steps={log_steps} y_shape={tuple(y.shape)} y_noisy_shape={tuple(y_noisy.shape)} "
                f"{self._gpu_memory_summary()}"
            )

        # Practical paired bridge rollout: start from the observation endpoint y_noisy at t=0
        # and evolve toward x1 using endpoint-conditioned transitions. This is not a full
        # general dynamic SB solver with KL path optimization.
        with self.ema.average_parameters():
            self.net.eval()
            for idx, step in enumerate(steps):
                step_tensor = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                pred = self.net(xt, step_tensor, cond=cond)
                pred_x1 = self.compute_pred_x1(step_tensor, y_noisy, pred, clip_denoise=clip_denoise)

                if verbose and idx in progress_points:
                    elapsed = time.time() - rollout_start
                    self.log.info(
                        f"[BridgeDebug] progress idx={idx + 1}/{len(steps)} diffusion_step={step} "
                        f"elapsed={_format_duration(elapsed)} xt({_tensor_stats(xt)}) "
                        f"pred_x1({_tensor_stats(pred_x1)}) {self._gpu_memory_summary()}"
                    )
                    self._log_bridge_tensor_health("[BridgeDebug] xt", xt)
                    self._log_bridge_tensor_health("[BridgeDebug] pred_x1", pred_x1)

                if keep_history and step in log_steps:
                    xs.append(xt.detach().cpu())
                    pred_x1s.append(pred_x1.detach().cpu())

                final_xt = xt.detach().cpu()
                final_pred_x1 = pred_x1.detach().cpu()

                if idx == len(steps) - 1:
                    break

                next_xt = self.diffusion.endpoint_conditioned_step(
                    step,
                    steps[idx + 1],
                    xt,
                    pred_x1,
                    y_noisy,
                    ot_ode=opt.ot_ode,
                )
                xt = next_xt

        if verbose:
            self.log.info(
                f"[BridgeDebug] rollout_finished elapsed={_format_duration(time.time() - rollout_start)} "
                f"saved_states={len(xs) if xs is not None else 1} "
                f"saved_predictions={len(pred_x1s) if pred_x1s is not None else 1}"
            )

        if keep_history:
            xs_out = torch.stack(xs, dim=1)
            pred_x1s_out = torch.stack(pred_x1s, dim=1)
        else:
            xs_out = final_xt.unsqueeze(1)
            pred_x1s_out = final_pred_x1.unsqueeze(1)

        return xs_out, pred_x1s_out, y_noisy.detach().cpu()

    @torch.no_grad()
    def _evaluate_dataset(self, opt, dataset, *, save_arrays=False, training_eval):
        eval_dataset, eval_meta = self._prepare_eval_dataset(opt, dataset, training_eval=training_eval)
        eval_batch_size = self._effective_eval_batch_size(opt, training_eval=training_eval)
        effective_nfe = self._resolve_eval_nfe(opt, training_eval=training_eval)
        loader = self._build_eval_dataloader(eval_dataset, batch_size=eval_batch_size)
        total_squared_error = 0.0
        total_elements = 0
        total_samples = 0
        saved = None
        dataset_size = eval_meta["effective_samples"]
        total_batches = math.ceil(dataset_size / eval_batch_size) if dataset_size > 0 else 0
        eval_start = time.time()
        progress_interval = 25
        first_batch_shapes = None

        for batch_index, batch in enumerate(loader):
            y = batch["y"].to(opt.device).float()
            x1 = batch["x1"].to(opt.device).float()
            self._assert_batch(y, x1)
            current_shapes = (tuple(y.shape), tuple(x1.shape))
            if first_batch_shapes is None:
                first_batch_shapes = current_shapes
                self.log.info(
                    f"[EvalBatch] first_batch_ready batch=1/{total_batches} "
                    f"y_shape={current_shapes[0]} x1_shape={current_shapes[1]} {self._gpu_memory_summary()}"
                )
                self.log.info("[EvalBatch] first_batch bridge_sampling start")
            elif current_shapes != first_batch_shapes:
                self.log.warning(
                    f"[EvalBatch] shape_change_detected batch={batch_index + 1}/{total_batches} "
                    f"y_shape={current_shapes[0]} x1_shape={current_shapes[1]} "
                    f"expected_y_shape={first_batch_shapes[0]} expected_x1_shape={first_batch_shapes[1]}"
                )
            xs, pred_x1s, _y_noisy = self.bridge_sampling(
                opt,
                y,
                clip_denoise=opt.clip_denoise,
                nfe=effective_nfe,
                log_count=opt.eval_log_count,
                verbose=batch_index == 0,
                keep_history=save_arrays and saved is None and opt.global_rank == 0,
            )
            if batch_index == 0:
                self.log.info("[EvalBatch] first_batch bridge_sampling finished")
            recon = self._prediction_to_save(pred_x1s).to(opt.device)
            squared_error = torch.sum((recon - x1) ** 2)
            total_squared_error += float(squared_error.item())
            total_elements += int(x1.numel())
            total_samples += int(x1.shape[0])

            if saved is None and save_arrays and opt.global_rank == 0:
                saved = {
                    "reconstruction": denormalize_sample(recon.detach().cpu(), self._get_target_denorm_stats()),
                    "target": denormalize_sample(x1.detach().cpu(), self._get_target_denorm_stats()),
                    "trajectory": xs,
                    "batch_index": batch_index,
                }

            should_log_progress = (
                batch_index == 0
                or (batch_index + 1) % progress_interval == 0
                or (batch_index + 1) == total_batches
            )
            if should_log_progress:
                elapsed = time.time() - eval_start
                batches_done = batch_index + 1
                sec_per_batch = elapsed / max(batches_done, 1)
                remaining_batches = max(total_batches - batches_done, 0)
                running_mse = total_squared_error / max(total_elements, 1)
                self.log.info(
                    f"[EvalBatch] progress batch={batches_done}/{total_batches} "
                    f"samples={total_samples}/{dataset_size} elapsed={_format_duration(elapsed)} "
                    f"sec_per_batch={sec_per_batch:.2f}s eta={_format_duration(sec_per_batch * remaining_batches)} "
                    f"running_mse={running_mse:.6f} {self._gpu_memory_summary()}"
                )

        total_squared_error, total_elements, total_samples = self._reduce_metric_sums(
            total_squared_error,
            total_elements,
            total_samples,
        )
        mean_mse = total_squared_error / max(total_elements, 1)
        return mean_mse, total_samples, int(total_elements), saved, eval_meta, eval_batch_size, effective_nfe

    @torch.no_grad()
    def evaluation(self, opt, it, val_dataset):
        if opt.distributed and opt.global_rank != 0:
            torch.distributed.barrier()
            return
        _preview_dataset, eval_meta = self._prepare_eval_dataset(opt, val_dataset, training_eval=True)
        eval_batch_size = self._effective_eval_batch_size(opt, training_eval=True)
        effective_nfe = self._resolve_eval_nfe(opt, training_eval=True)
        total_batches = math.ceil(eval_meta["effective_samples"] / eval_batch_size) if eval_meta["effective_samples"] > 0 else 0
        subset_mode = "deterministic_subset" if eval_meta["subset_enabled"] else "full_validation"
        self.log.info(f"========== Evaluation started: iter={it} ==========")
        self.log.info(
            f"[EvalDebug] start_time={datetime.now().isoformat(timespec='seconds')} iter={it} "
            f"task={opt.task} bridge_mode={getattr(opt, 'mag_bridge_mode', 'n/a')} "
            f"device={opt.device} val_samples={eval_meta['effective_samples']}/{eval_meta['full_samples']} "
            f"eval_batch_size={eval_batch_size} eval_batches={total_batches} eval_nfe={effective_nfe} "
            f"interval={opt.interval} eval_mode={subset_mode} eval_subset_seed={eval_meta['subset_seed']} "
            f"{self._gpu_memory_summary()}"
        )
        mean_mse, total_samples, total_elements, saved, _eval_meta, _eval_batch_size, _effective_nfe = self._evaluate_dataset(
            opt,
            val_dataset,
            save_arrays=True,
            training_eval=True,
        )
        self.writer.add_scalar(it, "eval_mse", torch.tensor(mean_mse))
        if saved is not None and opt.global_rank == 0:
            np.save(opt.ckpt_path / f"eval_reconstruction_iter_{it}.npy", saved["reconstruction"].numpy())
            np.save(opt.ckpt_path / f"eval_target_iter_{it}.npy", saved["target"].numpy())
        self.log.info(
            f"[Eval] mean_mse={mean_mse:.6f} | samples={total_samples} | elements={total_elements}"
        )
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.log.info(f"========== Evaluation finished: iter={it} ==========")
        if opt.distributed:
            torch.distributed.barrier()

    @torch.no_grad()
    def eval(self, opt, val_dataset):
        if opt.distributed and opt.global_rank != 0:
            torch.distributed.barrier()
            return
        self.writer = util.build_log_writer(opt)

        if opt.load is None:
            raise ValueError("Evaluation requires --ckpt or a resolved checkpoint path")

        mean_mse, total_samples, total_elements, saved, _eval_meta, _eval_batch_size, _effective_nfe = self._evaluate_dataset(
            opt,
            val_dataset,
            save_arrays=True,
            training_eval=False,
        )
        metrics = {
            "mean_mse": mean_mse,
            "num_samples": total_samples,
            "num_elements": total_elements,
        }
        if opt.global_rank == 0:
            np.save(opt.ckpt_path / "eval_metrics.npy", metrics, allow_pickle=True)
        if saved is not None and opt.global_rank == 0:
            np.save(opt.ckpt_path / "eval_reconstruction.npy", saved["reconstruction"].numpy())
            np.save(opt.ckpt_path / "eval_target.npy", saved["target"].numpy())
            np.save(opt.ckpt_path / "eval_trajectory.npy", saved["trajectory"].numpy())
        self.log.info(f"[Eval] Saved validation reconstructions and metrics to {opt.ckpt_path}")

        if opt.task == "density":
            self._run_taurus_inference(opt)

        self.writer.close()
        if opt.distributed:
            torch.distributed.barrier()

    @torch.no_grad()
    def _run_taurus_inference(self, opt):
        taurus_path = getattr(opt, "taurus_path", None)
        if taurus_path is None:
            taurus_path = Path(__file__).resolve().parents[1] / "data" / "taurus_L1495_column_density_map_rot_norm_128.npy"
        taurus_path = Path(taurus_path)
        if not taurus_path.exists():
            self.log.warning(f"[Eval] Taurus file not found at {taurus_path}; skipping")
            return

        observation = np.load(str(taurus_path))
        if observation.shape[0] < opt.image_size or observation.shape[1] < opt.image_size:
            raise ValueError(
                f"Taurus inference image must be at least patch-sized, got {observation.shape} for image_size={opt.image_size}"
            )
        patches, positions, img_size = crop_image(observation, patch_size=opt.image_size, step=opt.patch_stride)
        predictions = []
        obs_norm_stats = self._get_observation_norm_stats()
        for patch in patches:
            y = normalize_array(patch, obs_norm_stats).unsqueeze(0).to(opt.device)
            xs, pred_x1s, _ = self.bridge_sampling(
                opt,
                y,
                clip_denoise=opt.clip_denoise,
                nfe=self._resolve_eval_nfe(opt, training_eval=False),
                log_count=opt.eval_log_count,
                verbose=False,
                keep_history=False,
            )
            pred_patch = denormalize_sample(
                self._prediction_to_save(pred_x1s),
                self._get_target_denorm_stats(),
            ).squeeze().cpu().numpy()
            predictions.append(pred_patch)

        reconstructed = merge_patches(predictions, positions, img_size, patch_size=opt.image_size)
        np.save(opt.ckpt_path / "recons_taurus_inverse.npy", reconstructed)
        self.log.info(f"[Eval] Saved Taurus reconstruction to {opt.ckpt_path / 'recons_taurus_inverse.npy'}")
