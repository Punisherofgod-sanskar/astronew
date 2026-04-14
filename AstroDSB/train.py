from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import copy
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.multiprocessing import Process

try:
    import colored_traceback.always
except ImportError:  # pragma: no cover - optional dependency
    colored_traceback = None

from astrodsb import Runner
from datasets_mod import build_task_dataset
from datasets_mod.AllData import DATASET_ROOT
from distributed_util import init_processes
from logger import Logger
from astrodsb import util


RESULT_DIR = Path("results")


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _infer_device(gpu):
    if torch.cuda.is_available():
        return f"cuda:{gpu}" if gpu is not None else "cuda:0"
    return "cpu"


def _resolve_patch_stride(opt):
    alias = getattr(opt, "patch_overlap_alias", None)
    if alias is not None:
        opt.patch_stride = alias
        opt.patch_overlap = alias
        opt.patch_overlap_deprecated = True
    else:
        opt.patch_overlap = opt.patch_stride
        opt.patch_overlap_deprecated = False
    return opt


def validate_runtime_options(opt):
    if getattr(opt, "num_proc_node", 1) != 1:
        raise ValueError(
            "AstroDSB currently supports only single-node execution. Set --num-proc-node 1."
        )

    requested_gpu = getattr(opt, "gpu", None) is not None or getattr(opt, "n_gpu_per_node", 1) > 1
    if requested_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU execution was requested but CUDA is unavailable. Install a GPU-capable PyTorch build on the target machine."
        )

    if getattr(opt, "log_writer", None) == "wandb" and not util.wandb_is_available():
        raise ImportError("wandb logging was requested but the wandb package is not installed.")

    if getattr(opt, "n_gpu_per_node", 1) < 1:
        raise ValueError("--n-gpu-per-node must be at least 1")
    if getattr(opt, "microbatch", 1) < 1:
        raise ValueError("--microbatch must be at least 1")
    if getattr(opt, "batch_size", 1) < 1:
        raise ValueError("--batch-size must be at least 1")
    if getattr(opt, "eval_batch_size", None) is not None and getattr(opt, "eval_batch_size") < 1:
        raise ValueError("--eval-batch-size must be at least 1 when specified")

    distributed = getattr(opt, "distributed", False)
    num_proc_node = getattr(opt, "num_proc_node", 1)
    n_gpu_per_node = getattr(opt, "n_gpu_per_node", 1)
    microbatch = getattr(opt, "microbatch", 1)
    batch_size = getattr(opt, "batch_size", 1)
    world_size = num_proc_node * n_gpu_per_node if distributed else 1
    per_step_batch = world_size * microbatch
    if batch_size < per_step_batch:
        raise ValueError(
            f"--batch-size must be at least world_size * microbatch ({per_step_batch}), got {batch_size}."
        )
    if batch_size % per_step_batch != 0:
        raise ValueError(
            f"--batch-size must be divisible by world_size * microbatch ({per_step_batch}), got {batch_size}."
        )

    if getattr(opt, "patch_stride", getattr(opt, "patch_overlap", 0)) < 1:
        raise ValueError("--patch-stride must be at least 1")
    if getattr(opt, "patch_stride", getattr(opt, "patch_overlap", 0)) > getattr(opt, "image_size", 0):
        raise ValueError("--patch-stride must be less than or equal to --image-size")

    if getattr(opt, "task", "density") == "mag":
        channel_map = Runner.MAG_CHANNEL_SCHEMAS if hasattr(Runner, "MAG_CHANNEL_SCHEMAS") else None
        if channel_map is None:
            from astrodsb.runner import MAG_CHANNEL_SCHEMAS
            channel_map = MAG_CHANNEL_SCHEMAS
        schema = getattr(opt, "mag_channel_schema", None)
        bridge_mode = getattr(opt, "mag_bridge_mode", None)
        if schema not in channel_map:
            raise ValueError(f"Unsupported magnetic channel schema: {schema}")
        if bridge_mode not in channel_map[schema]:
            raise ValueError(f"Unsupported magnetic bridge mode: {bridge_mode}")

    return opt


def _populate_channel_metadata(opt, dataset):
    sample = dataset[0]
    opt.observation_channels = int(sample["y"].shape[0])
    opt.target_channels = int(sample["x1"].shape[0])
    if hasattr(dataset, "get_normalization_state"):
        norm_state = dataset.get_normalization_state()
        opt.observation_normalization = norm_state["observation"]
        opt.target_normalization = norm_state["target"]
    return opt


def _calibrate_density_log_effective_depth(opt, dataset):
    if opt.task != "density":
        return opt
    if getattr(opt, "physics_density_observation_operator", "none") != "slab_column_density":
        return opt
    raw_value = getattr(opt, "physics_density_log_effective_depth", "0.0")
    if str(raw_value).lower() != "auto":
        opt.physics_density_log_effective_depth = float(raw_value)
        return opt
    if not hasattr(dataset, "observation") or not hasattr(dataset, "target"):
        raise ValueError("Cannot auto-calibrate slab depth without raw observation/target arrays")

    observation = np.asarray(dataset.observation, dtype=np.float64)
    target = np.asarray(dataset.target, dtype=np.float64)
    if observation.ndim == 4:
        observation = observation[:, 0]
    if target.ndim == 4:
        target = target[:, 0]
    opt.physics_density_log_effective_depth = float(np.mean(observation - target))
    return opt


def _validate_dataset_runtime(opt):
    if opt.task != "mag":
        return

    from astrodsb.runner import MAG_CHANNEL_SCHEMAS

    channel_map = MAG_CHANNEL_SCHEMAS[opt.mag_channel_schema]
    required_index = channel_map[opt.mag_bridge_mode]
    if opt.observation_channels <= required_index:
        raise ValueError(
            f"Magnetic schema {opt.mag_channel_schema} with mode {opt.mag_bridge_mode} requires "
            f"observation channel index {required_index}, but dataset exposes only {opt.observation_channels} channels."
        )


def _log_dataset_summary(log, name, dataset):
    if not hasattr(dataset, "describe_ranges"):
        return
    summary = dataset.describe_ranges()
    log.info(
        f"[Data:{name}] samples={summary['num_samples']} "
        f"obs_range=[{summary['observation_min']:.6f}, {summary['observation_max']:.6f}] "
        f"target_range=[{summary['target_min']:.6f}, {summary['target_max']:.6f}] "
        f"normalized_obs_range=[{summary['normalized_observation_min']:.1f}, {summary['normalized_observation_max']:.1f}] "
        f"normalized_target_range=[{summary['normalized_target_min']:.1f}, {summary['normalized_target_max']:.1f}]"
    )


def _resolve_dataset_file_for_logging(opt):
    if getattr(opt, "dataset_file", None) is not None:
        return Path(opt.dataset_file)
    if opt.task == "mag":
        return DATASET_ROOT / "Magnetic_Field.npy"
    return DATASET_ROOT / "Density_Prediction.npy"


def create_training_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="astro_dsb")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--n-gpu-per-node", type=int, default=1)
    parser.add_argument("--master-address", type=str, default="localhost")
    parser.add_argument("--master-port", type=int, default=6020)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--num-proc-node", type=int, default=1)

    parser.add_argument("--task", type=str, default="density", choices=["density", "mag"])
    parser.add_argument("--dataset-file", type=Path, default=None)
    parser.add_argument("--dataset-observation-key", type=str, default=None)
    parser.add_argument("--dataset-target-key", type=str, default=None)
    parser.add_argument("--dataset-test-split", type=float, default=0.2)
    parser.add_argument("--dataset-split-seed", type=int, default=42)
    parser.add_argument(
        "--normalization-mode",
        type=str,
        default="auto",
        choices=["auto", "dataset", "strict_unit_interval"],
    )
    parser.add_argument(
        "--mag-channel-schema",
        type=str,
        default="default_xu2025",
        choices=["default_xu2025"],
    )
    parser.add_argument(
        "--mag-bridge-mode",
        type=str,
        default="projected_b_field",
        choices=[
            "projected_b_field",
            "column_density",
            "angle_dispersion",
            "los_velocity_dispersion",
        ],
    )

    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--t0", type=float, default=1e-4)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--beta-max", type=float, default=0.3)
    parser.add_argument("--ot-ode", action="store_true")
    parser.add_argument("--obs-noise-scale", type=float, default=0.1)
    parser.add_argument("--clip-denoise", action="store_true")
    parser.add_argument("--physics-weight", type=float, default=0.0)
    parser.add_argument("--physics-density-smooth-weight", type=float, default=0.02)
    parser.add_argument("--physics-density-range-weight", type=float, default=0.01)
    parser.add_argument("--physics-density-observation-weight", type=float, default=0.0)
    parser.add_argument("--physics-density-gradient-weight", type=float, default=0.0)
    parser.add_argument(
        "--physics-density-observation-operator",
        type=str,
        default="none",
        choices=["none", "identity", "slab_column_density"],
    )
    parser.add_argument("--physics-density-log-effective-depth", type=str, default="0.0")
    parser.add_argument("--physics-density-edge-scale", type=float, default=8.0)
    parser.add_argument("--physics-time-weight-power", type=float, default=1.0)
    parser.add_argument("--physics-mag-smooth-weight", type=float, default=0.02)
    parser.add_argument("--physics-mag-range-weight", type=float, default=0.01)
    parser.add_argument("--physics-mag-observation-weight", type=float, default=0.0)
    parser.add_argument("--physics-mag-gradient-weight", type=float, default=0.0)
    parser.add_argument("--physics-mag-dcf-weight", type=float, default=0.0)
    parser.add_argument("--physics-mag-dcf-q", type=float, default=1.0)
    parser.add_argument("--physics-mag-edge-scale", type=float, default=8.0)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--num-itr", type=int, default=35000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-gamma", type=float, default=0.99)
    parser.add_argument("--lr-step", type=int, default=1000)
    parser.add_argument("--l2-norm", type=float, default=0.0)
    parser.add_argument("--ema", type=float, default=0.99)
    parser.add_argument("--model-channels", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=3000)
    parser.add_argument("--eval-first-iter", type=int, default=500)
    parser.add_argument("--eval-nfe", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-max-samples", type=int, default=None)
    parser.add_argument("--eval-subset-seed", type=int, default=42)
    parser.add_argument("--eval-log-count", type=int, default=10)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--patch-overlap", dest="patch_overlap_alias", type=int, default=None)
    parser.add_argument("--taurus-path", type=Path, default=None)
    parser.add_argument("--data-range-tolerance", type=float, default=1e-3)

    parser.add_argument("--log-dir", type=Path, default=Path(".log"))
    parser.add_argument("--log-writer", type=str, default=None)
    parser.add_argument("--wandb-api-key", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="astrodsb")

    opt = parser.parse_args()

    opt.device = _infer_device(opt.gpu)
    opt.distributed = torch.cuda.is_available() and opt.n_gpu_per_node > 1
    opt.use_fp16 = False
    opt = _resolve_patch_stride(opt)
    opt.log_dir.mkdir(parents=True, exist_ok=True)
    opt.ckpt_path = RESULT_DIR / opt.name
    opt.ckpt_path.mkdir(parents=True, exist_ok=True)

    if opt.ckpt is not None:
        ckpt_candidate = Path(opt.ckpt)
        if ckpt_candidate.is_file():
            opt.load = ckpt_candidate
        else:
            opt.load = RESULT_DIR / opt.ckpt / "latest.pt"
            assert opt.load.exists(), f"Checkpoint not found: {opt.load}"
    else:
        opt.load = None

    return validate_runtime_options(opt)


def main(opt):
    log = Logger(opt.global_rank, opt.log_dir)
    log.info("===================================================================================")
    log.info("   Astro-DSB Trainer")
    log.info("===================================================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")
    log.info(
        f"[Runtime] cuda_available={torch.cuda.is_available()} device={opt.device} "
        f"distributed={opt.distributed} world_size={opt.global_size} "
        f"n_gpu_per_node={opt.n_gpu_per_node} microbatch={opt.microbatch} "
        f"effective_per_step_batch={opt.global_size * opt.microbatch} normalization={opt.normalization_mode}"
    )
    if opt.patch_overlap_deprecated:
        log.warning("--patch-overlap is deprecated and interpreted as patch stride; prefer --patch-stride.")

    if opt.seed is not None:
        set_seed(opt.seed + opt.global_rank)

    train_dataset = build_task_dataset(opt, train=True)
    val_dataset = build_task_dataset(opt, train=False)
    opt = _populate_channel_metadata(opt, train_dataset)
    opt = _calibrate_density_log_effective_depth(opt, train_dataset)
    _validate_dataset_runtime(opt)
    log.info(
        f"[Data] task={opt.task} obs_channels={opt.observation_channels} target_channels={opt.target_channels}"
    )
    if opt.task == "density" and getattr(opt, "physics_density_observation_operator", "none") == "slab_column_density":
        log.info(
            f"[Physics] density_observation_operator=slab_column_density "
            f"log_effective_depth={opt.physics_density_log_effective_depth:.6f}"
        )
    if opt.task == "mag":
        dataset_path = _resolve_dataset_file_for_logging(opt)
        first_sample = train_dataset[0]
        log.info(
            f"[Data] mag_channel_schema={opt.mag_channel_schema} mag_bridge_mode={opt.mag_bridge_mode}"
        )
        log.info(
            f"[Data] mag_dataset={dataset_path} train_keys=(X_train,Y_train) val_keys=(X_test,Y_test) "
            f"train_samples={len(train_dataset)} val_samples={len(val_dataset)} "
            f"first_y_shape={tuple(first_sample['y'].shape)} first_x1_shape={tuple(first_sample['x1'].shape)} "
            f"obs_channels={opt.observation_channels} target_channels={opt.target_channels}"
        )
    _log_dataset_summary(log, "train", train_dataset)
    _log_dataset_summary(log, "val", val_dataset)

    run = Runner(opt, log)
    run.train(opt, train_dataset, val_dataset)
    log.info("Finish!")


if __name__ == "__main__":
    opt = create_training_options()
    opt.global_rank = 0
    opt.local_rank = 0
    opt.global_size = 1

    if opt.distributed:
        size = opt.n_gpu_per_node
        processes = []
        for rank in range(size):
            rank_opt = copy.deepcopy(opt)
            rank_opt.local_rank = rank
            rank_opt.global_rank = rank + rank_opt.node_rank * rank_opt.n_gpu_per_node
            rank_opt.global_size = rank_opt.num_proc_node * rank_opt.n_gpu_per_node
            rank_opt.device = f"cuda:{rank}"
            process = Process(
                target=init_processes,
                args=(rank_opt.global_rank, rank_opt.global_size, main, rank_opt),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
    else:
        main(opt)
