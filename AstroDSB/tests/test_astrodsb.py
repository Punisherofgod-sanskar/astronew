import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics.losses import SlabColumnDensityOperator, build_physics_regularizer
from astrodsb.diffusion import Diffusion
from astrodsb.runner import Runner
from datasets_mod import build_task_dataset
from logger import Logger
from train import _calibrate_density_log_effective_depth, validate_runtime_options


def make_opt(tmpdir, **overrides):
    base = dict(
        seed=0,
        name="test_run",
        task="density",
        dataset_file=None,
        mag_channel_schema="default_xu2025",
        dataset_observation_key=None,
        dataset_target_key=None,
        dataset_test_split=0.5,
        dataset_split_seed=0,
        mag_bridge_mode="projected_b_field",
        image_size=64,
        t0=1e-4,
        T=1.0,
        interval=8,
        beta_max=0.3,
        ot_ode=False,
        obs_noise_scale=0.1,
        clip_denoise=False,
        batch_size=1,
        microbatch=1,
        num_itr=1,
        lr=1e-4,
        lr_gamma=1.0,
        lr_step=1000,
        l2_norm=0.0,
        ema=0.99,
        model_channels=32,
        num_res_blocks=1,
        num_workers=0,
        save_interval=100,
        eval_interval=100,
        eval_first_iter=100,
        eval_nfe=4,
        eval_batch_size=None,
        eval_max_samples=None,
        eval_subset_seed=42,
        eval_log_count=4,
        patch_stride=8,
        patch_overlap=8,
        patch_overlap_alias=None,
        log_dir=Path(tmpdir) / "logs",
        log_writer=None,
        wandb_api_key=None,
        wandb_entity=None,
        wandb_project="astrodsb-tests",
        device="cpu",
        distributed=False,
        use_fp16=False,
        global_rank=0,
        local_rank=0,
        global_size=1,
        ckpt_path=Path(tmpdir) / "ckpts",
        load=None,
        n_gpu_per_node=1,
        num_proc_node=1,
        master_address="localhost",
        master_port=6020,
        observation_channels=1,
        target_channels=1,
        taurus_path=Path(tmpdir) / "missing_taurus.npy",
        data_range_tolerance=1e-3,
        normalization_mode="auto",
        physics_weight=0.0,
        physics_density_smooth_weight=0.02,
        physics_density_range_weight=0.01,
        physics_density_observation_weight=0.0,
        physics_density_gradient_weight=0.0,
        physics_density_observation_operator="none",
        physics_density_log_effective_depth="0.0",
        physics_density_edge_scale=8.0,
        physics_time_weight_power=1.0,
    )
    base.update(overrides)
    base["log_dir"].mkdir(parents=True, exist_ok=True)
    base["ckpt_path"].mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(**base)


class AstroDSBTests(unittest.TestCase):
    def test_density_dataset_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "density.npy"
            array = np.stack(
                [
                    np.full((4, 64, 64), 0.25, dtype=np.float32),
                    np.full((4, 64, 64), 0.75, dtype=np.float32),
                ]
            )
            np.save(dataset_path, array)

            opt = make_opt(tmpdir, dataset_file=dataset_path, task="density")
            dataset = build_task_dataset(opt, train=True)
            sample = dataset[0]

            self.assertEqual(sample["y"].shape, (1, 64, 64))
            self.assertEqual(sample["x1"].shape, (1, 64, 64))
            self.assertTrue(torch.all(sample["y"] <= 1.0))
            self.assertTrue(torch.all(sample["x1"] >= -1.0))

    def test_validate_runtime_options_rejects_missing_cuda_for_gpu_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = make_opt(tmpdir, gpu=0, n_gpu_per_node=1, distributed=False)
            with mock.patch("train.torch.cuda.is_available", return_value=False):
                with self.assertRaises(RuntimeError):
                    validate_runtime_options(opt)

    def test_validate_runtime_options_rejects_invalid_batch_geometry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = make_opt(tmpdir, distributed=True, n_gpu_per_node=2, global_size=2, batch_size=2, microbatch=2)
            with mock.patch("train.torch.cuda.is_available", return_value=True):
                with self.assertRaises(ValueError):
                    validate_runtime_options(opt)

    def test_validate_runtime_options_rejects_multi_node(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = make_opt(tmpdir, num_proc_node=2)
            with self.assertRaises(ValueError):
                validate_runtime_options(opt)

    def test_validate_runtime_options_rejects_invalid_patch_stride(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = make_opt(tmpdir, patch_stride=0)
            with self.assertRaises(ValueError):
                validate_runtime_options(opt)

    def test_mag_dataset_preserves_channels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "mag.npz"
            np.savez(
                dataset_path,
                X_train=np.full((2, 4, 64, 64), 0.2, dtype=np.float32),
                Y_train=np.full((2, 1, 64, 64), 0.8, dtype=np.float32),
                X_test=np.full((1, 4, 64, 64), 0.3, dtype=np.float32),
                Y_test=np.full((1, 1, 64, 64), 0.7, dtype=np.float32),
            )

            opt = make_opt(
                tmpdir,
                task="mag",
                dataset_file=dataset_path,
                dataset_observation_key="X_train",
                dataset_target_key="Y_train",
                observation_channels=4,
            )
            dataset = build_task_dataset(opt, train=True)
            sample = dataset[0]

            self.assertEqual(sample["y"].shape, (4, 64, 64))
            self.assertEqual(sample["x1"].shape, (1, 64, 64))

    def test_mag_bridge_endpoint_uses_explicit_channel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = make_opt(tmpdir, observation_channels=4, target_channels=1, task="mag", mag_bridge_mode="projected_b_field")
            log = Logger(0, opt.log_dir)
            runner = Runner(opt, log)

            y = torch.stack(
                [
                    torch.zeros((64, 64)),
                    torch.ones((64, 64)),
                    torch.full((64, 64), 2.0),
                    torch.full((64, 64), 3.0),
                ],
                dim=0,
            ).unsqueeze(0)
            y_bridge = runner._project_observation(y)
            self.assertTrue(torch.allclose(y_bridge, torch.full((1, 1, 64, 64), 3.0)))

    def test_invalid_mag_bridge_mode_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = make_opt(tmpdir, observation_channels=4, target_channels=1, task="mag", mag_bridge_mode="not_real")
            log = Logger(0, opt.log_dir)
            runner = Runner(opt, log)
            y = torch.zeros((1, 4, 64, 64))
            with self.assertRaises(ValueError):
                runner._project_observation(y)

    def test_invalid_mag_channel_schema_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = make_opt(tmpdir, observation_channels=4, target_channels=1, task="mag", mag_channel_schema="not_real")
            log = Logger(0, opt.log_dir)
            runner = Runner(opt, log)
            y = torch.zeros((1, 4, 64, 64))
            with self.assertRaises(ValueError):
                runner._project_observation(y)

    def test_bridge_target_and_sampling_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = make_opt(tmpdir, observation_channels=4, target_channels=1, task="mag")
            log = Logger(0, opt.log_dir)
            runner = Runner(opt, log)

            y = torch.full((1, 4, 64, 64), 0.6)
            x1 = torch.full((1, 1, 64, 64), -0.2)
            step = torch.tensor([3], dtype=torch.long)

            y_bridge, y_noisy, _ = runner._sample_noisy_observation(y)
            target = runner.compute_target(step, y_noisy, x1)
            sigma = runner.diffusion.get_std_fwd(step, xdim=x1.shape[1:])
            expected = (y_noisy - x1) / sigma

            self.assertTrue(torch.allclose(target, expected))

            xs, pred_x1s, start_state = runner.bridge_sampling(
                opt,
                y,
                y_noisy=y_noisy,
                nfe=4,
                log_count=4,
                verbose=False,
            )
            self.assertTrue(torch.allclose(start_state, y_noisy.cpu()))
            self.assertEqual(xs.shape[0], 1)
            self.assertEqual(pred_x1s.shape[0], 1)

    def test_reverse_dynamics_depend_on_current_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = make_opt(tmpdir, observation_channels=1, target_channels=1, task="density", ot_ode=True)
            log = Logger(0, opt.log_dir)
            runner = Runner(opt, log)

            y_noisy = torch.zeros((1, 1, 64, 64))
            pred_x1 = torch.ones((1, 1, 64, 64))
            x_t_a = torch.zeros((1, 1, 64, 64))
            x_t_b = torch.full((1, 1, 64, 64), 0.5)

            x_next_a = runner.diffusion.endpoint_conditioned_step(1, 3, x_t_a, pred_x1, y_noisy, ot_ode=True)
            x_next_b = runner.diffusion.endpoint_conditioned_step(1, 3, x_t_b, pred_x1, y_noisy, ot_ode=True)

            self.assertFalse(torch.allclose(x_next_a, x_next_b))

    def test_reverse_bridge_api_removed(self):
        diffusion = Diffusion(np.linspace(1e-4, 2e-4, 8), "cpu")
        self.assertFalse(hasattr(diffusion, "reverse_bridge_step"))

    def test_diffusion_no_ddpm_posterior(self):
        diffusion = Diffusion(np.linspace(1e-4, 2e-4, 8), "cpu")
        self.assertFalse(hasattr(diffusion, "p_posterior"))

    def test_one_step_train_and_eval_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "density.npy"
            array = np.stack(
                [
                    np.random.rand(4, 64, 64).astype(np.float32) * 0.8,
                    np.random.rand(4, 64, 64).astype(np.float32) * 0.8,
                ]
            )
            np.save(dataset_path, array)

            opt = make_opt(tmpdir, dataset_file=dataset_path, task="density")
            train_dataset = build_task_dataset(opt, train=True)
            val_dataset = build_task_dataset(opt, train=False)
            sample = train_dataset[0]
            opt.observation_channels = int(sample["y"].shape[0])
            opt.target_channels = int(sample["x1"].shape[0])

            log = Logger(0, opt.log_dir)
            runner = Runner(opt, log)
            runner.train(opt, train_dataset, val_dataset)

            latest = opt.ckpt_path / "latest.pt"
            self.assertTrue(latest.exists())

            eval_opt = make_opt(tmpdir, dataset_file=dataset_path, task="density", load=latest)
            eval_opt.observation_channels = opt.observation_channels
            eval_opt.target_channels = opt.target_channels
            eval_opt.ckpt_path = opt.ckpt_path
            runner = Runner(eval_opt, log, save_opt=False)
            runner.eval(eval_opt, val_dataset)

            self.assertTrue((opt.ckpt_path / "eval_reconstruction.npy").exists())
            self.assertTrue((opt.ckpt_path / "eval_metrics.npy").exists())
            metrics = np.load(opt.ckpt_path / "eval_metrics.npy", allow_pickle=True).item()
            self.assertGreaterEqual(metrics["num_samples"], 1)
            self.assertIn("mean_mse", metrics)
            self.assertIn("num_elements", metrics)

    def test_taurus_uses_terminal_prediction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = make_opt(tmpdir, task="density", observation_channels=1, target_channels=1)
            log = Logger(0, opt.log_dir)
            runner = Runner(opt, log)

            observation = np.full((64, 64), 0.5, dtype=np.float32)
            taurus_path = Path(tmpdir) / "taurus.npy"
            np.save(taurus_path, observation)
            opt.taurus_path = taurus_path

            original_bridge_sampling = runner.bridge_sampling

            def fake_bridge_sampling(*args, **kwargs):
                y = args[1]
                pred_x1s = torch.stack(
                    [
                        torch.zeros((y.shape[0], 1, y.shape[-2], y.shape[-1])),
                        torch.ones((y.shape[0], 1, y.shape[-2], y.shape[-1])),
                    ],
                    dim=1,
                )
                xs = pred_x1s.clone()
                y_noisy = torch.zeros((y.shape[0], 1, y.shape[-2], y.shape[-1]))
                return xs, pred_x1s, y_noisy

            runner.bridge_sampling = fake_bridge_sampling
            runner._run_taurus_inference(opt)
            reconstructed = np.load(opt.ckpt_path / "recons_taurus_inverse.npy")
            self.assertTrue(np.allclose(reconstructed, 1.0))
            runner.bridge_sampling = original_bridge_sampling

    def test_taurus_rejects_undersized_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = make_opt(tmpdir, task="density", observation_channels=1, target_channels=1, image_size=64)
            log = Logger(0, opt.log_dir)
            runner = Runner(opt, log)

            observation = np.full((32, 32), 0.5, dtype=np.float32)
            taurus_path = Path(tmpdir) / "taurus_small.npy"
            np.save(taurus_path, observation)
            opt.taurus_path = taurus_path

            with self.assertRaises(ValueError):
                runner._run_taurus_inference(opt)

    def test_eval_dataloader_uses_eval_batch_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "density.npy"
            array = np.stack(
                [
                    np.random.rand(6, 64, 64).astype(np.float32) * 0.8,
                    np.random.rand(6, 64, 64).astype(np.float32) * 0.8,
                ]
            )
            np.save(dataset_path, array)

            opt = make_opt(tmpdir, dataset_file=dataset_path, task="density", microbatch=1, eval_batch_size=3)
            dataset = build_task_dataset(opt, train=False)
            sample = dataset[0]
            opt.observation_channels = int(sample["y"].shape[0])
            opt.target_channels = int(sample["x1"].shape[0])

            log = Logger(0, opt.log_dir)
            runner = Runner(opt, log)
            loader = runner._build_eval_dataloader(
                dataset,
                batch_size=runner._effective_eval_batch_size(opt, training_eval=True),
            )

            self.assertEqual(loader.batch_size, 3)

    def test_training_eval_subset_is_deterministic_and_capped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "density.npy"
            array = np.stack(
                [
                    np.random.rand(10, 64, 64).astype(np.float32) * 0.8,
                    np.random.rand(10, 64, 64).astype(np.float32) * 0.8,
                ]
            )
            np.save(dataset_path, array)

            opt = make_opt(tmpdir, dataset_file=dataset_path, task="density", eval_max_samples=4, eval_subset_seed=7)
            dataset = build_task_dataset(opt, train=False)
            sample = dataset[0]
            opt.observation_channels = int(sample["y"].shape[0])
            opt.target_channels = int(sample["x1"].shape[0])

            log = Logger(0, opt.log_dir)
            runner = Runner(opt, log)
            subset_a, meta_a = runner._prepare_eval_dataset(opt, dataset, training_eval=True)
            subset_b, meta_b = runner._prepare_eval_dataset(opt, dataset, training_eval=True)

            self.assertEqual(len(subset_a), 4)
            self.assertEqual(len(subset_b), 4)
            self.assertTrue(meta_a["subset_enabled"])
            self.assertEqual(meta_a["effective_samples"], 4)
            self.assertEqual(meta_a["full_samples"], len(dataset))
            self.assertEqual(subset_a.indices, subset_b.indices)
            self.assertEqual(meta_a["subset_seed"], meta_b["subset_seed"])

    def test_evaluate_dataset_uses_subset_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "density.npy"
            array = np.stack(
                [
                    np.random.rand(6, 64, 64).astype(np.float32) * 0.8,
                    np.random.rand(6, 64, 64).astype(np.float32) * 0.8,
                ]
            )
            np.save(dataset_path, array)

            opt = make_opt(
                tmpdir,
                dataset_file=dataset_path,
                task="density",
                eval_max_samples=2,
                eval_batch_size=2,
                eval_nfe=4,
            )
            dataset = build_task_dataset(opt, train=False)
            sample = dataset[0]
            opt.observation_channels = int(sample["y"].shape[0])
            opt.target_channels = int(sample["x1"].shape[0])

            log = Logger(0, opt.log_dir)
            runner = Runner(opt, log)
            mean_mse, total_samples, total_elements, saved, eval_meta, eval_batch_size, effective_nfe = runner._evaluate_dataset(
                opt,
                dataset,
                save_arrays=True,
                training_eval=True,
            )

            self.assertGreaterEqual(mean_mse, 0.0)
            self.assertEqual(total_samples, 2)
            self.assertEqual(total_elements, 2 * 64 * 64)
            self.assertIsNotNone(saved)
            self.assertTrue(eval_meta["subset_enabled"])
            self.assertEqual(eval_meta["effective_samples"], 2)
            self.assertEqual(eval_batch_size, 2)
            self.assertEqual(effective_nfe, 4)

    def test_training_eval_defaults_do_not_affect_manual_eval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "density.npy"
            array = np.stack(
                [
                    np.random.rand(5, 64, 64).astype(np.float32) * 0.8,
                    np.random.rand(5, 64, 64).astype(np.float32) * 0.8,
                ]
            )
            np.save(dataset_path, array)

            opt = make_opt(tmpdir, dataset_file=dataset_path, task="density", eval_nfe=None, eval_max_samples=None)
            dataset = build_task_dataset(opt, train=False)
            sample = dataset[0]
            opt.observation_channels = int(sample["y"].shape[0])
            opt.target_channels = int(sample["x1"].shape[0])

            log = Logger(0, opt.log_dir)
            runner = Runner(opt, log)
            _, meta = runner._prepare_eval_dataset(opt, dataset, training_eval=False)

            self.assertFalse(meta["subset_enabled"])
            self.assertEqual(meta["effective_samples"], len(dataset))
            self.assertEqual(runner._resolve_eval_nfe(opt, training_eval=False), opt.interval - 1)

    def test_one_step_mag_train_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "mag.npz"
            np.savez(
                dataset_path,
                X_train=np.random.rand(4, 4, 64, 64).astype(np.float32) * 0.8,
                Y_train=np.random.rand(4, 1, 64, 64).astype(np.float32) * 0.8,
                X_test=np.random.rand(2, 4, 64, 64).astype(np.float32) * 0.8,
                Y_test=np.random.rand(2, 1, 64, 64).astype(np.float32) * 0.8,
            )

            opt = make_opt(
                tmpdir,
                dataset_file=dataset_path,
                task="mag",
                observation_channels=4,
                target_channels=1,
            )
            train_dataset = build_task_dataset(opt, train=True)
            val_dataset = build_task_dataset(opt, train=False)
            sample = train_dataset[0]
            opt.observation_channels = int(sample["y"].shape[0])
            opt.target_channels = int(sample["x1"].shape[0])

            log = Logger(0, opt.log_dir)
            runner = Runner(opt, log)
            runner.train(opt, train_dataset, val_dataset)
            self.assertTrue((opt.ckpt_path / "latest.pt").exists())

    def test_one_step_mag_train_with_periodic_eval_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "mag.npz"
            np.savez(
                dataset_path,
                X_train=np.random.rand(4, 4, 64, 64).astype(np.float32) * 0.8,
                Y_train=np.random.rand(4, 1, 64, 64).astype(np.float32) * 0.8,
                X_test=np.random.rand(3, 4, 64, 64).astype(np.float32) * 0.8,
                Y_test=np.random.rand(3, 1, 64, 64).astype(np.float32) * 0.8,
            )

            opt = make_opt(
                tmpdir,
                dataset_file=dataset_path,
                task="mag",
                observation_channels=4,
                target_channels=1,
                num_itr=1,
                eval_first_iter=0,
                eval_interval=100,
                eval_nfe=2,
                eval_batch_size=2,
                eval_max_samples=1,
            )
            train_dataset = build_task_dataset(opt, train=True)
            val_dataset = build_task_dataset(opt, train=False)
            sample = train_dataset[0]
            opt.observation_channels = int(sample["y"].shape[0])
            opt.target_channels = int(sample["x1"].shape[0])

            log = Logger(0, opt.log_dir)
            runner = Runner(opt, log)
            runner.train(opt, train_dataset, val_dataset)

            self.assertTrue((opt.ckpt_path / "latest.pt").exists())
            self.assertTrue((opt.ckpt_path / "eval_reconstruction_iter_0.npy").exists())
            self.assertTrue((opt.ckpt_path / "eval_target_iter_0.npy").exists())

    def test_dataset_normalization_handles_small_overshoot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "density.npy"
            array = np.stack(
                [
                    np.full((2, 64, 64), 0.2, dtype=np.float32),
                    np.full((2, 64, 64), 1.056, dtype=np.float32),
                ]
            )
            np.save(dataset_path, array)

            opt = make_opt(tmpdir, dataset_file=dataset_path, task="density", normalization_mode="dataset")
            dataset = build_task_dataset(opt, train=True)
            sample = dataset[0]
            self.assertTrue(torch.all(sample["x1"] <= 1.0))
            self.assertTrue(torch.all(sample["x1"] >= -1.0))

    def test_out_of_range_inputs_fail_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "density.npy"
            array = np.stack(
                [
                    np.full((2, 64, 64), 0.2, dtype=np.float32),
                    np.full((2, 64, 64), 1.2, dtype=np.float32),
                ]
            )
            np.save(dataset_path, array)

            opt = make_opt(
                tmpdir,
                dataset_file=dataset_path,
                task="density",
                normalization_mode="strict_unit_interval",
            )
            with self.assertRaises(ValueError):
                _ = build_task_dataset(opt, train=True)

    def test_density_physics_regularizer_penalizes_out_of_range_predictions(self):
        opt = SimpleNamespace(
            task="density",
            physics_weight=1.0,
            physics_density_smooth_weight=0.0,
            physics_density_range_weight=1.0,
            physics_density_observation_weight=0.0,
            physics_density_gradient_weight=0.0,
            physics_density_observation_operator="none",
            physics_density_edge_scale=8.0,
            physics_time_weight_power=1.0,
        )
        regularizer = build_physics_regularizer(opt)
        prediction = torch.full((1, 1, 8, 8), 1.5)
        observation = torch.zeros((1, 1, 8, 8))

        loss, log = regularizer(prediction, observation)

        self.assertGreater(loss.item(), 0.0)
        self.assertGreater(log["in_range"], 0.0)

    def test_density_physics_regularizer_identity_operator_requires_matching_shapes(self):
        opt = SimpleNamespace(
            task="density",
            physics_weight=1.0,
            physics_density_smooth_weight=0.0,
            physics_density_range_weight=0.0,
            physics_density_observation_weight=1.0,
            physics_density_gradient_weight=0.0,
            physics_density_observation_operator="identity",
            physics_density_edge_scale=8.0,
            physics_time_weight_power=1.0,
        )
        regularizer = build_physics_regularizer(opt)
        prediction = torch.zeros((1, 1, 8, 8))
        observation = torch.zeros((1, 2, 8, 8))

        with self.assertRaises(ValueError):
            regularizer(prediction, observation)

    def test_magnetic_physics_regularizer_is_rejected_for_scalar_target_pipeline(self):
        opt = SimpleNamespace(
            task="mag",
            physics_weight=1.0,
        )

        with self.assertRaises(ValueError):
            build_physics_regularizer(opt)

    def test_slab_column_density_operator_applies_log_depth_shift_in_physical_space(self):
        operator = SlabColumnDensityOperator(
            target_min=2.0,
            target_max=6.0,
            observation_min=20.0,
            observation_max=24.0,
            log_effective_depth=18.0,
        )
        prediction = torch.zeros((1, 1, 4, 4))

        projected = operator(prediction)

        self.assertTrue(torch.allclose(projected, torch.zeros_like(projected)))

    def test_density_physics_regularizer_slab_operator_matches_consistent_observation(self):
        opt = SimpleNamespace(
            task="density",
            physics_weight=1.0,
            physics_density_smooth_weight=0.0,
            physics_density_range_weight=0.0,
            physics_density_observation_weight=1.0,
            physics_density_gradient_weight=0.0,
            physics_density_observation_operator="slab_column_density",
            physics_density_log_effective_depth=18.0,
            physics_density_edge_scale=8.0,
            physics_time_weight_power=1.0,
            observation_normalization={"min_value": 20.0, "max_value": 24.0},
            target_normalization={"min_value": 2.0, "max_value": 6.0},
        )
        regularizer = build_physics_regularizer(opt)
        prediction = torch.zeros((1, 1, 4, 4))
        observation = torch.zeros((1, 1, 4, 4))

        loss, log = regularizer(prediction, observation)

        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        self.assertAlmostEqual(log["obs_consistency"], 0.0, places=6)

    def test_density_physics_regularizer_reports_gradient_consistency(self):
        opt = SimpleNamespace(
            task="density",
            physics_weight=1.0,
            physics_density_smooth_weight=0.0,
            physics_density_range_weight=0.0,
            physics_density_observation_weight=1.0,
            physics_density_gradient_weight=1.0,
            physics_density_observation_operator="slab_column_density",
            physics_density_log_effective_depth=18.0,
            physics_density_edge_scale=8.0,
            physics_time_weight_power=1.0,
            obs_noise_scale=0.1,
            interval=8,
            observation_normalization={"min_value": 20.0, "max_value": 24.0},
            target_normalization={"min_value": 2.0, "max_value": 6.0},
        )
        regularizer = build_physics_regularizer(opt)
        prediction = torch.zeros((1, 1, 4, 4))
        observation = torch.zeros((1, 1, 4, 4))
        observation[:, :, :, 2:] = 0.5
        step = torch.tensor([7], dtype=torch.long)

        loss, log = regularizer(prediction, observation, step=step)

        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertGreater(log["grad_consistency"], 0.0)
        self.assertGreater(log["weight"], 0.0)

    def test_auto_calibrates_density_log_effective_depth_from_dataset_pairs(self):
        observation = np.full((2, 1, 4, 4), 23.0, dtype=np.float32)
        target = np.full((2, 1, 4, 4), 5.0, dtype=np.float32)
        dataset = SimpleNamespace(observation=observation, target=target)
        opt = SimpleNamespace(
            task="density",
            physics_density_observation_operator="slab_column_density",
            physics_density_log_effective_depth="auto",
        )

        updated = _calibrate_density_log_effective_depth(opt, dataset)

        self.assertAlmostEqual(updated.physics_density_log_effective_depth, 18.0, places=6)


if __name__ == "__main__":
    unittest.main()
