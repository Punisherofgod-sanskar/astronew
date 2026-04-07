from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "astrodsb_mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ITER_RE = re.compile(r"eval_reconstruction_iter_(\d+)\.npy$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot presentation-ready figures from AstroDSB density evaluation artifacts."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/density_run"),
        help="Directory containing density_run outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots should be written. Defaults to <results-dir>/plots.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=4,
        help="Maximum number of representative examples to render in example grids.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG output DPI.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Deterministic sample index for representative image and trajectory selection.",
    )
    return parser.parse_args()


def log(message):
    print(f"[plot_density_results] {message}")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_optional_npy(path: Path, *, allow_pickle: bool = True):
    if not path.exists():
        log(f"Missing optional file: {path}")
        return None
    return np.load(path, allow_pickle=allow_pickle)


def coerce_metric_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_metrics(path: Path):
    data = load_optional_npy(path, allow_pickle=True)
    if data is None:
        return None
    if hasattr(data, "item"):
        data = data.item()
    if not isinstance(data, dict):
        raise ValueError(f"Expected metrics dict in {path}, got {type(data)!r}")
    return {key: coerce_metric_value(val) for key, val in data.items()}


def normalize_to_samples(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr[None, None, :, :]
    if arr.ndim == 3:
        if arr.shape[0] <= 4:
            return arr[None, :, :, :]
        return arr[:, None, :, :]
    if arr.ndim == 4:
        return arr
    if arr.ndim == 5:
        return arr.reshape(-1, *arr.shape[-3:])
    raise ValueError(f"Unsupported array shape for image sampling: {arr.shape}")


def representative_image(array: np.ndarray, sample_index: int, channel_index: int = 0) -> np.ndarray:
    samples = normalize_to_samples(array)
    sample_idx = int(np.clip(sample_index, 0, samples.shape[0] - 1))
    channel_idx = int(np.clip(channel_index, 0, samples.shape[1] - 1))
    return np.asarray(samples[sample_idx, channel_idx])


def flatten_for_metrics(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64).reshape(-1)


def compute_pair_metrics(pred: np.ndarray, target: np.ndarray):
    pred_flat = flatten_for_metrics(pred)
    target_flat = flatten_for_metrics(target)
    residual = pred_flat - target_flat
    mse = float(np.mean(residual ** 2))
    mae = float(np.mean(np.abs(residual)))
    return mse, mae


def discover_iteration_pairs(results_dir: Path):
    recon_by_iter = {}
    for path in results_dir.glob("eval_reconstruction_iter_*.npy"):
        match = ITER_RE.match(path.name)
        if match:
            recon_by_iter[int(match.group(1))] = path

    pairs = []
    for iteration, recon_path in sorted(recon_by_iter.items()):
        target_path = results_dir / f"eval_target_iter_{iteration}.npy"
        if not target_path.exists():
            log(f"Skipping iter {iteration}: missing {target_path.name}")
            continue
        pairs.append((iteration, recon_path, target_path))
    return pairs


def load_image_pair(pred_path: Path, target_path: Path):
    pred = np.load(pred_path)
    target = np.load(target_path)
    pred_samples = normalize_to_samples(pred)
    target_samples = normalize_to_samples(target)
    if pred_samples.shape != target_samples.shape:
        raise ValueError(
            f"Mismatched reconstruction/target shapes for {pred_path.name} and {target_path.name}: "
            f"{pred_samples.shape} vs {target_samples.shape}"
        )
    return pred_samples, target_samples


def choose_training_iterations(iterations, max_count=5):
    if not iterations:
        return []
    if len(iterations) <= max_count:
        return list(iterations)
    chosen = np.linspace(0, len(iterations) - 1, num=max_count)
    unique = []
    for idx in chosen:
        value = iterations[int(round(idx))]
        if value not in unique:
            unique.append(value)
    if iterations[-1] not in unique:
        unique[-1] = iterations[-1]
    return unique


def save_figure(fig, path: Path, dpi: int):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {path}")


def plot_training_progress_grid(results_dir: Path, output_dir: Path, dpi: int, sample_index: int):
    pairs = discover_iteration_pairs(results_dir)
    if not pairs:
        log("No intermediate reconstruction/target pairs found; skipping training progress plots.")
        return None

    iteration_values = [it for it, _, _ in pairs]
    chosen_iterations = choose_training_iterations(iteration_values, max_count=5)
    chosen_pairs = [pair for pair in pairs if pair[0] in chosen_iterations]

    fig, axes = plt.subplots(
        nrows=len(chosen_pairs),
        ncols=3,
        figsize=(12, 3.6 * len(chosen_pairs)),
        squeeze=False,
    )
    fig.suptitle("Density Training Progress", fontsize=14)

    for row, (iteration, recon_path, target_path) in enumerate(chosen_pairs):
        pred, target = load_image_pair(recon_path, target_path)
        pred_img = representative_image(pred, sample_index)
        target_img = representative_image(target, sample_index)
        error_img = np.abs(pred_img - target_img)

        items = [
            (target_img, f"Target @ {iteration}", "viridis"),
            (pred_img, f"Prediction @ {iteration}", "viridis"),
            (error_img, f"|Error| @ {iteration}", "magma"),
        ]
        for col, (image, title, cmap) in enumerate(items):
            ax = axes[row, col]
            im = ax.imshow(image, cmap=cmap)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_figure(fig, output_dir / "training_progress_grid.png", dpi)

    metrics_by_iteration = []
    for iteration, recon_path, target_path in pairs:
        pred, target = load_image_pair(recon_path, target_path)
        mse, mae = compute_pair_metrics(pred, target)
        metrics_by_iteration.append({"iteration": iteration, "mse": mse, "mae": mae})

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(
        [item["iteration"] for item in metrics_by_iteration],
        [item["mse"] for item in metrics_by_iteration],
        marker="o",
        label="MSE",
    )
    ax.plot(
        [item["iteration"] for item in metrics_by_iteration],
        [item["mae"] for item in metrics_by_iteration],
        marker="s",
        label="MAE",
    )
    ax.set_title("Validation Error vs Training Iteration")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, output_dir / "metric_vs_iteration.png", dpi)

    return {
        "selected_iterations": chosen_iterations,
        "per_iteration_metrics": metrics_by_iteration,
    }


def plot_final_eval_examples(
    reconstruction: np.ndarray,
    target: np.ndarray,
    output_dir: Path,
    dpi: int,
    sample_index: int,
    max_examples: int,
):
    pred_samples = normalize_to_samples(reconstruction)
    target_samples = normalize_to_samples(target)
    num_examples = max(1, min(max_examples, pred_samples.shape[0], target_samples.shape[0]))
    start = int(np.clip(sample_index, 0, max(pred_samples.shape[0] - num_examples, 0)))

    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 3.6 * num_examples), squeeze=False)
    fig.suptitle("Final Density Evaluation Examples", fontsize=14)

    for row in range(num_examples):
        idx = start + row
        pred_img = pred_samples[idx, 0]
        target_img = target_samples[idx, 0]
        error_img = np.abs(pred_img - target_img)
        items = [
            (target_img, f"Target #{idx}", "viridis"),
            (pred_img, f"Prediction #{idx}", "viridis"),
            (error_img, f"|Error| #{idx}", "magma"),
        ]
        for col, (image, title, cmap) in enumerate(items):
            ax = axes[row, col]
            im = ax.imshow(image, cmap=cmap)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_figure(fig, output_dir / "final_eval_examples.png", dpi)


def plot_prediction_scatter_and_hist(
    reconstruction: np.ndarray,
    target: np.ndarray,
    output_dir: Path,
    dpi: int,
):
    pred_samples = normalize_to_samples(reconstruction)
    target_samples = normalize_to_samples(target)
    if pred_samples.shape != target_samples.shape:
        raise ValueError(
            f"Final reconstruction and target shapes do not match: {pred_samples.shape} vs {target_samples.shape}"
        )

    pred_flat = flatten_for_metrics(pred_samples)
    target_flat = flatten_for_metrics(target_samples)
    residual = pred_flat - target_flat

    sample_count = min(pred_flat.size, 50000)
    if sample_count < pred_flat.size:
        indices = np.linspace(0, pred_flat.size - 1, sample_count, dtype=int)
        pred_plot = pred_flat[indices]
        target_plot = target_flat[indices]
    else:
        pred_plot = pred_flat
        target_plot = target_flat

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    hb = ax.hexbin(target_plot, pred_plot, gridsize=60, bins="log", cmap="viridis", mincnt=1)
    low = float(min(target_plot.min(), pred_plot.min()))
    high = float(max(target_plot.max(), pred_plot.max()))
    ax.plot([low, high], [low, high], color="white", linestyle="--", linewidth=1.2, label="Ideal")
    ax.set_title("Prediction vs Target")
    ax.set_xlabel("Target value")
    ax.set_ylabel("Predicted value")
    ax.legend(loc="upper left")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log10(count)")
    save_figure(fig, output_dir / "prediction_vs_target_scatter.png", dpi)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residual, bins=80, color="#2a6f97", alpha=0.9)
    ax.set_title("Residual Histogram")
    ax.set_xlabel("Prediction - target")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    save_figure(fig, output_dir / "residual_histogram.png", dpi)


def plot_taurus(taurus: np.ndarray, output_dir: Path, dpi: int):
    taurus_arr = np.asarray(taurus)
    if taurus_arr.ndim != 2:
        raise ValueError(f"Taurus reconstruction must be 2D, got {taurus_arr.shape}")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(taurus_arr, cmap="cividis")
    ax.set_title("Taurus Density Reconstruction")
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Density")
    save_figure(fig, output_dir / "taurus_reconstruction.png", dpi)


def extract_trajectory_images(trajectory: np.ndarray, sample_index: int, max_steps: int = 6):
    arr = np.asarray(trajectory)
    if arr.ndim < 3:
        raise ValueError(f"Trajectory shape {arr.shape} is too small to visualize.")

    if arr.ndim == 3:
        if arr.shape[0] <= 4:
            raise ValueError(f"Ambiguous 3D trajectory shape: {arr.shape}")
        step_stack = arr
    elif arr.ndim == 4:
        if arr.shape[1] <= 4 and arr.shape[0] > 4:
            sample = arr
            step_stack = sample
        elif arr.shape[0] <= 4 and arr.shape[1] > 4:
            step_stack = np.moveaxis(arr, 1, 0)
        else:
            sample_idx = int(np.clip(sample_index, 0, arr.shape[0] - 1))
            sample = arr[sample_idx]
            if sample.ndim != 3:
                raise ValueError(f"Unexpected trajectory sample shape: {sample.shape}")
            if sample.shape[0] <= 4 and sample.shape[1] > 4:
                step_stack = np.moveaxis(sample, 1, 0)
            elif sample.shape[0] > 4:
                step_stack = sample
            else:
                raise ValueError(f"Ambiguous 4D trajectory sample shape: {sample.shape}")
    elif arr.ndim == 5:
        sample_idx = int(np.clip(sample_index, 0, arr.shape[0] - 1))
        sample = arr[sample_idx]
        if sample.ndim != 4:
            raise ValueError(f"Unexpected 5D trajectory sample shape: {sample.shape}")
        if sample.shape[1] <= 4 and sample.shape[0] > 4:
            step_stack = sample[:, 0, :, :]
        elif sample.shape[0] <= 4 and sample.shape[1] > 4:
            step_stack = sample[0]
        else:
            raise ValueError(f"Ambiguous 5D trajectory sample shape: {sample.shape}")
    else:
        raise ValueError(f"Unsupported trajectory shape: {arr.shape}")

    if step_stack.ndim != 3:
        raise ValueError(f"Unexpected trajectory stack shape after processing: {step_stack.shape}")

    indices = np.linspace(0, step_stack.shape[0] - 1, num=min(max_steps, step_stack.shape[0]), dtype=int)
    return [(int(i), step_stack[int(i)]) for i in indices]


def plot_trajectory(trajectory: np.ndarray, output_dir: Path, dpi: int, sample_index: int):
    try:
        steps = extract_trajectory_images(trajectory, sample_index)
    except ValueError as exc:
        log(f"Skipping trajectory preview: {exc}")
        return

    cols = min(3, len(steps))
    rows = int(math.ceil(len(steps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4 * rows), squeeze=False)
    fig.suptitle("Diffusion Trajectory Preview", fontsize=14)

    for ax in axes.flat:
        ax.axis("off")

    for ax, (step_idx, image) in zip(axes.flat, steps):
        im = ax.imshow(image, cmap="viridis")
        ax.set_title(f"Step {step_idx}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_figure(fig, output_dir / "trajectory_preview.png", dpi)


def main():
    args = parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir / "plots"

    ensure_dir(output_dir)
    log(f"Reading results from {results_dir}")
    log(f"Writing plots to {output_dir}")

    summary = {
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "final_metrics": None,
        "selected_iterations": [],
        "per_iteration_metrics": [],
        "generated_plots": [],
    }

    metrics = load_metrics(results_dir / "eval_metrics.npy")
    if metrics is not None:
        summary["final_metrics"] = metrics

    training_summary = plot_training_progress_grid(
        results_dir=results_dir,
        output_dir=output_dir,
        dpi=args.dpi,
        sample_index=args.sample_index,
    )
    if training_summary is not None:
        summary["selected_iterations"] = training_summary["selected_iterations"]
        summary["per_iteration_metrics"] = training_summary["per_iteration_metrics"]
        summary["generated_plots"].extend(
            ["training_progress_grid.png", "metric_vs_iteration.png"]
        )

    final_reconstruction = load_optional_npy(results_dir / "eval_reconstruction.npy")
    final_target = load_optional_npy(results_dir / "eval_target.npy")
    if final_reconstruction is not None and final_target is not None:
        try:
            pred_samples = normalize_to_samples(final_reconstruction)
            target_samples = normalize_to_samples(final_target)
            if pred_samples.shape != target_samples.shape:
                raise ValueError(
                    f"Final reconstruction and target shapes do not match: "
                    f"{pred_samples.shape} vs {target_samples.shape}"
                )
            plot_prediction_scatter_and_hist(pred_samples, target_samples, output_dir, args.dpi)
            plot_final_eval_examples(
                pred_samples,
                target_samples,
                output_dir,
                args.dpi,
                args.sample_index,
                args.max_examples,
            )
            summary["generated_plots"].extend(
                [
                    "prediction_vs_target_scatter.png",
                    "residual_histogram.png",
                    "final_eval_examples.png",
                ]
            )
        except ValueError as exc:
            log(f"Skipping final evaluation plots: {exc}")

    trajectory = load_optional_npy(results_dir / "eval_trajectory.npy")
    if trajectory is not None:
        existing = set(summary["generated_plots"])
        plot_trajectory(trajectory, output_dir, args.dpi, args.sample_index)
        if "trajectory_preview.png" not in existing and (output_dir / "trajectory_preview.png").exists():
            summary["generated_plots"].append("trajectory_preview.png")

    taurus = load_optional_npy(results_dir / "recons_taurus_inverse.npy")
    if taurus is not None:
        try:
            plot_taurus(taurus, output_dir, args.dpi)
            summary["generated_plots"].append("taurus_reconstruction.png")
        except ValueError as exc:
            log(f"Skipping Taurus reconstruction plot: {exc}")

    summary_path = output_dir / "plot_metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
