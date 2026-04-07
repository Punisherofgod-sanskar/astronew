from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import struct
import zipfile

import numpy as np
import torch


DATASET_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetPair:
    observation: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class NormalizationStats:
    mode: str
    min_value: float
    max_value: float


@dataclass(frozen=True)
class DatasetNormalization:
    observation: NormalizationStats
    target: NormalizationStats


def _default_density_file() -> Path:
    return DATASET_ROOT / "Density_Prediction.npy"


def _default_mag_file() -> Path:
    return DATASET_ROOT / "Magnetic_Field.npy"


def _resolve_normalization_mode(config, dataset_file: Path) -> str:
    mode = getattr(config, "normalization_mode", "auto")
    if mode == "auto":
        return "dataset" if dataset_file.resolve().parent == DATASET_ROOT.resolve() else "strict_unit_interval"
    return mode


def _build_stats(array: np.ndarray, mode: str) -> NormalizationStats:
    array = np.asarray(array)
    if not np.isfinite(array).all():
        raise ValueError("Dataset contains NaN or Inf values")

    min_value = float(np.nanmin(array))
    max_value = float(np.nanmax(array))
    if mode == "strict_unit_interval":
        tol = 1e-3
        if min_value < -tol or max_value > 1.0 + tol:
            raise ValueError(
                f"Expected normalized values in [0, 1] within tolerance {tol}; got range [{min_value}, {max_value}]"
            )
        return NormalizationStats(mode=mode, min_value=0.0, max_value=1.0)

    if max_value < min_value:
        raise ValueError(f"Cannot normalize invalid data range [{min_value}, {max_value}]")
    return NormalizationStats(mode=mode, min_value=min_value, max_value=max_value)


def normalize_array(sample: np.ndarray, stats: NormalizationStats) -> torch.Tensor:
    array = np.asarray(sample)
    if not np.isfinite(array).all():
        raise ValueError("Sample contains NaN or Inf values")

    tensor = torch.tensor(array, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected sample with 2 or 3 dims, got shape {tuple(tensor.shape)}")

    min_value = float(stats.min_value)
    max_value = float(stats.max_value)
    if max_value < min_value:
        raise ValueError(f"Invalid normalization stats [{min_value}, {max_value}]")
    if max_value == min_value:
        return torch.zeros_like(tensor)

    tensor = tensor.clamp(min_value, max_value)
    tensor = tensor.sub(min_value).div(max_value - min_value)
    return tensor.mul(2.0).sub(1.0)


def denormalize_sample(sample: torch.Tensor, stats: NormalizationStats | None = None) -> torch.Tensor:
    sample = sample.add(1.0).div(2.0)
    if stats is None:
        return torch.clamp(sample, 0.0, 1.0)
    return sample.mul(stats.max_value - stats.min_value).add(stats.min_value)


def _split_pair(pair: DatasetPair, train: bool, test_size: float, seed: int) -> DatasetPair:
    total = pair.observation.shape[0]
    if total < 2:
        return pair

    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)
    test_count = max(1, int(round(total * test_size)))
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    if train_idx.size == 0:
        train_idx = test_idx
    index = train_idx if train else test_idx
    return DatasetPair(pair.observation[index], pair.target[index])


def _load_density_pair(dataset_file: Path) -> DatasetPair:
    array = np.load(dataset_file, mmap_mode="r")
    if array.shape[0] != 2:
        raise ValueError(
            f"Density dataset must have shape [2, N, H, W], got {array.shape} from {dataset_file}"
        )
    return DatasetPair(array[0], array[1])


def _npz_member_memmap(dataset_file: Path, member_name: str) -> np.ndarray:
    with zipfile.ZipFile(dataset_file) as archive:
        info = archive.getinfo(member_name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                f"{dataset_file} member {member_name} is compressed; AstroDSB expects stored .npy members."
            )

        with dataset_file.open("rb") as handle:
            handle.seek(info.header_offset)
            header = handle.read(30)
            signature, *_rest, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", header)
            if signature != 0x04034B50:
                raise ValueError(f"Invalid zip local header for {member_name} in {dataset_file}")

            npy_offset = info.header_offset + 30 + name_len + extra_len
            handle.seek(npy_offset)
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"Unsupported .npy header version {version} in {dataset_file}:{member_name}")
            array_offset = handle.tell()

    order = "F" if fortran_order else "C"
    return np.memmap(dataset_file, mode="r", dtype=dtype, shape=shape, order=order, offset=array_offset)


def _load_npz_pair(dataset_file: Path, observation_key: str, target_key: str) -> DatasetPair:
    with zipfile.ZipFile(dataset_file) as archive:
        names = set(archive.namelist())

    observation_member = f"{observation_key}.npy"
    target_member = f"{target_key}.npy"
    if observation_member not in names or target_member not in names:
        raise KeyError(
            f"Expected keys {observation_key!r} and {target_key!r} in {dataset_file}, found {sorted(names)}"
        )

    return DatasetPair(
        observation=_npz_member_memmap(dataset_file, observation_member),
        target=_npz_member_memmap(dataset_file, target_member),
    )


def _validate_pair(pair: DatasetPair, *, task: str) -> DatasetPair:
    observation = np.asarray(pair.observation)
    target = np.asarray(pair.target)

    if not np.isfinite(observation).all() or not np.isfinite(target).all():
        raise ValueError(f"Found NaN/Inf in task={task} dataset")
    if observation.shape[0] != target.shape[0]:
        raise ValueError(
            f"Mismatched sample counts for task={task}: {observation.shape[0]} observations vs {target.shape[0]} targets"
        )

    if observation.ndim == 3:
        observation = observation[:, None, :, :]
    if target.ndim == 3:
        target = target[:, None, :, :]

    if observation.ndim != 4 or target.ndim != 4:
        raise ValueError(
            f"Expected observation/target arrays with shape [N,C,H,W], got {observation.shape} and {target.shape}"
        )
    if observation.shape[-2:] != target.shape[-2:]:
        raise ValueError(
            f"Observation/target spatial mismatch: {observation.shape[-2:]} vs {target.shape[-2:]}"
        )
    if target.shape[1] != 1:
        raise ValueError(f"AstroDSB expects a single-channel target x1, got {target.shape[1]} channels")
    if task == "density" and observation.shape[1] != 1:
        raise ValueError(f"Density task expects a single-channel observation, got {observation.shape[1]}")

    return DatasetPair(observation=observation, target=target)


class AstroDataset(torch.utils.data.Dataset):
    def __init__(self, pair: DatasetPair, normalization: DatasetNormalization):
        super().__init__()
        self.observation = pair.observation
        self.target = pair.target
        self.normalization = normalization

    def __len__(self) -> int:
        return int(self.observation.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        y = normalize_array(self.observation[index], self.normalization.observation)
        x1 = normalize_array(self.target[index], self.normalization.target)
        return {"y": y, "x1": x1}

    def describe_ranges(self) -> dict[str, float]:
        return {
            "observation_min": float(np.nanmin(self.observation)),
            "observation_max": float(np.nanmax(self.observation)),
            "target_min": float(np.nanmin(self.target)),
            "target_max": float(np.nanmax(self.target)),
            "normalized_observation_min": -1.0,
            "normalized_observation_max": 1.0,
            "normalized_target_min": -1.0,
            "normalized_target_max": 1.0,
            "num_samples": int(self.observation.shape[0]),
        }

    def get_normalization_state(self) -> dict[str, dict[str, float | str]]:
        return asdict(self.normalization)


def resolve_dataset_pair(config, train: bool) -> tuple[DatasetPair, DatasetNormalization]:
    task = getattr(config, "task", "density")
    dataset_file = getattr(config, "dataset_file", None)
    dataset_file = Path(dataset_file) if dataset_file is not None else (
        _default_density_file() if task == "density" else _default_mag_file()
    )
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    if task == "density":
        full_pair = _load_density_pair(dataset_file)
        pair = _split_pair(
            full_pair,
            train=train,
            test_size=getattr(config, "dataset_test_split", 0.2),
            seed=getattr(config, "dataset_split_seed", 42),
        )
    elif task == "mag":
        observation_key = getattr(config, "dataset_observation_key", None) or ("X_train" if train else "X_test")
        target_key = getattr(config, "dataset_target_key", None) or ("Y_train" if train else "Y_test")
        full_pair = _load_npz_pair(dataset_file, observation_key=observation_key, target_key=target_key)
        pair = full_pair
    else:
        raise ValueError(f"Unsupported AstroDSB task: {task}")

    pair = _validate_pair(pair, task=task)
    full_pair = _validate_pair(full_pair, task=task)
    mode = _resolve_normalization_mode(config, dataset_file)
    normalization = DatasetNormalization(
        observation=_build_stats(full_pair.observation, mode),
        target=_build_stats(full_pair.target, mode),
    )
    return pair, normalization


def build_astro_dataset(config, train: bool = True) -> AstroDataset:
    pair, normalization = resolve_dataset_pair(config, train=train)
    return AstroDataset(pair, normalization)
