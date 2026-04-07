from .AllData import build_astro_dataset, denormalize_sample, normalize_array, resolve_dataset_pair
from .MAGData import build_mag_dataset


def build_task_dataset(config, train=True):
    return build_astro_dataset(config, train=train) if config.task == "density" else build_mag_dataset(config, train=train)
