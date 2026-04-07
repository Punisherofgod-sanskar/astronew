from .AllData import AstroDataset, resolve_dataset_pair


def build_mag_dataset(config, train: bool = True) -> AstroDataset:
    pair, normalization = resolve_dataset_pair(config, train=train)
    return AstroDataset(pair, normalization)
