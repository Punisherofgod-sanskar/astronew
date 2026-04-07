# I2SB and AstroDSB

This repository contains two closely related codebases:

- `I2SB/` at the repository root: the original image-to-image Schrödinger Bridge implementation for image restoration and translation.
- `AstroDSB/`: an astrophysics-focused adaptation that applies the same diffusion-bridge idea to observational inverse problems such as density prediction and magnetic-field estimation.

The root project is based on NVIDIA's I2SB work, while `AstroDSB/` extends it with domain-specific datasets, evaluation scripts, plotting helpers, and single-node training workflows.

## Repository Layout

```text
.
├── train.py / sample.py / compute_metrices.py
│   Root training, sampling, and evaluation entry points for the original I2SB code.
├── i2sb/ , guided_diffusion/ , corruption/ , dataset/ , evaluation/
│   Core model, diffusion utilities, corruption operators, dataset loaders, and metrics.
├── scripts/
│   Helper scripts for training and checkpoint download.
├── assets/
│   Figures used by the original project README.
├── AstroDSB/
│   Astrophysics adaptation with its own training, evaluation, tests, assets, and docs.
└── 7015_Dynamic_Diffusion_Schr_di.pdf
    Local paper/reference artifact included in the workspace.
```

## Which README To Use

- Read this file for a high-level map of the repository.
- Read [`AstroDSB/README.md`](/home/slenderman/git/I2SB/AstroDSB/README.md) for the astrophysics-specific workflow, task definitions, and checkpoint links.

## Root I2SB Workflow

The root codebase is the original image restoration implementation. Main entry points:

- `train.py`: train a restoration model.
- `sample.py`: generate reconstructions from a checkpoint.
- `compute_metrices.py`: compute FID and classifier-based evaluation metrics.
- `scripts/download_checkpoint.sh`: download pretrained checkpoints for supported restoration tasks.

Core restoration tasks supported by the original code include:

- JPEG restoration
- 4x super-resolution
- Deblurring
- Inpainting

Typical environment setup:

```bash
conda env create --file requirements.yaml python=3
conda activate i2sb
```

Typical commands:

```bash
python train.py --name <experiment> --dataset-dir <lmdb_dir> --corrupt jpeg-5
python sample.py --ckpt <experiment> --dataset-dir <lmdb_dir> --n-gpu-per-node 1 --use-fp16
python compute_metrices.py --ckpt <experiment> --dataset-dir <lmdb_dir> --sample-dir samples_nfe1000
```

The original project expects ImageNet-style data prepared in LMDB format.

## AstroDSB Workflow

`AstroDSB/` is the actively adapted branch of the repository for astrophysical inversion tasks. It includes:

- paired dataset loading for density and magnetic tasks
- task-aware runtime validation
- evaluation for held-out samples and Taurus inference
- result plotting scripts for paper-quality figures
- a small test suite in `AstroDSB/tests/`

Primary AstroDSB entry points:

- [`AstroDSB/train.py`](/home/slenderman/git/I2SB/AstroDSB/train.py)
- [`AstroDSB/eval.py`](/home/slenderman/git/I2SB/AstroDSB/eval.py)
- [`AstroDSB/preprocess_taurus.py`](/home/slenderman/git/I2SB/AstroDSB/preprocess_taurus.py)
- [`AstroDSB/plot_density_results.py`](/home/slenderman/git/I2SB/AstroDSB/plot_density_results.py)
- [`AstroDSB/plot_magnetic_results.py`](/home/slenderman/git/I2SB/AstroDSB/plot_magnetic_results.py)

Important operational notes:

- AstroDSB currently validates for single-node execution only.
- GPU-enabled PyTorch is expected for real training runs.
- The tree already contains local runtime artifacts such as `AstroDSB/runs/`, `AstroDSB/__pycache__/`, and large observational data dumps under `AstroDSB/herschel_data/`.
- The new `.gitignore` prevents these local artifacts from being added in future, but it does not remove files that are already tracked.

## Development Notes

The repository currently mixes source code with generated artifacts and local experimental data. As a result:

- treat `results/`, `AstroDSB/results/`, `AstroDSB/runs/`, and log directories as disposable outputs
- keep large raw datasets outside version control when possible
- use the checked-in tests in `AstroDSB/tests/` to validate code changes in the astrophysics path

If you want to continue productizing this repository, a good next step would be separating source code from data and experiment outputs more aggressively.

## Licenses and Upstream Sources

- The original I2SB implementation is derived from NVIDIA's release and keeps the upstream license files in place.
- Additional third-party license notices are present under `guided_diffusion/` and `corruption/`.
- The astrophysics adaptation documents its paper and acknowledgements in [`AstroDSB/README.md`](/home/slenderman/git/I2SB/AstroDSB/README.md).
