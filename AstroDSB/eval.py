from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from pathlib import Path

import numpy as np
import torch

from astrodsb import Runner
from datasets_mod import build_task_dataset
from logger import Logger
from train import (
    _log_dataset_summary,
    _populate_channel_metadata,
    _validate_dataset_runtime,
    create_training_options,
    set_seed,
    validate_runtime_options,
)


def main(opt):
    opt = validate_runtime_options(opt)
    log = Logger(opt.global_rank, opt.log_dir)
    log.info("===================================================================================")
    log.info("   Astro-DSB Evaluator")
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

    val_dataset = build_task_dataset(opt, train=False)
    opt = _populate_channel_metadata(opt, val_dataset)
    _validate_dataset_runtime(opt)
    log.info(
        f"[Data] task={opt.task} obs_channels={opt.observation_channels} target_channels={opt.target_channels}"
    )
    if opt.task == "mag":
        log.info(
            f"[Data] mag_channel_schema={opt.mag_channel_schema} mag_bridge_mode={opt.mag_bridge_mode}"
        )
    _log_dataset_summary(log, "val", val_dataset)

    if opt.task == "mag":
        taurus_path = getattr(opt, "taurus_path", None)
        if taurus_path is not None and Path(taurus_path).exists():
            observation = np.load(str(taurus_path))
            if observation.shape[0] < opt.image_size or observation.shape[1] < opt.image_size:
                raise ValueError(
                    f"Taurus inference image must be at least patch-sized, got {observation.shape} for image_size={opt.image_size}"
                )

    run = Runner(opt, log, save_opt=False)
    run.eval(opt, val_dataset)
    log.info("Finish!")


if __name__ == "__main__":
    opt = create_training_options()
    opt.global_rank = 0
    opt.local_rank = 0
    opt.global_size = 1
    main(opt)
