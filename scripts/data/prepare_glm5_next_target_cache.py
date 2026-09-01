#!/usr/bin/env python3
"""Precompute the offline GLM-5.3 target cache with the distributed runner."""

# ruff: noqa: E402

from __future__ import annotations

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from scripts.data.prepare_deepseek_v4_target_cache import main


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        main(int(os.environ["LOCAL_RANK"]))
    else:
        torch.multiprocessing.spawn(main, nprocs=torch.cuda.device_count())
