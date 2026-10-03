"""Global seeding for random/numpy/torch and deterministic flags."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 0, deterministic: bool = False) -> None:
    """Seed every RNG used by the pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
