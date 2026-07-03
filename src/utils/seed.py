"""Deterministic seeding for Python, NumPy, PyTorch, CUDA, and environments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is a required runtime dependency
    torch = None


@dataclass(frozen=True)
class SeedState:
    """Record the seed used by a reproducible run."""

    seed: int


def set_seed(seed: int, deterministic_torch: bool = True) -> SeedState:
    """Seed all supported random generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return SeedState(seed=seed)


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Create a NumPy generator for deterministic local randomness."""
    return np.random.default_rng(seed)

