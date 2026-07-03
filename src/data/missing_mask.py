"""Missing-data mask generation for sequential trajectory inpainting."""

from __future__ import annotations

import numpy as np


def random_missing_mask(shape: tuple[int, int], missing_rate: float, seed: int = 0) -> np.ndarray:
    """Create an observed=1, missing=0 random mask."""
    rng = np.random.default_rng(seed)
    mask = rng.random(shape) >= missing_rate
    return mask.astype(np.float32)


def retained_mask(shape: tuple[int, int], retained_rate: float, seed: int = 0) -> np.ndarray:
    """Create a random mask that retains the requested fraction."""
    return random_missing_mask(shape, missing_rate=1.0 - retained_rate, seed=seed)


def block_missing_mask(shape: tuple[int, int], missing_rate: float, seed: int = 0) -> np.ndarray:
    """Create a continuous missing block of approximately missing_rate*T."""
    rng = np.random.default_rng(seed)
    t, f = shape
    mask = np.ones(shape, dtype=np.float32)
    block = max(1, int(round(t * missing_rate)))
    start = int(rng.integers(0, max(1, t - block + 1)))
    mask[start : start + block, :] = 0.0
    return mask


def interval_missing_mask(shape: tuple[int, int], interval: tuple[int, int]) -> np.ndarray:
    """Mask a fixed interval, clipping to the trajectory length."""
    t, _ = shape
    start = max(0, min(t, int(interval[0])))
    end = max(start, min(t, int(interval[1])))
    mask = np.ones(shape, dtype=np.float32)
    mask[start:end, :] = 0.0
    return mask


def compose_masks(*masks: np.ndarray) -> np.ndarray:
    """Combine multiple observed/missing masks."""
    if not masks:
        raise ValueError("At least one mask is required.")
    out = masks[0].copy()
    for mask in masks[1:]:
        out = out * mask
    return out.astype(np.float32)

