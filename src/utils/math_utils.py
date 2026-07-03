"""Small numerical helpers used by UAV controllers and metrics."""

from __future__ import annotations

import numpy as np


def norm(vec: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    """Euclidean norm with a small epsilon for numerical stability."""
    return np.linalg.norm(vec, axis=axis, keepdims=keepdims)


def unit(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a unit vector; zero vectors remain zero."""
    n = np.linalg.norm(vec)
    if n < eps:
        return np.zeros_like(vec)
    return vec / n


def clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip a vector by norm."""
    n = np.linalg.norm(vec)
    if n <= max_norm or n < 1e-12:
        return vec
    return vec * (max_norm / n)


def path_length(points: np.ndarray) -> float:
    """Compute cumulative length of a sequence of points."""
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

