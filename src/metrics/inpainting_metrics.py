"""MER, TMSE, and MSER metrics from the sequential inpainting experiment."""

from __future__ import annotations

import numpy as np


def mer(pred: np.ndarray, real: np.ndarray) -> float:
    """Mean error rate: sum |x'-x| / [N * (max(x)-min(x))]."""
    pred = np.asarray(pred)
    real = np.asarray(real)
    denom = real.size * max(float(real.max() - real.min()), 1e-8)
    return float(np.abs(pred - real).sum() / denom)


def tmse(pred: np.ndarray, real: np.ndarray) -> float:
    """Time mean square error."""
    pred = np.asarray(pred)
    real = np.asarray(real)
    return float(np.square(pred - real).sum() / real.size)


def mser(pred: np.ndarray, real: np.ndarray) -> float:
    """Maximum single error rate."""
    pred = np.asarray(pred)
    real = np.asarray(real)
    denom = max(float(real.max() - real.min()), 1e-8)
    return float(np.abs(pred - real).max() / denom)


def metric_table(pred: np.ndarray, real: np.ndarray, mask: np.ndarray, feature_names: list[str]) -> list[dict[str, float | str]]:
    """Compute metrics on missing entries for each feature."""
    rows = []
    for idx, name in enumerate(feature_names):
        missing = mask[:, idx] < 0.5
        if not missing.any():
            missing = np.ones(mask.shape[0], dtype=bool)
        p = pred[missing, idx]
        r = real[missing, idx]
        rows.append({"data_type": name, "MER": mer(p, r), "TMSE": tmse(p, r), "MSER": mser(p, r)})
    return rows

