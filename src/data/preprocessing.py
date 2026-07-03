"""Feature scaling for trajectory inpainting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FeatureScaler:
    """Min-max scaler that maps features to [-1, 1]."""

    min_: np.ndarray
    max_: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        denom = np.maximum(self.max_ - self.min_, 1e-8)
        return ((x - self.min_) / denom * 2.0 - 1.0).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        denom = np.maximum(self.max_ - self.min_, 1e-8)
        return ((x + 1.0) * 0.5 * denom + self.min_).astype(np.float32)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, min=self.min_, max=self.max_)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        data = np.load(Path(path))
        return cls(min_=data["min"], max_=data["max"])


def fit_feature_scaler(features: list[np.ndarray]) -> FeatureScaler:
    """Fit a scaler over a list of [T, F] feature trajectories."""
    stacked = np.concatenate(features, axis=0)
    return FeatureScaler(min_=stacked.min(axis=0), max_=stacked.max(axis=0))

