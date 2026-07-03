"""PyTorch datasets for trajectory inpainting and state-action imitation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.missing_mask import random_missing_mask
from src.data.preprocessing import FeatureScaler


class InpaintingDataset(Dataset):
    """Sequence windows with generated missing masks."""

    def __init__(
        self,
        trajectories: list[np.ndarray],
        scaler: FeatureScaler,
        seq_len: int,
        stride: int,
        missing_rate: float,
        seed: int = 0,
    ):
        self.scaler = scaler
        self.seq_len = seq_len
        self.missing_rate = missing_rate
        self.seed = seed
        self.windows: list[np.ndarray] = []
        for traj in trajectories:
            normalized = scaler.transform(traj)
            if len(normalized) <= seq_len:
                self.windows.append(normalized)
            else:
                for start in range(0, len(normalized) - seq_len + 1, stride):
                    self.windows.append(normalized[start : start + seq_len])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full = self.windows[idx].astype(np.float32)
        mask = random_missing_mask(full.shape, self.missing_rate, seed=self.seed + idx)
        incomplete = full * mask
        return (
            torch.from_numpy(incomplete),
            torch.from_numpy(mask),
            torch.from_numpy(full),
        )


class StateActionDataset(Dataset):
    """State-action pairs for GAIL discriminator training."""

    def __init__(self, path: str | Path):
        data = np.load(Path(path))
        self.states = data["states"].astype(np.float32)
        self.actions = data["actions"].astype(np.float32)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.states[idx]), torch.from_numpy(self.actions[idx])

