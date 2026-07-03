"""Value function network for PPO."""

from __future__ import annotations

import torch
from torch import nn


class ValueNetwork(nn.Module):
    """MLP critic with scalar output."""

    def __init__(self, state_dim: int, hidden_sizes: list[int] | tuple[int, ...] = (256, 128, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)

