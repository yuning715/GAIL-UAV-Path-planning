"""Continuous Gaussian policy network for PPO and GAIL-PPO."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

from src.models.attention import SelfAttentionBlock


class PolicyNetwork(nn.Module):
    """FC feature extractor, LSTM, self-attention, and Gaussian output heads."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | tuple[int, int] = (256, 128),
        lstm_hidden_size: int = 128,
        action_std_min: float = 0.05,
        action_std_max: float = 1.5,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        self.feature = nn.Sequential(*layers)
        self.lstm = nn.LSTM(in_dim, lstm_hidden_size, batch_first=True)
        self.attention = SelfAttentionBlock(lstm_hidden_size)
        self.mean_head = nn.Linear(lstm_hidden_size, action_dim)
        self.std_head = nn.Linear(lstm_hidden_size, action_dim)
        self.action_std_min = action_std_min
        self.action_std_max = action_std_max

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action mean and standard deviation."""
        squeeze_time = False
        if state.dim() == 2:
            state = state.unsqueeze(1)
            squeeze_time = True
        b, t, d = state.shape
        x = self.feature(state.reshape(b * t, d)).reshape(b, t, -1)
        x, _ = self.lstm(x)
        x = self.attention(x)
        pooled = x[:, -1]
        mean = self.mean_head(pooled)
        std = F.softplus(self.std_head(pooled)) + self.action_std_min
        std = torch.clamp(std, self.action_std_min, self.action_std_max)
        if squeeze_time:
            return mean, std
        return mean, std

    def distribution(self, state: torch.Tensor) -> Normal:
        """Create a diagonal Normal action distribution."""
        mean, std = self.forward(state)
        return Normal(mean, std)

    def act(self, state: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample or select an action and return its log probability."""
        dist = self.distribution(state)
        action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

