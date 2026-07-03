"""Temporal CNN-LSTM discriminator for complete trajectories."""

from __future__ import annotations

import torch
from torch import nn

from src.models.attention import TemporalAttention


class TrajectoryDiscriminator(nn.Module):
    """CNN-LSTM discriminator with attention pooling and sigmoid output."""

    def __init__(
        self,
        feature_dim: int,
        cnn_filters: list[int] | tuple[int, ...] = (64, 128, 256),
        lstm_hidden_size: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = feature_dim
        for out_channels in cnn_filters:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)
        self.lstm = nn.LSTM(in_channels, lstm_hidden_size, batch_first=True)
        self.attention = TemporalAttention(lstm_hidden_size)
        self.classifier = nn.Sequential(nn.Linear(lstm_hidden_size, 1), nn.Sigmoid())

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Return authenticity probabilities for [B, T, F] inputs."""
        x = trajectory.transpose(1, 2)
        x = self.cnn(x).transpose(1, 2)
        x, _ = self.lstm(x)
        context, _ = self.attention(x)
        return self.classifier(context).squeeze(-1)


class MLPRNNDiscriminator(nn.Module):
    """FC-RNN discriminator used by the without-CNN-LSTM ablation."""

    def __init__(self, feature_dim: int, hidden_size: int = 128):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(feature_dim, hidden_size), nn.ReLU())
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        x = self.embedding(trajectory)
        x, _ = self.rnn(x)
        return self.classifier(x[:, -1]).squeeze(-1)

