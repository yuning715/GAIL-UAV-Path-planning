"""GAN-LSTM generator for sequential trajectory inpainting."""

from __future__ import annotations

import torch
from torch import nn

from src.models.attention import TemporalAttention


class TrajectoryGenerator(nn.Module):
    """BiLSTM encoder, temporal attention, LSTM decoder, and FC output head."""

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_activation: str = "tanh",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.encoder = nn.LSTM(
            input_size=feature_dim * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.attention = TemporalAttention(hidden_size * 2)
        self.decoder = nn.LSTM(
            input_size=hidden_size * 4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, feature_dim)
        self.activation_name = output_activation

    def forward(self, incomplete: torch.Tensor, mask: torch.Tensor, preserve_observed: bool = False) -> torch.Tensor:
        """Repair a [B, T, F] trajectory given an observed=1 mask."""
        x = torch.cat([incomplete, mask], dim=-1)
        encoded, _ = self.encoder(x)
        context, _ = self.attention(encoded)
        repeated_context = context.unsqueeze(1).expand(-1, encoded.size(1), -1)
        decoded, _ = self.decoder(torch.cat([encoded, repeated_context], dim=-1))
        repaired = self.output(decoded)
        if self.activation_name == "tanh":
            repaired = torch.tanh(repaired)
        if preserve_observed:
            repaired = repaired * (1.0 - mask) + incomplete * mask
        return repaired

