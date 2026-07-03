"""GAIL discriminator over state-action sequences."""

from __future__ import annotations

import torch
from torch import nn

from src.models.attention import TemporalAttention


class GAILDiscriminator(nn.Module):
    """State-action embedding, two-layer LSTM, attention pooling, sigmoid head."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embedding_dim: int = 256,
        lstm_hidden_size: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(state_dim + action_dim, embedding_dim), nn.ReLU())
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_size, num_layers=num_layers, batch_first=True)
        self.attention = TemporalAttention(lstm_hidden_size)
        self.classifier = nn.Sequential(nn.Linear(lstm_hidden_size, 1), nn.Sigmoid())

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return D(s, a) probabilities."""
        if states.dim() == 2:
            states = states.unsqueeze(1)
            actions = actions.unsqueeze(1)
        x = torch.cat([states, actions], dim=-1)
        x = self.embedding(x)
        x, _ = self.lstm(x)
        context, _ = self.attention(x)
        return self.classifier(context).squeeze(-1)

