"""Attention layers used across sequence models."""

from __future__ import annotations

import torch
from torch import nn


class TemporalAttention(nn.Module):
    """Additive temporal attention over an LSTM output sequence."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Tanh(), nn.Linear(input_dim, 1))

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return context vector and attention weights for [B, T, D]."""
        logits = self.score(sequence).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return context, weights


class SelfAttentionBlock(nn.Module):
    """Single multi-head self-attention block with residual normalization."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        heads = max(1, min(num_heads, embed_dim))
        while embed_dim % heads != 0 and heads > 1:
            heads -= 1
        self.attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + out)

