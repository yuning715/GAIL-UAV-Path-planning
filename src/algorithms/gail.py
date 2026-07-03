"""GAIL discriminator updates and reward shaping."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GAILConfig:
    """GAIL reward and discriminator settings."""

    discriminator_learning_rate: float = 1e-4
    reward_weight: float = 1.0
    task_reward_beta: float = 0.1
    discriminator_update_frequency: int = 5


class GAILTrainer:
    """Update the state-action discriminator and compute imitation rewards."""

    def __init__(self, discriminator: nn.Module, config: GAILConfig):
        self.discriminator = discriminator
        self.config = config
        self.optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.discriminator_learning_rate)

    def discriminator_step(
        self,
        expert_states: torch.Tensor,
        expert_actions: torch.Tensor,
        policy_states: torch.Tensor,
        policy_actions: torch.Tensor,
    ) -> dict[str, float]:
        """Train discriminator with expert positives and policy negatives."""
        expert_prob = self.discriminator(expert_states, expert_actions)
        policy_prob = self.discriminator(policy_states, policy_actions)
        loss = F.binary_cross_entropy(expert_prob, torch.ones_like(expert_prob)) + F.binary_cross_entropy(
            policy_prob, torch.zeros_like(policy_prob)
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            tp = int((expert_prob >= 0.5).sum().item())
            fn = int((expert_prob < 0.5).sum().item())
            tn = int((policy_prob < 0.5).sum().item())
            fp = int((policy_prob >= 0.5).sum().item())
            acc = (tp + tn) / max(tp + tn + fp + fn, 1)
            ppv = tp / max(tp + fp, 1)
            tpr = tp / max(tp + fn, 1)
        return {
            "discriminator_loss": float(loss.detach().cpu()),
            "ACC": float(acc),
            "PPV": float(ppv),
            "TPR": float(tpr),
        }

    @torch.no_grad()
    def reward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute r_GAIL(s,a) = -log(1-D(s,a))."""
        prob = self.discriminator(states, actions).clamp(1e-6, 1.0 - 1e-6)
        return -torch.log(1.0 - prob) * self.config.reward_weight

