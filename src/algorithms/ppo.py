"""PPO update logic with clipped objective and GAE advantages."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.rollout_buffer import RolloutBuffer


@dataclass
class PPOConfig:
    """PPO hyperparameters."""

    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 0.99
    epochs_per_iteration: int = 10
    minibatch_size: int = 256
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5


class PPOTrainer:
    """Trainer for a Gaussian policy and value network."""

    def __init__(self, policy: nn.Module, value: nn.Module, config: PPOConfig):
        self.policy = policy
        self.value = value
        self.config = config
        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters()), lr=config.learning_rate
        )

    def update(self, buffer: RolloutBuffer, device: torch.device) -> dict[str, float]:
        """Run PPO epochs over the rollout buffer."""
        data = buffer.tensors(self.config.gamma, self.config.gae_lambda, device)
        dataset = TensorDataset(
            data["states"], data["actions"], data["old_log_probs"], data["advantages"], data["returns"]
        )
        loader = DataLoader(dataset, batch_size=min(self.config.minibatch_size, len(dataset)), shuffle=True)
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        steps = 0
        for _ in range(self.config.epochs_per_iteration):
            for states, actions, old_log_probs, advantages, returns in loader:
                dist = self.policy.distribution(states)
                log_probs = dist.log_prob(actions).sum(dim=-1)
                ratios = torch.exp(log_probs - old_log_probs)
                unclipped = ratios * advantages
                clipped = torch.clamp(
                    ratios, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon
                ) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                values = self.value(states)
                value_loss = torch.mean((values - returns) ** 2)
                entropy = dist.entropy().sum(dim=-1).mean()
                loss = policy_loss + self.config.value_coefficient * value_loss - self.config.entropy_coefficient * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()), self.config.max_grad_norm
                )
                self.optimizer.step()
                totals["policy_loss"] += float(policy_loss.detach().cpu())
                totals["value_loss"] += float(value_loss.detach().cpu())
                totals["entropy"] += float(entropy.detach().cpu())
                steps += 1
        return {k: v / max(steps, 1) for k, v in totals.items()}

