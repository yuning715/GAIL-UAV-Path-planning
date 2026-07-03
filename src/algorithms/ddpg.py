"""DDPG actor-critic baseline with replay buffer and target networks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.algorithms.replay_buffer import ReplayBuffer


class MLPActor(nn.Module):
    """Small actor network for controlled 2D experiments."""

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list[int], max_action: float = 4.0):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for hidden in hidden_layers:
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(state)) * self.max_action


class MLPCritic(nn.Module):
    """Q-value critic over state-action pairs."""

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim + action_dim
        for hidden in hidden_layers:
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


@dataclass
class DDPGConfig:
    """DDPG baseline hyperparameters."""

    gamma: float = 0.5
    learning_rate: float = 3e-4
    tau: float = 0.005
    batch_size: int = 128
    ou_beta: float = 0.1
    ou_sigma_1: float = 0.2
    gaussian_variance_initial: float = 0.5
    gaussian_attenuation_delta: float = 0.001


class DDPGAgent:
    """DDPG agent with OU and Gaussian exploration noise."""

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list[int], config: DDPGConfig, seed: int = 0):
        self.actor = MLPActor(state_dim, action_dim, hidden_layers)
        self.critic = MLPCritic(state_dim, action_dim, hidden_layers)
        self.target_actor = MLPActor(state_dim, action_dim, hidden_layers)
        self.target_critic = MLPCritic(state_dim, action_dim, hidden_layers)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.ou_state = np.zeros(action_dim, dtype=np.float32)

    def act(self, state: np.ndarray, episode: int = 0, explore: bool = True) -> np.ndarray:
        with torch.no_grad():
            action = self.actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0).cpu().numpy()
        if explore:
            self.ou_state = self.ou_state + self.config.ou_beta * (-self.ou_state) + self.config.ou_sigma_1 * self.rng.normal(
                size=self.ou_state.shape
            )
            variance = self.config.gaussian_variance_initial * np.exp(-self.config.gaussian_attenuation_delta * episode)
            action = action + self.ou_state + self.rng.normal(0.0, np.sqrt(variance), size=action.shape)
        return np.asarray(action, dtype=np.float32)

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        if len(buffer) < self.config.batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0}
        states, actions, rewards, next_states, dones = buffer.sample(self.config.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.float32)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)
        with torch.no_grad():
            next_actions = self.target_actor(next_states_t)
            target_q = rewards_t + self.config.gamma * (1.0 - dones_t) * self.target_critic(next_states_t, next_actions)
        q = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(q, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        self.soft_update()
        return {"actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach())}

    def soft_update(self) -> None:
        for target, source in zip(self.target_actor.parameters(), self.actor.parameters()):
            target.data.mul_(1.0 - self.config.tau).add_(source.data * self.config.tau)
        for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.data.mul_(1.0 - self.config.tau).add_(source.data * self.config.tau)

