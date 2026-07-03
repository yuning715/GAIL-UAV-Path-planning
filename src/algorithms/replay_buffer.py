"""Replay buffer for DDPG."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-capacity replay buffer."""

    def __init__(self, capacity: int, seed: int = 0):
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, done) -> None:
        self.buffer.append(
            Transition(
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[int(i)] for i in idx]
        return (
            np.stack([b.state for b in batch]),
            np.stack([b.action for b in batch]),
            np.asarray([b.reward for b in batch], dtype=np.float32),
            np.stack([b.next_state for b in batch]),
            np.asarray([b.done for b in batch], dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

