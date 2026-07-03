"""UAV state-transition models for 3D and controlled 2D experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.math_utils import clip_norm


@dataclass
class UAVDynamics3D:
    """Paper MDP dynamics with state [p, v, theta, phi] and 5-D action."""

    dt: float = 1.0
    max_speed: float = 25.0

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Advance the 3D state using the transition equations from the paper."""
        state = np.asarray(state, dtype=np.float32).copy()
        action = np.asarray(action, dtype=np.float32)
        position = state[:3]
        velocity = state[3:6]
        theta = state[6]
        phi = state[7]
        velocity = velocity + action[:3] * self.dt
        speed = np.linalg.norm(velocity)
        if speed > self.max_speed:
            velocity = velocity * (self.max_speed / speed)
        position = position + velocity * self.dt
        theta = theta + action[3]
        phi = phi + action[4]
        return np.array([*position, *velocity, theta, phi], dtype=np.float32)


@dataclass
class UAVDynamics2D:
    """Planar double-integrator dynamics for controlled experiments."""

    dt: float = 1.0
    max_speed: float = 18.0
    max_accel: float = 4.0

    def step(self, position: np.ndarray, velocity: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance position and velocity using clipped acceleration."""
        action = clip_norm(np.asarray(action, dtype=np.float32), self.max_accel)
        velocity = np.asarray(velocity, dtype=np.float32) + action * self.dt
        velocity = clip_norm(velocity, self.max_speed)
        position = np.asarray(position, dtype=np.float32) + velocity * self.dt
        return position.astype(np.float32), velocity.astype(np.float32)


def state6(position: np.ndarray, velocity: np.ndarray, goal: np.ndarray, scale: float = 1000.0) -> np.ndarray:
    """Build the controlled experiment state vector of size 6."""
    rel = (goal - position) / scale
    return np.array(
        [position[0] / scale, position[1] / scale, velocity[0] / 20.0, velocity[1] / 20.0, rel[0], rel[1]],
        dtype=np.float32,
    )

