"""Entity definitions shared by UAV environments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Obstacle:
    """Circular or spherical obstacle."""

    center: np.ndarray
    radius: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    def step(self, dt: float) -> None:
        """Advance the obstacle center using its velocity."""
        self.center = self.center + self.velocity * dt


@dataclass
class UAVState2D:
    """Planar UAV state used by controlled experiments."""

    position: np.ndarray
    velocity: np.ndarray
    theta: float = 0.0


@dataclass
class LoiteringUAV:
    """Patrolling UAV that switches to intercept mode after detection."""

    center: np.ndarray
    patrol_radius: float
    phase: float
    speed: float
    intercept_speed: float
    position: np.ndarray | None = None
    mode: str = "patrol"

    def __post_init__(self) -> None:
        if self.position is None:
            self.position = self.center + self.patrol_radius * np.array(
                [np.cos(self.phase), np.sin(self.phase)], dtype=np.float32
            )

    def step(self, dt: float, target: np.ndarray | None = None) -> None:
        """Patrol around the center or move toward the tested UAV."""
        if self.mode == "intercept" and target is not None:
            delta = target - self.position
            dist = np.linalg.norm(delta)
            if dist > 1e-8:
                self.position = self.position + delta / dist * self.intercept_speed * dt
            return
        angular_speed = self.speed / max(self.patrol_radius, 1.0)
        self.phase += angular_speed * dt
        self.position = self.center + self.patrol_radius * np.array(
            [np.cos(self.phase), np.sin(self.phase)], dtype=np.float32
        )

