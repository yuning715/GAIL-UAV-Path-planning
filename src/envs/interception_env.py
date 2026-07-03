"""Dynamic interception environment with target UAV groups and obstacles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.envs.entities import Obstacle
from src.envs.reward import shaped_progress_reward
from src.envs.uav_dynamics import UAVDynamics2D, state6
from src.utils.math_utils import path_length


@dataclass
class InterceptionRecord:
    """Metrics for one interception episode."""

    success: bool
    score: int
    intercepted: int
    contact_intercepted: int
    escaped: int
    total_targets: int
    episode_time: float
    path_length: float


class InterceptionEnv:
    """Controlled 2D task for blocking UAV groups inside an interception zone."""

    def __init__(self, config: dict[str, Any], seed: int = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.dt = float(config.get("dt", 1.0))
        self.max_group_steps = int(float(config.get("group_interval_minutes", 15)) * 60 / self.dt)
        self.start = np.array(config.get("start", [0.0, -360.0]), dtype=np.float32)
        self.map_size = np.array(config.get("map_size", [2000.0, 1300.0]), dtype=np.float32)
        self.zone_x = np.array(config.get("zone_x", [-650.0, 650.0]), dtype=np.float32)
        self.zone_y = np.array(config.get("zone_y", [-420.0, 420.0]), dtype=np.float32)
        self.intercept_radius = float(config.get("intercept_radius", 58.0))
        self.target_speed = float(config.get("target_speed", 8.0))
        self.dynamics = UAVDynamics2D(
            dt=self.dt,
            max_speed=float(config.get("max_speed", 19.0)),
            max_accel=float(config.get("max_accel", 4.0)),
        )
        self.obstacles = [
            Obstacle(center=np.array(o["center"], dtype=np.float32), radius=float(o["radius"]))
            for o in config.get("obstacles", [])
        ]
        self.reset(seed=seed)

    def reset(self, seed: int | None = None, group_size: int | None = None) -> np.ndarray:
        """Reset tested UAV and spawn a target group."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.position = self.start.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.group_size = int(group_size or self.rng.integers(1, int(self.config.get("max_group_size", 5)) + 1))
        offsets = np.linspace(-120.0, 120.0, self.group_size, dtype=np.float32)
        self.targets = np.stack(
            [
                np.array([self.zone_x[0] - 180.0 - 20.0 * i, offsets[i]], dtype=np.float32)
                for i in range(self.group_size)
            ]
        )
        self.target_active = np.ones(self.group_size, dtype=bool)
        self.target_intercepted = np.zeros(self.group_size, dtype=bool)
        self.contact_intercepted = np.zeros(self.group_size, dtype=bool)
        self.target_escaped = np.zeros(self.group_size, dtype=bool)
        self.path = [self.position.copy()]
        self.actions: list[np.ndarray] = []
        return self._obs()

    def nearest_active_target(self) -> np.ndarray:
        """Return the nearest target that has not been intercepted or escaped."""
        active_idx = np.where(self.target_active)[0]
        if len(active_idx) == 0:
            return np.array([self.zone_x[1], 0.0], dtype=np.float32)
        d = np.linalg.norm(self.targets[active_idx] - self.position[None, :], axis=1)
        return self.targets[active_idx[int(np.argmin(d))]].copy()

    def _obs(self) -> np.ndarray:
        return state6(self.position, self.velocity, self.nearest_active_target())

    def _move_targets(self, tested_position: np.ndarray) -> None:
        """Move targets through the zone with a deterministic evasive component."""
        for i in np.where(self.target_active)[0]:
            target = self.targets[i]
            forward = np.array([1.0, 0.0], dtype=np.float32)
            away = target - tested_position
            dist = np.linalg.norm(away)
            evasion = np.zeros(2, dtype=np.float32)
            if dist < 240.0 and dist > 1e-8:
                evasion = away / dist * 0.8
            obstacle_repulse = np.zeros(2, dtype=np.float32)
            for obs in self.obstacles:
                delta = target - obs.center
                od = np.linalg.norm(delta)
                if od < obs.radius + 110.0 and od > 1e-8:
                    obstacle_repulse += delta / od * (obs.radius + 110.0 - od) / 110.0
            direction = forward + evasion + obstacle_repulse
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                direction = forward
            else:
                direction = direction / norm
            self.targets[i] = target + direction * self.target_speed * self.dt

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance the interception episode."""
        previous = self.position.copy()
        self.position, self.velocity = self.dynamics.step(self.position, self.velocity, action)
        half = self.map_size / 2.0
        clipped = np.clip(self.position, -half, half)
        if not np.allclose(clipped, self.position):
            self.position = clipped.astype(np.float32)
            self.velocity = (self.velocity * 0.25).astype(np.float32)
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self._move_targets(self.position)
        for i in np.where(self.target_active)[0]:
            if np.linalg.norm(self.targets[i] - self.position) <= self.intercept_radius:
                self.target_active[i] = False
                self.target_intercepted[i] = True
                self.contact_intercepted[i] = True
            elif self.targets[i, 0] >= self.zone_x[1] + 120.0:
                self.target_active[i] = False
                self.target_escaped[i] = True
        self.path.append(self.position.copy())
        self.step_count += 1
        nearest = self.nearest_active_target()
        hazards = [obs.center for obs in self.obstacles]
        reward = shaped_progress_reward(previous, self.position, nearest, np.asarray(action), hazards, 80.0)
        reward += 20.0 * int(self.target_intercepted.sum()) - 15.0 * int(self.target_escaped.sum())
        done = not self.target_active.any() or self.step_count >= self.max_group_steps
        if done and self.step_count >= self.max_group_steps and bool(self.config.get("timeout_counts_as_success", True)):
            self.target_active[:] = False
            self.target_intercepted = np.logical_or(self.target_intercepted, ~self.target_escaped)
        elif done and self.step_count >= self.max_group_steps:
            self.target_escaped = np.logical_or(self.target_escaped, self.target_active)
            self.target_active[:] = False
        info = {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "targets": self.targets.copy(),
            "target_active": self.target_active.copy(),
            "intercepted": int(self.target_intercepted.sum()),
            "escaped": int(self.target_escaped.sum()),
            "success": int(self.target_escaped.sum()) == 0,
        }
        return self._obs(), float(reward), done, info

    def record(self) -> InterceptionRecord:
        """Compute metrics from the current episode."""
        intercepted = int(self.target_intercepted.sum())
        escaped = int(self.target_escaped.sum())
        score = intercepted - escaped
        return InterceptionRecord(
            success=escaped == 0,
            score=score,
            intercepted=intercepted,
            contact_intercepted=int(self.contact_intercepted.sum()),
            escaped=escaped,
            total_targets=int(self.group_size),
            episode_time=float(self.step_count * self.dt),
            path_length=path_length(np.array(self.path, dtype=np.float32)),
        )
