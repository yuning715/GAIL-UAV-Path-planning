"""Dynamic breakthrough environment with radar, loitering UAVs, and evacuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.envs.entities import LoiteringUAV
from src.envs.reward import shaped_progress_reward
from src.envs.uav_dynamics import UAVDynamics2D, state6
from src.utils.math_utils import path_length


@dataclass
class BreakthroughRecord:
    """Episode metrics collected from a breakthrough rollout."""

    success: bool
    failure_reason: str
    time_to_target: float
    cumulative_time: float
    path_length_to_target: float
    total_path_length: float
    minimum_relative_distance: float
    average_relative_distance: float


class BreakthroughEnv:
    """Controlled 2D defensive-line breakthrough task."""

    def __init__(self, config: dict[str, Any], seed: int = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.dt = float(config.get("dt", 1.0))
        self.max_steps = int(config.get("single_episode_max_steps", 300))
        self.start = np.array(config.get("start", [-920.0, 0.0]), dtype=np.float32)
        self.home = np.array(config.get("home", self.start), dtype=np.float32)
        self.target_device = np.array(config.get("target_device", [520.0, 0.0]), dtype=np.float32)
        self.radar_center = np.array(config.get("radar_center", [180.0, 0.0]), dtype=np.float32)
        self.radar_radius = float(config.get("radar_radius", 620.0))
        self.intercept_radius = float(config.get("intercept_radius", 45.0))
        self.target_radius = float(config.get("target_radius", 45.0))
        self.exit_radius = float(config.get("exit_radius", 80.0))
        self.dynamics = UAVDynamics2D(
            dt=self.dt,
            max_speed=float(config.get("max_speed", 18.0)),
            max_accel=float(config.get("max_accel", 4.0)),
        )
        self.loiter_cfg = config.get("loitering_uavs", {})
        self.reset(seed=seed)

    def reset(self, seed: int | None = None, target: np.ndarray | None = None) -> np.ndarray:
        """Reset the tested UAV and loitering UAVs."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.position = self.start.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.goal = np.array(target, dtype=np.float32) if target is not None else self.target_device.copy()
        self.phase = "approach"
        self.step_count = 0
        self.reached_target_step: int | None = None
        self.path: list[np.ndarray] = [self.position.copy()]
        self.actions: list[np.ndarray] = []
        self.distance_log: list[float] = []
        count = int(self.loiter_cfg.get("count", 3))
        patrol_radius = float(self.loiter_cfg.get("patrol_radius", 330.0))
        speed = float(self.loiter_cfg.get("speed", 13.0))
        intercept_speed = float(self.loiter_cfg.get("intercept_speed", 15.0))
        self.loiterers = [
            LoiteringUAV(
                center=self.radar_center.copy(),
                patrol_radius=patrol_radius,
                phase=2 * np.pi * i / count,
                speed=speed,
                intercept_speed=intercept_speed,
            )
            for i in range(count)
        ]
        return self._obs()

    def _obs(self) -> np.ndarray:
        goal = self.goal if self.phase == "approach" else self.home
        return state6(self.position, self.velocity, goal)

    def _in_radar(self) -> bool:
        return bool(np.linalg.norm(self.position - self.radar_center) <= self.radar_radius)

    def dynamic_obstacles(self) -> list[np.ndarray]:
        """Return current loitering UAV positions."""
        return [u.position.copy() for u in self.loiterers]

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance one time step and return Gym-like transition values."""
        previous = self.position.copy()
        self.position, self.velocity = self.dynamics.step(self.position, self.velocity, action)
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        detected = self._in_radar()
        for uav in self.loiterers:
            uav.mode = "intercept" if detected else "patrol"
            uav.step(self.dt, self.position if detected else None)
        distances = [float(np.linalg.norm(self.position - u.position)) for u in self.loiterers]
        min_distance = min(distances)
        self.distance_log.append(min_distance)
        self.path.append(self.position.copy())
        self.step_count += 1

        goal = self.goal if self.phase == "approach" else self.home
        reward = shaped_progress_reward(previous, self.position, goal, np.asarray(action), self.dynamic_obstacles(), 70.0)
        done = False
        reason = ""
        if min_distance <= self.intercept_radius:
            reward -= 120.0
            done = True
            reason = "intercepted"
        elif self.phase == "approach" and np.linalg.norm(self.position - self.goal) <= self.target_radius:
            self.phase = "evacuate"
            self.reached_target_step = self.step_count
            reward += 80.0
        elif self.phase == "evacuate" and (
            np.linalg.norm(self.position - self.home) <= self.exit_radius
            or np.linalg.norm(self.position - self.radar_center) >= self.radar_radius + self.exit_radius
        ):
            reward += 120.0
            done = True
            reason = "success"
        elif self.step_count >= self.max_steps:
            done = True
            reason = "timeout"
            reward -= 30.0

        info = {
            "phase": self.phase,
            "failure_reason": reason,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "loitering_positions": np.array(self.dynamic_obstacles()),
            "distance_to_loitering": np.array(distances, dtype=np.float32),
            "success": reason == "success",
            "reached_target": self.reached_target_step is not None,
        }
        return self._obs(), float(reward), done, info

    def record(self, success_override: bool | None = None, reason: str | None = None) -> BreakthroughRecord:
        """Compute path metrics from the rollout state logs."""
        points = np.array(self.path, dtype=np.float32)
        if self.reached_target_step is None:
            reach_idx = len(points) - 1
            time_to_target = float(self.step_count * self.dt)
        else:
            reach_idx = self.reached_target_step
            time_to_target = float(reach_idx * self.dt)
        distances = np.array(self.distance_log if self.distance_log else [np.inf], dtype=np.float32)
        success = bool(success_override) if success_override is not None else (
            self.phase == "evacuate"
            and (
                np.linalg.norm(self.position - self.home) <= self.exit_radius
                or np.linalg.norm(self.position - self.radar_center) >= self.radar_radius + self.exit_radius
            )
        )
        failure_reason = reason or ("success" if success else "timeout")
        return BreakthroughRecord(
            success=success,
            failure_reason=failure_reason,
            time_to_target=time_to_target,
            cumulative_time=float(self.step_count * self.dt),
            path_length_to_target=path_length(points[: reach_idx + 1]),
            total_path_length=path_length(points),
            minimum_relative_distance=float(np.min(distances)),
            average_relative_distance=float(np.mean(distances[np.isfinite(distances)])),
        )
