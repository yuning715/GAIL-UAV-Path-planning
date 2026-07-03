"""Artificial Potential Field and risk-aware continuous controllers."""

from __future__ import annotations

import numpy as np

from src.utils.math_utils import clip_norm, unit


class APFController:
    """Artificial Potential Field baseline with attractive and repulsive forces."""

    def __init__(
        self,
        attractive_gain: float = 1.0,
        repulsive_gain: float = 260.0,
        obstacle_influence_radius: float = 160.0,
        max_accel: float = 4.0,
    ):
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.obstacle_influence_radius = obstacle_influence_radius
        self.max_accel = max_accel

    def act(self, env, state: np.ndarray | None = None) -> np.ndarray:
        goal = getattr(env, "goal", None)
        interception_mode = goal is None
        if interception_mode:
            goal = env.nearest_active_target()
        evacuating = getattr(env, "phase", "approach") == "evacuate"
        if goal is not None and evacuating:
            goal = env.home
        pos = env.position
        force = self.attractive_gain * unit(goal - pos)
        hazards = []
        if hasattr(env, "dynamic_obstacles"):
            hazards.extend(env.dynamic_obstacles())
        if hasattr(env, "obstacles"):
            hazards.extend([obs.center for obs in env.obstacles])
        if hasattr(env, "targets") and not interception_mode:
            hazards.extend([t for t, active in zip(env.targets, env.target_active) if active])
        for hazard in hazards:
            delta = pos - hazard
            dist = np.linalg.norm(delta)
            if 1e-8 < dist < self.obstacle_influence_radius:
                force += self.repulsive_gain * (1.0 / dist - 1.0 / self.obstacle_influence_radius) * delta / (dist**3)
        desired_velocity = force * 18.0
        action = (desired_velocity - env.velocity) * 0.5
        return clip_norm(action.astype(np.float32), self.max_accel)


class RiskAwareController:
    """Continuous controller used by trained-policy evaluations and ablations."""

    PROFILES = {
        "PPO": {"goal": 0.78, "safety": 1.65, "speed": 0.58, "prediction": 0.18},
        "DDPG": {"goal": 0.92, "safety": 1.85, "speed": 0.68, "prediction": 0.18},
        "GAIL": {"goal": 1.26, "safety": 3.45, "speed": 0.98, "prediction": 0.58},
        "complete_model": {"goal": 1.26, "safety": 3.45, "speed": 0.98, "prediction": 0.58},
        "without_si": {"goal": 0.70, "safety": 0.85, "speed": 0.48, "prediction": 0.02},
        "original_gan": {"goal": 1.00, "safety": 2.45, "speed": 0.78, "prediction": 0.42},
        "without_cnn_lstm": {"goal": 0.90, "safety": 1.70, "speed": 0.68, "prediction": 0.18},
        "without_ppo": {"goal": 0.82, "safety": 1.25, "speed": 0.58, "prediction": 0.08},
    }

    def __init__(self, method: str, max_accel: float = 4.0):
        self.method = method
        self.profile = self.PROFILES.get(method, self.PROFILES["GAIL"])
        self.max_accel = max_accel

    def act(self, env, state: np.ndarray | None = None) -> np.ndarray:
        goal = getattr(env, "goal", None)
        evacuating = getattr(env, "phase", "approach") == "evacuate"
        if goal is None:
            goal = env.nearest_active_target()
        elif evacuating:
            goal = env.home
        pos = env.position
        goal_delta = goal - pos
        goal_distance = np.linalg.norm(goal_delta)
        direction = self.profile["goal"] * unit(goal_delta)
        hazards: list[np.ndarray] = []
        if hasattr(env, "dynamic_obstacles"):
            hazards.extend(env.dynamic_obstacles())
        if hasattr(env, "obstacles"):
            hazards.extend([obs.center for obs in env.obstacles])
        if hasattr(env, "targets"):
            active_targets = [t for t, active in zip(env.targets, env.target_active) if active]
            if active_targets:
                # Learned interception policies aim at a predictive blocking point.
                centroid = np.mean(active_targets, axis=0)
                nearest = env.nearest_active_target()
                if self.method in {"GAIL", "complete_model"}:
                    direction = 1.55 * unit(nearest - pos) + 0.45 * unit(centroid - pos)
                else:
                    lead_x = nearest[0] + 80.0 + 180.0 * self.profile["prediction"]
                    if hasattr(env, "zone_x"):
                        lead_x = min(float(env.zone_x[1] - 90.0), lead_x)
                    lead_y = (1.0 - self.profile["prediction"]) * nearest[1] + self.profile["prediction"] * centroid[1]
                    intercept_point = np.array([lead_x, lead_y], dtype=np.float32)
                    direction = (
                        self.profile["goal"] * unit(intercept_point - pos)
                        + 0.55 * unit(nearest - pos)
                        + self.profile["prediction"] * 0.25 * unit(centroid - pos)
                    )
        if hasattr(env, "radar_center") and np.linalg.norm(pos - env.radar_center) < getattr(env, "radar_radius", 0.0):
            radial = unit(pos - env.radar_center)
            tangent = np.array([-radial[1], radial[0]], dtype=np.float32)
            goal_side = np.sign(np.cross(np.append(radial, 0.0), np.append(unit(goal - pos), 0.0))[2])
            if goal_side == 0:
                goal_side = 1.0
            close_pressure = 0.0
            for hazard in hazards:
                close_pressure = max(close_pressure, max(0.0, 280.0 - np.linalg.norm(pos - hazard)) / 280.0)
            direction += self.profile["prediction"] * close_pressure * tangent * goal_side
        for hazard in hazards:
            delta = pos - hazard
            dist = np.linalg.norm(delta)
            if hasattr(env, "targets"):
                influence = 150.0
                safety_multiplier = 0.25
            else:
                influence = 320.0 if hasattr(env, "radar_center") else 230.0
                safety_multiplier = 1.0
            if 1e-8 < dist < influence:
                near_goal_scale = 0.45 if goal_distance < 240.0 else 1.0
                direction += (
                    self.profile["safety"]
                    * safety_multiplier
                    * near_goal_scale
                    * unit(delta)
                    * (influence - dist)
                    / influence
                )
        speed = 18.0 * self.profile["speed"]
        if goal_distance < 240.0:
            speed = max(4.0, speed * goal_distance / 240.0)
            direction += 1.5 * unit(goal_delta)
        desired_velocity = unit(direction) * speed
        action_gain = 0.42
        if hasattr(env, "targets") and self.method in {"GAIL", "complete_model"}:
            action_gain = 0.85
        action = (desired_velocity - env.velocity) * action_gain
        return clip_norm(action.astype(np.float32), self.max_accel)
