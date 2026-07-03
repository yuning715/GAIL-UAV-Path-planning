"""Ant Colony Optimization baseline with receding-horizon replanning."""

from __future__ import annotations

import numpy as np

from src.utils.math_utils import clip_norm, unit


class ACOController:
    """Grid-free ACO-inspired waypoint search for dynamic environments."""

    def __init__(
        self,
        ants: int = 24,
        iterations: int = 4,
        horizon_steps: int = 12,
        grid_resolution: float = 50.0,
        pheromone_evaporation: float = 0.45,
        risk_weight: float = 2.5,
        max_accel: float = 4.0,
        seed: int = 0,
    ):
        self.ants = ants
        self.iterations = iterations
        self.horizon_steps = horizon_steps
        self.grid_resolution = grid_resolution
        self.pheromone_evaporation = pheromone_evaporation
        self.risk_weight = risk_weight
        self.max_accel = max_accel
        self.rng = np.random.default_rng(seed)
        self.pheromone: dict[tuple[int, int], float] = {}

    def _hazards(self, env) -> list[np.ndarray]:
        hazards = []
        if hasattr(env, "dynamic_obstacles"):
            hazards.extend(env.dynamic_obstacles())
        if hasattr(env, "obstacles"):
            hazards.extend([obs.center for obs in env.obstacles])
        if hasattr(env, "targets") and hasattr(env, "goal"):
            hazards.extend([t for t, active in zip(env.targets, env.target_active) if active])
        return hazards

    def _score_path(self, points: np.ndarray, goal: np.ndarray, hazards: list[np.ndarray]) -> float:
        distance_cost = np.linalg.norm(points[-1] - goal) + 0.1 * np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        risk = 0.0
        for p in points:
            for h in hazards:
                d = np.linalg.norm(p - h)
                risk += max(0.0, 180.0 - d) / 180.0
        key = tuple(np.round(points[1] / self.grid_resolution).astype(int))
        pheromone_bonus = self.pheromone.get(key, 0.0)
        return float(distance_cost + self.risk_weight * risk - 20.0 * pheromone_bonus)

    def act(self, env, state: np.ndarray | None = None) -> np.ndarray:
        goal = getattr(env, "goal", None)
        if goal is None:
            goal = env.nearest_active_target()
        elif getattr(env, "phase", "approach") == "evacuate":
            goal = env.home
        hazards = self._hazards(env)
        best_path = None
        best_score = float("inf")
        self.pheromone = {k: v * (1.0 - self.pheromone_evaporation) for k, v in self.pheromone.items()}
        for _ in range(self.iterations):
            for _ant in range(self.ants):
                pos = env.position.copy()
                velocity = env.velocity.copy()
                points = [pos.copy()]
                for _step in range(self.horizon_steps):
                    bias = unit(goal - pos)
                    random_dir = self.rng.normal(0.0, 1.0, size=2)
                    direction = unit(0.75 * bias + 0.25 * random_dir)
                    for h in hazards:
                        d = np.linalg.norm(pos - h)
                        if 1e-8 < d < 160.0:
                            direction += unit(pos - h) * (160.0 - d) / 160.0
                    velocity = clip_norm(velocity + unit(direction) * 2.5, 18.0)
                    pos = pos + velocity
                    points.append(pos.copy())
                arr = np.array(points, dtype=np.float32)
                score = self._score_path(arr, goal, hazards)
                if score < best_score:
                    best_score = score
                    best_path = arr
            if best_path is not None:
                key = tuple(np.round(best_path[1] / self.grid_resolution).astype(int))
                self.pheromone[key] = self.pheromone.get(key, 0.0) + 1.0 / max(best_score, 1e-6)
        waypoint = best_path[1] if best_path is not None and len(best_path) > 1 else goal
        desired_velocity = unit(waypoint - env.position) * 16.0
        return clip_norm((desired_velocity - env.velocity) * 0.45, self.max_accel)
