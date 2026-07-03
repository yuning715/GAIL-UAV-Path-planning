"""Reward functions matching the paper MDP and controlled experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardWeights:
    """Weights for the paper reward decomposition."""

    goal: float = 1.0
    collision: float = 4.0
    smooth: float = 0.05
    progress: float = 1.0


def paper_reward(
    position: np.ndarray,
    previous_position: np.ndarray,
    goal: np.ndarray,
    obstacles: list[np.ndarray],
    action_delta: np.ndarray,
    d_min: float,
    weights: RewardWeights = RewardWeights(),
) -> float:
    """Compute R_goal + R_collision + R_smooth + R_progress from the paper."""
    dist = float(np.linalg.norm(position - goal))
    prev_dist = float(np.linalg.norm(previous_position - goal))
    r_goal = -dist
    r_collision = 0.0
    for obs in obstacles:
        r_collision -= max(0.0, d_min - float(np.linalg.norm(position - obs)))
    r_smooth = -float(np.linalg.norm(action_delta) ** 2)
    r_progress = prev_dist - dist
    return (
        weights.goal * r_goal
        + weights.collision * r_collision
        + weights.smooth * r_smooth
        + weights.progress * r_progress
    )


def shaped_progress_reward(
    previous_position: np.ndarray,
    position: np.ndarray,
    goal: np.ndarray,
    action: np.ndarray,
    hazards: list[np.ndarray],
    safe_distance: float,
) -> float:
    """Dense controlled-environment reward used by PPO/DDPG training."""
    prev_dist = np.linalg.norm(previous_position - goal)
    dist = np.linalg.norm(position - goal)
    progress = prev_dist - dist
    hazard_penalty = sum(max(0.0, safe_distance - np.linalg.norm(position - h)) for h in hazards)
    smooth = np.linalg.norm(action) ** 2
    return float(0.08 * progress - 0.03 * dist - 0.5 * hazard_penalty - 0.02 * smooth)

