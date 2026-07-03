"""Synthetic expert trajectory generation from UAV motion primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.envs.uav_dynamics import state6
from src.utils.math_utils import clip_norm, unit


def generate_feature_trajectory(points: int, seed: int, track_id: int) -> dict[str, np.ndarray]:
    """Generate a smooth expert trajectory with velocity, yaw, and pitch features."""
    rng = np.random.default_rng(seed + 7919 * track_id)
    t = np.linspace(0.0, 1.0, points, dtype=np.float32)
    base_speed = 40.0 + rng.normal(0.0, 1.2)
    velocity = (
        base_speed
        + 3.8 * np.sin(2 * np.pi * (1.5 + 0.1 * track_id) * t + rng.uniform(-1, 1))
        + 1.2 * np.sin(2 * np.pi * 7.0 * t + rng.uniform(-1, 1))
        + rng.normal(0.0, 0.25, size=points)
    )
    turning = (
        36.0 * np.sin(2 * np.pi * (0.75 + 0.04 * track_id) * t + rng.uniform(-2, 2))
        + 18.0 * np.sin(2 * np.pi * 3.5 * t + rng.uniform(-2, 2))
        + rng.normal(0.0, 2.0, size=points)
    )
    pitch = (
        9.0 * np.sin(2 * np.pi * (1.1 + 0.03 * track_id) * t + rng.uniform(-2, 2))
        + 4.0 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(-2, 2))
        + rng.normal(0.0, 0.4, size=points)
    )
    features = np.stack([velocity, turning, pitch], axis=1).astype(np.float32)
    speed_ms = velocity / 3.6
    heading = np.deg2rad(turning)
    climb = np.sin(np.deg2rad(pitch)) * speed_ms
    vx = np.cos(heading) * speed_ms
    vy = np.sin(heading) * speed_ms
    vz = climb
    pos = np.cumsum(np.stack([vx, vy, vz], axis=1), axis=0).astype(np.float32)
    states = np.concatenate(
        [pos, np.stack([vx, vy, vz], axis=1), turning[:, None], pitch[:, None]], axis=1
    ).astype(np.float32)
    state_action = states[:, [3, 4, 5, 6, 7]]
    actions = np.diff(state_action, axis=0, prepend=state_action[[0]])
    return {"features": features, "states": states, "actions": actions.astype(np.float32)}


def generate_state_action_demonstrations(
    num_trajectories: int,
    points: int,
    seed: int,
    state_dim: int = 6,
    action_dim: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate controlled 2D expert state-action pairs for imitation learning."""
    rng = np.random.default_rng(seed + 12345)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for traj in range(num_trajectories):
        position = np.array([-900.0 + rng.normal(0, 20), rng.normal(0, 120)], dtype=np.float32)
        velocity = np.zeros(2, dtype=np.float32)
        target = np.array([520.0 + rng.normal(0, 50), rng.normal(0, 160)], dtype=np.float32)
        phase_switch = int(points * 0.55)
        for i in range(points):
            goal = target if i < phase_switch else np.array([-900.0, rng.normal(0, 80)], dtype=np.float32)
            desired = unit(goal - position) * 15.0
            avoid_center = np.array([180.0, 0.0], dtype=np.float32)
            avoid = position - avoid_center
            avoid_dist = np.linalg.norm(avoid)
            repulse = np.zeros(2, dtype=np.float32)
            if avoid_dist < 420.0 and avoid_dist > 1e-8:
                repulse = unit(avoid) * (420.0 - avoid_dist) / 420.0 * 6.0
            accel = clip_norm((desired + repulse - velocity) * 0.35, 4.0)
            states.append(state6(position, velocity, goal))
            actions.append(accel.astype(np.float32))
            velocity = clip_norm(velocity + accel, 18.0)
            position = position + velocity
    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def generate_expert_dataset(
    output_dir: str | Path,
    num_trajectories: int,
    points_per_trajectory: int,
    seed: int,
    processed_dir: str | Path | None = None,
) -> list[Path]:
    """Generate expert trajectory files and optional GAIL state-action arrays."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx in range(num_trajectories):
        traj = generate_feature_trajectory(points_per_trajectory, seed, idx)
        path = output / f"trajectory_{idx:02d}.npz"
        np.savez_compressed(path, **traj)
        paths.append(path)
    if processed_dir is not None:
        processed = Path(processed_dir)
        processed.mkdir(parents=True, exist_ok=True)
        states, actions = generate_state_action_demonstrations(
            num_trajectories=num_trajectories,
            points=max(200, min(points_per_trajectory, 3000)),
            seed=seed,
        )
        np.savez_compressed(processed / "expert_state_actions.npz", states=states, actions=actions)
    return paths


def load_feature_trajectories(expert_dir: str | Path, indices: list[int] | None = None) -> list[np.ndarray]:
    """Load feature arrays from generated expert trajectories."""
    files = sorted(Path(expert_dir).glob("trajectory_*.npz"))
    if indices is not None:
        files = [files[i] for i in indices if i < len(files)]
    if not files:
        raise FileNotFoundError(f"No trajectory_*.npz files found in {expert_dir}")
    return [np.load(f)["features"].astype(np.float32) for f in files]
