from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from _common import REPO_ROOT, add_common_args, record_command, section_config, seed_from
from src.algorithms.aco import ACOController
from src.algorithms.apf import APFController, RiskAwareController
from src.envs.breakthrough_env import BreakthroughEnv
from src.metrics.planning_metrics import summarize_seed_values
from src.utils.runtime import runtime_stage
from src.utils.seed import set_seed
from src.utils.trace import write_trace
from src.visualization.plot_comparison import plot_breakthrough_results
from src.visualization.plot_paths import plot_path_comparison


def _controller(method: str, cfg: dict, seed: int):
    method_u = method.upper()
    if method_u == "APF":
        return APFController()
    if method_u == "ACO":
        return ACOController(seed=seed)
    return RiskAwareController(method_u)


def _designated_targets(cfg: dict, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    count = int(cfg.get("designated_targets", {}).get("count", 64))
    inside = float(cfg.get("designated_targets", {}).get("inside_radar_fraction", 0.60))
    center = np.array(cfg.get("radar_center", [180.0, 0.0]), dtype=np.float32)
    radius = float(cfg.get("radar_radius", 620.0))
    targets = []
    for i in range(count):
        if rng.random() < inside:
            r = radius * np.sqrt(rng.uniform(0.05, 0.95))
            a = rng.uniform(-np.pi, np.pi)
            targets.append(center + r * np.array([np.cos(a), np.sin(a)], dtype=np.float32))
        else:
            targets.append(np.array([rng.uniform(-950, 900), rng.uniform(-600, 600)], dtype=np.float32))
    return targets


def _run_episode(method: str, cfg: dict, seed: int, target: np.ndarray | None = None, save_path: Path | None = None):
    env = BreakthroughEnv(cfg, seed=seed)
    state = env.reset(seed=seed, target=target)
    controller = _controller(method, cfg, seed)
    loiter_hist = []
    done = False
    info = {"failure_reason": "timeout", "success": False}
    for _ in range(int(cfg.get("single_episode_max_steps", 300))):
        action = controller.act(env, state)
        state, _reward, done, info = env.step(action)
        loiter_hist.append(info["loitering_positions"])
        if done:
            break
    rec = env.record(success_override=info.get("success"), reason=info.get("failure_reason"))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            positions=np.array(env.path, dtype=np.float32),
            actions=np.array(env.actions, dtype=np.float32),
            loitering_positions=np.array(loiter_hist, dtype=np.float32),
            success=rec.success,
            failure_reason=rec.failure_reason,
        )
    return rec


def _append_seed_stats(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        fieldnames = ["experiment", "method", "metric", "mean", "std", "seeds"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--methods", nargs="+", default=["APF", "ACO", "PPO", "DDPG", "GAIL"])
    args = parser.parse_args()
    record_command()
    cfg = section_config(args.config, "breakthrough", "configs/breakthrough.yaml")
    seed = seed_from(args, cfg)
    set_seed(seed)
    with runtime_stage("eval_breakthrough", REPO_ROOT / "outputs/tables/runtime_complexity.csv"):
        raw_dir = REPO_ROOT / cfg.get("raw_rollout_dir", "outputs/raw_rollouts")
        tables_dir = REPO_ROOT / cfg.get("tables_dir", "outputs/tables")
        figures_dir = REPO_ROOT / cfg.get("figures_dir", "outputs/figures")
        tables_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        table_rows = []
        rollout_paths = {}
        for method in args.methods:
            path = raw_dir / f"breakthrough_{method.lower()}_single.npz"
            rollout_paths[method] = path
            rec = _run_episode(method, cfg, seed, target=np.array(cfg.get("target_device", [520.0, 0.0])), save_path=path)
            table_rows.append(
                {
                    "method": method,
                    "reach_target_time": rec.time_to_target,
                    "cumulative_time": rec.cumulative_time,
                    "reach_target_path_length": rec.path_length_to_target,
                    "total_path_length": rec.total_path_length,
                    "minimum_relative_distance": rec.minimum_relative_distance,
                    "average_relative_distance": rec.average_relative_distance,
                    "success": rec.success,
                    "failure_reason": rec.failure_reason,
                    "source_rollout": str(path),
                }
            )
        table_path = tables_dir / "table3_breakthrough_path_metrics.csv"
        with table_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
            writer.writeheader()
            writer.writerows(table_rows)
        write_trace(table_path, list(rollout_paths.values()))

        curve_rows = []
        seed_rows = []
        seeds = list(cfg.get("seeds", [seed]))
        targets = _designated_targets(cfg, seed)
        max_minutes = float(cfg.get("simulation_minutes", 300))
        for method in args.methods:
            per_seed_completion = []
            for sd in seeds:
                elapsed = 0.0
                count = 0
                curve_rows.append({"method": method, "seed": sd, "time_minutes": 0.0, "completion_count": 0})
                for idx, target in enumerate(targets):
                    rec = _run_episode(method, cfg, int(sd) + idx, target=target)
                    elapsed += max(rec.cumulative_time / 60.0, 0.1)
                    if rec.success:
                        count += 1
                    curve_rows.append(
                        {"method": method, "seed": sd, "time_minutes": min(elapsed, max_minutes), "completion_count": count}
                    )
                    if elapsed >= max_minutes:
                        break
                per_seed_completion.append(float(count))
            mean, std = summarize_seed_values(per_seed_completion)
            seed_rows.append(
                {
                    "experiment": "breakthrough",
                    "method": method,
                    "metric": "completion_count",
                    "mean": mean,
                    "std": std,
                    "seeds": str(seeds),
                }
            )
        curve_path = raw_dir / "breakthrough_completion_curve.csv"
        pd.DataFrame(curve_rows).to_csv(curve_path, index=False)
        _append_seed_stats(tables_dir / "seed_statistics.csv", seed_rows)
        plot_breakthrough_results(curve_path, figures_dir / "fig12_breakthrough_results.png")
        plot_path_comparison(rollout_paths, figures_dir / "fig13_path_comparison.png")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
