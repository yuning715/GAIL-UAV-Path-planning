from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from _common import REPO_ROOT, add_common_args, record_command, section_config, seed_from
from src.algorithms.aco import ACOController
from src.algorithms.apf import APFController, RiskAwareController
from src.envs.interception_env import InterceptionEnv
from src.metrics.planning_metrics import summarize_seed_values
from src.utils.runtime import runtime_stage
from src.utils.seed import set_seed
from src.utils.trace import write_trace
from src.visualization.plot_comparison import plot_interception_scores


def _controller(method: str, seed: int):
    method_u = method.upper()
    if method_u == "APF":
        return APFController(max_accel=0.2, attractive_gain=0.3)
    if method_u == "ACO":
        return ACOController(seed=seed, max_accel=0.35, ants=8, iterations=2)
    return RiskAwareController(method_u)


def _run_group(method: str, cfg: dict, seed: int, group_size: int):
    env = InterceptionEnv(cfg, seed=seed)
    state = env.reset(seed=seed, group_size=group_size)
    controller = _controller(method, seed)
    done = False
    for _ in range(env.max_group_steps):
        action = controller.act(env, state)
        state, _reward, done, _info = env.step(action)
        if done:
            break
    return env.record(), env


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
    cfg = section_config(args.config, "interception", "configs/interception.yaml")
    seed = seed_from(args, cfg)
    set_seed(seed)
    with runtime_stage("eval_interception", REPO_ROOT / "outputs/tables/runtime_complexity.csv"):
        raw_dir = REPO_ROOT / cfg.get("raw_rollout_dir", "outputs/raw_rollouts")
        tables_dir = REPO_ROOT / cfg.get("tables_dir", "outputs/tables")
        figures_dir = REPO_ROOT / cfg.get("figures_dir", "outputs/figures")
        raw_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        groups = int(cfg.get("groups", 33))
        group_sizes = rng.integers(1, int(cfg.get("max_group_size", 5)) + 1, size=groups)
        curve_rows = []
        table_rows = []
        seed_rows = []
        seeds = list(cfg.get("seeds", [seed]))
        source_paths = []
        for method in args.methods:
            score = 0
            intercepted_total = 0
            escaped_total = 0
            targets_total = 0
            payload = {}
            episode_rows = []
            for group_idx, group_size in enumerate(group_sizes):
                rec, env = _run_group(method, cfg, seed + group_idx, int(group_size))
                score += rec.score
                intercepted_total += rec.intercepted
                escaped_total += rec.escaped
                targets_total += rec.total_targets
                time_minutes = (group_idx + 1) * float(cfg.get("group_interval_minutes", 15))
                curve_rows.append({"method": method, "seed": seed, "time_minutes": time_minutes, "score": score})
                payload[f"positions_{group_idx}"] = np.array(env.path, dtype=np.float32)
                payload[f"actions_{group_idx}"] = np.array(env.actions, dtype=np.float32)
                payload[f"targets_{group_idx}"] = np.array(env.targets, dtype=np.float32)
                episode_rows.append(
                    {
                        "group": group_idx,
                        "group_size": int(group_size),
                        "score": rec.score,
                        "intercepted": rec.intercepted,
                        "escaped": rec.escaped,
                        "episode_time": rec.episode_time,
                        "path_length": rec.path_length,
                    }
                )
            source = raw_dir / f"interception_{method.lower()}_rollouts.npz"
            np.savez_compressed(source, **payload)
            pd.DataFrame(episode_rows).to_csv(raw_dir / f"interception_{method.lower()}_episodes.csv", index=False)
            source_paths.append(source)
            completion = intercepted_total / max(targets_total, 1)
            table_rows.append(
                {
                    "method": method,
                    "score": score,
                    "number_of_targets_intercepted": intercepted_total,
                    "completion_rate": completion * 100.0,
                    "total_targets": targets_total,
                    "failed_targets": escaped_total,
                    "source_rollout": str(source),
                }
            )
            per_seed_rates = []
            for sd in seeds:
                rng_sd = np.random.default_rng(int(sd))
                sizes_sd = rng_sd.integers(1, int(cfg.get("max_group_size", 5)) + 1, size=groups)
                intercepted = 0
                total = 0
                for group_idx, size in enumerate(sizes_sd):
                    rec, _env = _run_group(method, cfg, int(sd) + group_idx, int(size))
                    intercepted += rec.intercepted
                    total += rec.total_targets
                per_seed_rates.append(intercepted / max(total, 1))
            mean, std = summarize_seed_values(per_seed_rates)
            seed_rows.append(
                {
                    "experiment": "interception",
                    "method": method,
                    "metric": "completion_rate",
                    "mean": mean,
                    "std": std,
                    "seeds": str(seeds),
                }
            )
        table_path = tables_dir / "table4_interception_metrics.csv"
        pd.DataFrame(table_rows).to_csv(table_path, index=False)
        write_trace(table_path, source_paths)
        curve_path = raw_dir / "interception_score_curve.csv"
        pd.DataFrame(curve_rows).to_csv(curve_path, index=False)
        _append_seed_stats(tables_dir / "seed_statistics.csv", seed_rows)
        plot_interception_scores(curve_path, figures_dir / "fig15_interception_scores.png")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
