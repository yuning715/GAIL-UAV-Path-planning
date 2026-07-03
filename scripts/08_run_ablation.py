from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import REPO_ROOT, add_common_args, record_command, section_config, seed_from
from src.algorithms.apf import RiskAwareController
from src.envs.interception_env import InterceptionEnv
from src.metrics.planning_metrics import summarize_seed_values
from src.utils.runtime import runtime_stage
from src.utils.seed import set_seed
from src.utils.trace import write_trace
from src.visualization.plot_comparison import plot_ablation


def _run_variant(variant: str, env_cfg: dict, ablation_cfg: dict, seed: int):
    batches = int(ablation_cfg.get("batches", 5))
    uavs = int(ablation_cfg.get("uavs_per_batch", 3))
    delay = float(ablation_cfg.get("next_batch_delay_seconds", 30))
    controller = RiskAwareController(variant)
    intercepted = 0
    total_targets = batches * uavs
    elapsed = 0.0
    lengths = []
    payload = {}
    for batch in range(batches):
        env = InterceptionEnv(env_cfg, seed=seed + batch)
        state = env.reset(seed=seed + batch, group_size=uavs)
        for _ in range(env.max_group_steps):
            action = controller.act(env, state)
            state, _reward, done, _info = env.step(action)
            if done:
                break
        rec = env.record()
        intercepted += rec.contact_intercepted
        elapsed += rec.episode_time / 60.0
        if batch < batches - 1:
            elapsed += delay / 60.0
        lengths.append(rec.path_length)
        payload[f"positions_{batch}"] = np.array(env.path, dtype=np.float32)
        payload[f"actions_{batch}"] = np.array(env.actions, dtype=np.float32)
        payload[f"targets_{batch}"] = np.array(env.targets, dtype=np.float32)
    return {
        "success_rate": intercepted / max(total_targets, 1),
        "total_time": elapsed,
        "average_interception_path_length": float(np.mean(lengths)),
        "intercepted": intercepted,
        "total_targets": total_targets,
    }, payload


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    record_command()
    cfg = section_config(args.config, "ablation", "configs/ablation.yaml")
    int_cfg = section_config(args.config, "interception", "configs/interception.yaml")
    seed = seed_from(args, cfg)
    set_seed(seed)
    with runtime_stage("run_ablation", REPO_ROOT / "outputs/tables/runtime_complexity.csv"):
        env_cfg = {
            **int_cfg,
            "group_interval_minutes": float(cfg.get("max_batch_minutes", 15)),
            "max_group_size": int(cfg.get("uavs_per_batch", 3)),
            "timeout_counts_as_success": False,
        }
        raw_dir = REPO_ROOT / cfg.get("raw_rollout_dir", "outputs/raw_rollouts")
        tables_dir = REPO_ROOT / cfg.get("tables_dir", "outputs/tables")
        figures_dir = REPO_ROOT / cfg.get("figures_dir", "outputs/figures")
        raw_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        sources = []
        seeds = list(cfg.get("eval_seeds", [seed]))
        for variant in cfg.get("variants", []):
            per_seed = []
            payload_all = {}
            for sd in seeds:
                metrics, payload = _run_variant(variant, env_cfg, cfg, int(sd))
                per_seed.append(metrics)
                for key, value in payload.items():
                    payload_all[f"seed{sd}_{key}"] = value
            source = raw_dir / f"ablation_{variant}.npz"
            np.savez_compressed(source, **payload_all)
            sources.append(source)
            success_mean, success_std = summarize_seed_values([m["success_rate"] for m in per_seed])
            time_mean, _ = summarize_seed_values([m["total_time"] for m in per_seed])
            length_mean, _ = summarize_seed_values([m["average_interception_path_length"] for m in per_seed])
            rows.append(
                {
                    "variant": variant,
                    "success_rate": success_mean,
                    "success_rate_std": success_std,
                    "total_time": time_mean,
                    "average_interception_path_length": length_mean,
                    "seeds": str(seeds),
                    "source_rollout": str(source),
                }
            )
        table_path = tables_dir / "table5_ablation_metrics.csv"
        pd.DataFrame(rows).to_csv(table_path, index=False)
        write_trace(table_path, sources)
        plot_ablation(table_path, figures_dir / "ablation_comparison.png")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
