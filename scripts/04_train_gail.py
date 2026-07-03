from __future__ import annotations

import argparse
from pathlib import Path

from _common import REPO_ROOT, add_common_args, record_command, section_config, seed_from
from src.algorithms.train_gail_ppo import train_gail_ppo
from src.data.synthetic_expert_generator import generate_expert_dataset
from src.utils.runtime import runtime_stage


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    record_command()
    cfg = section_config(args.config, "gail", "configs/gail_ppo.yaml")
    env_cfg = section_config(args.config, "breakthrough", "configs/breakthrough.yaml")
    seed = seed_from(args, cfg)
    expert_path = REPO_ROOT / cfg.get("expert_dir", "data/processed") / "expert_state_actions.npz"
    if not expert_path.exists():
        generate_expert_dataset(REPO_ROOT / "data/expert", 10, 1200, seed, processed_dir=REPO_ROOT / "data/processed")
    if args.resume:
        cfg.setdefault("training", {})["resume"] = args.resume
    with runtime_stage("train_gail", REPO_ROOT / "outputs/tables/runtime_complexity.csv"):
        checkpoint = train_gail_ppo(cfg, env_cfg, seed=seed)
    print(f"Saved GAIL-PPO checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
