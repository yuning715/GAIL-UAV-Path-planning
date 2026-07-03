from __future__ import annotations

import argparse
from pathlib import Path

from _common import REPO_ROOT, add_common_args, record_command, section_config, seed_from
from src.data.synthetic_expert_generator import generate_expert_dataset
from src.utils.runtime import runtime_stage


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--num-trajectories", type=int, default=None)
    parser.add_argument("--points-per-trajectory", type=int, default=None)
    args = parser.parse_args()
    record_command()
    cfg = section_config(args.config, "expert_generation")
    seed = seed_from(args, cfg)
    num = int(args.num_trajectories or cfg.get("num_trajectories", 10))
    points = int(args.points_per_trajectory or cfg.get("points_per_trajectory", 10000))
    save_dir = REPO_ROOT / cfg.get("save_dir", "data/expert")
    processed_dir = REPO_ROOT / "data/processed"
    with runtime_stage("generate_expert_data", REPO_ROOT / "outputs/tables/runtime_complexity.csv"):
        paths = generate_expert_dataset(save_dir, num, points, seed, processed_dir=processed_dir)
    print(f"Generated {len(paths)} expert trajectories in {save_dir}")
    print(f"Generated state-action demonstrations in {processed_dir}")


if __name__ == "__main__":
    main()
