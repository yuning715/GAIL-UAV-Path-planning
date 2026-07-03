from __future__ import annotations

import argparse

from _common import REPO_ROOT, add_common_args, record_command, section_config, seed_from
from src.algorithms.train_inpainting import train_inpainting
from src.utils.runtime import runtime_stage


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    record_command()
    cfg = section_config(args.config, "inpainting", "configs/inpainting.yaml")
    seed = seed_from(args, cfg)
    if args.resume:
        cfg.setdefault("training", {})["resume"] = args.resume
    with runtime_stage("train_inpainting", REPO_ROOT / "outputs/tables/runtime_complexity.csv"):
        checkpoint = train_inpainting(cfg, seed=seed)
    print(f"Saved inpainting checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
