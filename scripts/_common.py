"""Shared script helpers."""

from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_section, load_config


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default="configs/quick.yaml", help="YAML config path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    return parser


def section_config(config_path: str, section: str, fallback: str | None = None) -> dict:
    cfg = load_config(config_path)
    if section in cfg:
        return get_section(cfg, section)
    include = cfg.get("include", {})
    if isinstance(include, dict) and section in include:
        return get_section(load_config(include[section]), section)
    if fallback:
        return get_section(load_config(fallback), section)
    return get_section(cfg, section)


def seed_from(args, cfg: dict) -> int:
    return int(args.seed if args.seed is not None else cfg.get("seed", 0))


def record_command(argv: list[str] | None = None) -> None:
    """Append the current script invocation to command history."""
    argv = argv or sys.argv
    log_path = REPO_ROOT / "outputs/logs/command_history.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} | python {' '.join(argv)}\n")
