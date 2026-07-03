"""YAML configuration loading and lightweight recursive overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``updates`` into ``base`` and return ``base``."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file from an absolute path or relative to the repo root."""
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(cfg_path)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def get_section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return a named section if present, otherwise return the full config."""
    if name in config and isinstance(config[name], Mapping):
        section = deepcopy(dict(config[name]))
        common = deepcopy(dict(config.get("common", {})))
        deep_update(common, section)
        common["_config_path"] = config.get("_config_path")
        common["_repo_root"] = config.get("_repo_root", str(REPO_ROOT))
        return common
    return deepcopy(dict(config))


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Resolve ``path`` relative to the repository root unless it is absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    base = Path(root) if root is not None else REPO_ROOT
    return base / p


def apply_cli_overrides(config: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Apply dotted-key command-line overrides to a config dictionary."""
    cfg = deepcopy(config)
    for dotted_key, value in overrides.items():
        target = cfg
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return cfg

