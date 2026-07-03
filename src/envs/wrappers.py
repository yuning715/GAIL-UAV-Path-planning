"""Small wrappers used by training loops."""

from __future__ import annotations

from typing import Any

from src.envs.breakthrough_env import BreakthroughEnv
from src.envs.interception_env import InterceptionEnv


def make_env(name: str, config: dict[str, Any], seed: int = 0):
    """Construct a named environment."""
    normalized = name.lower()
    if normalized in {"breakthrough", "defensive_line"}:
        return BreakthroughEnv(config, seed=seed)
    if normalized in {"interception", "intercept"}:
        return InterceptionEnv(config, seed=seed)
    raise ValueError(f"Unknown environment: {name}")

