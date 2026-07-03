"""Planning metrics computed from environment rollout records."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def completion_rate(successes: Iterable[bool]) -> float:
    values = list(successes)
    if not values:
        return 0.0
    return float(np.mean(values))


def summarize_seed_values(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def score_interception(intercepted: int, escaped: int) -> int:
    """Score is +1 per successful interception and -1 per failed crossing."""
    return int(intercepted - escaped)

