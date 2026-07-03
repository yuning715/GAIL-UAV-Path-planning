"""Runtime logging shared by command-line scripts."""

from __future__ import annotations

import csv
import platform
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def runtime_stage(stage: str, table_path: str | Path = "outputs/tables/runtime_complexity.csv") -> Iterator[None]:
    """Measure a stage and append runtime metadata to a CSV file."""
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        path = Path(table_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        row = {
            "stage": stage,
            "runtime_seconds": elapsed,
            "hardware": platform.platform(),
            "python": platform.python_version(),
        }
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

