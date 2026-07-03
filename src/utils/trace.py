"""Trace output tables back to raw logs, rollouts, or prediction files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_trace(table_path: str | Path, sources: list[str | Path], notes: str = "") -> Path:
    """Write a sidecar trace file for a generated table."""
    table = Path(table_path)
    trace_path = table.with_suffix(table.suffix + ".trace.json")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "table": str(table),
        "sources": [str(Path(s)) for s in sources],
        "notes": notes,
    }
    trace_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return trace_path

