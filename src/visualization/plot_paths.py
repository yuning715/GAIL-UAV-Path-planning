"""Path plotting from raw rollout files."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_path_comparison(rollout_paths: dict[str, str | Path], output_path: str | Path) -> None:
    """Plot UAV paths for all compared methods."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5.5))
    for method, path in rollout_paths.items():
        data = np.load(Path(path))
        pts = data["positions"]
        plt.plot(pts[:, 0], pts[:, 1], label=method, linewidth=1.8)
        if "loitering_positions" in data:
            loiter = data["loitering_positions"]
            if loiter.ndim == 3 and loiter.shape[0] > 0:
                plt.scatter(loiter[0, :, 0], loiter[0, :, 1], marker="x", s=40, color="black", alpha=0.4)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()

