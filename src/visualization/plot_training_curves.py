"""Training-curve plots for GAIL discriminator metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_gail_metrics(csv_path: str | Path, output_path: str | Path) -> None:
    """Plot ACC, PPV, and TPR curves from GAIL training logs."""
    df = pd.read_csv(csv_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    for col in ["ACC", "PPV", "TPR"]:
        if col in df:
            plt.plot(df["iteration"], df[col], label=col, linewidth=2)
    plt.xlabel("Episodes")
    plt.ylabel("Value")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()

