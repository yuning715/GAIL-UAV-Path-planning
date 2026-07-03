"""Comparison figures for path planning, interception, ablation, and runtime."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_breakthrough_results(csv_path: str | Path, output_path: str | Path) -> None:
    """Plot completion count over simulated time."""
    df = pd.read_csv(csv_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    for method, group in df.groupby("method"):
        plt.plot(group["time_minutes"], group["completion_count"], label=method, linewidth=2)
    plt.xlabel("Times(min)")
    plt.ylabel("Completion count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_interception_scores(csv_path: str | Path, output_path: str | Path) -> None:
    """Plot score over time for interception experiments."""
    df = pd.read_csv(csv_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    for method, group in df.groupby("method"):
        plt.plot(group["time_minutes"], group["score"], label=method, linewidth=2)
    plt.xlabel("Times(min)")
    plt.ylabel("The scores of models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_ablation(csv_path: str | Path, output_path: str | Path) -> None:
    """Plot ablation success rates and average path lengths."""
    df = pd.read_csv(csv_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.bar(df["variant"], df["success_rate"] * 100.0, color="#4c78a8", alpha=0.8)
    ax1.set_ylabel("Success rate (%)")
    ax1.tick_params(axis="x", rotation=25)
    ax2 = ax1.twinx()
    ax2.plot(df["variant"], df["average_interception_path_length"], color="#f58518", marker="o", linewidth=2)
    ax2.set_ylabel("Average path length (m)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_runtime(csv_path: str | Path, output_path: str | Path) -> None:
    """Plot runtime by stage/method."""
    df = pd.read_csv(csv_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.bar(df["stage"], df["runtime_seconds"], color="#54a24b")
    plt.ylabel("Runtime (s)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()

