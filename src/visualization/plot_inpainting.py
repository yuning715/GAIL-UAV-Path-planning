"""Plots for sequential trajectory inpainting experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_gan_convergence(csv_path: str | Path, output_path: str | Path) -> None:
    """Plot generator/discriminator loss curves from training CSV."""
    df = pd.read_csv(csv_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(df["epoch"], df["generator_loss"], label="The loss of G", linewidth=2)
    plt.plot(df["epoch"], df["discriminator_loss"], label="The loss of D", linewidth=2)
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_real_data(features: np.ndarray, feature_names: list[str], output_path: str | Path) -> None:
    """Plot full real velocity, turning angle, and pitch angle curves."""
    labels = ["Velocity(km/h)", "Turning angle(deg)", "Pitch angle(deg)"]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(feature_names), 1, figsize=(8, 7), sharex=True)
    x = np.arange(len(features))
    for idx, ax in enumerate(axes):
        ax.plot(x, features[:, idx], color="#1f77b4", linewidth=1.0)
        ax.set_ylabel(labels[idx] if idx < len(labels) else feature_names[idx])
    axes[-1].set_xlabel("Samples")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_repaired_vs_expert(
    repaired: np.ndarray,
    expert: np.ndarray,
    interval: tuple[int, int],
    feature_names: list[str],
    output_path: str | Path,
) -> None:
    """Plot repaired and expert curves over a sample interval."""
    start, end = interval
    start = max(0, min(len(expert), start))
    end = max(start + 1, min(len(expert), end))
    labels = ["Velocity(km/h)", "Turning angle(deg)", "Pitch angle(deg)"]
    x = np.arange(start, end)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(feature_names), 1, figsize=(8, 7), sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(x, expert[start:end, idx], color="#d62728", label="expert", linewidth=1.4)
        ax.plot(x, repaired[start:end, idx], color="#1f77b4", label="repaired", linewidth=1.2, linestyle="--")
        ax.set_ylabel(labels[idx] if idx < len(labels) else feature_names[idx])
        ax.legend(loc="best")
    axes[-1].set_xlabel("Sample")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

