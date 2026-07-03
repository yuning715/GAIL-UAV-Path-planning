from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from _common import REPO_ROOT, add_common_args, record_command, section_config, seed_from
from src.algorithms.train_inpainting import build_inpainting_models
from src.data.missing_mask import compose_masks, interval_missing_mask, random_missing_mask, retained_mask
from src.data.preprocessing import FeatureScaler
from src.data.synthetic_expert_generator import load_feature_trajectories
from src.metrics.inpainting_metrics import metric_table
from src.utils.checkpoint import load_checkpoint
from src.utils.device import get_device
from src.utils.runtime import runtime_stage
from src.utils.trace import write_trace
from src.visualization.plot_inpainting import plot_gan_convergence, plot_real_data, plot_repaired_vs_expert


def _repair(generator, scaler: FeatureScaler, features: np.ndarray, mask: np.ndarray, device: torch.device) -> np.ndarray:
    normalized = scaler.transform(features)
    incomplete = normalized * mask
    with torch.no_grad():
        repaired_norm = generator(
            torch.tensor(incomplete, dtype=torch.float32, device=device).unsqueeze(0),
            torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0),
            preserve_observed=True,
        ).squeeze(0)
    return scaler.inverse_transform(repaired_norm.cpu().numpy())


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    record_command()
    cfg = section_config(args.config, "inpainting", "configs/inpainting.yaml")
    seed = seed_from(args, cfg)
    with runtime_stage("eval_inpainting", REPO_ROOT / "outputs/tables/runtime_complexity.csv"):
        device = get_device(cfg.get("device", "auto") != "cpu")
        checkpoint_path = REPO_ROOT / cfg.get("checkpoint_dir", "outputs/checkpoints/inpainting") / "gan_lstm_final.pt"
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        generator, _discriminator = build_inpainting_models(cfg, device)
        generator.load_state_dict(checkpoint["generator"])
        generator.eval()
        scaler = FeatureScaler.load(REPO_ROOT / cfg.get("processed_dir", "data/processed") / "inpainting_scaler.npz")
        test_indices = list(cfg.get("test_indices", [8, 9]))
        tracks = load_feature_trajectories(REPO_ROOT / cfg.get("expert_dir", "data/expert"), test_indices)
        feature_names = list(cfg.get("feature_names", ["velocity", "turning_angle", "pitch_angle"]))
        missing_cfg = cfg.get("missing", {})
        intervals = [
            tuple(missing_cfg.get("first_plot_interval", [6000, 6400])),
            tuple(missing_cfg.get("second_plot_interval", [7000, 9000])),
        ]
        masks = [
            compose_masks(
                random_missing_mask(tracks[0].shape, float(missing_cfg.get("random_missing_rate", 0.02)), seed),
                interval_missing_mask(tracks[0].shape, intervals[0]),
            ),
            compose_masks(
                retained_mask(tracks[1].shape, float(missing_cfg.get("retained_rate", 0.90)), seed + 17),
                interval_missing_mask(tracks[1].shape, intervals[1]),
            ),
        ]
        raw_dir = REPO_ROOT / "outputs/raw_rollouts"
        raw_dir.mkdir(parents=True, exist_ok=True)
        repaired = []
        rows = []
        for idx, (track, mask, interval, label) in enumerate(zip(tracks, masks, intervals, ["The first track", "The second track"])):
            pred = _repair(generator, scaler, track, mask, device)
            repaired.append(pred)
            pred_path = raw_dir / f"inpainting_{idx + 1}.npz"
            np.savez_compressed(pred_path, expert=track, repaired=pred, mask=mask, interval=np.array(interval))
            for row in metric_table(pred, track, mask, feature_names):
                rows.append(
                    {
                        "track": label,
                        "data_type": row["data_type"],
                        "MER": row["MER"] * 100.0,
                        "TMSE": row["TMSE"],
                        "MSER": row["MSER"] * 100.0,
                        "source_prediction": str(pred_path),
                    }
                )
        tables_dir = REPO_ROOT / cfg.get("tables_dir", "outputs/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        table_path = tables_dir / "table2_inpainting_metrics.csv"
        with table_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        write_trace(table_path, [raw_dir / "inpainting_1.npz", raw_dir / "inpainting_2.npz"])
        figures_dir = REPO_ROOT / cfg.get("figures_dir", "outputs/figures")
        train_csv = REPO_ROOT / cfg.get("log_dir", "outputs/logs") / "inpainting_training.csv"
        if train_csv.exists():
            plot_gan_convergence(train_csv, figures_dir / "fig1_gan_convergence.png")
        plot_real_data(tracks[0], feature_names, figures_dir / "fig2_first_track_real_data.png")
        plot_repaired_vs_expert(repaired[0], tracks[0], intervals[0], feature_names, figures_dir / "fig3_first_track_repaired_vs_expert.png")
        plot_real_data(tracks[1], feature_names, figures_dir / "fig4_second_track_real_data.png")
        plot_repaired_vs_expert(repaired[1], tracks[1], intervals[1], feature_names, figures_dir / "fig5_second_track_repaired_vs_expert.png")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
