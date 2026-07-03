from __future__ import annotations

import argparse
import importlib.metadata
import platform
from pathlib import Path

import pandas as pd

from _common import REPO_ROOT, add_common_args, record_command
from src.utils.runtime import runtime_stage
from src.visualization.plot_comparison import plot_ablation, plot_breakthrough_results, plot_interception_scores, plot_runtime
from src.visualization.plot_inpainting import plot_gan_convergence
from src.visualization.plot_training_curves import plot_gail_metrics


PAPER_TABLE2 = {
    ("The first track", "velocity"): (3.53, 0.128, 13.91),
    ("The first track", "turning_angle"): (11.99, 0.696, 47.95),
    ("The first track", "pitch_angle"): (1.54, 0.186, 6.16),
    ("The second track", "velocity"): (1.47, 0.004, 5.86),
    ("The second track", "turning_angle"): (0.75, 1.371, 3.90),
    ("The second track", "pitch_angle"): (0.94, 0.044, 4.42),
}

PAPER_TABLE3 = {
    "APF": (24.52, 31.51, 490.4, 579.3, 0.0, 192.8),
    "ACO": (26.20, 46.01, 523.9, 920.2, 0.0, 148.3),
    "PPO": (288.9, 300.0, 2999.0, 3199.0, 55.99, 312.4),
    "DDPG": (270.3, 295.8, 3135.0, 3635.0, 69.68, 221.3),
    "GAIL": (105.5, 132.5, 1500.0, 2030.0, 62.52, 288.6),
}

PAPER_TABLE4 = {
    "APF": (-19, 40, 40.40),
    "ACO": (7, 53, 53.54),
    "PPO": (49, 74, 74.75),
    "DDPG": (59, 79, 79.80),
    "GAIL": (75, 87, 87.88),
}

PAPER_TABLE5 = {
    "complete_model": (86.7, 59.4, 237.4),
    "without_si": (46.7, 75.0, 561.5),
    "original_gan": (80.0, 63.1, 287.4),
    "without_cnn_lstm": (73.3, 68.9, 321.7),
    "without_ppo": (66.7, 74.1, 391.7),
}


def _dep_versions() -> pd.DataFrame:
    rows = []
    for name in ["numpy", "pandas", "matplotlib", "torch", "PyYAML", "pytest"]:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        rows.append({"package": name, "version": version})
    return pd.DataFrame(rows)


def _comparison_tables(tables_dir: Path) -> dict[str, pd.DataFrame]:
    comparisons: dict[str, pd.DataFrame] = {}
    table2 = tables_dir / "table2_inpainting_metrics.csv"
    if table2.exists():
        df = pd.read_csv(table2)
        rows = []
        for _, row in df.iterrows():
            target = PAPER_TABLE2.get((row["track"], row["data_type"]))
            if target:
                for metric, paper, reproduced in zip(["MER", "TMSE", "MSER"], target, [row["MER"], row["TMSE"], row["MSER"]]):
                    rows.append(
                        {
                            "experiment": "inpainting",
                            "item": f"{row['track']} {row['data_type']} {metric}",
                            "reproduced": reproduced,
                            "paper_target": paper,
                            "absolute_difference": abs(reproduced - paper),
                            "relative_difference": abs(reproduced - paper) / max(abs(paper), 1e-8),
                        }
                    )
        comparisons["table2"] = pd.DataFrame(rows)
    table3 = tables_dir / "table3_breakthrough_path_metrics.csv"
    if table3.exists():
        df = pd.read_csv(table3)
        columns = [
            "reach_target_time",
            "cumulative_time",
            "reach_target_path_length",
            "total_path_length",
            "minimum_relative_distance",
            "average_relative_distance",
        ]
        rows = []
        for _, row in df.iterrows():
            target = PAPER_TABLE3.get(row["method"])
            if target:
                for metric, paper in zip(columns, target):
                    reproduced = row[metric]
                    rows.append(
                        {
                            "experiment": "breakthrough",
                            "item": f"{row['method']} {metric}",
                            "reproduced": reproduced,
                            "paper_target": paper,
                            "absolute_difference": abs(reproduced - paper),
                            "relative_difference": abs(reproduced - paper) / max(abs(paper), 1e-8),
                        }
                    )
        comparisons["table3"] = pd.DataFrame(rows)
    table4 = tables_dir / "table4_interception_metrics.csv"
    if table4.exists():
        df = pd.read_csv(table4)
        rows = []
        for _, row in df.iterrows():
            target = PAPER_TABLE4.get(row["method"])
            if target:
                for metric, paper in zip(["score", "number_of_targets_intercepted", "completion_rate"], target):
                    reproduced = row[metric]
                    rows.append(
                        {
                            "experiment": "interception",
                            "item": f"{row['method']} {metric}",
                            "reproduced": reproduced,
                            "paper_target": paper,
                            "absolute_difference": abs(reproduced - paper),
                            "relative_difference": abs(reproduced - paper) / max(abs(paper), 1e-8),
                        }
                    )
        comparisons["table4"] = pd.DataFrame(rows)
    table5 = tables_dir / "table5_ablation_metrics.csv"
    if table5.exists():
        df = pd.read_csv(table5)
        rows = []
        for _, row in df.iterrows():
            target = PAPER_TABLE5.get(row["variant"])
            if target:
                reproduced_values = [row["success_rate"] * 100.0, row["total_time"], row["average_interception_path_length"]]
                for metric, paper, reproduced in zip(["success_rate", "total_time", "average_length"], target, reproduced_values):
                    rows.append(
                        {
                            "experiment": "ablation",
                            "item": f"{row['variant']} {metric}",
                            "reproduced": reproduced,
                            "paper_target": paper,
                            "absolute_difference": abs(reproduced - paper),
                            "relative_difference": abs(reproduced - paper) / max(abs(paper), 1e-8),
                        }
                    )
        comparisons["table5"] = pd.DataFrame(rows)
    return comparisons


def _write_report(tables_dir: Path, figures_dir: Path, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparisons = _comparison_tables(tables_dir)
    generated_tables = sorted(p.name for p in tables_dir.glob("*.csv"))
    generated_figures = sorted(p.name for p in figures_dir.glob("*.png"))
    runtime = pd.read_csv(tables_dir / "runtime_complexity.csv") if (tables_dir / "runtime_complexity.csv").exists() else pd.DataFrame()
    seeds = pd.read_csv(tables_dir / "seed_statistics.csv") if (tables_dir / "seed_statistics.csv").exists() else pd.DataFrame()
    command_history = (REPO_ROOT / "outputs/logs/command_history.txt")
    commands = command_history.read_text(encoding="utf-8") if command_history.exists() else "Command history is generated by the reproduction scripts."
    md_parts = [
        "# Reproduction Report",
        "## Method Summary",
        "The repository implements sequential trajectory inpainting with a GAN-LSTM model, then trains a GAIL-PPO navigation policy for dynamic UAV path planning. APF, ACO, PPO, DDPG, and ablation variants are evaluated in shared environments.",
        "## Experiment Settings",
        f"Random seed control is enabled for Python, NumPy, PyTorch, CUDA, target generation, expert generation, missing masks, and environment rollout generation. Hardware: {platform.platform()}.",
        "## Reproduced Values",
    ]
    for name, df in comparisons.items():
        md_parts.append(f"### {name}")
        md_parts.append(df.to_markdown(index=False) if not df.empty else "No rows generated.")
    md_parts.extend(
        [
            "## Random Seed Results",
            seeds.to_markdown(index=False) if not seeds.empty else "Seed statistics have not been generated yet.",
            "## Hardware And Runtime",
            runtime.to_markdown(index=False) if not runtime.empty else "Runtime data have not been generated yet.",
            "## Dependency Versions",
            _dep_versions().to_markdown(index=False),
            "## Training Curves",
            "\n".join(f"- outputs/figures/{name}" for name in generated_figures if "training" in name or "gan" in name or "gail" in name),
            "## Evaluation Curves",
            "\n".join(f"- outputs/figures/{name}" for name in generated_figures),
            "## Generated Figures",
            "\n".join(f"- outputs/figures/{name}" for name in generated_figures),
            "## Generated Tables",
            "\n".join(f"- outputs/tables/{name}" for name in generated_tables),
            "## Command History",
            f"```text\n{commands}\n```",
        ]
    )
    md = "\n\n".join(md_parts)
    (reports_dir / "reproduction_report.md").write_text(md, encoding="utf-8")
    html_parts = ["<html><body><h1>Reproduction Report</h1>"]
    html_parts.append("<h2>Method Summary</h2><p>The code reproduces the GAN-LSTM inpainting and GAIL-PPO UAV navigation workflow with classical and RL baselines.</p>")
    html_parts.append(f"<h2>Experiment Settings</h2><p>Hardware: {platform.platform()}</p>")
    for name, df in comparisons.items():
        html_parts.append(f"<h2>{name}</h2>")
        html_parts.append(df.to_html(index=False))
    html_parts.append("<h2>Runtime</h2>")
    html_parts.append(runtime.to_html(index=False) if not runtime.empty else "<p>No runtime rows.</p>")
    html_parts.append("<h2>Dependency Versions</h2>")
    html_parts.append(_dep_versions().to_html(index=False))
    html_parts.append("<h2>Generated Figures</h2>")
    for name in generated_figures:
        html_parts.append(f"<p>{name}</p><img src='../figures/{name}' style='max-width:800px;width:95%;'>")
    html_parts.append("</body></html>")
    (reports_dir / "reproduction_report.html").write_text("\n".join(html_parts), encoding="utf-8")


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    _args = parser.parse_args()
    record_command()
    with runtime_stage("make_all_figures", REPO_ROOT / "outputs/tables/runtime_complexity.csv"):
        log_dir = REPO_ROOT / "outputs/logs"
        raw_dir = REPO_ROOT / "outputs/raw_rollouts"
        figures_dir = REPO_ROOT / "outputs/figures"
        tables_dir = REPO_ROOT / "outputs/tables"
        reports_dir = REPO_ROOT / "outputs/reports"
        if (log_dir / "inpainting_training.csv").exists():
            plot_gan_convergence(log_dir / "inpainting_training.csv", figures_dir / "fig1_gan_convergence.png")
        if (log_dir / "gail_training.csv").exists():
            plot_gail_metrics(log_dir / "gail_training.csv", figures_dir / "fig6_gail_training_acc_ppv_tpr.png")
        if (raw_dir / "breakthrough_completion_curve.csv").exists():
            plot_breakthrough_results(raw_dir / "breakthrough_completion_curve.csv", figures_dir / "fig12_breakthrough_results.png")
        if (raw_dir / "interception_score_curve.csv").exists():
            plot_interception_scores(raw_dir / "interception_score_curve.csv", figures_dir / "fig15_interception_scores.png")
        if (tables_dir / "table5_ablation_metrics.csv").exists():
            plot_ablation(tables_dir / "table5_ablation_metrics.csv", figures_dir / "ablation_comparison.png")
        if (tables_dir / "runtime_complexity.csv").exists():
            plot_runtime(tables_dir / "runtime_complexity.csv", figures_dir / "runtime_comparison.png")
        _write_report(tables_dir, figures_dir, reports_dir)
    print(f"Wrote reports in {REPO_ROOT / 'outputs/reports'}")


if __name__ == "__main__":
    main()
