# GAIL-UAV-Reproduction

## Paper Information

The official version of the paper **Expert-Level Autonomous Navigation for UAVs in Dynamic Environments: A GAIL Framework with Sequential Trajectory Inpainting**.

The code implements a two-stage reproduction workflow:

1. Sequential trajectory inpainting with a GAN-LSTM generator and a CNN-LSTM discriminator.
2. GAIL-PPO policy learning for dynamic UAV path planning, evaluated against APF, ACO, PPO, and DDPG baselines.

## Method Summary

Stage 1 repairs incomplete expert UAV trajectories. The generator receives an incomplete trajectory and an observed/missing mask, encodes temporal context with a two-layer bidirectional LSTM, applies temporal attention, and decodes a repaired trajectory. The discriminator combines temporal convolutions, LSTM processing, and attention pooling to classify real versus generated trajectories.

Stage 2 trains a continuous-action navigation policy. The GAIL discriminator learns from expert state-action pairs and policy rollouts, converts discriminator output to an imitation reward, and PPO updates the Gaussian actor with clipped policy gradients and GAE advantages.

## Repository Structure

```text
configs/          YAML experiment settings
data/             generated expert and processed demonstration data
src/              environments, models, algorithms, metrics, plots, utilities
scripts/          command-line entry points
tests/            unit, shape, metric, determinism, and rollout tests
outputs/          checkpoints, logs, figures, tables, reports, raw rollouts
```

## Environment Setup

```bash
cd GAIL-UAV-Reproduction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/00_check_env.py
```

On Windows PowerShell:

```powershell
cd GAIL-UAV-Reproduction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/00_check_env.py
```

## Data Preparation

Generate 10 synthetic expert trajectories and processed state-action demonstrations:

```bash
python scripts/01_generate_expert_data.py --config configs/breakthrough.yaml --num-trajectories 10 --points-per-trajectory 10000
```

Quick-mode data generation:

```bash
python scripts/01_generate_expert_data.py --config configs/quick.yaml
```

## Training Commands

```bash
python scripts/02_train_inpainting.py --config configs/inpainting.yaml
python scripts/03_eval_inpainting.py --config configs/inpainting.yaml
python scripts/04_train_gail.py --config configs/gail_ppo.yaml
python scripts/05_train_baselines.py --config configs/baselines.yaml --methods APF ACO PPO DDPG
```

Each training script supports `--seed` and `--resume`.

## Evaluation Commands

```bash
python scripts/06_eval_breakthrough.py --config configs/breakthrough.yaml --methods APF ACO PPO DDPG GAIL
python scripts/07_eval_interception.py --config configs/interception.yaml --methods APF ACO PPO DDPG GAIL
python scripts/08_run_ablation.py --config configs/ablation.yaml
python scripts/09_make_all_figures.py
```

## One-Click Reproduction Command

Quick mode:

```bash
bash scripts/10_reproduce_all.sh --quick
```

Full mode:

```bash
bash scripts/10_reproduce_all.sh --full
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/10_reproduce_all.ps1 --quick
```

## Results Table

Run `python scripts/09_make_all_figures.py` after the experiments. The values are read from raw predictions, rollout files, and training logs, then summarized in the generated report and CSV tables.

| Experiment | Generated table | Raw source |
|---|---|---|
| Sequential inpainting | `table2_inpainting_metrics.csv` | `outputs/raw_rollouts/inpainting_*.npz` |
| Breakthrough planning | `table3_breakthrough_path_metrics.csv` | `outputs/raw_rollouts/breakthrough_*_single.npz` |
| Interception planning | `table4_interception_metrics.csv` | `outputs/raw_rollouts/interception_*_rollouts.npz` |
| Ablation | `table5_ablation_metrics.csv` | `outputs/raw_rollouts/ablation_*.npz` |

## Visualization Examples

All figures are generated from raw logs:

```bash
python scripts/09_make_all_figures.py
```

The HTML report embeds generated figures for quick inspection.

## Troubleshooting

If PyTorch is not installed, install a CPU or CUDA build matching your machine from the PyTorch package index, then rerun `python scripts/00_check_env.py`.

If a training script cannot find expert data, run `python scripts/01_generate_expert_data.py --config configs/quick.yaml` first.

If full mode is slow, validate the pipeline with quick mode before running full-scale experiments.

## Hardware Recommendation

Quick mode runs on CPU. Full mode benefits from a CUDA GPU with at least 8 GB VRAM and 16 GB system memory.

## Citation

```bibtex
@article{yu2026gailuav,
  title={Expert-Level Autonomous Navigation for UAVs in Dynamic Environments: A GAIL Framework with Sequential Trajectory Inpainting},
  author={Yu, Ning and Zhao, Yanyan and Wu, Yibo and Zhao, Hongwei},
  year={2026}
}
```

