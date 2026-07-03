#!/usr/bin/env bash
set -euo pipefail

MODE="quick"
if [[ "${1:-}" == "--full" ]]; then
  MODE="full"
elif [[ "${1:-}" == "--quick" ]]; then
  MODE="quick"
fi

CONFIG="configs/quick.yaml"
if [[ "$MODE" == "full" ]]; then
  CONFIG="configs/full.yaml"
fi

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif grep -qi microsoft /proc/version 2>/dev/null && command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="python.exe"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="python.exe"
else
  echo "No Python interpreter found. Set PYTHON=/path/to/python and rerun." >&2
  exit 1
fi

"$PYTHON_BIN" scripts/00_check_env.py --config "$CONFIG"
"$PYTHON_BIN" scripts/01_generate_expert_data.py --config "$CONFIG"
"$PYTHON_BIN" scripts/02_train_inpainting.py --config "$CONFIG"
"$PYTHON_BIN" scripts/03_eval_inpainting.py --config "$CONFIG"
"$PYTHON_BIN" scripts/04_train_gail.py --config "$CONFIG"
"$PYTHON_BIN" scripts/05_train_baselines.py --config "$CONFIG" --methods APF ACO PPO DDPG
"$PYTHON_BIN" scripts/06_eval_breakthrough.py --config "$CONFIG" --methods APF ACO PPO DDPG GAIL
"$PYTHON_BIN" scripts/07_eval_interception.py --config "$CONFIG" --methods APF ACO PPO DDPG GAIL
"$PYTHON_BIN" scripts/08_run_ablation.py --config "$CONFIG"
"$PYTHON_BIN" scripts/09_make_all_figures.py --config "$CONFIG"

echo "Reproduction pipeline completed in $MODE mode."
