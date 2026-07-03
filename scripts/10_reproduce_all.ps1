$ErrorActionPreference = "Stop"

$Mode = "quick"
if ($args.Count -gt 0) {
  if ($args[0] -eq "--full") { $Mode = "full" }
  if ($args[0] -eq "--quick") { $Mode = "quick" }
}

$Config = "configs/quick.yaml"
if ($Mode -eq "full") {
  $Config = "configs/full.yaml"
}

python scripts/00_check_env.py
python scripts/01_generate_expert_data.py --config $Config
python scripts/02_train_inpainting.py --config $Config
python scripts/03_eval_inpainting.py --config $Config
python scripts/04_train_gail.py --config $Config
python scripts/05_train_baselines.py --config $Config --methods APF ACO PPO DDPG
python scripts/06_eval_breakthrough.py --config $Config --methods APF ACO PPO DDPG GAIL
python scripts/07_eval_interception.py --config $Config --methods APF ACO PPO DDPG GAIL
python scripts/08_run_ablation.py --config $Config
python scripts/09_make_all_figures.py --config $Config

Write-Host "Reproduction pipeline completed in $Mode mode."

