# Data Directory

`data/expert` stores generated expert trajectories with velocity, turning angle, pitch angle, 3D state, and action arrays.

`data/processed` stores fitted scalers and state-action demonstrations used by GAIL.

`data/synthetic` is reserved for additional generated scenario data.

Generate the default data:

```bash
python scripts/01_generate_expert_data.py --config configs/quick.yaml
```

