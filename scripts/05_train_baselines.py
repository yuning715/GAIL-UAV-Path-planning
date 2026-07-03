from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from _common import REPO_ROOT, add_common_args, record_command, section_config, seed_from
from src.algorithms.aco import ACOController
from src.algorithms.apf import APFController
from src.algorithms.ddpg import DDPGAgent, DDPGConfig
from src.algorithms.ppo import PPOConfig, PPOTrainer
from src.algorithms.replay_buffer import ReplayBuffer
from src.algorithms.rollout_buffer import RolloutBuffer
from src.envs.breakthrough_env import BreakthroughEnv
from src.models.policy import PolicyNetwork
from src.models.value import ValueNetwork
from src.utils.checkpoint import save_checkpoint
from src.utils.runtime import runtime_stage
from src.utils.seed import set_seed


def _rollout_controller(controller, env, max_steps: int) -> dict:
    state = env.reset()
    total = 0.0
    for _ in range(max_steps):
        action = controller.act(env, state)
        state, reward, done, _info = env.step(action)
        total += reward
        if done:
            break
    rec = env.record()
    return {"reward": total, "success": rec.success, "steps": env.step_count}


def _train_ppo(cfg: dict, env_cfg: dict, seed: int, log_rows: list[dict]) -> None:
    device = torch.device("cpu")
    state_dim = int(cfg.get("state_dim", 6))
    action_dim = int(cfg.get("action_dim", 2))
    policy = PolicyNetwork(state_dim, action_dim, hidden_sizes=[32, 32], lstm_hidden_size=32).to(device)
    value = ValueNetwork(state_dim, [32, 32, 16]).to(device)
    ppo_cfg = PPOConfig(
        learning_rate=float(cfg.get("training", {}).get("learning_rate", 3e-4)),
        gamma=float(cfg.get("controlled_params", {}).get("gamma", 0.5)),
        epochs_per_iteration=2,
        minibatch_size=int(cfg.get("training", {}).get("batch_size", 128)),
    )
    trainer = PPOTrainer(policy, value, ppo_cfg)
    episodes = int(cfg.get("training", {}).get("episodes", 120))
    max_steps = int(cfg.get("training", {}).get("max_steps_per_episode", 300))
    for episode in range(1, episodes + 1):
        env = BreakthroughEnv({**env_cfg, "single_episode_max_steps": max_steps}, seed=seed + episode)
        state = env.reset(seed=seed + episode)
        buffer = RolloutBuffer()
        for _ in range(max_steps):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_t, log_prob_t = policy.act(state_t)
                value_t = value(state_t)
            action = action_t.squeeze(0).numpy()
            next_state, reward, done, _ = env.step(action)
            with torch.no_grad():
                next_value = value(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0))
            buffer.add(state, action, float(log_prob_t.item()), reward, done, float(value_t.item()), float(next_value.item()))
            state = next_state
            if done:
                break
        metrics = trainer.update(buffer, device)
        log_rows.append({"method": "PPO", "episode": episode, "reward": float(np.mean(buffer.rewards)), **metrics})
    save_checkpoint(
        REPO_ROOT / cfg.get("checkpoint_dir", "outputs/checkpoints/baselines") / "ppo_final.pt",
        policy=policy.state_dict(),
        value=value.state_dict(),
        config=cfg,
    )


def _train_ddpg(cfg: dict, env_cfg: dict, seed: int, log_rows: list[dict]) -> None:
    params = cfg.get("controlled_params", {})
    train = cfg.get("training", {})
    ddpg_cfg = DDPGConfig(
        gamma=float(params.get("gamma", 0.5)),
        learning_rate=float(train.get("learning_rate", 3e-4)),
        tau=float(train.get("tau", 0.005)),
        batch_size=int(train.get("batch_size", 128)),
        ou_beta=float(params.get("ou_beta", 0.1)),
        ou_sigma_1=float(params.get("ou_sigma_1", 0.2)),
        gaussian_variance_initial=float(params.get("gaussian_variance_initial", 0.5)),
        gaussian_attenuation_delta=float(params.get("gaussian_attenuation_delta", 0.001)),
    )
    agent = DDPGAgent(
        int(cfg.get("state_dim", 6)),
        int(cfg.get("action_dim", 2)),
        list(cfg.get("actor", {}).get("hidden_layers", [32, 32])),
        ddpg_cfg,
        seed=seed,
    )
    buffer = ReplayBuffer(int(train.get("replay_capacity", 50000)), seed=seed)
    episodes = int(train.get("episodes", 120))
    max_steps = int(train.get("max_steps_per_episode", 300))
    for episode in range(1, episodes + 1):
        env = BreakthroughEnv({**env_cfg, "single_episode_max_steps": max_steps}, seed=seed + episode)
        state = env.reset(seed=seed + episode)
        rewards = []
        for _ in range(max_steps):
            action = agent.act(state, episode=episode, explore=True)
            next_state, reward, done, _ = env.step(action)
            buffer.add(state, action, reward, next_state, done)
            metrics = agent.update(buffer)
            rewards.append(reward)
            state = next_state
            if done:
                break
        log_rows.append({"method": "DDPG", "episode": episode, "reward": float(np.mean(rewards)), **metrics})
    save_checkpoint(
        REPO_ROOT / cfg.get("checkpoint_dir", "outputs/checkpoints/baselines") / "ddpg_final.pt",
        actor=agent.actor.state_dict(),
        critic=agent.critic.state_dict(),
        config=cfg,
    )


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--methods", nargs="+", default=["APF", "ACO", "PPO", "DDPG"])
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    record_command()
    cfg = section_config(args.config, "baselines", "configs/baselines.yaml")
    if args.resume:
        cfg.setdefault("training", {})["resume"] = args.resume
    env_cfg = section_config(args.config, "breakthrough", "configs/breakthrough.yaml")
    seed = seed_from(args, cfg)
    set_seed(seed)
    rows: list[dict] = []
    with runtime_stage("train_baselines", REPO_ROOT / "outputs/tables/runtime_complexity.csv"):
        max_steps = int(cfg.get("training", {}).get("max_steps_per_episode", 300))
        for method in args.methods:
            method = method.upper()
            if method == "APF":
                env = BreakthroughEnv({**env_cfg, "single_episode_max_steps": max_steps}, seed=seed)
                rows.append({"method": "APF", "episode": 1, **_rollout_controller(APFController(**cfg.get("apf", {})), env, max_steps)})
            elif method == "ACO":
                env = BreakthroughEnv({**env_cfg, "single_episode_max_steps": max_steps}, seed=seed)
                rows.append({"method": "ACO", "episode": 1, **_rollout_controller(ACOController(**cfg.get("aco", {}), seed=seed), env, max_steps)})
            elif method == "PPO":
                _train_ppo(cfg, env_cfg, seed, rows)
            elif method == "DDPG":
                _train_ddpg(cfg, env_cfg, seed, rows)
    log_path = REPO_ROOT / cfg.get("log_dir", "outputs/logs") / "baseline_training.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
