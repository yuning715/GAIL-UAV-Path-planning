"""Training loop for GAIL-PPO UAV path planning."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.algorithms.gail import GAILConfig, GAILTrainer
from src.algorithms.ppo import PPOConfig, PPOTrainer
from src.algorithms.rollout_buffer import RolloutBuffer
from src.data.expert_dataset import StateActionDataset
from src.envs.breakthrough_env import BreakthroughEnv
from src.models.gail_discriminator import GAILDiscriminator
from src.models.policy import PolicyNetwork
from src.models.value import ValueNetwork
from src.utils.checkpoint import save_checkpoint
from src.utils.device import get_device
from src.utils.logger import JsonlLogger, setup_logger
from src.utils.seed import set_seed


def build_gail_components(config: dict[str, Any], device: torch.device):
    """Build policy, value, discriminator, and trainers."""
    state_dim = int(config.get("state_dim", 6))
    action_dim = int(config.get("action_dim", 2))
    policy = PolicyNetwork(state_dim, action_dim, **config.get("policy", {})).to(device)
    value = ValueNetwork(state_dim, config.get("value", {}).get("hidden_sizes", [256, 128, 64])).to(device)
    discriminator = GAILDiscriminator(state_dim, action_dim, **config.get("discriminator", {})).to(device)
    ppo_trainer = PPOTrainer(policy, value, PPOConfig(**config.get("ppo", {})))
    gail_trainer = GAILTrainer(discriminator, GAILConfig(**config.get("gail_params", {})))
    return policy, value, discriminator, ppo_trainer, gail_trainer


def _sample_expert_batch(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def train_gail_ppo(config: dict[str, Any], env_config: dict[str, Any], seed: int = 0) -> Path:
    """Train GAIL-PPO in the breakthrough environment and save checkpoints/logs."""
    set_seed(seed)
    device = get_device(config.get("device", "auto") != "cpu")
    logger = setup_logger("train_gail", config.get("log_dir", "outputs/logs"))
    policy, value, discriminator, ppo_trainer, gail_trainer = build_gail_components(config, device)
    expert_path = Path(config.get("expert_dir", "data/processed")) / "expert_state_actions.npz"
    expert_dataset = StateActionDataset(expert_path)
    expert_loader = DataLoader(expert_dataset, batch_size=int(config.get("ppo", {}).get("minibatch_size", 256)), shuffle=True)
    expert_iter = iter(expert_loader)
    checkpoint_dir = Path(config.get("checkpoint_dir", "outputs/checkpoints/gail"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config.get("log_dir", "outputs/logs"))
    jsonl = JsonlLogger(log_dir / "gail_training.jsonl")
    csv_path = log_dir / "gail_training.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(
                f,
                fieldnames=[
                    "iteration",
                    "reward_mean",
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "discriminator_loss",
                    "ACC",
                    "PPV",
                    "TPR",
                    "seconds",
                ],
            ).writeheader()
    train_cfg = config.get("training", {})
    iterations = int(train_cfg.get("iterations", 120))
    episodes_per_iteration = int(train_cfg.get("episodes_per_iteration", 4))
    max_steps = int(train_cfg.get("max_steps_per_episode", 300))
    beta = float(config.get("gail_params", {}).get("task_reward_beta", 0.1))
    final_path = checkpoint_dir / "gail_ppo_final.pt"
    for iteration in range(1, iterations + 1):
        started = time.perf_counter()
        env = BreakthroughEnv({**env_config, "single_episode_max_steps": max_steps}, seed=seed + iteration)
        buffer = RolloutBuffer()
        task_rewards: list[float] = []
        for ep in range(episodes_per_iteration):
            state = env.reset(seed=seed + 1000 * iteration + ep)
            for _ in range(max_steps):
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_t, log_prob_t = policy.act(state_t)
                    value_t = value(state_t)
                action = action_t.squeeze(0).cpu().numpy()
                next_state, reward, done, _info = env.step(action)
                with torch.no_grad():
                    next_value_t = value(torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0))
                buffer.add(
                    state,
                    action,
                    float(log_prob_t.item()),
                    reward,
                    done,
                    float(value_t.item()),
                    float(next_value_t.item()),
                )
                task_rewards.append(reward)
                state = next_state
                if done:
                    break
        if len(buffer) == 0:
            continue
        tensors = buffer.tensors(ppo_trainer.config.gamma, ppo_trainer.config.gae_lambda, device)
        (expert_states, expert_actions), expert_iter = _sample_expert_batch(expert_iter, expert_loader)
        expert_states = expert_states.to(device)
        expert_actions = expert_actions.to(device)
        sample_count = min(len(tensors["states"]), len(expert_states))
        policy_states = tensors["states"][:sample_count]
        policy_actions = tensors["actions"][:sample_count]
        disc_metrics = gail_trainer.discriminator_step(
            expert_states[:sample_count], expert_actions[:sample_count], policy_states, policy_actions
        )
        with torch.no_grad():
            gail_rewards = gail_trainer.reward(tensors["states"], tensors["actions"]).cpu().numpy()
        buffer.rewards = list(gail_rewards + beta * np.asarray(buffer.rewards, dtype=np.float32))
        ppo_metrics = ppo_trainer.update(buffer, device)
        row = {
            "iteration": iteration,
            "reward_mean": float(np.mean(buffer.rewards)),
            **ppo_metrics,
            **disc_metrics,
            "seconds": time.perf_counter() - started,
        }
        jsonl.write(**row)
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)
        if iteration % int(train_cfg.get("log_every", 1)) == 0:
            logger.info(
                "iter=%s reward=%.3f acc=%.3f ppv=%.3f tpr=%.3f",
                iteration,
                row["reward_mean"],
                row["ACC"],
                row["PPV"],
                row["TPR"],
            )
        if iteration % int(train_cfg.get("save_every", 10)) == 0:
            save_checkpoint(
                checkpoint_dir / f"gail_ppo_iter_{iteration:04d}.pt",
                iteration=iteration,
                policy=policy.state_dict(),
                value=value.state_dict(),
                discriminator=discriminator.state_dict(),
                config=config,
            )
    save_checkpoint(
        final_path,
        iteration=iterations,
        policy=policy.state_dict(),
        value=value.state_dict(),
        discriminator=discriminator.state_dict(),
        config=config,
    )
    return final_path

