"""Training loop for GAN-LSTM sequential trajectory inpainting."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.expert_dataset import InpaintingDataset
from src.data.preprocessing import FeatureScaler, fit_feature_scaler
from src.data.synthetic_expert_generator import load_feature_trajectories
from src.models.gan_lstm import TrajectoryGenerator
from src.models.trajectory_discriminator import TrajectoryDiscriminator
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_device
from src.utils.logger import JsonlLogger, setup_logger
from src.utils.seed import set_seed


def _gradient_penalty(discriminator, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    batch = real.size(0)
    alpha = torch.rand(batch, 1, 1, device=real.device)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    prob = discriminator(interpolated)
    grad = torch.autograd.grad(prob.sum(), interpolated, create_graph=True, retain_graph=True)[0]
    return ((grad.reshape(batch, -1).norm(2, dim=1) - 1.0) ** 2).mean()


def build_inpainting_models(config: dict[str, Any], device: torch.device):
    """Create generator and discriminator from config."""
    gen_cfg = config.get("generator", {})
    dis_cfg = config.get("discriminator", {})
    generator = TrajectoryGenerator(
        feature_dim=int(config.get("feature_dim", 3)),
        hidden_size=int(gen_cfg.get("hidden_size", 256)),
        num_layers=int(gen_cfg.get("num_layers", 2)),
        dropout=float(gen_cfg.get("dropout", 0.2)),
        output_activation=str(gen_cfg.get("output_activation", "tanh")),
    ).to(device)
    discriminator = TrajectoryDiscriminator(
        feature_dim=int(config.get("feature_dim", 3)),
        cnn_filters=list(dis_cfg.get("cnn_filters", [64, 128, 256])),
        lstm_hidden_size=int(dis_cfg.get("lstm_hidden_size", 128)),
        dropout=float(dis_cfg.get("dropout", 0.2)),
    ).to(device)
    return generator, discriminator


def train_inpainting(config: dict[str, Any], seed: int = 0) -> Path:
    """Train the GAN-LSTM inpainting model and return the final checkpoint path."""
    set_seed(seed)
    device = get_device(config.get("device", "auto") != "cpu")
    logger = setup_logger("train_inpainting", config.get("log_dir", "outputs/logs"))
    train_indices = list(config.get("train_indices", range(8)))
    trajectories = load_feature_trajectories(config["expert_dir"], train_indices)
    scaler = fit_feature_scaler(trajectories)
    processed_dir = Path(config.get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = processed_dir / "inpainting_scaler.npz"
    scaler.save(scaler_path)
    dataset = InpaintingDataset(
        trajectories=trajectories,
        scaler=scaler,
        seq_len=int(config.get("seq_len", 128)),
        stride=int(config.get("stride", 64)),
        missing_rate=float(config.get("missing", {}).get("random_missing_rate", 0.02)),
        seed=seed,
    )
    train_cfg = config.get("training", {})
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=True,
        drop_last=False,
    )
    generator, discriminator = build_inpainting_models(config, device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=float(train_cfg.get("learning_rate", 1e-4)))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=float(train_cfg.get("learning_rate", 1e-4)))
    resume = train_cfg.get("resume")
    start_epoch = 1
    if resume:
        ckpt = load_checkpoint(resume, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
    checkpoint_dir = Path(config.get("checkpoint_dir", "outputs/checkpoints/inpainting"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    jsonl = JsonlLogger(Path(config.get("log_dir", "outputs/logs")) / "inpainting_training.jsonl")
    csv_path = Path(config.get("log_dir", "outputs/logs")) / "inpainting_training.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(
                f, fieldnames=["epoch", "generator_loss", "discriminator_loss", "reconstruction_loss", "seconds"]
            ).writeheader()
    bce = torch.nn.BCELoss()
    alpha = float(train_cfg.get("alpha", 10.0))
    gp_weight = float(train_cfg.get("gradient_penalty_weight", 10.0))
    epochs = int(train_cfg.get("epochs", 2000))
    max_batches = train_cfg.get("max_batches_per_epoch")
    loss_type = train_cfg.get("gan_loss_type", "bce_recon")
    modified_lambda = float(train_cfg.get("modified_lambda", 1.0))
    final_path = checkpoint_dir / "gan_lstm_final.pt"
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        g_losses: list[float] = []
        d_losses: list[float] = []
        recon_losses: list[float] = []
        for batch_idx, (incomplete, mask, real) in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            incomplete = incomplete.to(device)
            mask = mask.to(device)
            real = real.to(device)
            fake = generator(incomplete, mask).detach()
            real_prob = discriminator(real)
            fake_prob = discriminator(fake)
            d_loss = bce(real_prob, torch.ones_like(real_prob)) + bce(fake_prob, torch.zeros_like(fake_prob))
            if gp_weight > 0:
                d_loss = d_loss + gp_weight * _gradient_penalty(discriminator, real, fake)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            fake = generator(incomplete, mask)
            fake_prob = discriminator(fake).clamp(1e-6, 1.0 - 1e-6)
            recon = F.mse_loss(fake * mask, real * mask)
            if loss_type == "modified_paper":
                adv = (-torch.log(1.0 - fake_prob)).mean() - modified_lambda * torch.log(fake_prob).mean()
            else:
                adv = bce(fake_prob, torch.ones_like(fake_prob))
            g_loss = adv + alpha * recon
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            g_losses.append(float(g_loss.detach().cpu()))
            d_losses.append(float(d_loss.detach().cpu()))
            recon_losses.append(float(recon.detach().cpu()))
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "generator_loss": float(np.mean(g_losses)),
            "discriminator_loss": float(np.mean(d_losses)),
            "reconstruction_loss": float(np.mean(recon_losses)),
            "seconds": elapsed,
        }
        jsonl.write(**row)
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)
        if epoch % int(train_cfg.get("log_every", 10)) == 0 or epoch == 1:
            logger.info(
                "epoch=%s g=%.4f d=%.4f recon=%.5f",
                epoch,
                row["generator_loss"],
                row["discriminator_loss"],
                row["reconstruction_loss"],
            )
        if epoch % int(train_cfg.get("save_every", 100)) == 0:
            save_checkpoint(
                checkpoint_dir / f"gan_lstm_epoch_{epoch:04d}.pt",
                epoch=epoch,
                generator=generator.state_dict(),
                discriminator=discriminator.state_dict(),
                opt_g=opt_g.state_dict(),
                opt_d=opt_d.state_dict(),
                scaler_path=str(scaler_path),
                config=config,
            )
    save_checkpoint(
        final_path,
        epoch=epochs,
        generator=generator.state_dict(),
        discriminator=discriminator.state_dict(),
        opt_g=opt_g.state_dict(),
        opt_d=opt_d.state_dict(),
        scaler_path=str(scaler_path),
        config=config,
    )
    return final_path

