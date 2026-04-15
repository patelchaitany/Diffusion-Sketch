"""Single-GPU training loop with stability best-practices.

Stability features:
  - Linear LR warmup before cosine decay
  - EMA (exponential moving average) of model weights for sampling
  - Min-SNR-γ per-timestep loss weighting
  - Gradient norm logging + NaN/Inf detection
  - TensorBoard + CSV logging
"""

import csv
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from diffusion_sketch.config import Config
from diffusion_sketch.data import SketchColorDataset
from diffusion_sketch.models import ConditionalUNet, GaussianDiffusion
from diffusion_sketch.losses import CombinedDiffusionLoss
from .utils import EMA, save_checkpoint


def _gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def _init_csv(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "epoch", "loss", "lr", "grad_norm", "gpu_mb"])


def _append_csv(path, row: dict):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            row["step"], row["epoch"], row["loss"],
            row["lr"], row["grad_norm"], row["gpu_mb"],
        ])


def _warmup_cosine_lr(step, warmup_steps, total_steps, base_lr, min_lr=1e-6):
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def _save_val_samples(diffusion, ema_model, val_loader, step, sample_dir, ddim_steps, writer, device):
    """Run DDIM on 4 validation sketches using EMA weights."""
    ema_model.eval()
    sketch, target = next(iter(val_loader))
    sketch, target = sketch.to(device), target.to(device)
    n = min(4, sketch.shape[0])
    sketch, target = sketch[:n], target[:n]

    generated = diffusion.sample_ddim(ema_model, sketch, sketch.shape, ddim_steps=ddim_steps)

    os.makedirs(sample_dir, exist_ok=True)
    save_image(sketch * 0.5 + 0.5, os.path.join(sample_dir, f"sketch_step{step}.png"), nrow=n)
    save_image(generated * 0.5 + 0.5, os.path.join(sample_dir, f"generated_step{step}.png"), nrow=n)
    save_image(target * 0.5 + 0.5, os.path.join(sample_dir, f"target_step{step}.png"), nrow=n)

    if writer is not None:
        writer.add_images("samples/sketch", sketch * 0.5 + 0.5, step)
        writer.add_images("samples/generated", generated * 0.5 + 0.5, step)
        writer.add_images("samples/ground_truth", target * 0.5 + 0.5, step)
        writer.flush()


def run_training(cfg: Config):
    """Plain single-GPU training with stability best-practices."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConditionalUNet(
        in_channels=cfg.model["in_channels"],
        cond_channels=cfg.model["cond_channels"],
        base_channels=cfg.model["base_channels"],
        channel_mults=tuple(cfg.model["channel_mults"]),
        attention_resolutions=tuple(cfg.model["attention_resolutions"]),
        num_res_blocks=cfg.model["num_res_blocks"],
        dropout=cfg.model["dropout"],
        gradient_checkpointing=cfg.model.get("gradient_checkpointing", False),
    ).to(device)

    ema = EMA(model, decay=cfg.training.get("ema_decay", 0.9999))

    diffusion = GaussianDiffusion(
        timesteps=cfg.diffusion["timesteps"],
        beta_start=cfg.diffusion["beta_start"],
        beta_end=cfg.diffusion["beta_end"],
    ).to(device)

    criterion = CombinedDiffusionLoss(
        lambda_l1=cfg.loss["lambda_l1"],
        lambda_laplacian=cfg.loss["lambda_laplacian"],
        lambda_gradient=cfg.loss["lambda_gradient"],
        lambda_histogram=cfg.loss["lambda_histogram"],
        lambda_perceptual=cfg.loss["lambda_perceptual"],
        use_perceptual=cfg.loss["use_perceptual"],
    ).to(device)

    base_lr = cfg.training["learning_rate"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.training["weight_decay"],
    )

    total_epochs = cfg.training["epochs"]
    train_ds = SketchColorDataset(cfg.data["train_dir"], image_size=cfg.data["image_size"])
    val_ds = SketchColorDataset(cfg.data["val_dir"], image_size=cfg.data["image_size"], augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training["batch_size"],
        shuffle=True,
        num_workers=cfg.data["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    steps_per_epoch = len(train_loader)
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = cfg.training.get("warmup_steps", 1000)
    warmup_aux = int(cfg.loss["aux_warmup_frac"] * total_steps)
    min_snr_gamma = cfg.training.get("min_snr_gamma", 5.0)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    timesteps = cfg.diffusion["timesteps"]
    global_step = 0

    tb_dir = os.path.abspath(cfg.paths.get("tensorboard_dir", "runs"))
    writer = SummaryWriter(log_dir=tb_dir)

    csv_path = os.path.join(tb_dir, "loss_log.csv")
    _init_csv(csv_path)

    sample_every = cfg.training["sample_every_steps"]
    sample_dir = cfg.paths["sample_dir"]
    ddim_steps = cfg.training["ddim_sample_steps"]
    grad_clip = cfg.training["grad_clip"]

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"\n{'='*60}")
    print(f"  Device:         {device}")
    print(f"  Parameters:     {param_count:.1f}M")
    print(f"  Train images:   {len(train_ds)}")
    print(f"  Val images:     {len(val_ds)}")
    print(f"  Epochs:         {total_epochs}")
    print(f"  Total steps:    {total_steps}")
    print(f"  Batch size:     {cfg.training['batch_size']}")
    print(f"  LR warmup:      {warmup_steps} steps")
    print(f"  Aux warmup:     {warmup_aux} steps")
    print(f"  EMA decay:      {ema.decay}")
    print(f"  Min-SNR-γ:      {min_snr_gamma}")
    print(f"  Sample every:   {sample_every} steps")
    print(f"  TensorBoard:    tensorboard --logdir {tb_dir}")
    print(f"  CSV log:        {csv_path}")
    print(f"{'='*60}\n")

    nan_count = 0

    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()
        if use_amp:
            torch.cuda.reset_peak_memory_stats()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=True)
        for sketch, target in pbar:
            sketch, target = sketch.to(device), target.to(device)
            t = torch.randint(0, timesteps, (target.shape[0],), device=device)
            x_noisy, noise = diffusion.q_sample(target, t)

            # --- warmup + cosine LR schedule (per step) ---
            lr = _warmup_cosine_lr(global_step, warmup_steps, total_steps, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with torch.amp.autocast("cuda", enabled=use_amp):
                noise_pred = model(x_noisy, t, sketch)

                # min-SNR-γ weighted noise loss (per-sample, then mean)
                snr_w = diffusion.min_snr_weight(t, gamma=min_snr_gamma)
                per_sample_mse = F.mse_loss(noise_pred, noise, reduction="none").mean(dim=[1, 2, 3])
                weighted_mse = (snr_w.squeeze() * per_sample_mse).mean()

                apply_aux = global_step >= warmup_aux
                if apply_aux:
                    x0_pred = diffusion.predict_x0_from_noise(x_noisy, t, noise_pred).clamp(-1, 1)
                    _, aux_dict = criterion(noise, noise_pred, target, x0_pred, apply_aux=True)
                    aux_loss = criterion._last_aux
                    loss = weighted_mse + aux_loss
                    loss_dict = {"mse_snr": weighted_mse.item()}
                    loss_dict.update(aux_dict)
                else:
                    loss = weighted_mse
                    loss_dict = {"mse_snr": weighted_mse.item()}

            # --- NaN guard ---
            if not torch.isfinite(loss):
                nan_count += 1
                optimizer.zero_grad()
                if nan_count % 10 == 1:
                    print(f"  [!] NaN/Inf loss at step {global_step}, skipping (total: {nan_count})")
                writer.add_scalar("debug/nan_count", nan_count, global_step)
                global_step += 1
                continue

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
                if math.isfinite(grad_norm):
                    scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
                if math.isfinite(grad_norm):
                    optimizer.step()

            ema.update(model)

            batch_loss = loss.item()
            epoch_loss += batch_loss
            num_batches += 1
            global_step += 1

            pbar.set_postfix(loss=f"{batch_loss:.4f}", gn=f"{grad_norm:.2f}", lr=f"{lr:.2e}")

            writer.add_scalar("loss/step", batch_loss, global_step)
            writer.add_scalar("debug/grad_norm", grad_norm, global_step)
            writer.add_scalar("training/lr", lr, global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f"loss_components/{k}", v, global_step)

            gpu_mb = round(_gpu_memory_mb(), 1)
            _append_csv(csv_path, {
                "step": global_step,
                "epoch": epoch + 1,
                "loss": round(batch_loss, 6),
                "lr": lr,
                "grad_norm": round(grad_norm, 4),
                "gpu_mb": gpu_mb,
            })

            if sample_every > 0 and global_step % sample_every == 0:
                _save_val_samples(
                    diffusion, ema.shadow, val_loader, global_step,
                    sample_dir, ddim_steps, writer, device,
                )
                model.train()

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(num_batches, 1)

        writer.add_scalar("loss/epoch_avg", avg_loss, epoch + 1)
        writer.add_scalar("training/epoch_time_sec", epoch_time, epoch + 1)
        writer.add_scalar("system/gpu_peak_memory_mb", _gpu_memory_mb(), epoch + 1)
        writer.flush()

        print(f"  -> avg loss: {avg_loss:.4f}  |  {epoch_time:.1f}s  |  GPU {_gpu_memory_mb():.0f}MB\n")

        ckpt_dir = cfg.paths["checkpoint_dir"]
        ckpt_path = os.path.join(ckpt_dir, f"diffusion_epoch{epoch+1}.pt")
        os.makedirs(ckpt_dir, exist_ok=True)
        save_checkpoint(model, optimizer, epoch + 1, ckpt_path, ema=ema)
        print(f"  [checkpoint] {ckpt_path}\n")

    writer.close()
    print(f"Training complete. Final avg loss: {avg_loss:.4f}")
