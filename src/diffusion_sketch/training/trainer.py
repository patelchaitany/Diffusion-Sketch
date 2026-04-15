"""Single-GPU training loop with TensorBoard + CSV logging."""

import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from diffusion_sketch.config import Config
from diffusion_sketch.data import SketchColorDataset
from diffusion_sketch.models import ConditionalUNet, GaussianDiffusion
from diffusion_sketch.losses import CombinedDiffusionLoss
from .utils import save_checkpoint


def _gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def _init_csv(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "loss", "lr", "gpu_mb"])


def _append_csv(path, row: dict):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([row["step"], row["epoch"], row["loss"], row["lr"], row["gpu_mb"]])


@torch.no_grad()
def _save_val_samples(diffusion, model, val_loader, step, sample_dir, ddim_steps, writer, device):
    """Run DDIM on 4 validation sketches and save grids to disk + TensorBoard."""
    model.eval()
    sketch, target = next(iter(val_loader))
    sketch, target = sketch.to(device), target.to(device)
    n = min(4, sketch.shape[0])
    sketch, target = sketch[:n], target[:n]

    generated = diffusion.sample_ddim(model, sketch, sketch.shape, ddim_steps=ddim_steps)

    os.makedirs(sample_dir, exist_ok=True)
    save_image(sketch * 0.5 + 0.5, os.path.join(sample_dir, f"sketch_step{step}.png"), nrow=n)
    save_image(generated * 0.5 + 0.5, os.path.join(sample_dir, f"generated_step{step}.png"), nrow=n)
    save_image(target * 0.5 + 0.5, os.path.join(sample_dir, f"target_step{step}.png"), nrow=n)

    if writer is not None:
        writer.add_images("samples/sketch", sketch * 0.5 + 0.5, step)
        writer.add_images("samples/generated", generated * 0.5 + 0.5, step)
        writer.add_images("samples/ground_truth", target * 0.5 + 0.5, step)
        writer.flush()

    model.train()


def run_training(cfg: Config):
    """Plain single-GPU training — no Ray, no distributed."""
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

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=cfg.training["weight_decay"],
    )
    total_epochs = cfg.training["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-6,
    )

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

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    warmup_steps = int(cfg.loss["aux_warmup_frac"] * total_epochs * len(train_loader))
    timesteps = cfg.diffusion["timesteps"]
    global_step = 0

    tb_dir = os.path.abspath(cfg.paths.get("tensorboard_dir", "runs"))
    writer = SummaryWriter(log_dir=tb_dir)

    csv_path = os.path.join(tb_dir, "loss_log.csv")
    _init_csv(csv_path)

    sample_every = cfg.training["sample_every_steps"]
    ckpt_every = cfg.training["checkpoint_every"]
    sample_dir = cfg.paths["sample_dir"]
    ddim_steps = cfg.training["ddim_sample_steps"]

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"\n{'='*60}")
    print(f"  Device:         {device}")
    print(f"  Parameters:     {param_count:.1f}M")
    print(f"  Train images:   {len(train_ds)}")
    print(f"  Val images:     {len(val_ds)}")
    print(f"  Epochs:         {total_epochs}")
    print(f"  Batch size:     {cfg.training['batch_size']}")
    print(f"  Sample every:   {sample_every} steps")
    print(f"  TensorBoard:    tensorboard --logdir {tb_dir}")
    print(f"  CSV log:        {csv_path}")
    print(f"{'='*60}\n")

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

            with torch.amp.autocast("cuda", enabled=use_amp):
                noise_pred = model(x_noisy, t, sketch)
                x0_pred = diffusion.predict_x0_from_noise(x_noisy, t, noise_pred).clamp(-1, 1)
                loss, loss_dict = criterion(
                    noise, noise_pred, target, x0_pred,
                    apply_aux=(global_step >= warmup_steps),
                )

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training["grad_clip"])
                optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            num_batches += 1
            global_step += 1

            pbar.set_postfix(loss=f"{batch_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            writer.add_scalar("loss/step", batch_loss, global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f"loss_components/{k}", v, global_step)

            gpu_mb = round(_gpu_memory_mb(), 1)
            _append_csv(csv_path, {
                "step": global_step,
                "epoch": epoch + 1,
                "loss": round(batch_loss, 6),
                "lr": scheduler.get_last_lr()[0],
                "gpu_mb": gpu_mb,
            })

            if sample_every > 0 and global_step % sample_every == 0:
                _save_val_samples(
                    diffusion, model, val_loader, global_step,
                    sample_dir, ddim_steps, writer, device,
                )

        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(num_batches, 1)

        writer.add_scalar("loss/epoch_avg", avg_loss, epoch + 1)
        writer.add_scalar("training/lr", scheduler.get_last_lr()[0], epoch + 1)
        writer.add_scalar("training/epoch_time_sec", epoch_time, epoch + 1)
        writer.add_scalar("system/gpu_peak_memory_mb", _gpu_memory_mb(), epoch + 1)
        writer.flush()

        print(f"  -> avg loss: {avg_loss:.4f}  |  {epoch_time:.1f}s  |  GPU {_gpu_memory_mb():.0f}MB\n")

        if (epoch + 1) % ckpt_every == 0:
            ckpt_dir = cfg.paths["checkpoint_dir"]
            ckpt_path = os.path.join(ckpt_dir, f"diffusion_epoch{epoch+1}.pt")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_checkpoint(model, optimizer, epoch + 1, ckpt_path)
            print(f"  [checkpoint] {ckpt_path}\n")

    writer.close()
    print(f"Training complete. Final avg loss: {avg_loss:.4f}")
