"""Ray Train-based distributed training loop with TensorBoard logging."""

import logging
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import ray
from ray import train as ray_train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

logging.getLogger("ray.train").setLevel(logging.WARNING)
logging.getLogger("ray._private").setLevel(logging.ERROR)

from diffusion_sketch.config import Config
from diffusion_sketch.data import SketchColorDataset
from diffusion_sketch.models import ConditionalUNet, GaussianDiffusion
from diffusion_sketch.losses import CombinedDiffusionLoss
from .utils import save_checkpoint, save_samples


def _gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def _log_tb(writer, epoch, metrics, loss_dict):
    """Write all scalars to TensorBoard."""
    writer.add_scalar("loss/total", metrics["loss"], epoch)
    for k, v in loss_dict.items():
        writer.add_scalar(f"loss/{k}", v, epoch)
    writer.add_scalar("training/lr", metrics["lr"], epoch)
    writer.add_scalar("training/epoch_time_sec", metrics["epoch_time_sec"], epoch)
    writer.add_scalar("training/images_per_sec", metrics["images_per_sec"], epoch)
    writer.add_scalar("system/gpu_peak_memory_mb", metrics["gpu_peak_memory_mb"], epoch)
    writer.flush()


def _log_epoch(metrics, loss_dict):
    """Print a formatted log line to the terminal."""
    e = metrics["epoch"]
    total = metrics.get("total_epochs", "?")
    pct = metrics["progress_pct"]
    t = metrics["epoch_time_sec"]
    lr = metrics["lr"]
    ips = metrics["images_per_sec"]
    mem = metrics["gpu_peak_memory_mb"]

    header = f"Epoch {e}/{total} ({pct}%)  |  {t:.1f}s  |  {ips:.0f} img/s  |  GPU {mem:.0f}MB  |  lr {lr:.2e}"
    parts = [f"{k}={v:.4f}" for k, v in loss_dict.items()]
    losses_str = "  ".join(parts)
    print(f"  {header}")
    print(f"    losses: {losses_str}")
    print()


def _train_loop(train_cfg: dict):
    cfg = Config(train_cfg)
    device = ray_train.torch.get_device()

    model = ConditionalUNet(
        in_channels=cfg.model["in_channels"],
        cond_channels=cfg.model["cond_channels"],
        base_channels=cfg.model["base_channels"],
        channel_mults=tuple(cfg.model["channel_mults"]),
        attention_resolutions=tuple(cfg.model["attention_resolutions"]),
        num_res_blocks=cfg.model["num_res_blocks"],
        dropout=cfg.model["dropout"],
        gradient_checkpointing=cfg.model.get("gradient_checkpointing", False),
    )
    model = ray_train.torch.prepare_model(model)

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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training["epochs"], eta_min=1e-6,
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
    train_loader = ray_train.torch.prepare_data_loader(train_loader)
    val_loader = ray_train.torch.prepare_data_loader(val_loader)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    warmup_steps = int(cfg.loss["aux_warmup_frac"] * cfg.training["epochs"] * len(train_loader))
    global_step = 0
    timesteps = cfg.diffusion["timesteps"]

    is_rank0 = ray_train.get_context().get_world_rank() == 0
    tb_dir = cfg.paths.get("tensorboard_dir", "runs")
    writer = SummaryWriter(log_dir=tb_dir) if is_rank0 else None

    total_epochs = cfg.training["epochs"]
    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()
        if use_amp:
            torch.cuda.reset_peak_memory_stats()

        for sketch, target in train_loader:
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

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(num_batches, 1)
        images_per_sec = (num_batches * cfg.training["batch_size"]) / max(epoch_time, 1e-6)

        sample_every = cfg.training["sample_every"]
        ckpt_every = cfg.training["checkpoint_every"]

        if is_rank0 and (epoch + 1) % sample_every == 0:
            val_sketch, val_target = next(iter(val_loader))
            val_sketch = val_sketch.to(device)
            unwrapped = model.module if hasattr(model, "module") else model
            n = min(4, val_sketch.shape[0])
            save_samples(
                diffusion, unwrapped, val_sketch, epoch + 1,
                folder=cfg.paths["sample_dir"],
                num_samples=n,
                ddim_steps=cfg.training["ddim_sample_steps"],
            )
            if writer is not None:
                with torch.no_grad():
                    gen = diffusion.sample_ddim(unwrapped, val_sketch[:n], val_sketch[:n].shape, ddim_steps=cfg.training["ddim_sample_steps"])
                writer.add_images("samples/sketch", val_sketch[:n] * 0.5 + 0.5, epoch + 1)
                writer.add_images("samples/generated", gen * 0.5 + 0.5, epoch + 1)
                writer.add_images("samples/ground_truth", val_target[:n].to(device) * 0.5 + 0.5, epoch + 1)

        metrics = {
            "loss": avg_loss,
            "epoch": epoch + 1,
            "total_epochs": total_epochs,
            "progress_pct": round(100 * (epoch + 1) / total_epochs, 1),
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_sec": round(epoch_time, 2),
            "images_per_sec": round(images_per_sec, 1),
            "global_step": global_step,
            "gpu_peak_memory_mb": round(_gpu_memory_mb(), 1),
        }
        for k, v in loss_dict.items():
            metrics[f"loss_{k}"] = v

        if is_rank0:
            _log_epoch(metrics, loss_dict)
            if writer is not None:
                _log_tb(writer, epoch + 1, metrics, loss_dict)

        if (epoch + 1) % ckpt_every == 0:
            unwrapped = model.module if hasattr(model, "module") else model
            ckpt_dir = cfg.paths["checkpoint_dir"]
            ckpt_path = os.path.join(ckpt_dir, f"diffusion_epoch{epoch + 1}.pt")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_checkpoint(unwrapped, optimizer, epoch + 1, ckpt_path)
            ray_train.report(metrics, checkpoint=ray_train.Checkpoint.from_directory(ckpt_dir))
        else:
            ray_train.report(metrics)

    if writer is not None:
        writer.close()


def run_training(cfg: Config):
    """Launch Ray distributed training from a loaded Config."""
    dashboard_host = cfg.ray.get("dashboard_host", "0.0.0.0")
    dashboard_port = cfg.ray.get("dashboard_port", 8265)

    ctx = ray.init(
        ignore_reinit_error=True,
        include_dashboard=True,
        dashboard_host=dashboard_host,
        dashboard_port=dashboard_port,
    )

    dashboard_url = getattr(ctx, "dashboard_url", None)
    dash = f"http://{dashboard_url}" if dashboard_url else f"http://{dashboard_host}:{dashboard_port}"
    tb_dir = os.path.abspath(cfg.paths.get("tensorboard_dir", "runs"))

    print(f"\n{'='*60}")
    print(f"  Ray Dashboard:  {dash}")
    print(f"  TensorBoard:    tensorboard --logdir {tb_dir} --bind_all")
    print(f"{'='*60}\n")

    trainer = TorchTrainer(
        train_loop_per_worker=_train_loop,
        train_loop_config=dict(cfg),
        scaling_config=ScalingConfig(
            num_workers=cfg.ray["num_workers"],
            use_gpu=cfg.ray["use_gpu"],
        ),
        run_config=RunConfig(
            storage_path=os.path.abspath(cfg.paths["ray_results_dir"]),
            name="diffusion_sketch_colorization",
            checkpoint_config=CheckpointConfig(num_to_keep=3),
        ),
    )

    result = trainer.fit()
    print(f"\nTraining complete. Best loss: {result.metrics.get('loss', 'N/A')}")
    print(f"Checkpoints: {result.checkpoint}")
    return result
