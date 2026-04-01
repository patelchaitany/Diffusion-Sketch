"""Ray Train-based distributed training loop."""

import logging
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import ray
from ray import train as ray_train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.util.metrics import Counter, Gauge

logging.getLogger("ray.train").setLevel(logging.WARNING)
logging.getLogger("ray._private").setLevel(logging.ERROR)

from diffusion_sketch.config import Config
from diffusion_sketch.data import SketchColorDataset
from diffusion_sketch.models import ConditionalUNet, GaussianDiffusion
from diffusion_sketch.losses import CombinedDiffusionLoss
from .utils import save_checkpoint, save_samples

# Custom Prometheus gauges — visible at http://localhost:8080 and queryable in Prometheus
_gauge_loss = Gauge("diffusion_loss_total", description="Total combined loss")
_gauge_mse = Gauge("diffusion_loss_mse", description="Noise prediction MSE loss")
_gauge_l1 = Gauge("diffusion_loss_l1", description="L1 reconstruction loss")
_gauge_laplacian = Gauge("diffusion_loss_laplacian", description="Laplacian edge loss")
_gauge_gradient = Gauge("diffusion_loss_gradient", description="Gradient shading loss")
_gauge_histogram = Gauge("diffusion_loss_histogram", description="Histogram color loss")
_gauge_epoch = Gauge("diffusion_epoch", description="Current training epoch")
_gauge_lr = Gauge("diffusion_learning_rate", description="Current learning rate")
_gauge_gpu_mem = Gauge("diffusion_gpu_peak_memory_mb", description="Peak GPU memory in MB")
_gauge_img_per_sec = Gauge("diffusion_images_per_sec", description="Training throughput")
_counter_steps = Counter("diffusion_global_steps", description="Total optimizer steps")


def _gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def _export_prometheus_metrics(metrics, loss_dict, steps_this_epoch):
    """Push current values to Ray's Prometheus-compatible gauges."""
    _gauge_loss.set(metrics["loss"])
    _gauge_epoch.set(metrics["epoch"])
    _gauge_lr.set(metrics["lr"])
    _gauge_gpu_mem.set(metrics["gpu_peak_memory_mb"])
    _gauge_img_per_sec.set(metrics["images_per_sec"])
    _counter_steps.inc(steps_this_epoch)

    _gauge_mse.set(loss_dict.get("mse", 0))
    _gauge_l1.set(loss_dict.get("l1", 0))
    _gauge_laplacian.set(loss_dict.get("laplacian", 0))
    _gauge_gradient.set(loss_dict.get("gradient", 0))
    _gauge_histogram.set(loss_dict.get("histogram", 0))


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

        is_rank0 = ray_train.get_context().get_world_rank() == 0
        sample_every = cfg.training["sample_every"]
        ckpt_every = cfg.training["checkpoint_every"]

        if is_rank0 and (epoch + 1) % sample_every == 0:
            val_sketch, _ = next(iter(val_loader))
            unwrapped = model.module if hasattr(model, "module") else model
            save_samples(
                diffusion, unwrapped, val_sketch.to(device), epoch + 1,
                folder=cfg.paths["sample_dir"],
                num_samples=min(4, val_sketch.shape[0]),
                ddim_steps=cfg.training["ddim_sample_steps"],
            )

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
            _export_prometheus_metrics(metrics, loss_dict, num_batches)

        if (epoch + 1) % ckpt_every == 0:
            unwrapped = model.module if hasattr(model, "module") else model
            ckpt_dir = cfg.paths["checkpoint_dir"]
            ckpt_path = os.path.join(ckpt_dir, f"diffusion_epoch{epoch + 1}.pt")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_checkpoint(unwrapped, optimizer, epoch + 1, ckpt_path)
            ray_train.report(metrics, checkpoint=ray_train.Checkpoint.from_directory(ckpt_dir))
        else:
            ray_train.report(metrics)


def run_training(cfg: Config):
    """Launch Ray distributed training from a loaded Config."""
    dashboard_host = cfg.ray.get("dashboard_host", "0.0.0.0")
    dashboard_port = cfg.ray.get("dashboard_port", 8265)
    metrics_port = cfg.ray.get("metrics_export_port", 8080)

    os.environ.setdefault("RAY_METRICS_EXPORT_PORT", str(metrics_port))

    ctx = ray.init(
        ignore_reinit_error=True,
        include_dashboard=True,
        dashboard_host=dashboard_host,
        dashboard_port=dashboard_port,
    )

    dashboard_url = getattr(ctx, "dashboard_url", None)
    if dashboard_url:
        dash_host = dashboard_url.split(":")[0]
        dash = f"http://{dashboard_url}"
    else:
        dash_host = dashboard_host
        dash = f"http://{dashboard_host}:{dashboard_port}"

    print(f"\n{'='*60}")
    print(f"  Ray Dashboard:   {dash}")
    print(f"  Metrics (raw):   http://{dash_host}:{metrics_port}")
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
