"""Checkpointing and sample generation utilities."""

import os
import torch
from torchvision.utils import save_image


def save_checkpoint(model, optimizer, epoch, filename):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, filename)


def load_checkpoint(filename, model, optimizer=None, lr=None):
    checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = lr
    return checkpoint.get("epoch", 0)


@torch.no_grad()
def save_samples(diffusion, model, sketch_batch, epoch, folder, num_samples=4, ddim_steps=50):
    os.makedirs(folder, exist_ok=True)
    model.eval()

    sketch = sketch_batch[:num_samples]
    generated = diffusion.sample_ddim(model, sketch, sketch.shape, ddim_steps=ddim_steps)

    save_image(sketch * 0.5 + 0.5, os.path.join(folder, f"input_epoch{epoch}.png"), nrow=num_samples)
    save_image(generated * 0.5 + 0.5, os.path.join(folder, f"generated_epoch{epoch}.png"), nrow=num_samples)
    model.train()
