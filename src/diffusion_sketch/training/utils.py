"""Checkpointing utilities."""

import os
import torch


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
