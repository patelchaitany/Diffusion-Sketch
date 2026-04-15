"""Checkpointing and EMA utilities."""

import copy
import os
import torch


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy: ema_param = decay * ema_param + (1-decay) * param.
    Use the EMA weights for sampling/validation — they produce much cleaner outputs
    than the raw (noisy) training weights.
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        self.shadow.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


def save_checkpoint(model, optimizer, epoch, filename, ema=None):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    data = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if ema is not None:
        data["ema_state_dict"] = ema.state_dict()
    torch.save(data, filename)


def load_checkpoint(filename, model, optimizer=None, lr=None, ema=None):
    checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = lr
    if ema is not None and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])
    return checkpoint.get("epoch", 0)
