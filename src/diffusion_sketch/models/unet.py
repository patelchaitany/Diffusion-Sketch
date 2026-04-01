"""Conditional U-Net for DDPM-based sketch colorization.

The network takes (noisy_color_image, sketch, timestep) and predicts noise.
Architecture follows the standard diffusion U-Net with:
  - Sinusoidal time embeddings
  - Group-norm + SiLU activations
  - Self-attention at low resolutions
  - Skip connections between encoder and decoder
  - Sketch condition concatenated channel-wise with the noisy input
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class ConditionalUNet(nn.Module):
    """U-Net conditioned on sketch input via channel concatenation.

    Args:
        in_channels: channels of the noisy image (3 for RGB).
        cond_channels: channels of the condition/sketch image (3 for RGB).
        base_channels: base feature dimension.
        channel_mults: multiplier for each resolution level.
        attention_resolutions: which downsampling levels get self-attention (0-indexed).
        num_res_blocks: residual blocks per level.
        dropout: dropout rate.
    """

    def __init__(
        self,
        in_channels=3,
        cond_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        attention_resolutions=(3,),
        num_res_blocks=2,
        dropout=0.1,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        time_dim = base_channels * 4

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(in_channels + cond_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        channels = [base_channels]
        ch = base_channels

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            block = nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(ResBlock(ch, out_ch, time_dim, dropout))
                ch = out_ch
                if level in attention_resolutions:
                    block.append(SelfAttention(ch))
            self.down_blocks.append(block)
            channels.append(ch)
            if level < len(channel_mults) - 1:
                self.down_samples.append(Downsample(ch))
            else:
                self.down_samples.append(nn.Identity())

        self.mid_block1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid_block2 = ResBlock(ch, ch, time_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            block = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop() if i == 0 else 0
                block.append(ResBlock(ch + skip_ch, out_ch, time_dim, dropout))
                ch = out_ch
                if level in attention_resolutions:
                    block.append(SelfAttention(ch))
            self.up_blocks.append(block)
            if level > 0:
                self.up_samples.append(Upsample(ch))
            else:
                self.up_samples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def _run_layer(self, layer, h, t_emb):
        if isinstance(layer, ResBlock):
            return layer(h, t_emb)
        return layer(h)

    def _ckpt(self, layer, h, t_emb):
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._run_layer, layer, h, t_emb, use_reentrant=False)
        return self._run_layer(layer, h, t_emb)

    def forward(self, x_noisy, t, sketch):
        t_emb = self.time_embed(t)
        h = self.input_conv(torch.cat([x_noisy, sketch], dim=1))

        skips = [h]
        for block, down in zip(self.down_blocks, self.down_samples):
            for layer in block:
                h = self._ckpt(layer, h, t_emb)
            if not isinstance(down, nn.Identity):
                skips.append(h)
                h = down(h)

        h = self._ckpt(self.mid_block1, h, t_emb)
        h = self.mid_attn(h)
        h = self._ckpt(self.mid_block2, h, t_emb)

        for block, up in zip(self.up_blocks, self.up_samples):
            first = True
            for layer in block:
                if isinstance(layer, ResBlock):
                    if first and skips:
                        h = torch.cat([h, skips.pop()], dim=1)
                        first = False
                    h = self._ckpt(layer, h, t_emb)
                else:
                    h = layer(h)
            if not isinstance(up, nn.Identity):
                h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))
