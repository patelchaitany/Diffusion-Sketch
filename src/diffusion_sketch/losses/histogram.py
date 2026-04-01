"""Histogram loss: Hellinger distance between soft color histograms for color consistency."""

import torch
import torch.nn as nn


class HistogramLoss(nn.Module):
    """L_histo = (1/sqrt(2)) * || H_gen^{1/2} - H_real^{1/2} ||_2

    Uses differentiable soft histograms via RBF kernel binning.
    """

    def __init__(self, num_bins=64):
        super().__init__()
        self.num_bins = num_bins
        self.sigma = 0.5 / num_bins
        bin_centers = torch.linspace(0, 1, num_bins)
        self.register_buffer("bin_centers", bin_centers)

    def _soft_histogram(self, img):
        B, C, H, W = img.shape
        pixels = img.reshape(B, C, -1).unsqueeze(-1)
        centers = self.bin_centers.reshape(1, 1, 1, -1)
        weights = torch.exp(-0.5 * ((pixels - centers) / self.sigma) ** 2)
        hist = weights.sum(dim=2)
        return hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)

    def forward(self, real, generated):
        real_01 = (real + 1) / 2
        gen_01 = (generated + 1) / 2
        hellinger = (1.0 / 2 ** 0.5) * torch.norm(
            self._soft_histogram(real_01).sqrt() - self._soft_histogram(gen_01).sqrt(),
            p=2, dim=-1,
        )
        return hellinger.mean()
