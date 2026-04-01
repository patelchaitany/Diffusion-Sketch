"""Laplacian loss: L1 between Laplacian-filtered images for edge preservation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianLoss(nn.Module):
    """L_laplacian = || I_real * L  -  I_gen * L ||_1"""

    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.register_buffer("kernel", kernel)

    def forward(self, real, generated):
        real_lap = F.conv2d(real, self.kernel, padding=1, groups=3)
        gen_lap = F.conv2d(generated, self.kernel, padding=1, groups=3)
        return F.l1_loss(gen_lap, real_lap)
