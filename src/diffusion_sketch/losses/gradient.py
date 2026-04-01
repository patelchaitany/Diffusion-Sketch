"""Gradient loss: L1 between Sobel gradient magnitudes for shading/texture."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    """L_gradient = || ||G(I_real)||_2  -  ||G(I_gen)||_2 ||_1"""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient_magnitude(self, img):
        gx = F.conv2d(img, self.sobel_x, padding=1, groups=3)
        gy = F.conv2d(img, self.sobel_y, padding=1, groups=3)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, real, generated):
        return F.l1_loss(self._gradient_magnitude(generated), self._gradient_magnitude(real))
