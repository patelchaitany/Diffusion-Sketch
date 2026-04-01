"""Combined loss for diffusion training, aggregating all paper losses."""

import torch.nn as nn
import torch.nn.functional as F

from .laplacian import LaplacianLoss
from .gradient import GradientLoss
from .histogram import HistogramLoss
from .perceptual import PerceptualLoss


class CombinedDiffusionLoss(nn.Module):
    """total = MSE(noise, noise_pred)
            + lambda_l1        * L1(x0, x0_pred)
            + lambda_laplacian * Laplacian(x0, x0_pred)
            + lambda_gradient  * Gradient(x0, x0_pred)
            + lambda_histogram * Histogram(x0, x0_pred)
           [+ lambda_perceptual * Perceptual(x0, x0_pred)]
    """

    def __init__(
        self,
        lambda_l1=100.0,
        lambda_laplacian=0.5,
        lambda_gradient=0.5,
        lambda_histogram=0.1,
        lambda_perceptual=0.0,
        use_perceptual=False,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_laplacian = lambda_laplacian
        self.lambda_gradient = lambda_gradient
        self.lambda_histogram = lambda_histogram
        self.lambda_perceptual = lambda_perceptual

        self.laplacian_loss = LaplacianLoss()
        self.gradient_loss = GradientLoss()
        self.histogram_loss = HistogramLoss()
        self.perceptual_loss = PerceptualLoss() if use_perceptual else None

    def forward(self, noise, noise_pred, x0, x0_pred, apply_aux=True):
        mse = F.mse_loss(noise_pred, noise)
        loss_dict = {"mse": mse.item()}

        if not apply_aux:
            return mse, loss_dict

        l1 = F.l1_loss(x0_pred, x0)
        lap = self.laplacian_loss(x0, x0_pred)
        grad = self.gradient_loss(x0, x0_pred)
        hist = self.histogram_loss(x0, x0_pred)

        total = (
            mse
            + self.lambda_l1 * l1
            + self.lambda_laplacian * lap
            + self.lambda_gradient * grad
            + self.lambda_histogram * hist
        )
        loss_dict.update({"l1": l1.item(), "laplacian": lap.item(), "gradient": grad.item(), "histogram": hist.item()})

        if self.perceptual_loss is not None:
            perc = self.perceptual_loss(x0, x0_pred)
            total += self.lambda_perceptual * perc
            loss_dict["perceptual"] = perc.item()

        loss_dict["total"] = total.item()
        return total, loss_dict
