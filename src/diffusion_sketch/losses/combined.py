"""Combined loss for diffusion training, aggregating all paper losses."""

import torch.nn as nn
import torch.nn.functional as F

from .laplacian import LaplacianLoss
from .gradient import GradientLoss
from .histogram import HistogramLoss
from .perceptual import PerceptualLoss


class CombinedDiffusionLoss(nn.Module):
    """Auxiliary losses on x0 predictions (L1, Laplacian, Gradient, Histogram).

    The noise MSE is handled separately in the trainer so min-SNR weighting
    can be applied per-sample. This module computes only the auxiliary terms.
    """

    def __init__(
        self,
        lambda_l1=1.0,
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
        self._last_aux = None

    def forward(self, noise, noise_pred, x0, x0_pred, apply_aux=True):
        mse = F.mse_loss(noise_pred, noise)
        loss_dict = {"mse": mse.item()}

        if not apply_aux:
            self._last_aux = None
            return mse, loss_dict

        l1 = F.l1_loss(x0_pred, x0)
        lap = self.laplacian_loss(x0, x0_pred)
        grad = self.gradient_loss(x0, x0_pred)
        hist = self.histogram_loss(x0, x0_pred)

        aux = (
            self.lambda_l1 * l1
            + self.lambda_laplacian * lap
            + self.lambda_gradient * grad
            + self.lambda_histogram * hist
        )
        loss_dict.update({
            "l1": l1.item(),
            "laplacian": lap.item(),
            "gradient": grad.item(),
            "histogram": hist.item(),
        })

        if self.perceptual_loss is not None:
            perc = self.perceptual_loss(x0, x0_pred)
            aux = aux + self.lambda_perceptual * perc
            loss_dict["perceptual"] = perc.item()

        self._last_aux = aux
        total = mse + aux
        loss_dict["total"] = total.item()
        loss_dict["aux"] = aux.item()
        return total, loss_dict
