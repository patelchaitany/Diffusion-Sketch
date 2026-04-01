"""VGG16 perceptual loss using feature maps at layers 4, 9, 16, 23."""

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """L_perceptual = sum_l || VGG_l(real) - VGG_l(generated) ||_2"""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = vgg.features
        self.slice1 = nn.Sequential(*features[:4])
        self.slice2 = nn.Sequential(*features[4:9])
        self.slice3 = nn.Sequential(*features[9:16])
        self.slice4 = nn.Sequential(*features[16:23])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, real, generated):
        loss = 0.0
        x_r, x_g = real, generated
        for s in [self.slice1, self.slice2, self.slice3, self.slice4]:
            x_r, x_g = s(x_r), s(x_g)
            loss += F.mse_loss(x_g, x_r)
        return loss
