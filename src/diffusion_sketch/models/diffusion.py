"""DDPM (Denoising Diffusion Probabilistic Model) scheduler.

Implements the forward (noising) and reverse (denoising) processes
with a linear beta schedule, plus accelerated DDIM sampling.
"""

import torch
import torch.nn as nn


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).float())
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod).float())
        self.register_buffer("sqrt_recip_alphas_cumprod_m1", torch.sqrt(1.0 / alphas_cumprod - 1.0).float())

        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_var.float())
        self.register_buffer("posterior_log_variance", torch.log(posterior_var.clamp(min=1e-20)).float())
        self.register_buffer(
            "posterior_mean_coef1",
            (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).float(),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)).float(),
        )

    def _extract(self, a, t, x_shape):
        return a.gather(-1, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def snr(self, t):
        """Signal-to-noise ratio: α̅_t / (1 - α̅_t)."""
        return self._extract(self.alphas_cumprod, t, t.shape) / \
               self._extract(1.0 - self.alphas_cumprod, t, t.shape).clamp(min=1e-8)

    def min_snr_weight(self, t, gamma=5.0):
        """Min-SNR-γ per-sample loss weight: min(SNR(t), γ) / SNR(t).

        Clamps timestep weighting so high-noise steps don't dominate.
        From "Efficient Diffusion Training via Min-SNR Weighting Strategy"
        (Hang et al., 2023).
        """
        s = self.snr(t)
        return (s.clamp(max=gamma) / s).clamp(min=0.0, max=1.0)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_a = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_1ma = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_a * x_start + sqrt_1ma * noise, noise

    def predict_x0_from_noise(self, x_t, t, noise):
        """Recover x_0 from x_t and predicted noise."""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recip_alphas_cumprod_m1, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Compute posterior q(x_{t-1} | x_t, x_0)."""
        mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = self._extract(self.posterior_variance, t, x_t.shape)
        log_var = self._extract(self.posterior_log_variance, t, x_t.shape)
        return mean, var, log_var

    @torch.no_grad()
    def p_sample(self, model, x_t, t, sketch):
        """Single reverse step."""
        noise_pred = model(x_t, t, sketch)
        x0_pred = self.predict_x0_from_noise(x_t, t, noise_pred).clamp(-1, 1)
        mean, _, log_var = self.q_posterior(x0_pred, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(-1, *((1,) * (len(x_t.shape) - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(self, model, sketch, shape):
        """Full DDPM reverse process."""
        device = sketch.device
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, sketch)
        return x.clamp(-1, 1)

    @torch.no_grad()
    def sample_ddim(self, model, sketch, shape, ddim_steps=50, eta=0.0):
        """Accelerated DDIM sampling."""
        device = sketch.device
        step_size = self.timesteps // ddim_steps
        timesteps = list(reversed(range(0, self.timesteps, step_size)))

        x = torch.randn(shape, device=device)
        for i in range(len(timesteps)):
            t_cur = timesteps[i]
            t_tensor = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            noise_pred = model(x, t_tensor, sketch)
            x0_pred = self.predict_x0_from_noise(x, t_tensor, noise_pred).clamp(-1, 1)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_cur = self.alphas_cumprod[t_cur]
                alpha_next = self.alphas_cumprod[t_next]
                sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_cur) * (1 - alpha_cur / alpha_next))
                dir_xt = torch.sqrt(1 - alpha_next - sigma ** 2) * noise_pred
                x = torch.sqrt(alpha_next) * x0_pred + dir_xt + sigma * torch.randn_like(x)
            else:
                x = x0_pred

        return x.clamp(-1, 1)
