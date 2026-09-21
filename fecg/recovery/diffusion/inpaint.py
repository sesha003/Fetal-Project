"""DDPM schedule + occlusion-conditioned inpainting. Requires PyTorch.

The inpainting loop is the payoff of the whole reframe. At each reverse-diffusion
step, the *visible* samples are overwritten with the (correctly noised) observation,
so the model only ever generates the *occluded* region, and it generates it
consistently with the visible context (RePaint-style known-region replacement). The
occlusion mask from `recovery.occlusion` is exactly the mask fed in here.
"""
from __future__ import annotations

import torch


class DDPM:
    def __init__(self, model, timesteps: int = 200, beta_start: float = 1e-4,
                 beta_end: float = 2e-2, device: str = "cpu"):
        self.model = model
        self.T = timesteps
        self.device = device
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.acp = torch.cumprod(self.alphas, dim=0)          # alpha-bar

    def q_sample(self, x0, t, noise):
        """Forward diffuse x0 to x_t."""
        ac = self.acp[t][:, None, None]
        return ac.sqrt() * x0 + (1 - ac).sqrt() * noise

    def p_loss(self, x0, cond):
        """Standard noise-prediction training loss."""
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self.model(x_t, t, cond)
        return ((pred - noise) ** 2).mean()

    @torch.no_grad()
    def inpaint(self, observed, mask_known, cond, n_resample: int = 1):
        """
        Reconstruct the occluded region.

        observed    : (B, 1, L) the beat window (visible part meaningful; occluded part ignored)
        mask_known  : (B, 1, L) 1 where the sample is VISIBLE/known, 0 where occluded
        cond        : (B, C, L) conditioning channels (context + occlusion mask)
        Returns the reconstructed window with visible samples preserved.
        """
        x = torch.randn_like(observed)
        for i in reversed(range(self.T)):
            t = torch.full((observed.shape[0],), i, device=self.device, dtype=torch.long)
            for _ in range(n_resample):
                # replace known region with correctly-noised observation
                noise = torch.randn_like(observed)
                x_known = self.q_sample(observed, t, noise)
                x = mask_known * x_known + (1 - mask_known) * x

                eps = self.model(x, t, cond)
                ac = self.acp[i]
                a = self.alphas[i]
                x0 = (x - (1 - ac).sqrt() * eps) / ac.sqrt()
                if i > 0:
                    beta = self.betas[i]
                    z = torch.randn_like(x)
                    x = a.rsqrt() * (x - beta / (1 - ac).sqrt() * eps) + beta.sqrt() * z
                else:
                    x = x0
        return mask_known * observed + (1 - mask_known) * x
