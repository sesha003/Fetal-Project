"""1D UNet noise predictor for the fetal-beat diffusion prior. Requires PyTorch.

Not executed in the build sandbox (torch does not fit), but written to standard,
runnable idioms. This is the morphology engine: trained on clean fetal beats, it
learns what a fetal beat *looks like* so the inpainting loop can reconstruct the
waveform hidden under a maternal QRS, conditioned on the visible context.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResBlock1D(nn.Module):
    def __init__(self, c, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c)
        self.conv1 = nn.Conv1d(c, c, 5, padding=2)
        self.temb = nn.Linear(t_dim, c)
        self.norm2 = nn.GroupNorm(8, c)
        self.conv2 = nn.Conv1d(c, c, 5, padding=2)
        self.act = nn.SiLU()

    def forward(self, x, temb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.temb(temb)[..., None]
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class DiffusionUNet1D(nn.Module):
    """Noise predictor eps_theta(x_t, t, cond). `cond_channels` carries the visible
    context + occlusion mask as extra input channels."""
    def __init__(self, in_ch: int = 1, cond_ch: int = 2, base: int = 32, t_dim: int = 128):
        super().__init__()
        self.t_dim = t_dim
        self.temb = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.inp = nn.Conv1d(in_ch + cond_ch, base, 3, padding=1)
        self.b1 = ResBlock1D(base, t_dim)
        self.down = nn.Conv1d(base, base * 2, 4, stride=2, padding=1)
        self.b2 = ResBlock1D(base * 2, t_dim)
        self.up = nn.ConvTranspose1d(base * 2, base, 4, stride=2, padding=1)
        self.b3 = ResBlock1D(base, t_dim)
        self.out = nn.Conv1d(base, in_ch, 3, padding=1)

    def forward(self, x_t, t, cond):
        temb = self.temb(timestep_embedding(t, self.t_dim))
        h = self.inp(torch.cat([x_t, cond], dim=1))
        h = self.b1(h, temb)
        skip = h
        h = self.b2(self.down(h), temb)
        h = self.up(h)
        if h.shape[-1] != skip.shape[-1]:
            h = nn.functional.pad(h, (0, skip.shape[-1] - h.shape[-1]))
        h = self.b3(h + skip, temb)
        return self.out(h)
