"""Train the fetal-beat diffusion prior on clean beats. Requires PyTorch.

Clean fetal beats come from the direct scalp channel in ADFECGDB (ground truth) or
from synthetic fetal beats. Each training window carries a conditioning stack of
(visible-context, occlusion-mask); at train time the mask is random so the model
learns to fill arbitrary gaps, at inference the mask is the real maternal occlusion.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .model import DiffusionUNet1D
from .inpaint import DDPM


class FetalBeatDataset(Dataset):
    """Windows of clean fetal beats + random occlusion masks for self-supervision."""
    def __init__(self, beats: np.ndarray, mask_frac_range=(0.1, 0.4)):
        # beats: (N, L) z-normed clean fetal-beat windows
        self.beats = beats.astype(np.float32)
        self.mask_frac_range = mask_frac_range

    def __len__(self):
        return len(self.beats)

    def __getitem__(self, i):
        x0 = self.beats[i][None]                        # (1, L)
        L = x0.shape[-1]
        frac = np.random.uniform(*self.mask_frac_range)
        w = int(frac * L)
        start = np.random.randint(0, max(1, L - w))
        known = np.ones((1, L), np.float32)
        known[:, start:start + w] = 0.0                 # 0 = occluded
        context = x0 * known                            # visible context
        cond = np.concatenate([context, known], axis=0)  # (2, L)
        return torch.from_numpy(x0), torch.from_numpy(cond)


def train_diffusion(beats: np.ndarray, epochs=50, batch_size=64, lr=2e-4,
                    timesteps=200, device="cpu"):
    ds = FetalBeatDataset(beats)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = DiffusionUNet1D(in_ch=1, cond_ch=2).to(device)
    ddpm = DDPM(model, timesteps=timesteps, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total = 0.0
        for x0, cond in dl:
            x0, cond = x0.to(device), cond.to(device)
            opt.zero_grad()
            loss = ddpm.p_loss(x0, cond)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {ep+1}/{epochs}  loss={total/max(1,len(dl)):.4f}")
    return ddpm
