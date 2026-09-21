"""Windowed dataset for the DL fetal-detector. Requires PyTorch."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import PipelineConfig
from ..io import Record
from ..preprocess import preprocess


def gaussian_target(length: int, peaks: np.ndarray, fs: float, sigma_s: float = 0.02):
    """Soft R-peak target: sum of Gaussians centered at each fetal R (in samples)."""
    y = np.zeros(length, dtype=np.float32)
    s = sigma_s * fs
    idx = np.arange(length)
    for p in peaks:
        if 0 <= p < length:
            y += np.exp(-0.5 * ((idx - p) / s) ** 2)
    return np.clip(y, 0, 1)


class WindowDataset(Dataset):
    """Slices records into fixed windows of preprocessed multichannel aECG + targets."""
    def __init__(self, records: list[Record], cfg: PipelineConfig,
                 win_s: float = 2.0, stride_s: float = 1.0):
        self.samples = []
        w = int(win_s * cfg.fs)
        stride = int(stride_s * cfg.fs)
        for rec in records:
            if rec.fet_r is None:
                continue
            clean = preprocess(rec.signal, cfg)                 # (n, C)
            n, C = clean.shape
            target = gaussian_target(n, rec.fet_r, cfg.fs)
            for start in range(0, max(1, n - w), stride):
                x = clean[start:start + w].T                    # (C, w)
                if x.shape[1] < w:
                    continue
                y = target[start:start + w]
                # per-window z-norm per channel
                x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
                self.samples.append((x.astype(np.float32), y.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.from_numpy(x), torch.from_numpy(y)
