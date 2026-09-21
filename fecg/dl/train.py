"""Training loop + inference wrapper for the DL fetal detector. Requires PyTorch.

The inference wrapper exposes a `detect(record)` that returns fetal R-peaks, so the
DL arm evaluates through the exact same harness as the classical arms. (It is a
detector rather than a maternal `Suppressor`, so it plugs in at the fetal-detection
stage instead of the suppression stage.)
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.signal import find_peaks

from ..config import PipelineConfig
from ..io import Record
from ..preprocess import preprocess
from .model import UNet1D
from .dataset import WindowDataset


def train_model(train_records, cfg: PipelineConfig, in_channels=4,
                win_s=2.0, epochs=20, batch_size=16, lr=1e-3, device="cpu",
                return_history=False):
    ds = WindowDataset(train_records, cfg, win_s=win_s)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = UNet1D(in_channels=in_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    history = []
    for ep in range(epochs):
        total = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / max(1, len(dl))
        history.append(avg)
        print(f"epoch {ep + 1}/{epochs}  loss={avg:.4f}", flush=True)
    return (model, history) if return_history else model


class DLDetector:
    """Wrap a trained UNet1D so it detects fetal R-peaks over a full record."""
    name = "dl_unet"

    def __init__(self, model, cfg: PipelineConfig, win_s=2.0, device="cpu"):
        self.model = model.eval()
        self.cfg = cfg
        self.win = int(win_s * cfg.fs)
        self.device = device

    @torch.no_grad()
    def probability(self, record: Record) -> np.ndarray:
        """Per-sample fetal R-peak probability over the full record."""
        clean = preprocess(record.signal, self.cfg)             # (n, C)
        n, C = clean.shape
        prob = np.zeros(n)
        counts = np.zeros(n)
        for start in range(0, max(1, n - self.win), self.win // 2):
            seg = clean[start:start + self.win].T
            if seg.shape[1] < self.win:
                continue
            seg = (seg - seg.mean(1, keepdims=True)) / (seg.std(1, keepdims=True) + 1e-8)
            x = torch.from_numpy(seg.astype(np.float32))[None].to(self.device)
            p = torch.sigmoid(self.model(x)).cpu().numpy()[0]
            prob[start:start + self.win] += p
            counts[start:start + self.win] += 1
        return prob / np.maximum(counts, 1)

    def detect(self, record: Record, threshold: float = 0.3) -> np.ndarray:
        prob = self.probability(record)
        refractory = int((60.0 / self.cfg.fhr_max_bpm) * self.cfg.fs)
        peaks, _ = find_peaks(prob, height=threshold, distance=refractory)
        return peaks
