"""1D U-Net that maps multichannel abdominal ECG -> fetal R-peak probability.

Requires PyTorch (`pip install torch`). Not executed in the build sandbox (no
space for the wheel), but written to standard, runnable PyTorch idioms.

Design: multichannel input so the network can exploit spatial diversity -- the
same property that lets BSS work at coincidence. The output is a per-sample
probability heatmap; peak-picking it yields fetal R-peaks that flow into the
usual FHR/HRV/evaluation code unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, 7, padding=3), nn.BatchNorm1d(c_out), nn.ReLU(),
            nn.Conv1d(c_out, c_out, 7, padding=3), nn.BatchNorm1d(c_out), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet1D(nn.Module):
    """Compact 1D U-Net. in_channels = number of abdominal leads."""
    def __init__(self, in_channels: int = 4, base: int = 16, depth: int = 3):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        c = in_channels
        chans = []
        for d in range(depth):
            oc = base * (2 ** d)
            self.downs.append(ConvBlock(c, oc))
            self.pools.append(nn.MaxPool1d(2))
            chans.append(oc)
            c = oc
        self.bottleneck = ConvBlock(c, c * 2)
        self.ups = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        c = c * 2
        for d in reversed(range(depth)):
            oc = chans[d]
            self.upconvs.append(nn.ConvTranspose1d(c, oc, 2, stride=2))
            self.ups.append(ConvBlock(c, oc))       # c = oc(skip) + oc(up)
            c = oc
        self.head = nn.Conv1d(c, 1, 1)

    def forward(self, x):                            # x: (B, C, L)
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for upconv, up, skip in zip(self.upconvs, self.ups, reversed(skips)):
            x = upconv(x)
            # pad if odd-length mismatch
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                x = nn.functional.pad(x, (0, diff))
            x = up(torch.cat([skip, x], dim=1))
        return self.head(x).squeeze(1)               # (B, L) logits
