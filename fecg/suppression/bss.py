"""Blind source separation (FastICA) maternal suppression.

BSS separates sources on a *spatial* axis, so it can pull maternal and fetal
apart even at exact temporal coincidence -- the core of the overlap hypothesis.
Needs >= 2 channels; on single-channel data it falls back to template subtraction.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.decomposition import FastICA

from .base import Suppressor
from .template import TemplateSubtraction


class ICASuppressor(Suppressor):
    name = "bss_ica"

    def __init__(self, cfg, seed: int = 0, pre_template: bool = True):
        super().__init__(cfg)
        self.seed = seed
        self.pre_template = pre_template            # remove bulk maternal first, then unmix

    def _dominant_rate_bpm(self, x):
        x = (x - x.mean()) / (x.std() + 1e-9)
        ac = np.correlate(x, x, mode="full")[len(x) - 1:]
        lo, hi = int(self.cfg.fs * 0.3), int(self.cfg.fs * 1.2)   # ~50-200 bpm
        if hi >= len(ac):
            return 0.0
        lag = lo + int(np.argmax(ac[lo:hi]))
        return self.cfg.fs / lag * 60.0

    def suppress(self, clean: np.ndarray, mat_r: Optional[np.ndarray] = None) -> np.ndarray:
        sig = self._as_2d(clean)
        base = sig
        if self.pre_template and mat_r is not None:
            base = TemplateSubtraction(self.cfg).suppress(sig, mat_r)
        n, n_ch = base.shape
        if n_ch < 2:
            return base                              # ICA needs spatial diversity

        ica = FastICA(n_components=n_ch, random_state=self.seed,
                      max_iter=600, whiten="unit-variance")
        try:
            S = ica.fit_transform(base)
        except Exception:
            return base
        # slowest-rate component == maternal; zero and re-mix
        rates = [self._dominant_rate_bpm(S[:, i]) for i in range(n_ch)]
        S[:, int(np.argmin(rates))] = 0.0
        return ica.inverse_transform(S)
