"""Kalman-based maternal suppression (beat-wise adaptive template tracker).

NOTE ON SCOPE. A faithful implementation of the Sameni EKF/EKS uses a McSharry
dynamic ECG model (Gaussian-kernel phase model) as the state. That is powerful but
heavy and easy to get subtly wrong. This arm implements a lighter, defensible
alternative: the maternal QRS *template* is treated as a slowly time-varying state,
and a Kalman filter tracks it across beats. Because the template adapts to
beat-to-beat maternal morphology drift, its prediction of the maternal contribution
at a coincident beat is better than a fixed median template -- which is exactly
where fixed template subtraction leaves a fetal-swamping residual.

For the full EKS, swap this class for a McSharry-model implementation (e.g. Sameni's
OSET). The `Suppressor` interface stays identical, so nothing else changes.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from .base import Suppressor


class KalmanTemplateSuppressor(Suppressor):
    name = "kalman"

    def __init__(self, cfg, q: float = 1e-4, r: float = 1e-1):
        super().__init__(cfg)
        self.q = q          # process variance: how fast maternal morphology may drift
        self.r = r          # measurement variance: fetal + noise contamination level

    def suppress(self, clean: np.ndarray, mat_r: Optional[np.ndarray] = None) -> np.ndarray:
        sig = self._as_2d(clean)
        if mat_r is None or len(mat_r) < 3:
            return sig
        n, n_ch = sig.shape
        hw = int(self.cfg.template_half_win_s * self.cfg.fs)
        L = 2 * hw
        residual = sig.copy()

        for c in range(n_ch):
            beats = [(r, sig[r - hw:r + hw, c]) for r in mat_r if r - hw >= 0 and r + hw < n]
            if len(beats) < 3:
                continue
            # state x = template (L,), diagonal covariance p (L,)
            x = np.median(np.array([b for _, b in beats]), axis=0)
            p = np.ones(L)
            for r, seg in beats:
                # predict (random-walk state model)
                p = p + self.q
                # update with this beat as a noisy observation of the maternal template
                k = p / (p + self.r)                 # per-sample Kalman gain
                x = x + k * (seg - x)
                p = (1 - k) * p
                # subtract the current tracked maternal estimate
                residual[r - hw:r + hw, c] = seg - x
        return residual
