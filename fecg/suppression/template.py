"""Adaptive template subtraction (classical temporal baseline)."""
from __future__ import annotations

from typing import Optional
import numpy as np

from .base import Suppressor


class TemplateSubtraction(Suppressor):
    """
    Per-channel median maternal template with a per-beat least-squares gain.

    The affine (gain) fit is the detail that matters at overlap: a fixed template
    leaves a residual proportional to maternal beat-to-beat amplitude drift, and
    that residual is what buries the small fetal peak at a coincident beat.
    """
    name = "template"

    def suppress(self, clean: np.ndarray, mat_r: Optional[np.ndarray] = None) -> np.ndarray:
        sig = self._as_2d(clean)
        if mat_r is None or len(mat_r) < 3:
            return sig
        n, n_ch = sig.shape
        hw = int(self.cfg.template_half_win_s * self.cfg.fs)
        residual = sig.copy()
        for c in range(n_ch):
            beats = [sig[r - hw:r + hw, c] for r in mat_r if r - hw >= 0 and r + hw < n]
            if len(beats) < 3:
                continue
            template = np.median(np.array(beats), axis=0)
            tnorm = float(template @ template)
            if tnorm == 0:
                continue
            for r in mat_r:
                a, b = r - hw, r + hw
                if a < 0 or b >= n:
                    continue
                seg = sig[a:b, c]
                gain = (seg @ template) / tnorm
                residual[a:b, c] = seg - gain * template
        return residual
