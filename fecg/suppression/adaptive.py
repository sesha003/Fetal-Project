"""Adaptive-filter maternal cancellation (LMS / RLS).

Classic Widrow noise-cancellation: a maternal reference is filtered to predict the
maternal component in each abdominal channel, then subtracted. Because the
reference is uncorrelated with the fetal signal, cancellation does not depend on
temporal separation, so it preserves the fetal complex through coincidence.

Caveat: it needs a good maternal reference. Datasets with a thoracic lead give
one directly; ADFECGDB has none, so here we build a synthetic reference from the
maternal-template reconstruction. That is a pragmatic stand-in -- with a real
chest lead, pass it in as `reference`.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from .base import Suppressor
from .template import TemplateSubtraction


def _lms(reference, desired, order, mu):
    n = len(desired)
    w = np.zeros(order)
    out = np.zeros(n)
    for i in range(order, n):
        x = reference[i - order:i][::-1]
        y = w @ x
        e = desired[i] - y
        w += 2 * mu * e * x / (x @ x + 1e-8)     # normalized LMS
        out[i] = e
    return out


def _rls(reference, desired, order, lam, delta):
    n = len(desired)
    w = np.zeros(order)
    P = np.eye(order) / delta
    out = np.zeros(n)
    for i in range(order, n):
        x = reference[i - order:i][::-1]
        Px = P @ x
        k = Px / (lam + x @ Px)
        e = desired[i] - w @ x
        w += k * e
        P = (P - np.outer(k, Px)) / lam
        out[i] = e
    return out


class AdaptiveSuppressor(Suppressor):
    name = "adaptive"

    def __init__(self, cfg, method: str = "rls", reference: Optional[np.ndarray] = None):
        super().__init__(cfg)
        assert method in ("lms", "rls")
        self.method = method
        self.reference = reference               # (n_samples,) maternal reference, if available

    def _build_reference(self, sig, mat_r):
        """Fallback reference: template reconstruction of the maternal component."""
        residual = TemplateSubtraction(self.cfg).suppress(sig, mat_r)
        maternal_est = sig - residual            # what template subtraction removed
        return maternal_est.mean(axis=1)         # collapse channels to one reference

    def suppress(self, clean: np.ndarray, mat_r: Optional[np.ndarray] = None) -> np.ndarray:
        sig = self._as_2d(clean)
        n, n_ch = sig.shape
        ref = self.reference
        if ref is None:
            if mat_r is None:
                return sig
            ref = self._build_reference(sig, mat_r)
        ref = np.asarray(ref, float)
        out = np.zeros_like(sig)
        for c in range(n_ch):
            if self.method == "lms":
                out[:, c] = _lms(ref, sig[:, c], self.cfg.adaptive_order, self.cfg.lms_mu)
            else:
                out[:, c] = _rls(ref, sig[:, c], self.cfg.adaptive_order,
                                 self.cfg.rls_lambda, self.cfg.rls_delta)
        return out
