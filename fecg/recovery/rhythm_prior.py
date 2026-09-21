"""Rhythm prior: predict where fetal beats fall inside occluded regions.

The fetal heart is a quasi-periodic oscillator whose rhythm is independent of the
mother's. So the visible beats around an occlusion tightly constrain the hidden one.
This module estimates a robust baseline fetal RR, finds gaps in the visible-beat
train that are a near-integer multiple of that RR (one or more beats went missing),
and predicts the hidden beats' phase locations. A prediction is kept only when an
occlusion sits near it -- i.e. a maternal QRS actually explains the miss. That
discipline is what stops the method inventing beats and wrecking precision.
"""
from __future__ import annotations

import numpy as np

from ..config import PipelineConfig


def robust_base_rr(visible_r: np.ndarray) -> float:
    """Baseline fetal RR (samples), robust to the gaps left by occluded beats.

    Uses the short intervals only: the median of intervals at or below the overall
    median, so double-length gaps (missing beats) don't inflate the estimate.
    """
    iv = np.diff(np.sort(visible_r)).astype(float)
    if len(iv) == 0:
        return 0.0
    m = np.median(iv)
    short = iv[iv <= m + 1e-9]
    return float(np.median(short)) if len(short) else float(m)


def predict_hidden_beats(visible_r: np.ndarray, mat_r: np.ndarray, mask: np.ndarray,
                         cfg: PipelineConfig, snap_tol_s: float = 0.08,
                         max_order: int = 2) -> np.ndarray:
    """
    Predict occluded fetal R-peak locations from the visible-beat rhythm.

    For each interval between consecutive visible beats, k = gap / base_RR estimates
    how many RR periods it spans; round(k)-1 beats are inferred missing and placed at
    phase-uniform positions. A predicted beat is accepted only if an occlusion lies
    within `snap_tol_s` of it (a maternal QRS that would hide it); when a maternal R
    is that close, the beat is snapped onto it, since that is where the occluded fetal
    beat physically sits. Gaps implying more than `max_order` missing beats are
    treated cautiously (their phase is least certain).
    """
    visible_r = np.sort(np.asarray(visible_r, int))
    mat_r = np.sort(np.asarray(mat_r, int))
    if len(visible_r) < 3:
        return np.array([], int)

    base = robust_base_rr(visible_r)
    if base <= 0:
        return np.array([], int)
    n = len(mask)
    tol = snap_tol_s * cfg.fs
    predicted = []

    for i in range(len(visible_r) - 1):
        a, b = visible_r[i], visible_r[i + 1]
        gap = b - a
        k = gap / base
        n_missing = int(round(k)) - 1
        if n_missing < 1 or n_missing > max_order:
            continue
        for j in range(1, n_missing + 1):
            loc = int(round(a + j * gap / (n_missing + 1)))
            if not (0 <= loc < n):
                continue
            if len(mat_r):
                nearest = mat_r[int(np.argmin(np.abs(mat_r - loc)))]
                if abs(nearest - loc) <= tol:
                    predicted.append(int(nearest))     # snap to the occluding QRS
                    continue
            if mask[loc]:
                predicted.append(loc)
    return np.array(sorted(set(predicted)), int)
