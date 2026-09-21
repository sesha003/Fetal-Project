"""Evaluation: QRS matching + overlap-stratified metrics + FHR error.

The stratified split (coincident vs non-coincident fetal beats) is the point of
the whole project: aggregate F1 hides exactly the beats that overlap costs you.
"""
from __future__ import annotations

import numpy as np

from .config import PipelineConfig


def match_peaks(detected, reference, tol_samples):
    """Greedy 1-1 nearest matching within tol. Returns TP, FP, FN, used-mask over refs."""
    detected = np.sort(np.asarray(detected))
    reference = np.sort(np.asarray(reference))
    used = np.zeros(len(reference), bool)
    tp = 0
    for d in detected:
        if len(reference) == 0:
            break
        i = int(np.argmin(np.abs(reference - d)))
        if not used[i] and abs(reference[i] - d) <= tol_samples:
            used[i] = True
            tp += 1
    fp = len(detected) - tp
    fn = len(reference) - int(used.sum())
    return tp, fp, fn, used


def prf(tp, fp, fn):
    se = tp / (tp + fn) if (tp + fn) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * se * ppv / (se + ppv) if (se + ppv) else 0.0
    return dict(TP=tp, FP=fp, FN=fn, Se=se, PPV=ppv, F1=f1)


def evaluate(detected_fet_r, ref_fet_r, ref_mat_r, cfg: PipelineConfig) -> dict:
    """Overall + coincident/non-coincident fetal QRS detection metrics."""
    tol = int(cfg.match_tol_s * cfg.fs)
    ref_fet_r = np.sort(np.asarray(ref_fet_r))
    ref_mat_r = np.sort(np.asarray(ref_mat_r)) if ref_mat_r is not None else np.array([])
    coin_win = cfg.coincidence_win_s * cfg.fs

    if len(ref_mat_r) and len(ref_fet_r):
        nearest = np.array([np.min(np.abs(ref_mat_r - r)) for r in ref_fet_r])
        is_coin = nearest <= coin_win
    else:
        is_coin = np.zeros(len(ref_fet_r), bool)

    tp, fp, fn, used = match_peaks(detected_fet_r, ref_fet_r, tol)

    def subset(mask):
        n_ref = int(mask.sum())
        n_hit = int((used & mask).sum())
        return dict(n_ref=n_ref, n_detected=n_hit,
                    Se=(n_hit / n_ref if n_ref else float("nan")))

    return {
        "overall": prf(tp, fp, fn),
        "coincident": subset(is_coin),
        "non_coincident": subset(~is_coin),
        "pct_beats_coincident": float(is_coin.mean()) if len(is_coin) else float("nan"),
    }


def fhr_error(fhr_t, fhr, ref_fet_r, cfg: PipelineConfig) -> dict:
    """FHR MAE/RMSE against a reference FHR resampled to the detected beat times."""
    ref_fet_r = np.sort(np.asarray(ref_fet_r))
    if len(ref_fet_r) < 2 or len(fhr) == 0:
        return dict(mae=np.nan, rmse=np.nan)
    ref_t = ref_fet_r[1:] / cfg.fs
    ref_fhr = 60.0 / (np.diff(ref_fet_r) / cfg.fs)
    ref_interp = np.interp(fhr_t, ref_t, ref_fhr)
    err = fhr - ref_interp
    return dict(mae=float(np.mean(np.abs(err))), rmse=float(np.sqrt(np.mean(err ** 2))))
