"""Fetal QRS detection, FHR series, and HRV metrics."""
from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from .config import PipelineConfig
from .preprocess import butter_bandpass


def detect_fetal_qrs(residual: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Detect fetal R-peaks on the maternal-suppressed residual (RMS across channels)."""
    residual = np.atleast_2d(residual)
    if residual.shape[0] < residual.shape[1]:
        residual = residual.T
    band = np.column_stack([
        butter_bandpass(residual[:, c], cfg.fs, cfg.fet_bp_low, cfg.fet_bp_high)
        for c in range(residual.shape[1])
    ])
    combined = np.sqrt(np.mean(band ** 2, axis=1))
    refractory = int((60.0 / cfg.fhr_max_bpm) * cfg.fs)
    thr = np.mean(combined) + 0.5 * np.std(combined)
    peaks, _ = sp_signal.find_peaks(combined, height=thr, distance=refractory)
    return peaks


def compute_fhr(fet_r: np.ndarray, cfg: PipelineConfig):
    """Instantaneous FHR (bpm) at each beat, with RR-based outlier rejection."""
    if len(fet_r) < 2:
        return np.array([]), np.array([])
    rr = np.diff(fet_r) / cfg.fs
    fhr = 60.0 / rr
    valid = (fhr >= cfg.fhr_min_bpm) & (fhr <= cfg.fhr_max_bpm)
    return (fet_r[1:] / cfg.fs)[valid], fhr[valid]


def compute_hrv(fet_r: np.ndarray, cfg: PipelineConfig) -> dict:
    """Basic HRV: mean FHR, SDNN, RMSSD, short-term variability (ms).

    Short-term (beat-to-beat) variability is the clinical payload that overlap
    dropout destroys, so it is worth reporting alongside detection metrics.
    """
    if len(fet_r) < 3:
        return dict(mean_fhr=np.nan, sdnn_ms=np.nan, rmssd_ms=np.nan, stv_ms=np.nan)
    rr = np.diff(fet_r) / cfg.fs * 1000.0        # ms
    good = (rr > 60000.0 / cfg.fhr_max_bpm) & (rr < 60000.0 / cfg.fhr_min_bpm)
    rr = rr[good]
    if len(rr) < 3:
        return dict(mean_fhr=np.nan, sdnn_ms=np.nan, rmssd_ms=np.nan, stv_ms=np.nan)
    return dict(
        mean_fhr=float(60000.0 / np.mean(rr)),
        sdnn_ms=float(np.std(rr)),
        rmssd_ms=float(np.sqrt(np.mean(np.diff(rr) ** 2))),
        stv_ms=float(np.mean(np.abs(np.diff(rr)))),
    )
