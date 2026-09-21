"""Preprocessing (bandpass + powerline notch) and Pan-Tompkins QRS detection."""
from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from .config import PipelineConfig


def butter_bandpass(x, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = sp_signal.butter(order, [low / nyq, min(high / nyq, 0.999)], btype="band")
    return sp_signal.filtfilt(b, a, x)


def notch(x, fs, f0, q):
    b, a = sp_signal.iirnotch(f0 / (0.5 * fs), q)
    return sp_signal.filtfilt(b, a, x)


def preprocess(sig: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Bandpass + notch, per channel. Accepts/returns (n_samples, n_channels)."""
    sig = np.atleast_2d(sig)
    if sig.shape[0] < sig.shape[1]:
        sig = sig.T
    out = np.zeros_like(sig, dtype=float)
    for c in range(sig.shape[1]):
        x = butter_bandpass(sig[:, c], cfg.fs, cfg.bp_low, cfg.bp_high)
        out[:, c] = notch(x, cfg.fs, cfg.powerline, cfg.notch_q)
    return out


def pan_tompkins(x, fs, bp_low, bp_high, refractory_s):
    """Pan-Tompkins-style R-peak detection on a single channel -> sample indices."""
    filt = butter_bandpass(x, fs, bp_low, bp_high)
    squared = np.gradient(filt) ** 2
    win = max(1, int(0.150 * fs))
    integ = np.convolve(squared, np.ones(win) / win, mode="same")
    thr = 0.5 * np.mean(integ) + 0.5 * np.median(integ)
    peaks, _ = sp_signal.find_peaks(integ, height=thr, distance=int(refractory_s * fs))
    # snap to local extremum of the bandpassed signal
    r, search = [], int(0.05 * fs)
    for p in peaks:
        a, b = max(0, p - search), min(len(filt), p + search)
        r.append(a + int(np.argmax(np.abs(filt[a:b]))))
    return np.array(sorted(set(r)), dtype=int)


def detect_maternal_qrs(sig: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Detect maternal R-peaks on the channel with strongest maternal QRS energy."""
    sig = np.atleast_2d(sig)
    if sig.shape[0] < sig.shape[1]:
        sig = sig.T
    energies = [np.sum(butter_bandpass(sig[:, c], cfg.fs, cfg.mat_bp_low,
                                       cfg.mat_bp_high) ** 2) for c in range(sig.shape[1])]
    best = int(np.argmax(energies))
    return pan_tompkins(sig[:, best], cfg.fs, cfg.mat_bp_low,
                        cfg.mat_bp_high, cfg.mat_refractory_s)
