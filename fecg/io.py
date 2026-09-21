"""Data loading: real PhysioNet records + a synthetic generator with known truth."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import PipelineConfig


def _fill_nans(sig: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN dropouts per channel (real PhysioNet records can
    contain them); filtfilt propagates a single NaN across the whole output, so
    this must happen before any filtering."""
    sig = sig.copy()
    idx = np.arange(sig.shape[0])
    for c in range(sig.shape[1]):
        col = sig[:, c]
        bad = np.isnan(col)
        if bad.any() and not bad.all():
            col[bad] = np.interp(idx[bad], idx[~bad], col[~bad])
    return sig


@dataclass
class Record:
    """A single multichannel abdominal ECG record with optional ground truth."""
    signal: np.ndarray                      # (n_samples, n_channels)
    fs: float
    name: str = "record"
    mat_r: Optional[np.ndarray] = None      # maternal R-peak samples (if known)
    fet_r: Optional[np.ndarray] = None      # fetal R-peak samples (ground truth)

    def __post_init__(self):
        self.signal = np.atleast_2d(self.signal)
        if self.signal.shape[0] < self.signal.shape[1]:
            self.signal = self.signal.T


# --------------------------------------------------------------------------- #
# Real data loaders (require `wfdb`)
# --------------------------------------------------------------------------- #
def load_adfecgdb_record(record_path: str, name: Optional[str] = None) -> Record:
    """
    Load an ADFECGDB record. The direct fetal *scalp* channel's annotation
    (tried as 'fqrs' -> 'qrs' -> 'atr') is the ground-truth fetal R-peak set.
    """
    import os
    import wfdb
    # PhysioNet ships ADFECGDB as EDF, which wfdb cannot open by bare record name.
    if record_path.endswith(".edf") or os.path.exists(record_path + ".edf"):
        path = record_path if record_path.endswith(".edf") else record_path + ".edf"
        return load_edf_record(path, ann_ext="qrs", name=name)
    rec = wfdb.rdrecord(record_path)
    fet_r = None
    for ext in ("fqrs", "qrs", "atr"):
        try:
            fet_r = wfdb.rdann(record_path, ext).sample
            break
        except Exception:
            continue
    return Record(signal=_fill_nans(rec.p_signal), fs=rec.fs,
                  name=name or record_path.split("/")[-1], fet_r=fet_r)


def load_physionet_record(record_path: str, ann_ext: str = "fqrs",
                          name: Optional[str] = None) -> Record:
    """Generic WFDB loader (CinC 2013, NIFECGDB, ...). Specify the annotation ext."""
    import wfdb
    rec = wfdb.rdrecord(record_path)
    fet_r = None
    try:
        fet_r = wfdb.rdann(record_path, ann_ext).sample
    except Exception:
        pass
    return Record(signal=_fill_nans(rec.p_signal), fs=rec.fs,
                  name=name or record_path.split("/")[-1], fet_r=fet_r)


# --------------------------------------------------------------------------- #
# Synthetic generator (controlled overlap; no download needed)
# --------------------------------------------------------------------------- #
def synthesize_record(cfg: PipelineConfig, duration_s: float = 30.0, n_channels: int = 4,
                      mhr: float = 75.0, fhr: float = 140.0, snr_fetal_db: float = -6.0,
                      overlap_frac: float = 0.15, mat_morph_jitter: float = 0.18,
                      seed: int = 0, name: str = "synthetic") -> Record:
    """
    Build a synthetic multichannel aECG with known maternal & fetal R-peaks and a
    controllable fraction of forced maternal/fetal coincidences. This is the knob
    that lets you study the overlap case directly (mirrors FECGSYN's purpose).
    """
    rng = np.random.default_rng(seed)
    fs = cfg.fs
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    def qrs_train(rate_bpm, amp, width_s, jitter=0.02):
        rr = 60.0 / rate_bpm
        peaks, tp = [], rr
        while tp < duration_s - 0.5:
            peaks.append(int(tp * fs))
            tp += rr * (1 + jitter * rng.standard_normal())
        return np.array(peaks)

    def render(peaks, amp, width_s, morph_jitter=0.0):
        """Render a QRS train. `morph_jitter` adds per-beat *shape* variation (not
        just amplitude) via an asymmetry component that per-beat gain fitting cannot
        absorb -- this is what makes template subtraction leave a residual, and thus
        what makes maternal/fetal overlap actually cause fetal dropout, as in real data.
        """
        sig = np.zeros(n)
        w = int(width_s * fs)
        tt = np.linspace(-3, 3, 2 * w)
        shape = (1 - tt ** 2) * np.exp(-tt ** 2 / 2)       # Mexican-hat QRS proxy
        dshape = np.gradient(shape)                        # asymmetry / timing component
        dshape /= (np.linalg.norm(dshape) + 1e-9)
        dshape *= np.linalg.norm(shape)
        for p in peaks:
            a, b = p - w, p + w
            if a >= 0 and b < n:
                eps = morph_jitter * rng.standard_normal()
                gain = 1.0 + 0.1 * rng.standard_normal()   # per-beat amplitude
                sig[a:b] += amp * (gain * shape + eps * dshape)
        return sig

    mat_r = qrs_train(mhr, 1.0, 0.05)
    fet_r = qrs_train(fhr, 1.0, 0.02)
    fet_amp = 10 ** (snr_fetal_db / 20.0)

    # Engineer overlaps by moving a fraction of MATERNAL beats onto fetal beats.
    # Crucially the fetal beats stay at their rhythm-correct positions -- that is how
    # real coincidence works (a fetal beat that happens to align with a maternal one),
    # and it is what any rhythm-based recovery must be tested against. Moving the fetal
    # beat instead would destroy the very rhythm the method relies on.
    if overlap_frac > 0 and len(fet_r) and len(mat_r):
        k = max(1, int(overlap_frac * len(fet_r)))
        chosen = rng.choice(fet_r, size=min(k, len(fet_r)), replace=False)
        for cf in chosen:
            mi = int(np.argmin(np.abs(mat_r - cf)))
            mat_r[mi] = cf                      # maternal QRS now hides this fetal beat
        mat_r = np.unique(mat_r)

    maternal = render(mat_r, 1.0, 0.05, morph_jitter=mat_morph_jitter)
    fetal = render(fet_r, fet_amp, 0.02)

    sig = np.zeros((n, n_channels))
    for c in range(n_channels):
        m_gain = 0.6 + 0.8 * rng.random()
        f_gain = 0.5 + 1.0 * rng.random()
        baseline = 0.3 * np.sin(2 * np.pi * 0.3 * t + rng.random() * 6)
        powerline = 0.05 * np.sin(2 * np.pi * cfg.powerline * t)
        noise = 0.05 * rng.standard_normal(n)
        sig[:, c] = m_gain * maternal + f_gain * fetal + baseline + powerline + noise

    return Record(signal=sig, fs=fs, name=name,
                  mat_r=np.sort(mat_r), fet_r=np.sort(fet_r))


def synthetic_cohort(cfg: PipelineConfig, n_records: int = 5, **kw) -> list[Record]:
    """A small cohort with varied HR/SNR/overlap for leave-one-record-out testing."""
    rng = np.random.default_rng(42)
    recs = []
    for i in range(n_records):
        recs.append(synthesize_record(
            cfg, duration_s=kw.get("duration_s", 30.0),
            mhr=float(rng.uniform(65, 90)),
            fhr=float(rng.uniform(120, 160)),
            snr_fetal_db=float(rng.uniform(-9, -3)),
            overlap_frac=float(rng.uniform(0.10, 0.25)),
            seed=i, name=f"synth_{i:02d}"))
    return recs


# --------------------------------------------------------------------------- #
# EDF loader (ADFECGDB / NIFECGDB)
# --------------------------------------------------------------------------- #
def read_edf_signals(path: str):
    """
    Minimal EDF/EDF+ reader returning ``(labels, signals, fs_per_label)``.

    Why not `wfdb`: these records carry an 'EDF Annotations' channel sampled at
    100 Hz alongside 1000 Hz ECG channels, and wfdb's converter resamples every
    channel down to the slowest one -- silently turning a 1000 Hz record into a
    100 Hz one, which destroys QRS timing at the 50 ms matching tolerance.
    """
    with open(path, "rb") as f:
        hdr = f.read(256)
        n_records = int(hdr[236:244])
        record_dur = float(hdr[244:252])
        ns = int(hdr[252:256])
        labels = [f.read(16).decode("latin-1").strip() for _ in range(ns)]
        f.read(80 * ns)                                   # transducer
        f.read(8 * ns)                                    # physical dimension
        p_min = np.array([float(f.read(8)) for _ in range(ns)])
        p_max = np.array([float(f.read(8)) for _ in range(ns)])
        d_min = np.array([float(f.read(8)) for _ in range(ns)])
        d_max = np.array([float(f.read(8)) for _ in range(ns)])
        f.read(80 * ns)                                   # prefiltering
        spr = np.array([int(f.read(8)) for _ in range(ns)])
        f.read(32 * ns)                                   # reserved
        raw = np.fromfile(f, dtype="<i2")

    per_record = int(spr.sum())
    raw = raw[: n_records * per_record].reshape(n_records, per_record)
    offsets = np.concatenate([[0], np.cumsum(spr)])
    gain = (p_max - p_min) / np.where((d_max - d_min) == 0, 1, d_max - d_min)
    base = p_min - gain * d_min

    signals, fs_per_label = {}, {}
    for i, lab in enumerate(labels):
        col = raw[:, offsets[i]:offsets[i + 1]].reshape(-1).astype(float)
        signals[lab] = col * gain[i] + base[i]
        fs_per_label[lab] = spr[i] / record_dur
    return labels, signals, fs_per_label


def load_edf_record(edf_path: str, ann_ext: str = "qrs",
                    name: Optional[str] = None,
                    exclude: tuple = ("EDF Annotations",)) -> Record:
    """Load an EDF record (ADFECGDB, NIFECGDB) + its WFDB annotation file.

    All ECG channels are kept, including a fetal scalp channel when present
    (ADFECGDB 'Direct_1'). That channel is the *ground truth source* -- exclude it
    from anything fed to a detector, or the evaluation is circular.
    """
    import wfdb
    labels, signals, fs_map = read_edf_signals(edf_path)
    keep = [l for l in labels if l not in exclude and fs_map[l] == max(fs_map.values())]
    sig = np.column_stack([signals[l] for l in keep])
    fs = float(fs_map[keep[0]])

    fet_r = None
    try:
        fet_r = wfdb.rdann(edf_path, ann_ext).sample
    except Exception:
        pass

    rec = Record(signal=_fill_nans(sig), fs=fs,
                 name=name or edf_path.split("/")[-1].replace(".edf", ""),
                 fet_r=fet_r)
    rec.channel_names = keep
    return rec
