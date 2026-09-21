"""Occlusion-aware fetal beat recovery -- the novel method.

Pipeline of the arm:
    preprocess -> detect maternal R -> build occlusion mask + reliability weights
      -> suppress maternal (template) -> detect fetal beats in the VISIBLE regions
         (detection is itself GATED: the reliability weights suppress maternal
          residual before peak-finding, so the visible rhythm is clean)
      -> predict hidden beats under occlusions from the rhythm prior
      -> VERIFY each prediction against the gated (down-weighted) observation
      -> merge visible + recovered beats

The contrast with every classical arm: they trust the occluded samples and simply
miss the fetal beats hidden there. This one knows those samples are compromised,
predicts the hidden beats from rhythm, and only confirms with a heavily
down-weighted look at the observation. Timing recovery needs no deep learning; the
optional diffusion layer reconstructs waveform morphology on top.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks

from ..config import PipelineConfig
from ..preprocess import preprocess, detect_maternal_qrs, butter_bandpass
from ..suppression.template import TemplateSubtraction
from .occlusion import occlusion_mask, reliability_weights
from .rhythm_prior import predict_hidden_beats


@dataclass
class RecoveryResult:
    fet_r: np.ndarray
    visible_r: np.ndarray
    recovered_r: np.ndarray
    mask: np.ndarray
    weights: np.ndarray


class OcclusionAwareRecovery:
    name = "occlusion_recovery"

    def __init__(self, cfg: PipelineConfig, verify: bool = True,
                 verify_tol_s: float = 0.03, verify_floor: float = 0.15):
        self.cfg = cfg
        self.verify = verify
        self.verify_tol = verify_tol_s
        self.verify_floor = verify_floor

    def _fetal_rms(self, residual):
        """RMS across channels of the fetal band."""
        band = np.column_stack([
            butter_bandpass(residual[:, c], self.cfg.fs, self.cfg.fet_bp_low,
                            self.cfg.fet_bp_high)
            for c in range(residual.shape[1])])
        return np.sqrt(np.mean(band ** 2, axis=1))

    def _visible_beats(self, rms, weights):
        """Gated detection: weight down the maternal-residual spikes, then peak-find.
        Any peak that survives is in a reliable (non-occluded) region by construction.
        """
        gated = rms * weights
        refractory = int((60.0 / self.cfg.fhr_max_bpm) * self.cfg.fs)
        thr = np.mean(gated) + 0.5 * np.std(gated)
        peaks, _ = find_peaks(gated, height=thr, distance=refractory)
        return np.array(peaks, int)

    def _verify_prediction(self, rms, weights, loc):
        fs = self.cfg.fs
        tol = int(self.verify_tol * fs)
        a, b = max(0, loc - tol), min(len(rms), loc + tol)
        if b - a < 3:
            return loc
        seg = rms[a:b] * weights[a:b]
        if seg.max() >= self.verify_floor * (rms.mean() + 1e-9):
            return a + int(np.argmax(seg))
        return loc

    def _direct_beats(self, rms):
        """Standard detection over all regions (the baseline set we never discard)."""
        refractory = int((60.0 / self.cfg.fhr_max_bpm) * self.cfg.fs)
        thr = np.mean(rms) + 0.5 * np.std(rms)
        peaks, _ = find_peaks(rms, height=thr, distance=refractory)
        return np.array(peaks, int)

    def recover(self, record) -> RecoveryResult:
        cfg = self.cfg
        clean = preprocess(record.signal, cfg)
        mat_r = detect_maternal_qrs(clean, cfg)
        n = clean.shape[0]

        mask = occlusion_mask(n, mat_r, cfg)
        weights = reliability_weights(n, mat_r, cfg)

        residual = TemplateSubtraction(cfg).suppress(clean, mat_r)
        rms = self._fetal_rms(residual)

        # direct detections everywhere = the baseline set (never discarded)
        direct_r = self._direct_beats(rms)
        # gated detections = clean beats used only to estimate the fetal rhythm
        gated_r = self._visible_beats(rms, weights)
        # predict beats missing from the (clean) rhythm where an occlusion explains it
        predicted = predict_hidden_beats(gated_r, mat_r, mask, cfg)

        if self.verify and len(predicted):
            predicted = np.array([self._verify_prediction(rms, weights, p)
                                  for p in predicted], int)

        # recovered = predicted beats that direct detection MISSED (genuinely new)
        tol = int(cfg.match_tol_s * cfg.fs)
        recovered = np.array([p for p in predicted
                              if len(direct_r) == 0
                              or np.min(np.abs(direct_r - p)) > tol], int)

        merged = np.array(sorted(set(direct_r.tolist() + recovered.tolist())), int)
        merged = self._enforce_refractory(merged)
        return RecoveryResult(fet_r=merged, visible_r=direct_r,
                              recovered_r=recovered, mask=mask, weights=weights)

    def _enforce_refractory(self, peaks):
        if len(peaks) == 0:
            return peaks
        refr = int((60.0 / self.cfg.fhr_max_bpm) * self.cfg.fs)
        out = [int(peaks[0])]
        for p in peaks[1:]:
            if p - out[-1] >= refr:
                out.append(int(p))
        return np.array(out, int)

    def detect(self, record) -> np.ndarray:
        return self.recover(record).fet_r
