"""Run the occlusion-aware recovery arm through the shared evaluation harness."""
from __future__ import annotations

import numpy as np

from ..config import PipelineConfig
from ..io import Record
from ..preprocess import detect_maternal_qrs, preprocess
from ..fetal import compute_fhr, compute_hrv
from ..evaluate import evaluate, fhr_error
from .gated_recovery import OcclusionAwareRecovery


def run_recovery_pipeline(record: Record, cfg: PipelineConfig,
                          recovery: OcclusionAwareRecovery | None = None) -> dict:
    """Full recovery pipeline; returns the same artifact/eval shape as run_pipeline,
    plus a breakdown of how many occluded beats were recovered."""
    recovery = recovery or OcclusionAwareRecovery(cfg)
    res = recovery.recover(record)
    fhr_t, fhr = compute_fhr(res.fet_r, cfg)
    hrv = compute_hrv(res.fet_r, cfg)

    out = dict(method=recovery.name, fet_r=res.fet_r, visible_r=res.visible_r,
               recovered_r=res.recovered_r, mask=res.mask, weights=res.weights,
               fhr_t=fhr_t, fhr=fhr, hrv=hrv,
               n_recovered=int(len(res.recovered_r)))

    if record.fet_r is not None:
        # reference maternal R for the coincidence labelling
        clean = preprocess(record.signal, cfg)
        ref_mat = record.mat_r if record.mat_r is not None else detect_maternal_qrs(clean, cfg)
        out["eval"] = evaluate(res.fet_r, record.fet_r, ref_mat, cfg)
        out["fhr_error"] = fhr_error(fhr_t, fhr, record.fet_r, cfg)
    return out
