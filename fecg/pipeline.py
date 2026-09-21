"""Pipeline orchestration + leave-one-record-out cross-validation."""
from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from .config import PipelineConfig
from .io import Record
from .preprocess import preprocess, detect_maternal_qrs
from .fetal import detect_fetal_qrs, compute_fhr, compute_hrv
from .evaluate import evaluate, fhr_error
from .suppression.base import Suppressor


def run_pipeline(record: Record, suppressor: Suppressor, cfg: PipelineConfig) -> dict:
    """preprocess -> maternal detect -> suppress -> fetal detect -> FHR -> evaluate."""
    clean = preprocess(record.signal, cfg)
    mat_r = detect_maternal_qrs(clean, cfg)
    residual = suppressor.suppress(clean, mat_r)
    fet_r = detect_fetal_qrs(residual, cfg)
    fhr_t, fhr = compute_fhr(fet_r, cfg)
    hrv = compute_hrv(fet_r, cfg)

    out = dict(method=suppressor.name, clean=clean, mat_r=mat_r, residual=residual,
               fet_r=fet_r, fhr_t=fhr_t, fhr=fhr, hrv=hrv)
    if record.fet_r is not None:
        ref_mat = record.mat_r if record.mat_r is not None else mat_r
        out["eval"] = evaluate(fet_r, record.fet_r, ref_mat, cfg)
        out["fhr_error"] = fhr_error(fhr_t, fhr, record.fet_r, cfg)
    return out


def leave_one_record_out(records: list[Record],
                         suppressor_factory: Callable[[PipelineConfig], Suppressor],
                         cfg: PipelineConfig) -> dict:
    """
    Run a method arm over a cohort, one record at a time, and aggregate.

    For unsupervised arms (template/BSS/adaptive/Kalman) there is nothing to fit,
    so this is a per-record evaluation loop. The same signature accepts a factory
    that trains on the other records first (for the DL arm), so the harness is
    identical either way.
    """
    per_record = []
    for held in records:
        supp = suppressor_factory(cfg)
        res = run_pipeline(held, supp, cfg)
        if "eval" in res:
            per_record.append({
                "record": held.name,
                "overall": res["eval"]["overall"],
                "coincident_Se": res["eval"]["coincident"]["Se"],
                "non_coincident_Se": res["eval"]["non_coincident"]["Se"],
                "pct_coincident": res["eval"]["pct_beats_coincident"],
                "fhr_mae": res.get("fhr_error", {}).get("mae", np.nan),
            })

    def _mean(key, sub=None):
        vals = [(r[key][sub] if sub else r[key]) for r in per_record]
        vals = [v for v in vals if v == v]        # drop NaNs
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "method": suppressor_factory(cfg).name,
        "n_records": len(per_record),
        "F1_mean": _mean("overall", "F1"),
        "Se_mean": _mean("overall", "Se"),
        "PPV_mean": _mean("overall", "PPV"),
        "coincident_Se_mean": _mean("coincident_Se"),
        "non_coincident_Se_mean": _mean("non_coincident_Se"),
        "fhr_mae_mean": _mean("fhr_mae"),
    }
    return {"summary": summary, "per_record": per_record}
