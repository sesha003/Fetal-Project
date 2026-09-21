#!/usr/bin/env python3
"""
run_recovery_experiment.py — does the reframe actually help at overlap?

Benchmarks the novel occlusion-aware recovery arm against the classical suppression
arms in the SAME leave-one-record-out harness, and prints the comparison. The claim
succeeds if `coincident_Se` rises for the recovery arm while `non_coincident_Se` and
PPV stay flat — i.e. it recovers hidden beats without inventing false ones.

    python run_recovery_experiment.py            # synthetic cohort (no download)
    python run_recovery_experiment.py --records 8
"""
from __future__ import annotations

import argparse
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numpy as np

from fecg import (PipelineConfig, synthetic_cohort, run_pipeline,
                  TemplateSubtraction, ICASuppressor)
from fecg.recovery import run_recovery_pipeline, OcclusionAwareRecovery


def _summarize(per_record):
    def m(key, sub=None):
        vals = [(r[key][sub] if sub else r[key]) for r in per_record]
        vals = [v for v in vals if v == v]
        return float(np.mean(vals)) if vals else float("nan")
    return dict(F1=m("overall", "F1"), Se=m("overall", "Se"), PPV=m("overall", "PPV"),
                non_coin_Se=m("non_coincident_Se"), coin_Se=m("coincident_Se"),
                fhr_mae=m("fhr_mae"))


def _collect(eval_dict, fhr_err):
    return dict(overall=eval_dict["overall"],
                coincident_Se=eval_dict["coincident"]["Se"],
                non_coincident_Se=eval_dict["non_coincident"]["Se"],
                fhr_mae=fhr_err.get("mae", np.nan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, default=6)
    ap.add_argument("--duration", type=float, default=30.0)
    args = ap.parse_args()

    cfg = PipelineConfig()
    cohort = synthetic_cohort(cfg, n_records=args.records, duration_s=args.duration)
    print(f"Cohort: {len(cohort)} records @ {cfg.fs:.0f} Hz\n")

    results = {}

    # classical baselines
    for label, factory in [("template", lambda c: TemplateSubtraction(c)),
                           ("bss_ica", lambda c: ICASuppressor(c))]:
        per = []
        for rec in cohort:
            out = run_pipeline(rec, factory(cfg), cfg)
            per.append(_collect(out["eval"], out.get("fhr_error", {})))
        results[label] = _summarize(per)

    # novel arm
    per, recov_counts = [], []
    for rec in cohort:
        out = run_recovery_pipeline(rec, cfg, OcclusionAwareRecovery(cfg))
        per.append(_collect(out["eval"], out.get("fhr_error", {})))
        recov_counts.append(out["n_recovered"])
    results["occlusion_recovery"] = _summarize(per)

    # table
    cols = ["F1", "Se", "PPV", "non_coin_Se", "coin_Se", "fhr_mae"]
    widths = [22] + [13] * len(cols)
    print("".join(h.ljust(w) for h, w in zip(["method"] + cols, widths)))
    print("-" * (widths[0] + 13 * len(cols)))
    for name, s in results.items():
        row = [name] + [f"{s[c]:.3f}" if s[c] == s[c] else "nan" for c in cols]
        print("".join(str(v).ljust(w) for v, w in zip(row, widths)))

    print(f"\nBeats recovered under occlusion (per record): {recov_counts}")
    print("Success = occlusion_recovery lifts `coin_Se` above the baselines")
    print("while holding `non_coin_Se` and `PPV` roughly flat.")


if __name__ == "__main__":
    main()
