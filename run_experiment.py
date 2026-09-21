#!/usr/bin/env python3
"""
run_experiment.py — end-to-end method comparison for the fECG overlap project.

Runs every classical suppression arm through the same leave-one-record-out harness
and prints one comparison table. The column that matters is `coincident_Se` — the
recovery rate on fetal beats that overlap a maternal QRS.

By default it runs on a synthetic cohort (no download needed). Point it at real
ADFECGDB records by editing `load_cohort()`.

Usage:
    python run_experiment.py                # synthetic cohort
    python run_experiment.py --records 8    # larger synthetic cohort
"""
from __future__ import annotations

import argparse
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from fecg import (PipelineConfig, synthetic_cohort, leave_one_record_out,
                  TemplateSubtraction, ICASuppressor, AdaptiveSuppressor,
                  KalmanTemplateSuppressor, load_physionet_record)


def load_cohort(cfg, n_records, duration_s, dataset="synthetic"):
    if dataset == "synthetic":
        return synthetic_cohort(cfg, n_records=n_records, duration_s=duration_s)
    if dataset == "cinc2013":
        import glob, os
        ids = sorted(os.path.basename(f)[:-4]
                     for f in glob.glob("data/cinc2013/set-a/a*.hea"))
        if n_records is not None:
            ids = ids[:n_records]
        return [load_physionet_record(f"data/cinc2013/set-a/{r}", ann_ext="fqrs", name=r)
                for r in ids]
    raise ValueError(f"unknown dataset: {dataset}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, default=None)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--fs", type=float, default=1000.0)
    ap.add_argument("--dataset", choices=["synthetic", "cinc2013"], default="synthetic")
    args = ap.parse_args()

    n_records = args.records if args.records is not None else (
        5 if args.dataset == "synthetic" else None)
    cfg = PipelineConfig(fs=args.fs)
    cohort = load_cohort(cfg, n_records, args.duration, dataset=args.dataset)
    print(f"Cohort: {len(cohort)} records @ {cfg.fs:.0f} Hz\n")

    arms = {
        "template":  lambda c: TemplateSubtraction(c),
        "adaptive":  lambda c: AdaptiveSuppressor(c, method="rls"),
        "bss_ica":   lambda c: ICASuppressor(c, pre_template=True),
        "kalman":    lambda c: KalmanTemplateSuppressor(c),
    }

    results = {}
    for name, factory in arms.items():
        results[name] = leave_one_record_out(cohort, factory, cfg)["summary"]

    # comparison table
    cols = ["F1_mean", "Se_mean", "PPV_mean",
            "non_coincident_Se_mean", "coincident_Se_mean", "fhr_mae_mean"]
    header = ["method"] + [c.replace("_mean", "") for c in cols]
    widths = [10] + [16] * len(cols)
    line = "".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for name, s in results.items():
        row = [name] + [f"{s[c]:.3f}" if s[c] == s[c] else "nan" for c in cols]
        print("".join(str(v).ljust(w) for v, w in zip(row, widths)))

    print("\nThe `coincident_Se` column is the headline result: recovery rate on")
    print("fetal beats that overlap a maternal QRS. The gap vs `non_coincident_Se`")
    print("is what each method's separation strategy buys you.")


if __name__ == "__main__":
    main()
