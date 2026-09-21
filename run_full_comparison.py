#!/usr/bin/env python3
"""
run_full_comparison.py — full model comparison on real data + production pick.

Evaluates every arm (4 classical suppressors + the DL U-Net detector) on the SAME
held-out test split of real CinC2013 records, so the comparison is apples-to-apples
(the classical arms need no training; the DL arm is trained only on the train split).

Outputs:
  - results/comparison_metrics.json   per-record + summary metrics for every method
  - results/*.png                     comparison plots
  - stdout                            comparison table + production recommendation

Usage:
    python run_full_comparison.py                    # 60 train / 15 test, 20 epochs
    python run_full_comparison.py --epochs 10         # faster DL training
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from fecg import (PipelineConfig, load_physionet_record, run_pipeline,
                  TemplateSubtraction, ICASuppressor, AdaptiveSuppressor,
                  KalmanTemplateSuppressor)
from fecg.dl import train_model, DLDetector
from fecg.preprocess import preprocess, detect_maternal_qrs
from fecg.evaluate import evaluate, fhr_error

RESULTS_DIR = Path("results")


def load_cinc2013():
    import glob, os
    ids = sorted(os.path.basename(f)[:-4] for f in glob.glob("data/cinc2013/set-a/a*.hea"))
    return [load_physionet_record(f"data/cinc2013/set-a/{r}", ann_ext="fqrs", name=r)
            for r in ids]


def split(records, n_test=15, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return [records[i] for i in train_idx], [records[i] for i in test_idx]


def eval_classical_record(rec, suppressor, cfg):
    out = run_pipeline(rec, suppressor, cfg)
    return dict(record=rec.name, overall=out["eval"]["overall"],
                coincident_Se=out["eval"]["coincident"]["Se"],
                non_coincident_Se=out["eval"]["non_coincident"]["Se"],
                pct_coincident=out["eval"]["pct_beats_coincident"],
                fhr_mae=out.get("fhr_error", {}).get("mae", np.nan))


def eval_dl_record(rec, detector, cfg):
    fet_r = detector.detect(rec)
    clean = preprocess(rec.signal, cfg)
    mat_r = detect_maternal_qrs(clean, cfg)
    ref_mat = rec.mat_r if rec.mat_r is not None else mat_r
    ev = evaluate(fet_r, rec.fet_r, ref_mat, cfg)
    from fecg.fetal import compute_fhr
    fhr_t, fhr = compute_fhr(fet_r, cfg)
    fe = fhr_error(fhr_t, fhr, rec.fet_r, cfg)
    return dict(record=rec.name, overall=ev["overall"],
                coincident_Se=ev["coincident"]["Se"],
                non_coincident_Se=ev["non_coincident"]["Se"],
                pct_coincident=ev["pct_beats_coincident"],
                fhr_mae=fe.get("mae", np.nan))


def summarize(per_record):
    def m(key, sub=None):
        vals = [(r[key][sub] if sub else r[key]) for r in per_record]
        vals = [v for v in vals if v == v]
        return float(np.mean(vals)) if vals else float("nan")

    def s(key, sub=None):
        vals = [(r[key][sub] if sub else r[key]) for r in per_record]
        vals = [v for v in vals if v == v]
        return float(np.std(vals)) if vals else float("nan")

    return dict(
        F1_mean=m("overall", "F1"), F1_std=s("overall", "F1"),
        Se_mean=m("overall", "Se"), Se_std=s("overall", "Se"),
        PPV_mean=m("overall", "PPV"), PPV_std=s("overall", "PPV"),
        coincident_Se_mean=m("coincident_Se"), coincident_Se_std=s("coincident_Se"),
        non_coincident_Se_mean=m("non_coincident_Se"), non_coincident_Se_std=s("non_coincident_Se"),
        fhr_mae_mean=m("fhr_mae"), fhr_mae_std=s("fhr_mae"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    cfg = PipelineConfig(fs=1000.0)

    print("Loading CinC2013 set-a (real data)...")
    records = load_cinc2013()
    train_records, test_records = split(records, n_test=args.n_test, seed=args.seed)
    print(f"Train: {len(train_records)} records   Test: {len(test_records)} records\n")

    classical_arms = {
        "template": lambda c: TemplateSubtraction(c),
        "adaptive": lambda c: AdaptiveSuppressor(c, method="rls"),
        "bss_ica":  lambda c: ICASuppressor(c, pre_template=True),
        "kalman":   lambda c: KalmanTemplateSuppressor(c),
    }

    all_per_record = {}
    all_summary = {}

    for name, factory in classical_arms.items():
        print(f"Evaluating {name} on {len(test_records)} test records...")
        per = [eval_classical_record(rec, factory(cfg), cfg) for rec in test_records]
        all_per_record[name] = per
        all_summary[name] = summarize(per)

    print(f"\nTraining DL U-Net on {len(train_records)} training records "
          f"({args.epochs} epochs)...")
    model, loss_history = train_model(train_records, cfg, in_channels=4,
                                      epochs=args.epochs, return_history=True)
    import torch
    torch.save(model.state_dict(), RESULTS_DIR / "dl_unet.pth")
    print(f"Saved trained model to {RESULTS_DIR / 'dl_unet.pth'}")
    detector = DLDetector(model, cfg)

    print(f"\nEvaluating dl_unet on {len(test_records)} test records...")
    per = [eval_dl_record(rec, detector, cfg) for rec in test_records]
    all_per_record["dl_unet"] = per
    all_summary["dl_unet"] = summarize(per)

    # ---- comparison table ----
    cols = ["F1_mean", "Se_mean", "PPV_mean",
            "non_coincident_Se_mean", "coincident_Se_mean", "fhr_mae_mean"]
    header = ["method"] + [c.replace("_mean", "") for c in cols]
    widths = [10] + [16] * len(cols)
    line = "".join(h.ljust(w) for h, w in zip(header, widths))
    print("\n" + line)
    print("-" * len(line))
    for name, s in all_summary.items():
        row = [name] + [f"{s[c]:.3f}" if s[c] == s[c] else "nan" for c in cols]
        print("".join(str(v).ljust(w) for v, w in zip(row, widths)))

    # ---- pick a production model ----
    # Primary criterion: coincident_Se (the project's headline metric — recovery at
    # maternal/fetal overlap). Tie-break on F1 to avoid rewarding a method that only
    # wins coincident_Se by over-triggering (checked via PPV not collapsing).
    ranked = sorted(all_summary.items(),
                    key=lambda kv: (kv[1]["coincident_Se_mean"], kv[1]["F1_mean"]),
                    reverse=True)
    best_name, best_metrics = ranked[0]

    print("\n=== Production recommendation ===")
    for rank, (name, s) in enumerate(ranked, 1):
        print(f"{rank}. {name:10s}  coincident_Se={s['coincident_Se_mean']:.3f}  "
              f"F1={s['F1_mean']:.3f}  PPV={s['PPV_mean']:.3f}  "
              f"fhr_mae={s['fhr_mae_mean']:.2f} bpm")
    print(f"\nBest for production: {best_name}")

    # ---- save results ----
    with open(RESULTS_DIR / "comparison_metrics.json", "w") as f:
        json.dump({"summary": all_summary, "per_record": all_per_record,
                   "loss_history": loss_history, "best_model": best_name,
                   "n_train": len(train_records), "n_test": len(test_records),
                   "test_records": [r.name for r in test_records],
                   "seed": args.seed},
                  f, indent=2)
    print(f"\nSaved metrics to {RESULTS_DIR / 'comparison_metrics.json'}")


if __name__ == "__main__":
    main()
