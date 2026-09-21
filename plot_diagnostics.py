#!/usr/bin/env python3
"""
plot_diagnostics.py — detailed diagnostic plots that explain *why* the methods
differ, beyond the aggregate bars in plot_comparison.py.

Reads results/comparison_metrics.json (+ results/dl_unet.pth when present) and writes:
  fig_signal_anatomy.png      raw -> residual -> DL probability, with overlap marked
  fig_dl_probability.png      U-Net probability trace vs ground-truth beats
  fig_fhr_trace.png           FHR over time per method vs ground truth
  fig_bland_altman.png        FHR agreement (clinical standard) for the best model
  fig_paired_records.png      per-record F1: best classical -> dl_unet
  fig_error_breakdown.png     TP / FP / FN composition per method
  fig_se_ppv_scatter.png      per-record operating points (Se vs PPV)
  fig_overlap_vs_f1.png       does overlap fraction predict record difficulty?
  fig_dl_threshold_sweep.png  Se/PPV/F1 vs detection threshold (production tuning)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from fecg import PipelineConfig, load_physionet_record, TemplateSubtraction
from fecg.preprocess import preprocess, detect_maternal_qrs, butter_bandpass
from fecg.fetal import detect_fetal_qrs, compute_fhr
from fecg.evaluate import match_peaks

RESULTS_DIR = Path("results")

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
GOOD, CRITICAL, WARNING = "#0ca30c", "#d03b3b", "#fab219"

METHOD_COLOR = {"template": BLUE, "adaptive": ORANGE, "bss_ica": AQUA,
                "kalman": YELLOW, "dl_unet": MAGENTA}
METHOD_ORDER = ["template", "adaptive", "bss_ica", "kalman", "dl_unet"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def load_record(name):
    return load_physionet_record(f"data/cinc2013/set-a/{name}", ann_ext="fqrs", name=name)


def get_detector(cfg):
    """Load the saved production model, or None if it hasn't been trained yet."""
    path = RESULTS_DIR / "dl_unet.pth"
    if not path.exists():
        return None
    import torch
    from fecg.dl import DLDetector
    from fecg.dl.model import UNet1D
    model = UNet1D(in_channels=4)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return DLDetector(model, cfg)


# --------------------------------------------------------------------------- #
def plot_signal_anatomy(rec, cfg, detector, out, t0=10.0, dur=6.0):
    """The money plot: what each stage actually sees, with overlap beats marked."""
    fs = cfg.fs
    a, b = int(t0 * fs), int((t0 + dur) * fs)
    t = np.arange(a, b) / fs

    clean = preprocess(rec.signal, cfg)
    mat_r = detect_maternal_qrs(clean, cfg)
    residual = TemplateSubtraction(cfg).suppress(clean, mat_r)
    fet_det = detect_fetal_qrs(residual, cfg)

    # fetal-band RMS of the residual — what the classical detector thresholds on
    band = np.column_stack([butter_bandpass(residual[:, c], fs, cfg.fet_bp_low,
                                            cfg.fet_bp_high)
                            for c in range(residual.shape[1])])
    combined = np.sqrt(np.mean(band ** 2, axis=1))

    gt = np.asarray(rec.fet_r)
    coin_win = cfg.coincidence_win_s * fs
    is_coin = np.array([np.min(np.abs(mat_r - r)) <= coin_win for r in gt]) if len(mat_r) else np.zeros(len(gt), bool)

    nrows = 3 if detector is not None else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(13, 3.0 * nrows), sharex=True,
                             facecolor=SURFACE)

    def mark_overlap(ax):
        for m in mat_r[(mat_r >= a) & (mat_r < b)]:
            ax.axvspan((m - coin_win) / fs, (m + coin_win) / fs,
                       color=CRITICAL, alpha=0.10, lw=0)

    # row 1: preprocessed abdominal signal + maternal QRS
    ax = axes[0]
    ax.plot(t, clean[a:b, 0], color=MUTED, linewidth=0.8)
    mark_overlap(ax)
    for m in mat_r[(mat_r >= a) & (mat_r < b)]:
        ax.axvline(m / fs, color=CRITICAL, linewidth=1.0, alpha=0.7)
    ax.set_ylabel("abdominal ECG")
    ax.set_title(f"Record {rec.name}: maternal QRS (red) dominates; shaded bands are "
                 f"the ±{cfg.coincidence_win_s * 1000:.0f} ms overlap zone")
    _style(ax)

    # row 2: template residual + detections vs truth
    ax = axes[1]
    ax.plot(t, combined[a:b], color=BLUE, linewidth=0.9)
    mark_overlap(ax)
    win_gt = gt[(gt >= a) & (gt < b)]
    win_coin = is_coin[(gt >= a) & (gt < b)]
    ax.plot(win_gt[~win_coin] / fs, np.full((~win_coin).sum(), combined[a:b].max() * 1.10),
            "v", color=GOOD, markersize=8, label="true fetal beat")
    ax.plot(win_gt[win_coin] / fs, np.full(win_coin.sum(), combined[a:b].max() * 1.10),
            "v", color=CRITICAL, markersize=8, label="true fetal beat (overlapped)")
    d = fet_det[(fet_det >= a) & (fet_det < b)]
    ax.plot(d / fs, np.full(len(d), combined[a:b].max() * 1.02), "^", color=INK,
            markersize=7, label="template detection")
    ax.set_ylabel("template residual\n(fetal band RMS)")
    ax.set_ylim(0, combined[a:b].max() * 1.30)
    ax.legend(frameon=False, labelcolor=INK, ncol=3, fontsize=9,
              loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0)
    _style(ax)

    # row 3: DL probability
    if detector is not None:
        prob = detector.probability(rec)
        dl_det = detector.detect(rec)
        ax = axes[2]
        ax.plot(t, prob[a:b], color=MAGENTA, linewidth=1.2)
        ax.axhline(0.3, color=MUTED, linestyle="--", linewidth=1.0, label="threshold 0.3")
        mark_overlap(ax)
        ax.plot(win_gt[~win_coin] / fs, np.full((~win_coin).sum(), 1.06), "v",
                color=GOOD, markersize=8)
        ax.plot(win_gt[win_coin] / fs, np.full(win_coin.sum(), 1.06), "v",
                color=CRITICAL, markersize=8)
        dd = dl_det[(dl_det >= a) & (dl_det < b)]
        ax.plot(dd / fs, np.full(len(dd), 0.98), "^", color=INK, markersize=7,
                label="U-Net detection")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("U-Net P(fetal R)")
        ax.legend(frameon=False, labelcolor=INK, ncol=2, fontsize=9,
                  loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0)
        _style(ax)

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_dl_probability(rec, cfg, detector, out, t0=10.0, dur=12.0):
    """Zoomed-out probability trace: is the model confident, or just barely clearing?"""
    if detector is None:
        return
    fs = cfg.fs
    a, b = int(t0 * fs), int((t0 + dur) * fs)
    t = np.arange(a, b) / fs
    prob = detector.probability(rec)
    gt = np.asarray(rec.fet_r)
    win_gt = gt[(gt >= a) & (gt < b)]

    fig, ax = plt.subplots(figsize=(13, 4.5), facecolor=SURFACE)
    ax.fill_between(t, 0, prob[a:b], color=MAGENTA, alpha=0.30, lw=0)
    ax.plot(t, prob[a:b], color=MAGENTA, linewidth=1.0)
    ax.axhline(0.3, color=MUTED, linestyle="--", linewidth=1.0)
    ax.text(t[0], 0.32, "detection threshold 0.3", color=MUTED, fontsize=9)
    for g in win_gt:
        ax.axvline(g / fs, color=GOOD, linewidth=0.9, alpha=0.55)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("P(fetal R-peak)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"U-Net probability trace vs ground-truth beats (green) — record {rec.name}")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_fhr_trace(rec, cfg, detector, out):
    """The clinician-facing output: FHR over time, truth vs each method."""
    fs = cfg.fs
    gt = np.asarray(rec.fet_r)
    gt_t, gt_fhr = compute_fhr(gt, cfg)

    clean = preprocess(rec.signal, cfg)
    mat_r = detect_maternal_qrs(clean, cfg)
    residual = TemplateSubtraction(cfg).suppress(clean, mat_r)
    tmpl_t, tmpl_fhr = compute_fhr(detect_fetal_qrs(residual, cfg), cfg)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=SURFACE)
    ax.plot(gt_t, gt_fhr, color=INK, linewidth=2.2, label="ground truth", zorder=3)
    ax.plot(tmpl_t, tmpl_fhr, color=BLUE, linewidth=1.3, alpha=0.85, label="template")
    if detector is not None:
        dl_t, dl_fhr = compute_fhr(detector.detect(rec), cfg)
        ax.plot(dl_t, dl_fhr, color=MAGENTA, linewidth=1.3, alpha=0.9, label="dl_unet")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("fetal heart rate (bpm)")
    ax.set_title(f"What the clinician sees: FHR trace vs ground truth — record {rec.name}")
    ax.legend(frameon=False, labelcolor=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_bland_altman(rec, cfg, detector, out):
    """Clinical agreement standard: mean vs difference, with 95% limits."""
    if detector is None:
        return
    gt = np.asarray(rec.fet_r)
    gt_t, gt_fhr = compute_fhr(gt, cfg)
    dl_t, dl_fhr = compute_fhr(detector.detect(rec), cfg)
    if len(gt_fhr) < 2 or len(dl_fhr) < 2:
        return
    ref = np.interp(dl_t, gt_t, gt_fhr)
    mean = (dl_fhr + ref) / 2
    diff = dl_fhr - ref
    bias, sd = float(np.mean(diff)), float(np.std(diff))

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    ax.scatter(mean, diff, s=26, color=MAGENTA, alpha=0.65, edgecolor=SURFACE, linewidth=0.5)
    ax.axhline(bias, color=INK, linewidth=1.6, label=f"bias {bias:+.2f} bpm")
    ax.axhline(bias + 1.96 * sd, color=CRITICAL, linestyle="--", linewidth=1.2,
               label=f"95% limits ±{1.96 * sd:.1f} bpm")
    ax.axhline(bias - 1.96 * sd, color=CRITICAL, linestyle="--", linewidth=1.2)
    ax.set_xlabel("mean of (U-Net, truth) FHR (bpm)")
    ax.set_ylabel("U-Net − truth (bpm)")
    ax.set_title(f"Bland–Altman FHR agreement, dl_unet — record {rec.name}")
    ax.legend(frameon=False, labelcolor=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_paired_records(per_record, out, baseline="template", champion="dl_unet"):
    """Does the champion win on every record, or just on average?"""
    if baseline not in per_record or champion not in per_record:
        return
    base = {r["record"]: r["overall"]["F1"] for r in per_record[baseline]}
    champ = {r["record"]: r["overall"]["F1"] for r in per_record[champion]}
    names = sorted(set(base) & set(champ))

    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor=SURFACE)
    n_win = 0
    for nm in names:
        y0, y1 = base[nm], champ[nm]
        improved = y1 >= y0
        n_win += improved
        ax.plot([0, 1], [y0, y1], color=GOOD if improved else CRITICAL,
                alpha=0.65, linewidth=1.6, marker="o", markersize=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([baseline, champion])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("F1 (per test record)")
    ax.set_title(f"Per-record F1: {champion} beats {baseline} on {n_win}/{len(names)} records\n"
                 f"(green = improved, red = regressed)")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_error_breakdown(per_record, methods, out):
    """Failure *shape*: missing beats (FN) vs inventing them (FP)."""
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    tp = [sum(r["overall"]["TP"] for r in per_record[m]) for m in methods]
    fp = [sum(r["overall"]["FP"] for r in per_record[m]) for m in methods]
    fn = [sum(r["overall"]["FN"] for r in per_record[m]) for m in methods]
    x = np.arange(len(methods))
    ax.bar(x, tp, 0.6, label="TP (correct)", color=GOOD, edgecolor=SURFACE, linewidth=2)
    ax.bar(x, fn, 0.6, bottom=tp, label="FN (missed beat)", color=WARNING,
           edgecolor=SURFACE, linewidth=2)
    ax.bar(x, fp, 0.6, bottom=np.array(tp) + np.array(fn), label="FP (invented beat)",
           color=CRITICAL, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("beats (summed over test set)")
    ax.set_title("Error composition — FP (red) is the clinically dangerous failure")
    ax.legend(frameon=False, labelcolor=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_se_ppv_scatter(per_record, methods, out):
    """Operating points: conservative (high PPV, low Se) vs trigger-happy."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor=SURFACE)
    for m in methods:
        se = [r["overall"]["Se"] for r in per_record[m]]
        ppv = [r["overall"]["PPV"] for r in per_record[m]]
        ax.scatter(se, ppv, s=55, color=METHOD_COLOR[m], alpha=0.75, label=m,
                   edgecolor=SURFACE, linewidth=1.2)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Sensitivity (Se) — beats found")
    ax.set_ylabel("PPV — detections that are real")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Per-record operating points\n(top-right is ideal; above the line = conservative)")
    ax.legend(frameon=False, labelcolor=INK, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_overlap_vs_f1(per_record, methods, out):
    """Is a record hard *because* of overlap, or for unrelated noise reasons?"""
    if not any("pct_coincident" in r for m in methods for r in per_record[m]):
        return
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    for m in methods:
        pts = [(r.get("pct_coincident"), r["overall"]["F1"]) for r in per_record[m]
               if r.get("pct_coincident") is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(np.array(xs) * 100, ys, s=55, color=METHOD_COLOR[m], alpha=0.75,
                   label=m, edgecolor=SURFACE, linewidth=1.2)
    ax.set_xlabel("% of fetal beats overlapping a maternal QRS")
    ax.set_ylabel("F1 on that record")
    ax.set_title("Does overlap fraction explain record difficulty?")
    ax.legend(frameon=False, labelcolor=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_threshold_sweep(test_names, cfg, detector, out):
    """Production tuning: the 0.3 threshold is arbitrary — show the whole curve."""
    if detector is None:
        return
    # Fine at the low end: F1 rises monotonically as the threshold falls, so a sweep
    # starting at 0.05 pins its "optimum" at the boundary and hides the real turn.
    thresholds = np.concatenate([np.arange(0.002, 0.02, 0.002),
                                 np.arange(0.02, 0.10, 0.01),
                                 np.arange(0.10, 0.96, 0.05)])
    refractory = int((60.0 / cfg.fhr_max_bpm) * cfg.fs)
    tol = int(cfg.match_tol_s * cfg.fs)

    traces = []
    for nm in test_names:
        rec = load_record(nm)
        traces.append((detector.probability(rec), np.asarray(rec.fet_r)))

    se_c, ppv_c, f1_c = [], [], []
    for th in thresholds:
        tp = fp = fn = 0
        for prob, gt in traces:
            peaks, _ = find_peaks(prob, height=th, distance=refractory)
            a, b, c, _ = match_peaks(peaks, gt, tol)
            tp, fp, fn = tp + a, fp + b, fn + c
        se = tp / (tp + fn) if (tp + fn) else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        se_c.append(se)
        ppv_c.append(ppv)
        f1_c.append(2 * se * ppv / (se + ppv) if (se + ppv) else 0.0)

    best = int(np.argmax(f1_c))
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    ax.plot(thresholds, se_c, color=ORANGE, linewidth=2, marker="o", markersize=4, label="Se")
    ax.plot(thresholds, ppv_c, color=AQUA, linewidth=2, marker="o", markersize=4, label="PPV")
    ax.plot(thresholds, f1_c, color=BLUE, linewidth=2.4, marker="o", markersize=4, label="F1")
    ax.axvline(0.3, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(0.305, 0.05, "current default 0.3", color=MUTED, fontsize=9, rotation=90)
    ax.axvline(thresholds[best], color=GOOD, linestyle="-", linewidth=1.4, alpha=0.8)
    ax.text(thresholds[best] + 0.01, 0.05,
            f"F1-optimal {thresholds[best]:.2f}", color=GOOD, fontsize=9, rotation=90)
    ax.set_xscale("log")
    ax.set_xlabel("detection threshold on P(fetal R)  — log scale")
    ax.set_ylabel("score (pooled over test set)")
    ax.set_ylim(0, 1.05)
    ax.set_title("dl_unet threshold sweep — the Se/PPV trade-off knob for deployment")
    ax.legend(frameon=False, labelcolor=INK, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    cur = int(np.argmin(np.abs(thresholds - 0.3)))
    print(f"  threshold sweep: current {thresholds[cur]:.2f} -> F1 {f1_c[cur]:.3f} | "
          f"optimal {thresholds[best]:.3f} -> F1 {f1_c[best]:.3f} "
          f"(Se {se_c[best]:.3f}, PPV {ppv_c[best]:.3f})")


def main():
    with open(RESULTS_DIR / "comparison_metrics.json") as f:
        data = json.load(f)
    per_record = data["per_record"]
    methods = [m for m in METHOD_ORDER if m in per_record]
    test_names = data.get("test_records") or [r["record"] for r in per_record[methods[0]]]

    cfg = PipelineConfig(fs=1000.0)
    detector = get_detector(cfg)
    if detector is None:
        print("No results/dl_unet.pth yet — skipping DL-specific plots.")

    # Illustrate both ends: where the champion wins most AND where it does worst,
    # so the signal-level story isn't cherry-picked.
    demo = worst = test_names[0]
    if "dl_unet" in per_record and "template" in per_record:
        base = {r["record"]: r["overall"]["F1"] for r in per_record["template"]}
        champ = {r["record"]: r["overall"]["F1"] for r in per_record["dl_unet"]}
        shared = set(base) & set(champ)
        if shared:
            demo = max(shared, key=lambda n: champ[n] - base[n])
            worst = min(shared, key=lambda n: champ[n])
    print(f"Signal-level plots: best-case {demo}, worst-case {worst}")
    rec = load_record(demo)

    plot_signal_anatomy(rec, cfg, detector, RESULTS_DIR / "fig_signal_anatomy.png")
    if worst != demo:
        plot_signal_anatomy(load_record(worst), cfg, detector,
                            RESULTS_DIR / "fig_signal_anatomy_worstcase.png")
    plot_dl_probability(rec, cfg, detector, RESULTS_DIR / "fig_dl_probability.png")
    plot_fhr_trace(rec, cfg, detector, RESULTS_DIR / "fig_fhr_trace.png")
    plot_bland_altman(rec, cfg, detector, RESULTS_DIR / "fig_bland_altman.png")
    plot_paired_records(per_record, RESULTS_DIR / "fig_paired_records.png")
    plot_error_breakdown(per_record, methods, RESULTS_DIR / "fig_error_breakdown.png")
    plot_se_ppv_scatter(per_record, methods, RESULTS_DIR / "fig_se_ppv_scatter.png")
    plot_overlap_vs_f1(per_record, methods, RESULTS_DIR / "fig_overlap_vs_f1.png")
    plot_threshold_sweep(test_names, cfg, detector,
                         RESULTS_DIR / "fig_dl_threshold_sweep.png")
    print(f"Saved diagnostic plots to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
