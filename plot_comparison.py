#!/usr/bin/env python3
"""
plot_comparison.py — visualize results/comparison_metrics.json from
run_full_comparison.py. Produces:
  results/fig_headline_metrics.png     F1 / Se / PPV per method
  results/fig_coincident_gap.png       non-coincident vs coincident Se per method
  results/fig_fhr_error.png            FHR MAE per method (lower is better)
  results/fig_f1_distribution.png      per-record F1 spread per method (boxplot)
  results/fig_training_loss.png        DL U-Net training loss curve
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")

# Fixed categorical order (never cycled) — dataviz skill's validated default palette.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

METHOD_COLOR = {
    "template": BLUE, "adaptive": ORANGE, "bss_ica": AQUA,
    "kalman": YELLOW, "dl_unet": MAGENTA,
}
METHOD_ORDER = ["template", "adaptive", "bss_ica", "kalman", "dl_unet"]


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def plot_headline_metrics(summary, methods, out):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    # No error bars: these metrics are bounded to [0,1], so a +/-1 SD whisker
    # overshoots the bound and misleads. Spread lives in fig_f1_distribution.
    metrics = [("F1_mean", BLUE), ("Se_mean", ORANGE), ("PPV_mean", AQUA)]
    x = np.arange(len(methods))
    width = 0.25
    for i, (mkey, color) in enumerate(metrics):
        vals = [summary[m][mkey] for m in methods]
        ax.bar(x + (i - 1) * width, vals, width,
               label=mkey.replace("_mean", ""), color=color,
               edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Overall detection quality by method (real CinC2013 test set)")
    ax.legend(frameon=False, labelcolor=INK)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_coincident_gap(summary, methods, out):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    x = np.arange(len(methods))
    width = 0.32
    non_coin = [summary[m]["non_coincident_Se_mean"] for m in methods]
    coin = [summary[m]["coincident_Se_mean"] for m in methods]
    ax.bar(x - width / 2, non_coin, width, label="non-coincident Se",
           color=BLUE, edgecolor=SURFACE, linewidth=2)
    ax.bar(x + width / 2, coin, width, label="coincident Se",
           color=ORANGE, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("sensitivity (Se)")
    ax.set_title("The headline result: fetal-beat recovery with vs. without\nmaternal QRS overlap")
    ax.legend(frameon=False, labelcolor=INK)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_fhr_error(summary, methods, out):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    vals = [summary[m]["fhr_mae_mean"] for m in methods]
    errs = [summary[m]["fhr_mae_std"] for m in methods]
    colors = [METHOD_COLOR[m] for m in methods]
    ax.bar(methods, vals, yerr=errs, capsize=3, color=colors,
           edgecolor=SURFACE, linewidth=2,
           error_kw=dict(ecolor=MUTED, elinewidth=1.2, capthick=1.2))
    ax.set_ylabel("FHR MAE (bpm)")
    ax.set_title("Fetal heart-rate error by method (lower is better)")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_f1_distribution(per_record, methods, out):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    data = [[r["overall"]["F1"] for r in per_record[m]] for m in methods]
    bp = ax.boxplot(data, tick_labels=methods, patch_artist=True, widths=0.5,
                    medianprops=dict(color=INK, linewidth=1.5),
                    whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                    flierprops=dict(markeredgecolor=MUTED, markersize=4))
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(METHOD_COLOR[m])
        patch.set_alpha(0.75)
        patch.set_edgecolor(SURFACE)
    ax.set_ylabel("F1 (per test record)")
    ax.set_title("Per-record F1 spread — robustness across the test set")
    ax.set_ylim(-0.02, 1.05)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_training_loss(loss_history, out):
    if not loss_history:
        return
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    epochs = np.arange(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, color=MAGENTA, linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("DL U-Net training loss")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    with open(RESULTS_DIR / "comparison_metrics.json") as f:
        data = json.load(f)
    summary = data["summary"]
    per_record = data["per_record"]
    loss_history = data.get("loss_history", [])
    methods = [m for m in METHOD_ORDER if m in summary]

    plot_headline_metrics(summary, methods, RESULTS_DIR / "fig_headline_metrics.png")
    plot_coincident_gap(summary, methods, RESULTS_DIR / "fig_coincident_gap.png")
    plot_fhr_error(summary, methods, RESULTS_DIR / "fig_fhr_error.png")
    plot_f1_distribution(per_record, methods, RESULTS_DIR / "fig_f1_distribution.png")
    plot_training_loss(loss_history, RESULTS_DIR / "fig_training_loss.png")
    print(f"Saved plots to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
