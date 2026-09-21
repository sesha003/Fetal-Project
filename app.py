"""
app.py — Streamlit test bench for the trained fetal-ECG detector.

    streamlit run app.py

Four views:
  1. Single record   — run the model (and classical baselines) on one recording,
                       inspect the signal, the probability trace and the FHR.
  2. Batch evaluation — sweep a set of records, aggregate overlap-stratified metrics.
  3. Threshold sweep  — where the peak-picking threshold should actually sit.
  4. Saved training run — the metrics/figures produced by run_full_comparison.py.

Design rule throughout: never show a fetal detection score without also showing
whether the record was in the model's training set and whether the annotations are
actually fetal. An impressive F1 on a training record is not a result.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import testbench as tb
from fecg import PipelineConfig
from fecg.preprocess import preprocess, detect_maternal_qrs

st.set_page_config(page_title="fECG Test Bench", page_icon="H", layout="wide")

ACCENT = "#d64550"      # detections
TRUTH = "#2a9d5c"       # ground truth
MAT = "#9aa0a6"         # maternal


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def get_model(ckpt_path: str, mtime: float):
    return tb.load_model(ckpt_path)


@st.cache_data(show_spinner="Loading record...")
def get_record(ds_name: str, rec_id: str):
    return tb.load_record(ds_name, rec_id)


@st.cache_data(show_spinner="Running model...")
def get_probability(ds_name: str, rec_id: str, ch: tuple, max_s: float,
                    ckpt_path: str, mtime: float):
    model, _ = get_model(ckpt_path, mtime)
    rec = tb.subset_record(get_record(ds_name, rec_id), list(ch), max_s)
    cfg = PipelineConfig(fs=rec.fs)
    return tb.dl_probability(model, rec, cfg)


@st.cache_data(show_spinner="Detecting maternal QRS...")
def get_maternal(ds_name: str, rec_id: str, ch: tuple, max_s: float):
    rec = tb.subset_record(get_record(ds_name, rec_id), list(ch), max_s)
    cfg = PipelineConfig(fs=rec.fs)
    return detect_maternal_qrs(preprocess(rec.signal, cfg), cfg)


@st.cache_data(show_spinner="Running classical arm...")
def get_classical(ds_name: str, rec_id: str, ch: tuple, max_s: float, arm: str):
    rec = tb.subset_record(get_record(ds_name, rec_id), list(ch), max_s)
    cfg = PipelineConfig(fs=rec.fs)
    res = tb.run_classical(rec, arm, cfg)
    return {k: v for k, v in res.items() if k not in ("residual", "clean")}


def metric_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    cols = ["method", "F1", "Se", "PPV", "non_coincident_Se", "coincident_Se",
            "pct_coincident", "fhr_mae", "TP", "FP", "FN", "n_coincident_beats",
            "mean_fhr", "stv_ms"]
    return df[[c for c in cols if c in df.columns]]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("fECG test bench")

datasets = tb.available_datasets()
if not datasets:
    st.error("No datasets found under data/. Expected data/cinc2013/set-a, "
             "data/adfecgdb or data/nifecgdb.")
    st.stop()

ds_name = st.sidebar.selectbox("Dataset", list(datasets))
spec = tb.DATASETS[ds_name]
st.sidebar.caption(spec.note)

ckpt_files = sorted(str(p) for p in tb.RESULTS_DIR.glob("*.pth"))
if not ckpt_files:
    st.error(f"No model checkpoint (*.pth) found in {tb.RESULTS_DIR}. "
             "Train one with `python run_full_comparison.py`.")
    st.stop()
ckpt_path = st.sidebar.selectbox("Model checkpoint", ckpt_files,
                                 format_func=lambda p: Path(p).name)
mtime = Path(ckpt_path).stat().st_mtime
model, in_channels = get_model(ckpt_path, mtime)
st.sidebar.caption(f"1D U-Net, expects {in_channels} input channels")

threshold = st.sidebar.slider("Detection threshold", 0.05, 0.95, 0.30, 0.05,
                              help="Peak height on the model's probability trace.")
max_seconds = st.sidebar.slider("Analyse first N seconds", 10, 300, 60, 10)

st.sidebar.divider()
baselines = st.sidebar.multiselect(
    "Classical baselines to compare",
    list(tb.CLASSICAL_ARMS), default=[],
    help="Unsupervised maternal-suppression arms. Slower than the U-Net, "
         "especially bss_ica on long records.")


# --------------------------------------------------------------------------- #
tab_single, tab_batch, tab_sweep, tab_saved = st.tabs(
    ["Single record", "Batch evaluation", "Threshold sweep", "Saved training run"])


# =========================================================================== #
# 1. Single record
# =========================================================================== #
def single_record_view():
    rec_ids = tb.list_records(ds_name)
    default_idx = 0
    if ds_name == "CinC2013 set-a":
        held = [r for r in rec_ids if r in set(tb.test_record_ids())]
        default_idx = rec_ids.index(held[0]) if held else 0

    c1, c2 = st.columns([1, 3])
    rec_id = c1.selectbox("Record", rec_ids, index=default_idx)

    rec_full = get_record(ds_name, rec_id)
    names = tb.channel_names(rec_full)
    usable = [n for n in names if n not in spec.truth_channels]
    default_ch = tb.default_channels(ds_name, names, in_channels)
    ch = c2.multiselect(f"Model input channels (need exactly {in_channels})",
                        usable, default=default_ch)

    label = tb.split_label(ds_name, rec_id)
    ann_bpm = tb.annotation_bpm(rec_full)

    b1, b2, b3 = st.columns(3)
    b1.metric("Split", label)
    b2.metric("Annotation rate", f"{ann_bpm:.0f} bpm" if ann_bpm == ann_bpm else "n/a")
    b3.metric("Duration", f"{rec_full.signal.shape[0] / rec_full.fs:.0f} s")

    if label.startswith("SEEN IN TRAINING"):
        st.warning("This record was in the U-Net's training set. Its score measures "
                   "memorisation, not generalisation — read it as a sanity check only.")
    if ds_name == "ADFECGDB":
        twin = tb.DUPLICATE_OF_CINC.get(rec_id)
        if twin:
            st.info(f"ADFECGDB **{rec_id}** is the same recording as CinC2013 "
                    f"**{twin}** (correlation 1.000, identical annotations), so this "
                    "is not an independent dataset — its split is inherited above.")
    if spec.truth_kind != "fetal":
        st.error(f"{ds_name} annotations mark the **{spec.truth_kind}** QRS, not the "
                 "fetal one. Detection metrics below are therefore NOT fetal accuracy. "
                 "Use this dataset for visual inspection of the probability trace.")
    elif ann_bpm == ann_bpm and ann_bpm < 100:
        st.warning(f"Annotations imply {ann_bpm:.0f} bpm, which is in the maternal "
                   "range. The ground truth for this record may be mislabelled.")
    if spec.truth_channels:
        st.info(f"Channel(s) {', '.join(spec.truth_channels)} are the ground-truth "
                "source (direct fetal lead) and are excluded from model input.")

    if len(ch) != in_channels:
        st.warning(f"Select exactly {in_channels} channels to run the model.")
        return

    rec = tb.subset_record(rec_full, ch, max_seconds)
    cfg = PipelineConfig(fs=rec.fs)
    ch_t = tuple(ch)

    prob = get_probability(ds_name, rec_id, ch_t, float(max_seconds), ckpt_path, mtime)
    dl_peaks = tb.peaks_from_probability(prob, cfg, threshold)
    mat_r = get_maternal(ds_name, rec_id, ch_t, float(max_seconds))

    rows, detections = [], {}
    if rec.fet_r is not None and len(rec.fet_r):
        res = tb.score(rec, dl_peaks, cfg, mat_r=mat_r)
        rows.append(tb.flat_metrics("dl_unet", rec_id, res))
    detections["dl_unet"] = dl_peaks

    for arm in baselines:
        cres = get_classical(ds_name, rec_id, ch_t, float(max_seconds), arm)
        detections[arm] = cres["fet_r"]
        if rec.fet_r is not None and len(rec.fet_r):
            rows.append(tb.flat_metrics(arm, rec_id, tb.score(rec, cres["fet_r"], cfg,
                                                              mat_r=mat_r)))

    st.subheader("Metrics")
    if rows:
        st.dataframe(metric_frame(rows).style.format(
            {c: "{:.3f}" for c in ["F1", "Se", "PPV", "non_coincident_Se",
                                   "coincident_Se", "pct_coincident", "fhr_mae",
                                   "mean_fhr", "stv_ms"]}),
            width="stretch")
        dl_row = rows[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("F1", f"{dl_row['F1']:.3f}")
        k2.metric("Se (non-coincident)", f"{dl_row['non_coincident_Se']:.3f}")
        k3.metric("Se (coincident)", f"{dl_row['coincident_Se']:.3f}",
                  delta=f"{dl_row['coincident_Se'] - dl_row['non_coincident_Se']:+.3f}",
                  help="The headline number: recovery of fetal beats buried under a "
                       "maternal QRS. The delta is the overlap penalty.")
        k4.metric("FHR MAE", f"{dl_row['fhr_mae']:.1f} bpm")
    else:
        st.info("No ground-truth annotations for this record — showing detections only.")

    st.subheader("Signal and detections")
    total_s = rec.signal.shape[0] / rec.fs
    win = st.slider("Window (s)", 0.0, float(total_s), (0.0, min(10.0, float(total_s))),
                    step=0.5)
    a, b = int(win[0] * rec.fs), int(win[1] * rec.fs)
    view_ch = st.selectbox("Channel to display", ch, index=0)
    ci = ch.index(view_ch)

    clean = preprocess(rec.signal, cfg)
    t = np.arange(a, b) / rec.fs

    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1.2]})
    ax = axes[0]
    ax.plot(t, clean[a:b, ci], lw=0.7, color="#333")
    for m in mat_r[(mat_r >= a) & (mat_r < b)]:
        ax.axvline(m / rec.fs, color=MAT, lw=1.0, alpha=0.7)
    if rec.fet_r is not None:
        gt = np.asarray(rec.fet_r)
        gt = gt[(gt >= a) & (gt < b)]
        ax.plot(gt / rec.fs, np.full(len(gt), clean[a:b, ci].max() * 1.05),
                "v", color=TRUTH, ms=7, label="ground truth")
    d = dl_peaks[(dl_peaks >= a) & (dl_peaks < b)]
    ax.plot(d / rec.fs, np.full(len(d), clean[a:b, ci].min() * 1.05),
            "^", color=ACCENT, ms=7, label="detected")
    ax.set_ylabel(f"{view_ch}\n(filtered)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_title(f"{rec_id} — grey verticals = maternal QRS "
                 f"(fetal beats on them are the hard case)", fontsize=10)

    ax = axes[1]
    ax.plot(t, prob[a:b], lw=0.9, color="#1f6feb")
    ax.axhline(threshold, color=ACCENT, ls="--", lw=1, label=f"threshold {threshold}")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("P(fetal R)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    fhr_t, fhr = tb.compute_fhr(dl_peaks, cfg)
    sel = (fhr_t >= win[0]) & (fhr_t <= win[1])
    ax.plot(fhr_t[sel], fhr[sel], "-o", ms=3, lw=1, color=ACCENT, label="detected FHR")
    if rec.fet_r is not None and len(rec.fet_r) > 2:
        gt = np.sort(np.asarray(rec.fet_r))
        gt_t, gt_fhr = tb.compute_fhr(gt, cfg)
        gsel = (gt_t >= win[0]) & (gt_t <= win[1])
        ax.plot(gt_t[gsel], gt_fhr[gsel], "-", lw=1.2, color=TRUTH, alpha=0.8,
                label="reference FHR")
    ax.set_ylabel("FHR (bpm)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Export detections"):
        exp = pd.DataFrame({"sample": dl_peaks, "time_s": dl_peaks / rec.fs})
        st.download_button("Download detected fetal R-peaks (CSV)",
                           exp.to_csv(index=False).encode(),
                           file_name=f"{ds_name.split()[0]}_{rec_id}_fetal_rpeaks.csv",
                           mime="text/csv")


with tab_single:
    single_record_view()


# =========================================================================== #
# 2. Batch evaluation
# =========================================================================== #
with tab_batch:
    st.subheader("Evaluate a set of records")
    if spec.truth_kind != "fetal":
        st.error(f"{ds_name} has {spec.truth_kind} annotations only — batch fetal "
                 "metrics would be meaningless. Pick another dataset.")
    else:
        all_ids = tb.list_records(ds_name)
        if ds_name == "CinC2013 set-a":
            scope = st.radio("Records", ["Held-out test split (honest)",
                                         "All records (includes training data)"],
                             horizontal=True)
            ids = ([r for r in all_ids if r in set(tb.test_record_ids())]
                   if scope.startswith("Held-out") else all_ids)
        elif ds_name == "ADFECGDB":
            seen = [r for r in all_ids
                    if tb.split_label(ds_name, r).startswith("SEEN")]
            ids = [r for r in all_ids if r not in seen]
            st.warning(f"{len(seen)} of {len(all_ids)} ADFECGDB records "
                       f"({', '.join(seen)}) are duplicates of CinC2013 training "
                       "records and are excluded by default.")
        else:
            ids = all_ids
            st.caption("Every record in this dataset is unseen by the model.")

        ids = st.multiselect("Records to run", all_ids, default=ids)
        include_baselines = st.checkbox(
            "Also run the selected classical baselines (much slower)",
            value=False, disabled=not baselines)

        if st.button("Run batch", type="primary") and ids:
            prog = st.progress(0.0, text="Starting...")
            rows, t0 = [], time.time()
            for i, rid in enumerate(ids):
                prog.progress(i / len(ids), text=f"{rid} ({i + 1}/{len(ids)})")
                try:
                    r_full = get_record(ds_name, rid)
                    names_i = tb.channel_names(r_full)
                    ch_i = tb.default_channels(ds_name, names_i, in_channels)
                    if len(ch_i) < in_channels or r_full.fet_r is None:
                        continue
                    r = tb.subset_record(r_full, ch_i, float(max_seconds))
                    cfg_i = PipelineConfig(fs=r.fs)
                    p = get_probability(ds_name, rid, tuple(ch_i), float(max_seconds),
                                        ckpt_path, mtime)
                    m = get_maternal(ds_name, rid, tuple(ch_i), float(max_seconds))
                    pk = tb.peaks_from_probability(p, cfg_i, threshold)
                    row = tb.flat_metrics("dl_unet", rid, tb.score(r, pk, cfg_i, mat_r=m))
                    row["split"] = tb.split_label(ds_name, rid)
                    rows.append(row)
                    if include_baselines:
                        for arm in baselines:
                            c = get_classical(ds_name, rid, tuple(ch_i),
                                              float(max_seconds), arm)
                            br = tb.flat_metrics(arm, rid,
                                                 tb.score(r, c["fet_r"], cfg_i, mat_r=m))
                            br["split"] = tb.split_label(ds_name, rid)
                            rows.append(br)
                except Exception as e:                      # keep the batch going
                    st.warning(f"{rid}: {type(e).__name__}: {e}")
            prog.progress(1.0, text=f"Done in {time.time() - t0:.1f}s")
            st.session_state["batch_rows"] = rows
            st.session_state["batch_ds"] = ds_name

        rows = (st.session_state.get("batch_rows")
                if st.session_state.get("batch_ds") == ds_name else None)
        if rows:
            df = pd.DataFrame(rows)
            st.markdown("**Summary (mean ± std over records)**")
            summary = []
            for meth, g in df.groupby("method"):
                s = tb.summarize(g.to_dict("records"))
                summary.append({
                    "method": meth, "n": s["n_records"],
                    "F1": f"{s['F1_mean']:.3f} ± {s['F1_std']:.3f}",
                    "Se": f"{s['Se_mean']:.3f} ± {s['Se_std']:.3f}",
                    "PPV": f"{s['PPV_mean']:.3f} ± {s['PPV_std']:.3f}",
                    "Se non-coincident": f"{s['non_coincident_Se_mean']:.3f}",
                    "Se coincident": f"{s['coincident_Se_mean']:.3f}",
                    "overlap penalty": f"{s['coincident_Se_mean'] - s['non_coincident_Se_mean']:+.3f}",
                    "FHR MAE (bpm)": f"{s['fhr_mae_mean']:.2f}",
                })
            st.dataframe(pd.DataFrame(summary), width="stretch", hide_index=True)

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            meths = sorted(df["method"].unique())
            axes[0].boxplot([df[df.method == m]["F1"].dropna() for m in meths],
                            tick_labels=meths)
            axes[0].set_title("F1 across records"); axes[0].set_ylim(0, 1)
            axes[0].tick_params(axis="x", rotation=30)

            x = np.arange(len(meths))
            nc = [df[df.method == m]["non_coincident_Se"].mean() for m in meths]
            co = [df[df.method == m]["coincident_Se"].mean() for m in meths]
            axes[1].bar(x - 0.2, nc, 0.4, label="non-coincident", color=MAT)
            axes[1].bar(x + 0.2, co, 0.4, label="coincident", color=ACCENT)
            axes[1].set_xticks(x); axes[1].set_xticklabels(meths, rotation=30)
            axes[1].set_title("Sensitivity: the overlap gap")
            axes[1].set_ylim(0, 1); axes[1].legend(fontsize=8)

            dl = df[df.method == "dl_unet"]
            axes[2].scatter(dl["pct_coincident"], dl["F1"], color=ACCENT)
            for _, r in dl.iterrows():
                axes[2].annotate(r["record"], (r["pct_coincident"], r["F1"]),
                                 fontsize=7, alpha=0.7)
            axes[2].set_xlabel("fraction of beats coincident")
            axes[2].set_ylabel("F1"); axes[2].set_title("Does overlap explain failure?")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("**Per record**")
            st.dataframe(df.sort_values(["method", "F1"]), width="stretch",
                         hide_index=True)
            st.download_button("Download per-record metrics (CSV)",
                               df.to_csv(index=False).encode(),
                               file_name=f"batch_{st.session_state.get('batch_ds','')}"
                                         f"_thr{threshold}.csv".replace(" ", "_"),
                               mime="text/csv")


# =========================================================================== #
# 3. Threshold sweep
# =========================================================================== #
with tab_sweep:
    st.subheader("Where should the detection threshold sit?")
    st.caption("The model outputs a probability trace; the threshold converts it to "
               "beats. Sweeping it separates 'the model cannot see this beat' from "
               "'the threshold is wrong'.")

    if spec.truth_kind != "fetal":
        st.error(f"Needs fetal annotations; {ds_name} has {spec.truth_kind} ones.")
    else:
        all_ids = tb.list_records(ds_name)
        default = ([r for r in all_ids if r in set(tb.test_record_ids())]
                   if ds_name == "CinC2013 set-a" else all_ids)
        sweep_ids = st.multiselect("Records", all_ids, default=default, key="sweep_ids")
        if st.button("Run sweep") and sweep_ids:
            # finer at the low end: that is where Se/PPV actually trade off, and
            # a sweep starting at 0.05 can hit its optimum on the boundary
            thrs = np.round(np.concatenate([
                np.arange(0.01, 0.10, 0.01), np.arange(0.10, 0.96, 0.05)]), 3)
            acc = {t: [] for t in thrs}
            prog = st.progress(0.0)
            for i, rid in enumerate(sweep_ids):
                prog.progress(i / len(sweep_ids), text=rid)
                r_full = get_record(ds_name, rid)
                ch_i = tb.default_channels(ds_name, tb.channel_names(r_full), in_channels)
                if len(ch_i) < in_channels or r_full.fet_r is None:
                    continue
                r = tb.subset_record(r_full, ch_i, float(max_seconds))
                cfg_i = PipelineConfig(fs=r.fs)
                p = get_probability(ds_name, rid, tuple(ch_i), float(max_seconds),
                                    ckpt_path, mtime)
                m = get_maternal(ds_name, rid, tuple(ch_i), float(max_seconds))
                for t_ in thrs:
                    pk = tb.peaks_from_probability(p, cfg_i, float(t_))
                    acc[t_].append(tb.flat_metrics("dl_unet", rid,
                                                   tb.score(r, pk, cfg_i, mat_r=m)))
            prog.progress(1.0, text="Done")
            st.session_state["sweep"] = {
                float(t): tb.summarize(v) for t, v in acc.items() if v}

        sweep = st.session_state.get("sweep")
        if sweep:
            ts = sorted(sweep)
            f1 = [sweep[t]["F1_mean"] for t in ts]
            se = [sweep[t]["Se_mean"] for t in ts]
            ppv = [sweep[t]["PPV_mean"] for t in ts]
            cse = [sweep[t]["coincident_Se_mean"] for t in ts]
            best = ts[int(np.nanargmax(f1))]

            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.plot(ts, f1, "-o", ms=4, color=ACCENT, label="F1")
            ax.plot(ts, se, "-", color="#1f6feb", label="Se")
            ax.plot(ts, ppv, "-", color=TRUTH, label="PPV")
            ax.plot(ts, cse, "--", color="#8250df", label="Se (coincident beats)")
            ax.axvline(best, color="#333", ls=":", lw=1)
            ax.annotate(f"best F1 @ {best:.2f}", (best, max(f1)), fontsize=9,
                        xytext=(5, -12), textcoords="offset points")
            ax.axvline(threshold, color=MAT, ls="-", lw=1, alpha=0.6)
            ax.set_xlabel("threshold"); ax.set_ylabel("score"); ax.set_ylim(0, 1)
            ax.legend(fontsize=9); ax.grid(alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.metric("Best mean F1", f"{max(f1):.3f}", f"at threshold {best:.3f}")
            interior = [v for t_, v in zip(ts, f1) if t_ not in (ts[0], ts[-1])]
            if best in (ts[0], ts[-1]) and max(f1) - max(interior) > 0.002:
                st.warning("The optimum landed on the edge of the swept range — the "
                           "true best threshold is outside it, so treat this value "
                           "as a lower/upper bound, not a recommendation.")
            elif max(f1) - min(f1) < 0.02:
                st.info("F1 is flat across the whole range — this model's ranking of "
                        "beats is what limits it, not where the threshold sits. "
                        "Pick the threshold from the Se/PPV trade-off you want.")
            st.caption(f"Currently applied threshold: {threshold:.2f} "
                       f"(F1 {np.interp(threshold, ts, f1):.3f}). Lowering it trades "
                       "PPV for recovery of coincident beats.")
            st.dataframe(pd.DataFrame({
                "threshold": ts, "F1": f1, "Se": se, "PPV": ppv,
                "Se_coincident": cse}).style.format("{:.3f}"),
                width="stretch", hide_index=True)


# =========================================================================== #
# 4. Saved training run
# =========================================================================== #
with tab_saved:
    run = tb.saved_run()
    if not run:
        st.info("No results/comparison_metrics.json yet — run "
                "`python run_full_comparison.py` to produce one.")
    else:
        st.subheader("Metrics from the last run_full_comparison.py")
        c1, c2, c3 = st.columns(3)
        c1.metric("Train records", run.get("n_train", "?"))
        c2.metric("Test records", run.get("n_test", "?"))
        c3.metric("Selected model", run.get("best_model", "?"))

        summ = pd.DataFrame(run["summary"]).T
        show = ["F1_mean", "Se_mean", "PPV_mean", "non_coincident_Se_mean",
                "coincident_Se_mean", "fhr_mae_mean"]
        st.dataframe(summ[show].style.format("{:.3f}"), width="stretch")
        st.caption("Held-out test records: " + ", ".join(run.get("test_records", [])))

        if run.get("loss_history"):
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(range(1, len(run["loss_history"]) + 1), run["loss_history"],
                    "-o", ms=3, color=ACCENT)
            ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss")
            ax.set_title("U-Net training loss"); ax.grid(alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        figs = sorted(tb.RESULTS_DIR.glob("fig_*.png"))
        if figs:
            st.subheader("Saved figures")
            pick = st.selectbox("Figure", [f.name for f in figs])
            st.image(str(tb.RESULTS_DIR / pick), width="stretch")
