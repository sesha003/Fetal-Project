"""
testbench.py — dataset registry, model loading and evaluation helpers for app.py.

Kept separate from the Streamlit UI so every number the app shows can also be
reproduced from a plain Python session:

    from testbench import load_record, load_model, run_dl, run_classical
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from fecg import (PipelineConfig, Record, load_physionet_record, load_edf_record,
                  run_pipeline, TemplateSubtraction, ICASuppressor,
                  AdaptiveSuppressor, KalmanTemplateSuppressor)
from fecg.evaluate import evaluate, fhr_error
from fecg.fetal import compute_fhr, compute_hrv
from fecg.preprocess import preprocess, detect_maternal_qrs

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DEFAULT_CKPT = RESULTS_DIR / "dl_unet.pth"
METRICS_JSON = RESULTS_DIR / "comparison_metrics.json"


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
@dataclass
class DatasetSpec:
    key: str
    dir: str
    kind: str                 # "wfdb" | "edf"
    ann_ext: str
    fs: float
    note: str
    truth_kind: str = "fetal"      # what the annotation files actually mark
    # channels that must never reach a detector (they *are* the ground truth)
    truth_channels: tuple = ()


DATASETS = {
    "CinC2013 set-a": DatasetSpec(
        key="cinc2013", dir="data/cinc2013/set-a", kind="wfdb", ann_ext="fqrs",
        fs=1000.0,
        note="4 abdominal leads, 60 s, 1000 Hz. This is the dataset the U-Net was "
             "trained on — only the held-out test records are an honest score."),
    "ADFECGDB": DatasetSpec(
        key="adfecgdb", dir="data/adfecgdb", kind="edf", ann_ext="qrs", fs=1000.0,
        note="4 abdominal leads + 1 direct fetal scalp lead, 5 min, 1000 Hz. "
             "NOT an independent test set: every record here is duplicated inside "
             "CinC2013 set-a, and 3 of the 5 were in the U-Net's training split.",
        truth_channels=("Direct_1",)),
    "NIFECGDB": DatasetSpec(
        key="nifecgdb", dir="data/nifecgdb", kind="edf", ann_ext="qrs", fs=1000.0,
        truth_kind="maternal",
        note="2 thoracic + 3 abdominal leads, 1000 Hz. WARNING: this database's "
             ".qrs files annotate the MATERNAL QRS (~83 bpm here), not the fetal "
             "one. Detection metrics against them are not fetal metrics — use this "
             "dataset for visual inspection only."),
}


def available_datasets() -> dict:
    return {k: v for k, v in DATASETS.items() if (ROOT / v.dir).is_dir()}


def list_records(ds_name: str) -> list[str]:
    spec = DATASETS[ds_name]
    d = ROOT / spec.dir
    if spec.kind == "wfdb":
        return sorted(os.path.basename(f)[:-4] for f in glob.glob(str(d / "*.hea")))
    return sorted(os.path.basename(f)[:-4] for f in glob.glob(str(d / "*.edf")))


def load_record(ds_name: str, rec_id: str) -> Record:
    spec = DATASETS[ds_name]
    path = str(ROOT / spec.dir / rec_id)
    if spec.kind == "wfdb":
        rec = load_physionet_record(path, ann_ext=spec.ann_ext, name=rec_id)
        rec.channel_names = [f"AECG{i + 1}" for i in range(rec.signal.shape[1])]
        return rec
    return load_edf_record(path + ".edf", ann_ext=spec.ann_ext, name=rec_id)


def channel_names(rec: Record) -> list[str]:
    return list(getattr(rec, "channel_names",
                        [f"ch{i + 1}" for i in range(rec.signal.shape[1])]))


def default_channels(ds_name: str, names: list[str], n_needed: int) -> list[str]:
    """Pick sensible model-input channels: abdominal leads first, never a truth lead."""
    spec = DATASETS[ds_name]
    usable = [n for n in names if n not in spec.truth_channels]
    abd = [n for n in usable if "abdomen" in n.lower() or n.lower().startswith("aecg")]
    ordered = abd + [n for n in usable if n not in abd]
    return ordered[:n_needed]


def subset_record(rec: Record, ch_names: list[str], max_seconds: float | None) -> Record:
    """Select channels and (optionally) trim duration, keeping annotations consistent."""
    names = channel_names(rec)
    idx = [names.index(c) for c in ch_names]
    sig = rec.signal[:, idx]
    n = sig.shape[0]
    if max_seconds:
        n = min(n, int(max_seconds * rec.fs))
    sig = sig[:n]

    def _trim(a):
        return None if a is None else np.asarray(a)[np.asarray(a) < n]

    out = Record(signal=sig, fs=rec.fs, name=rec.name,
                 mat_r=_trim(rec.mat_r), fet_r=_trim(rec.fet_r))
    out.channel_names = list(ch_names)
    return out


# --------------------------------------------------------------------------- #
# Train / test provenance — which records the U-Net actually saw
# --------------------------------------------------------------------------- #
def saved_run() -> dict | None:
    if METRICS_JSON.exists():
        with open(METRICS_JSON) as f:
            return json.load(f)
    return None


def test_record_ids() -> list[str]:
    """Held-out CinC2013 records from the saved training run (falls back to the
    same deterministic split run_full_comparison.py uses)."""
    run = saved_run()
    if run and run.get("test_records"):
        return list(run["test_records"])
    ids = list_records("CinC2013 set-a")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ids))[:15]
    return sorted(ids[i] for i in idx)


# ADFECGDB records are re-published inside CinC2013 set-a. Verified by correlating
# every ADFECGDB abdominal lead against every CinC2013 record: each pair matches at
# r = 1.000 with identical annotation samples. Without this map, evaluating on
# ADFECGDB looks like a cross-dataset generalisation test while 3 of the 5 records
# were actually in the training split.
DUPLICATE_OF_CINC = {
    "r01": "a05", "r04": "a25", "r07": "a19", "r08": "a17", "r10": "a03",
}


def split_label(ds_name: str, rec_id: str) -> str:
    """'held-out test' / 'SEEN IN TRAINING' / 'unseen dataset'."""
    test = set(test_record_ids())
    if ds_name == "CinC2013 set-a":
        return "held-out test" if rec_id in test else "SEEN IN TRAINING"
    if ds_name == "ADFECGDB":
        twin = DUPLICATE_OF_CINC.get(rec_id)
        if twin is None:
            return "unseen dataset"
        return (f"held-out test (= {twin})" if twin in test
                else f"SEEN IN TRAINING (= {twin})")
    return "unseen dataset"


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def checkpoint_in_channels(ckpt_path: str) -> int:
    import torch
    sd = torch.load(ckpt_path, map_location="cpu")
    for k, v in sd.items():
        if k.endswith("downs.0.net.0.weight") or k == "downs.0.net.0.weight":
            return int(v.shape[1])
    raise ValueError("could not infer in_channels from checkpoint")


def load_model(ckpt_path: str):
    import torch
    from fecg.dl import UNet1D
    in_ch = checkpoint_in_channels(ckpt_path)
    model = UNet1D(in_channels=in_ch)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model, in_ch


# --------------------------------------------------------------------------- #
# Inference + evaluation
# --------------------------------------------------------------------------- #
def dl_probability(model, rec: Record, cfg: PipelineConfig, win_s: float = 2.0) -> np.ndarray:
    from fecg.dl import DLDetector
    return DLDetector(model, cfg, win_s=win_s).probability(rec)


def peaks_from_probability(prob: np.ndarray, cfg: PipelineConfig, threshold: float) -> np.ndarray:
    refractory = int((60.0 / cfg.fhr_max_bpm) * cfg.fs)
    peaks, _ = find_peaks(prob, height=threshold, distance=refractory)
    return peaks


def score(rec: Record, fet_r: np.ndarray, cfg: PipelineConfig,
          mat_r: np.ndarray | None = None) -> dict:
    """Overlap-stratified metrics + FHR error + HRV for one detection set."""
    if mat_r is None:
        mat_r = detect_maternal_qrs(preprocess(rec.signal, cfg), cfg)
    ref_mat = rec.mat_r if rec.mat_r is not None else mat_r
    ev = evaluate(fet_r, rec.fet_r, ref_mat, cfg)
    fhr_t, fhr = compute_fhr(fet_r, cfg)
    fe = fhr_error(fhr_t, fhr, rec.fet_r, cfg)
    return dict(eval=ev, fhr_t=fhr_t, fhr=fhr, fhr_error=fe,
                hrv=compute_hrv(fet_r, cfg), mat_r=mat_r, fet_r=fet_r)


def flat_metrics(name: str, rec_name: str, res: dict) -> dict:
    ev = res["eval"]
    return {
        "record": rec_name, "method": name,
        "F1": ev["overall"]["F1"], "Se": ev["overall"]["Se"], "PPV": ev["overall"]["PPV"],
        "coincident_Se": ev["coincident"]["Se"],
        "non_coincident_Se": ev["non_coincident"]["Se"],
        "pct_coincident": ev["pct_beats_coincident"],
        "fhr_mae": res["fhr_error"].get("mae", np.nan),
        "TP": ev["overall"]["TP"], "FP": ev["overall"]["FP"], "FN": ev["overall"]["FN"],
        "n_coincident_beats": ev["coincident"]["n_ref"],
        "mean_fhr": res["hrv"]["mean_fhr"], "stv_ms": res["hrv"]["stv_ms"],
    }


CLASSICAL_ARMS = {
    "template": lambda c: TemplateSubtraction(c),
    "adaptive": lambda c: AdaptiveSuppressor(c, method="rls"),
    "bss_ica": lambda c: ICASuppressor(c, pre_template=True),
    "kalman": lambda c: KalmanTemplateSuppressor(c),
}


def run_classical(rec: Record, arm: str, cfg: PipelineConfig) -> dict:
    out = run_pipeline(rec, CLASSICAL_ARMS[arm](cfg), cfg)
    res = dict(eval=out.get("eval"), fhr_t=out["fhr_t"], fhr=out["fhr"],
               fhr_error=out.get("fhr_error", {}), hrv=out["hrv"],
               mat_r=out["mat_r"], fet_r=out["fet_r"], residual=out["residual"],
               clean=out["clean"])
    return res


def summarize(rows: list[dict]) -> dict:
    """Mean ± std over records, NaN-safe."""
    keys = ["F1", "Se", "PPV", "coincident_Se", "non_coincident_Se", "fhr_mae"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        out[f"{k}_mean"] = float(vals.mean()) if len(vals) else float("nan")
        out[f"{k}_std"] = float(vals.std()) if len(vals) else float("nan")
    out["n_records"] = len(rows)
    return out


def annotation_bpm(rec: Record) -> float:
    """Mean beat rate implied by a record's annotation file — a cheap sanity check
    that the 'ground truth' really is fetal (~110-160 bpm) and not maternal (~60-90)."""
    if rec.fet_r is None or len(rec.fet_r) < 3:
        return float("nan")
    rr = np.diff(np.sort(np.asarray(rec.fet_r))) / rec.fs
    rr = rr[(rr > 0.2) & (rr < 2.0)]
    return float(60.0 / np.median(rr)) if len(rr) else float("nan")
