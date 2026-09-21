"""Minimal smoke tests: every classical arm runs and evaluation is well-formed.
Run with:  python -m pytest -q   (or)   python tests/test_smoke.py
"""
import os
import sys
import warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fecg import (PipelineConfig, synthesize_record, run_pipeline,
                  TemplateSubtraction, ICASuppressor, AdaptiveSuppressor,
                  KalmanTemplateSuppressor)

CFG = PipelineConfig()
ARMS = [TemplateSubtraction, lambda c: ICASuppressor(c),
        lambda c: AdaptiveSuppressor(c, method="rls"), KalmanTemplateSuppressor]


def _make():
    return synthesize_record(CFG, duration_s=15.0, overlap_frac=0.2, seed=7)


def test_arms_run_and_evaluate():
    rec = _make()
    for arm in ARMS:
        supp = arm(CFG)
        out = run_pipeline(rec, supp, CFG)
        assert "eval" in out
        ev = out["eval"]["overall"]
        assert 0.0 <= ev["Se"] <= 1.0 and 0.0 <= ev["PPV"] <= 1.0
        assert 0.0 <= ev["F1"] <= 1.0


def test_coincidence_labels_present():
    rec = _make()
    out = run_pipeline(rec, TemplateSubtraction(CFG), CFG)
    ev = out["eval"]
    assert ev["coincident"]["n_ref"] + ev["non_coincident"]["n_ref"] == len(rec.fet_r)
    assert 0.0 <= ev["pct_beats_coincident"] <= 1.0


if __name__ == "__main__":
    test_arms_run_and_evaluate()
    test_coincidence_labels_present()
    print("all smoke tests passed")


def test_recovery_runs_and_is_safe():
    """Recovery pipeline runs and never wrecks precision on clean synthetic data."""
    from fecg.recovery import run_recovery_pipeline, OcclusionAwareRecovery
    rec = _make()
    out = run_recovery_pipeline(rec, CFG, OcclusionAwareRecovery(CFG))
    assert "eval" in out
    assert out["eval"]["overall"]["PPV"] >= 0.85          # doesn't invent many false beats
    assert out["n_recovered"] >= 0


def test_rhythm_prior_recovers_injected_dropout():
    """Core claim: knock out coincident beats, rhythm prior puts most back, few FPs."""
    import numpy as np
    from fecg.recovery.occlusion import occlusion_mask
    from fecg.recovery.rhythm_prior import predict_hidden_beats
    rec = synthesize_record(CFG, duration_s=40.0, overlap_frac=0.25, seed=3)
    gt, mat_r = np.sort(rec.fet_r), np.sort(rec.mat_r)
    n = rec.signal.shape[0]
    mask = occlusion_mask(n, mat_r, CFG)
    win = CFG.coincidence_win_s * CFG.fs
    tol = int(CFG.match_tol_s * CFG.fs)
    coin = np.array([f for f in gt if np.min(np.abs(mat_r - f)) <= win], int)
    observed = np.array([f for f in gt if f not in set(coin.tolist())], int)  # drop all coincident
    pred = predict_hidden_beats(observed, mat_r, mask, CFG)
    if len(coin) >= 5:
        rec_rate = np.mean([np.min(np.abs(pred - c)) <= tol if len(pred) else False for c in coin])
        false = sum(1 for p in pred if np.min(np.abs(gt - p)) > tol)
        assert rec_rate >= 0.5           # recovers at least half the knocked-out beats
        assert false <= 0.2 * len(pred) + 1   # few false insertions
