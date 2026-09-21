"""Central configuration for the fECG extraction pipeline."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    fs: float = 1000.0                      # sampling rate (Hz); ADFECGDB/CinC2013 = 1000

    # --- preprocessing ---
    bp_low: float = 1.0                     # bandpass low cutoff (Hz) -> removes baseline wander
    bp_high: float = 100.0                  # bandpass high cutoff (Hz)
    powerline: float = 50.0                 # notch (Hz): 50 EU, 60 US
    notch_q: float = 30.0

    # --- maternal QRS detection ---
    mat_bp_low: float = 8.0
    mat_bp_high: float = 20.0
    mat_refractory_s: float = 0.25          # min maternal RR (~240 bpm ceiling)

    # --- maternal suppression (shared) ---
    template_half_win_s: float = 0.12       # +/- window around maternal R

    # adaptive filter (LMS/RLS)
    adaptive_order: int = 8
    lms_mu: float = 0.01
    rls_lambda: float = 0.999
    rls_delta: float = 1.0

    # --- fetal QRS detection ---
    fet_bp_low: float = 15.0
    fet_bp_high: float = 70.0
    fhr_min_bpm: float = 90.0
    fhr_max_bpm: float = 200.0

    # --- evaluation ---
    match_tol_s: float = 0.05               # +/-50 ms QRS matching (CinC 2013 standard)
    coincidence_win_s: float = 0.05         # fetal R within this of a maternal R => "coincident"
