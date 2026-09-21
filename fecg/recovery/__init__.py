"""Occlusion-aware fetal beat recovery — the novel project arm.

Core (numpy only):
    from fecg.recovery import OcclusionAwareRecovery, run_recovery_pipeline
Diffusion morphology layer (optional, torch):
    from fecg.recovery.diffusion import DiffusionUNet1D, DDPM, train_diffusion
"""
from .occlusion import occlusion_mask, reliability_weights, occluded_intervals
from .rhythm_prior import predict_hidden_beats, robust_base_rr
from .gated_recovery import OcclusionAwareRecovery, RecoveryResult
from .pipeline import run_recovery_pipeline

__all__ = [
    "occlusion_mask", "reliability_weights", "occluded_intervals",
    "predict_hidden_beats", "robust_base_rr",
    "OcclusionAwareRecovery", "RecoveryResult", "run_recovery_pipeline",
]
