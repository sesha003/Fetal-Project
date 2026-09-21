"""Occlusion model: turn maternal R-peaks into an occlusion mask + reliability weights.

This is the conceptual heart of the project. The maternal QRS occludes the fetal
beat at known, predictable locations. So instead of trusting every sample equally,
we mark the samples under each maternal QRS as *unreliable* and down-weight the
observation there. Downstream stages (rhythm prediction, diffusion inpainting) use
this to decide where to trust the signal and where to reconstruct.
"""
from __future__ import annotations

import numpy as np

from ..config import PipelineConfig


def occlusion_mask(n_samples: int, mat_r: np.ndarray, cfg: PipelineConfig,
                   half_width_s: float | None = None) -> np.ndarray:
    """Boolean mask, True where a maternal QRS occludes the fetal signal.

    Width defaults to the maternal QRS extent (~template half-window). These are the
    samples where the observation is a liar about the fetal beat.
    """
    hw = int((half_width_s if half_width_s is not None
              else cfg.template_half_win_s) * cfg.fs)
    mask = np.zeros(n_samples, dtype=bool)
    for r in np.asarray(mat_r, int):
        a, b = max(0, r - hw), min(n_samples, r + hw)
        mask[a:b] = True
    return mask


def reliability_weights(n_samples: int, mat_r: np.ndarray, cfg: PipelineConfig,
                        floor: float = 0.02, soft_s: float = 0.03) -> np.ndarray:
    """Per-sample reliability in [floor, 1].

    ~1 away from maternal QRS, dropping to `floor` at each maternal R-peak, with a
    smooth (Gaussian) shoulder so the transition isn't a hard cliff. This is the
    "gated measurement model": the weight the observation gets in any fusion.
    """
    w = np.ones(n_samples, dtype=float)
    sigma = max(1.0, soft_s * cfg.fs)
    idx = np.arange(n_samples)
    for r in np.asarray(mat_r, int):
        dip = (1.0 - floor) * np.exp(-0.5 * ((idx - r) / sigma) ** 2)
        w = np.minimum(w, 1.0 - dip)
    return np.clip(w, floor, 1.0)


def occluded_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) sample ranges of contiguous occluded regions."""
    if not mask.any():
        return []
    edges = np.diff(mask.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]
    return list(zip(starts, ends))
