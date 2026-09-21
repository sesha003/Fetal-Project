"""Shared interface for maternal-suppression method arms.

Every arm (template, adaptive, BSS, Kalman, DL) implements `.suppress(...)` and
returns a residual of the same shape as the input, so they all plug into the same
pipeline and the same overlap-stratified evaluation harness. That uniformity is
what makes the method comparison apples-to-apples.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..config import PipelineConfig


class Suppressor(ABC):
    name: str = "base"

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    @abstractmethod
    def suppress(self, clean: np.ndarray, mat_r: Optional[np.ndarray] = None) -> np.ndarray:
        """Remove the maternal component. `clean` is preprocessed (n_samples, n_channels).

        `mat_r` (maternal R-peak samples) is supplied for arms that need it
        (template, adaptive, Kalman) and ignored by arms that don't (BSS).
        """
        raise NotImplementedError

    @staticmethod
    def _as_2d(sig: np.ndarray) -> np.ndarray:
        sig = np.atleast_2d(sig)
        if sig.shape[0] < sig.shape[1]:
            sig = sig.T
        return sig
