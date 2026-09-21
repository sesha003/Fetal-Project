"""Maternal-suppression method arms, all sharing the Suppressor interface."""
from .base import Suppressor
from .template import TemplateSubtraction
from .bss import ICASuppressor
from .adaptive import AdaptiveSuppressor
from .kalman import KalmanTemplateSuppressor

__all__ = [
    "Suppressor",
    "TemplateSubtraction",
    "ICASuppressor",
    "AdaptiveSuppressor",
    "KalmanTemplateSuppressor",
]
