"""Deep-learning arm. Imports are lazy so the classical pipeline runs without torch.

Usage:
    from fecg.dl import UNet1D, WindowDataset, train_model, DLDetector
"""

def __getattr__(name):
    if name in ("UNet1D",):
        from .model import UNet1D
        return UNet1D
    if name in ("WindowDataset",):
        from .dataset import WindowDataset
        return WindowDataset
    if name in ("train_model", "DLDetector"):
        from . import train
        return getattr(train, name)
    raise AttributeError(name)
