"""Diffusion morphology layer (optional, requires torch). Lazy imports keep the
numpy-only recovery core usable without torch installed."""

def __getattr__(name):
    if name == "DiffusionUNet1D":
        from .model import DiffusionUNet1D
        return DiffusionUNet1D
    if name == "DDPM":
        from .inpaint import DDPM
        return DDPM
    if name in ("train_diffusion", "FetalBeatDataset"):
        from . import train
        return getattr(train, name)
    raise AttributeError(name)
