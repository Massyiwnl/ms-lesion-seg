"""Factory dei modelli dal config.  [Implementazione: Fase 3]

Mappa `model.arch` -> costruttore:
    unet            -> UNet2D
    attention_unet  -> AttentionUNet2D
    dynunet         -> monai.networks.nets.DynUNet   (deep supervision nativa)
    swin_unetr      -> monai.networks.nets.SwinUNETR (stretch, Fase 8)
"""
from __future__ import annotations


def build_model(cfg):
    """Istanzia il modello secondo cfg.model. Fase 3."""
    raise NotImplementedError
