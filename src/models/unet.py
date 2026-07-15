"""U-Net 2D parametrica (baseline).  [Implementazione: Fase 3]

Rielaborazione pulita e generalizzata della U-Net del lab CAE+U-Net:
- in_channels configurabile (per la fusione multimodale),
- larghezze encoder da `model.features`,
- blocchi Conv-BN-ReLU x2, down = maxpool, up = convtranspose + skip concat.
"""
from __future__ import annotations


def UNet2D(in_channels: int, out_channels: int = 1, features=(32, 64, 128, 256, 512),
           dropout: float = 0.0):
    """Costruisce la U-Net 2D baseline. Fase 3."""
    raise NotImplementedError
