"""Attention U-Net 2D + deep supervision.

Wrapper esplicito su `UNet2D(attention=True)`: condivide l'implementazione (stessi
blocchi, stessa inizializzazione) così il confronto con la baseline isola davvero
l'effetto degli Attention Gate. La classe `AttentionGate` è definita in `unet.py`.
"""
from __future__ import annotations
from src.models.unet import UNet2D, AttentionGate  # noqa: F401  (riesportata)


def AttentionUNet2D(in_channels: int, out_channels: int = 1,
                    features=(32, 64, 128, 256, 512),
                    deep_supervision: bool = False, dropout: float = 0.0) -> UNet2D:
    """Costruisce la Attention U-Net 2D."""
    return UNet2D(in_channels=in_channels, out_channels=out_channels, features=features,
                  attention=True, deep_supervision=deep_supervision, dropout=dropout)
