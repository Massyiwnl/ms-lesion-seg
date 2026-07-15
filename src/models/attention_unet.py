"""Attention U-Net 2D + deep supervision.  [Implementazione: Fase 3]

- Attention Gate sugli skip (Oktay et al. 2018): il decoder "pesa" le feature
  dell'encoder prima della concatenazione.
- Deep supervision: teste di segmentazione ausiliarie a più livelli del decoder;
  in training la loss somma i contributi (pesi in loss.deep_supervision_weights),
  in inference si usa solo l'uscita a piena risoluzione.
"""
from __future__ import annotations


def AttentionUNet2D(in_channels: int, out_channels: int = 1,
                    features=(32, 64, 128, 256, 512),
                    deep_supervision: bool = False, dropout: float = 0.0):
    """Costruisce la Attention U-Net 2D. Fase 3."""
    raise NotImplementedError
