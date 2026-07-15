"""Funzioni di costo e wrapper deep supervision.  [Implementazione: Fase 4]

- dice        -> monai.losses.DiceLoss(sigmoid=True)
- dice_focal  -> monai.losses.DiceFocalLoss(sigmoid=True)   (sbilanciamento)
- tversky     -> monai.losses.TverskyLoss(alpha, beta)      (controllo FN/FP)
- DeepSupervisionLoss: somma pesata della loss su output principale + ausiliari,
  con la GT ridimensionata alla risoluzione di ciascuna testa.
"""
from __future__ import annotations


def build_loss(cfg):
    """Ritorna il criterio di loss secondo cfg.loss. Fase 4."""
    raise NotImplementedError
