"""Metriche di segmentazione.  [Implementazione: Fase 4]

- Dice, IoU (MeanIoU), HD95 (HausdorffDistanceMetric percentile=95),
  ASSD (SurfaceDistanceMetric) da MONAI.
- Lesion-wise: via componenti connesse (scipy.ndimage.label) su GT e predizione,
  con matching per overlap -> TPR (sensibilità per-lesione), FP per volume,
  ignorando componenti < eval.min_lesion_size.
"""
from __future__ import annotations


def compute_overlap_metrics(pred, target, cfg):
    """Dice/IoU/HD95/ASSD su un batch o volume. Fase 4."""
    raise NotImplementedError


def lesion_wise_metrics(pred_mask, gt_mask, min_size: int = 3):
    """TPR per-lesione e falsi positivi per volume. Fase 4."""
    raise NotImplementedError
