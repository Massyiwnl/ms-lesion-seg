"""Explainability: Grad-CAM e mappe delle attention gate.  [Impl.: Fase 7]

- Grad-CAM (monai.visualize.GradCAM/GradCAMpp) su uno strato conv scelto
  (explain.gradcam_layer) per evidenziare le regioni che guidano la predizione.
- Per la Attention U-Net: hook sui coefficienti delle attention gate per visualizzare
  "dove guarda" il modello.
- Error analysis qualitativa: overlay TP/FP/FN e focus sulle lesioni piccole.
"""
from __future__ import annotations


def gradcam_maps(model, inputs, cfg):
    """Mappe Grad-CAM per un batch di esempi. Fase 7."""
    raise NotImplementedError
