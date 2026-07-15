"""Valutazione sul test set, complessiva e per sottogruppo.  [Impl.: Fasi 5-7]

- Ricostruisce le predizioni per volume (aggregando le slice 2.5D) e calcola le
  metriche di metrics.py.
- Aggrega i risultati per sottogruppo (sesso, età, field strength) -> tabella pandas/CSV.
- Supporta la valutazione cross-dominio (train_domain -> test_domain) per la Fase 6.
"""
from __future__ import annotations


def evaluate_model(cfg, model, test_loader, cases):
    """Metriche complessive + per sottogruppo. Fasi 5-7."""
    raise NotImplementedError
