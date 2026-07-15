"""Regolarizzazione dello Jacobiano (robustezza / Lipschitz).  [Impl.: Fase 6]

Penalizza ||dOutput/dInput||_F, stimata con il metodo di Hutchinson (proiezioni
casuali) come in Hoffman et al. 2019 — lo stesso approccio del paper PJI del corso.
Aggiunta come termine di loss (peso `regularization.jacobian_lambda`) per ridurre
la sensibilità del modello a perturbazioni/artefatti e migliorare il transfer 3T<->1.5T.
"""
from __future__ import annotations


def jacobian_regularization(logits, inputs, n_proj: int = 1):
    """Stima della norma del Jacobiano tramite proiezioni casuali. Fase 6."""
    raise NotImplementedError
