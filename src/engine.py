"""Loop di training e validazione.  [Implementazione: Fase 4]

- AMP (autocast + GradScaler) per la T4, optimizer AdamW, scheduler cosine/plateau.
- Deep supervision gestita dal criterio di loss; opz. termine Jacobiano (Fase 6).
- Validazione periodica con Dice; early stopping + salvataggio del best checkpoint.
- Logging di loss/metriche su CSV (+ opz. TensorBoard) e salvataggio della config risolta.
"""
from __future__ import annotations


def train_one_epoch(model, loader, criterion, optimizer, device, cfg, scaler=None):
    raise NotImplementedError


def validate(model, loader, device, cfg):
    raise NotImplementedError


def fit(cfg, model, train_loader, val_loader):
    """Ciclo completo di addestramento. Fase 4."""
    raise NotImplementedError
