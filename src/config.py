"""Caricamento e gestione della configurazione (OmegaConf).

Uso tipico:
    from src.config import load_config
    cfg = load_config("configs/exp_unet_flair.yaml",
                      overrides=["train.lr=0.0005", "train.epochs=120"])

Ogni file d'esperimento sovrascrive `configs/base.yaml`; gli `overrides` da CLI
(sintassi dotlist, es. "train.lr=0.0005") hanno priorità massima.
"""
from __future__ import annotations
import os
from omegaconf import OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG = os.path.normpath(os.path.join(_HERE, "..", "configs", "base.yaml"))


def load_config(exp_config: str | None = None, overrides: list[str] | None = None):
    """Carica base.yaml, applica l'esperimento e gli override, poi finalizza."""
    cfg = OmegaConf.load(BASE_CONFIG)
    if exp_config and os.path.abspath(exp_config) != BASE_CONFIG:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(exp_config))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    _finalize(cfg)
    return cfg


def _finalize(cfg) -> None:
    """Risolve i campi derivati e crea la cartella di output."""
    # in_channels: auto = n_modalità * (2*context_slices + 1)
    if str(cfg.model.in_channels).lower() == "auto":
        n_mod = len(cfg.data.modalities)
        ctx = 2 * int(cfg.data.context_slices) + 1
        cfg.model.in_channels = n_mod * ctx
    os.makedirs(cfg.project.output_dir, exist_ok=True)


def save_config(cfg, path: str) -> None:
    """Salva la config risolta accanto ai risultati (riproducibilità)."""
    with open(path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
