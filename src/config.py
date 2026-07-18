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
CONFIG_DIR = os.path.normpath(os.path.join(_HERE, "..", "configs"))
BASE_CONFIG = os.path.join(CONFIG_DIR, "base.yaml")
# Config locale (path della macchina): NON versionata, sovrascrive base + esperimento.
LOCAL_CONFIG = os.environ.get("MSLESSEG_LOCAL_CONFIG",
                              os.path.join(CONFIG_DIR, "local.yaml"))


def load_config(exp_config: str | None = None, overrides: list[str] | None = None):
    """Carica la configurazione con questa precedenza (dal più debole al più forte):

        configs/base.yaml  ->  config d'esperimento  ->  configs/local.yaml  ->  override CLI

    `configs/local.yaml` contiene i path (e le impostazioni hardware) specifici della
    macchina ed è in .gitignore: così aggiornare i file del repo non sovrascrive mai la
    tua configurazione locale, e lo stesso codice gira su PC e su Colab senza modifiche.
    """
    cfg = OmegaConf.load(BASE_CONFIG)
    if exp_config and os.path.abspath(exp_config) != BASE_CONFIG:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(exp_config))
    if os.path.exists(LOCAL_CONFIG):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(LOCAL_CONFIG))
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
