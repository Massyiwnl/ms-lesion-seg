"""Utility comuni: seeding, device, logging, conteggio parametri."""
from __future__ import annotations
import os, sys, random, logging
import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Fissa i seed di random/numpy/torch per la riproducibilità."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger(name: str = "mslesseg", logfile: str | None = None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
