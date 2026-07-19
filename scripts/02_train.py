"""Addestramento di un esperimento (Fase 4).

    python scripts/02_train.py --config configs/exp_unet_flair.yaml
    python scripts/02_train.py --config configs/exp_attunet_multimodal_ds.yaml \
           --override train.epochs=60 train.batch_size=8

Split a livello di paziente (nessun timepoint dello stesso paziente finisce sia in
train sia in validazione) e selezione del modello sul Dice per volume.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import load_config
from src.data.dataset import build_dataloaders
from src.data.indexing import make_patient_splits, filter_by_domain
from src.data.preprocessing import load_index
from src.engine import fit
from src.models.build import build_model
from src.utils import set_seed, get_device, setup_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "base.yaml"))
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    set_seed(int(cfg.project.seed), bool(cfg.project.deterministic))
    device = get_device()
    logger = setup_logger("train", os.path.join(str(cfg.project.output_dir), "train.log"))

    cases = load_index(cfg.paths.processed_root, "train")

    # Esperimento di dominio (Fase 6): allena solo su un sottoinsieme di protocollo
    # (es. thickness_band=thick) per poi misurare il calo su quello non visto.
    if cfg.domain.split_by and cfg.domain.train_domain:
        n0 = len(cases)
        cases = filter_by_domain(cases, str(cfg.domain.split_by), cfg.domain.train_domain)
        logger.info(f"Dominio: {cfg.domain.split_by}={cfg.domain.train_domain} "
                    f"-> {len(cases)}/{n0} casi di training")
        if not cases:
            raise ValueError("Nessun caso nel dominio richiesto: esegui prima "
                             "scripts/01b_refresh_metadata.py")

    train_cases, val_cases = make_patient_splits(cases, args.val_fraction, int(cfg.project.seed))
    n_tr = len({c["patient"] for c in train_cases})
    n_va = len({c["patient"] for c in val_cases})
    logger.info(f"Esperimento: {os.path.basename(args.config)} -> {cfg.project.output_dir}")
    logger.info(f"Casi: {len(train_cases)} train ({n_tr} pazienti) / "
                f"{len(val_cases)} val ({n_va} pazienti)")
    logger.info(f"Modalità: {list(cfg.data.modalities)} | context_slices: "
                f"{cfg.data.context_slices} -> in_channels={cfg.model.in_channels}")

    if len(train_cases) < 2:
        raise ValueError(
            f"Solo {len(train_cases)} casi di training (indice: {len(cases)} casi).\n"
            f"L'indice {cfg.paths.processed_root}/index_train.json sembra incompleto.\n"
            f"Rilancia il preprocessing COMPLETO (senza --limit):\n"
            f"    python scripts/01_preprocess.py --split train\n"
            f"I volumi già presenti vengono saltati, quindi dura pochi secondi.")

    train_loader, val_loader, train_ds, val_ds = build_dataloaders(cfg, train_cases, val_cases)
    logger.info(f"Slice: train {len(train_ds)} (con lesione {train_ds.is_positive.mean()*100:.1f}% "
                f"-> ricampionate al {float(cfg.data.pos_neg_ratio)*100:.0f}%) | val {len(val_ds)}")

    model = build_model(cfg)
    fit(cfg, model, train_loader, val_loader, device=device, logger=logger)


if __name__ == "__main__":
    main()
