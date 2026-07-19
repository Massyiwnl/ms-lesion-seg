"""Valutazione di un modello addestrato su uno split (Fase 5).

    python scripts/03_evaluate.py --run runs/unet_flair --split test

Legge `config_resolved.yaml` e `best.pth` dalla cartella del run, quindi la valutazione
usa esattamente la configurazione con cui il modello è stato addestrato.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
from omegaconf import OmegaConf

from src.data.preprocessing import load_index
from src.evaluate import evaluate_model
from src.models.build import build_model
from src.utils import get_device, set_seed, setup_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="cartella del run (es. runs/unet_flair)")
    ap.add_argument("--split", default="test", help="test | train")
    ap.add_argument("--checkpoint", default="best.pth")
    ap.add_argument("--out", default=None, help="cartella dei risultati (default: <run>/eval_<split>)")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    run_dir = args.run if os.path.isabs(args.run) else os.path.join(REPO_ROOT, args.run)
    cfg_path = os.path.join(run_dir, "config_resolved.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config del run non trovata: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.override)))

    out_dir = args.out or os.path.join(run_dir, f"eval_{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logger("eval", os.path.join(out_dir, "eval.log"))
    set_seed(int(cfg.project.seed))
    device = get_device()

    model = build_model(cfg)
    ckpt = torch.load(os.path.join(run_dir, args.checkpoint), map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt["model"])
    logger.info(f"Run: {run_dir} | checkpoint: {args.checkpoint} "
                f"(epoca {ckpt.get('epoch','?')}, Dice val {ckpt.get('val_dice', float('nan')):.4f})")
    logger.info(f"Modello: {cfg.model.arch} | modalità {list(cfg.data.modalities)} "
                f"| in_channels {cfg.model.in_channels} | device {device}")

    cases = load_index(cfg.paths.processed_root, args.split)
    logger.info(f"Valutazione su split '{args.split}': {len(cases)} casi "
                f"({len({c['patient'] for c in cases})} pazienti)\n")
    evaluate_model(cfg, model, cases, device=device, out_dir=out_dir, logger=logger)


if __name__ == "__main__":
    main()
