"""Genera il pacchetto di figure per l'analisi qualitativa e per il report (Fase 7).

    python scripts/05_visualize.py --run runs/unet_flair --split test
    python scripts/05_visualize.py --run runs/unet_flair --split test --device cpu
    python scripts/05_visualize.py --run runs/unet_flair --split test --cases P59_T1 P72_T1

Produce in <run>/figures/:
    training_curves.png     loss e Dice per epoca
    results_overview.png    distribuzione del Dice, relazione col carico lesionale, sottogruppi
    detection_by_size.png   tasso di rilevamento per dimensione della lesione
    overlay_<caso>.png      TP/FN/FP sulle slice più lesionate (casi migliori e peggiori)
    lesion_zoom_<caso>.png  zoom sulle singole lesioni, dalla più piccola alla più grande

Usa `--device cpu` se la GPU è occupata da un training in corso.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from omegaconf import OmegaConf

from src.data.preprocessing import load_index
from src.data.transforms import build_transforms
from src.evaluate import predict_case, _remove_small_components
from src.models.build import build_model
from src.visualize import (plot_training_curves, plot_results_overview,
                           plot_case_overlay, plot_lesion_zoom, plot_detection_by_size)


def lesion_sizes_and_detection(pred, gt, min_size: int = 3):
    from scipy import ndimage
    lab, n = ndimage.label(gt, structure=ndimage.generate_binary_structure(3, 2))
    if n == 0:
        return []
    sizes = ndimage.sum(gt, lab, index=np.arange(1, n + 1))
    out = []
    for i in range(n):
        if sizes[i] < min_size:
            continue
        out.append((float(sizes[i]), bool((pred[lab == i + 1]).any())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default="best.pth")
    ap.add_argument("--cases", nargs="*", default=None, help="casi specifici (default: 2 migliori + 2 peggiori)")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--max-cases", type=int, default=None, help="limita l'analisi per dimensione")
    args = ap.parse_args()

    run_dir = args.run if os.path.isabs(args.run) else os.path.join(REPO_ROOT, args.run)
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    cfg = OmegaConf.load(os.path.join(run_dir, "config_resolved.yaml"))
    eval_dir = os.path.join(run_dir, f"eval_{args.split}")

    # --- 1) curve di apprendimento (nessun modello richiesto) ---
    if os.path.exists(os.path.join(run_dir, "history.csv")):
        plot_training_curves(run_dir)
        print("  [+] training_curves.png")

    # --- 2) panoramica dei risultati (dal CSV della valutazione) ---
    res_csv = os.path.join(eval_dir, "results_per_case.csv")
    if os.path.exists(res_csv):
        plot_results_overview(res_csv, save_dir=run_dir)
        print("  [+] results_overview.png")
    else:
        print(f"  [!] {res_csv} assente: esegui prima 03_evaluate.py")

    # --- 3) figure che richiedono il modello ---
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    ckpt = torch.load(os.path.join(run_dir, args.checkpoint), map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    _, eval_tf = build_transforms(cfg)
    cases = load_index(cfg.paths.processed_root, args.split)
    by_name = {f"{c['patient']}_{c['timepoint']}": c for c in cases}
    print(f"  device: {device} | casi disponibili: {len(cases)}")

    # scelta dei casi da illustrare: i 2 migliori e i 2 peggiori
    if args.cases:
        names = [n for n in args.cases if n in by_name]
    elif os.path.exists(res_csv):
        import pandas as pd
        df = pd.read_csv(res_csv).sort_values("dice")
        names = list(df["case"].head(2)) + list(df["case"].tail(2))
    else:
        names = list(by_name)[:2]

    thr = float(cfg.eval.threshold)
    for name in names:
        meta = by_name[name]
        prob = predict_case(model, meta, cfg, device, eval_tf)
        pred = _remove_small_components(prob > thr, int(cfg.eval.get("postproc_min_size", 0)))
        gt = np.load(os.path.join(meta["dir"], "mask.npy")) > 0
        img = np.load(os.path.join(meta["dir"], f"{cfg.data.modalities[0]}.npy")).astype(np.float32)
        plot_case_overlay(img, gt, pred, case_name=name, save_dir=run_dir)
        plot_lesion_zoom(img, gt, pred, case_name=name, save_dir=run_dir)
        d = 2 * (gt & pred).sum() / max(gt.sum() + pred.sum(), 1)
        print(f"  [+] overlay_{name}.png / lesion_zoom_{name}.png (Dice {d:.3f})")

    # --- 4) rilevamento per dimensione della lesione (su tutti i casi) ---
    subset = cases[: args.max_cases] if args.max_cases else cases
    pairs = []
    for i, meta in enumerate(subset, 1):
        prob = predict_case(model, meta, cfg, device, eval_tf)
        pred = _remove_small_components(prob > thr, int(cfg.eval.get("postproc_min_size", 0)))
        gt = np.load(os.path.join(meta["dir"], "mask.npy")) > 0
        pairs += lesion_sizes_and_detection(pred, gt, int(cfg.eval.min_lesion_size))
        print(f"      analisi dimensioni {i}/{len(subset)}", end="\r")
    if pairs:
        plot_detection_by_size(pairs, save_dir=run_dir)
        s = np.array([p[0] for p in pairs]); d = np.array([p[1] for p in pairs])
        small, big = s < 30, s >= 100
        print(f"\n  [+] detection_by_size.png")
        print(f"      lesioni <30 voxel : rilevate {d[small].mean()*100:.0f}% (n={small.sum()})")
        print(f"      lesioni >=100 voxel: rilevate {d[big].mean()*100:.0f}% (n={big.sum()})")

    print(f"\nFigure salvate in {fig_dir}")


if __name__ == "__main__":
    main()
