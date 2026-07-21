"""Explainability: Grad-CAM e mappe di attenzione (Fase 7).

    python scripts/07_explain.py --run runs/unet_flair_ds --split test --device cpu
    python scripts/07_explain.py --run runs/best_combo --split test --cases P69_T1 P59_T1

Produce in <run>/figures/:
    explain_<caso>.png   immagine, overlay TP/FN/FP e mappe di rilevanza affiancate
    cam_alignment.png    quanto la rilevanza si concentra sulla lesione, per strato

La domanda a cui rispondono queste figure non è "il modello segmenta bene?" ma
"si sta basando sulla lesione o su qualcos'altro?". Un modello con buon Dice ma
rilevanza diffusa è fragile: probabilmente sfrutta regolarità del dataset che non
si ritrovano in un altro ospedale.
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
from src.explain.gradcam import GradCAM2D, attention_maps, suggest_layers
from src.models.build import build_model
from src.visualize import plot_explanation, plot_cam_alignment


def selectivity(cam, gt_sl, brain_sl):
    """Rilevanza media dentro la lesione / nel tessuto sano."""
    les = gt_sl > 0
    healthy = (brain_sl != 0) & ~les
    if les.sum() < 3 or healthy.sum() < 100:
        return None
    return float(cam[les].mean() / (cam[healthy].mean() + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default="best.pth")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--layers", nargs="*", default=None)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--n-cases", type=int, default=4)
    args = ap.parse_args()

    run_dir = args.run if os.path.isabs(args.run) else os.path.join(REPO_ROOT, args.run)
    cfg = OmegaConf.load(os.path.join(run_dir, "config_resolved.yaml"))
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
    layers = args.layers or suggest_layers(model)
    print(f"Modello: {cfg.model.arch} | device {device}")
    print(f"Strati per la Grad-CAM: {layers}")

    if args.cases:
        names = [n for n in args.cases if n in by_name]
    else:
        res = os.path.join(run_dir, f"eval_{args.split}", "results_per_case.csv")
        if os.path.exists(res):
            import pandas as pd
            df = pd.read_csv(res).sort_values("dice")
            k = max(1, args.n_cases // 2)
            names = list(df["case"].head(k)) + list(df["case"].tail(k))
        else:
            names = list(by_name)[: args.n_cases]

    agg: dict[str, list] = {}
    thr = float(cfg.eval.threshold)
    for name in names:
        meta = by_name[name]
        prob = predict_case(model, meta, cfg, device, eval_tf)
        pred = _remove_small_components(prob > thr, int(cfg.eval.get("postproc_min_size", 0)))
        gt = np.load(os.path.join(meta["dir"], "mask.npy")) > 0
        img = np.load(os.path.join(meta["dir"], f"{cfg.data.modalities[0]}.npy")).astype(np.float32)
        z = int(np.argmax(gt.reshape(len(gt), -1).sum(1)))

        # ricostruisci l'input 2.5D della slice scelta
        ctx = int(cfg.data.context_slices)
        chans = []
        for m in cfg.data.modalities:
            vol = np.load(os.path.join(meta["dir"], f"{m}.npy"), mmap_mode="r")
            zs = [int(np.clip(z + d, 0, len(vol) - 1)) for d in range(-ctx, ctx + 1)]
            chans.append(np.asarray(vol[zs], dtype=np.float32))
        x = torch.from_numpy(np.concatenate(chans, 0))[None].to(device)

        cams, degenerate = {}, []
        for lname in layers:
            with GradCAM2D(model, lname) as cam:
                m = cam(x)[0, 0].cpu().numpy()
                if cam.degenerate is not None and bool(cam.degenerate[0]):
                    degenerate.append(lname)     # nessun contributo positivo: non informativa
                    continue
                cams[f"Grad-CAM · {lname}"] = m
        att = attention_maps(model, x)
        if att:                                   # la più grossolana e la più fine
            cams["attention gate (grossolana)"] = att[0]
            if len(att) > 1:
                cams["attention gate (fine)"] = att[-1]
        if degenerate:
            print(f"      [i] mappe nulle (nessun contributo positivo), escluse: {degenerate}")

        for label, c in cams.items():
            s = selectivity(c, gt[z], img[z])
            if s is not None:
                agg.setdefault(label, []).append(s)

        plot_explanation(img[z], gt[z], pred[z], cams, case_name=name, z=z, save_dir=run_dir)
        print(f"  [+] explain_{name}.png (z={z})")

    if agg:
        means = {k: float(np.mean(v)) for k, v in agg.items()}
        plot_cam_alignment(means, save_dir=run_dir)
        print("\n  Selettività (rilevanza lesione / tessuto sano):")
        for k, v in means.items():
            print(f"    {k:28s} {v:5.2f}x")
        print("  [+] cam_alignment.png")
    print(f"\nFigure in {os.path.join(run_dir, 'figures')}")


if __name__ == "__main__":
    main()
