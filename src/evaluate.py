"""Valutazione per volume su un intero split, con analisi per sottogruppo.

Perché la valutazione è "per volume" e non per slice:
- HD95, ASSD e le metriche lesion-wise richiedono le componenti connesse **3D**: una
  lesione è un oggetto tridimensionale, contarla slice per slice la conterebbe più volte.
- Il Dice per paziente è la metrica riportata dai benchmark, quindi rende i risultati
  confrontabili con le baseline pubblicate.

Per ogni caso si ricostruisce il volume predetto impilando le predizioni di tutte le
slice (qui il filtro sul cervello viene disattivato: si predice l'intero volume), poi si
confronta con la maschera di riferimento.
"""
from __future__ import annotations
import copy
import json
import os
import numpy as np
import torch

from src.metrics import dice_score, iou_score, surface_distances, lesion_wise_metrics


def _remove_small_components(mask, min_size: int):
    """Post-processing: elimina le componenti connesse più piccole di `min_size` voxel."""
    if min_size <= 0:
        return mask
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(mask.ndim, 2))
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_size
    return keep[lab]


@torch.no_grad()
def predict_case(model, case_meta: dict, cfg, device, transforms, batch_size: int = 32):
    """Ricostruisce il volume di probabilità predetto per un caso: [S, H, W] float32."""
    from torch.utils.data import DataLoader
    from src.data.dataset import SliceDataset

    cfg_eval = copy.deepcopy(cfg)
    cfg_eval.data.min_brain_fraction = 0.0     # predici TUTTE le slice del volume
    ds = SliceDataset([case_meta], cfg_eval, transforms=transforms, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    n, h, w = int(case_meta["n_slices"]), *map(int, case_meta["spatial_size"])
    prob = np.zeros((n, h, w), dtype=np.float32)
    model.eval()
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        p = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
        for k, z in enumerate(batch["z"].tolist()):
            prob[int(z)] = p[k]
    return prob


def evaluate_case(model, case_meta, cfg, device, transforms) -> dict:
    """Metriche complete per un singolo caso."""
    prob = predict_case(model, case_meta, cfg, device, transforms)
    pred = prob > float(cfg.eval.threshold)
    pred = _remove_small_components(pred, int(cfg.eval.get("postproc_min_size", 0)))

    mask_path = os.path.join(case_meta["dir"], "mask.npy")
    if not os.path.exists(mask_path):
        return {"case": f"{case_meta['patient']}_{case_meta['timepoint']}", "has_mask": False}
    gt = np.load(mask_path) > 0

    row = {
        "case": f"{case_meta['patient']}_{case_meta['timepoint']}",
        "patient": case_meta["patient"],
        "timepoint": case_meta["timepoint"],
        "has_mask": True,
        "dice": dice_score(pred, gt),
        "iou": iou_score(pred, gt),
        "vol_gt": int(gt.sum()),
        "vol_pred": int(pred.sum()),
    }
    row["vol_error_pct"] = (100.0 * (row["vol_pred"] - row["vol_gt"]) / row["vol_gt"]
                            if row["vol_gt"] > 0 else float("nan"))
    tp = float((pred & gt).sum())
    row["precision"] = tp / max(float(pred.sum()), 1e-8)
    row["recall"] = tp / max(float(gt.sum()), 1e-8)

    if bool(cfg.eval.hd95):
        hd, assd = surface_distances(pred, gt)
        row["hd95"], row["assd"] = hd, assd
    if bool(cfg.eval.lesion_wise):
        row.update(lesion_wise_metrics(pred, gt, int(cfg.eval.min_lesion_size)))

    for k in ("sex", "age", "age_band", "ms_type", "edss",
              "vendor", "scanner_model", "slice_thickness", "thickness_band"):
        if k in case_meta:
            row[k] = case_meta[k]
    return row


def evaluate_model(cfg, model, cases: list[dict], device=None, out_dir: str | None = None,
                   logger=None) -> dict:
    """Valuta il modello su tutti i casi; salva CSV per caso, riepilogo e tabella fairness."""
    import pandas as pd
    from src.data.transforms import build_transforms
    from src.utils import get_device, setup_logger

    device = device or get_device()
    logger = logger or setup_logger("eval")
    out_dir = out_dir or str(cfg.project.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    _, eval_tf = build_transforms(cfg)
    model = model.to(device)

    rows = []
    for i, case in enumerate(cases, 1):
        row = evaluate_case(model, case, cfg, device, eval_tf)
        rows.append(row)
        if row.get("has_mask"):
            extra = f" | HD95 {row['hd95']:.1f}mm" if not np.isnan(row.get("hd95", np.nan)) else ""
            logger.info(f"  [{i}/{len(cases)}] {row['case']}: Dice {row['dice']:.4f} "
                        f"(P {row['precision']:.3f} R {row['recall']:.3f}){extra} | "
                        f"lesioni {row.get('lesions_detected','-')}/{row.get('n_lesions_gt','-')} "
                        f"FP {row.get('lesion_fp','-')}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "results_per_case.csv"), index=False)

    num = df.select_dtypes(include=[np.number])
    summary = {"n_cases": int(len(df))}
    for c in ("dice", "iou", "precision", "recall", "hd95", "assd",
              "lesion_tpr", "lesion_f1", "lesion_fp", "vol_error_pct"):
        if c in num:
            summary[f"{c}_mean"] = float(np.nanmean(num[c]))
            summary[f"{c}_std"] = float(np.nanstd(num[c]))
            summary[f"{c}_median"] = float(np.nanmedian(num[c]))
    with open(os.path.join(out_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # --- fairness: metriche per sottogruppo ---
    sub_rows = []
    for key in list(cfg.eval.subgroups):
        if key not in df.columns:
            continue
        for val, g in df.groupby(df[key].astype(str)):
            sub_rows.append({"attributo": key, "gruppo": val, "n": int(len(g)),
                             "dice_mean": float(np.nanmean(g["dice"])),
                             "dice_std": float(np.nanstd(g["dice"])),
                             "recall_mean": float(np.nanmean(g["recall"])),
                             "precision_mean": float(np.nanmean(g["precision"]))})
    if sub_rows:
        sdf = pd.DataFrame(sub_rows)
        sdf.to_csv(os.path.join(out_dir, "results_by_subgroup.csv"), index=False)
        summary["subgroups"] = sub_rows

    logger.info(f"\nDice medio: {summary.get('dice_mean', float('nan')):.4f} "
                f"± {summary.get('dice_std', float('nan')):.4f} "
                f"(mediana {summary.get('dice_median', float('nan')):.4f}) su {len(df)} casi")
    logger.info(f"Risultati salvati in {out_dir}")
    return summary
