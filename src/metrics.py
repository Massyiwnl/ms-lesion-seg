"""Metriche di segmentazione: sovrapposizione, distanza e lesion-wise.

Perché più di una metrica:
- **Dice/IoU** misurano la sovrapposizione, ma sono dominati dai pazienti con molte
  lesioni (nel nostro dataset il carico varia di 98 volte tra un paziente e l'altro).
- **HD95/ASSD** misurano l'errore di contorno: un Dice alto con contorni sbagliati
  è clinicamente diverso da un Dice alto con contorni precisi.
- **Lesion-wise** conta quante lesioni sono state *trovate*, non quanti voxel:
  clinicamente conta il numero di lesioni nuove, e una lesione piccola mancata pesa
  poco sul Dice ma moltissimo sulla diagnosi.

Nota metodologica: il Dice va calcolato **per volume** (per paziente) e poi mediato,
non mediando i Dice delle singole slice. Per questo l'accumulatore somma TP/FP/FN
per caso e calcola il Dice solo alla fine.
"""
from __future__ import annotations
import numpy as np


class CaseAccumulator:
    """Accumula TP/FP/FN per caso, per calcolare il Dice a livello di volume."""

    def __init__(self):
        self.stats: dict[str, np.ndarray] = {}

    def update(self, pred_bin, target_bin, case_ids):
        """pred_bin/target_bin: tensori/array [B, 1, H, W] binari; case_ids: lista di str."""
        p = np.asarray(pred_bin).reshape(len(case_ids), -1).astype(bool)
        t = np.asarray(target_bin).reshape(len(case_ids), -1).astype(bool)
        tp = (p & t).sum(1)
        fp = (p & ~t).sum(1)
        fn = (~p & t).sum(1)
        for i, cid in enumerate(case_ids):
            s = self.stats.setdefault(cid, np.zeros(3, dtype=np.int64))
            s += np.array([tp[i], fp[i], fn[i]], dtype=np.int64)

    def per_case_dice(self) -> dict[str, float]:
        out = {}
        for cid, (tp, fp, fn) in self.stats.items():
            denom = 2 * tp + fp + fn
            out[cid] = float(2 * tp / denom) if denom > 0 else float("nan")
        return out

    def mean_dice(self) -> float:
        d = np.array(list(self.per_case_dice().values()), dtype=float)
        return float(np.nanmean(d)) if d.size else float("nan")

    def summary(self) -> dict:
        per_case = self.per_case_dice()
        d = np.array(list(per_case.values()), dtype=float)
        tot = np.sum(list(self.stats.values()), axis=0) if self.stats else np.zeros(3)
        tp, fp, fn = tot
        return {
            "dice_mean": float(np.nanmean(d)) if d.size else float("nan"),
            "dice_std": float(np.nanstd(d)) if d.size else float("nan"),
            "dice_median": float(np.nanmedian(d)) if d.size else float("nan"),
            "iou_global": float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else float("nan"),
            "precision_global": float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan"),
            "recall_global": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
            "n_cases": len(per_case),
        }


def dice_score(pred_bin, target_bin) -> float:
    p = np.asarray(pred_bin).astype(bool)
    t = np.asarray(target_bin).astype(bool)
    denom = p.sum() + t.sum()
    return float(2 * (p & t).sum() / denom) if denom > 0 else float("nan")


def iou_score(pred_bin, target_bin) -> float:
    p = np.asarray(pred_bin).astype(bool)
    t = np.asarray(target_bin).astype(bool)
    union = (p | t).sum()
    return float((p & t).sum() / union) if union > 0 else float("nan")


def surface_distances(pred_bin, target_bin, spacing=(1.0, 1.0, 1.0)):
    """HD95 e ASSD (mm) tramite MONAI. Ritorna (hd95, assd), NaN se una maschera è vuota."""
    import torch
    from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric

    p = np.asarray(pred_bin).astype(np.float32)
    t = np.asarray(target_bin).astype(np.float32)
    if p.sum() == 0 or t.sum() == 0:
        return float("nan"), float("nan")
    pt = torch.from_numpy(p)[None, None]
    tt = torch.from_numpy(t)[None, None]
    hd = HausdorffDistanceMetric(include_background=True, percentile=95)
    sd = SurfaceDistanceMetric(include_background=True, symmetric=True)
    return float(hd(pt, tt).item()), float(sd(pt, tt).item())


def lesion_wise_metrics(pred_bin, gt_bin, min_size: int = 3, connectivity: int = 2) -> dict:
    """Metriche per-lesione su componenti connesse (volume 3D).

    Una lesione della GT è "trovata" se almeno un voxel della predizione la interseca;
    una componente predetta che non interseca nessuna lesione vera è un falso positivo.
    Le componenti più piccole di `min_size` voxel vengono ignorate (rumore).
    """
    from scipy import ndimage

    p = np.asarray(pred_bin).astype(bool)
    g = np.asarray(gt_bin).astype(bool)
    struct = ndimage.generate_binary_structure(p.ndim, connectivity)

    def _components(mask):
        lab, n = ndimage.label(mask, structure=struct)
        if n == 0:
            return lab, []
        sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
        return lab, [i + 1 for i, s in enumerate(sizes) if s >= min_size]

    g_lab, g_ids = _components(g)
    p_lab, p_ids = _components(p)

    detected = sum(1 for i in g_ids if np.any(p[g_lab == i]))
    fp = sum(1 for j in p_ids if not np.any(g[p_lab == j]))
    n_gt, n_pred = len(g_ids), len(p_ids)
    tpr = detected / n_gt if n_gt else float("nan")
    ppv = (n_pred - fp) / n_pred if n_pred else float("nan")
    f1 = (2 * tpr * ppv / (tpr + ppv)) if (n_gt and n_pred and (tpr + ppv) > 0) else float("nan")
    return {"n_lesions_gt": n_gt, "n_lesions_pred": n_pred, "lesions_detected": detected,
            "lesion_tpr": tpr, "lesion_fp": fp, "lesion_ppv": ppv, "lesion_f1": f1}
