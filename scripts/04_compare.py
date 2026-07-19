"""Tabella comparativa di tutti gli esperimenti (Fase 5).

    python scripts/04_compare.py --runs-dir runs --split test

Scansiona le cartelle dei run, legge la configurazione e i risultati di valutazione e
produce una tabella unica (CSV + Markdown pronto per il report), con le colonne degli
assi sperimentali (architettura, modalità, loss, deep supervision, ...) accanto alle
metriche. Se disponibile, esegue anche un test statistico appaiato rispetto alla
baseline: con pochi pazienti una differenza di Dice può essere solo rumore.
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

METRICS = [("dice_mean", "Dice"), ("dice_std", "±"), ("iou_mean", "IoU"),
           ("hd95_mean", "HD95 (mm)"), ("lesion_tpr_mean", "Les. TPR"),
           ("lesion_f1_mean", "Les. F1"), ("lesion_fp_mean", "Les. FP"),
           ("precision_mean", "Prec."), ("recall_mean", "Rec.")]


def collect(runs_dir: str, split: str) -> pd.DataFrame:
    rows = []
    for name in sorted(os.listdir(runs_dir)):
        run = os.path.join(runs_dir, name)
        cfg_p = os.path.join(run, "config_resolved.yaml")
        res_p = os.path.join(run, f"eval_{split}", "results_summary.json")
        if not (os.path.exists(cfg_p) and os.path.exists(res_p)):
            continue
        cfg = OmegaConf.load(cfg_p)
        with open(res_p) as f:
            res = json.load(f)
        row = {
            "esperimento": name,
            "arch": str(cfg.model.arch),
            "modalità": "+".join(list(cfg.data.modalities)),
            "loss": str(cfg.loss.name),
            "deep_sup": bool(cfg.model.deep_supervision),
            "jacobian": bool(cfg.regularization.jacobian),
            "n_casi": res.get("n_cases"),
        }
        summ_p = os.path.join(run, "summary.json")
        if os.path.exists(summ_p):
            with open(summ_p) as f:
                row["dice_val"] = json.load(f).get("best_val_dice")
        row.update({k: res.get(k) for k, _ in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_test(runs_dir: str, split: str, baseline: str, other: str):
    """Wilcoxon appaiato sui Dice per caso (stessi pazienti nei due esperimenti)."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    pa = os.path.join(runs_dir, baseline, f"eval_{split}", "results_per_case.csv")
    pb = os.path.join(runs_dir, other, f"eval_{split}", "results_per_case.csv")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return None
    a = pd.read_csv(pa).set_index("case")["dice"]
    b = pd.read_csv(pb).set_index("case")["dice"]
    common = a.index.intersection(b.index)
    if len(common) < 6:
        return None
    x, y = a.loc[common].values, b.loc[common].values
    if np.allclose(x, y):
        return None
    stat, p = wilcoxon(x, y)
    return {"baseline": baseline, "confronto": other, "n": len(common),
            "delta_dice": float(np.mean(y - x)), "p_value": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--baseline", default=None, help="run di riferimento per il test statistico")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs_dir = args.runs_dir if os.path.isabs(args.runs_dir) else os.path.join(REPO_ROOT, args.runs_dir)
    df = collect(runs_dir, args.split)
    if df.empty:
        print(f"[-] Nessun risultato in {runs_dir} per lo split '{args.split}'.")
        print("    Esegui prima: python scripts/03_evaluate.py --run runs/<nome> --split", args.split)
        return

    df = df.sort_values("dice_mean", ascending=False)
    out_dir = args.out or os.path.join(runs_dir, "_comparison")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"comparison_{args.split}.csv"), index=False)

    # tabella Markdown per il report
    cols = ["esperimento", "arch", "modalità", "loss", "deep_sup"] + [k for k, _ in METRICS]
    hdr = ["Esperimento", "Arch.", "Modalità", "Loss", "DS"] + [lbl for _, lbl in METRICS]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r.get(c)
            vals.append("-" if v is None or (isinstance(v, float) and np.isnan(v))
                        else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(vals) + " |")
    md = "\n".join(lines)
    with open(os.path.join(out_dir, f"comparison_{args.split}.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")

    print(f"\n=== Confronto esperimenti (split: {args.split}) ===\n")
    print(md)

    base = args.baseline or df.iloc[-1]["esperimento"]
    tests = [t for t in (paired_test(runs_dir, args.split, base, o)
                         for o in df["esperimento"] if o != base) if t]
    if tests:
        print(f"\n=== Test di Wilcoxon appaiato (riferimento: {base}) ===")
        for t in tests:
            sig = "significativo" if t["p_value"] < 0.05 else "non significativo"
            print(f"  {t['confronto']:32s} ΔDice {t['delta_dice']:+.4f}  "
                  f"p={t['p_value']:.4f}  ({sig}, n={t['n']})")
        pd.DataFrame(tests).to_csv(os.path.join(out_dir, f"stats_{args.split}.csv"), index=False)
    print(f"\nSalvato in {out_dir}")


if __name__ == "__main__":
    main()
