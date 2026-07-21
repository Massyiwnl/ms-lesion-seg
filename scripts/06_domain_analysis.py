"""Analisi di robustezza al dominio: in-dominio vs fuori dominio (Fase 6).

    python scripts/06_domain_analysis.py --runs runs/domain_thick_base runs/domain_thick_robust runs/unet_flair_ds
    python scripts/06_domain_analysis.py --runs ... --attribute vendor --in-domain Philips

Per ogni run legge `eval_<split>/results_per_case.csv`, divide i pazienti di test in
"in dominio" e "fuori dominio" secondo un attributo di acquisizione, e calcola il
divario di prestazione. Esegue anche il test appaiato tra i run sugli STESSI pazienti.

ATTENZIONE metodologica: un divario che si restringe NON significa che il modello sia
diventato più robusto. Può restringersi perché il gruppo che andava meglio peggiora.
Per questo lo script riporta sempre i due gruppi separatamente e il test appaiato, non
solo il divario.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd


def load_run(run_dir: str, split: str) -> pd.DataFrame:
    p = os.path.join(run_dir, f"eval_{split}", "results_per_case.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Risultati mancanti: {p}\nEsegui prima 03_evaluate.py")
    df = pd.read_csv(p).set_index("case")
    df["_run"] = os.path.basename(run_dir.rstrip("/\\"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--attribute", default="thickness_band")
    ap.add_argument("--in-domain", default="thick", help="valore corrispondente al dominio di training")
    ap.add_argument("--metric", default="dice")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = [r if os.path.isabs(r) else os.path.join(REPO_ROOT, r) for r in args.runs]
    dfs = {os.path.basename(r.rstrip("/\\")): load_run(r, args.split) for r in runs}
    ref = next(iter(dfs.values()))
    if args.attribute not in ref.columns:
        raise SystemExit(f"Attributo '{args.attribute}' assente. Esegui 01b_refresh_metadata.py "
                         f"e poi rilancia 03_evaluate.py.\nColonne: {list(ref.columns)}")

    grp = ref[args.attribute].astype(str)
    IN = (grp == args.in_domain).values
    OUT = (~IN) & (grp != "unknown").values
    names = list(ref.index)
    print(f"Attributo di dominio: {args.attribute} | in dominio = '{args.in_domain}'")
    print(f"  IN    ({IN.sum():2d}): {', '.join(n for n, m in zip(names, IN) if m)}")
    print(f"  FUORI ({OUT.sum():2d}): {', '.join(n for n, m in zip(names, OUT) if m)}\n")

    rows = []
    for name, df in dfs.items():
        v = df.loc[names, args.metric].values
        rows.append({"run": name,
                     "in_dominio": float(np.nanmean(v[IN])),
                     "fuori_dominio": float(np.nanmean(v[OUT])),
                     "divario": float(np.nanmean(v[IN]) - np.nanmean(v[OUT])),
                     "complessivo": float(np.nanmean(v))})
    tab = pd.DataFrame(rows)
    print("=== Divario di dominio ===")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if len(dfs) >= 2:
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            wilcoxon = None
        base_name = list(dfs)[0]
        base = dfs[base_name].loc[names, args.metric].values
        print(f"\n=== Test appaiati rispetto a '{base_name}' ===")
        for name, df in list(dfs.items())[1:]:
            v = df.loc[names, args.metric].values
            for label, mask in (("FUORI dominio", OUT), ("IN dominio", IN)):
                d = v[mask] - base[mask]
                txt = f"  {name:26s} {label:14s} Δ {d.mean():+.4f}  migliorati {(d > 0).sum()}/{mask.sum()}"
                if wilcoxon is not None and mask.sum() >= 6 and not np.allclose(v[mask], base[mask]):
                    _, pv = wilcoxon(base[mask], v[mask])
                    txt += f"  p={pv:.4f} ({'signif.' if pv < 0.05 else 'non signif.'})"
                print(txt)
        print("\n  Nota: un divario che si restringe non implica maggiore robustezza. "
              "Verifica\n  che il gruppo FUORI dominio sia MIGLIORATO, non che quello IN sia peggiorato.")

    out_dir = args.out or os.path.join(REPO_ROOT, "runs", "_comparison")
    os.makedirs(out_dir, exist_ok=True)
    tab.to_csv(os.path.join(out_dir, f"domain_{args.attribute}_{args.split}.csv"), index=False)

    # figura a barre raggruppate
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(tab)); w = 0.38
        fig, ax = plt.subplots(figsize=(1.9 * len(tab) + 4, 4.4))
        ax.bar(x - w/2, tab["in_dominio"], w, label=f"in dominio ({args.in_domain}, n={IN.sum()})")
        ax.bar(x + w/2, tab["fuori_dominio"], w, label=f"fuori dominio (n={OUT.sum()})")
        for i, r in tab.iterrows():
            ax.text(i, max(r["in_dominio"], r["fuori_dominio"]) + 0.02,
                    f"divario {r['divario']:+.3f}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(tab["run"], fontsize=9)
        ax.set_ylabel(f"{args.metric} medio"); ax.set_ylim(0, 1.0)
        ax.set_title(f"Robustezza al dominio — {args.attribute}")
        ax.legend(frameon=False); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        path = os.path.join(out_dir, f"domain_{args.attribute}_{args.split}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\nFigura: {path}")
    except Exception as e:
        print(f"[i] figura non generata: {e}")
    print(f"Tabella: {out_dir}")


if __name__ == "__main__":
    main()
