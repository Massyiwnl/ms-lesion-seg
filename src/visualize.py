"""Figure per l'analisi qualitativa e per il report.

Le metriche dicono *quanto* il modello sbaglia, le figure dicono *come*: se
sovrasegmenta, se manca le lesioni piccole, se sbaglia solo i bordi. La codifica
a colori usata ovunque è:

    verde  = TP  (lesione trovata)
    rosso  = FN  (lesione mancata)
    blu    = FP  (falso allarme)

Tutte le funzioni restituiscono la Figure matplotlib, così si possono usare sia da
script (salvataggio su file) sia inline in un notebook.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

COL_TP = np.array([0.15, 0.85, 0.30])
COL_FN = np.array([1.00, 0.20, 0.20])
COL_FP = np.array([0.25, 0.55, 1.00])


def _to_display(sl, p_lo=1.0, p_hi=99.0):
    """Slice z-scored -> immagine in scala di grigi con buon contrasto."""
    sl = np.rot90(np.asarray(sl, dtype=np.float32))
    brain = sl != 0
    if brain.any():
        lo, hi = np.percentile(sl[brain], [p_lo, p_hi])
        sl = np.clip((sl - lo) / (hi - lo + 1e-8), 0, 1)
    return sl


def overlay_rgb(img_sl, gt_sl, pred_sl, alpha=0.6):
    """Immagine RGB con TP/FN/FP sovrapposti."""
    base = _to_display(img_sl)
    rgb = np.dstack([base] * 3)
    gt = np.rot90(np.asarray(gt_sl) > 0)
    pr = np.rot90(np.asarray(pred_sl) > 0)
    for mask, col in ((gt & pr, COL_TP), (gt & ~pr, COL_FN), (~gt & pr, COL_FP)):
        if mask.any():
            rgb[mask] = (1 - alpha) * rgb[mask] + alpha * col
    return rgb


def _legend(ax):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COL_TP, label="TP (trovata)"),
                       Patch(color=COL_FN, label="FN (mancata)"),
                       Patch(color=COL_FP, label="FP (falso allarme)")],
              loc="lower center", ncol=3, frameon=False, fontsize=9,
              bbox_to_anchor=(0.5, -0.08))


# ------------------------------------------------------------------ curve
def plot_training_curves(run_dir: str, save: bool = True):
    """Loss e Dice per epoca: mostra convergenza, overfitting ed early stopping."""
    import pandas as pd
    df = pd.read_csv(os.path.join(run_dir, "history.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(df["epoch"], df["train_loss"], label="train", lw=1.8)
    v = df.dropna(subset=["val_loss"])
    ax[0].plot(v["epoch"], v["val_loss"], label="validazione", lw=1.8)
    ax[0].set_xlabel("epoca"); ax[0].set_ylabel("loss"); ax[0].set_title("Loss")
    ax[0].legend(frameon=False); ax[0].grid(alpha=0.3)

    d = df.dropna(subset=["val_dice"])
    ax[1].plot(d["epoch"], d["val_dice"], color="tab:green", lw=1.8, label="Dice (per volume)")
    if "val_precision" in d:
        ax[1].plot(d["epoch"], d["val_precision"], "--", lw=1.2, alpha=0.7, label="precisione")
        ax[1].plot(d["epoch"], d["val_recall"], "--", lw=1.2, alpha=0.7, label="recall")
    if len(d):
        i = d["val_dice"].idxmax()
        ax[1].scatter([d.loc[i, "epoch"]], [d.loc[i, "val_dice"]], color="black", zorder=5, s=40)
        ax[1].annotate(f"best {d.loc[i,'val_dice']:.4f}\n(epoca {int(d.loc[i,'epoch'])})",
                       (d.loc[i, "epoch"], d.loc[i, "val_dice"]),
                       textcoords="offset points", xytext=(-10, -34), fontsize=9)
    ax[1].set_xlabel("epoca"); ax[1].set_ylabel("metrica"); ax[1].set_title("Validazione")
    ax[1].set_ylim(0, 1); ax[1].legend(frameon=False); ax[1].grid(alpha=0.3)

    fig.suptitle(f"Curve di apprendimento — {os.path.basename(run_dir)}")
    fig.tight_layout()
    if save:
        _savefig(fig, run_dir, "training_curves.png")
    return fig


# ------------------------------------------------------------------ overlay
def plot_case_overlay(image_vol, gt_vol, pred_vol, case_name: str = "",
                      n_slices: int = 3, save_dir: str | None = None):
    """Le slice con più lesione: immagine, ground truth, predizione, overlay."""
    gt = np.asarray(gt_vol) > 0
    per_slice = gt.reshape(len(gt), -1).sum(1)
    zs = np.argsort(per_slice)[::-1][:n_slices]
    zs = sorted(int(z) for z in zs if per_slice[z] > 0) or [int(np.argmax(per_slice))]

    fig, axes = plt.subplots(len(zs), 4, figsize=(13, 3.4 * len(zs)), squeeze=False)
    for r, z in enumerate(zs):
        pr = np.asarray(pred_vol[z]) > 0
        dice_z = 2 * (gt[z] & pr).sum() / max(gt[z].sum() + pr.sum(), 1)
        panels = [(_to_display(image_vol[z]), "FLAIR", "gray"),
                  (np.rot90(gt[z].astype(float)), "ground truth", "gray"),
                  (np.rot90(pr.astype(float)), "predizione", "gray"),
                  (overlay_rgb(image_vol[z], gt[z], pr), f"overlay — Dice {dice_z:.3f}", None)]
        for c, (im, title, cmap) in enumerate(panels):
            axes[r][c].imshow(im, cmap=cmap)
            axes[r][c].set_title(f"{title}" + (f"  (z={z})" if c == 0 else ""), fontsize=10)
            axes[r][c].axis("off")
    _legend(axes[-1][3])
    fig.suptitle(f"Analisi qualitativa — {case_name}", fontsize=12)
    fig.tight_layout()
    if save_dir:
        _savefig(fig, save_dir, f"overlay_{case_name}.png")
    return fig


def plot_lesion_zoom(image_vol, gt_vol, pred_vol, case_name: str = "",
                     n_lesions: int = 6, margin: int = 12, save_dir: str | None = None):
    """Zoom sulle singole lesioni, dalla più piccola alla più grande.

    Risponde alla domanda che le metriche aggregate nascondono: il modello sbaglia
    sulle lesioni piccole o su quelle grandi?
    """
    from scipy import ndimage
    gt = np.asarray(gt_vol) > 0
    pred = np.asarray(pred_vol) > 0
    lab, n = ndimage.label(gt, structure=ndimage.generate_binary_structure(3, 2))
    if n == 0:
        return None
    sizes = ndimage.sum(gt, lab, index=np.arange(1, n + 1))
    order = np.argsort(sizes)
    picks = [order[int(round(i))] for i in np.linspace(0, n - 1, min(n_lesions, n))]

    cols = min(len(picks), 3)
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for k, idx in enumerate(picks):
        comp = lab == (idx + 1)
        zc, hc, wc = (np.array(np.where(comp)).mean(axis=1)).astype(int)
        h0, h1 = max(0, hc - margin), min(gt.shape[1], hc + margin)
        w0, w1 = max(0, wc - margin), min(gt.shape[2], wc + margin)
        ax = axes[k // cols][k % cols]
        ax.imshow(overlay_rgb(image_vol[zc][h0:h1, w0:w1], gt[zc][h0:h1, w0:w1],
                              pred[zc][h0:h1, w0:w1], alpha=0.45), interpolation="nearest")
        found = (comp & pred).any()
        ax.set_title(f"{int(sizes[idx])} voxel — {'TROVATA' if found else 'MANCATA'}",
                     fontsize=10, color=("darkgreen" if found else "darkred"))
        ax.axis("off")
    for k in range(len(picks), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"Lesioni a scale diverse — {case_name}", fontsize=12)
    fig.tight_layout()
    if save_dir:
        _savefig(fig, save_dir, f"lesion_zoom_{case_name}.png")
    return fig


# ------------------------------------------------------------------ risultati
def plot_results_overview(results_csv: str, save_dir: str | None = None):
    """Distribuzione del Dice, relazione col carico lesionale, metriche per sottogruppo."""
    import pandas as pd
    df = pd.read_csv(results_csv)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].hist(df["dice"], bins=12, color="tab:blue", alpha=0.8, edgecolor="white")
    ax[0].axvline(df["dice"].mean(), color="black", ls="--",
                  label=f"media {df['dice'].mean():.3f}")
    ax[0].set_xlabel("Dice per paziente"); ax[0].set_ylabel("n. pazienti")
    ax[0].set_title("Distribuzione del Dice"); ax[0].legend(frameon=False)

    if "vol_gt" in df:
        ax[1].scatter(df["vol_gt"], df["dice"], s=40, alpha=0.8)
        ax[1].set_xscale("log")
        ax[1].set_xlabel("carico lesionale (voxel, scala log)"); ax[1].set_ylabel("Dice")
        try:
            from scipy import stats
            r, p = stats.pearsonr(np.log10(df["vol_gt"].clip(lower=1)), df["dice"])
            ax[1].set_title(f"Dice vs carico lesionale (r={r:+.2f}, p={p:.3f})")
        except Exception:
            ax[1].set_title("Dice vs carico lesionale")
        ax[1].grid(alpha=0.3)

    key = next((k for k in ("thickness_band", "vendor", "sex") if k in df.columns), None)
    if key:
        g = df.groupby(df[key].astype(str))["dice"]
        m, s, n = g.mean(), g.std().fillna(0), g.size()
        ax[2].bar(range(len(m)), m.values, yerr=s.values, capsize=4,
                  color="tab:green", alpha=0.8)
        ax[2].set_xticks(range(len(m)))
        ax[2].set_xticklabels([f"{k}\n(n={n[k]})" for k in m.index], fontsize=9)
        ax[2].set_ylabel("Dice medio"); ax[2].set_title(f"Dice per {key}")
        ax[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    if save_dir:
        _savefig(fig, save_dir, "results_overview.png")
    return fig


def plot_detection_by_size(size_detected: list[tuple], save_dir: str | None = None,
                           bins=(0, 10, 30, 100, 300, 1000, 10**9)):
    """Tasso di rilevamento per fascia di dimensione della lesione.

    Verifica quantitativamente l'ipotesi più importante dell'error analysis: le lesioni
    che sfuggono sono quelle piccole, che sono anche le più rilevanti clinicamente
    (le lesioni nuove).
    """
    sizes = np.array([s for s, _ in size_detected], dtype=float)
    det = np.array([d for _, d in size_detected], dtype=bool)
    labels, rates, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (sizes >= lo) & (sizes < hi)
        if m.sum() == 0:
            continue
        labels.append(f"{lo}-{hi if hi < 10**8 else '∞'}")
        rates.append(det[m].mean())
        counts.append(int(m.sum()))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(range(len(rates)), rates, color="tab:purple", alpha=0.85)
    for i, (b, c) in enumerate(zip(bars, counts)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"n={c}",
                ha="center", fontsize=9)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_xlabel("dimensione della lesione (voxel)"); ax.set_ylabel("tasso di rilevamento")
    ax.set_ylim(0, 1.12); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"Rilevamento per dimensione — {len(sizes)} lesioni totali")
    fig.tight_layout()
    if save_dir:
        _savefig(fig, save_dir, "detection_by_size.png")
    return fig


def _savefig(fig, out_dir: str, name: str):
    d = os.path.join(out_dir, "figures") if not out_dir.endswith("figures") else out_dir
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
