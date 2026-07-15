"""Ispezione e sanity-check del dataset MSLesSeg.

Esegui DOPO aver scaricato il dataset e impostato `paths.data_root`:
    python scripts/00_inspect_data.py --config configs/base.yaml
    python scripts/00_inspect_data.py --split test

Verifica: numero pazienti/casi, esempi di filename (per validare filename_pattern),
copertura modalità, shape/spacing/intensità di un volume campione, sbilanciamento di
classe della maschera, e distribuzione dei sottogruppi (se c'è metadata_csv).
"""
import argparse, os, sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config
from src.data.indexing import discover_cases, attach_metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--split", default="train", help="train | test | '' (nessuna sottocartella)")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)

    root = os.path.join(cfg.paths.data_root, args.split) if args.split else cfg.paths.data_root
    if not os.path.isdir(root):
        print(f"[i] {root} inesistente: uso data_root ({cfg.paths.data_root})")
        root = cfg.paths.data_root
    print(f"[*] Scansione: {root}\n")

    cases = discover_cases(root, cfg.data.filename_pattern, cfg.data.modality_aliases)
    cases = attach_metadata(cases, cfg.paths.metadata_csv)

    if not cases:
        print("[-] Nessun caso trovato. Controlla `paths.data_root` e `data.filename_pattern`.")
        return

    patients = sorted({c["patient"] for c in cases})
    print(f"[+] Pazienti: {len(patients)} | Casi (paziente x timepoint): {len(cases)}\n")

    print("--- Esempi (primi 3 casi) ---")
    for c in cases[:3]:
        print(f"  paziente={c['patient']}  timepoint={c['timepoint']}  "
              f"modalità={sorted(c['images'])}  mask={'sì' if c['mask'] else 'NO'}")

    print("\n--- Copertura modalità ---")
    cov = Counter(tuple(sorted(c["images"])) for c in cases)
    for k, v in cov.items():
        print(f"  {k}: {v} casi")
    n_mask = sum(1 for c in cases if c["mask"])
    print(f"  Casi con maschera: {n_mask}/{len(cases)}")

    # Ispezione di un volume campione
    import numpy as np, nibabel as nib
    sample = next((c for c in cases if c["mask"]), cases[0])
    print(f"\n--- Volume campione: paziente {sample['patient']} tp {sample['timepoint']} ---")
    shapes = {}
    for mod, p in sorted(sample["images"].items()):
        img = nib.load(p)
        arr = np.asarray(img.dataobj)  # lettura lazy: veloce e senza caricare tutto in RAM
        shapes[mod] = arr.shape
        zooms = tuple(round(float(z), 2) for z in img.header.get_zooms()[:3])
        print(f"  {mod}: shape={arr.shape} spacing={zooms} dtype={arr.dtype} "
              f"range=[{float(arr.min()):.1f}, {float(arr.max()):.1f}]")
    if len(set(shapes.values())) == 1:
        print("  [OK] Tutte le modalità hanno la stessa shape (co-registrazione plausibile).")
    else:
        print("  [!] Shape diverse tra modalità: verificare registrazione/risoluzione.")

    if sample["mask"]:
        m = np.asarray(nib.load(sample["mask"]).dataobj)
        uniq = np.unique(m)
        pos, tot = float((m > 0).sum()), float(m.size)
        head = ", ".join(str(u) for u in uniq[:5]) + ("..." if len(uniq) > 5 else "")
        print(f"  maschera: valori=[{head}]  lesione={pos / tot * 100:.3f}% dei voxel "
              f"(-> forte sbilanciamento di classe)")

    # Sottogruppi
    if cfg.paths.metadata_csv and any("sex" in c for c in cases):
        print("\n--- Sottogruppi (da metadata) ---")
        for key in ("sex", "age_band", "field_strength"):
            cc = Counter(str(c.get(key, "NA")) for c in cases)
            print(f"  {key}: {dict(cc)}")
    else:
        print("\n[i] metadata_csv non impostato: fairness e split di dominio richiedono i "
              "metadati per paziente (sesso, età, field strength).")


if __name__ == "__main__":
    main()
