"""Aggiorna i metadati dei casi già preprocessati, senza ricalcolare i volumi (Fase 6).

    python scripts/01b_refresh_metadata.py --split train
    python scripts/01b_refresh_metadata.py --split test

Rilegge `clinical_data.csv` e `patient_scanners_info_FLAIR.csv` e riscrive i campi
demografici e di acquisizione dentro ogni `meta.json` e dentro `index_<split>.json`.
Utile quando si aggiungono metadati (qui: vendor / spessore di slice) dopo aver già
eseguito il preprocessing: i file .npy non vengono toccati, dura pochi secondi.
"""
import argparse
import json
import os
import sys
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import load_config
from src.data.indexing import attach_metadata, attach_scanner_metadata

FIELDS = ("sex", "age", "age_band", "ms_type", "edss", "lesion_volume",
          "vendor", "scanner_model", "slice_thickness", "thickness_band", "pixel_spacing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "base.yaml"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    index_path = os.path.join(cfg.paths.processed_root, f"index_{args.split}.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Indice non trovato: {index_path}")
    with open(index_path) as f:
        metas = json.load(f)["cases"]

    # riusa le stesse funzioni del preprocessing
    stub = [{"patient": m["patient"], "timepoint": m["timepoint"]} for m in metas]
    stub = attach_metadata(stub, cfg.paths.metadata_csv)
    stub = attach_scanner_metadata(stub, cfg.paths.get("scanner_csv", None))

    n_up = 0
    for meta, info in zip(metas, stub):
        for k in FIELDS:
            if k in info:
                meta[k] = info[k]
        mp = os.path.join(meta["dir"], "meta.json")
        if os.path.exists(mp):
            with open(mp) as f:
                on_disk = json.load(f)
            on_disk.update({k: meta[k] for k in FIELDS if k in meta})
            with open(mp, "w") as f:
                json.dump(on_disk, f, indent=1)
            n_up += 1

    with open(index_path, "w") as f:
        json.dump({"split": args.split, "cases": metas}, f, indent=1)

    print(f"[+] Aggiornati {n_up}/{len(metas)} casi dello split '{args.split}'\n")
    for key in ("sex", "age_band", "vendor", "thickness_band", "scanner_model"):
        c = Counter(str(m.get(key, "unknown")) for m in metas)
        print(f"  {key:16s}: {dict(sorted(c.items(), key=lambda x: -x[1]))}")
    miss = [m["patient"] for m in metas if m.get("thickness_band", "unknown") == "unknown"]
    if miss:
        print(f"\n  [!] {len(miss)} casi senza metadati di acquisizione: {sorted(set(miss))}")
        print("      Verranno esclusi dagli esperimenti di dominio.")


if __name__ == "__main__":
    main()
