"""Preprocessing offline del dataset MSLesSeg (Fase 2).

    python scripts/01_preprocess.py --split train
    python scripts/01_preprocess.py --split test

Normalizza le intensità per modalità, riorienta con l'asse-slice per primo, ritaglia al
cervello, porta le slice a `data.spatial_size` e salva .npy + meta.json per ogni caso,
più un indice globale index_<split>.json in `paths.processed_root`.
"""
import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config
from src.data.indexing import discover_cases, attach_metadata
from src.data.preprocessing import process_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--split", default="train", help="train | test")
    ap.add_argument("--limit", type=int, default=None, help="processa solo N casi (test rapido)")
    ap.add_argument("--force", action="store_true", help="riprocessa anche i casi già presenti")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    src_root = os.path.join(cfg.paths.data_root, args.split)
    if not os.path.isdir(src_root):
        src_root = cfg.paths.data_root
    out_root = os.path.join(cfg.paths.processed_root, args.split)
    os.makedirs(out_root, exist_ok=True)

    cases = discover_cases(src_root, cfg.data.filename_pattern, cfg.data.modality_aliases)
    cases = attach_metadata(cases, cfg.paths.metadata_csv)
    cases.sort(key=lambda c: (c["patient"], c["timepoint"]))
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print(f"[-] Nessun caso trovato in {src_root}")
        return

    print(f"[*] Sorgente : {src_root}")
    print(f"[*] Output   : {out_root}")
    print(f"[*] Casi     : {len(cases)} | modalità: {list(cfg.data.modalities)} "
          f"| slice {list(cfg.data.spatial_size)}\n")

    metas, t0 = [], time.time()
    for i, case in enumerate(cases, 1):
        name = f"{case['patient']}_{case['timepoint']}"
        out_dir = os.path.join(out_root, name)
        meta_path = os.path.join(out_dir, "meta.json")
        if os.path.exists(meta_path) and not args.force:
            with open(meta_path) as f:
                metas.append(json.load(f))
            print(f"  [{i}/{len(cases)}] {name}: già presente (salto)")
            continue
        meta = process_case(case, cfg, out_dir)
        if meta is None:
            print(f"  [{i}/{len(cases)}] {name}: MODALITÀ MANCANTI -> saltato")
            continue
        metas.append(meta)
        pos = int((__import__("numpy").array(meta["lesion_voxels"]) > 0).sum())
        print(f"  [{i}/{len(cases)}] {name}: {meta['n_slices']} slice "
              f"({pos} con lesione) | lesione tot={meta['lesion_total']} voxel")

    index_path = os.path.join(cfg.paths.processed_root, f"index_{args.split}.json")
    with open(index_path, "w") as f:
        json.dump({"split": args.split, "cases": metas}, f, indent=1)

    size_mb = sum(os.path.getsize(os.path.join(dp, fn))
                  for dp, _, fns in os.walk(out_root) for fn in fns) / 1e6
    n_pat = len({m["patient"] for m in metas})
    tot_slices = sum(m["n_slices"] for m in metas)
    pos_slices = sum(int(v > 0) for m in metas for v in m["lesion_voxels"])
    print(f"\n[+] Completato in {time.time() - t0:.1f}s")
    print(f"    Casi: {len(metas)} | pazienti: {n_pat}")
    print(f"    Slice totali: {tot_slices} | con lesione: {pos_slices} "
          f"({pos_slices / max(tot_slices, 1) * 100:.1f}%)")
    print(f"    Spazio su disco: {size_mb:.0f} MB")
    print(f"    Indice: {index_path}")


if __name__ == "__main__":
    main()
