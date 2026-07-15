# MS Lesion Segmentation — MSLesSeg

Studio comparativo encoder–decoder per la segmentazione delle lesioni di sclerosi
multipla su MRI cerebrale multimodale (FLAIR / T1 / T2), in **2.5D**, eseguibile su
Google Colab. Progetto per il corso di Medical Imaging.

## Idea in breve
Non un solo modello, ma una **pipeline completa** + uno **studio comparativo controllato**
(architettura × modalità × loss) e l'**analisi** del modello migliore su tre assi:
robustezza al cambio di scanner (1.5T↔3T), fairness demografica, interpretabilità.

## Struttura del progetto
```
ms-lesion-seg/
├── configs/            # configurazioni YAML (base + un file per esperimento)
├── src/
│   ├── config.py       # caricamento/merge config (OmegaConf)
│   ├── utils.py        # seed, device, logging
│   ├── data/           # indicizzazione dataset + dataset 2.5D + transforms
│   ├── models/         # U-Net, Attention U-Net, factory
│   ├── losses.py       # Dice / Dice+Focal / Tversky (+ deep supervision)
│   ├── metrics.py      # Dice, IoU, HD95, lesion-wise
│   ├── regularization.py  # regolarizzazione dello Jacobiano
│   ├── engine.py       # loop di training/validazione
│   ├── evaluate.py     # valutazione su test + per sottogruppo
│   └── explain/        # Grad-CAM, mappe di attenzione
├── scripts/            # entry-point da riga di comando (00_inspect ... 05_explain)
├── notebooks/          # colab_runner.ipynb (driver Colab)
└── report/             # report LaTeX + model card (finale)
```

## Setup
### Colab (consigliato)
Apri `notebooks/colab_runner.ipynb` ed esegui le celle: monta Drive, installa le
dipendenze, clona/aggiorna il repo, esegue gli script.

### Locale
```
pip install -r requirements.txt
```

## Sistema di configurazione
Ogni esperimento è un piccolo file YAML che sovrascrive `configs/base.yaml`:
```
python scripts/02_train.py --config configs/exp_attunet_multimodal_ds.yaml \
       --override train.lr=0.0005 train.epochs=120
```
`model.in_channels: auto` viene risolto come `n_modalità × (2·context_slices + 1)`.

## Roadmap di sviluppo (checklist)
### Fase 0 — Fondamenta  [FATTO] (questo scaffold)
- [x] Struttura repo, ambiente, `.gitignore`, `requirements.txt`
- [x] Sistema di config (OmegaConf) + config base + config d'esempio
- [x] Utility (seed, device, logging)
- [x] Indicizzazione dataset flessibile + join metadati + split per paziente
- [x] Script di ispezione dati (`00_inspect_data.py`)

### Fase 1 — Dati
- [ ] Scaricare MSLesSeg su Drive; impostare `paths.data_root`
- [ ] Eseguire `00_inspect_data.py`, verificare `filename_pattern` e copertura modalità
- [ ] Reperire/creare `metadata.csv` (sesso, età, field strength per paziente)
- [ ] EDA: sbilanciamento di classe, lesion-load, visualizzazione MPR

### Fase 2 — Preprocessing & Dataset 2.5D
- [ ] `01_preprocess.py`: normalizzazione intensità per-modalità -> volumi .npy (memmap)
- [ ] Indice slice + filtro brain + oversampling slice con lesione
- [ ] `data/dataset.py`: Dataset 2.5D (canali = modalità × slice adiacenti)
- [ ] `data/transforms.py`: pipeline augmentation train/val (pattern append/extend)
- [ ] Sanity check: visualizzare un batch, verificare allineamento immagine/maschera

### Fase 3 — Modelli
- [ ] `models/unet.py`: U-Net baseline parametrica (in_channels configurabile)
- [ ] `models/attention_unet.py`: Attention U-Net + deep supervision
- [ ] `models/build.py`: factory da config (+ hook opzionale MONAI DynUNet/SwinUNETR)

### Fase 4 — Loss, metriche, engine
- [ ] `losses.py`: Dice / Dice+Focal / Tversky + wrapper deep supervision
- [ ] `metrics.py`: Dice, IoU, HD95, ASSD, lesion-wise TPR/FPR
- [ ] `regularization.py`: Jacobian (stima di Hutchinson)
- [ ] `engine.py`: loop con AMP, scheduler, early stopping, checkpoint, logging CSV
- [ ] `02_train.py`: entry-point di training

### Fase 5 — Esperimenti core
- [ ] Baseline U-Net (FLAIR) -> risultato di riferimento
- [ ] Ablation modalità: FLAIR vs FLAIR+T1+T2
- [ ] Ablation architettura: U-Net vs Attention U-Net (+deep supervision)
- [ ] Ablation loss: Dice vs Dice+Focal/Tversky
- [ ] Tabella riassuntiva dei risultati (Dice/IoU/HD95/lesion-wise)

### Fase 6 — Robustezza al dominio + regolarizzazione
- [ ] Split per field strength; train 3T / test 1.5T (e viceversa) -> misurare il calo
- [ ] Attivare bias-field augmentation e Jacobian -> misurare il recupero
- [ ] `03_evaluate.py`: valutazione cross-dominio

### Fase 7 — Analisi
- [ ] `04_fairness.py`: metriche per sottogruppo (sesso, età, scanner)
- [ ] `05_explain.py`: Grad-CAM + mappe di attenzione; error analysis (FP/FN, lesioni piccole)
- [ ] Figure qualitative (overlay predizione/GT)

### Fase 8 — Estensioni (opzionali)
- [ ] SwinUNETR / UNETR (confronto Transformer)
- [ ] MedSAM zero-shot come baseline foundation
- [ ] Analisi radiomica sulle lesioni segmentate (PyRadiomics)

### Fase 9 — Report & consegna
- [ ] Report LaTeX (intro clinica -> dati -> metodi -> risultati -> analisi -> limiti)
- [ ] Model card (uso previsto, dati, metriche, limiti, rischi)
- [ ] README finale, seed/config salvati, requirements congelati, notebook pulito
