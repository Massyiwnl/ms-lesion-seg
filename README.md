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

**Prima cosa da fare su una macchina nuova**: copia `configs/local.yaml.example` in
`configs/local.yaml` e metti i tuoi path. Quel file è in `.gitignore`, quindi non
finisce su GitHub e **non viene mai sovrascritto** quando aggiorni gli altri file.

Precedenza (dal più debole al più forte):
```
configs/base.yaml  ->  config d'esperimento  ->  configs/local.yaml  ->  --override da CLI
```
Così lo stesso comando gira identico su PC e su Colab: cambia solo `local.yaml`.

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

### Fase 2 — Preprocessing & Dataset 2.5D  [FATTO]
- [x] `01_preprocess.py`: normalizzazione intensità per-modalità -> volumi .npy (memmap)
- [x] Indice slice + filtro brain + oversampling slice con lesione
- [x] `data/dataset.py`: Dataset 2.5D (canali = modalità × slice adiacenti)
- [x] `data/transforms.py`: pipeline augmentation train/val (pattern append/extend)
- [x] Sanity check: visualizzare un batch, verificare allineamento immagine/maschera

### Fase 3 — Modelli  [FATTO]
- [x] `models/unet.py`: U-Net baseline parametrica (in_channels configurabile)
- [x] `models/attention_unet.py`: Attention U-Net + deep supervision
- [x] `models/build.py`: factory da config (+ hook opzionale MONAI DynUNet/SwinUNETR)

**Statistiche del dataset (train, 93 casi):** 15 044 slice utili, 6 782 con lesione
(45.1%); a livello di pixel la lesione è lo **0.204%** (sfondo:lesione = 490:1).
Carico lesionale per caso: mediana 6 217 mm3, min 744, max 72 872 (98x).

### Fase 4 — Loss, metriche, engine  [FATTO]
- [x] `losses.py`: Dice / Dice+Focal / Tversky + wrapper deep supervision
- [x] `metrics.py`: Dice, IoU, HD95, ASSD, lesion-wise TPR/FPR
- [x] `regularization.py`: Jacobian (stima di Hutchinson)
- [x] `engine.py`: loop con AMP, scheduler, early stopping, checkpoint, logging CSV
- [x] `02_train.py`: entry-point di training

Il Dice di validazione è calcolato **per volume** (TP/FP/FN accumulati per caso), non
mediando i Dice delle singole slice: è la metrica usata dai benchmark e non è gonfiata
dalle slice facili.

### Nota tecnica — collasso della Dice loss (risolto)
Al primo training la baseline è collassata all'epoca 9: recall a 0, nessuna lesione
predetta, train loss ferma a 0.50. Causa: con `batch=False` (default MONAI) il Dice è
mediato per slice e una slice vuota predetta vuota vale loss 0; con `pos_neg_ratio=0.5`
predire "tutto sfondo" garantisce 0.50, meglio dello 0.72 di un modello che sta
imparando. L'AMP amplifica il problema: in float16 la sigmoide di logit < -20 va in
underflow a zero esatto, rendendo la ricompensa perfetta. Fix: `loss.batch_dice: true`
e loss calcolata sempre in float32 (`_to_fp32` in engine.py). Aggiunto anche un
rilevatore che avvisa se la recall resta a 0 per due validazioni.

### Fase 5 — Esperimenti core  [strumenti pronti]
- [x] `03_evaluate.py`: valutazione per volume su test (Dice/IoU/HD95/ASSD/lesion-wise)
- [x] `04_compare.py`: tabella comparativa + Wilcoxon appaiato vs baseline
**Baseline completata (E0)**: U-Net, FLAIR, Dice -> val 0.7919 (epoca 46),
**test 0.6696 ± 0.1083** su 22 pazienti. In linea con le baseline 3D pubblicate
(UNETR 0.642, MSSegDiff 0.685). Lesion-wise TPR 0.771 (665/863 lesioni trovate),
mediana 10 falsi positivi per paziente. Il Dice correla col carico lesionale
(Pearson su log-volume r=+0.50, p=0.018): 0.622 sui pazienti con poche lesioni contro
0.717 su quelli con molte.

Piano ablation (una variabile alla volta rispetto a E0, stesso seed e stesso split):
| ID | Config | Variabile isolata |
|---|---|---|
| E0 | `exp_unet_flair.yaml` | baseline |
| E1 | `exp_unet_multimodal.yaml` | modalità: +T1+T2 |
| E2 | `exp_attunet_flair.yaml` | architettura: attention gate |
| E3 | `exp_unet_flair_ds.yaml` | deep supervision |
| E4 | `exp_unet_flair_focal.yaml` | loss: Dice+Focal |
| E5 | `exp_unet_flair_tversky.yaml` | loss: Tversky |
| E6 | `exp_best.yaml` | combinazione delle scelte vincenti |

- [x] Baseline U-Net (FLAIR) -> risultato di riferimento
- [ ] Ablation modalità: FLAIR vs FLAIR+T1+T2
- [ ] Ablation architettura: U-Net vs Attention U-Net (+deep supervision)
- [ ] Ablation loss: Dice vs Dice+Focal/Tversky
- [ ] Tabella riassuntiva dei risultati

Con 22 pazienti di test una differenza di Dice di pochi punti può essere rumore: per
questo il confronto include un **test di Wilcoxon appaiato** sui Dice per caso.

### Fase 6 — Robustezza al dominio + regolarizzazione
**Il dataset NON riporta il field strength.** `patient_scanners_info_*.csv` contiene
produttore, modello e parametri di sequenza, ma non il tag DICOM MagneticFieldStrength;
dedurre 1.5T/3T dal modello non è affidabile (Achieva, Ingenia, Signa HDxt esistono in
entrambe le versioni). L'asse di dominio è quindi ridefinito su basi verificabili:

- **Primario — spessore di slice** (protocollo di acquisizione): thin <=2.5mm vs
  thick >=3mm. Train su thick: 57 casi / 30 pazienti. Test: 10 casi in-dominio (thick)
  contro 12 fuori dominio (thin) — due lati ben bilanciati.
- **Secondario — produttore**: il training è quasi tutto Philips (84/93); il test ha
  16 Philips e 6 non-Philips (5 GE + 1 Siemens). Valutazione stratificata, senza
  training aggiuntivo.

- [x] Metadati di acquisizione agganciati (`attach_scanner_metadata`)
- [x] `01b_refresh_metadata.py`: aggiorna i meta senza rifare il preprocessing
- [x] Filtro di dominio nel training (`domain.split_by` / `train_domain`)
- [ ] F6-A: train su thick -> misurare il calo su thin
- [ ] F6-B: stesso training + bias-field aug + Jacobian -> misurare il recupero

### Fase 7 — Analisi
- [x] Metriche per sottogruppo (sesso, età, vendor, spessore) -> `results_by_subgroup.csv`
- [x] `05_visualize.py` + `src/visualize.py`: pacchetto di figure per il report
      (curve di apprendimento, overlay TP/FN/FP, zoom sulle lesioni per dimensione,
      distribuzione del Dice, rilevamento per dimensione della lesione)
- [x] Sezione 8 del notebook: esplorazione interattiva con slider sulle slice
- [ ] Grad-CAM + mappe di attenzione (richiede il modello Attention U-Net)
- [ ] Error analysis scritta nel report

### Fase 8 — Estensioni (opzionali)
- [ ] SwinUNETR / UNETR (confronto Transformer)
- [ ] MedSAM zero-shot come baseline foundation
- [ ] Analisi radiomica sulle lesioni segmentate (PyRadiomics)

### Fase 9 — Report & consegna
- [ ] Report LaTeX (intro clinica -> dati -> metodi -> risultati -> analisi -> limiti)
- [ ] Model card (uso previsto, dati, metriche, limiti, rischi)
- [ ] README finale, seed/config salvati, requirements congelati, notebook pulito
