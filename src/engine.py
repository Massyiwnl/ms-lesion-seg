"""Loop di training e validazione.

Scelte principali:
- **AMP (mixed precision)**: quasi raddoppia la velocità e dimezza la memoria su GPU
  con Tensor Core (T4/RTX). Viene disattivata automaticamente se è attiva la
  regolarizzazione dello Jacobiano, che richiede un doppio backward incompatibile
  con il gradient scaling.
- **Dice per volume** come metrica di selezione del modello: mediare i Dice delle
  singole slice darebbe un numero ottimista e instabile (vedi metrics.py).
- **Early stopping + best checkpoint**: si salva il modello con il Dice di validazione
  migliore, non l'ultimo.
- **Log CSV**: una riga per epoca, per poter tracciare le curve di apprendimento nel
  report senza dipendere da TensorBoard.
"""
from __future__ import annotations
import csv
import json
import os
import time
import numpy as np
import torch

from src.metrics import CaseAccumulator


def _to_fp32(out):
    """Riporta i logit a float32 prima della loss.

    Sotto AMP i logit sono in float16: la sigmoide di valori < -20 va in underflow a
    zero esatto, e la Dice loss su una slice vuota diventa esattamente 0, premiando la
    predizione vuota. Calcolando la loss in float32 il problema sparisce.
    """
    if isinstance(out, (list, tuple)):
        return [o.float() for o in out]
    return out.float()


def _autocast(device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.amp.autocast("cpu", enabled=False)


def build_optimizer(cfg, model):
    name = str(cfg.train.optimizer).lower()
    lr, wd = float(cfg.train.lr), float(cfg.train.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                               momentum=0.99, nesterov=True)
    raise ValueError(f"Optimizer sconosciuto: {cfg.train.optimizer}")


def build_scheduler(cfg, optimizer):
    name = str(cfg.train.scheduler).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.train.epochs))
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                          factor=0.5, patience=8)
    return None


def train_one_epoch(model, loader, criterion, optimizer, device, cfg, scaler=None):
    model.train()
    use_jac = bool(cfg.regularization.jacobian)
    amp = bool(cfg.train.amp) and device.type == "cuda" and not use_jac
    lam = float(cfg.regularization.jacobian_lambda)
    losses, regs = [], []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_jac:
            from src.regularization import jacobian_regularization
            x.requires_grad_(True)
            out = model(x)
            loss = criterion(_to_fp32(out), y)
            reg = jacobian_regularization(out, x, int(cfg.regularization.jacobian_n_proj))
            total = loss + lam * reg
            total.backward()
            regs.append(float(reg.detach()))
            if cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
            optimizer.step()
        else:
            with _autocast(device, amp):
                out = model(x)
            loss = criterion(_to_fp32(out), y)   # loss sempre in float32
            if scaler is not None and amp:
                scaler.scale(loss).backward()
                if cfg.train.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.train.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
                optimizer.step()
        losses.append(float(loss.detach()))

    return {"loss": float(np.mean(losses)) if losses else float("nan"),
            "jac": float(np.mean(regs)) if regs else None}


@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    """Dice aggregato per caso (per volume) + loss media."""
    model.eval()
    acc = CaseAccumulator()
    thr = float(cfg.eval.threshold)
    losses = []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with _autocast(device, bool(cfg.train.amp) and device.type == "cuda"):
            logits = model(x)
        logits = logits.float()
        loss = criterion(logits, y)
        losses.append(float(loss))
        prob = torch.sigmoid(logits)
        acc.update((prob > thr).cpu().numpy(), y.cpu().numpy(), list(batch["case"]))

    summ = acc.summary()
    summ["loss"] = float(np.mean(losses)) if losses else float("nan")
    return summ


def fit(cfg, model, train_loader, val_loader, device=None, logger=None):
    """Ciclo completo di addestramento con early stopping e salvataggio del best."""
    from src.config import save_config
    from src.losses import build_loss
    from src.utils import get_device, setup_logger, count_parameters

    device = device or get_device()
    out_dir = str(cfg.project.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    logger = logger or setup_logger("train", os.path.join(out_dir, "train.log"))
    save_config(cfg, os.path.join(out_dir, "config_resolved.yaml"))

    model = model.to(device)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    use_amp = bool(cfg.train.amp) and device.type == "cuda" and not bool(cfg.regularization.jacobian)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    logger.info(f"Device: {device} | parametri: {count_parameters(model)/1e6:.2f}M "
                f"| AMP: {use_amp} | loss: {cfg.loss.name} | arch: {cfg.model.arch}")
    logger.info(f"Batch train: {len(train_loader)} | batch val: {len(val_loader)}")

    csv_path = os.path.join(out_dir, "history.csv")
    fields = ["epoch", "train_loss", "val_loss", "val_dice", "val_dice_median",
              "val_precision", "val_recall", "lr", "sec"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best, best_epoch, patience = -1.0, -1, int(cfg.train.early_stopping_patience)
    collapsed = 0
    history = []

    for epoch in range(1, int(cfg.train.epochs) + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, cfg, scaler)

        row = {"epoch": epoch, "train_loss": round(tr["loss"], 5), "val_loss": "",
               "val_dice": "", "val_dice_median": "", "val_precision": "", "val_recall": "",
               "lr": optimizer.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)}

        if epoch % int(cfg.train.val_interval) == 0:
            va = validate(model, val_loader, criterion, device, cfg)
            row.update({"val_loss": round(va["loss"], 5),
                        "val_dice": round(va["dice_mean"], 5),
                        "val_dice_median": round(va["dice_median"], 5),
                        "val_precision": round(va["precision_global"], 5),
                        "val_recall": round(va["recall_global"], 5)})
            msg = (f"Epoca {epoch:3d}/{cfg.train.epochs} | train {tr['loss']:.4f} | "
                   f"val {va['loss']:.4f} | Dice {va['dice_mean']:.4f} "
                   f"(mediana {va['dice_median']:.4f}) | P {va['precision_global']:.3f} "
                   f"R {va['recall_global']:.3f} | {row['sec']}s")
            if tr["jac"] is not None:
                msg += f" | jac {tr['jac']:.4f}"
            logger.info(msg)

            if va["recall_global"] == 0.0 or np.isnan(va["precision_global"]):
                collapsed += 1
                if collapsed == 2:
                    logger.warning("  [!] COLLASSO: il modello non predice più alcuna "
                                   "lesione (recall=0). Verifica loss.batch_dice=true e "
                                   "considera loss.name=dice_ce o un learning rate minore.")
            else:
                collapsed = 0

            if va["dice_mean"] > best:
                best, best_epoch = va["dice_mean"], epoch
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_dice": best, "cfg": str(cfg)},
                           os.path.join(out_dir, "best.pth"))
                logger.info(f"  -> nuovo best: Dice {best:.4f} (salvato best.pth)")

            if scheduler is not None:
                scheduler.step(va["dice_mean"]) if str(cfg.train.scheduler) == "plateau" \
                    else scheduler.step()
        else:
            logger.info(f"Epoca {epoch:3d}/{cfg.train.epochs} | train {tr['loss']:.4f} "
                        f"| {row['sec']}s")
            if scheduler is not None and str(cfg.train.scheduler) != "plateau":
                scheduler.step()

        history.append(row)
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        if patience > 0 and best_epoch > 0 and (epoch - best_epoch) >= patience:
            logger.info(f"Early stopping: nessun miglioramento da {patience} epoche.")
            break

    torch.save({"model": model.state_dict(), "epoch": epoch}, os.path.join(out_dir, "last.pth"))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"best_val_dice": best, "best_epoch": best_epoch,
                   "epochs_run": epoch, "output_dir": out_dir}, f, indent=1)
    logger.info(f"Fine. Miglior Dice di validazione: {best:.4f} (epoca {best_epoch})")
    return {"best_val_dice": best, "best_epoch": best_epoch, "history": history}
