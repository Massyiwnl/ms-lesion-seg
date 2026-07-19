"""Funzioni di costo per la segmentazione con forte sbilanciamento di classe.

Nel nostro dataset la lesione occupa lo 0.204% dei pixel (sfondo:lesione = 490:1).
Una cross-entropy semplice verrebbe minimizzata predicendo "tutto sfondo": per questo
si usano loss basate sulla sovrapposizione (Dice) o che ripesano i campioni difficili.

    dice        DiceLoss: ottimizza direttamente la metrica di valutazione.
    dice_focal  Dice + Focal: la Focal abbassa il peso dei pixel già classificati bene
                (l'oceano di sfondo) e concentra il gradiente sui casi difficili.
    dice_ce     Dice + Cross-Entropy: alternativa robusta (default di nnU-Net).
    tversky     Generalizza il Dice con pesi asimmetrici su FP e FN: con alpha<beta si
                penalizzano di più i falsi negativi, cioè le lesioni mancate.

ATTENZIONE — minimo degenere della Dice loss. Con `batch=False` (default di MONAI) il
Dice è mediato sulle singole slice: una slice SENZA lesione predetta vuota dà loss 0.
Se metà del batch è composta da slice vuote, predire "tutto sfondo" garantisce loss 0.5,
che può essere inferiore alla loss di un modello che sta davvero imparando: il training
collassa e la recall va a 0. Il fenomeno è amplificato dall'AMP, perché in float16 la
sigmoide di logit < -20 va in underflow a zero esatto, rendendo la ricompensa perfetta.
Per questo si usa `batch=True` (il Dice è calcolato sul batch aggregato, che contiene
sempre lesioni) e la loss è calcolata in float32 (vedi engine.py).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSupervisionLoss(nn.Module):
    """Somma pesata della loss sulle uscite a più risoluzioni.

    Dettaglio importante: la ground truth viene ridimensionata con **max-pooling**, non
    con interpolazione nearest. Con lesioni di pochi pixel, un nearest a 1/8 di
    risoluzione le cancellerebbe quasi tutte, e le teste ausiliarie riceverebbero una
    maschera vuota; il max-pooling garantisce che una lesione presente sopravviva.
    """

    def __init__(self, base_loss: nn.Module, weights=(1.0, 0.5, 0.25, 0.125)):
        super().__init__()
        self.base = base_loss
        self.weights = list(weights)

    def forward(self, preds, target):
        if not isinstance(preds, (list, tuple)):
            return self.base(preds, target)
        total, wsum = 0.0, 0.0
        for w, p in zip(self.weights, preds):
            if p.shape[-2:] == target.shape[-2:]:
                t = target
            else:
                t = F.adaptive_max_pool2d(target, output_size=p.shape[-2:])
            total = total + w * self.base(p, t)
            wsum += w
        return total / max(wsum, 1e-8)


def build_loss(cfg):
    """Costruisce il criterio di loss secondo cfg.loss (+ deep supervision)."""
    from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss, TverskyLoss

    name = str(cfg.loss.name).lower()
    sig = bool(cfg.loss.sigmoid)
    # batch=True: il Dice è calcolato sull'INTERO batch aggregato, non mediando i Dice
    # delle singole slice. Senza questo, una slice vuota predetta vuota vale loss 0 e il
    # modello impara a non predire nulla (minimo degenere). Vedi nota nel docstring.
    bd = bool(cfg.loss.get("batch_dice", True))

    if name == "dice":
        base = DiceLoss(sigmoid=sig, include_background=True, batch=bd)
    elif name in ("dice_focal", "dicefocal"):
        base = DiceFocalLoss(sigmoid=sig, include_background=True, batch=bd,
                             lambda_dice=1.0, lambda_focal=1.0, gamma=2.0)
    elif name in ("dice_ce", "dicece"):
        base = DiceCELoss(sigmoid=sig, include_background=True, batch=bd,
                          lambda_dice=1.0, lambda_ce=1.0)
    elif name == "tversky":
        base = TverskyLoss(sigmoid=sig, include_background=True, batch=bd,
                           alpha=float(cfg.loss.tversky_alpha),
                           beta=float(cfg.loss.tversky_beta))
    else:
        raise ValueError(f"Loss sconosciuta: {cfg.loss.name}")

    if bool(cfg.model.deep_supervision):
        return DeepSupervisionLoss(base, list(cfg.loss.deep_supervision_weights))
    return base
