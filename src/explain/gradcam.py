"""Explainability: Grad-CAM e mappe delle attention gate (Fase 7).

Perché serve. Le metriche dicono QUANTO il modello sbaglia, le figure qualitative
dicono DOVE, ma nessuna delle due dice SU COSA il modello si stia basando. Una rete
può ottenere un buon Dice guardando la cosa sbagliata (per esempio un artefatto
sistematico o la posizione anatomica invece dell'intensità della lesione): è il tipo
di problema che emerge solo quando il modello viene messo su dati di un altro ospedale.

Grad-CAM per la SEGMENTAZIONE. La formulazione originale è per la classificazione:
si prende il logit della classe e se ne calcola il gradiente rispetto alle feature map
di uno strato convoluzionale. Qui l'uscita non è uno scalare ma una mappa, quindi si
sceglie uno scalare bersaglio: la somma dei logit sui pixel predetti come lesione
(`mode="predicted"`) oppure sui pixel della ground truth (`mode="target"`, utile per
chiedersi "cosa avrebbe dovuto guardare per trovare QUESTA lesione").

    peso del canale c:  w_c = media spaziale di  dScore/dA_c
    mappa:              CAM = ReLU( somma_c  w_c * A_c )

La ReLU tiene solo i contributi che spingono VERSO la classe lesione.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM2D:
    """Grad-CAM per reti di segmentazione 2D.

    Uso:
        cam = GradCAM2D(model, "bottleneck")
        heat = cam(x)            # [B, 1, H, W] normalizzata in [0,1]
        cam.close()
    """

    def __init__(self, model, target_layer: str):
        self.model = model
        self.layer_name = target_layer
        modules = dict(model.named_modules())
        if target_layer not in modules:
            raise KeyError(f"Strato '{target_layer}' non trovato. Disponibili (esempi): "
                           f"{[n for n in modules if n][:12]}")
        self.layer = modules[target_layer]
        self.acts = None
        self.grads = None
        self.degenerate = None
        self._h1 = self.layer.register_forward_hook(self._save_acts)
        self._h2 = self.layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, module, inp, out):
        self.acts = out.detach()

    def _save_grads(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def __call__(self, x, target=None, mode: str = "predicted", threshold: float = 0.5):
        """Ritorna la mappa Grad-CAM [B, 1, H, W] normalizzata per campione."""
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        x = x.clone().requires_grad_(True)
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]

        if mode == "target" and target is not None:
            mask = (target > 0.5).float()
        else:
            mask = (torch.sigmoid(out) > threshold).float()
        if mask.sum() == 0:                       # nessun pixel selezionato: usa tutta l'uscita
            mask = torch.ones_like(out)
        score = (out * mask).sum()
        score.backward()

        w = self.grads.mean(dim=(2, 3), keepdim=True)          # [B, C, 1, 1]
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))  # [B, 1, h, w]
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        b = cam.shape[0]
        flat = cam.view(b, -1)
        mn = flat.min(dim=1)[0].view(b, 1, 1, 1)
        mx = flat.max(dim=1)[0].view(b, 1, 1, 1)
        # Se dopo la ReLU la mappa è identicamente nulla, quello strato non dà contributi
        # positivi al punteggio: la mappa non è interpretabile e va segnalata, non
        # normalizzata (dividere per ~0 produrrebbe rumore dall'aspetto significativo).
        self.degenerate = (mx.view(b) < 1e-12).cpu().numpy()
        cam = (cam - mn) / (mx - mn + 1e-8)
        cam[torch.from_numpy(self.degenerate).to(cam.device).view(b, 1, 1, 1).expand_as(cam)] = 0.0

        self.model.zero_grad(set_to_none=True)
        if was_training:
            self.model.train()
        return cam.detach()

    def close(self):
        self._h1.remove()
        self._h2.remove()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def suggest_layers(model) -> list[str]:
    """Strati convoluzionali sensati su cui calcolare la Grad-CAM, dal profondo al fine."""
    names = dict(model.named_modules())
    out = []
    for cand in ("bottleneck", "decoders.0", "decoders.1", "decoders.2", "decoders.3"):
        if cand in names:
            out.append(cand)
    return out or [n for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)][-4:]


@torch.no_grad()
def attention_maps(model, x) -> list[np.ndarray]:
    """Coefficienti alpha delle attention gate, dal livello più grossolano al più fine.

    Restituisce una lista di array [H, W] già riportati alla risoluzione dell'input.
    Vale solo per la Attention U-Net; per gli altri modelli ritorna lista vuota.
    """
    if not getattr(model, "attention", False):
        return []
    was = model.store_attention
    model.store_attention = True
    model.eval()
    model(x)
    model.store_attention = was
    maps = []
    for a in model.attention_maps:
        up = F.interpolate(a.float(), size=x.shape[-2:], mode="bilinear", align_corners=False)
        maps.append(up[0, 0].cpu().numpy())
    return maps
