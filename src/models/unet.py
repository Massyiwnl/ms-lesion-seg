"""U-Net 2D parametrica, con Attention Gate e deep supervision opzionali.

Una sola implementazione con due interruttori (`attention`, `deep_supervision`) invece
di tre classi separate: così l'ablation è pulita, perché i quattro modelli confrontati
(U-Net / +DS / +Attention / +Attention+DS) condividono esattamente lo stesso codice e
la stessa inizializzazione. Qualsiasi differenza nei risultati è attribuibile al
componente attivato, non a dettagli implementativi diversi.

Struttura (features=[32,64,128,256,512], input 192x192):
    encoder  192 -> 96 -> 48 -> 24        (4 blocchi + maxpool)
    bottleneck                12x12
    decoder   24 -> 48 -> 96 -> 192       (convtranspose + skip)
"""
from __future__ import annotations
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Conv-BN-ReLU x2, il blocco base della U-Net."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention Gate additivo (Oktay et al., 2018).

    Il segnale del decoder (g) "interroga" la skip connection (x) e produce una mappa
    di coefficienti alpha in [0,1] con cui la skip viene pesata prima della
    concatenazione. Serve a sopprimere le regioni irrilevanti: utile con lesioni
    piccole e sparse, dove le skip a piena risoluzione portano molto background.
    """

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.W_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        alpha = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        return x * alpha, alpha


class UNet2D(nn.Module):
    """U-Net 2D.

    Args:
        in_channels: canali d'ingresso (n_modalità x slice 2.5D).
        out_channels: 1 = segmentazione binaria (logit, sigmoid applicata nella loss).
        features: larghezze dei livelli; l'ultima è il bottleneck.
        attention: attiva gli Attention Gate sulle skip.
        deep_supervision: aggiunge teste ausiliarie ai livelli intermedi del decoder.

    Forward:
        - deep_supervision attiva **e** modello in training -> lista di logit
          [piena risoluzione, 1/2, 1/4, 1/8] (la loss li pesa; vedi losses.py)
        - altrimenti -> un solo tensore [B, out_channels, H, W]
    """

    def __init__(self, in_channels: int, out_channels: int = 1,
                 features=(32, 64, 128, 256, 512), attention: bool = False,
                 deep_supervision: bool = False, dropout: float = 0.0):
        super().__init__()
        features = list(features)
        self.attention = attention
        self.deep_supervision = deep_supervision
        self.store_attention = False      # se True salva le mappe alpha (explainability)
        self.attention_maps: list = []

        # --- encoder ---
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features[:-1]:
            self.encoders.append(DoubleConv(ch, f, dropout))
            ch = f
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(features[-2], features[-1], dropout)

        # --- decoder ---
        self.ups = nn.ModuleList()
        self.gates = nn.ModuleList() if attention else None
        self.decoders = nn.ModuleList()
        ch = features[-1]
        for f in reversed(features[:-1]):
            self.ups.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            if attention:
                self.gates.append(AttentionGate(f, f, max(f // 2, 1)))
            self.decoders.append(DoubleConv(2 * f, f, dropout))
            ch = f

        # --- teste di segmentazione ---
        self.head = nn.Conv2d(features[0], out_channels, 1)
        if deep_supervision:
            # una testa ausiliaria per ogni livello del decoder tranne l'ultimo
            self.aux_heads = nn.ModuleList(
                [nn.Conv2d(f, out_channels, 1) for f in list(reversed(features[:-1]))[:-1]])

    def forward(self, x):
        if self.store_attention:
            self.attention_maps = []

        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)

        decoder_outputs = []
        for i, (up, dec) in enumerate(zip(self.ups, self.decoders)):
            x = up(x)
            skip = skips[-(i + 1)]
            if self.attention:
                skip, alpha = self.gates[i](x, skip)
                if self.store_attention:
                    self.attention_maps.append(alpha.detach())
            x = dec(torch.cat([skip, x], dim=1))
            decoder_outputs.append(x)

        main = self.head(decoder_outputs[-1])
        if self.deep_supervision and self.training:
            # dal più fine al più grossolano, coerente con loss.deep_supervision_weights
            aux = [h(o) for h, o in zip(reversed(self.aux_heads), reversed(decoder_outputs[:-1]))]
            return [main] + aux
        return main
