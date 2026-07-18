"""Factory dei modelli a partire dal config.

    unet            -> U-Net 2D baseline
    attention_unet  -> Attention U-Net 2D (+ deep supervision opzionale)
    dynunet         -> MONAI DynUNet (stile nnU-Net, deep supervision nativa)
    swin_unetr      -> MONAI SwinUNETR (Transformer gerarchico; stretch, Fase 8)
"""
from __future__ import annotations


def build_model(cfg):
    """Istanzia il modello descritto da cfg.model."""
    arch = str(cfg.model.arch).lower()
    in_ch = int(cfg.model.in_channels)
    out_ch = int(cfg.model.out_channels)
    feats = list(cfg.model.features)
    ds = bool(cfg.model.deep_supervision)
    drop = float(cfg.model.dropout)

    if arch == "unet":
        from src.models.unet import UNet2D
        return UNet2D(in_ch, out_ch, feats, attention=False, deep_supervision=ds, dropout=drop)

    if arch in ("attention_unet", "attunet"):
        from src.models.attention_unet import AttentionUNet2D
        return AttentionUNet2D(in_ch, out_ch, feats, deep_supervision=ds, dropout=drop)

    if arch == "dynunet":
        from monai.networks.nets import DynUNet
        n = len(feats)
        return DynUNet(spatial_dims=2, in_channels=in_ch, out_channels=out_ch,
                       kernel_size=[3] * n, strides=[1] + [2] * (n - 1),
                       upsample_kernel_size=[2] * (n - 1),
                       filters=feats, res_block=True,
                       deep_supervision=ds, deep_supr_num=max(1, min(3, n - 2)))

    if arch == "swin_unetr":
        from monai.networks.nets import SwinUNETR
        try:
            return SwinUNETR(in_channels=in_ch, out_channels=out_ch, spatial_dims=2,
                             feature_size=24, use_checkpoint=True)
        except TypeError:   # firme diverse tra versioni di MONAI
            return SwinUNETR(img_size=tuple(cfg.data.spatial_size), in_channels=in_ch,
                             out_channels=out_ch, spatial_dims=2, feature_size=24,
                             use_checkpoint=True)

    raise ValueError(f"Architettura sconosciuta: {cfg.model.arch}")
