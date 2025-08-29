"""Minimal Vision-Transformer wrapper for EEG spectrograms.

Highlights
---------
* Accepts any timm ViT family model; we expose *model_name*, *img_size*,
  and *in_chans*.
* Replaces the classifier head so num_classes is task-specific.
* Optionally inserts a small "domain-adapter" MLP before the head.  This is
  cheap (~<1 M parameters) and can help cross-subject generalisation.

Typical usage
-------------
>>> from models.eeg_vit import build_model
>>> model = build_model('vit_small_patch16_224', num_classes=10, in_chans=128)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
import torch.nn as nn

try:
    import timm  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError('timm is required for eeg_vit models: pip install timm') from e

__all__ = ['DAEEGViT', 'build_model']


class DAEEGViT(nn.Module):
    """Domain-Adaptive EEG Vision Transformer.

    This is essentially a *timm* ViT backbone with:
      • customised input patch embedding (`in_chans = 128`)
      • replaced classification head (num_classes)
      • optional 2-layer MLP domain adapter in front of the head.
    """

    def __init__(self, *, model_name: str = 'vit_small_patch16_224',
                 num_classes: int = 10, img_size: int = 128,
                 in_chans: int = 128, domain_adapter: bool = True,
                 adapter_dim: int = 384,
                 drop_path_rate: float = 0.1,
                 attn_drop_rate: float = 0.0,
                 drop_rate: float = 0.1,
                 qkv_bias: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,  # remove original classifier
            in_chans=in_chans,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
        )
        embed_dim = self.backbone.num_features

        self.domain_adapter = None
        if domain_adapter:
            self.domain_adapter = nn.Sequential(
                nn.Linear(embed_dim, adapter_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(adapter_dim, embed_dim),
                nn.ReLU(),
            )
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor):  # (B, C, H, W)
        # timm ViT models expect channels-first images
        feat = self.backbone.forward_features(x)
        # timm ViT returns (B, seq, dim) if global_pool disabled; pool CLS token
        if feat.dim() == 3:
            feat = feat[:, 0]
        if self.domain_adapter is not None:
            feat = self.domain_adapter(feat)
        return self.head(feat)


# Expose new kwargs so the training script can forward them without surprises.
def build_model(model_name: str, num_classes: int, *,
                img_size: int = 128, in_chans: int = 128,
                domain_adapter: bool = True, adapter_dim: int = 384,
                drop_path_rate: float = 0.1,
                attn_drop_rate: float = 0.0,
                drop_rate: float = 0.1,
                qkv_bias: bool = True) -> nn.Module:
    """Factory helper to stay consistent with CLI/yaml parameters."""
    return DAEEGViT(model_name=model_name,
                    num_classes=num_classes,
                    img_size=img_size,
                    in_chans=in_chans,
                    domain_adapter=domain_adapter,
                    adapter_dim=adapter_dim,
                    drop_path_rate=drop_path_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    qkv_bias=qkv_bias) 