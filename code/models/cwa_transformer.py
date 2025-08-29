"""Minimal self-contained implementation of the Channel-wise Auto-Encoder + Transformer
(CwA-Transformer) model described in arXiv:2412.14522.

This is **not** an official reproduction; it is a pragmatic subset that is
sufficient for our landing-digit smoke-tests:
  • depth-wise (grouped) 1-D conv per channel for latent extraction
  • sequence dimension = time-steps after down-sampling
  • Transformer encoder blocks over the sequence (multi-head attention)

Input shape expected by forward:  (B, C, T)
Output: logits (B, num_classes)
"""
from __future__ import annotations

import math
from typing import Optional

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    SDP_IS_AVAILABLE = True
except ImportError:
    SDP_IS_AVAILABLE = False

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (same as Vaswani 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        l = x.size(1)
        return x + self.pe[:, :l, :]


class CwaTransformer(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        enc_kernel_size: int = 7,
        enc_latent_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2,
        enc_stride: int = 2,
    ) -> None:
        super().__init__()

        padding = enc_kernel_size // 2
        # Channel-wise encoder implemented as grouped conv:
        self.encoder = nn.Conv1d(
            in_channels,
            in_channels,  # keep channel dim
            kernel_size=enc_kernel_size,
            stride=enc_stride,  # configurable down-sample factor
            padding=padding,
            groups=in_channels,  # each channel independent
            bias=False,
        )

        self.norm_enc = nn.BatchNorm1d(in_channels)

        d_model = in_channels  # token dimension = #channels (128 for us)
        self.pos_enc = _PositionalEncoding(d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (B, C, T)
        # Channel-wise conv → (B, C, T')
        x = self.encoder(x)
        x = self.norm_enc(x)
        x = x.permute(0, 2, 1)  # (B, T', C)  ‑> sequence first

        x = self.pos_enc(x)
        # Use the explicit SDP backend context manager if available
        if SDP_IS_AVAILABLE:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                x = F.scaled_dot_product_attention(x, x, x, is_causal=False)
        else:
            # Fallback for older PyTorch versions
            x = F.scaled_dot_product_attention(x, x, x, is_causal=False)

        x = self.transformer(x)  # (B, T', C)

        # Global average pooling over the sequence tokens
        x = x.mean(dim=1)
        return self.head(x) 