"""
FNO + U-Net Decoder model for phase-resolved sea-surface rollout prediction.

Recommended location in your project:
    neuralop/models/fno_unet_decoder.py

Input:
    x:      (B, input_steps, H, W)
Output:
    pred:   (B, output_steps, H, W)

Architecture:
    coarse = FNO(x)
    residual = U-Net([x, coarse])
    pred = coarse + residual_scale * residual

The FNO branch learns global/nonlocal spectral evolution. The U-Net decoder/refiner
recovers local spatial details and high-wavenumber residuals, which often helps
reduce accumulated long-rollout smoothing error.
"""

from __future__ import annotations

import inspect
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from neuralop.models.fno import FNO, TFNO
except ImportError as e:
    raise ImportError("Please install neuralop before using FNOUNetDecoder.") from e


def _filter_model_kwargs(model_cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(model_cls)
    return {k: v for k, v in kwargs.items() if v is not None and k in sig.parameters}


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm is stable for small batches; choose a divisor of num_channels."""
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_group_norm(out_channels),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(float(dropout)))
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_group_norm(out_channels),
            nn.GELU(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetRefiner(nn.Module):
    """
    A compact 2D U-Net used as the decoder/refiner after the FNO coarse branch.
    Time frames are treated as channels, consistent with the current FNO baseline.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        depth: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        depth = int(depth)
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.inc = ConvBlock(in_channels, base_channels, dropout=dropout)

        downs = []
        channels = [base_channels]
        ch = base_channels
        for _ in range(depth):
            next_ch = ch * 2
            downs.append(DownBlock(ch, next_ch, dropout=dropout))
            channels.append(next_ch)
            ch = next_ch
        self.downs = nn.ModuleList(downs)

        ups = []
        for level in range(depth - 1, -1, -1):
            skip_ch = channels[level]
            out_ch = skip_ch
            ups.append(UpBlock(ch, skip_ch, out_ch, dropout=dropout))
            ch = out_ch
        self.ups = nn.ModuleList(ups)

        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # Do not use the bottom feature as a skip for the first up block.
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)

        return self.out_conv(x)


class FNOUNetDecoder(nn.Module):
    """
    FNO coarse predictor + U-Net residual decoder.

    Parameters are intentionally close to neuralop.models.FNO so the training and
    validation scripts can switch model_arch from "fno" to "fno_unet_decoder".
    """

    def __init__(
        self,
        n_modes,
        hidden_channels: int,
        in_channels: int,
        out_channels: int,
        n_layers: int = 4,
        lifting_channels: int = 64,
        projection_channels: int = 64,
        fno_arch: str = "fno",
        unet_base_channels: int = 32,
        unet_depth: int = 3,
        unet_dropout: float = 0.0,
        use_context: bool = True,
        use_residual: bool = True,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        fno_arch = str(fno_arch).lower()
        if fno_arch == "fno":
            fno_cls = FNO
        elif fno_arch == "tfno":
            fno_cls = TFNO
        else:
            raise ValueError(f"Unsupported fno_arch: {fno_arch}. Expected 'fno' or 'tfno'.")

        fno_kwargs = dict(
            n_modes=tuple(n_modes),
            hidden_channels=int(hidden_channels),
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            n_layers=int(n_layers),
            lifting_channels=int(lifting_channels),
            projection_channels=int(projection_channels),
        )
        self.fno = fno_cls(**_filter_model_kwargs(fno_cls, fno_kwargs))

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.use_context = bool(use_context)
        self.use_residual = bool(use_residual)
        self.residual_scale = float(residual_scale)

        decoder_in_channels = self.out_channels + (self.in_channels if self.use_context else 0)
        self.decoder = UNetRefiner(
            in_channels=decoder_in_channels,
            out_channels=self.out_channels,
            base_channels=int(unet_base_channels),
            depth=int(unet_depth),
            dropout=float(unet_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coarse = self.fno(x)
        if isinstance(coarse, (tuple, list)):
            coarse = coarse[0]
        if coarse.shape[-2:] != x.shape[-2:]:
            coarse = F.interpolate(coarse, size=x.shape[-2:], mode="bilinear", align_corners=False)

        decoder_input = torch.cat([x, coarse], dim=1) if self.use_context else coarse
        residual_or_direct = self.decoder(decoder_input)

        if self.use_residual:
            return coarse + self.residual_scale * residual_or_direct
        return residual_or_direct
