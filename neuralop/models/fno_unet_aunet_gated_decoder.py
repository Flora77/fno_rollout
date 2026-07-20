"""
FNO + single U-Net / Attention U-Net residual refiner + gated residual for
phase-resolved sea-surface rollout prediction.

Recommended location:
    neuralop/models/fno_unet_aunet_gated_decoder.py

Input:
    x:    (B, input_steps, H, W)
Output:
    pred: (B, output_steps, H, W)

Architecture:
    1) coarse = FNO(x)
    2) z = concat(x, coarse) if use_context else coarse
    3) residual = SingleUNet(z) or AttentionUNet(z)
    4) pred = coarse + residual_scale * sigmoid(GateNet(z)) * residual
       if use_gated_residual=True, otherwise pred = coarse + residual_scale * residual

Compared with the frequency-band version, this module removes Fourier band
splitting and multi-branch fusion. The only difference between U-Net and AU-Net
is whether attention gates are applied to the encoder skip features in the
U-Net decoder path.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from neuralop.models.fno import FNO, TFNO
except ImportError as e:
    raise ImportError("Please install neuralop before using FNOGlobalUNetGatedDecoder.") from e


def _filter_model_kwargs(model_cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(model_cls)
    return {k: v for k, v in kwargs.items() if v is not None and k in sig.parameters}


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(int(max_groups), int(num_channels))
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class PeriodicConv2d(nn.Module):
    """Conv2d with circular/periodic padding for spatial wave fields."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        padding_mode: str = "periodic",
    ):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("PeriodicConv2d expects odd kernel_size >= 1.")
        self.pad = kernel_size // 2
        self.padding_mode = str(padding_mode).lower()
        self.conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=kernel_size,
            padding=0,
            bias=bool(bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            if self.padding_mode in {"periodic", "circular"}:
                x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="circular")
            elif self.padding_mode in {"zero", "zeros"}:
                x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="constant", value=0.0)
            elif self.padding_mode in {"reflect", "reflection"}:
                x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
            else:
                raise ValueError(f"Unsupported padding_mode: {self.padding_mode}")
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        padding_mode: str = "periodic",
    ):
        super().__init__()
        layers = [
            PeriodicConv2d(in_channels, out_channels, kernel_size=3, bias=False, padding_mode=padding_mode),
            _make_group_norm(out_channels),
            nn.GELU(),
        ]
        if float(dropout) > 0.0:
            layers.append(nn.Dropout2d(float(dropout)))
        layers += [
            PeriodicConv2d(out_channels, out_channels, kernel_size=3, bias=False, padding_mode=padding_mode),
            _make_group_norm(out_channels),
            nn.GELU(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        padding_mode: str = "periodic",
    ):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Standard U-Net decoder block."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        padding_mode: str = "periodic",
    ):
        super().__init__()
        self.reduce = PeriodicConv2d(in_channels, out_channels, kernel_size=1, bias=False, padding_mode=padding_mode)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout=dropout, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Additive attention gate for AU-Net skip connections.

    Given decoder gating feature g and encoder skip feature x, it learns
        alpha = sigmoid(psi(GELU(W_g g + W_x x)))
    and returns alpha * x.
    """

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: Optional[int] = None,
        padding_mode: str = "periodic",
    ):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(1, min(int(gate_channels), int(skip_channels)) // 2)
        inter_channels = int(inter_channels)
        self.gate_proj = PeriodicConv2d(gate_channels, inter_channels, kernel_size=1, bias=False, padding_mode=padding_mode)
        self.skip_proj = PeriodicConv2d(skip_channels, inter_channels, kernel_size=1, bias=False, padding_mode=padding_mode)
        self.psi = PeriodicConv2d(inter_channels, 1, kernel_size=1, bias=True, padding_mode=padding_mode)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        alpha = self.sigmoid(self.psi(self.act(self.gate_proj(gate) + self.skip_proj(skip))))
        return skip * alpha


class AttentionUpBlock(nn.Module):
    """AU-Net decoder block: attention-filter skip features before concatenation."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        attention_inter_channels: Optional[int] = None,
        padding_mode: str = "periodic",
    ):
        super().__init__()
        self.reduce = PeriodicConv2d(in_channels, out_channels, kernel_size=1, bias=False, padding_mode=padding_mode)
        self.att = AttentionGate(
            gate_channels=out_channels,
            skip_channels=skip_channels,
            inter_channels=attention_inter_channels,
            padding_mode=padding_mode,
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout=dropout, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        skip = self.att(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class PeriodicUNetRefiner(nn.Module):
    """Single periodic-padding U-Net residual refiner."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        depth: int = 3,
        dropout: float = 0.0,
        padding_mode: str = "periodic",
    ):
        super().__init__()
        depth = int(depth)
        if depth < 1:
            raise ValueError("depth must be >= 1")
        base_channels = int(base_channels)

        self.inc = ConvBlock(in_channels, base_channels, dropout=dropout, padding_mode=padding_mode)
        downs = []
        channels = [base_channels]
        ch = base_channels
        for _ in range(depth):
            next_ch = ch * 2
            downs.append(DownBlock(ch, next_ch, dropout=dropout, padding_mode=padding_mode))
            channels.append(next_ch)
            ch = next_ch
        self.downs = nn.ModuleList(downs)

        ups = []
        for level in range(depth - 1, -1, -1):
            skip_ch = channels[level]
            out_ch = skip_ch
            ups.append(UpBlock(ch, skip_ch, out_ch, dropout=dropout, padding_mode=padding_mode))
            ch = out_ch
        self.ups = nn.ModuleList(ups)
        self.out_conv = PeriodicConv2d(ch, out_channels, kernel_size=1, bias=True, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        return self.out_conv(x)


class PeriodicAttentionUNetRefiner(nn.Module):
    """Single periodic-padding Attention U-Net residual refiner."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        depth: int = 3,
        dropout: float = 0.0,
        attention_inter_channels: Optional[int] = None,
        padding_mode: str = "periodic",
    ):
        super().__init__()
        depth = int(depth)
        if depth < 1:
            raise ValueError("depth must be >= 1")
        base_channels = int(base_channels)

        self.inc = ConvBlock(in_channels, base_channels, dropout=dropout, padding_mode=padding_mode)
        downs = []
        channels = [base_channels]
        ch = base_channels
        for _ in range(depth):
            next_ch = ch * 2
            downs.append(DownBlock(ch, next_ch, dropout=dropout, padding_mode=padding_mode))
            channels.append(next_ch)
            ch = next_ch
        self.downs = nn.ModuleList(downs)

        ups = []
        for level in range(depth - 1, -1, -1):
            skip_ch = channels[level]
            out_ch = skip_ch
            ups.append(
                AttentionUpBlock(
                    ch,
                    skip_ch,
                    out_ch,
                    dropout=dropout,
                    attention_inter_channels=attention_inter_channels,
                    padding_mode=padding_mode,
                )
            )
            ch = out_ch
        self.ups = nn.ModuleList(ups)
        self.out_conv = PeriodicConv2d(ch, out_channels, kernel_size=1, bias=True, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        return self.out_conv(x)


class ResidualGate(nn.Module):
    """Spatial-temporal gate for residual correction strength."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 32,
        bias_init: float = 0.0,
        padding_mode: str = "periodic",
    ):
        super().__init__()
        hidden_channels = int(hidden_channels)
        self.net = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, dropout=0.0, padding_mode=padding_mode),
            PeriodicConv2d(hidden_channels, out_channels, kernel_size=1, bias=True, padding_mode=padding_mode),
        )
        last = self.net[-1]
        if isinstance(last, PeriodicConv2d):
            nn.init.constant_(last.conv.bias, float(bias_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class FNOGlobalUNetGatedDecoder(nn.Module):
    """FNO coarse predictor + single U-Net/AU-Net residual refiner + optional gated residual."""

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
        decoder_type: str = "unet",
        decoder_base_channels: int = 32,
        unet_depth: int = 3,
        unet_dropout: float = 0.0,
        use_context: bool = True,
        use_residual: bool = True,
        residual_scale: float = 1.0,
        use_gated_residual: bool = True,
        gate_hidden_channels: int = 32,
        gate_bias_init: float = 0.0,
        attention_inter_channels: Optional[int] = None,
        padding_mode: str = "periodic",
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
        self.use_gated_residual = bool(use_gated_residual)
        self.decoder_type = str(decoder_type).lower()
        self.padding_mode = str(padding_mode).lower()

        decoder_in_channels = self.out_channels + (self.in_channels if self.use_context else 0)
        if self.decoder_type in {"unet", "u-net", "single_unet"}:
            self.decoder = PeriodicUNetRefiner(
                in_channels=decoder_in_channels,
                out_channels=self.out_channels,
                base_channels=int(decoder_base_channels),
                depth=int(unet_depth),
                dropout=float(unet_dropout),
                padding_mode=self.padding_mode,
            )
        elif self.decoder_type in {"aunet", "attention_unet", "attention-u-net", "au-net"}:
            self.decoder = PeriodicAttentionUNetRefiner(
                in_channels=decoder_in_channels,
                out_channels=self.out_channels,
                base_channels=int(decoder_base_channels),
                depth=int(unet_depth),
                dropout=float(unet_dropout),
                attention_inter_channels=attention_inter_channels,
                padding_mode=self.padding_mode,
            )
        else:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}. Expected 'unet' or 'aunet'.")

        if self.use_gated_residual:
            self.residual_gate = ResidualGate(
                in_channels=decoder_in_channels,
                out_channels=self.out_channels,
                hidden_channels=int(gate_hidden_channels),
                bias_init=float(gate_bias_init),
                padding_mode=self.padding_mode,
            )
        else:
            self.residual_gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coarse = self.fno(x)
        if isinstance(coarse, (tuple, list)):
            coarse = coarse[0]
        if coarse.shape[-2:] != x.shape[-2:]:
            coarse = F.interpolate(coarse, size=x.shape[-2:], mode="bilinear", align_corners=False)

        decoder_input = torch.cat([x, coarse], dim=1) if self.use_context else coarse
        residual_or_direct = self.decoder(decoder_input)

        if not self.use_residual:
            return residual_or_direct

        if self.residual_gate is not None:
            gate = self.residual_gate(decoder_input)
            return coarse + self.residual_scale * gate * residual_or_direct
        return coarse + self.residual_scale * residual_or_direct


# Aliases for clearer imports in experiment scripts.
FNOUNetGatedDecoder = FNOGlobalUNetGatedDecoder
FNOAUNetGatedDecoder = FNOGlobalUNetGatedDecoder
