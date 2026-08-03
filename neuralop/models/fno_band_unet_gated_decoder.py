"""
FNO + frequency-band U-Net decoder + gated residual for phase-resolved
sea-surface rollout prediction.

Recommended location:
    neuralop/models/fno_band_unet_gated_decoder.py

Input:
    x:    (B, input_steps, H, W)
Output:
    pred: (B, output_steps, H, W)

Architecture:
    1) coarse = FNO(x)
    2) z = concat(x, coarse) if use_context else coarse
    3) Split z into low/mid/high spatial-frequency bands with Fourier masks.
    4) Each band is refined by a periodic-padding U-Net branch.
    5) Band residuals are fused by either sum, learned scalar gates, or learned
       spatial-temporal gates.
    6) Final output uses gated residual correction:
           pred = coarse + residual_scale * sigmoid(gate(z)) * residual

Why these changes matter for wave fields:
    - Periodic padding matches periodic numerical wave tanks / HOS-like domains.
    - Band branches let the model separate long-wave phase/bulk structure from
      short-wave/high-wavenumber details.
    - Gated residual prevents the residual decoder from over-correcting regions
      where the FNO coarse solution is already reliable.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from neuralop.models.fno import FNO, TFNO
except ImportError as e:
    raise ImportError("Please install neuralop before using FNOBandUNetGatedDecoder.") from e


def _filter_model_kwargs(model_cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(model_cls)
    return {k: v for k, v in kwargs.items() if v is not None and k in sig.parameters}


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(int(max_groups), int(num_channels))
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


def _as_tuple_float(vals: Sequence[float] | str | None) -> Tuple[float, ...]:
    if vals is None:
        return tuple()
    if isinstance(vals, str):
        vals = vals.strip()
        if not vals:
            return tuple()
        vals = vals.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        return tuple(float(v.strip()) for v in vals.split(",") if v.strip())
    return tuple(float(v) for v in vals)


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
            raise ValueError("PeriodicConv2d currently expects odd kernel_size >= 1.")
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


class PeriodicUNetBranch(nn.Module):
    """Compact periodic-padding U-Net branch for one frequency band."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 16,
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


class FourierBandSplitter(nn.Module):
    """
    Fixed Fourier band splitter.

    It builds smooth radial masks on the fly for the current H, W, dtype and device.
    Bands are defined on normalized radial wavenumber rho in [0, 1]. For cutoffs
    (0.33, 0.67), the three bands are low / mid / high.
    """

    def __init__(
        self,
        cutoffs: Sequence[float] = (0.33, 0.67),
        transition_width: float = 0.04,
    ):
        super().__init__()
        cutoffs = tuple(sorted(float(c) for c in cutoffs if 0.0 < float(c) < 1.0))
        self.cutoffs = cutoffs
        self.transition_width = max(float(transition_width), 0.0)

    @property
    def n_bands(self) -> int:
        return len(self.cutoffs) + 1

    def _soft_lowpass(self, rho: torch.Tensor, cutoff: float) -> torch.Tensor:
        tw = self.transition_width
        if tw <= 0.0:
            return (rho <= float(cutoff)).to(rho.dtype)
        # Smooth step from 1 to 0 across [cutoff - tw, cutoff + tw].
        x = (rho - (float(cutoff) - tw)) / max(2.0 * tw, 1e-12)
        x = torch.clamp(x, 0.0, 1.0)
        smooth = x * x * (3.0 - 2.0 * x)
        return 1.0 - smooth

    def _build_masks(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ky = torch.fft.fftfreq(h, d=1.0, device=device).to(dtype=dtype)
        kx = torch.fft.rfftfreq(w, d=1.0, device=device).to(dtype=dtype)
        gy, gx = torch.meshgrid(ky, kx, indexing="ij")
        rho = torch.sqrt(gx * gx + gy * gy)
        rho = rho / torch.clamp(rho.max(), min=torch.tensor(1e-12, device=device, dtype=dtype))

        lowpasses = [self._soft_lowpass(rho, c) for c in self.cutoffs]
        masks = []
        if len(lowpasses) == 0:
            masks.append(torch.ones_like(rho))
        else:
            masks.append(lowpasses[0])
            for i in range(1, len(lowpasses)):
                masks.append(torch.clamp(lowpasses[i] - lowpasses[i - 1], min=0.0, max=1.0))
            masks.append(torch.clamp(1.0 - lowpasses[-1], min=0.0, max=1.0))

        mask = torch.stack(masks, dim=0)  # (bands, H, W_rfft)
        # Normalize masks so bands approximately partition unity even with smooth overlaps.
        mask = mask / torch.clamp(mask.sum(dim=0, keepdim=True), min=1e-8)
        return mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        bsz, channels, h, w = x.shape
        x_fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        masks = self._build_masks(h, w, x.device, x.real.dtype)
        out = []
        for i in range(masks.shape[0]):
            band_fft = x_fft * masks[i].view(1, 1, h, -1)
            band = torch.fft.irfft2(band_fft, s=(h, w), dim=(-2, -1), norm="ortho")
            out.append(band)
        return tuple(out)


class BandFusion(nn.Module):
    """Fuse band-wise residual predictions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_bands: int,
        mode: str = "learnable_gate",
        gate_hidden_channels: int = 32,
        padding_mode: str = "periodic",
    ):
        super().__init__()
        self.mode = str(mode).lower()
        self.n_bands = int(n_bands)
        self.out_channels = int(out_channels)

        if self.mode in {"sum", "fixed_sum", "mean"}:
            self.gate_net = None
            self.scalar_logits = None
        elif self.mode in {"learnable_scalar", "scalar", "softmax_scalar"}:
            self.scalar_logits = nn.Parameter(torch.zeros(self.n_bands))
            self.gate_net = None
        elif self.mode in {"learnable_gate", "spatial_gate", "softmax_spatial"}:
            hidden = int(gate_hidden_channels)
            self.gate_net = nn.Sequential(
                ConvBlock(in_channels, hidden, dropout=0.0, padding_mode=padding_mode),
                PeriodicConv2d(hidden, self.n_bands * self.out_channels, kernel_size=1, bias=True, padding_mode=padding_mode),
            )
            self.scalar_logits = None
        else:
            raise ValueError(
                "Unsupported band_fusion mode. Expected sum, learnable_scalar, or learnable_gate; "
                f"got {mode}."
            )

    def forward(self, residuals: Sequence[torch.Tensor], context: torch.Tensor) -> torch.Tensor:
        if len(residuals) != self.n_bands:
            raise RuntimeError(f"Expected {self.n_bands} residuals, got {len(residuals)}")
        stack = torch.stack(list(residuals), dim=1)  # (B, bands, Cout, H, W)

        if self.mode in {"sum", "fixed_sum"}:
            return stack.sum(dim=1)
        if self.mode == "mean":
            return stack.mean(dim=1)
        if self.scalar_logits is not None:
            weights = torch.softmax(self.scalar_logits, dim=0).view(1, self.n_bands, 1, 1, 1)
            return (stack * weights).sum(dim=1)

        logits = self.gate_net(context)
        bsz, _, h, w = logits.shape
        logits = logits.view(bsz, self.n_bands, self.out_channels, h, w)
        weights = torch.softmax(logits, dim=1)
        return (stack * weights).sum(dim=1)


class FrequencyBandUNetRefiner(nn.Module):
    """
    Frequency-band U-Net refiner.

    The input is split into Fourier bands; each band has its own periodic U-Net
    branch; branch residuals are fused by fixed or learnable gates.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        branch_base_channels: int = 16,
        depth: int = 3,
        dropout: float = 0.0,
        band_cutoffs: Sequence[float] = (0.33, 0.67),
        band_transition_width: float = 0.04,
        band_fusion: str = "learnable_gate",
        band_gate_hidden_channels: int = 32,
        padding_mode: str = "periodic",
    ):
        super().__init__()
        self.splitter = FourierBandSplitter(cutoffs=band_cutoffs, transition_width=band_transition_width)
        self.n_bands = self.splitter.n_bands
        self.branches = nn.ModuleList(
            [
                PeriodicUNetBranch(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    base_channels=int(branch_base_channels),
                    depth=int(depth),
                    dropout=float(dropout),
                    padding_mode=padding_mode,
                )
                for _ in range(self.n_bands)
            ]
        )
        self.fusion = BandFusion(
            in_channels=in_channels,
            out_channels=out_channels,
            n_bands=self.n_bands,
            mode=band_fusion,
            gate_hidden_channels=int(band_gate_hidden_channels),
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        band_inputs = self.splitter(x)
        band_residuals = [branch(x_band) for branch, x_band in zip(self.branches, band_inputs)]
        return self.fusion(band_residuals, x)


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


class FNOBandUNetGatedDecoder(nn.Module):
    """
    FNO coarse predictor + frequency-band U-Net residual refiner + gated residual.
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
        unet_depth: int = 3,
        unet_dropout: float = 0.0,
        use_context: bool = True,
        use_residual: bool = True,
        residual_scale: float = 1.0,
        use_gated_residual: bool = True,
        gate_hidden_channels: int = 32,
        gate_bias_init: float = 0.0,
        band_branch_base_channels: int = 16,
        band_cutoffs: Sequence[float] = (0.33, 0.67),
        band_transition_width: float = 0.04,
        band_fusion: str = "learnable_gate",
        band_gate_hidden_channels: int = 32,
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
        self.padding_mode = str(padding_mode).lower()

        decoder_in_channels = self.out_channels + (self.in_channels if self.use_context else 0)
        cutoffs = _as_tuple_float(band_cutoffs)
        self.decoder = FrequencyBandUNetRefiner(
            in_channels=decoder_in_channels,
            out_channels=self.out_channels,
            branch_base_channels=int(band_branch_base_channels),
            depth=int(unet_depth),
            dropout=float(unet_dropout),
            band_cutoffs=cutoffs,
            band_transition_width=float(band_transition_width),
            band_fusion=str(band_fusion),
            band_gate_hidden_channels=int(band_gate_hidden_channels),
            padding_mode=self.padding_mode,
        )

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
