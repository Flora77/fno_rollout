"""Periodic ordinary Mask U-Net for sparse sea-surface reconstruction."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, int(channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class PeriodicConv2d(nn.Module):
    """Conv2d with explicit circular spatial padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("PeriodicConv2d requires a positive odd kernel size")
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size,
            stride=int(stride),
            padding=0,
            bias=bias,
        )

    def forward(self, value: Tensor) -> Tensor:
        if self.padding:
            value = F.pad(
                value,
                (self.padding, self.padding, self.padding, self.padding),
                mode="circular",
            )
        return self.conv(value)


class PeriodicConvBlock(nn.Module):
    """Two ordinary periodic convolutions; no PartialConv normalization."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            PeriodicConv2d(in_channels, out_channels, 3),
            _group_norm(out_channels),
            nn.GELU(),
            PeriodicConv2d(out_channels, out_channels, 3),
            _group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


class PeriodicMaskUNet(nn.Module):
    """Reconstruct ``(B,T,H,W)`` history from masked values and a binary mask.

    The 60 history frames are treated as channels.  The explicit observation mask is
    concatenated with values after missing entries have been removed, so arbitrary
    storage fill values at ``mask=0`` cannot influence the output.
    """

    def __init__(
        self,
        input_steps: int = 60,
        channels: Sequence[int] = (12, 24, 40),
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("PeriodicMaskUNet requires at least two resolution levels")
        parsed_channels = tuple(int(value) for value in channels)
        if any(value <= 0 for value in parsed_channels):
            raise ValueError("PeriodicMaskUNet channels must be positive")
        self.input_steps = int(input_steps)
        self.channels = parsed_channels

        self.stem = PeriodicConvBlock(2 * self.input_steps, parsed_channels[0])
        self.downsamples = nn.ModuleList()
        self.encoder_blocks = nn.ModuleList()
        for in_channels, out_channels in zip(
            parsed_channels[:-1], parsed_channels[1:]
        ):
            self.downsamples.append(
                nn.Sequential(
                    PeriodicConv2d(in_channels, out_channels, 3, stride=2),
                    _group_norm(out_channels),
                    nn.GELU(),
                )
            )
            self.encoder_blocks.append(
                PeriodicConvBlock(out_channels, out_channels)
            )

        self.decoder_blocks = nn.ModuleList(
            PeriodicConvBlock(current_channels + skip_channels, skip_channels)
            for current_channels, skip_channels in zip(
                reversed(parsed_channels[1:]), reversed(parsed_channels[:-1])
            )
        )
        self.output_projection = PeriodicConv2d(
            parsed_channels[0], self.input_steps, kernel_size=1
        )

    def forward(self, x_obs: Tensor, obs_mask: Tensor) -> Tensor:
        if x_obs.ndim != 4 or obs_mask.ndim != 4:
            raise ValueError("PeriodicMaskUNet expects x_obs and obs_mask as (B,T,H,W)")
        if x_obs.shape != obs_mask.shape:
            raise ValueError("x_obs and obs_mask must have identical shapes")
        if x_obs.shape[1] != self.input_steps:
            raise ValueError(
                f"Expected {self.input_steps} history frames, got {x_obs.shape[1]}"
            )
        if not torch.all((obs_mask == 0) | (obs_mask == 1)):
            raise ValueError("obs_mask must be binary with 1=observed")

        mask = obs_mask.to(device=x_obs.device, dtype=x_obs.dtype)
        value = torch.cat((x_obs * mask, mask), dim=1)
        skips = []
        value = self.stem(value)
        skips.append(value)
        for downsample, block in zip(self.downsamples, self.encoder_blocks):
            value = block(downsample(value))
            skips.append(value)

        for block, skip in zip(self.decoder_blocks, reversed(skips[:-1])):
            # Nearest upsampling commutes with periodic shifts; the following
            # circular convolutions perform the learned spatial refinement.
            value = F.interpolate(value, size=skip.shape[-2:], mode="nearest")
            value = block(torch.cat((value, skip), dim=1))
        reconstruction = self.output_projection(value)
        if reconstruction.shape != x_obs.shape:
            raise RuntimeError(
                f"Mask U-Net output {tuple(reconstruction.shape)} differs from "
                f"input {tuple(x_obs.shape)}"
            )
        if not torch.isfinite(reconstruction).all():
            raise FloatingPointError("Mask U-Net reconstruction contains NaN/Inf")
        return reconstruction


__all__ = ["PeriodicConv2d", "PeriodicConvBlock", "PeriodicMaskUNet"]
