"""PartialConv encoder with a lightweight asymmetric masked-autoencoder decoder."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neuralop.layers.partial_convolution import PeriodicPartialConv2d


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, int(channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _PeriodicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("Periodic decoder convolutions require odd kernel sizes")
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, value: Tensor) -> Tensor:
        if self.padding:
            value = F.pad(
                value,
                (self.padding, self.padding, self.padding, self.padding),
                mode="circular",
            )
        return self.conv(value)


class PartialConvEncoderStage(nn.Module):
    """Downsample once while propagating the PartialConv validity mask."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.partial_conv = PeriodicPartialConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            bias=True,
        )
        self.norm = _group_norm(out_channels)
        self.activation = nn.GELU()

    def forward(self, value: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        value, next_mask = self.partial_conv(value, mask)
        value = self.activation(self.norm(value)) * next_mask
        return value, next_mask


class PartialConvEncoder(nn.Module):
    """Mask-propagating encoder for a ``(B,T,H,W)`` sea-surface history."""

    def __init__(self, in_channels: int = 60, channels: Sequence[int] = (24, 48, 72)):
        super().__init__()
        if not channels:
            raise ValueError("PartialConvEncoder requires at least one stage")
        stages = []
        current_channels = int(in_channels)
        for output_channels in channels:
            stages.append(PartialConvEncoderStage(current_channels, int(output_channels)))
            current_channels = int(output_channels)
        self.in_channels = int(in_channels)
        self.out_channels = current_channels
        self.stages = nn.ModuleList(stages)

    def forward(
        self, value: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tuple[Tuple[int, int], ...]]:
        if value.ndim != 4 or value.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected value (B,{self.in_channels},H,W), got {tuple(value.shape)}"
            )
        spatial_sizes: List[Tuple[int, int]] = [tuple(value.shape[-2:])]
        for stage in self.stages:
            value, mask = stage(value, mask)
            spatial_sizes.append(tuple(value.shape[-2:]))
        return value, mask, tuple(spatial_sizes)


class LightweightAsymmetricMAEDecoder(nn.Module):
    """Small no-skip decoder; lighter than the PartialConv encoder by design."""

    def __init__(
        self,
        latent_channels: int,
        output_channels: int = 60,
        channels: Sequence[int] = (32, 24, 16),
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("MAE decoder requires at least one upsampling stage")
        self.latent_projection = _PeriodicConv2d(latent_channels, int(channels[0]), 1)
        blocks = []
        current_channels = int(channels[0])
        for output_channels_stage in channels:
            output_channels_stage = int(output_channels_stage)
            blocks.append(
                nn.Sequential(
                    _PeriodicConv2d(current_channels, output_channels_stage, 3),
                    _group_norm(output_channels_stage),
                    nn.GELU(),
                )
            )
            current_channels = output_channels_stage
        self.blocks = nn.ModuleList(blocks)
        self.output_projection = _PeriodicConv2d(
            current_channels, int(output_channels), 3
        )

    def forward(
        self, latent: Tensor, target_sizes: Sequence[Tuple[int, int]]
    ) -> Tensor:
        if len(target_sizes) != len(self.blocks):
            raise ValueError(
                f"Decoder has {len(self.blocks)} stages but received "
                f"{len(target_sizes)} target sizes"
            )
        value = self.latent_projection(latent)
        for block, target_size in zip(self.blocks, target_sizes):
            value = F.interpolate(
                value, size=target_size, mode="bilinear", align_corners=False
            )
            value = block(value)
        return self.output_projection(value)


class PartialConvMaskedAutoencoder(nn.Module):
    """Reconstruct a complete history from sparse values and an explicit mask."""

    def __init__(
        self,
        input_steps: int = 60,
        encoder_channels: Sequence[int] = (24, 48, 72),
        decoder_channels: Sequence[int] = (32, 24, 16),
    ) -> None:
        super().__init__()
        if len(encoder_channels) != len(decoder_channels):
            raise ValueError("Encoder and decoder must have the same number of scales")
        self.input_steps = int(input_steps)
        self.encoder = PartialConvEncoder(self.input_steps, encoder_channels)
        self.decoder = LightweightAsymmetricMAEDecoder(
            self.encoder.out_channels,
            output_channels=self.input_steps,
            channels=decoder_channels,
        )

    def forward(self, x_obs: Tensor, obs_mask: Tensor) -> Tensor:
        if x_obs.shape != obs_mask.shape:
            raise ValueError("x_obs and obs_mask must have identical (B,T,H,W) shapes")
        latent, _, spatial_sizes = self.encoder(x_obs, obs_mask)
        target_sizes = tuple(reversed(spatial_sizes[:-1]))
        reconstruction = self.decoder(latent, target_sizes)
        if reconstruction.shape != x_obs.shape:
            raise RuntimeError(
                f"Reconstruction shape {tuple(reconstruction.shape)} differs from "
                f"input shape {tuple(x_obs.shape)}"
            )
        return reconstruction


__all__ = [
    "LightweightAsymmetricMAEDecoder",
    "PartialConvEncoder",
    "PartialConvEncoderStage",
    "PartialConvMaskedAutoencoder",
]
