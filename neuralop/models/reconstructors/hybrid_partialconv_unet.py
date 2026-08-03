"""Periodic U-Net variants with PartialConv restricted to selected encoder levels."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neuralop.layers.partial_convolution import PeriodicPartialConv2d
from neuralop.models.reconstructors.mask_unet import (
    PeriodicConv2d,
    PeriodicConvBlock,
    _group_norm,
)


class _PartialConvInputBlock(nn.Module):
    """One mask-aware input convolution followed by an ordinary refinement."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.partial_conv = PeriodicPartialConv2d(
            in_channels, out_channels, kernel_size=3, stride=1, bias=True
        )
        self.norm = _group_norm(out_channels)
        self.refine = PeriodicConvBlock(out_channels, out_channels)

    def forward(self, value: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        value, next_mask = self.partial_conv(value, mask)
        value = F.gelu(self.norm(value)) * next_mask
        return self.refine(value), next_mask


class _PartialConvDownsample(nn.Module):
    """Mask-aware downsampling used by the all-PartialConv U-Net ablation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.partial_conv = PeriodicPartialConv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=True
        )
        self.norm = _group_norm(out_channels)
        self.refine = PeriodicConvBlock(out_channels, out_channels)

    def forward(self, value: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        value, next_mask = self.partial_conv(value, mask)
        value = F.gelu(self.norm(value)) * next_mask
        return self.refine(value), next_mask


class PeriodicHybridPartialConvUNet(nn.Module):
    """Reconstruct history with PartialConv near input and ordinary U-Net thereafter.

    Parameters
    ----------
    partialconv_levels:
        Number of encoder resolutions that use PartialConv. ``1`` is the proposed
        hybrid: only the input-nearest convolution is mask-aware. Setting it to the
        number of ``channels`` produces the all-encoder PartialConv U-Net ablation.
    use_skip_connections:
        Disabling skips isolates the value of U-Net feature transfer while retaining
        the same encoder and decoder resolutions.
    """

    def __init__(
        self,
        input_steps: int = 60,
        channels: Sequence[int] = (12, 24, 40),
        *,
        partialconv_levels: int = 1,
        use_skip_connections: bool = True,
    ) -> None:
        super().__init__()
        parsed_channels = tuple(int(value) for value in channels)
        if len(parsed_channels) < 2 or any(value <= 0 for value in parsed_channels):
            raise ValueError("Hybrid PartialConv U-Net requires >=2 positive channels")
        partialconv_levels = int(partialconv_levels)
        if not 1 <= partialconv_levels <= len(parsed_channels):
            raise ValueError(
                "partialconv_levels must be between 1 and the encoder depth"
            )
        self.input_steps = int(input_steps)
        self.channels = parsed_channels
        self.partialconv_levels = partialconv_levels
        self.use_skip_connections = bool(use_skip_connections)

        self.input_block = _PartialConvInputBlock(
            self.input_steps, parsed_channels[0]
        )
        self.downsamples = nn.ModuleList()
        self.encoder_blocks = nn.ModuleList()
        for level, (in_channels, out_channels) in enumerate(
            zip(parsed_channels[:-1], parsed_channels[1:]), start=1
        ):
            if level < partialconv_levels:
                self.downsamples.append(
                    _PartialConvDownsample(in_channels, out_channels)
                )
                self.encoder_blocks.append(nn.Identity())
            else:
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

        decoder_blocks = []
        for current_channels, skip_channels in zip(
            reversed(parsed_channels[1:]), reversed(parsed_channels[:-1])
        ):
            decoder_in = current_channels + (
                skip_channels if self.use_skip_connections else 0
            )
            decoder_blocks.append(PeriodicConvBlock(decoder_in, skip_channels))
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.output_projection = PeriodicConv2d(
            parsed_channels[0], self.input_steps, kernel_size=1
        )

    def forward(self, x_obs: Tensor, obs_mask: Tensor) -> Tensor:
        if x_obs.ndim != 4 or x_obs.shape != obs_mask.shape:
            raise ValueError(
                "Hybrid PartialConv U-Net expects matching x_obs/obs_mask (B,T,H,W)"
            )
        if x_obs.shape[1] != self.input_steps:
            raise ValueError(
                f"Expected {self.input_steps} history frames, got {x_obs.shape[1]}"
            )
        if not torch.all((obs_mask == 0) | (obs_mask == 1)):
            raise ValueError("obs_mask must be binary with 1=observed")

        mask = obs_mask.to(device=x_obs.device, dtype=x_obs.dtype)
        value, feature_mask = self.input_block(x_obs, mask)
        skips = [value]
        target_sizes = [tuple(value.shape[-2:])]
        for level, (downsample, block) in enumerate(
            zip(self.downsamples, self.encoder_blocks), start=1
        ):
            if level < self.partialconv_levels:
                value, feature_mask = downsample(value, feature_mask)
            else:
                value = block(downsample(value))
            skips.append(value)
            target_sizes.append(tuple(value.shape[-2:]))

        for index, decoder in enumerate(self.decoder_blocks):
            skip = skips[-2 - index]
            value = F.interpolate(value, size=skip.shape[-2:], mode="nearest")
            if self.use_skip_connections:
                value = torch.cat((value, skip), dim=1)
            value = decoder(value)
        reconstruction = self.output_projection(value)
        if reconstruction.shape != x_obs.shape:
            raise RuntimeError(
                f"Hybrid reconstruction {tuple(reconstruction.shape)} != "
                f"input {tuple(x_obs.shape)}"
            )
        if not torch.isfinite(reconstruction).all():
            raise FloatingPointError("Hybrid reconstruction contains NaN/Inf")
        return reconstruction


__all__ = ["PeriodicHybridPartialConvUNet"]
