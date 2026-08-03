"""Partial convolution with periodic spatial boundaries."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.utils import _pair


class PeriodicPartialConv2d(nn.Module):
    """Apply a 2D convolution using valid values only.

    ``mask`` must contain binary values with ``1=valid`` and ``0=missing``.
    It may have one channel shared by all value channels, or one channel per
    input value channel. Both value and mask use circular spatial padding.

    The valid weighted sum is rescaled by ``full_support / valid_support``.
    The bias, when present, is added after that rescaling. The returned mask has
    shape ``(B,1,Hout,Wout)`` and is valid wherever local support is non-empty.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        *,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        if min(self.in_channels, self.out_channels, *self.kernel_size) <= 0:
            raise ValueError("channels and kernel dimensions must be positive")
        if min(*self.stride, *self.dilation) <= 0:
            raise ValueError("stride and dilation must be positive")

        if padding is None:
            effective_kernel = tuple(
                dilation_value * (kernel_value - 1) + 1
                for kernel_value, dilation_value in zip(
                    self.kernel_size, self.dilation
                )
            )
            if any(size % 2 == 0 for size in effective_kernel):
                raise ValueError(
                    "Automatic same padding requires odd effective kernel sizes"
                )
            self.padding = tuple(size // 2 for size in effective_kernel)
        else:
            self.padding = _pair(padding)
        if min(*self.padding) < 0:
            raise ValueError("padding must be non-negative")

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            bias=bias,
        )

    @property
    def weight(self) -> nn.Parameter:
        return self.conv.weight

    @property
    def bias(self) -> Optional[nn.Parameter]:
        return self.conv.bias

    def forward(self, value: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        mask = self._validate_inputs(value, mask).to(
            device=value.device, dtype=value.dtype
        )
        if mask.shape[1] == 1:
            value_mask = mask.expand(-1, self.in_channels, -1, -1)
            support_mask = mask
            support_channels = 1
        else:
            value_mask = mask
            support_mask = mask
            support_channels = self.in_channels

        padding = (
            self.padding[1],
            self.padding[1],
            self.padding[0],
            self.padding[0],
        )
        padded_value = F.pad(value, padding, mode="circular")
        padded_value_mask = F.pad(value_mask, padding, mode="circular")
        padded_support_mask = F.pad(support_mask, padding, mode="circular")

        masked_output = self.conv(padded_value * padded_value_mask)
        mask_kernel = torch.ones(
            1,
            support_channels,
            self.kernel_size[0],
            self.kernel_size[1],
            device=value.device,
            dtype=value.dtype,
        )
        valid_count = F.conv2d(
            padded_support_mask,
            mask_kernel,
            stride=self.stride,
            dilation=self.dilation,
        )
        next_mask = (valid_count > 0).to(dtype=value.dtype)
        full_count = float(
            support_channels * self.kernel_size[0] * self.kernel_size[1]
        )
        normalization = torch.where(
            valid_count > 0,
            full_count / valid_count.clamp_min(1.0),
            torch.zeros_like(valid_count),
        )

        if self.bias is None:
            output = masked_output * normalization
        else:
            bias = self.bias.view(1, -1, 1, 1)
            output = (masked_output - bias) * normalization + bias
        output = output * next_mask
        return output, next_mask

    def _validate_inputs(self, value: Tensor, mask: Tensor) -> Tensor:
        if not isinstance(value, Tensor) or not isinstance(mask, Tensor):
            raise TypeError("value and mask must be torch.Tensor values")
        if value.ndim != 4 or mask.ndim != 4:
            raise ValueError("value and mask must have shape (B,C,H,W)")
        if value.shape[0] != mask.shape[0] or value.shape[-2:] != mask.shape[-2:]:
            raise ValueError(
                f"value and mask batch/spatial shapes differ: {value.shape} vs {mask.shape}"
            )
        if value.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} value channels, got {value.shape[1]}"
            )
        if mask.shape[1] not in (1, self.in_channels):
            raise ValueError(
                "mask must have one shared channel or match value input channels; "
                f"got {mask.shape[1]}"
            )
        if value.device != mask.device:
            raise ValueError("value and mask must be on the same device")
        if not torch.is_floating_point(value):
            raise TypeError("value must use a floating-point dtype")
        if not torch.isfinite(value).all():
            raise ValueError("value contains NaN or Inf")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("mask must be binary with 1=valid and 0=missing")
        return mask


__all__ = ["PeriodicPartialConv2d"]
