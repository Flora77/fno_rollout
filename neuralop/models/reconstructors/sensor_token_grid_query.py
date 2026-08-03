"""Periodic sensor-token encoder with full-grid cross-attention queries."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn


def _periodic_grid_coordinates(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    yy = torch.arange(height, device=device, dtype=dtype) / float(height)
    xx = torch.arange(width, device=device, dtype=dtype) / float(width)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack((grid_y, grid_x), dim=-1).reshape(-1, 2)


def periodic_fourier_coordinates(coordinates: Tensor, bands: int) -> Tensor:
    """Embed normalized ``(y,x)`` coordinates with exact unit-period wrapping."""

    if coordinates.shape[-1] != 2:
        raise ValueError("coordinates must end in normalized (y,x)")
    if int(bands) <= 0:
        raise ValueError("bands must be positive")
    wrapped = torch.remainder(coordinates, 1.0)
    frequencies = torch.arange(
        1,
        int(bands) + 1,
        device=coordinates.device,
        dtype=coordinates.dtype,
    )
    angles = 2.0 * math.pi * wrapped.unsqueeze(-1) * frequencies
    features = torch.stack(
        (
            torch.sin(angles[..., 0, :]),
            torch.cos(angles[..., 0, :]),
            torch.sin(angles[..., 1, :]),
            torch.cos(angles[..., 1, :]),
        ),
        dim=-2,
    )
    return features.flatten(start_dim=-2)


class CrossAttentionQueryBlock(nn.Module):
    """Cross-attention plus pointwise feed-forward refinement, without query self-attention."""

    def __init__(
        self,
        *,
        dimension: int,
        heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        if dimension <= 0 or heads <= 0 or dimension % heads:
            raise ValueError("Cross-attention dimension must be divisible by heads")
        hidden = int(round(dimension * float(mlp_ratio)))
        if hidden <= 0:
            raise ValueError("Cross-attention MLP width must be positive")
        self.query_norm = nn.LayerNorm(dimension)
        self.memory_norm = nn.LayerNorm(dimension)
        self.cross_attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Linear(hidden, dimension),
        )

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        sensor_mask: Tensor,
    ) -> Tensor:
        update, _ = self.cross_attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=~sensor_mask,
            need_weights=False,
        )
        queries = queries + update
        return queries + self.feed_forward(self.feed_forward_norm(queries))


class PeriodicSensorTokenGridQueryReconstructor(nn.Module):
    """Encode one token per fixed sensor and cross-attend periodic grid queries."""

    def __init__(
        self,
        *,
        input_steps: int = 60,
        dimension: int = 128,
        encoder_depth: int = 4,
        encoder_heads: int = 4,
        encoder_mlp_ratio: float = 4.0,
        decoder_depth: int = 3,
        decoder_heads: int = 4,
        decoder_mlp_ratio: float = 2.0,
        coordinate_bands: int = 8,
        query_chunk_size: int = 512,
        hard_observation_consistency: bool = False,
    ) -> None:
        super().__init__()
        self.input_steps = int(input_steps)
        self.dimension = int(dimension)
        self.coordinate_bands = int(coordinate_bands)
        self.query_chunk_size = int(query_chunk_size)
        self.hard_observation_consistency = bool(hard_observation_consistency)
        if self.input_steps <= 0 or self.dimension <= 0:
            raise ValueError("input_steps and dimension must be positive")
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        if self.hard_observation_consistency:
            raise ValueError("ST-D0 main path disables hard observation consistency")

        coordinate_features = 4 * self.coordinate_bands
        self.history_embedding = nn.Linear(self.input_steps, self.dimension)
        self.sensor_coordinate_embedding = nn.Linear(
            coordinate_features, self.dimension
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dimension,
            nhead=int(encoder_heads),
            dim_feedforward=int(round(self.dimension * float(encoder_mlp_ratio))),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sensor_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(encoder_depth),
            norm=nn.LayerNorm(self.dimension),
        )
        self.grid_query_embedding = nn.Linear(
            coordinate_features, self.dimension
        )
        self.query_decoder = nn.ModuleList(
            CrossAttentionQueryBlock(
                dimension=self.dimension,
                heads=int(decoder_heads),
                mlp_ratio=float(decoder_mlp_ratio),
            )
            for _ in range(int(decoder_depth))
        )
        self.output_projection = nn.Linear(self.dimension, self.input_steps)
        self.empty_sensor_token = nn.Parameter(
            torch.empty(1, 1, self.dimension)
        )
        nn.init.trunc_normal_(self.empty_sensor_token, std=0.02)

    def _fixed_sensor_tokens(
        self,
        x_obs: Tensor,
        obs_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if x_obs.ndim != 4 or x_obs.shape != obs_mask.shape:
            raise ValueError("x_obs and obs_mask must have identical (B,T,H,W) shapes")
        if x_obs.shape[1] != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} history frames")
        if not torch.all((obs_mask == 0) | (obs_mask == 1)):
            raise ValueError("obs_mask must be binary with 1=observed")
        binary = obs_mask.bool()
        spatial = binary[:, 0]
        if not torch.equal(binary, spatial[:, None].expand_as(binary)):
            raise ValueError("ST-D0 requires a time-invariant fixed-sensor mask")

        batch, _, height, width = x_obs.shape
        grid = _periodic_grid_coordinates(
            height,
            width,
            device=x_obs.device,
            dtype=x_obs.dtype,
        )
        selectors = [sample.reshape(-1) for sample in spatial]
        maximum = max(1, max(int(selector.sum()) for selector in selectors))
        sensor_values = x_obs.new_zeros((batch, maximum, self.input_steps))
        sensor_coordinates = x_obs.new_zeros((batch, maximum, 2))
        sensor_mask = torch.zeros(
            (batch, maximum),
            device=x_obs.device,
            dtype=torch.bool,
        )
        safe_values = (x_obs * obs_mask.to(dtype=x_obs.dtype)).permute(
            0, 2, 3, 1
        ).reshape(batch, -1, self.input_steps)
        for sample, selector in enumerate(selectors):
            count = int(selector.sum())
            if count:
                sensor_values[sample, :count] = safe_values[sample, selector]
                sensor_coordinates[sample, :count] = grid[selector]
                sensor_mask[sample, :count] = True
        return sensor_values, sensor_coordinates, sensor_mask

    def _encode_sensor_tokens(
        self,
        sensor_values: Tensor,
        sensor_coordinates: Tensor,
        sensor_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if sensor_values.ndim != 3 or sensor_values.shape[-1] != self.input_steps:
            raise ValueError("sensor_values must be (B,N,input_steps)")
        if sensor_coordinates.shape != (*sensor_values.shape[:2], 2):
            raise ValueError("sensor_coordinates must be (B,N,2)")
        if sensor_mask.shape != sensor_values.shape[:2]:
            raise ValueError("sensor_mask must be (B,N)")
        sensor_mask = sensor_mask.bool()
        active_columns = sensor_mask.any(dim=0)
        if bool(active_columns.any()):
            sensor_values = sensor_values[:, active_columns]
            sensor_coordinates = sensor_coordinates[:, active_columns]
            sensor_mask = sensor_mask[:, active_columns]
        else:
            sensor_values = sensor_values[:, :1]
            sensor_coordinates = sensor_coordinates[:, :1]
            sensor_mask = sensor_mask[:, :1]
        safe_values = torch.where(
            sensor_mask.unsqueeze(-1),
            sensor_values,
            torch.zeros_like(sensor_values),
        )
        safe_coordinates = torch.where(
            sensor_mask.unsqueeze(-1),
            sensor_coordinates,
            torch.zeros_like(sensor_coordinates),
        )
        coordinate_features = periodic_fourier_coordinates(
            safe_coordinates, self.coordinate_bands
        )
        tokens = self.history_embedding(safe_values)
        tokens = tokens + self.sensor_coordinate_embedding(coordinate_features)
        tokens = torch.where(sensor_mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))

        effective_mask = sensor_mask.clone()
        empty_samples = ~effective_mask.any(dim=1)
        if bool(empty_samples.any()):
            tokens[empty_samples, 0] = self.empty_sensor_token[0, 0]
            effective_mask[empty_samples, 0] = True
        encoded = self.sensor_encoder(
            tokens,
            src_key_padding_mask=~effective_mask,
        )
        return encoded, effective_mask, empty_samples

    def reconstruct_from_sensor_tokens(
        self,
        sensor_values: Tensor,
        sensor_coordinates: Tensor,
        sensor_mask: Tensor,
        *,
        height: int,
        width: int,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        encoded, effective_mask, empty_samples = self._encode_sensor_tokens(
            sensor_values,
            sensor_coordinates,
            sensor_mask,
        )
        output_coordinates = _periodic_grid_coordinates(
            int(height),
            int(width),
            device=sensor_values.device,
            dtype=sensor_values.dtype,
        )
        query_features = periodic_fourier_coordinates(
            output_coordinates, self.coordinate_bands
        )
        base_queries = self.grid_query_embedding(query_features)
        decoded_chunks = []
        for query_chunk in base_queries.split(self.query_chunk_size, dim=0):
            queries = query_chunk.unsqueeze(0).expand(
                sensor_values.shape[0], -1, -1
            )
            for block in self.query_decoder:
                queries = block(queries, encoded, effective_mask)
            decoded_chunks.append(self.output_projection(queries))
        reconstruction = torch.cat(decoded_chunks, dim=1)
        reconstruction = reconstruction.view(
            sensor_values.shape[0], int(height), int(width), self.input_steps
        )
        reconstruction = reconstruction.permute(0, 3, 1, 2).contiguous()
        if not torch.isfinite(reconstruction).all():
            raise FloatingPointError(
                "Sensor-token reconstruction contains NaN/Inf"
            )
        return reconstruction, {
            "sensor_mask": sensor_mask.bool(),
            "effective_sensor_mask": effective_mask,
            "sensor_count": sensor_mask.bool().sum(dim=1),
            "empty_context": empty_samples,
        }

    def forward_with_aux(
        self,
        x_obs: Tensor,
        obs_mask: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        sensor_values, sensor_coordinates, sensor_mask = self._fixed_sensor_tokens(
            x_obs,
            obs_mask,
        )
        reconstruction, aux = self.reconstruct_from_sensor_tokens(
            sensor_values,
            sensor_coordinates,
            sensor_mask,
            height=x_obs.shape[-2],
            width=x_obs.shape[-1],
        )
        if reconstruction.shape != x_obs.shape:
            raise RuntimeError("ST-D0 reconstruction shape differs from input")
        return reconstruction, {
            **aux,
            "sensor_coordinates": sensor_coordinates,
        }

    def forward(self, x_obs: Tensor, obs_mask: Tensor) -> Tensor:
        reconstruction, _ = self.forward_with_aux(x_obs, obs_mask)
        return reconstruction


__all__ = [
    "CrossAttentionQueryBlock",
    "PeriodicSensorTokenGridQueryReconstructor",
    "periodic_fourier_coordinates",
]
