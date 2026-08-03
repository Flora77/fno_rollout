"""Coordinate-aware periodic GNO and GINO-style sea-surface reconstructors.

The ``scalar_legacy`` kernel mode preserves the first B5/B6 checkpoint contract.
New experiments should use ``channelwise_nonlinear``: it applies signed,
channel-specific periodic kernels to nonlinear sensor features instead of reducing
the reconstructor to one sample-independent affine interpolation stencil.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from neuralop.models.fno import FNO
from neuralop.models.reconstructors.mask_unet import PeriodicConv2d


def _grid_coordinates(
    height: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    yy = torch.arange(height, device=device, dtype=dtype) / float(height)
    xx = torch.arange(width, device=device, dtype=dtype) / float(width)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack((grid_y, grid_x), dim=-1).reshape(-1, 2)


def _fixed_sensor_values(
    x_obs: Tensor, obs_mask: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    if x_obs.ndim != 4 or x_obs.shape != obs_mask.shape:
        raise ValueError("GNO/GINO expects matching x_obs/obs_mask (B,T,H,W)")
    if not torch.all((obs_mask == 0) | (obs_mask == 1)):
        raise ValueError("obs_mask must be binary with 1=observed")
    binary = obs_mask.bool()
    spatial = binary[:, 0]
    if not torch.equal(binary, spatial[:, None].expand_as(binary)):
        raise ValueError(
            "Coordinate reconstructors currently require a time-invariant sensor mask"
        )
    if not torch.equal(spatial, spatial[:1].expand_as(spatial)):
        raise ValueError(
            "Coordinate reconstructors require one fixed sensor layout per batch"
        )
    sensor_selector = spatial[0].reshape(-1)
    if not bool(sensor_selector.any()):
        raise ValueError("GNO/GINO requires at least one observed sensor")
    batch, steps, height, width = x_obs.shape
    coordinates = _grid_coordinates(
        height, width, device=x_obs.device, dtype=x_obs.dtype
    )
    sensor_coordinates = coordinates[sensor_selector]
    safe_values = x_obs * obs_mask.to(dtype=x_obs.dtype)
    values = safe_values.permute(0, 2, 3, 1).reshape(batch, -1, steps)
    return sensor_coordinates, values[:, sensor_selector], coordinates


class PeriodicKernelIntegral(nn.Module):
    """Vectorized periodic graph-kernel integral without optional scatter packages.

    The learned scalar kernel acts on the shortest periodic relative displacement.
    Query chunks bound memory for the 64x64 output grid.  A nearest-point fallback
    ensures finite output when a radius contains no sensor.
    """

    def __init__(
        self,
        channels: int,
        radius: float,
        mlp_channels: Sequence[int] = (64, 64),
        query_chunk_size: int = 512,
        kernel_mode: str = "channelwise_nonlinear",
        kernel_rank: int = 16,
        normalization: str = "l1",
        relative_fourier_frequencies: Sequence[int] = (),
        cache_geometry: bool = False,
    ) -> None:
        super().__init__()
        if radius <= 0.0 or radius > 0.75:
            raise ValueError("Periodic GNO radius must be in (0, 0.75]")
        kernel_mode = str(kernel_mode).lower()
        if kernel_mode not in {"scalar_legacy", "channelwise_nonlinear"}:
            raise ValueError(
                "kernel_mode must be 'scalar_legacy' or 'channelwise_nonlinear'"
            )
        self.channels = int(channels)
        self.kernel_mode = kernel_mode
        self.normalization = str(normalization).lower()
        if self.normalization not in {"l1", "neighbor_mean"}:
            raise ValueError("GNO normalization must be 'l1' or 'neighbor_mean'")
        self.kernel_rank = (
            1
            if self.kernel_mode == "scalar_legacy"
            else min(int(kernel_rank), self.channels)
        )
        if self.kernel_rank <= 0:
            raise ValueError("kernel_rank must be positive")
        self.relative_fourier_frequencies = tuple(
            int(value) for value in relative_fourier_frequencies
        )
        if any(value <= 0 for value in self.relative_fourier_frequencies):
            raise ValueError("Relative Fourier frequencies must be positive")
        self.cache_geometry = bool(cache_geometry)
        self._geometry_cache: Dict[tuple, Tuple[Tensor, Tensor]] = {}
        self._geometry_source_signature = None
        self.geometry_cache_hits = 0
        self.geometry_cache_misses = 0
        layers = []
        in_channels = 2 + 4 * len(self.relative_fourier_frequencies)
        for out_channels in mlp_channels:
            layers.extend((nn.Linear(in_channels, int(out_channels)), nn.GELU()))
            in_channels = int(out_channels)
        kernel_out_channels = self.kernel_rank
        layers.append(nn.Linear(in_channels, kernel_out_channels))
        self.kernel = nn.Sequential(*layers)
        if self.kernel_mode == "scalar_legacy":
            # Keep the original module names and parameter shapes so locked B5/B6
            # checkpoints remain loadable when the resolved config omits kernel_mode.
            self.feature_mix = nn.Linear(self.channels, self.channels)
            self.source_encoder = None
            self.source_gate = None
            self.source_to_kernel = None
            self.kernel_to_feature = None
        else:
            self.source_encoder = nn.Sequential(
                nn.Linear(self.channels, self.channels),
                nn.GELU(),
                nn.Linear(self.channels, self.channels),
            )
            self.source_gate = nn.Linear(self.channels, self.channels)
            self.source_to_kernel = nn.Linear(self.channels, self.kernel_rank)
            self.kernel_to_feature = nn.Linear(self.kernel_rank, self.channels)
            self.feature_mix = nn.Sequential(
                nn.Linear(self.channels, self.channels),
                nn.GELU(),
                nn.Linear(self.channels, self.channels),
            )
        self.radius = float(radius)
        self.query_chunk_size = int(query_chunk_size)

    def _kernel_input(self, delta: Tensor) -> Tensor:
        if not self.relative_fourier_frequencies:
            return delta
        features = [delta]
        for frequency in self.relative_fourier_frequencies:
            phase = 2.0 * torch.pi * float(frequency) * delta
            features.extend((torch.sin(phase), torch.cos(phase)))
        return torch.cat(features, dim=-1)

    def clear_geometry_cache(self) -> None:
        self._geometry_cache.clear()
        self._geometry_source_signature = None

    def geometry_cache_report(self) -> Dict[str, int]:
        return {
            "entries": len(self._geometry_cache),
            "hits": int(self.geometry_cache_hits),
            "misses": int(self.geometry_cache_misses),
        }

    def _relative_geometry(
        self, source_coords: Tensor, query: Tensor
    ) -> Tuple[Tensor, Tensor]:
        source_signature = (
            source_coords.data_ptr(),
            tuple(source_coords.shape),
            source_coords.device,
            source_coords.dtype,
        )
        if self.cache_geometry:
            if source_signature != self._geometry_source_signature:
                self._geometry_cache.clear()
                self._geometry_source_signature = source_signature
            query_signature = (
                query.data_ptr(),
                tuple(query.shape),
                tuple(query.stride()),
            )
            cached = self._geometry_cache.get(query_signature)
            if cached is not None:
                self.geometry_cache_hits += 1
                return cached
        delta = query[:, None, :] - source_coords[None, :, :]
        delta = delta - torch.round(delta)
        distance = torch.linalg.vector_norm(delta, dim=-1)
        support = distance < self.radius
        has_support = support.any(dim=1, keepdim=True)
        nearest = F.one_hot(
            distance.argmin(dim=1), num_classes=source_coords.shape[0]
        ).bool()
        effective_support = torch.where(has_support, support, nearest)
        normalized_distance = (distance / self.radius).clamp(0.0, 1.0)
        cutoff = 0.5 * (1.0 + torch.cos(torch.pi * normalized_distance))
        cutoff = torch.where(effective_support, cutoff, torch.zeros_like(cutoff))
        cutoff = torch.where(
            (~has_support) & nearest, torch.ones_like(cutoff), cutoff
        )
        if self.cache_geometry:
            self.geometry_cache_misses += 1
            self._geometry_cache[query_signature] = (delta.detach(), cutoff.detach())
        return delta, cutoff

    def _nonlinear_chunk(
        self,
        source_coords: Tensor,
        query: Tensor,
        kernel_features: Tensor,
    ) -> Tensor:
        if self.kernel_to_feature is None:
            raise RuntimeError("Nonlinear GNO kernel projection is unavailable")
        delta, cutoff = self._relative_geometry(source_coords, query)
        logits = self.kernel(self._kernel_input(delta)).clamp(-10.0, 10.0)
        # Signed low-rank kernels represent oscillatory, channel-dependent wave
        # structure without materializing a Q x N x hidden_channels tensor.
        channel_kernel = torch.tanh(logits) * cutoff.unsqueeze(-1)
        if self.normalization == "l1":
            normalizer = channel_kernel.abs().sum(
                dim=1, keepdim=True
            ).clamp_min(1.0e-6)
        else:
            # Monte-Carlo/Riemann-style local averaging: the learned signed
            # kernel retains its amplitude instead of being forced to unit L1
            # mass independently in every channel.
            normalizer = cutoff.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
            normalizer = normalizer.unsqueeze(-1)
        channel_kernel = channel_kernel / normalizer
        aggregated = torch.einsum(
            "qnr,bnr->bqr", channel_kernel, kernel_features
        )
        projected = self.kernel_to_feature(aggregated)
        return projected + self.feature_mix(projected)

    def forward(
        self, source_coords: Tensor, query_coords: Tensor, source_features: Tensor
    ) -> Tensor:
        if source_coords.ndim != 2 or query_coords.ndim != 2:
            raise ValueError("GNO coordinates must be (N,2) and (Q,2)")
        if source_features.ndim != 3:
            raise ValueError("GNO source_features must be (B,N,C)")
        if source_features.shape[1] != source_coords.shape[0]:
            raise ValueError("GNO coordinate and feature point counts differ")
        if source_features.shape[-1] != self.channels:
            raise ValueError(
                f"GNO expected {self.channels} feature channels, "
                f"got {source_features.shape[-1]}"
            )
        if self.kernel_mode == "channelwise_nonlinear":
            if (
                self.source_encoder is None
                or self.source_gate is None
                or self.source_to_kernel is None
                or self.kernel_to_feature is None
            ):
                raise RuntimeError("Nonlinear GNO source modules were not initialized")
            encoded_features = self.source_encoder(source_features)
            # A feature-dependent gate makes the operator adapt its spatial response
            # to each observed wave history while keeping aggregation memory bounded.
            source_gate = 2.0 * torch.sigmoid(self.source_gate(source_features))
            encoded_features = encoded_features * source_gate
            kernel_features = self.source_to_kernel(encoded_features)
        else:
            encoded_features = source_features
            kernel_features = None
        outputs = []
        for query in query_coords.split(self.query_chunk_size, dim=0):
            if self.kernel_mode == "channelwise_nonlinear":
                if kernel_features is None:
                    raise RuntimeError("Nonlinear GNO kernel features are unavailable")
                if torch.is_grad_enabled():
                    outputs.append(
                        checkpoint(
                            self._nonlinear_chunk,
                            source_coords,
                            query,
                            kernel_features,
                            use_reentrant=False,
                        )
                    )
                else:
                    outputs.append(
                        self._nonlinear_chunk(
                            source_coords, query, kernel_features
                        )
                    )
                continue

            delta, cutoff = self._relative_geometry(source_coords, query)
            logits = self.kernel(self._kernel_input(delta)).clamp(-10.0, 10.0)
            weights = torch.exp(logits.squeeze(-1)) * cutoff
            weights = weights / weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0e-12)
            aggregated = torch.einsum(
                "qn,bnc->bqc", weights, encoded_features
            )
            outputs.append(self.feature_mix(aggregated))
        return torch.cat(outputs, dim=1)


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, int(channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualTemporalConvBlock(nn.Module):
    """Non-periodic residual temporal encoding for one sensor history."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        dilation = int(dilation)
        if dilation <= 0:
            raise ValueError("Temporal dilation must be positive")
        self.layers = nn.Sequential(
            _group_norm(channels),
            nn.GELU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                padding_mode="replicate",
            ),
            _group_norm(channels),
            nn.GELU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                padding_mode="replicate",
            ),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.layers(value)


class SensorHistoryEncoder(nn.Module):
    """Encode all 60 ordered frames without collapsing them in one shallow lift."""

    def __init__(
        self,
        input_steps: int,
        hidden_channels: int,
        temporal_channels: int,
        temporal_dilations: Sequence[int],
    ) -> None:
        super().__init__()
        self.input_steps = int(input_steps)
        self.hidden_channels = int(hidden_channels)
        self.temporal_channels = int(temporal_channels)
        if min(self.input_steps, self.hidden_channels, self.temporal_channels) <= 0:
            raise ValueError("History encoder dimensions must be positive")
        dilations = tuple(int(value) for value in temporal_dilations)
        if not dilations:
            raise ValueError("At least one temporal residual block is required")

        # The explicit time coordinate distinguishes the beginning and end of
        # the non-periodic observation window.
        self.input_projection = nn.Conv1d(
            2,
            self.temporal_channels,
            kernel_size=5,
            padding=2,
            padding_mode="replicate",
        )
        self.blocks = nn.ModuleList(
            ResidualTemporalConvBlock(self.temporal_channels, dilation)
            for dilation in dilations
        )
        self.sequence_projection = nn.Linear(
            self.temporal_channels * self.input_steps,
            self.hidden_channels,
        )
        self.direct_projection = nn.Linear(self.input_steps, self.hidden_channels)
        self.output_norm = nn.LayerNorm(self.hidden_channels)

    def forward(self, sensor_values: Tensor) -> Tensor:
        if sensor_values.ndim != 3:
            raise ValueError("Sensor histories must be (B,N,T)")
        if sensor_values.shape[-1] != self.input_steps:
            raise ValueError(
                f"Expected {self.input_steps} sensor history frames, "
                f"got {sensor_values.shape[-1]}"
            )
        batch, sensors, _ = sensor_values.shape
        flattened = sensor_values.reshape(batch * sensors, 1, self.input_steps)
        time = torch.linspace(
            -1.0,
            1.0,
            self.input_steps,
            device=sensor_values.device,
            dtype=sensor_values.dtype,
        ).view(1, 1, self.input_steps)
        time = time.expand(batch * sensors, -1, -1)
        sequence = self.input_projection(torch.cat((flattened, time), dim=1))
        for block in self.blocks:
            sequence = block(sequence)
        encoded = self.sequence_projection(sequence.flatten(start_dim=1))
        encoded = encoded + self.direct_projection(flattened[:, 0])
        encoded = F.gelu(self.output_norm(encoded))
        return encoded.view(batch, sensors, self.hidden_channels)


class TemporalTokenMixingBlock(nn.Module):
    """Mix a short ordered token sequence while preserving every token."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(channels)
        self.temporal = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels),
        )

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim < 3:
            raise ValueError("Temporal tokens must end in (tokens,channels)")
        token_count, channels = value.shape[-2:]
        flattened = value.reshape(-1, token_count, channels)
        normalized = self.input_norm(flattened)
        mixed = self.temporal(normalized.transpose(1, 2)).transpose(1, 2)
        flattened = flattened + mixed
        flattened = flattened + self.update(flattened)
        return flattened.view_as(value)


class SensorHistoryTokenEncoder(nn.Module):
    """Encode history as several ordered tokens instead of one early summary."""

    def __init__(
        self,
        input_steps: int,
        token_count: int,
        token_channels: int,
        temporal_channels: int,
        temporal_dilations: Sequence[int],
    ) -> None:
        super().__init__()
        self.input_steps = int(input_steps)
        self.token_count = int(token_count)
        self.token_channels = int(token_channels)
        self.temporal_channels = int(temporal_channels)
        if min(
            self.input_steps,
            self.token_count,
            self.token_channels,
            self.temporal_channels,
        ) <= 0:
            raise ValueError("Temporal-token encoder dimensions must be positive")
        if self.input_steps % self.token_count != 0:
            raise ValueError("input_steps must be divisible by temporal token count")
        dilations = tuple(int(value) for value in temporal_dilations)
        if not dilations:
            raise ValueError("At least one temporal residual block is required")
        self.frames_per_token = self.input_steps // self.token_count
        self.input_projection = nn.Conv1d(
            2,
            self.temporal_channels,
            kernel_size=5,
            padding=2,
            padding_mode="replicate",
        )
        self.blocks = nn.ModuleList(
            ResidualTemporalConvBlock(self.temporal_channels, dilation)
            for dilation in dilations
        )
        self.token_projection = nn.Linear(
            self.temporal_channels, self.token_channels
        )
        self.direct_projection = nn.Linear(
            self.frames_per_token, self.token_channels
        )
        self.token_embedding = nn.Parameter(
            torch.zeros(1, self.token_count, self.token_channels)
        )
        nn.init.normal_(self.token_embedding, std=0.02)
        self.token_mixer = TemporalTokenMixingBlock(self.token_channels)
        self.output_norm = nn.LayerNorm(self.token_channels)

    def forward(self, sensor_values: Tensor) -> Tensor:
        if sensor_values.ndim != 3:
            raise ValueError("Sensor histories must be (B,N,T)")
        if sensor_values.shape[-1] != self.input_steps:
            raise ValueError(
                f"Expected {self.input_steps} sensor history frames, "
                f"got {sensor_values.shape[-1]}"
            )
        batch, sensors, _ = sensor_values.shape
        flattened = sensor_values.reshape(batch * sensors, 1, self.input_steps)
        time = torch.linspace(
            -1.0,
            1.0,
            self.input_steps,
            device=sensor_values.device,
            dtype=sensor_values.dtype,
        ).view(1, 1, self.input_steps)
        time = time.expand(batch * sensors, -1, -1)
        sequence = self.input_projection(torch.cat((flattened, time), dim=1))
        for block in self.blocks:
            sequence = block(sequence)
        pooled = F.adaptive_avg_pool1d(sequence, self.token_count)
        tokens = self.token_projection(pooled.transpose(1, 2))
        direct = flattened[:, 0].view(
            batch * sensors, self.token_count, self.frames_per_token
        )
        tokens = tokens + self.direct_projection(direct) + self.token_embedding
        tokens = self.token_mixer(tokens)
        tokens = F.gelu(self.output_norm(tokens))
        return tokens.view(
            batch,
            sensors,
            self.token_count,
            self.token_channels,
        )


class PeriodicGraphOperatorBlock(nn.Module):
    """Residual graph-kernel layer matching the iterative GNO formulation."""

    def __init__(
        self,
        channels: int,
        radius: float,
        mlp_channels: Sequence[int],
        *,
        kernel_rank: int,
        normalization: str,
        query_chunk_size: int,
        relative_fourier_frequencies: Sequence[int] = (),
        cache_geometry: bool = False,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(channels)
        self.integral = PeriodicKernelIntegral(
            channels,
            radius,
            mlp_channels,
            query_chunk_size=query_chunk_size,
            kernel_mode="channelwise_nonlinear",
            kernel_rank=kernel_rank,
            normalization=normalization,
            relative_fourier_frequencies=relative_fourier_frequencies,
            cache_geometry=cache_geometry,
        )
        # W v(x) is the pointwise term in a neural-operator layer.
        self.local = nn.Linear(channels, channels)
        self.update = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels),
        )

    def forward(self, coordinates: Tensor, value: Tensor) -> Tensor:
        normalized = self.input_norm(value)
        integral = self.integral(coordinates, coordinates, normalized)
        value = value + F.gelu(self.local(normalized) + integral)
        return value + self.update(value)


class PeriodicGridResidualBlock(nn.Module):
    """Full-resolution periodic refinement without a spatial bottleneck."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        bottleneck = max(8, channels // 2)
        self.layers = nn.Sequential(
            PeriodicConv2d(channels, bottleneck, kernel_size=1),
            _group_norm(bottleneck),
            nn.GELU(),
            PeriodicConv2d(bottleneck, bottleneck, kernel_size=3),
            _group_norm(bottleneck),
            nn.GELU(),
            PeriodicConv2d(bottleneck, channels, kernel_size=3),
            _group_norm(channels),
        )

    def forward(self, value: Tensor) -> Tensor:
        return F.gelu(value + self.layers(value))


def _enforce_observation_consistency(
    reconstruction: Tensor,
    x_obs: Tensor,
    obs_mask: Tensor,
    *,
    enabled: bool,
) -> Tensor:
    if not enabled:
        return reconstruction
    return torch.where(obs_mask.bool(), x_obs, reconstruction)


class PeriodicGNOReconstructor(nn.Module):
    """Periodic sparse-history GNO with a checkpoint-compatible legacy path.

    ``single_integral_v2`` preserves the previous B5 module names and tensor
    shapes. ``residual_graph_v3`` uses a residual temporal encoder, iterative
    sensor-graph neural-operator layers, a full-rank sensor-to-grid integral,
    and full-resolution periodic grid refinement. ``residual_graph_v4`` keeps
    several ordered temporal tokens through the graph, augments relative
    geometry with periodic Fourier features, and caches fixed-layout geometry.
    """

    def __init__(
        self,
        input_steps: int = 60,
        hidden_channels: int = 32,
        radius: float = 0.20,
        mlp_channels: Sequence[int] = (64, 64),
        kernel_mode: str = "channelwise_nonlinear",
        kernel_rank: int = 16,
        hard_observation_consistency: bool = False,
        architecture: str = "single_integral_v2",
        temporal_channels: int = 32,
        temporal_dilations: Sequence[int] = (1, 2, 4),
        graph_layers: int = 3,
        grid_refinement_layers: int = 3,
        kernel_normalization: str = "l1",
        query_chunk_size: int = 512,
        raw_observation_supervision: bool = False,
        temporal_tokens: int = 4,
        relative_fourier_frequencies: Sequence[int] = (),
        cache_geometry: bool = False,
    ) -> None:
        super().__init__()
        self.input_steps = int(input_steps)
        self.hidden_channels = int(hidden_channels)
        self.radius = float(radius)
        self.kernel_mode = str(kernel_mode).lower()
        self.hard_observation_consistency = bool(hard_observation_consistency)
        self.architecture = str(architecture).lower()
        self.raw_observation_supervision = bool(raw_observation_supervision)
        self.kernel_rank = int(kernel_rank)
        self.kernel_normalization = str(kernel_normalization).lower()
        self.query_chunk_size = int(query_chunk_size)
        self.temporal_tokens = int(temporal_tokens)
        self.relative_fourier_frequencies = tuple(
            int(value) for value in relative_fourier_frequencies
        )
        self.cache_geometry = bool(cache_geometry)
        self._cached_sensor_layout = None
        self._cached_sensor_coordinates = None
        self._cached_output_coordinates = None
        if self.architecture not in {
            "single_integral_v2",
            "residual_graph_v3",
            "residual_graph_v4",
        }:
            raise ValueError(
                "GNO architecture must be 'single_integral_v2' or "
                "'residual_graph_v3' or 'residual_graph_v4'"
            )

        if self.architecture == "single_integral_v2":
            # Preserve the exact module names used by legacy B5 checkpoints.
            self.value_lift = nn.Linear(self.input_steps, self.hidden_channels)
            self.gno = PeriodicKernelIntegral(
                self.hidden_channels,
                self.radius,
                mlp_channels,
                kernel_mode=self.kernel_mode,
                kernel_rank=self.kernel_rank,
            )
            self.value_activation = (
                nn.Identity()
                if self.kernel_mode == "scalar_legacy"
                else nn.GELU()
            )
            self.output_refinement = (
                None
                if self.kernel_mode == "scalar_legacy"
                else nn.Sequential(
                    nn.Linear(self.hidden_channels, self.hidden_channels),
                    nn.GELU(),
                )
            )
            self.output_projection = nn.Linear(
                self.hidden_channels, self.input_steps
            )
            return

        if self.kernel_mode != "channelwise_nonlinear":
            raise ValueError(
                "Residual graph GNO requires "
                "kernel_mode='channelwise_nonlinear'"
            )
        graph_layers = int(graph_layers)
        grid_refinement_layers = int(grid_refinement_layers)
        if graph_layers < 1:
            raise ValueError("residual_graph_v3 requires at least one graph layer")
        if grid_refinement_layers < 0:
            raise ValueError(
                "residual_graph_v3 grid_refinement_layers must be non-negative"
            )
        if self.architecture == "residual_graph_v3" and (
            self.kernel_rank < self.hidden_channels
        ):
            raise ValueError(
                "residual_graph_v3 kernel_rank must be at least hidden_channels "
                "to avoid a low-rank aggregation bottleneck"
            )
        if self.architecture == "residual_graph_v4":
            if self.temporal_tokens < 2:
                raise ValueError("residual_graph_v4 requires at least two time tokens")
            if self.hidden_channels % self.temporal_tokens != 0:
                raise ValueError(
                    "hidden_channels must be divisible by temporal token count"
                )
            self.temporal_token_channels = (
                self.hidden_channels // self.temporal_tokens
            )
            if self.kernel_rank < self.temporal_token_channels:
                raise ValueError(
                    "residual_graph_v4 kernel_rank must be at least token channels"
                )
            self.history_token_encoder = SensorHistoryTokenEncoder(
                self.input_steps,
                self.temporal_tokens,
                self.temporal_token_channels,
                int(temporal_channels),
                temporal_dilations,
            )
            graph_channels = self.temporal_token_channels
            graph_fourier_frequencies = self.relative_fourier_frequencies
            graph_cache_geometry = self.cache_geometry
        else:
            self.history_encoder = SensorHistoryEncoder(
                self.input_steps,
                self.hidden_channels,
                int(temporal_channels),
                temporal_dilations,
            )
            graph_channels = self.hidden_channels
            graph_fourier_frequencies = ()
            graph_cache_geometry = False
        self.sensor_operator_blocks = nn.ModuleList(
            PeriodicGraphOperatorBlock(
                graph_channels,
                self.radius,
                mlp_channels,
                kernel_rank=self.kernel_rank,
                normalization=self.kernel_normalization,
                query_chunk_size=self.query_chunk_size,
                relative_fourier_frequencies=graph_fourier_frequencies,
                cache_geometry=graph_cache_geometry,
            )
            for _ in range(graph_layers)
        )
        if self.architecture == "residual_graph_v4":
            self.sensor_token_mixers = nn.ModuleList(
                TemporalTokenMixingBlock(self.temporal_token_channels)
                for _ in range(graph_layers)
            )
        self.sensor_to_grid = PeriodicKernelIntegral(
            graph_channels,
            self.radius,
            mlp_channels,
            query_chunk_size=self.query_chunk_size,
            kernel_mode="channelwise_nonlinear",
            kernel_rank=self.kernel_rank,
            normalization=self.kernel_normalization,
            relative_fourier_frequencies=graph_fourier_frequencies,
            cache_geometry=graph_cache_geometry,
        )
        if self.architecture == "residual_graph_v4":
            self.token_grid_fusion = nn.Sequential(
                PeriodicConv2d(
                    self.hidden_channels,
                    self.hidden_channels,
                    kernel_size=1,
                ),
                _group_norm(self.hidden_channels),
                nn.GELU(),
            )
        self.grid_refinement = nn.Sequential(
            *(
                PeriodicGridResidualBlock(self.hidden_channels)
                for _ in range(grid_refinement_layers)
            )
        )
        self.grid_output_projection = PeriodicConv2d(
            self.hidden_channels,
            self.input_steps,
            kernel_size=1,
        )

    def _sensor_values_with_cached_coordinates(
        self, x_obs: Tensor, obs_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        sensor_coords, sensor_values, output_coords = _fixed_sensor_values(
            x_obs, obs_mask
        )
        if not self.cache_geometry:
            return sensor_coords, sensor_values, output_coords
        layout = obs_mask[0, 0].bool().detach()
        cache_valid = (
            self._cached_sensor_layout is not None
            and self._cached_sensor_layout.device == layout.device
            and self._cached_sensor_coordinates is not None
            and self._cached_sensor_coordinates.dtype == sensor_coords.dtype
            and torch.equal(self._cached_sensor_layout, layout)
        )
        if not cache_valid:
            self._cached_sensor_layout = layout.clone()
            self._cached_sensor_coordinates = sensor_coords.detach()
            self._cached_output_coordinates = output_coords.detach()
        return (
            self._cached_sensor_coordinates,
            sensor_values,
            self._cached_output_coordinates,
        )

    def _forward_raw(self, x_obs: Tensor, obs_mask: Tensor) -> Tensor:
        sensor_coords, sensor_values, output_coords = (
            self._sensor_values_with_cached_coordinates(x_obs, obs_mask)
        )
        if x_obs.shape[1] != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} history frames")
        if self.architecture == "single_integral_v2":
            sensor_features = self.value_activation(self.value_lift(sensor_values))
            grid_features = self.gno(sensor_coords, output_coords, sensor_features)
            if self.output_refinement is not None:
                grid_features = grid_features + self.output_refinement(grid_features)
            reconstruction = self.output_projection(grid_features)
            batch, _, height, width = x_obs.shape
            reconstruction = reconstruction.view(
                batch, height, width, self.input_steps
            )
            return reconstruction.permute(0, 3, 1, 2).contiguous()

        batch, _, height, width = x_obs.shape
        if self.architecture == "residual_graph_v4":
            sensor_tokens = self.history_token_encoder(sensor_values)
            sensor_count = sensor_tokens.shape[1]
            for block, token_mixer in zip(
                self.sensor_operator_blocks, self.sensor_token_mixers
            ):
                token_features = sensor_tokens.permute(0, 2, 1, 3).reshape(
                    batch * self.temporal_tokens,
                    sensor_count,
                    self.temporal_token_channels,
                )
                token_features = block(sensor_coords, token_features)
                sensor_tokens = token_features.view(
                    batch,
                    self.temporal_tokens,
                    sensor_count,
                    self.temporal_token_channels,
                ).permute(0, 2, 1, 3).contiguous()
                sensor_tokens = token_mixer(sensor_tokens)
            token_features = sensor_tokens.permute(0, 2, 1, 3).reshape(
                batch * self.temporal_tokens,
                sensor_count,
                self.temporal_token_channels,
            )
            grid_tokens = self.sensor_to_grid(
                sensor_coords, output_coords, token_features
            )
            grid_features = grid_tokens.view(
                batch,
                self.temporal_tokens,
                height,
                width,
                self.temporal_token_channels,
            ).permute(0, 1, 4, 2, 3).contiguous()
            grid_features = grid_features.view(
                batch, self.hidden_channels, height, width
            )
            grid_features = self.token_grid_fusion(grid_features)
            grid_features = self.grid_refinement(grid_features)
            return self.grid_output_projection(grid_features)

        sensor_features = self.history_encoder(sensor_values)
        for block in self.sensor_operator_blocks:
            sensor_features = block(sensor_coords, sensor_features)
        grid_features = self.sensor_to_grid(
            sensor_coords, output_coords, sensor_features
        )
        grid_features = grid_features.view(
            batch, height, width, self.hidden_channels
        )
        grid_features = grid_features.permute(0, 3, 1, 2).contiguous()
        grid_features = self.grid_refinement(grid_features)
        return self.grid_output_projection(grid_features)

    def forward_with_aux(
        self, x_obs: Tensor, obs_mask: Tensor
    ) -> Dict[str, Tensor]:
        raw_reconstruction = self._forward_raw(x_obs, obs_mask)
        reconstruction = _enforce_observation_consistency(
            raw_reconstruction,
            x_obs,
            obs_mask,
            enabled=self.hard_observation_consistency,
        )
        if not torch.isfinite(raw_reconstruction).all() or not torch.isfinite(
            reconstruction
        ).all():
            raise FloatingPointError("GNO reconstruction contains NaN/Inf")
        loss_reconstruction = (
            raw_reconstruction
            if self.raw_observation_supervision
            else reconstruction
        )
        return {
            "raw_reconstruction": raw_reconstruction,
            "reconstruction": reconstruction,
            "loss_reconstruction": loss_reconstruction,
        }

    def forward(self, x_obs: Tensor, obs_mask: Tensor) -> Tensor:
        reconstruction = self.forward_with_aux(x_obs, obs_mask)["reconstruction"]
        return reconstruction


class PeriodicGINOReconstructor(nn.Module):
    """GNO lift -> latent-grid FNO propagation -> GNO projection to full grid."""

    def __init__(
        self,
        input_steps: int = 60,
        hidden_channels: int = 32,
        latent_shape: Sequence[int] = (16, 16),
        radius: float = 0.20,
        n_modes: Sequence[int] = (8, 8),
        fno_layers: int = 3,
        mlp_channels: Sequence[int] = (64, 64),
        kernel_mode: str = "channelwise_nonlinear",
        kernel_rank: int = 16,
        hard_observation_consistency: bool = False,
        latent_positional_embedding: str = "grid",
    ) -> None:
        super().__init__()
        if len(latent_shape) != 2 or any(int(value) <= 1 for value in latent_shape):
            raise ValueError("latent_shape must contain two integers greater than one")
        if len(n_modes) != 2:
            raise ValueError("GINO latent FNO requires two spatial mode counts")
        self.input_steps = int(input_steps)
        self.hidden_channels = int(hidden_channels)
        self.latent_shape = tuple(int(value) for value in latent_shape)
        self.radius = float(radius)
        self.kernel_mode = str(kernel_mode).lower()
        self.hard_observation_consistency = bool(hard_observation_consistency)
        self.latent_positional_embedding = str(latent_positional_embedding).lower()
        if self.latent_positional_embedding not in {"grid", "none"}:
            raise ValueError(
                "latent_positional_embedding must be 'grid' or 'none'"
            )
        self.value_lift = nn.Linear(self.input_steps, self.hidden_channels)
        self.value_activation = (
            nn.Identity()
            if self.kernel_mode == "scalar_legacy"
            else nn.GELU()
        )
        self.input_gno = PeriodicKernelIntegral(
            self.hidden_channels,
            self.radius,
            mlp_channels,
            kernel_mode=self.kernel_mode,
            kernel_rank=kernel_rank,
        )
        self.latent_operator = FNO(
            n_modes=tuple(int(value) for value in n_modes),
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            hidden_channels=self.hidden_channels,
            n_layers=int(fno_layers),
            positional_embedding=(
                "grid" if self.latent_positional_embedding == "grid" else None
            ),
            domain_padding=None,
        )
        self.output_gno = PeriodicKernelIntegral(
            self.hidden_channels,
            self.radius,
            mlp_channels,
            kernel_mode=self.kernel_mode,
            kernel_rank=kernel_rank,
        )
        self.output_projection = nn.Linear(self.hidden_channels, self.input_steps)

    def forward(self, x_obs: Tensor, obs_mask: Tensor) -> Tensor:
        sensor_coords, sensor_values, output_coords = _fixed_sensor_values(
            x_obs, obs_mask
        )
        if x_obs.shape[1] != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} history frames")
        sensor_features = self.value_activation(self.value_lift(sensor_values))
        latent_coords = _grid_coordinates(
            *self.latent_shape, device=x_obs.device, dtype=x_obs.dtype
        )
        latent = self.input_gno(sensor_coords, latent_coords, sensor_features)
        batch = x_obs.shape[0]
        latent = latent.view(batch, *self.latent_shape, self.hidden_channels)
        latent = latent.permute(0, 3, 1, 2).contiguous()
        latent = self.latent_operator(latent)
        latent_values = latent.permute(0, 2, 3, 1).reshape(
            batch, -1, self.hidden_channels
        )
        output_features = self.output_gno(
            latent_coords, output_coords, latent_values
        )
        reconstruction = self.output_projection(output_features)
        _, _, height, width = x_obs.shape
        reconstruction = reconstruction.view(batch, height, width, self.input_steps)
        reconstruction = reconstruction.permute(0, 3, 1, 2).contiguous()
        reconstruction = _enforce_observation_consistency(
            reconstruction,
            x_obs,
            obs_mask,
            enabled=self.hard_observation_consistency,
        )
        if not torch.isfinite(reconstruction).all():
            raise FloatingPointError("GINO reconstruction contains NaN/Inf")
        return reconstruction


__all__ = [
    "PeriodicGINOReconstructor",
    "PeriodicGNOReconstructor",
    "PeriodicKernelIntegral",
]
