"""Coordinate-aware temporal FNO-DeepONet models for sea-surface experiments.

The implementation follows the paper's central factorization:

    sparse history -> temporal FNO branch code
    (query time, query position) -> Fourier-feature trunk code
    prediction = branch/trunk inner product

Public sea-surface tensors remain ``(B,T,H,W)``. Missing entries are always
excluded with an explicit ``1=observed`` mask; zero is never used to infer
validity.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def _periodic_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return flattened periodic ``(y,x)`` coordinates in ``[0,1)``."""

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    yy = torch.arange(height, device=device, dtype=dtype) / float(height)
    xx = torch.arange(width, device=device, dtype=dtype) / float(width)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack((grid_y, grid_x), dim=-1).reshape(-1, 2)


class TemporalSpectralConv1d(nn.Module):
    """Learned low-mode convolution along the history-time dimension."""

    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.modes = int(modes)
        if self.channels <= 0 or self.modes <= 0:
            raise ValueError("channels and modes must be positive")
        scale = 1.0 / max(1, self.channels)
        self.weight = nn.Parameter(
            scale
            * torch.randn(
                self.channels,
                self.channels,
                self.modes,
                2,
            )
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3 or values.shape[1] != self.channels:
            raise ValueError("TemporalSpectralConv1d expects (B,C,T)")
        spectrum = torch.fft.rfft(values.float(), dim=-1, norm="ortho")
        retained = min(self.modes, spectrum.shape[-1])
        output_spectrum = torch.zeros(
            values.shape[0],
            self.channels,
            spectrum.shape[-1],
            device=values.device,
            dtype=spectrum.dtype,
        )
        weights = torch.view_as_complex(self.weight.contiguous()).to(spectrum.dtype)
        output_spectrum[..., :retained] = torch.einsum(
            "bim,iom->bom",
            spectrum[..., :retained],
            weights[..., :retained],
        )
        output = torch.fft.irfft(
            output_spectrum,
            n=values.shape[-1],
            dim=-1,
            norm="ortho",
        )
        return output.to(dtype=values.dtype)


class TemporalFNOLayer(nn.Module):
    """Paper-style temporal spectral path plus pointwise residual transform."""

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = TemporalSpectralConv1d(width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, values: Tensor) -> Tensor:
        return self.activation(self.spectral(values) + self.pointwise(values))


class FourierQueryEmbedding(nn.Module):
    """Fourier features for normalized query time and periodic grid position."""

    def __init__(
        self,
        *,
        time_bands: int,
        spatial_bands: int,
        time_max_frequency: float,
    ) -> None:
        super().__init__()
        self.time_bands = int(time_bands)
        self.spatial_bands = int(spatial_bands)
        self.time_max_frequency = float(time_max_frequency)
        if self.time_bands <= 0 or self.spatial_bands <= 0:
            raise ValueError("Fourier band counts must be positive")
        if self.time_max_frequency < 1.0:
            raise ValueError("time_max_frequency must be at least 1")
        time_frequencies = math.pi * torch.logspace(
            0.0,
            math.log10(self.time_max_frequency),
            self.time_bands,
        )
        spatial_frequencies = 2.0 * math.pi * torch.arange(
            1,
            self.spatial_bands + 1,
            dtype=torch.float32,
        )
        self.register_buffer("time_frequencies", time_frequencies, persistent=True)
        self.register_buffer(
            "spatial_frequencies", spatial_frequencies, persistent=True
        )

    @property
    def output_features(self) -> int:
        return 2 * self.time_bands + 4 * self.spatial_bands

    def forward(self, times: Tensor, coordinates: Tensor) -> Tensor:
        if times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        if coordinates.ndim != 2 or coordinates.shape[-1] != 2:
            raise ValueError("coordinates must be (Q,2)")
        if times.shape[0] != coordinates.shape[0]:
            raise ValueError("times and coordinates must contain the same queries")
        time_angles = times[:, None] * self.time_frequencies[None].to(
            device=times.device,
            dtype=times.dtype,
        )
        spatial_angles = (
            torch.remainder(coordinates, 1.0)[..., None]
            * self.spatial_frequencies[None, None].to(
                device=coordinates.device,
                dtype=coordinates.dtype,
            )
        )
        return torch.cat(
            (
                torch.cos(time_angles),
                torch.sin(time_angles),
                torch.cos(spatial_angles[:, 0]),
                torch.sin(spatial_angles[:, 0]),
                torch.cos(spatial_angles[:, 1]),
                torch.sin(spatial_angles[:, 1]),
            ),
            dim=-1,
        )


class FNODeepONet(nn.Module):
    """Temporal-FNO branch and coordinate-query trunk on a periodic grid."""

    def __init__(
        self,
        *,
        input_steps: int = 60,
        height: int = 64,
        width: int = 64,
        branch_width: int = 64,
        branch_modes: int = 24,
        branch_layers: int = 4,
        branch_hidden: int = 512,
        latent_dim: int = 256,
        trunk_hidden: int = 384,
        trunk_layers: int = 4,
        time_fourier_bands: int = 24,
        spatial_fourier_bands: int = 16,
        time_max_frequency: float = 48.0,
        query_chunk_size: int = 8192,
        checkpoint_trunk: bool = True,
    ) -> None:
        super().__init__()
        self.input_steps = int(input_steps)
        self.height = int(height)
        self.width = int(width)
        self.branch_width = int(branch_width)
        self.branch_modes = int(branch_modes)
        self.latent_dim = int(latent_dim)
        self.query_chunk_size = int(query_chunk_size)
        self.checkpoint_trunk = bool(checkpoint_trunk)
        if self.input_steps <= 0 or min(self.height, self.width) <= 0:
            raise ValueError("input_steps and spatial dimensions must be positive")
        if min(self.branch_width, self.latent_dim, self.query_chunk_size) <= 0:
            raise ValueError("branch width, latent dim, and query chunk must be positive")
        if branch_layers <= 0 or trunk_layers < 2:
            raise ValueError("branch_layers must be positive and trunk_layers >= 2")

        # Per grid slot: elevation, relative y, relative x, and explicit validity.
        branch_input_features = 4 * self.height * self.width
        self.branch_lifting = nn.Linear(branch_input_features, self.branch_width)
        self.branch_layers = nn.ModuleList(
            TemporalFNOLayer(self.branch_width, self.branch_modes)
            for _ in range(int(branch_layers))
        )
        self.branch_projection = nn.Sequential(
            nn.Linear(self.branch_width * self.input_steps, int(branch_hidden)),
            nn.GELU(),
            nn.Linear(int(branch_hidden), int(branch_hidden)),
            nn.GELU(),
            nn.Linear(int(branch_hidden), self.latent_dim),
        )

        self.query_embedding = FourierQueryEmbedding(
            time_bands=int(time_fourier_bands),
            spatial_bands=int(spatial_fourier_bands),
            time_max_frequency=float(time_max_frequency),
        )
        trunk_modules = [
            nn.Linear(self.query_embedding.output_features, int(trunk_hidden)),
            nn.GELU(),
        ]
        for _ in range(int(trunk_layers) - 2):
            trunk_modules.extend(
                (nn.Linear(int(trunk_hidden), int(trunk_hidden)), nn.GELU())
            )
        trunk_modules.append(nn.Linear(int(trunk_hidden), self.latent_dim))
        self.trunk = nn.Sequential(*trunk_modules)
        self.output_bias = nn.Parameter(torch.zeros(()))

    def _validate_sparse_history(self, x_obs: Tensor, obs_mask: Tensor) -> None:
        expected = (self.input_steps, self.height, self.width)
        if x_obs.ndim != 4 or x_obs.shape != obs_mask.shape:
            raise ValueError("x_obs and obs_mask must have identical (B,T,H,W)")
        if tuple(x_obs.shape[1:]) != expected:
            raise ValueError(
                f"Expected history tail {expected}, got {tuple(x_obs.shape[1:])}"
            )
        if not bool(torch.all((obs_mask == 0) | (obs_mask == 1))):
            raise ValueError("obs_mask must be binary with 1=observed")

    def encode_sparse_history(self, x_obs: Tensor, obs_mask: Tensor) -> Tensor:
        """Encode only explicitly observed history values into one branch code."""

        self._validate_sparse_history(x_obs, obs_mask)
        mask = obs_mask.to(dtype=x_obs.dtype)
        safe_values = (x_obs * mask).flatten(start_dim=2)
        grid = _periodic_grid(
            self.height,
            self.width,
            device=x_obs.device,
            dtype=x_obs.dtype,
        )
        # Signed shortest periodic displacement from the grid center.
        relative_grid = torch.remainder(grid - 0.5 + 0.5, 1.0) - 0.5
        coordinate_channels = (
            relative_grid.reshape(1, 1, -1, 2)
            * mask.flatten(start_dim=2).unsqueeze(-1)
        ).flatten(start_dim=2)
        branch_input = torch.cat(
            (safe_values, coordinate_channels, mask.flatten(start_dim=2)),
            dim=-1,
        )
        features = self.branch_lifting(branch_input).permute(0, 2, 1)
        for layer in self.branch_layers:
            features = layer(features)
        branch_code = self.branch_projection(features.flatten(start_dim=1))
        if not torch.isfinite(branch_code).all():
            raise FloatingPointError("FNO-DeepONet branch code contains NaN/Inf")
        return branch_code

    def decode_grid(
        self,
        branch_code: Tensor,
        query_times: Tensor,
    ) -> Tensor:
        """Decode all spatial grid points for each normalized query time."""

        if branch_code.ndim != 2 or branch_code.shape[-1] != self.latent_dim:
            raise ValueError("branch_code must be (B,latent_dim)")
        query_times = torch.as_tensor(
            query_times,
            device=branch_code.device,
            dtype=branch_code.dtype,
        ).flatten()
        if query_times.numel() == 0:
            raise ValueError("At least one query time is required")
        grid = _periodic_grid(
            self.height,
            self.width,
            device=branch_code.device,
            dtype=branch_code.dtype,
        )
        points = self.height * self.width
        total_queries = int(query_times.numel()) * points
        decoded = []
        for start in range(0, total_queries, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, total_queries)
            flat_indices = torch.arange(start, stop, device=branch_code.device)
            time_indices = torch.div(flat_indices, points, rounding_mode="floor")
            point_indices = torch.remainder(flat_indices, points)
            query_features = self.query_embedding(
                query_times[time_indices],
                grid[point_indices],
            )
            trunk_code = (
                checkpoint(self.trunk, query_features, use_reentrant=False)
                if self.training and self.checkpoint_trunk
                else self.trunk(query_features)
            )
            decoded.append(
                torch.einsum("bl,ql->bq", branch_code, trunk_code)
                + self.output_bias
            )
        prediction = torch.cat(decoded, dim=1).view(
            branch_code.shape[0],
            query_times.numel(),
            self.height,
            self.width,
        )
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("FNO-DeepONet decoded field contains NaN/Inf")
        return prediction

    def forward(
        self,
        x_obs: Tensor,
        obs_mask: Tensor,
        query_times: Tensor,
    ) -> Tensor:
        return self.decode_grid(
            self.encode_sparse_history(x_obs, obs_mask),
            query_times,
        )


class FNODeepONetReconstructor(nn.Module):
    """Decode the 60-frame full history from sparse observations."""

    def __init__(
        self,
        operator: FNODeepONet,
        *,
        hard_observation_consistency: bool = False,
    ) -> None:
        super().__init__()
        self.operator = operator
        self.input_steps = operator.input_steps
        self.hard_observation_consistency = bool(hard_observation_consistency)

    def history_query_times(self, reference: Tensor) -> Tensor:
        return torch.linspace(
            -1.0,
            0.0,
            self.input_steps,
            device=reference.device,
            dtype=reference.dtype,
        )

    def forward_with_aux(
        self,
        x_obs: Tensor,
        obs_mask: Tensor,
    ) -> Mapping[str, Tensor]:
        raw = self.operator(
            x_obs,
            obs_mask,
            self.history_query_times(x_obs),
        )
        mask = obs_mask.to(dtype=x_obs.dtype)
        reconstruction = (
            raw * (1.0 - mask) + x_obs * mask
            if self.hard_observation_consistency
            else raw
        )
        return {
            "reconstruction": reconstruction,
            "raw_reconstruction": raw,
            "loss_reconstruction": raw,
        }

    def forward(self, x_obs: Tensor, obs_mask: Tensor) -> Tensor:
        return self.forward_with_aux(x_obs, obs_mask)["reconstruction"]


class FNODeepONetGridForecaster(nn.Module):
    """One-shot full-grid forecaster used in autoregressive curriculum rollout."""

    def __init__(
        self,
        operator: FNODeepONet,
        *,
        output_steps: int = 30,
    ) -> None:
        super().__init__()
        self.operator = operator
        self.output_steps = int(output_steps)
        if self.output_steps <= 0:
            raise ValueError("output_steps must be positive")

    def forward(self, history: Tensor) -> Tensor:
        if history.ndim != 4:
            raise ValueError("FNO-DeepONet forecaster expects (B,T,H,W)")
        all_valid = torch.ones_like(history)
        query_times = torch.arange(
            1,
            self.output_steps + 1,
            device=history.device,
            dtype=history.dtype,
        ) / float(self.output_steps)
        return self.operator(history, all_valid, query_times)


def fno_deeponet_kwargs(
    model_config: Mapping[str, object],
    *,
    input_steps: int,
    height: int,
    width: int,
    prefix: str = "fno_deeponet",
) -> Dict[str, object]:
    """Resolve one prefixed model configuration into constructor arguments."""

    def value(name: str, default: object) -> object:
        specific_key = f"{prefix}_{name}"
        if specific_key in model_config:
            return model_config[specific_key]
        return model_config.get(f"fno_deeponet_{name}", default)

    return {
        "input_steps": int(input_steps),
        "height": int(height),
        "width": int(width),
        "branch_width": int(value("branch_width", 64)),
        "branch_modes": int(value("branch_modes", 24)),
        "branch_layers": int(value("branch_layers", 4)),
        "branch_hidden": int(value("branch_hidden", 512)),
        "latent_dim": int(value("latent_dim", 256)),
        "trunk_hidden": int(value("trunk_hidden", 384)),
        "trunk_layers": int(value("trunk_layers", 4)),
        "time_fourier_bands": int(value("time_fourier_bands", 24)),
        "spatial_fourier_bands": int(value("spatial_fourier_bands", 16)),
        "time_max_frequency": float(value("time_max_frequency", 48.0)),
        "query_chunk_size": int(value("query_chunk_size", 8192)),
        "checkpoint_trunk": bool(value("checkpoint_trunk", True)),
    }


__all__ = [
    "FNODeepONet",
    "FNODeepONetGridForecaster",
    "FNODeepONetReconstructor",
    "FourierQueryEmbedding",
    "TemporalFNOLayer",
    "TemporalSpectralConv1d",
    "fno_deeponet_kwargs",
]
