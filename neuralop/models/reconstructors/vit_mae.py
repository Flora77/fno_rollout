"""Sparse ViT/Tubelet masked autoencoders for periodic sea-surface histories.

The implementation follows the central MAE design choice: the encoder processes
visible tokens only, while mask tokens are introduced exclusively in the smaller
decoder.  The spatial-patch path follows He et al., *Masked Autoencoders Are
Scalable Vision Learners* (CVPR 2022).  The spacetime-agnostic tubelet path follows
Feichtenhofer et al., *Masked Autoencoders As Spatiotemporal Learners* (NeurIPS
2022).  Public inputs and outputs remain ``(B,T,H,W)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

import torch
from torch import Tensor, nn


Tokenization = Literal["spatial_patch", "tubelet"]


def _transformer_stack(
    *,
    dimension: int,
    depth: int,
    heads: int,
    mlp_ratio: float,
) -> nn.TransformerEncoder:
    if dimension <= 0 or depth <= 0 or heads <= 0:
        raise ValueError("Transformer dimensions, depth, and heads must be positive")
    if dimension % heads != 0:
        raise ValueError("Transformer dimension must be divisible by heads")
    layer = nn.TransformerEncoderLayer(
        d_model=int(dimension),
        nhead=int(heads),
        dim_feedforward=int(round(dimension * float(mlp_ratio))),
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=int(depth), norm=nn.LayerNorm(dimension))


def _periodic_fourier_positions(
    grid_shape: Sequence[int],
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return deterministic Fourier positions in token order ``(t,y,x)``.

    Spatial coordinates are periodic.  Tubelet time coordinates receive the same
    bounded Fourier representation, but time is not wrapped by the tokenizer.
    """

    if len(grid_shape) != 3:
        raise ValueError("grid_shape must be (time_groups, height_patches, width_patches)")
    time_groups, height_patches, width_patches = (int(value) for value in grid_shape)
    if min(time_groups, height_patches, width_patches, dimension) <= 0:
        raise ValueError("Position grid and embedding dimension must be positive")

    axes = []
    for size in (time_groups, height_patches, width_patches):
        if size == 1:
            axes.append(torch.zeros(1, device=device, dtype=dtype))
        else:
            axes.append(torch.arange(size, device=device, dtype=dtype) / float(size))
    tt, yy, xx = torch.meshgrid(*axes, indexing="ij")
    coordinates = (tt.reshape(-1), yy.reshape(-1), xx.reshape(-1))

    # Four sine/cosine values per frequency for the periodic spatial axes and,
    # when applicable, two for the temporal tubelet axis.
    include_time = time_groups > 1
    values_per_frequency = 6 if include_time else 4
    bands = max(1, math.ceil(dimension / values_per_frequency))
    features = []
    for frequency in range(1, bands + 1):
        angle = 2.0 * math.pi * float(frequency)
        if include_time:
            features.extend(
                (torch.sin(angle * coordinates[0]), torch.cos(angle * coordinates[0]))
            )
        features.extend(
            (
                torch.sin(angle * coordinates[1]),
                torch.cos(angle * coordinates[1]),
                torch.sin(angle * coordinates[2]),
                torch.cos(angle * coordinates[2]),
            )
        )
    return torch.stack(features, dim=-1)[:, :dimension]


@dataclass(frozen=True)
class TokenGrid:
    time_groups: int
    height_patches: int
    width_patches: int
    temporal_patch: int
    spatial_patch: int

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.time_groups, self.height_patches, self.width_patches

    @property
    def token_count(self) -> int:
        return self.time_groups * self.height_patches * self.width_patches

    @property
    def values_per_token(self) -> int:
        return self.temporal_patch * self.spatial_patch * self.spatial_patch


class PeriodicViTMaskedAutoencoder(nn.Module):
    """Reconstruct a 60-frame history with visible-only ViT encoding.

    ``spatial_patch`` treats time as channels and creates one token per spatial
    patch. ``tubelet`` creates explicit non-overlapping ``(time,y,x)`` tokens.
    In both cases a token is visible when at least one entry in its explicit
    observation mask is valid.
    """

    def __init__(
        self,
        *,
        input_steps: int = 60,
        patch_size: int = 4,
        tokenization: Tokenization = "spatial_patch",
        tubelet_size: int = 10,
        encoder_dim: int = 128,
        encoder_depth: int = 4,
        encoder_heads: int = 4,
        decoder_dim: int = 64,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.input_steps = int(input_steps)
        self.patch_size = int(patch_size)
        self.tokenization = str(tokenization)
        self.tubelet_size = int(tubelet_size)
        if self.input_steps <= 0 or self.patch_size <= 0:
            raise ValueError("input_steps and patch_size must be positive")
        if self.tokenization not in {"spatial_patch", "tubelet"}:
            raise ValueError("tokenization must be 'spatial_patch' or 'tubelet'")
        if self.tokenization == "tubelet":
            if self.tubelet_size <= 0 or self.input_steps % self.tubelet_size:
                raise ValueError("tubelet_size must divide input_steps")
            temporal_patch = self.tubelet_size
        else:
            temporal_patch = self.input_steps

        values_per_token = temporal_patch * self.patch_size * self.patch_size
        self.encoder_dim = int(encoder_dim)
        self.decoder_dim = int(decoder_dim)
        self.patch_embedding = nn.Linear(2 * values_per_token, self.encoder_dim)
        self.encoder = _transformer_stack(
            dimension=self.encoder_dim,
            depth=int(encoder_depth),
            heads=int(encoder_heads),
            mlp_ratio=float(mlp_ratio),
        )
        self.encoder_to_decoder = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.decoder = _transformer_stack(
            dimension=self.decoder_dim,
            depth=int(decoder_depth),
            heads=int(decoder_heads),
            mlp_ratio=float(mlp_ratio),
        )
        self.output_projection = nn.Linear(self.decoder_dim, values_per_token)
        self.mask_token = nn.Parameter(torch.empty(1, 1, self.decoder_dim))
        self.empty_context_token = nn.Parameter(torch.empty(1, 1, self.encoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.empty_context_token, std=0.02)

    def _grid(self, height: int, width: int) -> TokenGrid:
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Spatial dimensions must be divisible by patch_size")
        temporal_patch = (
            self.input_steps
            if self.tokenization == "spatial_patch"
            else self.tubelet_size
        )
        return TokenGrid(
            time_groups=self.input_steps // temporal_patch,
            height_patches=height // self.patch_size,
            width_patches=width // self.patch_size,
            temporal_patch=temporal_patch,
            spatial_patch=self.patch_size,
        )

    @staticmethod
    def _patchify(value: Tensor, grid: TokenGrid) -> Tensor:
        batch, time_steps, height, width = value.shape
        expected = (
            grid.time_groups * grid.temporal_patch,
            grid.height_patches * grid.spatial_patch,
            grid.width_patches * grid.spatial_patch,
        )
        if (time_steps, height, width) != expected:
            raise ValueError(
                f"Tensor shape {(time_steps, height, width)} does not match token grid "
                f"{expected}"
            )
        patches = value.reshape(
            batch,
            grid.time_groups,
            grid.temporal_patch,
            grid.height_patches,
            grid.spatial_patch,
            grid.width_patches,
            grid.spatial_patch,
        )
        patches = patches.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        return patches.reshape(batch, grid.token_count, grid.values_per_token)

    @staticmethod
    def _unpatchify(patches: Tensor, grid: TokenGrid) -> Tensor:
        batch = patches.shape[0]
        value = patches.reshape(
            batch,
            grid.time_groups,
            grid.height_patches,
            grid.width_patches,
            grid.temporal_patch,
            grid.spatial_patch,
            grid.spatial_patch,
        )
        value = value.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        return value.reshape(
            batch,
            grid.time_groups * grid.temporal_patch,
            grid.height_patches * grid.spatial_patch,
            grid.width_patches * grid.spatial_patch,
        )

    def token_visibility(self, obs_mask: Tensor) -> Tensor:
        if obs_mask.ndim != 4 or obs_mask.shape[1] != self.input_steps:
            raise ValueError("obs_mask must be (B,input_steps,H,W)")
        grid = self._grid(obs_mask.shape[-2], obs_mask.shape[-1])
        return self._patchify(obs_mask, grid).bool().any(dim=-1)

    def _encode_visible(
        self,
        embeddings: Tensor,
        positions: Tensor,
        visible: Tensor,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        visible_indices = [
            torch.nonzero(sample_visible, as_tuple=False).flatten()
            for sample_visible in visible
        ]
        maximum = max(1, max((int(index.numel()) for index in visible_indices), default=0))
        batch = embeddings.shape[0]
        sequences = embeddings.new_zeros((batch, maximum, self.encoder_dim))
        padding = torch.ones((batch, maximum), device=embeddings.device, dtype=torch.bool)
        for sample, indices in enumerate(visible_indices):
            count = int(indices.numel())
            if count:
                sequences[sample, :count] = embeddings[sample, indices] + positions[indices]
                padding[sample, :count] = False
            else:
                sequences[sample, 0] = self.empty_context_token[0, 0]
                padding[sample, 0] = False
        encoded = self.encoder(sequences, src_key_padding_mask=padding)
        return encoded, padding, visible_indices

    def forward_with_aux(
        self, x_obs: Tensor, obs_mask: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor | str]]:
        if x_obs.ndim != 4 or x_obs.shape != obs_mask.shape:
            raise ValueError("x_obs and obs_mask must have identical (B,T,H,W) shapes")
        if x_obs.shape[1] != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} history frames")
        if not torch.all((obs_mask == 0) | (obs_mask == 1)):
            raise ValueError("obs_mask must be binary with 1=observed")

        grid = self._grid(x_obs.shape[-2], x_obs.shape[-1])
        mask = obs_mask.to(device=x_obs.device, dtype=x_obs.dtype)
        # Multiplication makes the model invariant to arbitrary storage values at M=0.
        value_patches = self._patchify(x_obs * mask, grid)
        mask_patches = self._patchify(mask, grid)
        embeddings = self.patch_embedding(torch.cat((value_patches, mask_patches), dim=-1))
        visible = mask_patches.bool().any(dim=-1)
        encoder_positions = _periodic_fourier_positions(
            grid.shape,
            self.encoder_dim,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        encoded, padding, visible_indices = self._encode_visible(
            embeddings, encoder_positions, visible
        )

        decoder_positions = _periodic_fourier_positions(
            grid.shape,
            self.decoder_dim,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        projected = self.encoder_to_decoder(encoded)
        decoder_tokens = (
            self.mask_token.to(device=projected.device, dtype=projected.dtype)
            .expand(x_obs.shape[0], grid.token_count, self.decoder_dim)
            .clone()
        )
        decoder_positions = decoder_positions.to(dtype=projected.dtype)
        for sample, indices in enumerate(visible_indices):
            count = int(indices.numel())
            if count:
                decoder_tokens[sample, indices] = projected[sample, :count]
        decoder_tokens = decoder_tokens + decoder_positions.unsqueeze(0)
        decoded = self.decoder(decoder_tokens)
        reconstruction = self._unpatchify(self.output_projection(decoded), grid)
        if reconstruction.shape != x_obs.shape:
            raise RuntimeError("ViT-MAE reconstruction shape differs from input")
        if not torch.isfinite(reconstruction).all():
            raise FloatingPointError("ViT-MAE reconstruction contains NaN/Inf")
        return reconstruction, {
            "tokenization": self.tokenization,
            "visible_token_count": visible.sum(dim=1),
            "encoder_sequence_length": (~padding).sum(dim=1),
        }

    def forward(self, x_obs: Tensor, obs_mask: Tensor) -> Tensor:
        reconstruction, _ = self.forward_with_aux(x_obs, obs_mask)
        return reconstruction


class PeriodicMaskAwarePoolingViTReconstructor(PeriodicViTMaskedAutoencoder):
    """Spatial ViT whose patch tokens average only observed spatial samples.

    The encoder and decoder modules are inherited unchanged from the V0 spatial
    ViT.  For each time step and spatial patch, observed values are summed and
    divided by the number of valid spatial samples.  The resulting 60-value
    history is expanded back to the V0 patch-embedding width, so this
    reconstructor has exactly the same parameter count as V0.  Observation
    counts are returned for diagnostics but are not embedded, gated, or used as
    token confidence.
    """

    tokenization_name = "spatial_mask_aware_pool"

    def __init__(self, **kwargs: object) -> None:
        requested = str(kwargs.pop("tokenization", "spatial_patch"))
        if requested not in {"spatial_patch", self.tokenization_name}:
            raise ValueError(
                "Mask-aware pooling supports spatial patches only, got "
                f"{requested!r}"
            )
        super().__init__(tokenization="spatial_patch", **kwargs)

    def _mask_aware_token_inputs(
        self, x_obs: Tensor, obs_mask: Tensor, grid: TokenGrid
    ) -> Tuple[Tensor, Tensor, Tensor]:
        mask = obs_mask.to(device=x_obs.device, dtype=x_obs.dtype)
        batch = x_obs.shape[0]
        spatial_samples = grid.spatial_patch * grid.spatial_patch

        value_locations = self._patchify(x_obs * mask, grid).reshape(
            batch, grid.token_count, self.input_steps, spatial_samples
        )
        mask_locations = self._patchify(mask, grid).reshape(
            batch, grid.token_count, self.input_steps, spatial_samples
        )
        observed_count = mask_locations.sum(dim=-1)
        pooled_history = value_locations.sum(dim=-1) / observed_count.clamp_min(1.0)
        pooled_history = torch.where(
            observed_count > 0,
            pooled_history,
            torch.zeros_like(pooled_history),
        )

        # Preserve the V0 patch-embedding width without exposing the count as a
        # confidence feature.  The binary validity history records only whether
        # a time step has any observation in the patch.
        pooled_values = (
            pooled_history.unsqueeze(-1)
            .expand(batch, grid.token_count, self.input_steps, spatial_samples)
            .reshape(batch, grid.token_count, grid.values_per_token)
        )
        pooled_validity = (
            (observed_count > 0)
            .to(dtype=x_obs.dtype)
            .unsqueeze(-1)
            .expand(batch, grid.token_count, self.input_steps, spatial_samples)
            .reshape(batch, grid.token_count, grid.values_per_token)
        )
        visible = (observed_count > 0).any(dim=-1)
        embedding_inputs = torch.cat((pooled_values, pooled_validity), dim=-1)
        return embedding_inputs, visible, observed_count

    def forward_with_aux(
        self, x_obs: Tensor, obs_mask: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor | str]]:
        if x_obs.ndim != 4 or x_obs.shape != obs_mask.shape:
            raise ValueError("x_obs and obs_mask must have identical (B,T,H,W) shapes")
        if x_obs.shape[1] != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} history frames")
        if not torch.all((obs_mask == 0) | (obs_mask == 1)):
            raise ValueError("obs_mask must be binary with 1=observed")

        grid = self._grid(x_obs.shape[-2], x_obs.shape[-1])
        embedding_inputs, visible, observed_count = self._mask_aware_token_inputs(
            x_obs, obs_mask, grid
        )
        embeddings = self.patch_embedding(embedding_inputs)
        encoder_positions = _periodic_fourier_positions(
            grid.shape,
            self.encoder_dim,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        encoded, padding, visible_indices = self._encode_visible(
            embeddings, encoder_positions, visible
        )

        decoder_positions = _periodic_fourier_positions(
            grid.shape,
            self.decoder_dim,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        projected = self.encoder_to_decoder(encoded)
        decoder_tokens = (
            self.mask_token.to(device=projected.device, dtype=projected.dtype)
            .expand(x_obs.shape[0], grid.token_count, self.decoder_dim)
            .clone()
        )
        decoder_positions = decoder_positions.to(dtype=projected.dtype)
        for sample, indices in enumerate(visible_indices):
            count = int(indices.numel())
            if count:
                decoder_tokens[sample, indices] = projected[sample, :count]
        decoder_tokens = decoder_tokens + decoder_positions.unsqueeze(0)
        decoded = self.decoder(decoder_tokens)
        reconstruction = self._unpatchify(self.output_projection(decoded), grid)
        if reconstruction.shape != x_obs.shape:
            raise RuntimeError("Mask-aware ViT reconstruction shape differs from input")
        if not torch.isfinite(reconstruction).all():
            raise FloatingPointError(
                "Mask-aware ViT reconstruction contains NaN/Inf"
            )
        return reconstruction, {
            "tokenization": self.tokenization_name,
            "visible_token_count": visible.sum(dim=1),
            "encoder_sequence_length": (~padding).sum(dim=1),
            "observed_count_per_time": observed_count,
        }


class PeriodicConfidenceMaskAwarePoolingViTReconstructor(
    PeriodicMaskAwarePoolingViTReconstructor
):
    """PC-D1 with a learned additive embedding of patch observation confidence.

    A spatial patch has confidence ``c_p = mean_t(n_observed(t)) / patch_area``.
    The scalar is independently mapped to the encoder width and added to the
    corresponding patch embedding.  Confidence does not change token
    visibility and is not used as a gate or attention bias.
    """

    tokenization_name = "spatial_mask_aware_pool_confidence"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.confidence_embedding = nn.Linear(1, self.encoder_dim)

    @staticmethod
    def _patch_confidence(observed_count: Tensor, grid: TokenGrid) -> Tensor:
        spatial_samples = grid.spatial_patch * grid.spatial_patch
        confidence = observed_count.mean(dim=-1) / float(spatial_samples)
        return confidence.clamp(0.0, 1.0)

    def forward_with_aux(
        self, x_obs: Tensor, obs_mask: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor | str]]:
        if x_obs.ndim != 4 or x_obs.shape != obs_mask.shape:
            raise ValueError("x_obs and obs_mask must have identical (B,T,H,W) shapes")
        if x_obs.shape[1] != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} history frames")
        if not torch.all((obs_mask == 0) | (obs_mask == 1)):
            raise ValueError("obs_mask must be binary with 1=observed")

        grid = self._grid(x_obs.shape[-2], x_obs.shape[-1])
        embedding_inputs, visible, observed_count = self._mask_aware_token_inputs(
            x_obs, obs_mask, grid
        )
        patch_confidence = self._patch_confidence(observed_count, grid)
        embeddings = self.patch_embedding(embedding_inputs)
        embeddings = embeddings + self.confidence_embedding(
            patch_confidence.unsqueeze(-1).to(dtype=embeddings.dtype)
        )
        encoder_positions = _periodic_fourier_positions(
            grid.shape,
            self.encoder_dim,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        encoded, padding, visible_indices = self._encode_visible(
            embeddings, encoder_positions, visible
        )

        decoder_positions = _periodic_fourier_positions(
            grid.shape,
            self.decoder_dim,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        projected = self.encoder_to_decoder(encoded)
        decoder_tokens = (
            self.mask_token.to(device=projected.device, dtype=projected.dtype)
            .expand(x_obs.shape[0], grid.token_count, self.decoder_dim)
            .clone()
        )
        decoder_positions = decoder_positions.to(dtype=projected.dtype)
        for sample, indices in enumerate(visible_indices):
            count = int(indices.numel())
            if count:
                decoder_tokens[sample, indices] = projected[sample, :count]
        decoder_tokens = decoder_tokens + decoder_positions.unsqueeze(0)
        decoded = self.decoder(decoder_tokens)
        reconstruction = self._unpatchify(self.output_projection(decoded), grid)
        if reconstruction.shape != x_obs.shape:
            raise RuntimeError("Confidence ViT reconstruction shape differs from input")
        if not torch.isfinite(reconstruction).all():
            raise FloatingPointError(
                "Confidence ViT reconstruction contains NaN/Inf"
            )
        return reconstruction, {
            "tokenization": self.tokenization_name,
            "visible_token_count": visible.sum(dim=1),
            "encoder_sequence_length": (~padding).sum(dim=1),
            "observed_count_per_time": observed_count,
            "patch_confidence": patch_confidence,
        }


def random_token_observation_mask(
    reference: Tensor,
    *,
    patch_size: int,
    mask_ratio: float,
    tokenization: Tokenization = "spatial_patch",
    tubelet_size: int = 10,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate an exact-ratio random MAE mask and expand it to ``(B,T,H,W)``.

    The returned convention is the repository convention: ``1=visible/observed``.
    Randomness comes from PyTorch's checkpointed RNG state, so resume remains exact.
    """

    if reference.ndim != 4:
        raise ValueError("reference must be (B,T,H,W)")
    if not 0.0 < float(mask_ratio) < 1.0:
        raise ValueError("mask_ratio must lie strictly between zero and one")
    batch, time_steps, height, width = reference.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Spatial dimensions must be divisible by patch_size")
    temporal_patch = time_steps if tokenization == "spatial_patch" else int(tubelet_size)
    if temporal_patch <= 0 or time_steps % temporal_patch:
        raise ValueError("tubelet_size must divide the number of history frames")
    time_groups = time_steps // temporal_patch
    height_patches = height // patch_size
    width_patches = width // patch_size
    token_count = time_groups * height_patches * width_patches
    keep_count = max(1, int(round(token_count * (1.0 - float(mask_ratio)))))
    noise = torch.rand(
        batch,
        token_count,
        device=reference.device,
        generator=generator,
    )
    keep_indices = noise.argsort(dim=1)[:, :keep_count]
    token_mask = torch.zeros(
        batch, token_count, device=reference.device, dtype=torch.bool
    )
    token_mask.scatter_(1, keep_indices, True)
    token_mask = token_mask.reshape(
        batch, time_groups, height_patches, width_patches, 1, 1, 1
    ).expand(
        batch,
        time_groups,
        height_patches,
        width_patches,
        temporal_patch,
        patch_size,
        patch_size,
    )
    token_mask = token_mask.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    return token_mask.reshape(batch, time_steps, height, width).to(reference.dtype)


__all__ = [
    "PeriodicConfidenceMaskAwarePoolingViTReconstructor",
    "PeriodicMaskAwarePoolingViTReconstructor",
    "PeriodicViTMaskedAutoencoder",
    "TokenGrid",
    "random_token_observation_mask",
]
