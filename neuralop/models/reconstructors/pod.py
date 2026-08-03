"""Train-only POD basis utilities and gappy-POD spatial reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

import torch
from torch import Tensor, nn


def fit_pod_basis_from_snapshots(
    snapshots: Tensor,
    *,
    max_rank: int,
    seed: int,
    niter: int = 4,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return mean, orthonormal spatial modes and singular values.

    ``snapshots`` must contain flattened, already train-normalized fields as
    ``(N,H*W)``.  Randomized PCA is deterministic for a fixed torch runtime and seed.
    """

    if snapshots.ndim != 2:
        raise ValueError("POD snapshots must have shape (N,H*W)")
    max_rank = int(max_rank)
    if max_rank <= 0 or max_rank > min(snapshots.shape):
        raise ValueError("POD max_rank must be in [1,min(N,H*W)]")
    if not torch.isfinite(snapshots).all():
        raise FloatingPointError("POD snapshots contain NaN/Inf")
    mean = snapshots.mean(dim=0)
    centered = snapshots - mean
    devices = (
        [snapshots.device.index or torch.cuda.current_device()]
        if snapshots.is_cuda
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if snapshots.is_cuda:
            torch.cuda.manual_seed_all(int(seed))
        _, singular_values, right_vectors = torch.pca_lowrank(
            centered,
            q=max_rank,
            center=False,
            niter=int(niter),
        )
    modes = right_vectors.transpose(0, 1).contiguous()
    return mean, modes, singular_values


def load_pod_basis_artifact(
    path: Union[str, Path],
) -> Tuple[Tensor, Tensor, Mapping[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"POD basis artifact not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("POD basis artifact must contain a mapping")
    mean = payload.get("spatial_mean")
    modes = payload.get("spatial_modes")
    metadata = payload.get("metadata", {})
    if not isinstance(mean, Tensor) or not isinstance(modes, Tensor):
        raise KeyError("POD artifact requires spatial_mean and spatial_modes tensors")
    if mean.ndim != 1 or modes.ndim != 2 or modes.shape[1] != mean.numel():
        raise ValueError("POD artifact tensors have incompatible shapes")
    if not isinstance(metadata, Mapping):
        raise TypeError("POD artifact metadata must be a mapping")
    return mean.float(), modes.float(), metadata


class PODSpatialReconstructor(nn.Module):
    """Frame-wise gappy-POD reconstruction from sparse grid observations."""

    def __init__(
        self,
        spatial_mean: Tensor,
        spatial_modes: Tensor,
        *,
        rank: int,
        height: int,
        width: int,
        ridge: float = 1.0e-6,
    ) -> None:
        super().__init__()
        rank = int(rank)
        height = int(height)
        width = int(width)
        if spatial_mean.ndim != 1 or spatial_mean.numel() != height * width:
            raise ValueError("POD spatial_mean does not match height*width")
        if spatial_modes.ndim != 2 or spatial_modes.shape[1] != height * width:
            raise ValueError("POD spatial_modes must have shape (R,H*W)")
        if rank <= 0 or rank > spatial_modes.shape[0]:
            raise ValueError("POD rank exceeds the available basis")
        if float(ridge) < 0.0:
            raise ValueError("POD ridge must be non-negative")
        self.rank = rank
        self.height = height
        self.width = width
        self.ridge = float(ridge)
        self.register_buffer("spatial_mean", spatial_mean.detach().float().clone())
        self.register_buffer(
            "spatial_modes", spatial_modes[:rank].detach().float().clone()
        )

    @classmethod
    def from_artifact(
        cls,
        path: Union[str, Path],
        *,
        rank: int,
        ridge: float = 1.0e-6,
    ) -> "PODSpatialReconstructor":
        mean, modes, metadata = load_pod_basis_artifact(path)
        height = int(metadata.get("height", 0))
        width = int(metadata.get("width", 0))
        if height <= 0 or width <= 0:
            raise ValueError("POD artifact metadata requires positive height and width")
        return cls(
            mean,
            modes,
            rank=rank,
            height=height,
            width=width,
            ridge=ridge,
        )

    def _solve_mask_group(self, values: Tensor, observed: Tensor) -> Tensor:
        if not bool(observed.any()):
            return self.spatial_mean.unsqueeze(0).expand(values.shape[0], -1)
        modes_at_observations = self.spatial_modes[:, observed].transpose(0, 1)
        centered = values[:, observed] - self.spatial_mean[observed]
        gram = modes_at_observations.transpose(0, 1) @ modes_at_observations
        if self.ridge:
            gram = gram + self.ridge * torch.eye(
                self.rank, device=gram.device, dtype=gram.dtype
            )
        right_hand_side = centered @ modes_at_observations
        coefficients = torch.linalg.solve(
            gram, right_hand_side.transpose(0, 1)
        ).transpose(0, 1)
        return self.spatial_mean + coefficients @ self.spatial_modes

    def forward(self, x_obs: Tensor, obs_mask: Tensor) -> Tensor:
        if x_obs.ndim != 4 or obs_mask.ndim != 4:
            raise ValueError("POD expects x_obs and obs_mask as (B,T,H,W)")
        if x_obs.shape != obs_mask.shape:
            raise ValueError("x_obs and obs_mask must have identical shapes")
        if tuple(x_obs.shape[-2:]) != (self.height, self.width):
            raise ValueError("POD input spatial shape differs from its basis")
        if not torch.all((obs_mask == 0) | (obs_mask == 1)):
            raise ValueError("obs_mask must be binary with 1=observed")

        original_shape = x_obs.shape
        values = x_obs.reshape(-1, self.height * self.width).float()
        masks = obs_mask.reshape(-1, self.height * self.width).bool()
        unique_masks, inverse = torch.unique(masks, dim=0, return_inverse=True)
        reconstruction = torch.empty_like(values)
        for mask_index, observed in enumerate(unique_masks):
            rows = inverse == mask_index
            reconstruction[rows] = self._solve_mask_group(values[rows], observed)
        reconstruction = reconstruction.reshape(original_shape).to(dtype=x_obs.dtype)
        if not torch.isfinite(reconstruction).all():
            raise FloatingPointError("POD reconstruction contains NaN/Inf")
        return reconstruction

    def basis_state(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "height": self.height,
            "width": self.width,
            "ridge": self.ridge,
        }


__all__ = [
    "PODSpatialReconstructor",
    "fit_pod_basis_from_snapshots",
    "load_pod_basis_artifact",
]
