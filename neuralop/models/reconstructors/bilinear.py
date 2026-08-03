"""Deterministic frame-wise spatial interpolation for sparse sea surfaces."""

from __future__ import annotations

from itertools import product
from typing import Tuple

import numpy as np
import torch
from scipy.interpolate import griddata
from scipy.spatial import QhullError
from torch import Tensor, nn


class BilinearSpatialReconstructor(nn.Module):
    """Reconstruct each time frame independently from observed grid points.

    On arbitrary point masks, piecewise-linear interpolation on a Delaunay
    triangulation is the scattered-data extension of regular-grid bilinear
    interpolation. Periodic copies of observations are used by default for the
    HOS spatial domain. Nearest-neighbor values fill degenerate triangulations.

    This B1 baseline is deterministic and intentionally non-differentiable.
    """

    def __init__(self, *, periodic: bool = True, empty_fill_value: float = 0.0):
        super().__init__()
        self.periodic = bool(periodic)
        self.empty_fill_value = float(empty_fill_value)

    def forward(self, x_obs: Tensor, obs_mask: Tensor) -> Tensor:
        batched_x, batched_mask, squeezed = self._validate_inputs(x_obs, obs_mask)
        device = batched_x.device
        dtype = batched_x.dtype

        x_numpy = batched_x.detach().to(device="cpu", dtype=torch.float64).numpy()
        mask_numpy = batched_mask.detach().to(device="cpu").bool().numpy()
        reconstruction = np.empty_like(x_numpy)

        for batch_index in range(x_numpy.shape[0]):
            for time_index in range(x_numpy.shape[1]):
                reconstruction[batch_index, time_index] = self._interpolate_frame(
                    x_numpy[batch_index, time_index],
                    mask_numpy[batch_index, time_index],
                )

        result = torch.from_numpy(reconstruction).to(device=device, dtype=dtype)
        return result[0] if squeezed else result

    @staticmethod
    def _validate_inputs(x_obs: Tensor, obs_mask: Tensor) -> Tuple[Tensor, Tensor, bool]:
        if not isinstance(x_obs, Tensor) or not isinstance(obs_mask, Tensor):
            raise TypeError("x_obs and obs_mask must be torch.Tensor values")
        if x_obs.shape != obs_mask.shape:
            raise ValueError(
                f"x_obs and obs_mask shapes differ: {x_obs.shape} vs {obs_mask.shape}"
            )
        if x_obs.ndim not in (3, 4):
            raise ValueError(
                "Expected x_obs and obs_mask with shape (T,H,W) or (B,T,H,W), "
                f"got {tuple(x_obs.shape)}"
            )
        if not torch.isfinite(x_obs).all():
            raise ValueError("x_obs contains NaN or Inf")
        if not torch.all((obs_mask == 0) | (obs_mask == 1)):
            raise ValueError("obs_mask must be binary with 1=observed and 0=missing")

        squeezed = x_obs.ndim == 3
        if squeezed:
            return x_obs.unsqueeze(0), obs_mask.unsqueeze(0), True
        return x_obs, obs_mask, False

    def _interpolate_frame(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.all():
            return values.copy()
        if not mask.any():
            return np.full_like(values, self.empty_fill_value)

        height, width = values.shape
        observed_points = np.argwhere(mask).astype(np.float64)
        observed_values = values[mask]
        query_y, query_x = np.mgrid[0:height, 0:width]

        points, samples = self._periodic_observations(
            observed_points, observed_values, height, width
        )
        try:
            interpolated = griddata(
                points,
                samples,
                (query_y, query_x),
                method="linear",
                fill_value=np.nan,
            )
        except QhullError:
            interpolated = np.full_like(values, np.nan, dtype=np.float64)

        invalid = ~np.isfinite(interpolated)
        if invalid.any():
            nearest = griddata(
                points,
                samples,
                (query_y, query_x),
                method="nearest",
            )
            interpolated[invalid] = nearest[invalid]

        # Preserve observations exactly; validity is determined only by mask.
        interpolated[mask] = observed_values
        return np.nan_to_num(
            interpolated,
            nan=self.empty_fill_value,
            posinf=self.empty_fill_value,
            neginf=self.empty_fill_value,
        )

    def _periodic_observations(
        self,
        points: np.ndarray,
        values: np.ndarray,
        height: int,
        width: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.periodic:
            return points, values

        shifted_points = []
        shifted_values = []
        for row_shift, col_shift in product((-height, 0, height), (-width, 0, width)):
            shifted_points.append(
                points + np.asarray([row_shift, col_shift], dtype=np.float64)
            )
            shifted_values.append(values)
        return np.concatenate(shifted_points), np.concatenate(shifted_values)
