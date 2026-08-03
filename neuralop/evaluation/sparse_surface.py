"""Per-source reconstruction and forecast metrics for sparse sea surfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, Mapping, Sequence

import torch
from torch import Tensor


def denormalize_sea_surface(field: Tensor, mean: float, std: float) -> Tensor:
    mean = float(mean)
    std = float(std)
    if not math.isfinite(mean):
        raise ValueError("Training-set normalization mean must be finite")
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("Training-set normalization std must be finite and positive")
    return field * std + mean


@dataclass
class _FieldMetricAccumulator:
    """Streaming equivalent of the established B0 paper metric accumulator."""

    count: int = 0
    sqerr_sum: float = 0.0
    true_sum: float = 0.0
    pred_sum: float = 0.0
    true_sq_sum: float = 0.0
    pred_sq_sum: float = 0.0
    cross_sum: float = 0.0
    fft_diff_sq_sum: float = 0.0
    fft_true_sq_sum: float = 0.0
    fft_pred_sq_sum: float = 0.0
    gradient_count: int = 0
    gradient_diff_sq_sum: float = 0.0
    gradient_true_sq_sum: float = 0.0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes differ: {prediction.shape} vs {target.shape}"
            )
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("Metric inputs contain NaN or Inf")
        prediction = prediction.detach().cpu().to(dtype=torch.float64)
        target = target.detach().cpu().to(dtype=torch.float64)
        difference = prediction - target
        self.count += target.numel()
        self.sqerr_sum += float(difference.square().sum())
        self.true_sum += float(target.sum())
        self.pred_sum += float(prediction.sum())
        self.true_sq_sum += float(target.square().sum())
        self.pred_sq_sum += float(prediction.square().sum())
        self.cross_sum += float((target * prediction).sum())

        true_fft = torch.fft.rfft2(target, dim=(-2, -1), norm="ortho")
        pred_fft = torch.fft.rfft2(prediction, dim=(-2, -1), norm="ortho")
        self.fft_diff_sq_sum += float((pred_fft - true_fft).abs().square().sum())
        self.fft_true_sq_sum += float(true_fft.abs().square().sum())
        self.fft_pred_sq_sum += float(pred_fft.abs().square().sum())

        pred_dx = torch.roll(prediction, shifts=-1, dims=-1) - prediction
        pred_dy = torch.roll(prediction, shifts=-1, dims=-2) - prediction
        true_dx = torch.roll(target, shifts=-1, dims=-1) - target
        true_dy = torch.roll(target, shifts=-1, dims=-2) - target
        self.gradient_count += 2 * target.numel()
        self.gradient_diff_sq_sum += float(
            (pred_dx - true_dx).square().sum() + (pred_dy - true_dy).square().sum()
        )
        self.gradient_true_sq_sum += float(
            true_dx.square().sum() + true_dy.square().sum()
        )

    def compute(self) -> Dict[str, float]:
        if self.count <= 0:
            raise RuntimeError("Cannot finalize an empty field metric accumulator")
        eps = 1.0e-12
        count = float(self.count)
        rmse = math.sqrt(max(self.sqerr_sum / count, 0.0))
        true_mean = self.true_sum / count
        true_variance = max(self.true_sq_sum / count - true_mean * true_mean, 0.0)
        hs_reference = 4.0 * math.sqrt(max(true_variance, eps))
        nrmse = rmse / max(hs_reference, eps)
        ssp = math.sqrt(max(self.fft_diff_sq_sum, 0.0)) / (
            math.sqrt(max(self.fft_true_sq_sum, 0.0))
            + math.sqrt(max(self.fft_pred_sq_sum, 0.0))
            + eps
        )
        covariance = self.cross_sum - self.true_sum * self.pred_sum / count
        true_ss = self.true_sq_sum - self.true_sum * self.true_sum / count
        pred_ss = self.pred_sq_sum - self.pred_sum * self.pred_sum / count
        correlation = covariance / (
            math.sqrt(max(true_ss, eps)) * math.sqrt(max(pred_ss, eps))
        )
        gradient_rmse = math.sqrt(
            max(self.gradient_diff_sq_sum / max(self.gradient_count, 1), 0.0)
        )
        gradient_reference = math.sqrt(
            max(self.gradient_true_sq_sum / max(self.gradient_count, 1), eps)
        )
        return {
            "rmse": rmse,
            "rmse_phys": rmse,
            "nrmse": nrmse,
            "nrmse_hs": nrmse,
            "hs_ref": hs_reference,
            "ssp": ssp,
            "correlation": correlation,
            "corr": correlation,
            "gradient_rmse": gradient_rmse,
            "gradient_nrmse": gradient_rmse / max(gradient_reference, eps),
        }


@dataclass
class _PointMetricAccumulator:
    """Metrics for an arbitrary observed or missing subset without FFT assumptions."""

    count: int = 0
    sqerr_sum: float = 0.0
    true_sum: float = 0.0
    pred_sum: float = 0.0
    true_sq_sum: float = 0.0
    pred_sq_sum: float = 0.0
    cross_sum: float = 0.0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        prediction = prediction.detach().cpu().reshape(-1).to(dtype=torch.float64)
        target = target.detach().cpu().reshape(-1).to(dtype=torch.float64)
        if prediction.shape != target.shape or prediction.numel() == 0:
            raise ValueError("Region metric inputs must be non-empty and equally shaped")
        difference = prediction - target
        self.count += target.numel()
        self.sqerr_sum += float(difference.square().sum())
        self.true_sum += float(target.sum())
        self.pred_sum += float(prediction.sum())
        self.true_sq_sum += float(target.square().sum())
        self.pred_sq_sum += float(prediction.square().sum())
        self.cross_sum += float((target * prediction).sum())

    def compute(self) -> Dict[str, float]:
        if self.count <= 0:
            return {
                "rmse": 0.0,
                "rmse_phys": 0.0,
                "nrmse": 0.0,
                "nrmse_hs": 0.0,
                "hs_ref": 0.0,
                "correlation": 0.0,
                "corr": 0.0,
            }
        eps = 1.0e-12
        count = float(self.count)
        rmse = math.sqrt(max(self.sqerr_sum / count, 0.0))
        true_mean = self.true_sum / count
        true_variance = max(self.true_sq_sum / count - true_mean * true_mean, 0.0)
        hs_reference = 4.0 * math.sqrt(max(true_variance, eps))
        covariance = self.cross_sum - self.true_sum * self.pred_sum / count
        true_ss = self.true_sq_sum - self.true_sum * self.true_sum / count
        pred_ss = self.pred_sq_sum - self.pred_sum * self.pred_sum / count
        correlation = covariance / (
            math.sqrt(max(true_ss, eps)) * math.sqrt(max(pred_ss, eps))
        )
        return {
            "rmse": rmse,
            "rmse_phys": rmse,
            "nrmse": rmse / max(hs_reference, eps),
            "nrmse_hs": rmse / max(hs_reference, eps),
            "hs_ref": hs_reference,
            "correlation": correlation,
            "corr": correlation,
        }


@dataclass
class _FrameGrowthAccumulator:
    """Streaming per-forecast-frame error growth statistics."""

    count: Tensor | None = None
    sqerr_sum: Tensor | None = None
    true_sum: Tensor | None = None
    pred_sum: Tensor | None = None
    true_sq_sum: Tensor | None = None
    pred_sq_sum: Tensor | None = None
    cross_sum: Tensor | None = None

    def update(self, prediction: Tensor, target: Tensor) -> None:
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("Growth-curve inputs must share shape (T,H,W)")
        prediction = prediction.detach().cpu().to(dtype=torch.float64)
        target = target.detach().cpu().to(dtype=torch.float64)
        reduce_dims = (-2, -1)
        spatial_count = prediction.shape[-2] * prediction.shape[-1]
        values = {
            "count": torch.full(
                (prediction.shape[0],), spatial_count, dtype=torch.float64
            ),
            "sqerr_sum": (prediction - target).square().sum(dim=reduce_dims),
            "true_sum": target.sum(dim=reduce_dims),
            "pred_sum": prediction.sum(dim=reduce_dims),
            "true_sq_sum": target.square().sum(dim=reduce_dims),
            "pred_sq_sum": prediction.square().sum(dim=reduce_dims),
            "cross_sum": (prediction * target).sum(dim=reduce_dims),
        }
        if self.count is None:
            for name, value in values.items():
                setattr(self, name, value.clone())
            return
        if self.count.shape != values["count"].shape:
            raise ValueError("Forecast length changed while accumulating growth curve")
        for name, value in values.items():
            setattr(self, name, getattr(self, name) + value)

    def compute(self, frame_interval: float) -> list[Dict[str, float]]:
        if self.count is None:
            raise RuntimeError("Cannot finalize an empty growth-curve accumulator")
        eps = 1.0e-12
        rows = []
        for index in range(self.count.numel()):
            count = float(self.count[index])
            sqerr = float(self.sqerr_sum[index])
            true_sum = float(self.true_sum[index])
            pred_sum = float(self.pred_sum[index])
            true_sq_sum = float(self.true_sq_sum[index])
            pred_sq_sum = float(self.pred_sq_sum[index])
            cross_sum = float(self.cross_sum[index])
            rmse = math.sqrt(max(sqerr / count, 0.0))
            true_mean = true_sum / count
            variance = max(true_sq_sum / count - true_mean * true_mean, 0.0)
            hs_reference = 4.0 * math.sqrt(max(variance, eps))
            covariance = cross_sum - true_sum * pred_sum / count
            true_ss = true_sq_sum - true_sum * true_sum / count
            pred_ss = pred_sq_sum - pred_sum * pred_sum / count
            correlation = covariance / (
                math.sqrt(max(true_ss, eps)) * math.sqrt(max(pred_ss, eps))
            )
            frame = index + 1
            rows.append(
                {
                    "frame": float(frame),
                    "time_seconds": float(frame * frame_interval),
                    "rmse": rmse,
                    "nrmse": rmse / max(hs_reference, eps),
                    "correlation": correlation,
                }
            )
        return rows


@dataclass
class _SourceAccumulator:
    history: _FieldMetricAccumulator = field(default_factory=_FieldMetricAccumulator)
    forecasts: Dict[int, _FieldMetricAccumulator] = field(default_factory=dict)
    missing: _PointMetricAccumulator = field(default_factory=_PointMetricAccumulator)
    observed: _PointMetricAccumulator = field(default_factory=_PointMetricAccumulator)
    growth: _FrameGrowthAccumulator = field(default_factory=_FrameGrowthAccumulator)


@dataclass
class _HistorySourceAccumulator:
    full: _FieldMetricAccumulator = field(default_factory=_FieldMetricAccumulator)
    missing: _PointMetricAccumulator = field(default_factory=_PointMetricAccumulator)
    observed: _PointMetricAccumulator = field(default_factory=_PointMetricAccumulator)


def _normalize_source_ids(
    source_ids: Sequence[Hashable] | Tensor, batch_size: int
) -> list[Hashable]:
    if isinstance(source_ids, Tensor):
        source_ids = source_ids.detach().cpu().reshape(-1).tolist()
    ids = list(source_ids)
    if len(ids) != batch_size:
        raise ValueError(
            f"source_ids length {len(ids)} does not match batch size {batch_size}"
        )
    for source_id in ids:
        try:
            hash(source_id)
        except TypeError as error:
            raise TypeError(f"source_id must be hashable, got {source_id!r}") from error
    return ids


def _mean_metric_dict(metrics: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    rows = list(metrics)
    if not rows:
        raise RuntimeError("Cannot average an empty metric collection")
    keys = tuple(rows[0])
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


class SparseSurfaceMetricAccumulator:
    """Accumulate physical metrics per source MAT file, then average files equally."""

    def __init__(
        self,
        *,
        normalization_mean: float,
        normalization_std: float,
        forecast_horizons: Sequence[int] = (30, 60, 120, 180, 240, 300),
        frame_interval: float = 0.25,
    ) -> None:
        # Validate provenance values immediately; callers must pass B0 training stats.
        denormalize_sea_surface(torch.zeros(()), normalization_mean, normalization_std)
        horizons = tuple(sorted({int(horizon) for horizon in forecast_horizons}))
        if not horizons or horizons[0] <= 0:
            raise ValueError("forecast_horizons must contain positive frame counts")
        self.normalization_mean = float(normalization_mean)
        self.normalization_std = float(normalization_std)
        self.forecast_horizons = horizons
        self.frame_interval = float(frame_interval)
        if not math.isfinite(self.frame_interval) or self.frame_interval <= 0.0:
            raise ValueError("frame_interval must be finite and positive")
        self._sources: Dict[Hashable, _SourceAccumulator] = {}

    @torch.no_grad()
    def update(
        self,
        history_reconstruction: Tensor,
        x_full: Tensor,
        forecast: Tensor,
        y: Tensor,
        obs_mask: Tensor,
        *,
        source_ids: Sequence[Hashable] | Tensor,
    ) -> None:
        if history_reconstruction.shape != x_full.shape:
            raise ValueError("History reconstruction and x_full must have identical shapes")
        if obs_mask.shape != x_full.shape:
            raise ValueError("obs_mask and x_full must have identical shapes")
        if forecast.ndim != 4 or y.ndim != 4 or y.shape[1] < forecast.shape[1]:
            raise ValueError("y must contain at least as many future frames as forecast")
        if forecast.shape[0] != y.shape[0] or forecast.shape[2:] != y.shape[2:]:
            raise ValueError("Forecast and y batch/spatial shapes differ")
        if not torch.all((obs_mask == 0) | (obs_mask == 1)):
            raise ValueError("obs_mask must be binary with 1=observed and 0=missing")

        ids = _normalize_source_ids(source_ids, x_full.shape[0])
        history_phy = denormalize_sea_surface(
            history_reconstruction, self.normalization_mean, self.normalization_std
        ).detach().cpu()
        x_full_phy = denormalize_sea_surface(
            x_full, self.normalization_mean, self.normalization_std
        ).detach().cpu()
        forecast_phy = denormalize_sea_surface(
            forecast, self.normalization_mean, self.normalization_std
        ).detach().cpu()
        y_phy = denormalize_sea_surface(
            y[:, : forecast.shape[1]], self.normalization_mean, self.normalization_std
        ).detach().cpu()
        mask = obs_mask.detach().cpu().bool()

        for local_index, source_id in enumerate(ids):
            source = self._sources.setdefault(source_id, _SourceAccumulator())
            source.history.update(history_phy[local_index], x_full_phy[local_index])
            local_mask = mask[local_index]
            local_missing = ~local_mask
            if local_missing.any():
                source.missing.update(
                    history_phy[local_index][local_missing],
                    x_full_phy[local_index][local_missing],
                )
            if local_mask.any():
                source.observed.update(
                    history_phy[local_index][local_mask],
                    x_full_phy[local_index][local_mask],
                )

            source.growth.update(forecast_phy[local_index], y_phy[local_index])

            for horizon in self.forecast_horizons:
                if horizon > forecast.shape[1]:
                    continue
                accumulator = source.forecasts.setdefault(
                    horizon, _FieldMetricAccumulator()
                )
                accumulator.update(
                    forecast_phy[local_index, :horizon],
                    y_phy[local_index, :horizon],
                )

    def compute(self) -> Dict[str, Any]:
        if not self._sources:
            raise RuntimeError("No sparse-surface metric samples were accumulated")
        per_source: Dict[str, Dict[str, Any]] = {}
        for source_id, source in self._sources.items():
            history = source.history.compute()
            missing_region = source.missing.compute()
            observed_region = source.observed.compute()
            history.update(
                {
                    "missing_rmse": missing_region["rmse"],
                    "observation_consistency_rmse": observed_region["rmse"],
                    "missing_count": float(source.missing.count),
                    "observed_count": float(source.observed.count),
                    "missing_region": missing_region,
                    "observed_region": observed_region,
                }
            )
            source_result: Dict[str, Any] = {
                "history_reconstruction": history,
                "error_growth_curve": source.growth.compute(self.frame_interval),
            }
            for horizon, accumulator in sorted(source.forecasts.items()):
                source_result[f"forecast_{horizon}"] = accumulator.compute()
            per_source[str(source_id)] = source_result

        history_rows = [
            value["history_reconstruction"] for value in per_source.values()
        ]
        history_metrics = _mean_metric_dict(
            {
                key: value
                for key, value in row.items()
                if key
                not in {"missing_count", "observed_count", "missing_region", "observed_region"}
            }
            for row in history_rows
        )
        history_metrics["missing_count"] = sum(
            row["missing_count"] for row in history_rows
        )
        history_metrics["observed_count"] = sum(
            row["observed_count"] for row in history_rows
        )
        history_metrics["missing_region"] = _mean_metric_dict(
            row["missing_region"] for row in history_rows
        )
        history_metrics["observed_region"] = _mean_metric_dict(
            row["observed_region"] for row in history_rows
        )
        growth_rows = [value["error_growth_curve"] for value in per_source.values()]
        error_growth_curve = []
        for frame_index in range(len(growth_rows[0])):
            frame_rows = [rows[frame_index] for rows in growth_rows]
            error_growth_curve.append(_mean_metric_dict(frame_rows))
        result: Dict[str, Any] = {
            "history_reconstruction": history_metrics,
            "error_growth_curve": error_growth_curve,
            "source_count": len(per_source),
            "per_source": per_source,
        }
        for horizon in self.forecast_horizons:
            key = f"forecast_{horizon}"
            rows = [value[key] for value in per_source.values() if key in value]
            if rows:
                if len(rows) != len(per_source):
                    raise RuntimeError(f"Not every source has metrics for horizon {horizon}")
                result[key] = _mean_metric_dict(rows)
        return result


class SparseHistoryMetricAccumulator:
    """Per-MAT physical reconstruction metrics used for P1 model selection."""

    def __init__(self, *, normalization_mean: float, normalization_std: float) -> None:
        denormalize_sea_surface(
            torch.zeros(()), normalization_mean, normalization_std
        )
        self.normalization_mean = float(normalization_mean)
        self.normalization_std = float(normalization_std)
        self._sources: Dict[Hashable, _HistorySourceAccumulator] = {}

    @torch.no_grad()
    def update(
        self,
        reconstruction: Tensor,
        x_full: Tensor,
        obs_mask: Tensor,
        *,
        source_ids: Sequence[Hashable] | Tensor,
    ) -> None:
        if reconstruction.shape != x_full.shape or obs_mask.shape != x_full.shape:
            raise ValueError("P1 metric tensors must share shape (B,T,H,W)")
        if reconstruction.ndim != 4:
            raise ValueError("P1 metric tensors must be (B,T,H,W)")
        if not torch.all((obs_mask == 0) | (obs_mask == 1)):
            raise ValueError("obs_mask must be binary with 1=observed and 0=missing")

        ids = _normalize_source_ids(source_ids, x_full.shape[0])
        reconstruction_phy = denormalize_sea_surface(
            reconstruction, self.normalization_mean, self.normalization_std
        ).detach().cpu()
        x_full_phy = denormalize_sea_surface(
            x_full, self.normalization_mean, self.normalization_std
        ).detach().cpu()
        masks = obs_mask.detach().cpu().bool()
        for local_index, source_id in enumerate(ids):
            source = self._sources.setdefault(
                source_id, _HistorySourceAccumulator()
            )
            prediction = reconstruction_phy[local_index]
            target = x_full_phy[local_index]
            mask = masks[local_index]
            source.full.update(prediction, target)
            if (~mask).any():
                source.missing.update(prediction[~mask], target[~mask])
            if mask.any():
                source.observed.update(prediction[mask], target[mask])

    def compute(self) -> Dict[str, Any]:
        if not self._sources:
            raise RuntimeError("No P1 reconstruction metrics were accumulated")
        per_source: Dict[str, Dict[str, Any]] = {}
        for source_id, source in self._sources.items():
            per_source[str(source_id)] = {
                "full": source.full.compute(),
                "missing_region": source.missing.compute(),
                "observed_region": source.observed.compute(),
                "missing_count": float(source.missing.count),
                "observed_count": float(source.observed.count),
            }
        return {
            "full": _mean_metric_dict(
                row["full"] for row in per_source.values()
            ),
            "missing_region": _mean_metric_dict(
                row["missing_region"] for row in per_source.values()
            ),
            "observed_region": _mean_metric_dict(
                row["observed_region"] for row in per_source.values()
            ),
            "missing_count": sum(
                row["missing_count"] for row in per_source.values()
            ),
            "observed_count": sum(
                row["observed_count"] for row in per_source.values()
            ),
            "source_count": len(per_source),
            "per_source": per_source,
        }


@torch.no_grad()
def compute_sparse_metrics(
    history_reconstruction: Tensor,
    x_full: Tensor,
    forecast: Tensor,
    y: Tensor,
    obs_mask: Tensor,
    *,
    source_ids: Sequence[Hashable] | Tensor,
    normalization_mean: float,
    normalization_std: float,
    forecast_horizons: Sequence[int] = (30, 60, 120, 180, 240, 300),
    frame_interval: float = 0.25,
) -> Dict[str, Any]:
    accumulator = SparseSurfaceMetricAccumulator(
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        forecast_horizons=forecast_horizons,
        frame_interval=frame_interval,
    )
    accumulator.update(
        history_reconstruction,
        x_full,
        forecast,
        y,
        obs_mask,
        source_ids=source_ids,
    )
    return accumulator.compute()


def compute_b1_metrics(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Backward-compatible B1 name using the shared B1/P1/P2 evaluation path."""

    return compute_sparse_metrics(*args, **kwargs)


__all__ = [
    "SparseHistoryMetricAccumulator",
    "SparseSurfaceMetricAccumulator",
    "compute_b1_metrics",
    "compute_sparse_metrics",
    "denormalize_sea_surface",
]
