"""Unified P1/P2 trainer for frozen or joint PartialConv-MAE-RFNO coupling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from neuralop.training.masked_reconstruction import MaskedReconstructionLoss


@dataclass
class SparseCoupledTrainingConfig:
    coupling: str = "frozen"
    reconstructor_lr: float = 2.0e-3
    rfno_lr: float = 2.0e-4
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0

    input_steps: int = 60
    one_shot_steps: int = 30
    max_rollout_steps: int = 300
    use_rollout_curriculum: bool = True
    rollout_train_steps: Tuple[int, ...] = (30, 60, 120, 180, 240, 300)
    rollout_curriculum_boundaries: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6)
    rollout_detach_context: bool = False
    forecast_supervision: Optional[bool] = None

    use_segment_weighting: bool = True
    segment_weight_type: str = "linear"
    segment_weight_min: float = 1.0
    segment_weight_max: float = 2.5
    segment_weight_power: float = 2.0

    hidden_weight: float = 1.0
    observation_weight: float = 0.1
    history_gradient_weight: float = 0.05
    rollout_weight: float = 1.0
    forecast_gradient_weight: float = 0.05
    spectrum_weight: float = 0.01

    def __post_init__(self) -> None:
        self.coupling = str(self.coupling).lower()
        if self.coupling not in {"frozen", "joint", "direct"}:
            raise ValueError("coupling must be 'frozen', 'joint', or 'direct'")
        if self.forecast_supervision is None:
            # Preserve P1 as reconstruction-only unless forecast-aware training is
            # explicitly requested. P2 joint remains forecast-supervised by default.
            self.forecast_supervision = self.coupling in {"joint", "direct"}
        else:
            self.forecast_supervision = bool(self.forecast_supervision)
        if self.input_steps != 60 or self.one_shot_steps != 30:
            raise ValueError("P1/P2 uses the established 60-to-30 RFNO contract")
        if self.max_rollout_steps < self.one_shot_steps:
            raise ValueError("max_rollout_steps must be at least 30")
        if min(self.reconstructor_lr, self.rfno_lr) <= 0.0:
            raise ValueError("reconstructor_lr and rfno_lr must be positive")
        weights = (
            self.hidden_weight,
            self.observation_weight,
            self.history_gradient_weight,
            self.rollout_weight,
            self.forecast_gradient_weight,
            self.spectrum_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("loss weights must be non-negative")

    def curriculum_steps(self) -> Tuple[int, ...]:
        if not self.use_rollout_curriculum:
            return (int(self.max_rollout_steps),)
        steps = tuple(
            sorted(
                {
                    int(step)
                    for step in self.rollout_train_steps
                    if 0 < int(step) <= self.max_rollout_steps
                }
            )
        )
        if not steps or steps[0] != self.one_shot_steps:
            raise ValueError("rollout curriculum must start with 30 frames")
        return steps

    def active_rollout_steps(self, epoch: int, total_epochs: int) -> int:
        steps = self.curriculum_steps()
        if len(steps) == 1:
            return steps[0]
        boundaries = list(float(value) for value in self.rollout_curriculum_boundaries)
        if len(boundaries) != len(steps):
            boundaries = [index / len(steps) for index in range(len(steps))]
        boundaries = sorted(boundaries)
        boundaries[0] = 0.0
        progress = (
            0.0
            if total_epochs <= 1
            else float(epoch - 1) / float(max(total_epochs - 1, 1))
        )
        index = 0
        for candidate, boundary in enumerate(boundaries):
            if progress >= boundary:
                index = candidate
        return steps[min(index, len(steps) - 1)]


def periodic_forecast_gradient_mse(prediction: Tensor, target: Tensor) -> Tensor:
    """Periodic spatial-gradient MSE for the HOS domain."""

    pred_dx = torch.roll(prediction, shifts=-1, dims=-1) - prediction
    target_dx = torch.roll(target, shifts=-1, dims=-1) - target
    pred_dy = torch.roll(prediction, shifts=-1, dims=-2) - prediction
    target_dy = torch.roll(target, shifts=-1, dims=-2) - target
    return (pred_dx - target_dx).square().mean() + (
        pred_dy - target_dy
    ).square().mean()


def spectrum_shape_loss(prediction: Tensor, target: Tensor) -> Tensor:
    prediction_fft = torch.fft.rfft2(prediction.float(), dim=(-2, -1), norm="ortho")
    target_fft = torch.fft.rfft2(target.float(), dim=(-2, -1), norm="ortho")
    prediction_power = prediction_fft.real.square() + prediction_fft.imag.square()
    target_power = target_fft.real.square() + target_fft.imag.square()
    power_error = (prediction_power - target_power).square().mean()
    reference_power = target_power.square().mean().clamp_min(1.0e-8)
    return power_error / reference_power


class SparseCoupledForecastLoss(nn.Module):
    """Named reconstruction, consistency, rollout, gradient, and spectrum losses."""

    def __init__(self, config: SparseCoupledTrainingConfig):
        super().__init__()
        self.config = config
        self.reconstruction_loss = MaskedReconstructionLoss(
            hidden_weight=1.0,
            observation_weight=1.0,
            history_gradient_weight=1.0,
        )

    def _segment_weights(self, segments: int, device: torch.device) -> Tensor:
        config = self.config
        if not config.use_segment_weighting or config.segment_weight_type == "none":
            weights = torch.ones(segments, device=device)
        elif config.segment_weight_type == "linear":
            weights = torch.linspace(
                config.segment_weight_min,
                config.segment_weight_max,
                segments,
                device=device,
            )
        elif config.segment_weight_type == "power":
            position = torch.linspace(0.0, 1.0, segments, device=device)
            weights = config.segment_weight_min + (
                config.segment_weight_max - config.segment_weight_min
            ) * position.pow(config.segment_weight_power)
        elif config.segment_weight_type == "exp":
            minimum = max(config.segment_weight_min, 1.0e-8)
            maximum = max(config.segment_weight_max, 1.0e-8)
            growth = math.log(maximum / minimum) / max(segments - 1, 1)
            weights = minimum * torch.exp(
                growth * torch.arange(segments, device=device)
            )
        else:
            raise ValueError(
                f"Unsupported segment_weight_type: {config.segment_weight_type}"
            )
        return weights / weights.mean().clamp_min(1.0e-8)

    def forward(
        self,
        history_reconstruction: Tensor,
        x_full: Tensor,
        forecast: Tensor,
        y: Optional[Tensor],
        obs_mask: Tensor,
    ) -> Dict[str, Tensor]:
        reconstruction_terms = self.reconstruction_loss(
            history_reconstruction, x_full, obs_mask
        )
        if self.config.forecast_supervision:
            if y is None or forecast.ndim != 4 or y.ndim != 4:
                raise ValueError("forecast-supervised training requires a 4D future y")
            if y.shape[1] < forecast.shape[1]:
                raise ValueError("y must contain at least as many future frames as forecast")
            target = y[:, : forecast.shape[1]]
            if forecast.shape != target.shape:
                raise ValueError("forecast and selected rollout target shapes differ")
            chunks = forecast.split(self.config.one_shot_steps, dim=1)
            target_chunks = target.split(self.config.one_shot_steps, dim=1)
            segment_weights = self._segment_weights(len(chunks), forecast.device)
            rollout_loss = sum(
                segment_weights[index] * (prediction - truth).square().mean()
                for index, (prediction, truth) in enumerate(zip(chunks, target_chunks))
            ) / segment_weights.sum().clamp_min(1.0e-8)
            forecast_gradient_loss = periodic_forecast_gradient_mse(forecast, target)
            spectrum_loss = spectrum_shape_loss(forecast, target)
        else:
            # Keep future targets completely outside the P1 reconstruction-only
            # objective while retaining finite named logging fields.
            zero = history_reconstruction.sum() * 0.0
            rollout_loss = zero
            forecast_gradient_loss = zero
            spectrum_loss = zero

        hidden_loss = reconstruction_terms["hidden_reconstruction"]
        observation_loss = reconstruction_terms["observation_consistency"]
        history_gradient_loss = reconstruction_terms["history_gradient"]
        weighted_hidden = self.config.hidden_weight * hidden_loss
        weighted_observation = self.config.observation_weight * observation_loss
        weighted_history_gradient = (
            self.config.history_gradient_weight * history_gradient_loss
        )
        weighted_rollout = self.config.rollout_weight * rollout_loss
        weighted_forecast_gradient = (
            self.config.forecast_gradient_weight * forecast_gradient_loss
        )
        weighted_spectrum = self.config.spectrum_weight * spectrum_loss
        total = (
            weighted_hidden
            + weighted_observation
            + weighted_history_gradient
            + weighted_rollout
            + weighted_forecast_gradient
            + weighted_spectrum
        )
        losses = {
            "hidden_reconstruction": hidden_loss,
            "observation_consistency": observation_loss,
            "history_gradient": history_gradient_loss,
            "rollout": rollout_loss,
            "forecast_gradient": forecast_gradient_loss,
            "spectrum": spectrum_loss,
            "weighted_hidden_reconstruction": weighted_hidden,
            "weighted_observation_consistency": weighted_observation,
            "weighted_history_gradient": weighted_history_gradient,
            "weighted_rollout": weighted_rollout,
            "weighted_forecast_gradient": weighted_forecast_gradient,
            "weighted_spectrum": weighted_spectrum,
            "total": total,
        }
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise FloatingPointError("Sparse coupled loss contains NaN/Inf")
        return losses


class SparseCoupledForecastTrainer:
    """Single trainer shared by P1 frozen and P2 joint coupling."""

    def __init__(
        self,
        pipeline: nn.Module,
        config: SparseCoupledTrainingConfig,
    ) -> None:
        self.pipeline = pipeline
        self.config = config
        self.pipeline.configure_coupling(config.coupling)
        self.criterion = SparseCoupledForecastLoss(config)

    def build_optimizer(self) -> torch.optim.Optimizer:
        groups = self.pipeline.optimizer_param_groups(
            reconstructor_lr=self.config.reconstructor_lr,
            rfno_lr=self.config.rfno_lr,
            weight_decay=self.config.weight_decay,
        )
        return torch.optim.AdamW(groups)

    def forward_batch(
        self, batch: Mapping[str, Any], *, active_rollout_steps: int
    ) -> Dict[str, Any]:
        required = ("x_obs", "obs_mask", "x_full")
        if self.config.forecast_supervision:
            required = (*required, "y")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Sparse coupled batch is missing fields: {missing}")
        tensors = [batch[key] for key in required]
        if not all(isinstance(value, Tensor) for value in tensors):
            raise TypeError("Sparse coupled batch fields must be torch.Tensor values")

        # Future target y is consumed only below by the loss; pipeline inputs are
        # explicitly limited to sparse history values and their validity mask.
        outputs = self.pipeline.rollout(
            batch["x_obs"],
            batch["obs_mask"],
            rollout_steps=active_rollout_steps,
            detach_context=self.config.rollout_detach_context,
        )
        losses = self.criterion(
            outputs.get(
                "history_loss_reconstruction",
                outputs["history_reconstruction"],
            ),
            batch["x_full"],
            outputs["forecast"],
            batch.get("y"),
            batch["obs_mask"],
        )
        return {
            **outputs,
            "losses": losses,
            "active_rollout_steps": int(active_rollout_steps),
        }

    def train_batch(
        self,
        batch: Mapping[str, Any],
        optimizer: torch.optim.Optimizer,
        *,
        active_rollout_steps: int = 30,
    ) -> Dict[str, Any]:
        self.pipeline.train(True)
        optimizer.zero_grad(set_to_none=True)
        result = self.forward_batch(batch, active_rollout_steps=active_rollout_steps)
        result["losses"]["total"].backward()
        if self.config.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                tuple(self.pipeline.optimizer_parameters()), self.config.grad_clip_norm
            )
        optimizer.step()
        return result

    def train_batch_for_epoch(
        self,
        batch: Mapping[str, Any],
        optimizer: torch.optim.Optimizer,
        *,
        epoch: int,
        total_epochs: int,
    ) -> Dict[str, Any]:
        active_steps = self.config.active_rollout_steps(epoch, total_epochs)
        return self.train_batch(
            batch, optimizer, active_rollout_steps=active_steps
        )

    @torch.no_grad()
    def validate_batch(
        self, batch: Mapping[str, Any], *, active_rollout_steps: int = 30
    ) -> Dict[str, Any]:
        self.pipeline.eval()
        return self.forward_batch(batch, active_rollout_steps=active_rollout_steps)


__all__ = [
    "SparseCoupledForecastLoss",
    "SparseCoupledForecastTrainer",
    "SparseCoupledTrainingConfig",
    "periodic_forecast_gradient_mse",
    "spectrum_shape_loss",
]
