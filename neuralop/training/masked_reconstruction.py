"""Loss decomposition and batch interface for masked-history pretraining."""

from __future__ import annotations

import base64
from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor, nn

from neuralop.models.reconstructors.vit_mae import random_token_observation_mask


def _expanded_mask(mask: Tensor, reference: Tensor) -> Tensor:
    if mask.ndim != 4 or reference.ndim != 4:
        raise ValueError("mask and reference must be (B,C,H,W)")
    if mask.shape == reference.shape:
        return mask
    if (
        mask.shape[0] == reference.shape[0]
        and mask.shape[1] == 1
        and mask.shape[-2:] == reference.shape[-2:]
    ):
        return mask.expand_as(reference)
    raise ValueError(f"Cannot broadcast mask {mask.shape} to {reference.shape}")


def _masked_mse(prediction: Tensor, target: Tensor, selection: Tensor) -> Tensor:
    selection = selection.to(device=prediction.device, dtype=prediction.dtype)
    squared_error = (prediction - target).square() * selection
    return squared_error.sum() / selection.sum().clamp_min(1.0)


def periodic_spatial_gradient_mse(prediction: Tensor, target: Tensor) -> Tensor:
    pred_dx = torch.roll(prediction, shifts=-1, dims=-1) - prediction
    target_dx = torch.roll(target, shifts=-1, dims=-1) - target
    pred_dy = torch.roll(prediction, shifts=-1, dims=-2) - prediction
    target_dy = torch.roll(target, shifts=-1, dims=-2) - target
    return (pred_dx - target_dx).square().mean() + (
        pred_dy - target_dy
    ).square().mean()


class MaskedReconstructionLoss(nn.Module):
    """Named P1 losses for hidden reconstruction and observed consistency."""

    def __init__(
        self,
        *,
        hidden_weight: float = 1.0,
        observation_weight: float = 0.1,
        history_gradient_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_weight = float(hidden_weight)
        self.observation_weight = float(observation_weight)
        self.history_gradient_weight = float(history_gradient_weight)
        if min(
            self.hidden_weight,
            self.observation_weight,
            self.history_gradient_weight,
        ) < 0.0:
            raise ValueError("P1 loss weights must be non-negative")

    def forward(
        self, reconstruction: Tensor, x_full: Tensor, obs_mask: Tensor
    ) -> Dict[str, Tensor]:
        if reconstruction.shape != x_full.shape:
            raise ValueError("reconstruction and x_full must have identical shapes")
        observed = _expanded_mask(obs_mask, x_full).to(dtype=x_full.dtype)
        if not torch.all((observed == 0) | (observed == 1)):
            raise ValueError("obs_mask must be binary with 1=observed")
        hidden = 1.0 - observed

        hidden_loss = _masked_mse(reconstruction, x_full, hidden)
        observation_loss = _masked_mse(reconstruction, x_full, observed)
        if self.history_gradient_weight > 0.0:
            gradient_loss = periodic_spatial_gradient_mse(reconstruction, x_full)
        else:
            gradient_loss = reconstruction.sum() * 0.0

        weighted_hidden = self.hidden_weight * hidden_loss
        weighted_observation = self.observation_weight * observation_loss
        weighted_gradient = self.history_gradient_weight * gradient_loss
        total = weighted_hidden + weighted_observation + weighted_gradient
        losses = {
            "hidden_reconstruction": hidden_loss,
            "observation_consistency": observation_loss,
            "history_gradient": gradient_loss,
            "weighted_hidden_reconstruction": weighted_hidden,
            "weighted_observation_consistency": weighted_observation,
            "weighted_history_gradient": weighted_gradient,
            "total": total,
        }
        if not all(torch.isfinite(loss).all() for loss in losses.values()):
            raise FloatingPointError("Masked reconstruction loss contains NaN/Inf")
        return losses


class MaskedReconstructionPretrainer(nn.Module):
    """One-batch train/validation interface that never reads future targets."""

    def __init__(
        self,
        reconstructor: nn.Module,
        criterion: MaskedReconstructionLoss,
        *,
        mae_mask_ratio: float = 0.0,
        mae_patch_size: int = 4,
        mae_tokenization: str = "spatial_patch",
        mae_tubelet_size: int = 10,
        training_mask_seed: int = 42,
        evaluation_mask_seed: int = 1729,
    ):
        super().__init__()
        self.reconstructor = reconstructor
        self.criterion = criterion
        self.mae_mask_ratio = float(mae_mask_ratio)
        self.mae_patch_size = int(mae_patch_size)
        self.mae_tokenization = str(mae_tokenization)
        self.mae_tubelet_size = int(mae_tubelet_size)
        self.training_mask_seed = int(training_mask_seed)
        self.evaluation_mask_seed = int(evaluation_mask_seed)
        self._randomize_mask = True
        self._training_generator: Optional[torch.Generator] = None
        self._pending_training_generator_state: Optional[Tensor] = None
        self._evaluation_generator: Optional[torch.Generator] = None
        if not 0.0 <= self.mae_mask_ratio < 1.0:
            raise ValueError("mae_mask_ratio must be in [0,1)")
        if self.mae_mask_ratio and self.mae_tokenization not in {
            "spatial_patch",
            "tubelet",
        }:
            raise ValueError("Unsupported MAE tokenization")

    def set_mask_mode(self, *, training: bool) -> None:
        """Use random train masks and a reset, deterministic validation sequence."""

        self._randomize_mask = bool(training)
        self._evaluation_generator = None

    def _mask_generator(self, device: torch.device) -> Optional[torch.Generator]:
        if self._randomize_mask:
            if self._training_generator is None:
                self._training_generator = torch.Generator(device=device)
                self._training_generator.manual_seed(self.training_mask_seed)
                if self._pending_training_generator_state is not None:
                    self._training_generator.set_state(
                        self._pending_training_generator_state
                    )
                    self._pending_training_generator_state = None
            return self._training_generator
        if self._evaluation_generator is None:
            self._evaluation_generator = torch.Generator(device=device)
            self._evaluation_generator.manual_seed(self.evaluation_mask_seed)
        return self._evaluation_generator

    def training_mask_state(self) -> Optional[str]:
        """Return a JSON-safe state for exact mask-sequence resume."""

        if self._training_generator is None:
            return None
        raw_state = bytes(self._training_generator.get_state().cpu().tolist())
        return base64.b64encode(raw_state).decode("ascii")

    def load_training_mask_state(self, encoded_state: str) -> None:
        """Restore the dedicated training-mask generator on its next use."""

        raw_state = base64.b64decode(str(encoded_state).encode("ascii"))
        self._training_generator = None
        self._pending_training_generator_state = torch.tensor(
            list(raw_state), dtype=torch.uint8
        )

    def _pretraining_inputs(
        self,
        x_full: Tensor,
        x_obs: Tensor,
        obs_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.mae_mask_ratio == 0.0:
            return x_obs, obs_mask
        # Complete history is passed through a stochastic observation operator.
        # The model receives only the resulting visible values and explicit mask.
        synthetic_mask = random_token_observation_mask(
            x_full,
            patch_size=self.mae_patch_size,
            mask_ratio=self.mae_mask_ratio,
            tokenization=self.mae_tokenization,  # type: ignore[arg-type]
            tubelet_size=self.mae_tubelet_size,
            generator=self._mask_generator(x_full.device),
        )
        return x_full * synthetic_mask, synthetic_mask

    def forward_batch(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        required = ("x_obs", "obs_mask", "x_full")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Masked pretraining batch is missing fields: {missing}")
        x_obs = batch["x_obs"]
        obs_mask = batch["obs_mask"]
        x_full = batch["x_full"]
        if not all(isinstance(value, Tensor) for value in (x_obs, obs_mask, x_full)):
            raise TypeError("x_obs, obs_mask, and x_full must be torch.Tensor values")

        model_x_obs, supervision_mask = self._pretraining_inputs(
            x_full, x_obs, obs_mask
        )
        forward_with_aux = getattr(self.reconstructor, "forward_with_aux", None)
        if callable(forward_with_aux):
            reconstruction_result = forward_with_aux(
                model_x_obs, supervision_mask
            )
            if isinstance(reconstruction_result, Mapping):
                reconstruction = reconstruction_result["reconstruction"]
                loss_reconstruction = reconstruction_result.get(
                    "loss_reconstruction", reconstruction
                )
                raw_reconstruction = reconstruction_result.get(
                    "raw_reconstruction", reconstruction
                )
            elif (
                isinstance(reconstruction_result, tuple)
                and len(reconstruction_result) == 2
                and isinstance(reconstruction_result[0], Tensor)
            ):
                # ViT and sensor-token reconstructors expose
                # ``(reconstruction, diagnostics)`` through the same method name.
                reconstruction = reconstruction_result[0]
                loss_reconstruction = reconstruction
                raw_reconstruction = reconstruction
            else:
                raise TypeError(
                    "forward_with_aux must return a reconstruction mapping or "
                    "(Tensor, diagnostics)"
                )
        else:
            reconstruction = self.reconstructor(model_x_obs, supervision_mask)
            loss_reconstruction = reconstruction
            raw_reconstruction = reconstruction
        losses = self.criterion(
            loss_reconstruction, x_full, supervision_mask
        )
        result = {
            "reconstruction": reconstruction,
            "raw_reconstruction": raw_reconstruction,
            "loss_reconstruction": loss_reconstruction,
            "losses": losses,
            "supervision_mask": supervision_mask,
        }
        return result

    def train_batch(
        self,
        batch: Mapping[str, Any],
        optimizer: torch.optim.Optimizer,
        *,
        max_grad_norm: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.train()
        self.set_mask_mode(training=True)
        optimizer.zero_grad(set_to_none=True)
        result = self.forward_batch(batch)
        result["losses"]["total"].backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.reconstructor.parameters(), max_grad_norm)
        optimizer.step()
        return result

    @torch.no_grad()
    def validate_batch(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        self.eval()
        self.set_mask_mode(training=False)
        return self.forward_batch(batch)


__all__ = [
    "MaskedReconstructionLoss",
    "MaskedReconstructionPretrainer",
    "periodic_spatial_gradient_mse",
]
