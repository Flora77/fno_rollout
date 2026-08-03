"""Composable sparse-history reconstruction and frozen-RFNO forecasting."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from neuralop.models.fno_unet_aunet_gated_decoder import (
    FNOGlobalUNetGatedDecoder,
)
from neuralop.models.fno_deeponet import FNODeepONet
from neuralop.models.reconstructors import BilinearSpatialReconstructor


def _optional_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def _decoder_type(config: Mapping[str, Any]) -> str:
    architecture = str(config.get("model_arch", "fno_unet_gated_decoder")).lower()
    if architecture in {"fno_aunet_gated_decoder", "fno_attention_unet_gated_decoder"}:
        return "aunet"
    return str(config.get("fno_unet_refiner_type", "unet"))


def build_rfno_from_b0_config(config: Mapping[str, Any]) -> nn.Module:
    """Build the established RFNO architecture without changing its implementation."""

    architecture = str(config.get("model_arch", "")).lower()
    supported = {
        "fno_unet_gated_decoder",
        "fno_aunet_gated_decoder",
        "fno_attention_unet_gated_decoder",
        "fno_global_unet_gated_decoder",
    }
    if architecture not in supported:
        raise ValueError(
            f"B1 loader supports the established RFNO architectures {sorted(supported)}, "
            f"got {architecture!r}"
        )

    return FNOGlobalUNetGatedDecoder(
        n_modes=tuple(config.get("n_modes", (28, 28))),
        hidden_channels=int(config.get("hidden_channels", 32)),
        in_channels=int(config.get("input_steps", 60)),
        out_channels=int(config.get("output_steps", 30)),
        n_layers=int(config.get("n_layers", 4)),
        lifting_channels=int(config.get("lifting_channels", 64)),
        projection_channels=int(config.get("projection_channels", 64)),
        fno_arch=str(config.get("fno_unet_fno_arch", "fno")),
        decoder_type=_decoder_type(config),
        decoder_base_channels=int(config.get("fno_unet_base_channels", 32)),
        unet_depth=int(config.get("fno_unet_depth", 3)),
        unet_dropout=float(config.get("fno_unet_decoder_dropout", 0.0)),
        use_context=bool(config.get("fno_unet_use_context", True)),
        use_residual=bool(config.get("fno_unet_use_residual", True)),
        residual_scale=float(config.get("fno_unet_residual_scale", 1.0)),
        use_gated_residual=bool(config.get("fno_unet_use_gated_residual", True)),
        gate_hidden_channels=int(config.get("fno_unet_gate_hidden_channels", 32)),
        gate_bias_init=float(config.get("fno_unet_gate_bias_init", 0.0)),
        attention_inter_channels=_optional_positive_int(
            config.get("fno_aunet_attention_inter_channels")
        ),
        padding_mode=str(config.get("fno_unet_padding_mode", "periodic")),
    )


def load_rfno_from_b0_checkpoint(
    checkpoint_path: Union[str, Path],
    *,
    map_location: Union[str, torch.device] = "cpu",
) -> Tuple[nn.Module, Dict[str, Any], float, float, Path]:
    """Strictly load the B0 RFNO and its training-set normalization metadata."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"B0 RFNO checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("B0 checkpoint must contain a mapping")
    config = checkpoint.get("config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(config, Mapping) or not isinstance(state_dict, Mapping):
        raise KeyError("B0 checkpoint requires 'config' and 'model_state_dict'")

    normalization_keys = ("train_dataset_mean", "train_dataset_std")
    missing_normalization = [key for key in normalization_keys if key not in checkpoint]
    if missing_normalization:
        raise KeyError(
            "B0 checkpoint is missing training-set normalization metadata: "
            f"{missing_normalization}"
        )
    normalization_mean = float(checkpoint["train_dataset_mean"])
    normalization_std = float(checkpoint["train_dataset_std"])
    if not math.isfinite(normalization_mean):
        raise ValueError("B0 training-set normalization mean must be finite")
    if not math.isfinite(normalization_std) or normalization_std <= 0.0:
        raise ValueError("B0 training-set normalization std must be finite and positive")

    rfno = build_rfno_from_b0_config(config)
    if "_metadata" in state_dict:
        state_dict = {
            key: value for key, value in state_dict.items() if key != "_metadata"
        }
    rfno.load_state_dict(state_dict, strict=True)
    input_steps = int(config.get("input_steps", 60))
    output_steps = int(config.get("output_steps", 30))
    if input_steps != 60 or output_steps != 30:
        raise ValueError(
            "Sparse frozen pipelines require the established B0 60-to-30 RFNO, "
            f"got {input_steps}-to-{output_steps}"
        )
    return (
        rfno,
        dict(config),
        normalization_mean,
        normalization_std,
        checkpoint_path,
    )


def _autoregressive_rfno_rollout(
    rfno: nn.Module,
    history: Tensor,
    *,
    input_steps: int,
    output_steps: int,
    rollout_steps: int,
    detach_context: bool,
) -> Tensor:
    """Roll out RFNO predictions without reading any future ground-truth frame."""

    rollout_steps = int(rollout_steps)
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    context = history
    forecast_chunks = []
    generated = 0
    while generated < rollout_steps:
        prediction = rfno(context)
        if prediction.ndim != 4 or prediction.shape[1] != output_steps:
            raise RuntimeError(
                "RFNO must return "
                f"(B,{output_steps},H,W), got {tuple(prediction.shape)}"
            )
        take = min(output_steps, rollout_steps - generated)
        prediction = prediction[:, :take]
        forecast_chunks.append(prediction)
        generated += take
        context_prediction = prediction.detach() if detach_context else prediction
        context = torch.cat([context, context_prediction], dim=1)[:, -input_steps:]
    return torch.cat(forecast_chunks, dim=1)


class BilinearFrozenRFNOPipeline(nn.Module):
    """B1 pipeline: frame-wise interpolation followed by a frozen B0 RFNO."""

    def __init__(
        self,
        rfno: nn.Module,
        *,
        reconstructor: Optional[nn.Module] = None,
        input_steps: int = 60,
        output_steps: int = 30,
        normalization_mean: float = 0.0,
        normalization_std: float = 1.0,
        checkpoint_path: Optional[Union[str, Path]] = None,
        checkpoint_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.reconstructor = (
            BilinearSpatialReconstructor() if reconstructor is None else reconstructor
        )
        self.rfno = rfno
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.normalization_mean = float(normalization_mean)
        self.normalization_std = float(normalization_std)
        self.checkpoint_path = None if checkpoint_path is None else str(checkpoint_path)
        self.checkpoint_config = dict(checkpoint_config or {})
        if self.normalization_std <= 0.0:
            raise ValueError("normalization_std must be positive")
        self._freeze_rfno()

    @classmethod
    def from_b0_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        *,
        map_location: Union[str, torch.device] = "cpu",
        reconstructor: Optional[nn.Module] = None,
    ) -> "BilinearFrozenRFNOPipeline":
        rfno, config, mean, std, checkpoint_path = load_rfno_from_b0_checkpoint(
            checkpoint_path, map_location=map_location
        )
        input_steps = int(config.get("input_steps", 60))
        output_steps = int(config.get("output_steps", 30))

        return cls(
            rfno,
            reconstructor=reconstructor,
            input_steps=input_steps,
            output_steps=output_steps,
            normalization_mean=mean,
            normalization_std=std,
            checkpoint_path=checkpoint_path,
            checkpoint_config=config,
        )

    def _freeze_rfno(self) -> None:
        self.rfno.requires_grad_(False)
        self.rfno.eval()
        self.rfno.zero_grad(set_to_none=True)

    def train(self, mode: bool = True) -> "BilinearFrozenRFNOPipeline":
        super().train(mode)
        # The deterministic reconstructor has no train/eval distinction. RFNO
        # always stays in eval mode to reproduce B0 inference behavior.
        self.rfno.eval()
        return self

    def optimizer_parameters(self) -> Iterable[nn.Parameter]:
        """Return trainable non-RFNO parameters only (empty for deterministic B1)."""

        return tuple(
            parameter
            for parameter in self.reconstructor.parameters()
            if parameter.requires_grad
        )

    def forward(self, x_obs: Tensor, obs_mask: Tensor) -> Dict[str, Tensor]:
        return self.rollout(x_obs, obs_mask, rollout_steps=self.output_steps)

    def rollout(
        self, x_obs: Tensor, obs_mask: Tensor, *, rollout_steps: int
    ) -> Dict[str, Tensor]:
        if x_obs.ndim != 4 or obs_mask.ndim != 4:
            raise ValueError("B1 pipeline expects x_obs and obs_mask as (B,T,H,W)")
        if x_obs.shape != obs_mask.shape:
            raise ValueError("x_obs and obs_mask must have identical shapes")
        if x_obs.shape[1] != self.input_steps:
            raise ValueError(
                f"Expected {self.input_steps} history frames, got {x_obs.shape[1]}"
            )

        history = self.reconstructor(x_obs, obs_mask)
        forecast = _autoregressive_rfno_rollout(
            self.rfno,
            history,
            input_steps=self.input_steps,
            output_steps=self.output_steps,
            rollout_steps=rollout_steps,
            detach_context=True,
        )
        expected_forecast_shape = (
            x_obs.shape[0],
            int(rollout_steps),
            x_obs.shape[2],
            x_obs.shape[3],
        )
        if tuple(forecast.shape) != expected_forecast_shape:
            raise RuntimeError(
                f"RFNO forecast shape {tuple(forecast.shape)} differs from "
                f"expected {expected_forecast_shape}"
            )
        if not torch.isfinite(history).all() or not torch.isfinite(forecast).all():
            raise FloatingPointError("B1 history reconstruction or forecast has NaN/Inf")

        return {
            "history_reconstruction": history,
            "forecast": forecast,
        }

    def rfno_gradients_are_none(self) -> bool:
        return all(parameter.grad is None for parameter in self.rfno.parameters())


class LearnedReconstructionRFNOPipeline(nn.Module):
    """Shared learned-reconstruction pipeline for B4/P1/P2."""

    def __init__(
        self,
        rfno: nn.Module,
        reconstructor: nn.Module,
        *,
        input_steps: int = 60,
        output_steps: int = 30,
        normalization_mean: float = 0.0,
        normalization_std: float = 1.0,
        coupling: str = "frozen",
        checkpoint_path: Optional[Union[str, Path]] = None,
        checkpoint_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.reconstructor = reconstructor
        self.rfno = rfno
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.normalization_mean = float(normalization_mean)
        self.normalization_std = float(normalization_std)
        self.checkpoint_path = None if checkpoint_path is None else str(checkpoint_path)
        self.checkpoint_config = dict(checkpoint_config or {})
        if self.normalization_std <= 0.0:
            raise ValueError("normalization_std must be positive")
        self.coupling = ""
        self.configure_coupling(coupling)

    @classmethod
    def from_b0_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        reconstructor: nn.Module,
        *,
        map_location: Union[str, torch.device] = "cpu",
        coupling: str = "frozen",
    ) -> "LearnedReconstructionRFNOPipeline":
        rfno, config, mean, std, checkpoint_path = load_rfno_from_b0_checkpoint(
            checkpoint_path, map_location=map_location
        )
        return cls(
            rfno,
            reconstructor,
            input_steps=int(config["input_steps"]),
            output_steps=int(config["output_steps"]),
            normalization_mean=mean,
            normalization_std=std,
            coupling=coupling,
            checkpoint_path=checkpoint_path,
            checkpoint_config=config,
        )

    def configure_coupling(self, coupling: str) -> None:
        coupling = str(coupling).lower()
        if coupling not in {"frozen", "joint"}:
            raise ValueError("Learned reconstruction coupling must be 'frozen' or 'joint'")
        self.coupling = coupling
        self.rfno.zero_grad(set_to_none=True)
        if coupling == "frozen":
            self.rfno.requires_grad_(False)
            self.rfno.eval()
        else:
            self.rfno.requires_grad_(True)
            self.rfno.train(self.training)

    def train(self, mode: bool = True) -> "LearnedReconstructionRFNOPipeline":
        super().train(mode)
        if self.coupling == "frozen":
            self.rfno.eval()
        return self

    def optimizer_parameters(self) -> Iterable[nn.Parameter]:
        parameters = list(
            parameter
            for parameter in self.reconstructor.parameters()
            if parameter.requires_grad
        )
        if self.coupling == "joint":
            parameters.extend(
                parameter for parameter in self.rfno.parameters() if parameter.requires_grad
            )
        return tuple(parameters)

    def optimizer_param_groups(
        self,
        *,
        reconstructor_lr: float,
        rfno_lr: Optional[float] = None,
        weight_decay: float = 0.0,
    ) -> list:
        if reconstructor_lr <= 0.0:
            raise ValueError("reconstructor_lr must be positive")
        if rfno_lr is None:
            rfno_lr = 0.1 * float(reconstructor_lr)
        if rfno_lr <= 0.0:
            raise ValueError("rfno_lr must be positive")
        groups = [
            {
                "name": "reconstructor",
                "params": [
                    parameter
                    for parameter in self.reconstructor.parameters()
                    if parameter.requires_grad
                ],
                "lr": float(reconstructor_lr),
                "weight_decay": float(weight_decay),
            }
        ]
        if self.coupling == "joint":
            groups.append(
                {
                    "name": "rfno",
                    "params": [
                        parameter
                        for parameter in self.rfno.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": float(rfno_lr),
                    "weight_decay": float(weight_decay),
                }
            )
        return groups

    def forward(self, x_obs: Tensor, obs_mask: Tensor) -> Dict[str, Tensor]:
        return self.rollout(
            x_obs,
            obs_mask,
            rollout_steps=self.output_steps,
            detach_context=False,
        )

    def rollout(
        self,
        x_obs: Tensor,
        obs_mask: Tensor,
        *,
        rollout_steps: int,
        detach_context: bool = False,
    ) -> Dict[str, Tensor]:
        if x_obs.shape != obs_mask.shape or x_obs.ndim != 4:
            raise ValueError(
                "Learned reconstruction expects x_obs and obs_mask with identical "
                "(B,T,H,W)"
            )
        if x_obs.shape[1] != self.input_steps:
            raise ValueError(
                f"Expected {self.input_steps} history frames, got {x_obs.shape[1]}"
            )
        forward_with_aux = getattr(self.reconstructor, "forward_with_aux", None)
        if callable(forward_with_aux):
            reconstruction_result = forward_with_aux(x_obs, obs_mask)
            if isinstance(reconstruction_result, Mapping):
                history = reconstruction_result["reconstruction"]
                raw_history = reconstruction_result.get(
                    "raw_reconstruction", history
                )
                loss_history = reconstruction_result.get(
                    "loss_reconstruction", history
                )
            elif (
                isinstance(reconstruction_result, tuple)
                and len(reconstruction_result) == 2
                and isinstance(reconstruction_result[0], Tensor)
            ):
                history = reconstruction_result[0]
                raw_history = history
                loss_history = history
            else:
                raise TypeError(
                    "Reconstructor forward_with_aux must return a reconstruction "
                    "mapping or (Tensor, diagnostics)"
                )
        else:
            history = self.reconstructor(x_obs, obs_mask)
            raw_history = history
            loss_history = history
        rollout_steps = int(rollout_steps)
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        forecast = _autoregressive_rfno_rollout(
            self.rfno,
            history,
            input_steps=self.input_steps,
            output_steps=self.output_steps,
            rollout_steps=rollout_steps,
            detach_context=detach_context,
        )
        expected_history = tuple(x_obs.shape)
        expected_forecast = (
            x_obs.shape[0],
            rollout_steps,
            x_obs.shape[2],
            x_obs.shape[3],
        )
        if tuple(history.shape) != expected_history:
            raise RuntimeError(
                "Learned reconstruction shape "
                f"{tuple(history.shape)} != {expected_history}"
            )
        if tuple(forecast.shape) != expected_forecast:
            raise RuntimeError(
                f"Learned forecast shape {tuple(forecast.shape)} != {expected_forecast}"
            )
        if (
            not torch.isfinite(history).all()
            or not torch.isfinite(raw_history).all()
            or not torch.isfinite(loss_history).all()
            or not torch.isfinite(forecast).all()
        ):
            raise FloatingPointError(
                "Learned reconstruction or forecast contains NaN/Inf"
            )
        return {
            "history_reconstruction": history,
            "history_raw_reconstruction": raw_history,
            "history_loss_reconstruction": loss_history,
            "forecast": forecast,
        }

    def rfno_gradients_are_none(self) -> bool:
        return all(parameter.grad is None for parameter in self.rfno.parameters())


class FNODeepONetDirectSparsePipeline(nn.Module):
    """B7-style sparse history-to-history-and-future one-shot operator."""

    def __init__(
        self,
        operator: FNODeepONet,
        *,
        forecast_steps: int = 300,
        hard_observation_consistency: bool = False,
        normalization_mean: float = 0.0,
        normalization_std: float = 1.0,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        self.direct_operator = operator
        self.input_steps = int(operator.input_steps)
        self.output_steps = 30
        self.forecast_steps = int(forecast_steps)
        self.hard_observation_consistency = bool(hard_observation_consistency)
        self.normalization_mean = float(normalization_mean)
        self.normalization_std = float(normalization_std)
        self.checkpoint_path = None if checkpoint_path is None else str(checkpoint_path)
        self.coupling = "direct"
        if self.forecast_steps <= 0:
            raise ValueError("forecast_steps must be positive")
        if self.normalization_std <= 0.0:
            raise ValueError("normalization_std must be positive")

    @property
    def reconstructor(self) -> nn.Module:
        """Compatibility view used by shared checkpoint and gradient reporting."""

        return self.direct_operator

    def configure_coupling(self, coupling: str) -> None:
        if str(coupling).lower() != "direct":
            raise ValueError("Direct FNO-DeepONet pipeline requires coupling='direct'")
        self.coupling = "direct"
        self.direct_operator.requires_grad_(True)

    def optimizer_parameters(self) -> Iterable[nn.Parameter]:
        return tuple(
            parameter
            for parameter in self.direct_operator.parameters()
            if parameter.requires_grad
        )

    def optimizer_param_groups(
        self,
        *,
        reconstructor_lr: float,
        rfno_lr: Optional[float] = None,
        weight_decay: float = 0.0,
    ) -> list:
        del rfno_lr
        if reconstructor_lr <= 0.0:
            raise ValueError("direct operator learning rate must be positive")
        return [
            {
                "name": "direct_operator",
                "params": list(self.optimizer_parameters()),
                "lr": float(reconstructor_lr),
                "weight_decay": float(weight_decay),
            }
        ]

    def forward(self, x_obs: Tensor, obs_mask: Tensor) -> Dict[str, Tensor]:
        return self.rollout(x_obs, obs_mask, rollout_steps=self.forecast_steps)

    def rollout(
        self,
        x_obs: Tensor,
        obs_mask: Tensor,
        *,
        rollout_steps: int,
        detach_context: bool = False,
    ) -> Dict[str, Tensor]:
        del detach_context
        rollout_steps = int(rollout_steps)
        if rollout_steps <= 0 or rollout_steps > self.forecast_steps:
            raise ValueError(
                f"rollout_steps must be in [1,{self.forecast_steps}] for direct decoding"
            )
        branch_code = self.direct_operator.encode_sparse_history(x_obs, obs_mask)
        history_times = torch.linspace(
            -1.0,
            0.0,
            self.input_steps,
            device=x_obs.device,
            dtype=x_obs.dtype,
        )
        future_times = torch.arange(
            1,
            rollout_steps + 1,
            device=x_obs.device,
            dtype=x_obs.dtype,
        ) / float(self.forecast_steps)
        raw_history = self.direct_operator.decode_grid(branch_code, history_times)
        forecast = self.direct_operator.decode_grid(branch_code, future_times)
        mask = obs_mask.to(dtype=x_obs.dtype)
        history = (
            raw_history * (1.0 - mask) + x_obs * mask
            if self.hard_observation_consistency
            else raw_history
        )
        if history.shape != x_obs.shape:
            raise RuntimeError("Direct FNO-DeepONet reconstruction shape mismatch")
        expected_forecast = (
            x_obs.shape[0],
            rollout_steps,
            x_obs.shape[-2],
            x_obs.shape[-1],
        )
        if tuple(forecast.shape) != expected_forecast:
            raise RuntimeError(
                "Direct FNO-DeepONet forecast shape "
                f"{tuple(forecast.shape)} != {expected_forecast}"
            )
        return {
            "history_reconstruction": history,
            "history_raw_reconstruction": raw_history,
            "history_loss_reconstruction": raw_history,
            "forecast": forecast,
        }


# Backward-compatible method-specific names. The implementation stays shared so B4 and
# P1 use exactly the same frozen-RFNO behavior and optimizer filtering.
PartialConvMAERFNOPipeline = LearnedReconstructionRFNOPipeline
PartialConvMAEFrozenRFNOPipeline = LearnedReconstructionRFNOPipeline
MaskUNetFrozenRFNOPipeline = LearnedReconstructionRFNOPipeline


__all__ = [
    "BilinearFrozenRFNOPipeline",
    "FNODeepONetDirectSparsePipeline",
    "LearnedReconstructionRFNOPipeline",
    "MaskUNetFrozenRFNOPipeline",
    "PartialConvMAERFNOPipeline",
    "PartialConvMAEFrozenRFNOPipeline",
    "build_rfno_from_b0_config",
    "load_rfno_from_b0_checkpoint",
]
