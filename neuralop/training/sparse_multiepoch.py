"""Formal multi-epoch training loop shared by learned sparse reconstructors."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor, nn

from neuralop.evaluation import (
    SparseHistoryMetricAccumulator,
    SparseSurfaceMetricAccumulator,
)
from neuralop.training.masked_reconstruction import (
    MaskedReconstructionLoss,
    MaskedReconstructionPretrainer,
)
from neuralop.training.sparse_coupled_forecast import SparseCoupledForecastTrainer
from neuralop.training.sparse_experiment_runner import (
    SparseCheckpointManager,
    SparseExperimentContext,
    coupled_training_config,
    move_sparse_batch_to_device,
)


@dataclass(frozen=True)
class SparseMultiEpochConfig:
    epochs: int
    scheduler_name: str
    min_lr: float
    step_size: int
    gamma: float
    amp: bool
    early_stopping_patience: int
    primary_loss: str
    primary_mode: str
    abort_gradient_norm: float
    max_peak_memory_mb: float

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "SparseMultiEpochConfig":
        training = config.get("training", {})
        scheduler = training.get("scheduler", {})
        if isinstance(scheduler, str):
            scheduler = {"name": scheduler}
        result = cls(
            epochs=int(training.get("epochs", 0)),
            scheduler_name=str(scheduler.get("name", "cosine")).lower(),
            min_lr=float(scheduler.get("min_lr", 1.0e-6)),
            step_size=int(scheduler.get("step_size", 20)),
            gamma=float(scheduler.get("gamma", 0.5)),
            amp=bool(training.get("amp", True)),
            early_stopping_patience=int(
                training.get("early_stopping_patience", 0)
            ),
            primary_loss=str(training.get("primary_loss", "total")),
            primary_mode=str(training.get("primary_mode", "min")).lower(),
            abort_gradient_norm=float(training.get("abort_gradient_norm", math.inf)),
            max_peak_memory_mb=float(training.get("max_peak_memory_mb", math.inf)),
        )
        if result.epochs <= 0:
            raise ValueError("training.epochs must be positive for formal training")
        if result.scheduler_name not in {"cosine", "step", "constant"}:
            raise ValueError("training.scheduler.name must be cosine, step or constant")
        if result.min_lr < 0.0 or result.step_size <= 0 or result.gamma <= 0.0:
            raise ValueError("Invalid scheduler configuration")
        if result.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if result.primary_mode not in {"min", "max"}:
            raise ValueError("training.primary_mode must be min or max")
        if result.abort_gradient_norm <= 0.0 or result.max_peak_memory_mb <= 0.0:
            raise ValueError("Safety thresholds must be positive")
        return result


def _make_grad_scaler(device: torch.device, enabled: bool) -> Any:
    use_amp = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler(device.type, enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


class SparseMultiEpochTrainer:
    """One loop for learned reconstruction and forecast-aware training."""

    def __init__(
        self,
        context: SparseExperimentContext,
        pipeline: nn.Module,
        device: torch.device,
    ) -> None:
        self.context = context
        self.pipeline = pipeline
        self.device = device
        self.experiment_id = str(context.config["experiment_id"]).upper()
        learned_ids = {
            "B4", "B5", "B5-GNO", "B5-GINO", "B6",
            "H1", "H1-A1", "H1-A2", "H1-A3", "H1-A4", "H1-A5", "H1-A6",
            "P1", "P2", "V0", "V1", "V2", "V3", "V4", "PC-D1", "PC-D2",
            "ST-D0", "MU-V3", "FD-R1", "FD-A1", "FD-L1",
        }
        if self.experiment_id not in learned_ids:
            raise ValueError(f"Formal learned training supports {sorted(learned_ids)}")
        if set(("train", "val")) - set(context.data.loaders):
            raise ValueError("Formal training requires train and val loaders")

        self.loop_config = SparseMultiEpochConfig.from_mapping(context.config)
        self.coupled_config = coupled_training_config(context.config)
        self.forecast_supervision = bool(self.coupled_config.forecast_supervision)
        if self.experiment_id == "B4" and self.forecast_supervision:
            raise ValueError("B4 training does not support forecast_supervision=true")
        if self.experiment_id in {"P2", "B6", "V4", "FD-A1", "FD-L1"} and not self.forecast_supervision:
            raise ValueError(
                f"{self.experiment_id} joint training requires forecast_supervision=true"
            )
        if (
            self.experiment_id
            in {
                "P1", "H1", "H1-A1", "H1-A2", "H1-A3",
                "H1-A4", "H1-A5", "H1-A6",
                "B5", "B5-GNO", "B5-GINO",
                "V3", "MU-V3",
            }
            and self.forecast_supervision
        ):
            if self.coupled_config.use_rollout_curriculum:
                raise ValueError(
                    "Frozen forecast-supervised training requires a fixed 30-frame rollout"
                )
            if self.coupled_config.max_rollout_steps != 30:
                raise ValueError(
                    "Frozen forecast-supervised training requires max_rollout_steps=30"
                )
        self.reconstruction_only = not self.forecast_supervision
        if self.experiment_id in {"P2", "B6", "V4", "FD-R1", "FD-A1", "FD-L1"} and self.loop_config.amp:
            label = {
                "P2": "P2 RFNO",
                "B6": "B6 joint RFNO",
                "V4": "V4 joint RFNO",
                "FD-R1": "FD-R1 FNO-DeepONet reconstruction",
                "FD-A1": "FD-A1 autoregressive FNO-DeepONet",
                "FD-L1": "FD-L1 direct FNO-DeepONet",
            }[self.experiment_id]
            raise ValueError(
                f"{label} training requires training.amp=false because the "
                "spectral FFT path is not reliable in float16 on the supported "
                "Windows/CUDA runtime"
            )
        self.max_grad_norm = float(self.coupled_config.grad_clip_norm)
        if self.reconstruction_only:
            if self.coupled_config.coupling != "frozen":
                raise ValueError("Reconstruction-only training requires coupling=frozen")
            criterion = MaskedReconstructionLoss(
                hidden_weight=self.coupled_config.hidden_weight,
                observation_weight=self.coupled_config.observation_weight,
                history_gradient_weight=self.coupled_config.history_gradient_weight,
            )
            self.batch_trainer: Any = MaskedReconstructionPretrainer(
                pipeline.reconstructor,  # type: ignore[attr-defined]
                criterion,
                mae_mask_ratio=float(
                    context.config["training"].get("mae_mask_ratio", 0.0)
                ),
                mae_patch_size=int(
                    context.config["model"].get("vit_patch_size", 4)
                ),
                mae_tokenization=str(
                    context.config["model"].get(
                        "vit_tokenization", "spatial_patch"
                    )
                ),
                mae_tubelet_size=int(
                    context.config["model"].get("vit_tubelet_size", 10)
                ),
                training_mask_seed=int(
                    context.config["training"].get(
                        "mae_mask_seed", context.config["runtime"].get("seed", 42)
                    )
                ),
                evaluation_mask_seed=int(
                    context.config["training"].get(
                        "mae_validation_mask_seed",
                        int(context.config["runtime"].get("seed", 42)) + 1729,
                    )
                )
            )
            parameters = tuple(
                parameter
                for parameter in pipeline.reconstructor.parameters()  # type: ignore[attr-defined]
                if parameter.requires_grad
            )
            self.optimizer = torch.optim.AdamW(
                parameters,
                lr=self.coupled_config.reconstructor_lr,
                weight_decay=self.coupled_config.weight_decay,
            )
        else:
            expected_coupling = (
                "direct"
                if self.experiment_id == "FD-L1"
                else (
                    "joint"
                    if self.experiment_id in {"P2", "B6", "V4", "FD-A1"}
                    else "frozen"
                )
            )
            if self.coupled_config.coupling != expected_coupling:
                raise ValueError(
                    f"{self.experiment_id} forecast-supervised training requires "
                    f"coupling={expected_coupling}"
                )
            self.batch_trainer = SparseCoupledForecastTrainer(
                pipeline, self.coupled_config  # type: ignore[arg-type]
            )
            self.optimizer = self.batch_trainer.build_optimizer()

        self.scheduler = self._build_scheduler()
        self.scaler = _make_grad_scaler(device, self.loop_config.amp)
        self.checkpoints = SparseCheckpointManager(
            context.run_dir, context.config_sha256
        )
        self.log_path = context.run_dir / "logs" / "epochs.jsonl"

    def _build_scheduler(self) -> Any:
        name = self.loop_config.scheduler_name
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.loop_config.epochs,
                eta_min=self.loop_config.min_lr,
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.loop_config.step_size,
                gamma=self.loop_config.gamma,
            )
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda _: 1.0
        )

    def _active_rollout_steps(self, epoch: int) -> int:
        if self.reconstruction_only:
            return 0
        return self.coupled_config.active_rollout_steps(
            epoch, self.loop_config.epochs
        )

    def _forward_batch(
        self, batch: Mapping[str, Any], active_rollout_steps: int
    ) -> Dict[str, Any]:
        if self.reconstruction_only:
            return self.batch_trainer.forward_batch(batch)
        return self.batch_trainer.forward_batch(
            batch, active_rollout_steps=active_rollout_steps
        )

    def _optimizer_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        )

    def _verify_gradients(self) -> None:
        trainable_gradients = [
            parameter.grad for parameter in self._optimizer_parameters()
        ]
        if not any(
            gradient is not None and bool(torch.count_nonzero(gradient).item())
            for gradient in trainable_gradients
        ):
            raise RuntimeError("No non-zero gradient reached trainable parameters")
        if not all(
            gradient is None or torch.isfinite(gradient).all()
            for gradient in trainable_gradients
        ):
            raise FloatingPointError("Trainable parameter gradient contains NaN/Inf")
        if self.coupled_config.coupling == "direct":
            direct_gradients = [
                parameter.grad
                for parameter in self.pipeline.direct_operator.parameters()  # type: ignore[attr-defined]
            ]
            if not any(
                gradient is not None and bool(torch.count_nonzero(gradient).item())
                for gradient in direct_gradients
            ):
                raise RuntimeError(
                    "Direct FNO-DeepONet did not receive a non-zero gradient"
                )
            return
        rfno_gradients = [
            parameter.grad
            for parameter in self.pipeline.rfno.parameters()  # type: ignore[attr-defined]
        ]
        reconstructor_gradients = [
            parameter.grad
            for parameter in self.pipeline.reconstructor.parameters()  # type: ignore[attr-defined]
        ]
        if not all(
            gradient is None or torch.isfinite(gradient).all()
            for gradient in reconstructor_gradients
        ):
            raise FloatingPointError("Reconstructor gradient contains NaN/Inf")
        if not any(
            gradient is not None and bool(torch.count_nonzero(gradient).item())
            for gradient in reconstructor_gradients
        ):
            raise RuntimeError("Reconstructor did not receive a non-zero gradient")
        if self.coupled_config.coupling == "frozen":
            if any(gradient is not None for gradient in rfno_gradients):
                raise RuntimeError("Frozen RFNO unexpectedly received parameter gradients")
        elif not any(
            gradient is not None and bool(torch.count_nonzero(gradient).item())
            for gradient in rfno_gradients
        ):
            raise RuntimeError("Joint RFNO did not receive a non-zero gradient")

    def _autocast_context(self) -> Any:
        if not self.scaler.is_enabled():
            return nullcontext()
        return torch.autocast(device_type=self.device.type, enabled=True)

    def _run_epoch(
        self,
        *,
        training: bool,
        active_rollout_steps: int,
        max_batches: Optional[int],
        verify_first_gradient: bool = False,
    ) -> Dict[str, float]:
        loader = self.context.data.loaders["train" if training else "val"]
        history_accumulator = (
            SparseHistoryMetricAccumulator(
                normalization_mean=self.context.data.normalization_mean,
                normalization_std=self.context.data.normalization_std,
            )
            if self.reconstruction_only and not training
            else None
        )
        forecast_accumulator = (
            SparseSurfaceMetricAccumulator(
                normalization_mean=self.context.data.normalization_mean,
                normalization_std=self.context.data.normalization_std,
                forecast_horizons=tuple(
                    horizon
                    for horizon in self.context.config["evaluation"][
                        "forecast_horizons"
                    ]
                    if int(horizon) <= active_rollout_steps
                ),
                frame_interval=float(self.context.config["data"].get("dt", 0.25)),
            )
            if self.forecast_supervision and not training
            else None
        )
        totals: Dict[str, float] = {}
        samples = 0
        batches = 0
        gradient_norm_total = 0.0
        gradient_norm_max = 0.0
        amp_overflow_batches = 0
        self.pipeline.train(training)
        if self.reconstruction_only and hasattr(
            self.batch_trainer, "set_mask_mode"
        ):
            self.batch_trainer.set_mask_mode(training=training)
        for raw_batch in loader:
            batch = move_sparse_batch_to_device(raw_batch, self.device)
            batch_size = int(batch["x_obs"].shape[0])
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    result = self._forward_batch(batch, active_rollout_steps)
                    loss = result["losses"]["total"]
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self._optimizer_parameters(),
                        self.max_grad_norm if self.max_grad_norm > 0.0 else math.inf,
                    )
                )
                if not math.isfinite(gradient_norm):
                    if not self.scaler.is_enabled():
                        raise FloatingPointError("Gradient norm contains NaN/Inf")
                    # A finite forward loss can overflow after AMP scaling. Let
                    # GradScaler skip the unsafe optimizer update and reduce its
                    # scale instead of treating this expected recovery path as a
                    # model-divergence safety stop.
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    amp_overflow_batches += 1
                else:
                    if gradient_norm > self.loop_config.abort_gradient_norm:
                        raise RuntimeError(
                            "Gradient explosion safety stop: "
                            f"norm={gradient_norm:.6g} exceeds "
                            f"{self.loop_config.abort_gradient_norm:.6g}"
                        )
                    gradient_norm_total += gradient_norm * batch_size
                    gradient_norm_max = max(gradient_norm_max, gradient_norm)
                    if verify_first_gradient and batches == 0:
                        self._verify_gradients()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                with torch.no_grad(), self._autocast_context():
                    result = self._forward_batch(batch, active_rollout_steps)
            for name, value in result["losses"].items():
                scalar = float(value.detach().float().cpu())
                if not math.isfinite(scalar):
                    raise FloatingPointError(f"Non-finite {name} loss")
                totals[name] = totals.get(name, 0.0) + scalar * batch_size
            if history_accumulator is not None:
                history_accumulator.update(
                    result["reconstruction"],
                    batch["x_full"],
                    result.get("supervision_mask", batch["obs_mask"]),
                    source_ids=batch["source_id"],
                )
            if forecast_accumulator is not None:
                forecast_accumulator.update(
                    result["history_reconstruction"],
                    batch["x_full"],
                    result["forecast"],
                    batch["y"],
                    batch["obs_mask"],
                    source_ids=batch["source_id"],
                )
            samples += batch_size
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
        if not samples:
            raise RuntimeError("No samples were processed in epoch")
        result = {name: value / samples for name, value in totals.items()}
        result["samples"] = float(samples)
        result["batches"] = float(batches)
        if training:
            result["gradient_norm_mean"] = gradient_norm_total / samples
            result["gradient_norm_max"] = gradient_norm_max
            result["amp_overflow_batches"] = float(amp_overflow_batches)
        if history_accumulator is not None:
            history = history_accumulator.compute()
            full = history["full"]
            missing = history["missing_region"]
            observed = history["observed_region"]
            result.update(
                {
                    "reconstruction_full_rmse": float(full["rmse"]),
                    "reconstruction_full_nrmse": float(full["nrmse"]),
                    "reconstruction_full_correlation": float(full["correlation"]),
                    "reconstruction_full_gradient_rmse": float(
                        full["gradient_rmse"]
                    ),
                    "reconstruction_full_gradient_nrmse": float(
                        full["gradient_nrmse"]
                    ),
                    "reconstruction_full_spectrum_ssp": float(full["ssp"]),
                    "reconstruction_missing_rmse": float(missing["rmse"]),
                    "reconstruction_missing_nrmse": float(missing["nrmse"]),
                    "reconstruction_missing_correlation": float(
                        missing["correlation"]
                    ),
                    "observation_consistency_rmse": float(observed["rmse"]),
                    "observation_consistency_nrmse": float(observed["nrmse"]),
                    "observation_consistency_correlation": float(
                        observed["correlation"]
                    ),
                    "reconstruction_metric_source_count": float(
                        history["source_count"]
                    ),
                }
            )
            if not all(
                math.isfinite(value)
                for key, value in result.items()
                if key.startswith("reconstruction_")
                or key.startswith("observation_consistency_")
            ):
                raise FloatingPointError(
                    "Non-finite reconstruction validation metric"
                )
        if forecast_accumulator is not None:
            metrics = forecast_accumulator.compute()
            history = metrics["history_reconstruction"]
            result.update(
                {
                    "reconstruction_full_nrmse": float(history["nrmse"]),
                    "reconstruction_full_correlation": float(
                        history["correlation"]
                    ),
                    "reconstruction_missing_nrmse": float(
                        history["missing_region"]["nrmse"]
                    ),
                    "observation_consistency_nrmse": float(
                        history["observed_region"]["nrmse"]
                    ),
                    "forecast_metric_source_count": float(metrics["source_count"]),
                }
            )
            for horizon in self.context.config["evaluation"]["forecast_horizons"]:
                key = f"forecast_{int(horizon)}"
                if key not in metrics:
                    continue
                values = metrics[key]
                result.update(
                    {
                        f"{key}_rmse": float(values["rmse"]),
                        f"{key}_nrmse": float(values["nrmse"]),
                        f"{key}_correlation": float(values["correlation"]),
                        f"{key}_ssp": float(values["ssp"]),
                        f"{key}_gradient_rmse": float(values["gradient_rmse"]),
                        f"{key}_gradient_nrmse": float(
                            values["gradient_nrmse"]
                        ),
                    }
                )
            metric_values = [
                value
                for key, value in result.items()
                if key.startswith("forecast_")
                or key.startswith("reconstruction_")
                or key.startswith("observation_consistency_")
            ]
            if not all(math.isfinite(float(value)) for value in metric_values):
                raise FloatingPointError("Non-finite forecast validation metric")
        return result

    @staticmethod
    def _improved(value: float, best: float, mode: str) -> bool:
        return value < best if mode == "min" else value > best

    def _append_epoch_log(self, payload: Mapping[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")

    def fit(
        self,
        *,
        resume_checkpoint: Optional[Path] = None,
        continuation_checkpoint: Optional[Path] = None,
        max_epochs_this_run: Optional[int] = None,
        max_train_batches: Optional[int] = None,
        max_val_batches: Optional[int] = None,
        run_kind: str = "formal",
    ) -> Dict[str, Any]:
        if resume_checkpoint is not None and continuation_checkpoint is not None:
            raise ValueError("Specify only resume_checkpoint or continuation_checkpoint")
        if run_kind not in {"formal", "smoke"}:
            raise ValueError("run_kind must be formal or smoke")
        if run_kind == "formal" and any(
            value is not None
            for value in (max_epochs_this_run, max_train_batches, max_val_batches)
        ):
            raise ValueError("Formal training cannot use smoke limits")

        start_epoch = 1
        global_step = 0
        best_metric = math.inf if self.loop_config.primary_mode == "min" else -math.inf
        best_epoch = 0
        epochs_without_improvement = 0
        elapsed_before = 0.0
        peak_allocated_before = 0
        peak_reserved_before = 0
        continuation_state: Optional[Dict[str, Any]] = None
        restore_checkpoint = resume_checkpoint or continuation_checkpoint
        if restore_checkpoint is not None:
            restored = self.checkpoints.restore(
                restore_checkpoint,
                self.pipeline,
                self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                data_loader_generator=self.context.data.generators["train"],
                restore_random_state=True,
                require_training_state=True,
                allow_config_mismatch=continuation_checkpoint is not None,
            )
            start_epoch = int(restored["epoch"]) + 1
            global_step = int(restored["global_step"])
            progress = restored["curriculum_state"]
            if (
                self.reconstruction_only
                and hasattr(self.batch_trainer, "load_training_mask_state")
                and progress.get("pretraining_mask_generator_state") is not None
            ):
                self.batch_trainer.load_training_mask_state(
                    str(progress["pretraining_mask_generator_state"])
                )
            if continuation_checkpoint is None:
                restored_best = progress.get("best_metric")
                if restored_best is not None:
                    best_metric = float(restored_best)
                best_epoch = int(progress.get("best_epoch", best_epoch))
                epochs_without_improvement = int(
                    progress.get("epochs_without_improvement", 0)
                )
            else:
                continuation_state = {
                    "checkpoint_path": str(Path(continuation_checkpoint).resolve()),
                    "source_scientific_config_sha256": restored[
                        "source_scientific_config_sha256"
                    ],
                    "source_epoch": int(restored["epoch"]),
                    "source_global_step": int(restored["global_step"]),
                    "source_curriculum_state": dict(progress),
                    "restored_components": dict(restored["restored_components"]),
                }
                # The parent selected a 30-frame checkpoint with a different metric.
                # Restore every training state first, then start a fresh, explicitly
                # recorded 300-frame model-selection contract in the child run.
                best_metric = (
                    math.inf
                    if self.loop_config.primary_mode == "min"
                    else -math.inf
                )
                best_epoch = 0
                epochs_without_improvement = 0
            elapsed_before = float(progress.get("elapsed_seconds", 0.0))
            peak_allocated_before = int(
                progress.get("peak_memory_allocated_bytes", 0)
            )
            peak_reserved_before = int(
                progress.get("peak_memory_reserved_bytes", 0)
            )

        if continuation_state is not None:
            transition_progress = {
                "active_rollout_steps": int(
                    continuation_state["source_curriculum_state"].get(
                        "active_rollout_steps", 30
                    )
                ),
                "total_epochs": self.loop_config.epochs,
                "best_metric": None,
                "best_epoch": 0,
                "epochs_without_improvement": 0,
                "run_kind": run_kind,
                "elapsed_seconds": elapsed_before,
                "peak_memory_allocated_bytes": peak_allocated_before,
                "peak_memory_reserved_bytes": peak_reserved_before,
                "continuation": continuation_state,
            }
            self.checkpoints.save_last(
                self.pipeline,
                self.optimizer,
                epoch=start_epoch - 1,
                global_step=global_step,
                metrics={},
                scheduler=self.scheduler,
                scaler=self.scaler,
                curriculum_state=transition_progress,
                data_loader_generator=self.context.data.generators["train"],
                require_complete_training_state=True,
            )

        final_epoch = self.loop_config.epochs
        if max_epochs_this_run is not None:
            if max_epochs_this_run <= 0:
                raise ValueError("max_epochs_this_run must be positive")
            final_epoch = min(
                final_epoch, start_epoch + int(max_epochs_this_run) - 1
            )

        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        completed_epochs = 0
        stopped_early = False
        safety_stop_reason: Optional[str] = None
        last_record: Dict[str, Any] = {}
        for epoch in range(start_epoch, final_epoch + 1):
            active_steps = self._active_rollout_steps(epoch)
            train_metrics = self._run_epoch(
                training=True,
                active_rollout_steps=active_steps,
                max_batches=max_train_batches,
                verify_first_gradient=completed_epochs == 0,
            )
            global_step += int(train_metrics["batches"])
            val_metrics = self._run_epoch(
                training=False,
                active_rollout_steps=active_steps,
                max_batches=max_val_batches,
            )
            eligible_for_best = (
                self.reconstruction_only
                or active_steps == self.coupled_config.max_rollout_steps
            )
            primary_value: Optional[float] = None
            if eligible_for_best:
                if self.loop_config.primary_loss not in val_metrics:
                    raise KeyError(
                        "Primary validation loss "
                        f"{self.loop_config.primary_loss!r} missing"
                    )
                primary_value = float(val_metrics[self.loop_config.primary_loss])
            improved = eligible_for_best and self._improved(
                float(primary_value), best_metric, self.loop_config.primary_mode
            )
            if improved:
                best_metric = primary_value
                best_epoch = epoch
                epochs_without_improvement = 0
            elif eligible_for_best:
                epochs_without_improvement += 1
            self.scheduler.step()

            flat_metrics = {
                **{f"train.{key}": value for key, value in train_metrics.items()},
                **{f"val.{key}": value for key, value in val_metrics.items()},
            }
            elapsed_total = elapsed_before + (time.perf_counter() - started)
            peak_allocated = peak_allocated_before
            peak_reserved = peak_reserved_before
            if self.device.type == "cuda":
                peak_allocated = max(
                    peak_allocated,
                    int(torch.cuda.max_memory_allocated(self.device)),
                )
                peak_reserved = max(
                    peak_reserved,
                    int(torch.cuda.max_memory_reserved(self.device)),
                )
            progress = {
                "active_rollout_steps": active_steps,
                "total_epochs": self.loop_config.epochs,
                "best_metric": best_metric if best_epoch > 0 else None,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "run_kind": run_kind,
                "elapsed_seconds": elapsed_total,
                "peak_memory_allocated_bytes": peak_allocated,
                "peak_memory_reserved_bytes": peak_reserved,
            }
            if self.reconstruction_only and hasattr(
                self.batch_trainer, "training_mask_state"
            ):
                progress["pretraining_mask_generator_state"] = (
                    self.batch_trainer.training_mask_state()
                )
            if continuation_state is not None:
                progress["continuation"] = continuation_state
            checkpoint_arguments = {
                "scheduler": self.scheduler,
                "scaler": self.scaler,
                "curriculum_state": progress,
                "data_loader_generator": self.context.data.generators["train"],
                "require_complete_training_state": True,
            }
            self.checkpoints.save_last(
                self.pipeline,
                self.optimizer,
                epoch=epoch,
                global_step=global_step,
                metrics=flat_metrics,
                **checkpoint_arguments,
            )
            if improved:
                self.checkpoints.save_best(
                    self.pipeline,
                    self.optimizer,
                    metric_name=f"val.{self.loop_config.primary_loss}",
                    metric_value=float(primary_value),
                    epoch=epoch,
                    global_step=global_step,
                    mode=self.loop_config.primary_mode,
                    metrics=flat_metrics,
                    **checkpoint_arguments,
                )

            last_record = {
                "epoch": epoch,
                "global_step": global_step,
                "active_rollout_steps": active_steps,
                "train": train_metrics,
                "val": val_metrics,
                "learning_rates": [
                    float(group["lr"]) for group in self.optimizer.param_groups
                ],
                "best_metric": best_metric if best_epoch > 0 else None,
                "best_epoch": best_epoch,
                "improved": improved,
                "run_kind": run_kind,
            }
            self._append_epoch_log(last_record)
            completed_epochs += 1

            peak_memory_mb = peak_allocated / (1024.0 * 1024.0)
            if peak_memory_mb > self.loop_config.max_peak_memory_mb:
                safety_stop_reason = (
                    f"peak memory {peak_memory_mb:.3f} MB exceeded configured "
                    f"limit {self.loop_config.max_peak_memory_mb:.3f} MB"
                )
                break

            patience = self.loop_config.early_stopping_patience
            if eligible_for_best and patience > 0 and epochs_without_improvement >= patience:
                stopped_early = True
                break

        elapsed_total = elapsed_before + (time.perf_counter() - started)
        peak_allocated = peak_allocated_before
        peak_reserved = peak_reserved_before
        if self.device.type == "cuda":
            peak_allocated = max(
                peak_allocated, int(torch.cuda.max_memory_allocated(self.device))
            )
            peak_reserved = max(
                peak_reserved, int(torch.cuda.max_memory_reserved(self.device))
            )
        return {
            "experiment_id": self.experiment_id,
            "run_kind": run_kind,
            "start_epoch": start_epoch,
            "completed_epochs": completed_epochs,
            "last_epoch": last_record.get("epoch", start_epoch - 1),
            "global_step": global_step,
            "best_metric": best_metric if best_epoch > 0 else None,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "safety_stop_reason": safety_stop_reason,
            "elapsed_seconds": elapsed_total,
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
            "peak_memory_allocated_mb": peak_allocated / (1024.0 * 1024.0),
            "peak_memory_reserved_mb": peak_reserved / (1024.0 * 1024.0),
            "last_record": last_record,
            "continuation": continuation_state,
        }


__all__ = ["SparseMultiEpochConfig", "SparseMultiEpochTrainer"]
