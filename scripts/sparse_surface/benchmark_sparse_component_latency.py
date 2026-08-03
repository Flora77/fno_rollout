#!/usr/bin/env python
"""Benchmark reconstruction and 300-frame forecast latency separately on validation.

The benchmark deliberately excludes data loading and host-to-device transfer.  For the
direct FD-L1 operator, sparse-history encoding plus history decoding is charged to the
reconstruction stage; decoding future query times is charged to forecasting.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from neuralop.models.sparse_forecast_pipeline import (
    FNODeepONetDirectSparsePipeline,
    LearnedReconstructionRFNOPipeline,
    _autoregressive_rfno_rollout,
)
from neuralop.training.sparse_experiment_runner import (
    SparseCheckpointManager,
    build_sparse_pipeline,
    move_sparse_batch_to_device,
    prepare_sparse_experiment,
    sparse_model_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rollout-steps", type=int, default=300)
    parser.add_argument("--max-batches", type=int, default=30)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help=(
            "Permit validation-only restore after an external protocol audit. "
            "This is needed by legacy checkpoints whose hash predates unused defaults."
        ),
    )
    return parser


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reconstruct_learned(
    pipeline: LearnedReconstructionRFNOPipeline,
    x_obs: Tensor,
    obs_mask: Tensor,
) -> Tensor:
    forward_with_aux = getattr(pipeline.reconstructor, "forward_with_aux", None)
    if callable(forward_with_aux):
        result = forward_with_aux(x_obs, obs_mask)
        if isinstance(result, Mapping):
            return result["reconstruction"]
        if isinstance(result, tuple) and result and isinstance(result[0], Tensor):
            return result[0]
        raise TypeError("Unsupported reconstructor forward_with_aux result")
    return pipeline.reconstructor(x_obs, obs_mask)


def _run_stages(
    pipeline: torch.nn.Module,
    x_obs: Tensor,
    obs_mask: Tensor,
    *,
    rollout_steps: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, float, float]:
    _sync(device)
    started = time.perf_counter()
    if isinstance(pipeline, LearnedReconstructionRFNOPipeline):
        history = _reconstruct_learned(pipeline, x_obs, obs_mask)
        _sync(device)
        reconstruction_seconds = time.perf_counter() - started

        started = time.perf_counter()
        forecast = _autoregressive_rfno_rollout(
            pipeline.rfno,
            history,
            input_steps=pipeline.input_steps,
            output_steps=pipeline.output_steps,
            rollout_steps=rollout_steps,
            detach_context=False,
        )
        _sync(device)
        forecast_seconds = time.perf_counter() - started
        return history, forecast, reconstruction_seconds, forecast_seconds

    if isinstance(pipeline, FNODeepONetDirectSparsePipeline):
        branch_code = pipeline.direct_operator.encode_sparse_history(x_obs, obs_mask)
        history_times = torch.linspace(
            -1.0,
            0.0,
            pipeline.input_steps,
            device=x_obs.device,
            dtype=x_obs.dtype,
        )
        raw_history = pipeline.direct_operator.decode_grid(branch_code, history_times)
        mask = obs_mask.to(dtype=x_obs.dtype)
        history = (
            raw_history * (1.0 - mask) + x_obs * mask
            if pipeline.hard_observation_consistency
            else raw_history
        )
        _sync(device)
        reconstruction_seconds = time.perf_counter() - started

        started = time.perf_counter()
        future_times = torch.arange(
            1,
            rollout_steps + 1,
            device=x_obs.device,
            dtype=x_obs.dtype,
        ) / float(pipeline.forecast_steps)
        forecast = pipeline.direct_operator.decode_grid(branch_code, future_times)
        _sync(device)
        forecast_seconds = time.perf_counter() - started
        return history, forecast, reconstruction_seconds, forecast_seconds

    raise TypeError(f"Unsupported benchmark pipeline: {type(pipeline).__name__}")


@torch.no_grad()
def main() -> int:
    args = _parser().parse_args()
    if args.max_batches <= 0 or args.warmup_batches < 0:
        raise ValueError("max-batches must be positive and warmup-batches non-negative")
    project_root = args.project_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if checkpoint.name != "best.pt":
        raise PermissionError("Component latency must benchmark the selected best.pt")

    context = prepare_sparse_experiment(
        args.config,
        project_root=project_root,
        run_dir=args.run_dir,
        device=args.device,
        loader_splits=("val",),
    )
    device = torch.device(context.config["runtime"]["device"])
    pipeline = build_sparse_pipeline(context.config, device)
    restore = SparseCheckpointManager(
        context.run_dir, context.config_sha256
    ).restore(
        checkpoint,
        pipeline,
        restore_random_state=False,
        allow_config_mismatch=bool(args.allow_config_mismatch),
    )
    pipeline.eval()

    loader = context.data.loaders["val"]
    reconstruction_seconds = 0.0
    forecast_seconds = 0.0
    samples = 0
    measured_batches = 0
    warmup_left = int(args.warmup_batches)
    for raw_batch in loader:
        batch = move_sparse_batch_to_device(raw_batch, device)
        inputs = sparse_model_inputs(batch)
        history, forecast, recon_s, forecast_s = _run_stages(
            pipeline,
            inputs["x_obs"],
            inputs["obs_mask"],
            rollout_steps=int(args.rollout_steps),
            device=device,
        )
        if not torch.isfinite(history).all() or not torch.isfinite(forecast).all():
            raise FloatingPointError("Latency benchmark produced NaN/Inf")
        expected_forecast = (
            history.shape[0],
            int(args.rollout_steps),
            history.shape[-2],
            history.shape[-1],
        )
        if history.shape != inputs["x_obs"].shape or forecast.shape != expected_forecast:
            raise RuntimeError("Latency benchmark output shape mismatch")
        if warmup_left:
            warmup_left -= 1
            continue
        reconstruction_seconds += recon_s
        forecast_seconds += forecast_s
        samples += int(history.shape[0])
        measured_batches += 1
        if measured_batches >= int(args.max_batches):
            break

    if measured_batches == 0 or samples == 0:
        raise RuntimeError("No validation samples were benchmarked")
    payload: dict[str, Any] = {
        "experiment_id": str(context.config["experiment_id"]),
        "experiment_name": str(context.config["experiment_name"]),
        "split": "val",
        "scientific_config_sha256": context.config_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(restore.get("epoch", 0)),
        "checkpoint_restore_mode": (
            "validation_benchmark_config_mismatch"
            if args.allow_config_mismatch
            else "strict"
        ),
        "device": str(device),
        "rollout_steps": int(args.rollout_steps),
        "warmup_batches": int(args.warmup_batches),
        "measured_batches": measured_batches,
        "measured_samples": samples,
        "timing_scope": "model_only_excludes_dataloader_and_host_to_device_transfer",
        "direct_latency_allocation": (
            "history_encoding_and_history_decoding_to_reconstruction; "
            "future_time_decoding_to_forecast"
        ),
        "reconstruction_seconds": reconstruction_seconds,
        "reconstruction_seconds_per_sample": reconstruction_seconds / samples,
        "forecast_300_seconds": forecast_seconds,
        "forecast_300_seconds_per_sample": forecast_seconds / samples,
        "total_seconds": reconstruction_seconds + forecast_seconds,
        "total_seconds_per_sample": (
            reconstruction_seconds + forecast_seconds
        ) / samples,
    }
    output = context.run_dir / "component_latency.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
