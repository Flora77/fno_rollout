"""Gate B/C diagnostics for the residual-graph B5 GNO.

Only the frozen train split is verified and opened. The script never requests test
or validation data, never consumes a future target, and never trains RFNO.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch import Tensor

from neuralop.training.masked_reconstruction import MaskedReconstructionLoss
from neuralop.training.sparse_experiment_runner import (
    build_sparse_dataloaders,
    build_sparse_pipeline,
    load_frozen_data_split,
    load_resolved_sparse_config,
)


def _to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Tensor]:
    required = ("x_full", "x_obs", "obs_mask")
    result: Dict[str, Tensor] = {}
    for key in required:
        value = batch[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"{key} must be a tensor")
        result[key] = value.to(device, non_blocking=True)
    return result


def _stack_two_samples(dataset: Any, device: torch.device) -> Dict[str, Tensor]:
    samples = [dataset[0], dataset[1]]
    return {
        key: torch.stack([sample[key] for sample in samples]).to(device)
        for key in ("x_full", "x_obs", "obs_mask")
    }


def _losses(
    model: torch.nn.Module,
    criterion: MaskedReconstructionLoss,
    batch: Mapping[str, Tensor],
) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    result = model.forward_with_aux(batch["x_obs"], batch["obs_mask"])
    losses = criterion(
        result["loss_reconstruction"],
        batch["x_full"],
        batch["obs_mask"],
    )
    return result, losses


def _module_gradient_report(model: torch.nn.Module) -> Dict[str, bool]:
    temporal_prefix = (
        "history_token_encoder."
        if getattr(model, "architecture", "") == "residual_graph_v4"
        else "history_encoder."
    )
    groups = {
        "temporal_encoder": temporal_prefix,
        "sensor_graph_blocks": "sensor_operator_blocks.",
        "sensor_to_grid": "sensor_to_grid.",
        "grid_refinement": "grid_refinement.",
        "output_projection": "grid_output_projection.",
    }
    report: Dict[str, bool] = {}
    for label, prefix in groups.items():
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        ]
        report[label] = bool(gradients) and all(
            torch.isfinite(gradient).all() for gradient in gradients
        ) and any(bool(torch.count_nonzero(gradient)) for gradient in gradients)
    return report


def _field_diagnostics(raw: Tensor) -> Dict[str, float]:
    dx = torch.roll(raw, shifts=-1, dims=-1) - raw
    dy = torch.roll(raw, shifts=-1, dims=-2) - raw
    spectrum = torch.fft.rfft2(raw.float(), dim=(-2, -1), norm="ortho")
    power = spectrum.real.square() + spectrum.imag.square()
    return {
        "raw_variance": float(raw.var().detach()),
        "raw_gradient_energy": float(
            (dx.square().mean() + dy.square().mean()).detach()
        ),
        "raw_spectral_energy": float(power.mean().detach()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/sparse_experiments/b5_gno_frozen.example.json"),
    )
    parser.add_argument(
        "--mode", choices=("two-sample", "fixed-batch"), required=True
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    default_steps = 500 if args.mode == "two-sample" else 100
    steps = default_steps if args.steps is None else int(args.steps)
    if steps <= 0 or args.log_every <= 0:
        raise ValueError("steps and log-every must be positive")

    config = load_resolved_sparse_config(args.config, device=args.device)
    if config["pipeline"]["reconstructor"] != "gno":
        raise ValueError("B5 residual-graph diagnostics require reconstructor=gno")
    if config["model"].get("gno_architecture") not in {
        "residual_graph_v3",
        "residual_graph_v4",
    }:
        raise ValueError("Diagnostics require a residual-graph GNO architecture")
    split = load_frozen_data_split(
        config["data"]["split_manifest_path"],
        config,
        verify_splits=("train",),
    )
    bundle = build_sparse_dataloaders(
        config,
        split,
        splits=("train",),
        include_targets=False,
    )

    seed = int(config["runtime"]["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    if args.mode == "two-sample":
        batch = _stack_two_samples(bundle.sparse_datasets["train"], device)
    else:
        batch = _to_device(next(iter(bundle.loaders["train"])), device)
    if "y" in batch:
        raise RuntimeError("Diagnostic batch must not contain future target y")

    pipeline = build_sparse_pipeline(config, device)
    model = pipeline.reconstructor
    if pipeline.coupling != "frozen":
        raise RuntimeError("B5 diagnostic requires frozen coupling")
    optimizer_parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    rfno_parameter_ids = {id(parameter) for parameter in pipeline.rfno.parameters()}
    if any(id(parameter) in rfno_parameter_ids for parameter in optimizer_parameters):
        raise RuntimeError("RFNO parameter leaked into GNO optimizer parameters")
    training = config["training"]
    weights = training["loss_weights"]
    criterion = MaskedReconstructionLoss(
        hidden_weight=float(weights["hidden_reconstruction"]),
        observation_weight=float(weights["observation_consistency"]),
        history_gradient_weight=float(weights["history_gradient"]),
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=float(training["reconstructor_lr"]),
        weight_decay=float(training["weight_decay"]),
    )
    grad_clip = float(training["grad_clip_norm"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    with torch.no_grad():
        initial_result, initial_losses = _losses(model, criterion, batch)
    initial = {
        name: float(value.detach()) for name, value in initial_losses.items()
    }
    logs = []
    memory_samples = []
    start = time.perf_counter()
    last_gradient_norm = math.nan
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _, losses = _losses(model, criterion, batch)
        losses["total"].backward()
        last_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(optimizer_parameters, grad_clip)
        )
        if not math.isfinite(last_gradient_norm):
            raise FloatingPointError("Diagnostic gradient norm is non-finite")
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == steps:
            record = {
                "step": step,
                "total": float(losses["total"].detach()),
                "hidden": float(losses["hidden_reconstruction"].detach()),
                "raw_observation": float(
                    losses["observation_consistency"].detach()
                ),
                "gradient_loss": float(losses["history_gradient"].detach()),
                "gradient_norm_preclip": last_gradient_norm,
            }
            if device.type == "cuda":
                record["memory_allocated_mb"] = float(
                    torch.cuda.memory_allocated(device) / (1024**2)
                )
                memory_samples.append(record["memory_allocated_mb"])
            logs.append(record)
            print(json.dumps(record), flush=True)

    model.eval()
    with torch.no_grad():
        final_result, final_losses = _losses(model, criterion, batch)
    final = {name: float(value.detach()) for name, value in final_losses.items()}
    ratios = {
        name: final[name] / max(initial[name], 1.0e-12)
        for name in ("total", "hidden_reconstruction", "observation_consistency")
    }
    field = _field_diagnostics(final_result["raw_reconstruction"])
    module_gradients = _module_gradient_report(model)
    elapsed = time.perf_counter() - start
    report = {
        "mode": args.mode,
        "steps": steps,
        "seed": seed,
        "test_accessed": False,
        "future_target_present": False,
        "sample_count": int(batch["x_full"].shape[0]),
        "initial": initial,
        "final": final,
        "ratios": ratios,
        "reduction_percent": {
            name: 100.0 * (1.0 - ratio) for name, ratio in ratios.items()
        },
        "field_diagnostics": field,
        "module_gradients": module_gradients,
        "rfno_gradients_none": pipeline.rfno_gradients_are_none(),
        "elapsed_seconds": elapsed,
        "peak_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else None
        ),
        "memory_samples_mb": memory_samples,
        "logs": logs,
    }
    gate_ratio = 0.30 if args.mode == "two-sample" else 0.70
    gate_passed = (
        ratios["total"] <= gate_ratio
        and ratios["hidden_reconstruction"] <= gate_ratio
        and ratios["observation_consistency"] < 1.0
        and initial["observation_consistency"] > 0.0
        and final["observation_consistency"] > 0.0
        and all(value > 1.0e-10 for value in field.values())
        and all(module_gradients.values())
        and report["rfno_gradients_none"]
    )
    report["gate_ratio"] = gate_ratio
    report["gate_passed"] = gate_passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"gate_passed": gate_passed, "output": str(args.output)}))
    if not gate_passed:
        raise RuntimeError(
            f"{args.mode} diagnostic failed its loss/gradient/field gate"
        )


if __name__ == "__main__":
    main()
