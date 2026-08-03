#!/usr/bin/env python3
"""Run one real B4 train batch through Mask U-Net and frozen B0 RFNO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neuralop.models.reconstructors import PartialConvMaskedAutoencoder
from neuralop.training.sparse_experiment_runner import (
    build_sparse_pipeline,
    coupled_training_config,
    move_sparse_batch_to_device,
    prepare_sparse_experiment,
    run_sparse_one_batch,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        default=Path("config/sparse_experiments/b4_mask_unet_frozen.example.json"),
        nargs="?",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(42)
    context = prepare_sparse_experiment(
        args.config,
        project_root=Path.cwd(),
        run_dir=args.run_dir,
        device=args.device,
        loader_splits=("train", "val"),
    )
    if "test" in context.data.loaders or "test" in context.data.dense_datasets:
        raise RuntimeError("B4 smoke test must not construct the test split")
    device = torch.device(context.config["runtime"]["device"])
    pipeline = build_sparse_pipeline(context.config, device)
    raw_batch = next(iter(context.data.loaders["train"]))
    batch = move_sparse_batch_to_device(raw_batch, device)
    result = run_sparse_one_batch(
        "B4",
        pipeline,
        batch,
        rollout_steps=30,
        training_config=coupled_training_config(context.config),
        backward=True,
    )

    optimizer = result["optimizer"]
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    rfno_parameter_ids = {id(parameter) for parameter in pipeline.rfno.parameters()}
    b4_parameter_count = sum(
        parameter.numel() for parameter in pipeline.reconstructor.parameters()
    )
    p1_parameter_count = sum(
        parameter.numel()
        for parameter in PartialConvMaskedAutoencoder().parameters()
    )
    losses = {
        key: float(value.detach().cpu()) for key, value in result["losses"].items()
    }
    summary = {
        "source_id": [int(value) for value in batch["source_id"]],
        "mask_id": list(batch["mask_id"]),
        "history_shape": list(result["history_reconstruction"].shape),
        "forecast_shape": list(result["forecast"].shape),
        "outputs_finite": bool(
            torch.isfinite(result["history_reconstruction"]).all()
            and torch.isfinite(result["forecast"]).all()
        ),
        "losses_finite": all(torch.isfinite(torch.tensor(value)) for value in losses.values()),
        "b4_reconstructor_parameters": b4_parameter_count,
        "p1_reconstructor_parameters": p1_parameter_count,
        "b4_over_p1_parameters": b4_parameter_count / p1_parameter_count,
        "rfno_in_optimizer": any(
            id(parameter) in rfno_parameter_ids for parameter in optimizer_parameters
        ),
        "rfno_requires_grad_false": all(
            not parameter.requires_grad for parameter in pipeline.rfno.parameters()
        ),
        "rfno_gradients_none": pipeline.rfno_gradients_are_none(),
        "test_loader_constructed": False,
        "normalization_mean": context.data.normalization_mean,
        "normalization_std": context.data.normalization_std,
    }
    print(json.dumps(summary, indent=2))
    passed = (
        summary["history_shape"] == [len(batch["mask_id"]), 60, 64, 64]
        and summary["forecast_shape"] == [len(batch["mask_id"]), 30, 64, 64]
        and summary["outputs_finite"]
        and summary["losses_finite"]
        and 0.8 <= summary["b4_over_p1_parameters"] <= 1.25
        and not summary["rfno_in_optimizer"]
        and summary["rfno_requires_grad_false"]
        and summary["rfno_gradients_none"]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
