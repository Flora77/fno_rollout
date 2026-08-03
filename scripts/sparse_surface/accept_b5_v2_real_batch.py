#!/usr/bin/env python3
"""Acceptance checks for nonlinear periodic B5 GNO/GINO on one real batch."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from neuralop.training.sparse_experiment_runner import (
    build_sparse_pipeline,
    move_sparse_batch_to_device,
    prepare_sparse_experiment,
)
from neuralop.training.sparse_multiepoch import SparseMultiEpochTrainer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    context = prepare_sparse_experiment(
        args.config,
        project_root=args.project_root,
        run_dir=args.run_dir,
        device=args.device,
        loader_splits=("train", "val"),
        frozen_split_manifest=args.split_manifest,
    )
    seed = int(context.config["runtime"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(context.config["runtime"]["device"])
    pipeline = build_sparse_pipeline(context.config, device)
    trainer = SparseMultiEpochTrainer(context, pipeline, device)
    optimizer = trainer.optimizer
    batch = move_sparse_batch_to_device(
        next(iter(context.data.loaders["train"])), device
    )
    batch_without_y = {key: value for key, value in batch.items() if key != "y"}

    pipeline.train(True)
    optimizer.zero_grad(set_to_none=True)
    result = trainer.batch_trainer.forward_batch(batch_without_y)
    total = result["losses"]["total"]
    total.backward()

    gradients = {
        name: parameter.grad
        for name, parameter in pipeline.reconstructor.named_parameters()
        if parameter.requires_grad
    }
    missing_gradients = sorted(
        name for name, gradient in gradients.items() if gradient is None
    )
    nonfinite_gradients = sorted(
        name
        for name, gradient in gradients.items()
        if gradient is not None and not bool(torch.isfinite(gradient).all())
    )
    zero_gradients = sorted(
        name
        for name, gradient in gradients.items()
        if gradient is not None and not bool(torch.count_nonzero(gradient).item())
    )

    reconstruction = result["reconstruction"]
    observed = batch["obs_mask"].bool()
    observation_max_abs = float(
        (reconstruction[observed] - batch["x_obs"][observed]).abs().max().item()
    )

    pipeline.eval()
    zero_fill = torch.where(observed, batch["x_obs"], torch.zeros_like(batch["x_obs"]))
    hundred_fill = torch.where(
        observed, batch["x_obs"], torch.full_like(batch["x_obs"], 100.0)
    )
    random_fill = torch.where(
        observed, batch["x_obs"], torch.randn_like(batch["x_obs"]) * 100.0
    )
    with torch.no_grad():
        zero_output = pipeline.reconstructor(zero_fill, batch["obs_mask"])
        hundred_output = pipeline.reconstructor(hundred_fill, batch["obs_mask"])
        random_output = pipeline.reconstructor(random_fill, batch["obs_mask"])
        rollout = pipeline.rollout(
            batch["x_obs"], batch["obs_mask"], rollout_steps=300
        )

    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    rfno_parameters = tuple(pipeline.rfno.parameters())
    payload = {
        "experiment_id": context.config["experiment_id"],
        "scientific_config_sha256": context.config_sha256,
        "loader_splits": sorted(context.data.loaders),
        "train_samples": len(context.data.sparse_datasets["train"]),
        "val_samples": len(context.data.sparse_datasets["val"]),
        "y_removed_before_forward": "y" not in batch_without_y,
        "reconstruction_shape": list(reconstruction.shape),
        "rollout_shape": list(rollout["forecast"].shape),
        "reconstruction_finite": bool(torch.isfinite(reconstruction).all()),
        "loss_finite": bool(torch.isfinite(total)),
        "forecast_finite": bool(torch.isfinite(rollout["forecast"]).all()),
        "trainable_reconstructor_parameter_tensors": len(gradients),
        "missing_gradient_names": missing_gradients,
        "nonfinite_gradient_names": nonfinite_gradients,
        "zero_gradient_names": zero_gradients,
        "hard_observation_consistency_max_abs": observation_max_abs,
        "fill_0_vs_100_max_abs": float(
            (zero_output - hundred_output).abs().max().item()
        ),
        "fill_0_vs_random_max_abs": float(
            (zero_output - random_output).abs().max().item()
        ),
        "rfno_eval": not pipeline.rfno.training,
        "rfno_requires_grad_false": all(
            not parameter.requires_grad for parameter in rfno_parameters
        ),
        "rfno_excluded_from_optimizer": all(
            id(parameter) not in optimizer_ids for parameter in rfno_parameters
        ),
        "rfno_gradients_none": all(
            parameter.grad is None for parameter in rfno_parameters
        ),
    }
    passed = (
        payload["loader_splits"] == ["train", "val"]
        and payload["val_samples"] == 777
        and payload["y_removed_before_forward"]
        and payload["reconstruction_shape"][1:] == [60, 64, 64]
        and payload["rollout_shape"][1:] == [300, 64, 64]
        and payload["reconstruction_finite"]
        and payload["loss_finite"]
        and payload["forecast_finite"]
        and not missing_gradients
        and not nonfinite_gradients
        and observation_max_abs == 0.0
        and payload["fill_0_vs_100_max_abs"] == 0.0
        and payload["fill_0_vs_random_max_abs"] == 0.0
        and payload["rfno_eval"]
        and payload["rfno_requires_grad_false"]
        and payload["rfno_excluded_from_optimizer"]
        and payload["rfno_gradients_none"]
    )
    payload["passed"] = passed
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output = args.run_dir / "acceptance.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
