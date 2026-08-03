#!/usr/bin/env python3
"""Audit a saved P1 pilot on one real validation batch without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neuralop.training.sparse_experiment_runner import (
    SparseCheckpointManager,
    build_sparse_pipeline,
    move_sparse_batch_to_device,
    prepare_sparse_experiment,
)
from neuralop.training.sparse_multiepoch import SparseMultiEpochTrainer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda")
    return parser


@torch.no_grad()
def main() -> int:
    args = _parser().parse_args()
    context = prepare_sparse_experiment(
        args.config,
        project_root=args.project_root,
        resume_checkpoint=args.checkpoint,
        loader_splits=("train", "val"),
        device=args.device,
    )
    if str(context.config["experiment_id"]).upper() != "P1":
        raise ValueError("P1 pilot audit requires experiment_id=P1")

    device = torch.device(context.config["runtime"]["device"])
    pipeline = build_sparse_pipeline(context.config, device)
    restored = SparseCheckpointManager(
        context.run_dir, context.config_sha256
    ).restore(args.checkpoint, pipeline, restore_random_state=False)
    trainer = SparseMultiEpochTrainer(context, pipeline, device)

    batch = move_sparse_batch_to_device(
        next(iter(context.data.loaders["val"])), device
    )
    mask = batch["obs_mask"].bool()
    x_zero = batch["x_obs"]
    x_hundred = torch.where(mask, x_zero, torch.full_like(x_zero, 100.0))
    x_random = torch.where(mask, x_zero, torch.randn_like(x_zero) * 100.0)

    pipeline.reconstructor.eval()
    reconstruction_zero = pipeline.reconstructor(x_zero, batch["obs_mask"])
    reconstruction_hundred = pipeline.reconstructor(
        x_hundred, batch["obs_mask"]
    )
    reconstruction_random = pipeline.reconstructor(x_random, batch["obs_mask"])

    # A NaN future target is a fail-fast proof that P1 reconstruction pretraining
    # does not inspect y. All named losses must remain finite.
    nan_future_batch = dict(batch)
    nan_future_batch["y"] = torch.full_like(batch["y"], float("nan"))
    no_future_result = trainer.batch_trainer.forward_batch(nan_future_batch)

    optimizer_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    rfno_ids = {id(parameter) for parameter in pipeline.rfno.parameters()}
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "restored": restored,
        "loader_splits": sorted(context.data.loaders),
        "batch_shape": list(x_zero.shape),
        "mask_ids": list(batch["mask_id"]),
        "reconstruction_shape": list(reconstruction_zero.shape),
        "fill_0_vs_100_max_abs": float(
            (reconstruction_zero - reconstruction_hundred).abs().max()
        ),
        "fill_0_vs_random_max_abs": float(
            (reconstruction_zero - reconstruction_random).abs().max()
        ),
        "observed_values_unchanged": bool(
            torch.equal(x_zero[mask], x_hundred[mask])
            and torch.equal(x_zero[mask], x_random[mask])
        ),
        "reconstruction_finite": bool(
            torch.isfinite(reconstruction_zero).all()
            and torch.isfinite(reconstruction_hundred).all()
            and torch.isfinite(reconstruction_random).all()
        ),
        "nan_y_pretraining_losses_finite": bool(
            all(
                torch.isfinite(value).all()
                for value in no_future_result["losses"].values()
            )
        ),
        "optimizer_rfno_parameter_overlap": len(optimizer_ids & rfno_ids),
        "rfno_requires_grad_false": all(
            not parameter.requires_grad for parameter in pipeline.rfno.parameters()
        ),
        "rfno_gradients_none": all(
            parameter.grad is None for parameter in pipeline.rfno.parameters()
        ),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
