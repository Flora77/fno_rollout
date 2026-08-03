#!/usr/bin/env python3
"""Run one P1 pretraining batch and one frozen-B0 validation batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neuralop.data.datasets.sparse_sea_surface import SparseMaskManifest
from neuralop.models.reconstructors import PartialConvMaskedAutoencoder
from neuralop.models.sparse_forecast_pipeline import (
    PartialConvMAEFrozenRFNOPipeline,
)
from neuralop.training.masked_reconstruction import (
    MaskedReconstructionLoss,
    MaskedReconstructionPretrainer,
)


def _smooth_history(batch_size: int, size: int = 64) -> torch.Tensor:
    coordinates = torch.linspace(0.0, 2.0 * torch.pi, size + 1)[:-1]
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    time = torch.linspace(0.0, 1.0, 60).view(1, 60, 1, 1)
    phases = torch.linspace(0.0, 0.7, batch_size).view(batch_size, 1, 1, 1)
    return torch.sin(xx + phases + time) + 0.5 * torch.cos(yy - time)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/hparam_sensitivity/mode_m28x28_h32_lp64.pt"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/masks/point_masks.json")
    )
    parser.add_argument("--mask-id", default="mask_00006")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device(args.device)
    manifest = SparseMaskManifest(args.manifest)
    mask = manifest.get_mask(args.mask_id).float()
    if tuple(mask.shape) != (60, 64, 64):
        raise ValueError(f"P1 smoke test expects mask (60,64,64), got {mask.shape}")
    x_full = _smooth_history(args.batch_size)
    obs_mask = mask.unsqueeze(0).expand(args.batch_size, -1, -1, -1).clone()
    batch = {
        "x_full": x_full.to(device),
        "x_obs": (x_full * obs_mask).to(device),
        "obs_mask": obs_mask.to(device),
    }

    reconstructor = PartialConvMaskedAutoencoder()
    pipeline = PartialConvMAEFrozenRFNOPipeline.from_b0_checkpoint(
        args.checkpoint,
        reconstructor,
        map_location=device,
    ).to(device)
    criterion = MaskedReconstructionLoss(
        hidden_weight=1.0,
        observation_weight=0.1,
        history_gradient_weight=0.05,
    )
    pretrainer = MaskedReconstructionPretrainer(
        pipeline.reconstructor, criterion
    ).to(device)
    optimizer_parameters = tuple(pipeline.optimizer_parameters())
    optimizer = torch.optim.Adam(optimizer_parameters, lr=2.0e-3)
    rfno_parameter_ids = {id(parameter) for parameter in pipeline.rfno.parameters()}

    train_result = pretrainer.train_batch(batch, optimizer, max_grad_norm=1.0)
    validation_result = pretrainer.validate_batch(batch)
    pipeline.eval()
    with torch.no_grad():
        outputs = pipeline(batch["x_obs"], batch["obs_mask"])

    train_losses = {
        key: float(value.detach().cpu())
        for key, value in train_result["losses"].items()
    }
    validation_losses = {
        key: float(value.detach().cpu())
        for key, value in validation_result["losses"].items()
    }
    result = {
        "history_shape": list(outputs["history_reconstruction"].shape),
        "forecast_shape": list(outputs["forecast"].shape),
        "history_finite": bool(torch.isfinite(outputs["history_reconstruction"]).all()),
        "forecast_finite": bool(torch.isfinite(outputs["forecast"]).all()),
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "all_losses_finite": all(
            torch.isfinite(torch.tensor(value))
            for value in (*train_losses.values(), *validation_losses.values())
        ),
        "optimizer_parameter_count": sum(
            parameter.numel() for parameter in optimizer_parameters
        ),
        "optimizer_contains_rfno": any(
            id(parameter) in rfno_parameter_ids for parameter in optimizer_parameters
        ),
        "rfno_gradients_none": pipeline.rfno_gradients_are_none(),
        "rfno_in_eval_mode": not pipeline.rfno.training,
        "normalization_mean": pipeline.normalization_mean,
        "normalization_std": pipeline.normalization_std,
    }
    print(json.dumps(result, indent=2))

    passed = (
        result["history_shape"] == [args.batch_size, 60, 64, 64]
        and result["forecast_shape"] == [args.batch_size, 30, 64, 64]
        and result["history_finite"]
        and result["forecast_finite"]
        and result["all_losses_finite"]
        and result["optimizer_parameter_count"] > 0
        and not result["optimizer_contains_rfno"]
        and result["rfno_gradients_none"]
        and result["rfno_in_eval_mode"]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
