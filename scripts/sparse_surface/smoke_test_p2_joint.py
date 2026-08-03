#!/usr/bin/env python3
"""Run one 30-frame P2 joint batch with the real B0 RFNO checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neuralop.data.datasets.sparse_sea_surface import SparseMaskManifest
from neuralop.models.reconstructors import PartialConvMaskedAutoencoder
from neuralop.models.sparse_forecast_pipeline import PartialConvMAERFNOPipeline
from neuralop.training.sparse_coupled_forecast import (
    SparseCoupledForecastTrainer,
    SparseCoupledTrainingConfig,
)


def _smooth_sequence(batch_size: int, steps: int, size: int = 64) -> torch.Tensor:
    coordinates = torch.linspace(0.0, 2.0 * torch.pi, size + 1)[:-1]
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    time = torch.linspace(0.0, 1.5, steps).view(1, steps, 1, 1)
    phases = torch.linspace(0.0, 0.7, batch_size).view(batch_size, 1, 1, 1)
    return torch.sin(xx + phases + time) + 0.5 * torch.cos(yy - 0.7 * time)


def _gradient_summary(module: torch.nn.Module):
    gradients = [parameter.grad for parameter in module.parameters()]
    present = [gradient for gradient in gradients if gradient is not None]
    return {
        "parameter_tensors_with_grad": len(present),
        "all_present_gradients_finite": bool(present)
        and all(torch.isfinite(gradient).all() for gradient in present),
    }


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
    sequence = _smooth_sequence(args.batch_size, 90)
    x_full = sequence[:, :60]
    y = sequence[:, 60:90]
    mask = SparseMaskManifest(args.manifest).get_mask(args.mask_id).float()
    obs_mask = mask.unsqueeze(0).expand(args.batch_size, -1, -1, -1).clone()
    batch = {
        "x_full": x_full.to(device),
        "x_obs": (x_full * obs_mask).to(device),
        "obs_mask": obs_mask.to(device),
        "y": y.to(device),
    }

    pipeline = PartialConvMAERFNOPipeline.from_b0_checkpoint(
        args.checkpoint,
        PartialConvMaskedAutoencoder(),
        map_location=device,
        coupling="joint",
    ).to(device)
    config = SparseCoupledTrainingConfig(
        coupling="joint",
        reconstructor_lr=2.0e-4,
        rfno_lr=2.0e-5,
        hidden_weight=1.0,
        observation_weight=0.1,
        rollout_weight=1.0,
        history_gradient_weight=0.05,
        forecast_gradient_weight=0.05,
        spectrum_weight=0.01,
    )
    trainer = SparseCoupledForecastTrainer(pipeline, config)
    optimizer = trainer.build_optimizer()
    result = trainer.train_batch(batch, optimizer, active_rollout_steps=30)

    losses = {
        key: float(value.detach().cpu()) for key, value in result["losses"].items()
    }
    reconstructor_gradients = _gradient_summary(pipeline.reconstructor)
    rfno_gradients = _gradient_summary(pipeline.rfno)
    summary = {
        "coupling": pipeline.coupling,
        "history_shape": list(result["history_reconstruction"].shape),
        "forecast_shape": list(result["forecast"].shape),
        "losses": losses,
        "all_losses_finite": all(
            torch.isfinite(torch.tensor(value)) for value in losses.values()
        ),
        "optimizer_groups": [
            {
                "name": group["name"],
                "lr": group["lr"],
                "parameter_count": sum(
                    parameter.numel() for parameter in group["params"]
                ),
            }
            for group in optimizer.param_groups
        ],
        "reconstructor_gradients": reconstructor_gradients,
        "rfno_gradients": rfno_gradients,
    }
    print(json.dumps(summary, indent=2))
    passed = (
        summary["coupling"] == "joint"
        and summary["history_shape"] == [args.batch_size, 60, 64, 64]
        and summary["forecast_shape"] == [args.batch_size, 30, 64, 64]
        and summary["all_losses_finite"]
        and reconstructor_gradients["all_present_gradients_finite"]
        and rfno_gradients["all_present_gradients_finite"]
        and summary["optimizer_groups"][1]["lr"]
        < summary["optimizer_groups"][0]["lr"]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
