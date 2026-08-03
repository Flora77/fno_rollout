#!/usr/bin/env python3
"""Run a synthetic one-batch B1 forward check with the real B0 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neuralop.data.datasets.sparse_sea_surface import SparseMaskManifest
from neuralop.evaluation.sparse_surface import compute_b1_metrics
from neuralop.models.sparse_forecast_pipeline import BilinearFrozenRFNOPipeline


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    manifest = SparseMaskManifest(args.manifest)
    mask = manifest.get_mask(args.mask_id).to(dtype=torch.float32)
    if tuple(mask.shape) != (60, 64, 64):
        raise ValueError(f"Smoke test expects mask (60,64,64), got {tuple(mask.shape)}")

    generator = torch.Generator().manual_seed(args.seed)
    x_full = torch.randn(args.batch_size, 60, 64, 64, generator=generator)
    obs_mask = mask.unsqueeze(0).expand(args.batch_size, -1, -1, -1).clone()
    x_obs = x_full * obs_mask
    y = torch.randn(args.batch_size, 300, 64, 64, generator=generator)

    pipeline = BilinearFrozenRFNOPipeline.from_b0_checkpoint(
        args.checkpoint, map_location=device
    ).to(device)
    pipeline.train()
    with torch.inference_mode():
        outputs = pipeline.rollout(
            x_obs.to(device), obs_mask.to(device), rollout_steps=300
        )
        all_valid_history = pipeline.reconstructor(
            x_full[:1].to(device), torch.ones_like(x_full[:1], device=device)
        )
        metrics = compute_b1_metrics(
            outputs["history_reconstruction"],
            x_full.to(device),
            outputs["forecast"],
            y.to(device),
            obs_mask.to(device),
            source_ids=list(range(args.batch_size)),
            normalization_mean=pipeline.normalization_mean,
            normalization_std=pipeline.normalization_std,
        )

    result = {
        "history_shape": list(outputs["history_reconstruction"].shape),
        "forecast_shape": list(outputs["forecast"].shape),
        "history_finite": bool(torch.isfinite(outputs["history_reconstruction"]).all()),
        "forecast_finite": bool(torch.isfinite(outputs["forecast"]).all()),
        "all_valid_identity": bool(
            torch.equal(all_valid_history.cpu(), x_full[:1])
        ),
        "rfno_gradients_none": pipeline.rfno_gradients_are_none(),
        "rfno_in_eval_mode": not pipeline.rfno.training,
        "optimizer_parameter_count": sum(
            parameter.numel() for parameter in pipeline.optimizer_parameters()
        ),
        "normalization_mean": pipeline.normalization_mean,
        "normalization_std": pipeline.normalization_std,
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2))

    required = (
        result["history_shape"] == [args.batch_size, 60, 64, 64]
        and result["forecast_shape"] == [args.batch_size, 300, 64, 64]
        and result["history_finite"]
        and result["forecast_finite"]
        and result["all_valid_identity"]
        and result["rfno_gradients_none"]
        and result["rfno_in_eval_mode"]
        and result["optimizer_parameter_count"] == 0
    )
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main())
