"""One-batch forward/backward smoke test for the B5 residual-graph GNO.

This script uses the real fixed-point mask but synthetic normalized fields. It checks
one frozen-RFNO forecast chunk, but does not load future targets, save a checkpoint,
or start formal training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neuralop.models.reconstructors import PeriodicGNOReconstructor
from neuralop.models.sparse_forecast_pipeline import (
    LearnedReconstructionRFNOPipeline,
)
from neuralop.training.masked_reconstruction import (
    MaskedReconstructionLoss,
    MaskedReconstructionPretrainer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask-manifest",
        type=Path,
        default=Path("data/masks/point_masks.npz"),
    )
    parser.add_argument("--mask-id", default="mask_00006")
    parser.add_argument(
        "--rfno-checkpoint",
        type=Path,
        default=Path("checkpoints/hparam_sensitivity/mode_m28x28_h32_lp64.pt"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--rollout-steps", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(20260731)
    device = torch.device(args.device)
    with np.load(args.mask_manifest, allow_pickle=False) as archive:
        spatial_temporal_mask = archive[args.mask_id]
    if spatial_temporal_mask.shape != (60, 64, 64):
        raise ValueError(
            "B5 smoke test expects mask shape (60,64,64), got "
            f"{spatial_temporal_mask.shape}"
        )
    obs_mask = torch.from_numpy(
        spatial_temporal_mask.astype(np.float32, copy=False)
    ).unsqueeze(0).to(device)
    x_full = torch.randn(1, 60, 64, 64, device=device)
    x_obs = x_full * obs_mask

    model = PeriodicGNOReconstructor(
        input_steps=60,
        hidden_channels=64,
        radius=0.20,
        mlp_channels=(64, 64),
        kernel_mode="channelwise_nonlinear",
        kernel_rank=64,
        hard_observation_consistency=True,
        architecture="residual_graph_v3",
        temporal_channels=32,
        temporal_dilations=(1, 2, 4),
        graph_layers=3,
        grid_refinement_layers=3,
        kernel_normalization="neighbor_mean",
        query_chunk_size=128,
        raw_observation_supervision=True,
    ).to(device)
    trainer = MaskedReconstructionPretrainer(
        model,
        MaskedReconstructionLoss(
            hidden_weight=1.0,
            observation_weight=0.1,
            history_gradient_weight=0.05,
        ),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.0e-4, weight_decay=1.0e-4
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    result = trainer.train_batch(
        {
            "x_full": x_full,
            "x_obs": x_obs,
            "obs_mask": obs_mask,
        },
        optimizer,
        max_grad_norm=1.0,
    )
    pipeline = LearnedReconstructionRFNOPipeline.from_b0_checkpoint(
        args.rfno_checkpoint,
        model,
        map_location=device,
        coupling="frozen",
    ).to(device)
    with torch.no_grad():
        pipeline_result = pipeline.rollout(
            x_obs,
            obs_mask,
            rollout_steps=args.rollout_steps,
            detach_context=True,
        )
    reconstruction = result["reconstruction"]
    raw_reconstruction = result["raw_reconstruction"]
    forecast = pipeline_result["forecast"]
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    observed = obs_mask.bool()
    report = {
        "device": str(device),
        "sensor_count": int(obs_mask[0, 0].sum().item()),
        "shape": list(reconstruction.shape),
        "raw_finite": bool(torch.isfinite(raw_reconstruction).all()),
        "consistent_finite": bool(torch.isfinite(reconstruction).all()),
        "forecast_shape": list(forecast.shape),
        "forecast_finite": bool(torch.isfinite(forecast).all()),
        "rfno_gradients_none": pipeline.rfno_gradients_are_none(),
        "hard_consistency_max_abs": float(
            (reconstruction[observed] - x_obs[observed]).abs().max().item()
        ),
        "observation_loss": float(
            result["losses"]["observation_consistency"].detach().item()
        ),
        "total_loss": float(result["losses"]["total"].detach().item()),
        "finite_nonzero_gradient_tensors": sum(
            bool(torch.isfinite(gradient).all() and torch.count_nonzero(gradient))
            for gradient in gradients
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "peak_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else None
        ),
    }
    if not all(
        (
            report["raw_finite"],
            report["consistent_finite"],
            report["forecast_finite"],
            report["forecast_shape"] == [1, args.rollout_steps, 64, 64],
            report["rfno_gradients_none"],
            report["hard_consistency_max_abs"] == 0.0,
            report["observation_loss"] > 0.0,
            report["finite_nonzero_gradient_tensors"] > 0,
        )
    ):
        raise RuntimeError(f"B5 residual-graph smoke test failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
