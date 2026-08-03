#!/usr/bin/env python3
"""Overfit PC-D2 on two real normalized 60-frame training windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neuralop.data.datasets.sea_surface_simple import _load_mat_array
from neuralop.data.datasets.sparse_sea_surface import SparseMaskManifest
from neuralop.models.reconstructors import (
    PeriodicConfidenceMaskAwarePoolingViTReconstructor,
)
from neuralop.training.masked_reconstruction import (
    MaskedReconstructionLoss,
    MaskedReconstructionPretrainer,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--variable", default="height")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/hparam_sensitivity/mode_m28x28_h32_lp64.pt"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/masks/point_masks.npz")
    )
    parser.add_argument("--mask-id", default="mask_00006")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.data_file is None:
        candidates = sorted(Path("data/bimodal/train").glob("*.mat"))
        if not candidates:
            raise FileNotFoundError("No training MAT files found in data/bimodal/train")
        args.data_file = candidates[0]

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mean = float(checkpoint["train_dataset_mean"])
    std = float(checkpoint["train_dataset_std"])
    field = _load_mat_array(str(args.data_file), variable=args.variable)
    if field.shape[0] < 64:
        raise ValueError("Two-window overfit requires at least 64 frames")
    x_full = torch.stack(
        (
            torch.from_numpy(field[0:60].copy()),
            torch.from_numpy(field[4:64].copy()),
        )
    ).float()
    x_full = (x_full - mean) / std

    mask = SparseMaskManifest(args.manifest).get_mask(args.mask_id).float()
    obs_mask = mask.unsqueeze(0).expand(2, -1, -1, -1).clone()
    device = torch.device(args.device)
    batch = {
        "x_full": x_full.to(device),
        "x_obs": (x_full * obs_mask).to(device),
        "obs_mask": obs_mask.to(device),
    }

    model = PeriodicConfidenceMaskAwarePoolingViTReconstructor().to(device)
    trainer = MaskedReconstructionPretrainer(
        model,
        MaskedReconstructionLoss(
            hidden_weight=1.0,
            observation_weight=0.0,
            history_gradient_weight=0.05,
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial = float(trainer.validate_batch(batch)["losses"]["total"].cpu())
    gradients_finite = True
    nonzero_gradient_seen = False
    confidence_gradient_seen = False
    for _ in range(args.steps):
        result = trainer.train_batch(batch, optimizer, max_grad_norm=1.0)
        gradients = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        gradients_finite = gradients_finite and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        nonzero_gradient_seen = nonzero_gradient_seen or any(
            bool(torch.count_nonzero(gradient).item()) for gradient in gradients
        )
        confidence_gradients = [
            parameter.grad for parameter in model.confidence_embedding.parameters()
        ]
        confidence_gradient_seen = confidence_gradient_seen or any(
            gradient is not None and bool(torch.count_nonzero(gradient).item())
            for gradient in confidence_gradients
        )
        if not bool(torch.isfinite(result["losses"]["total"])):
            gradients_finite = False
            break

    final_result = trainer.validate_batch(batch)
    final = float(final_result["losses"]["total"].cpu())
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )
    summary = {
        "data_file": str(args.data_file.resolve()),
        "mask_id": args.mask_id,
        "samples": 2,
        "steps": args.steps,
        "initial_total_loss": initial,
        "final_total_loss": final,
        "final_over_initial": final / max(initial, 1.0e-12),
        "reduction_percent": 100.0 * (1.0 - final / max(initial, 1.0e-12)),
        "all_losses_finite": all(
            bool(torch.isfinite(value).all())
            for value in final_result["losses"].values()
        ),
        "gradients_finite": gradients_finite,
        "nonzero_gradient_seen": nonzero_gradient_seen,
        "confidence_gradient_seen": confidence_gradient_seen,
        "peak_memory_mb": peak_memory_mb,
    }
    print(json.dumps(summary, indent=2))
    passed = (
        summary["all_losses_finite"]
        and gradients_finite
        and nonzero_gradient_seen
        and confidence_gradient_seen
        and final < initial * 0.5
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
