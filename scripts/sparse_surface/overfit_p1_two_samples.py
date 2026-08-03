#!/usr/bin/env python3
"""Overfit P1 reconstruction on two real 60-frame windows only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neuralop.data.datasets.sea_surface_simple import _load_mat_array
from neuralop.data.datasets.sparse_sea_surface import SparseMaskManifest
from neuralop.models.reconstructors import PartialConvMaskedAutoencoder
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
        "--manifest", type=Path, default=Path("data/masks/point_masks.json")
    )
    parser.add_argument("--mask-id", default="mask_00006")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=5.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
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
    first = torch.from_numpy(field[0:60].copy())
    second = torch.from_numpy(field[4:64].copy())
    x_full = (torch.stack([first, second]).float() - mean) / std

    manifest = SparseMaskManifest(args.manifest)
    mask = manifest.get_mask(args.mask_id).float()
    obs_mask = mask.unsqueeze(0).expand(2, -1, -1, -1).clone()
    device = torch.device(args.device)
    batch = {
        "x_full": x_full.to(device),
        "x_obs": (x_full * obs_mask).to(device),
        "obs_mask": obs_mask.to(device),
    }

    model = PartialConvMaskedAutoencoder().to(device)
    pretrainer = MaskedReconstructionPretrainer(
        model,
        MaskedReconstructionLoss(
            hidden_weight=1.0,
            observation_weight=0.1,
            history_gradient_weight=0.0,
        ),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    initial = float(
        pretrainer.validate_batch(batch)["losses"]["total"].detach().cpu()
    )
    for _ in range(args.steps):
        pretrainer.train_batch(batch, optimizer, max_grad_norm=1.0)
    final_result = pretrainer.validate_batch(batch)
    final = float(final_result["losses"]["total"].detach().cpu())
    result = {
        "data_file": str(args.data_file),
        "samples": 2,
        "steps": args.steps,
        "initial_total_loss": initial,
        "final_total_loss": final,
        "final_over_initial": final / max(initial, 1.0e-12),
        "reduction_percent": 100.0 * (1.0 - final / max(initial, 1.0e-12)),
        "all_losses_finite": all(
            torch.isfinite(value).all() for value in final_result["losses"].values()
        ),
    }
    print(json.dumps(result, indent=2))
    return 0 if result["all_losses_finite"] and final < initial * 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
