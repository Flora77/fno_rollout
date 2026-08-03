"""Tiny synthetic overfit check for the nonlinear periodic B5 reconstructors.

This is an implementation smoke test, not a scientific training entry. It uses two
smooth periodic samples, never reads repository train/validation data, and defaults
to CPU.
"""

from __future__ import annotations

import argparse

import torch

from neuralop.models.reconstructors import (
    PeriodicGINOReconstructor,
    PeriodicGNOReconstructor,
)
from neuralop.training.masked_reconstruction import MaskedReconstructionLoss


def _two_periodic_samples(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.arange(size, dtype=torch.float32) / float(size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    time = torch.arange(60, dtype=torch.float32)[:, None, None] / 60.0
    samples = []
    for sample_index in range(2):
        phase = 0.17 * sample_index
        field = (
            torch.sin(2.0 * torch.pi * (xx[None] + 0.30 * time + phase))
            + 0.55
            * torch.cos(2.0 * torch.pi * (yy[None] - 0.20 * time - phase))
            + 0.20
            * torch.sin(
                2.0
                * torch.pi
                * (xx[None] + yy[None] + 0.10 * time + 0.5 * phase)
            )
        )
        samples.append(field)
    full = torch.stack(samples)
    mask = torch.zeros_like(full)
    mask[:, :, ::2, ::2] = 1.0
    return full, mask


def _build_model(name: str, hidden_channels: int) -> torch.nn.Module:
    common = {
        "input_steps": 60,
        "hidden_channels": hidden_channels,
        "radius": 0.6,
        "mlp_channels": (32, 32),
        "kernel_mode": "channelwise_nonlinear",
        "hard_observation_consistency": True,
    }
    if name == "gno":
        return PeriodicGNOReconstructor(**common)
    return PeriodicGINOReconstructor(
        **common,
        latent_shape=(8, 8),
        n_modes=(4, 4),
        fno_layers=2,
        latent_positional_embedding="none",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("gno", "gino"), default="gno")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-final-ratio", type=float, default=0.75)
    args = parser.parse_args()
    if args.steps <= 0 or args.size < 4 or args.hidden_channels <= 0:
        raise ValueError("steps/hidden-channels must be positive and size must be >= 4")

    torch.manual_seed(20260729)
    device = torch.device(args.device)
    full, mask = (value.to(device) for value in _two_periodic_samples(args.size))
    observed = full * mask
    model = _build_model(args.model, args.hidden_channels).to(device)
    criterion = MaskedReconstructionLoss(
        hidden_weight=1.0,
        observation_weight=0.1,
        history_gradient_weight=0.2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    with torch.no_grad():
        initial = float(criterion(model(observed, mask), full, mask)["total"])
    final = initial
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(observed, mask)
        losses = criterion(reconstruction, full, mask)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = float(losses["total"].detach())

    ratio = final / max(initial, 1.0e-12)
    print(
        {
            "model": args.model,
            "steps": args.steps,
            "initial_loss": initial,
            "final_loss": final,
            "final_ratio": ratio,
        }
    )
    if not torch.isfinite(torch.tensor(final)):
        raise FloatingPointError("Two-sample overfit produced a non-finite loss")
    if ratio > args.max_final_ratio:
        raise RuntimeError(
            f"Two-sample overfit ratio {ratio:.4f} exceeds "
            f"{args.max_final_ratio:.4f}"
        )


if __name__ == "__main__":
    main()
