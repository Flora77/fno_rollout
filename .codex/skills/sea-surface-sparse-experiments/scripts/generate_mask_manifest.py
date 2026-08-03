#!/usr/bin/env python3
"""Generate deterministic sparse-observation masks and a JSON manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


MASK_TYPES = {"fixed_points", "random_points", "blocks", "stripes"}


def _parse_csv(text: str, cast):
    values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _make_exact(mask: np.ndarray, target: int, rng: np.random.Generator) -> np.ndarray:
    flat = mask.reshape(-1)
    current = int(flat.sum())
    if current > target:
        indices = np.flatnonzero(flat)
        flat[rng.choice(indices, size=current - target, replace=False)] = False
    elif current < target:
        indices = np.flatnonzero(~flat)
        flat[rng.choice(indices, size=target - current, replace=False)] = True
    return mask


def fixed_points(t: int, h: int, w: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    count = max(1, min(h * w, round(rate * h * w)))
    spatial = np.zeros(h * w, dtype=bool)
    spatial[rng.choice(h * w, size=count, replace=False)] = True
    return np.broadcast_to(spatial.reshape(1, h, w), (t, h, w)).copy()


def random_points(t: int, h: int, w: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    count = max(1, min(h * w, round(rate * h * w)))
    result = np.zeros((t, h * w), dtype=bool)
    for frame in range(t):
        result[frame, rng.choice(h * w, size=count, replace=False)] = True
    return result.reshape(t, h, w)


def blocks(t: int, h: int, w: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    target = max(1, min(h * w, round(rate * h * w)))
    spatial = np.zeros((h, w), dtype=bool)
    max_h = max(1, h // 3)
    max_w = max(1, w // 3)
    while int(spatial.sum()) < target:
        block_h = int(rng.integers(1, max_h + 1))
        block_w = int(rng.integers(1, max_w + 1))
        row = int(rng.integers(0, h))
        col = int(rng.integers(0, w))
        rows = np.arange(row, row + block_h) % h
        cols = np.arange(col, col + block_w) % w
        spatial[np.ix_(rows, cols)] = True
    spatial = _make_exact(spatial, target, rng)
    return np.broadcast_to(spatial[None], (t, h, w)).copy()


def stripes(t: int, h: int, w: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    target = max(1, min(h * w, round(rate * h * w)))
    spatial = np.zeros((h, w), dtype=bool)
    orientation = "rows" if rng.random() < 0.5 else "cols"
    order = rng.permutation(h if orientation == "rows" else w)
    for index in order:
        if orientation == "rows":
            spatial[index, :] = True
        else:
            spatial[:, index] = True
        if int(spatial.sum()) >= target:
            break
    spatial = _make_exact(spatial, target, rng)
    return np.broadcast_to(spatial[None], (t, h, w)).copy()


GENERATORS = {
    "fixed_points": fixed_points,
    "random_points": random_points,
    "blocks": blocks,
    "stripes": stripes,
}


def apply_temporal_dropout(
    mask: np.ndarray, probability: float, rng: np.random.Generator
) -> np.ndarray:
    if probability <= 0.0:
        return mask
    keep = rng.random(mask.shape) >= probability
    return mask & keep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="output .npz path")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--time-steps", type=int, default=60)
    parser.add_argument("--rates", default="0.01,0.02,0.05,0.10,0.20")
    parser.add_argument("--mask-types", default="fixed_points,random_points,blocks,stripes")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temporal-dropout", type=float, default=0.0)
    args = parser.parse_args()

    if min(args.height, args.width, args.time_steps, args.replicates) <= 0:
        parser.error("height, width, time-steps, and replicates must be positive")
    rates = _parse_csv(args.rates, float)
    if any(not 0.0 < rate <= 1.0 for rate in rates):
        parser.error("all observation rates must be in (0, 1]")
    mask_types = _parse_csv(args.mask_types, str)
    unknown = sorted(set(mask_types) - MASK_TYPES)
    if unknown:
        parser.error(f"unsupported mask types: {unknown}; choose from {sorted(MASK_TYPES)}")
    if not 0.0 <= args.temporal_dropout < 1.0:
        parser.error("temporal-dropout must be in [0, 1)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, object]] = []
    index = 0
    for mask_type in mask_types:
        for rate in rates:
            for replicate in range(args.replicates):
                sequence = np.random.SeedSequence(
                    [args.seed, mask_types.index(mask_type), round(rate * 1_000_000), replicate]
                )
                rng = np.random.default_rng(sequence)
                mask = GENERATORS[mask_type](
                    args.time_steps, args.height, args.width, rate, rng
                )
                mask = apply_temporal_dropout(mask, args.temporal_dropout, rng)
                key = f"mask_{index:05d}"
                mask_uint8 = mask.astype(np.uint8)
                arrays[key] = mask_uint8
                digest = hashlib.sha256(mask_uint8.tobytes()).hexdigest()
                records.append(
                    {
                        "mask_id": key,
                        "mask_type": mask_type,
                        "requested_observation_rate": rate,
                        "effective_observation_rate": float(mask_uint8.mean()),
                        "replicate": replicate,
                        "seed": args.seed,
                        "temporal_dropout": args.temporal_dropout,
                        "shape": list(mask_uint8.shape),
                        "sha256": digest,
                    }
                )
                index += 1

    np.savez_compressed(args.output, **arrays)
    manifest_path = args.output.with_suffix(".json")
    manifest = {
        "version": 1,
        "npz_file": args.output.name,
        "height": args.height,
        "width": args.width,
        "time_steps": args.time_steps,
        "base_seed": args.seed,
        "entries": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} masks: {args.output}")
    print(f"wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
