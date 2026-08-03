#!/usr/bin/env python3
"""Strictly restore every nonlinear B5 v2 best/last checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from neuralop.training.sparse_experiment_runner import (
    build_sparse_pipeline,
    prepare_sparse_experiment,
)
from neuralop.training.sparse_multiepoch import SparseMultiEpochTrainer


RUNS = tuple(
    (method, seed)
    for method in ("gno", "gino")
    for seed in (42, 43, 44)
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_equal(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> bool:
    return first.keys() == second.keys() and all(
        torch.equal(first[key].detach().cpu(), second[key].detach().cpu())
        for key in first
    )


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    split = root / "data" / "splits" / "sea_surface_bimodal_v1.json"
    rows: list[dict[str, object]] = []
    for method, seed in RUNS:
        label = f"b5_v2_{method}_seed{seed}"
        config = (
            root
            / "config"
            / "sparse_experiments"
            / f"b5_v2_{method}_frozen_seed{seed}.formal.json"
        )
        run_dir = root / "runs" / "sparse" / f"{label}_formal_20260729_v1"
        audit_dir = (
            root / "runs" / "sparse" / f"{label}_checkpoint_audit_20260730_v2"
        )
        context = prepare_sparse_experiment(
            config,
            project_root=root,
            run_dir=audit_dir,
            device="cuda",
            loader_splits=("train", "val"),
            frozen_split_manifest=split,
        )
        device = torch.device(context.config["runtime"]["device"])
        pipeline = build_sparse_pipeline(context.config, device)
        trainer = SparseMultiEpochTrainer(context, pipeline, device)
        rfno_initial = {
            key: value.detach().cpu().clone()
            for key, value in pipeline.rfno.state_dict().items()
            if isinstance(value, torch.Tensor)
        }

        restored: dict[str, object] = {}
        for name in ("best", "last"):
            path = run_dir / "checkpoints" / f"{name}.pt"
            result = trainer.checkpoints.restore(
                path,
                pipeline,
                trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                data_loader_generator=context.data.generators["train"],
                restore_random_state=True,
                require_training_state=True,
                strict=True,
            )
            restored[name] = {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "epoch": result["epoch"],
                "global_step": result["global_step"],
                "restored_components": result["restored_components"],
                "rfno_unchanged_from_b0": _state_equal(
                    rfno_initial,
                    {
                        key: value.detach().cpu()
                        for key, value in pipeline.rfno.state_dict().items()
                        if isinstance(value, torch.Tensor)
                    },
                ),
            }
        rows.append(
            {
                "method": method,
                "seed": seed,
                "scientific_config_sha256": context.config_sha256,
                "loader_splits": sorted(context.data.loaders),
                "train_samples": len(context.data.sparse_datasets["train"]),
                "val_samples": len(context.data.sparse_datasets["val"]),
                "restored": restored,
            }
        )

    passed = all(
        row["loader_splits"] == ["train", "val"]
        and row["train_samples"] == 1221
        and row["val_samples"] == 777
        and all(
            item["rfno_unchanged_from_b0"]
            and all(item["restored_components"].values())
            for item in row["restored"].values()
        )
        for row in rows
    )
    payload = {"passed": passed, "runs": rows}
    output = (
        root
        / "results"
        / "min_b5_gno_gino_v2_three_seed_20260729_v1"
        / "checkpoint_restore_audit.json"
    )
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": passed, "runs": len(rows)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
