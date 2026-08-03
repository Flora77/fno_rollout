#!/usr/bin/env python3
"""Run formal or limited smoke training for learned sparse experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from neuralop.training.sparse_experiment_runner import (
    build_sparse_pipeline,
    load_resolved_sparse_config,
    prepare_sparse_experiment,
    validate_continuation_checkpoint,
)
from neuralop.training.sparse_multiepoch import SparseMultiEpochTrainer


def _save_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _seed_runtime(seed: int) -> None:
    """Make runtime.seed control fresh model initialization and stochastic ops."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume", type=Path, default=None)
    continuation.add_argument(
        "--continue-from",
        type=Path,
        default=None,
        help=(
            "Create a new curriculum run from a complete parent checkpoint while "
            "strictly restoring optimizer/scheduler/scaler/RNG state."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    configured_manifest = raw_config.get("data", {}).get("split_manifest_path")
    if (
        args.resume is None
        and args.continue_from is None
        and args.split_manifest is None
        and not configured_manifest
    ):
        raise ValueError("New formal training requires --split-manifest")
    smoke_limits = (
        args.max_epochs,
        args.max_train_batches,
        args.max_val_batches,
    )
    if args.smoke:
        max_epochs = args.max_epochs or 1
        max_train_batches = args.max_train_batches or 1
        max_val_batches = args.max_val_batches or 1
    else:
        if any(value is not None for value in smoke_limits):
            raise ValueError("Training limits require explicit --smoke")
        max_epochs = None
        max_train_batches = None
        max_val_batches = None

    continuation_metadata = None
    frozen_split_manifest = args.split_manifest
    if args.continue_from is not None:
        candidate_config = load_resolved_sparse_config(
            args.config,
            project_root=args.project_root,
            run_dir=args.run_dir,
            device=args.device,
        )
        configured_continuation = candidate_config["pipeline"].get(
            "continuation_checkpoint"
        )
        if configured_continuation is None:
            raise ValueError(
                "--continue-from requires pipeline.continuation_checkpoint in config"
            )
        if Path(configured_continuation).resolve() != args.continue_from.resolve():
            raise ValueError(
                "--continue-from differs from pipeline.continuation_checkpoint"
            )
        continuation_metadata = validate_continuation_checkpoint(
            candidate_config, args.continue_from
        )
        frozen_split_manifest = Path(continuation_metadata["parent_split_path"])

    context = prepare_sparse_experiment(
        args.config,
        project_root=args.project_root,
        run_dir=args.run_dir,
        device=args.device,
        resume_checkpoint=args.resume,
        loader_splits=("train", "val"),
        frozen_split_manifest=frozen_split_manifest,
        training_mode=True,
    )
    experiment_id = str(context.config["experiment_id"]).upper()
    forecast_supervision = bool(
        context.config.get("training", {}).get("forecast_supervision", False)
    )
    target_flags = {
        split: bool(dataset.include_target)
        for split, dataset in context.data.sparse_datasets.items()
    }
    if any(flag != forecast_supervision for flag in target_flags.values()):
        raise RuntimeError(
            "Formal training target policy differs from forecast_supervision: "
            f"{target_flags}"
        )
    learned_ids = {
        "B4", "B5", "B5-GNO", "B5-GINO", "B6",
        "H1", "H1-A1", "H1-A2", "H1-A3", "H1-A4", "H1-A5", "H1-A6",
        "P1", "P2", "V0", "V1", "V2", "V3", "V4", "PC-D1", "PC-D2",
        "ST-D0", "MU-V3", "FD-R1", "FD-A1", "FD-L1",
    }
    if experiment_id not in learned_ids:
        raise ValueError(f"Formal sparse training supports {sorted(learned_ids)}")

    _seed_runtime(int(context.config["runtime"]["seed"]))
    device = torch.device(context.config["runtime"]["device"])
    pipeline = build_sparse_pipeline(context.config, device)
    trainer = SparseMultiEpochTrainer(context, pipeline, device)
    summary = trainer.fit(
        resume_checkpoint=args.resume,
        continuation_checkpoint=args.continue_from,
        max_epochs_this_run=max_epochs,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        run_kind="smoke" if args.smoke else "formal",
    )
    summary["training_data_contract"] = {
        "forecast_supervision": forecast_supervision,
        "future_target_in_loader": forecast_supervision,
        "model_inputs": ["x_obs", "obs_mask"],
        "x_full_role": "reconstruction_supervision_only",
    }
    if continuation_metadata is not None:
        summary["continuation_source"] = continuation_metadata
    prefix = "smoke_training_summary" if args.smoke else "training_summary"
    output_name = f"{prefix}_epoch{int(summary['last_epoch']):04d}.json"
    _save_exclusive(context.run_dir / output_name, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
