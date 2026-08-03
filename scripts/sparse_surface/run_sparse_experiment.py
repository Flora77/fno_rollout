#!/usr/bin/env python3
"""Prepare, dry-run, or evaluate a frozen sparse sea-surface experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from neuralop.training.sparse_experiment_runner import (
    SparseCheckpointManager,
    build_sparse_pipeline,
    coupled_training_config,
    evaluate_sparse_loader,
    move_sparse_batch_to_device,
    prepare_sparse_experiment,
    run_sparse_one_batch,
    sha256_file,
    summarize_dry_run,
)


LEARNED_EXPERIMENTS = frozenset(
    {
        "B4", "B5", "B5-GNO", "B5-GINO", "B6",
        "H1", "H1-A1", "H1-A2", "H1-A3", "H1-A4", "H1-A5", "H1-A6",
        "P1", "P2", "V0", "V1", "V2", "V3", "V4", "PC-D1", "PC-D2",
        "ST-D0", "MU-V3", "FD-R1", "FD-A1", "FD-L1",
    }
)


def _save_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume", type=Path, default=None)
    continuation.add_argument("--existing-run", type=Path, default=None)
    continuation.add_argument(
        "--evaluation-checkpoint",
        type=Path,
        default=None,
        help=(
            "Load trained model weights into a new evaluation-only configuration. "
            "This permits observation-mask changes for validation OOD studies while "
            "still requiring a strict architecture/state-dict match."
        ),
    )
    parser.add_argument(
        "--allow-validation-protocol-mismatch",
        action="store_true",
        help=(
            "Explicitly permit validation-only OOD configuration changes. "
            "Formal in-distribution validation must omit this flag and pass "
            "the strict checkpoint scientific-config hash check."
        ),
    )
    parser.add_argument("--device", default=None)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--evaluate-split", choices=("val", "test"))
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="Dry-run defaults to 30; formal evaluation defaults to config rollout_steps.",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required before touching the test split.",
    )
    return parser


def _configured_experiment_id(config_path: Path) -> str:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Sparse experiment config root must be an object")
    return str(payload.get("experiment_id", "")).upper()


def _evaluation_output_name(
    split: str, *, rollout_steps: int, formal_rollout_steps: int, max_batches: int | None
) -> str:
    if rollout_steps == formal_rollout_steps and max_batches is None:
        return f"{split}_metrics.json"
    batch_label = "all" if max_batches is None else str(max_batches)
    return f"{split}_metrics_partial_rollout{rollout_steps}_max{batch_label}.json"


def main() -> int:
    args = _parser().parse_args()
    experiment_id = _configured_experiment_id(args.config)
    if args.evaluate_split == "test" and not args.allow_test:
        raise PermissionError("Test evaluation requires explicit --allow-test")
    if args.evaluate_split == "test":
        if args.resume is not None or args.existing_run is not None:
            raise PermissionError(
                "Final test must use --evaluation-checkpoint in a new unique run; "
                "--resume/--existing-run would reuse the training directory"
            )
        if args.run_dir is None:
            raise PermissionError("Final test requires an explicit unique --run-dir")
        if args.run_dir.resolve().exists():
            raise FileExistsError(
                f"Final test run directory already exists: {args.run_dir.resolve()}"
            )
        if args.split_manifest is None:
            raise PermissionError(
                "Final test requires the frozen training --split-manifest"
            )
    if args.evaluation_checkpoint is not None:
        if args.evaluate_split not in {"val", "test"}:
            raise PermissionError(
                "--evaluation-checkpoint requires validation or final test evaluation"
            )
        if args.evaluation_checkpoint.name != "best.pt":
            raise PermissionError(
                "--evaluation-checkpoint requires an explicitly selected best.pt"
            )
    if args.allow_validation_protocol_mismatch and (
        args.evaluation_checkpoint is None or args.evaluate_split != "val"
    ):
        raise PermissionError(
            "--allow-validation-protocol-mismatch is restricted to explicit "
            "validation-only checkpoint evaluation"
        )
    trained_checkpoint = args.resume or args.evaluation_checkpoint
    if args.evaluate_split is not None and experiment_id in LEARNED_EXPERIMENTS:
        if trained_checkpoint is None:
            raise PermissionError(
                f"{experiment_id} evaluation requires an explicit trained checkpoint "
                "via --resume or --evaluation-checkpoint; --existing-run alone does "
                "not restore learned weights"
            )
        if args.evaluate_split == "test" and (
            args.evaluation_checkpoint is None
            or args.evaluation_checkpoint.name != "best.pt"
        ):
            raise PermissionError(
                "Learned test evaluation requires --evaluation-checkpoint .../best.pt"
            )
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")
    if args.evaluate_split == "test" and args.max_batches is not None:
        raise ValueError("Formal test evaluation cannot use --max-batches")

    loader_splits = (
        ("train", "val") if args.dry_run else (str(args.evaluate_split),)
    )

    context = prepare_sparse_experiment(
        args.config,
        project_root=args.project_root,
        run_dir=args.run_dir,
        device=args.device,
        resume_checkpoint=args.resume,
        existing_run_dir=args.existing_run,
        loader_splits=loader_splits,
        frozen_split_manifest=args.split_manifest,
    )
    device = torch.device(context.config["runtime"]["device"])
    pipeline = build_sparse_pipeline(context.config, device)
    restore_result = None
    if trained_checkpoint is not None:
        restore_result = SparseCheckpointManager(
            context.run_dir, context.config_sha256
        ).restore(
            trained_checkpoint,
            pipeline,
            restore_random_state=False,
            allow_config_mismatch=args.allow_validation_protocol_mismatch,
        )

    if args.dry_run:
        rollout_steps = int(args.rollout_steps or 30)
        raw_batch = next(iter(context.data.loaders["val"]))
        batch = move_sparse_batch_to_device(raw_batch, device)
        experiment_id = str(context.config["experiment_id"])
        training_config = (
            None
            if experiment_id == "B1"
            else coupled_training_config(context.config)
        )
        result = run_sparse_one_batch(
            experiment_id,
            pipeline,
            batch,
            rollout_steps=rollout_steps,
            training_config=training_config,
            backward=False,
        )
        summary = summarize_dry_run(context, pipeline, result)
        _save_exclusive(context.run_dir / "dry_run.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    split = str(args.evaluate_split)
    rollout_steps = int(
        args.rollout_steps or context.config["data"]["rollout_steps"]
    )
    formal_rollout_steps = int(context.config["data"]["rollout_steps"])
    if rollout_steps <= 0 or rollout_steps > formal_rollout_steps:
        raise ValueError(
            f"--rollout-steps must be in [1,{formal_rollout_steps}], got {rollout_steps}"
        )
    if split == "test" and rollout_steps != formal_rollout_steps:
        raise ValueError("Formal test evaluation must use the configured 300-frame rollout")
    formal = rollout_steps == formal_rollout_steps and args.max_batches is None
    if experiment_id in LEARNED_EXPERIMENTS and formal:
        progress = (restore_result or {}).get("curriculum_state", {})
        if progress.get("run_kind") != "formal":
            raise PermissionError(
                "Formal learned evaluation rejects smoke or unclassified checkpoints"
            )
    metrics = evaluate_sparse_loader(
        pipeline,
        context.data.loaders[split],
        normalization_mean=context.data.normalization_mean,
        normalization_std=context.data.normalization_std,
        rollout_steps=rollout_steps,
        forecast_horizons=context.config["evaluation"]["forecast_horizons"],
        frame_interval=float(context.config["data"]["dt"]),
        device=device,
        max_batches=args.max_batches,
        source_entries=context.data.split_manifest["splits"][split],
    )
    evaluated_checkpoint = (
        Path(trained_checkpoint).resolve()
        if trained_checkpoint is not None
        else Path(context.config["pipeline"]["rfno_checkpoint"])
    )
    payload = {
        "experiment_id": context.config["experiment_id"],
        "experiment_name": context.config["experiment_name"],
        "split": split,
        "evaluation_kind": "formal" if formal else "partial",
        "scientific_config_sha256": context.config_sha256,
        "evaluated_checkpoint": {
            "path": str(evaluated_checkpoint),
            "sha256": sha256_file(evaluated_checkpoint),
        },
        "checkpoint_restore": restore_result,
        "checkpoint_restore_mode": (
            "explicit_validation_ood_protocol_mismatch"
            if args.allow_validation_protocol_mismatch
            else "strict_new_test_run"
            if args.evaluation_checkpoint is not None and split == "test"
            else "strict_scientific_config_hash"
            if args.evaluation_checkpoint is not None
            else "same_config"
        ),
        "parameters": {
            "total": sum(parameter.numel() for parameter in pipeline.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in pipeline.parameters()
                if parameter.requires_grad
            ),
        },
        "freeze_checks": {
            "rfno_requires_grad_false": all(
                not parameter.requires_grad for parameter in pipeline.rfno.parameters()
            ),
            "rfno_gradients_none": all(
                parameter.grad is None for parameter in pipeline.rfno.parameters()
            ),
        },
        "metrics": metrics,
    }
    output_path = context.run_dir / "evaluation" / _evaluation_output_name(
        split,
        rollout_steps=rollout_steps,
        formal_rollout_steps=formal_rollout_steps,
        max_batches=args.max_batches,
    )
    _save_exclusive(output_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
