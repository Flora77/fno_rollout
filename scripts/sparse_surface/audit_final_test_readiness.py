#!/usr/bin/env python3
"""Metadata-only GO/NO-GO audit for the locked final test plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neuralop.training.sparse_experiment_runner import (
    SparseCheckpointManager,
    build_sparse_pipeline,
    load_resolved_sparse_config,
    scientific_config_sha256,
    sha256_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path("config/sparse_experiments/final_test_locked_20260722_v1.json"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def audit(manifest_path: Path, project_root: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blockers: list[str] = []
    shared = manifest["shared"]
    run_root = _resolve(project_root, manifest["run_root"])
    result_root = _resolve(project_root, manifest["result_root"])
    if run_root.exists():
        blockers.append(f"locked test run root already exists: {run_root}")
    if result_root.exists():
        blockers.append(f"locked result root already exists: {result_root}")

    mask_manifest = _resolve(project_root, shared["mask_manifest_path"])
    mask_archive = _resolve(project_root, shared["mask_archive_path"])
    for path, expected in (
        (mask_manifest, shared["mask_manifest_sha256"]),
        (mask_archive, shared["mask_archive_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            blockers.append(f"mask artifact hash mismatch: {path}")

    targets = set()
    for method in manifest["methods"]:
        method_id = method["method_id"]
        config_path = _resolve(project_root, method["config_path"])
        checkpoint_path = _resolve(project_root, method["checkpoint_path"])
        split_path = _resolve(project_root, method["split_manifest_path"])
        target = _resolve(project_root, method["test_run_dir"])
        if target in targets:
            blockers.append(f"duplicate test directory: {target}")
        targets.add(target)
        if target.parent != run_root:
            blockers.append(f"test directory escapes locked run root: {target}")
        if target.exists():
            blockers.append(f"test directory already exists: {target}")
        missing_required = False
        for required in (config_path, checkpoint_path, split_path):
            if not required.is_file():
                blockers.append(f"missing {method_id} artifact: {required}")
                missing_required = True
        if missing_required:
            continue

        resolved = load_resolved_sparse_config(
            config_path,
            project_root=project_root,
            run_dir=target,
        )
        config_hash = scientific_config_sha256(resolved)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if config_hash != method["scientific_config_sha256"]:
            blockers.append(f"{method_id} resolved scientific config hash changed")
        if checkpoint.get("scientific_config_sha256") != config_hash:
            blockers.append(f"{method_id} checkpoint strict config mismatch")
        else:
            # Exercise the real strict state-dict restore without constructing a
            # data loader or reading any MAT file.
            pipeline = build_sparse_pipeline(resolved, torch.device("cpu"))
            # Point the read-only manager at the checkpoint's existing parent
            # run. Its constructor creates a checkpoints directory, so using the
            # future target here would accidentally reserve the test directory.
            SparseCheckpointManager(checkpoint_path.parent.parent, config_hash).restore(
                checkpoint_path,
                pipeline,
                restore_random_state=False,
                allow_config_mismatch=False,
            )
        if sha256_file(checkpoint_path) != method["checkpoint_sha256"]:
            blockers.append(f"{method_id} checkpoint file hash mismatch")
        if checkpoint_path.name != "best.pt":
            blockers.append(f"{method_id} checkpoint is not best.pt")
        if sha256_file(split_path) != shared["split_sha256"]:
            blockers.append(f"{method_id} split hash mismatch")
        if resolved["observation"]["mask_id"] != shared["mask_id"]:
            blockers.append(f"{method_id} mask_id mismatch")
        if resolved["normalization"] != shared["normalization"]:
            blockers.append(f"{method_id} normalization mismatch")
        if resolved["evaluation"]["forecast_horizons"] != shared["forecast_horizons"]:
            blockers.append(f"{method_id} forecast horizons mismatch")
        if sha256_file(resolved["pipeline"]["rfno_checkpoint"]) != shared[
            "rfno_checkpoint_sha256"
        ]:
            blockers.append(f"{method_id} B0 RFNO hash mismatch")

    return {
        "decision": "GO" if not blockers else "NO-GO",
        "blockers": blockers,
        "methods": [method["method_id"] for method in manifest["methods"]],
    }


def main() -> int:
    args = _parser().parse_args()
    result = audit(args.manifest.resolve(), args.project_root.resolve())
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
