#!/usr/bin/env python3
"""Fit one train-only POD basis and select rank using full validation rollout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from neuralop.data.datasets.sea_surface_simple import _load_mat_array
from neuralop.models.reconstructors import fit_pod_basis_from_snapshots
from neuralop.training.sparse_experiment_runner import (
    build_sparse_pipeline,
    evaluate_sparse_loader,
    move_sparse_batch_to_device,
    prepare_sparse_experiment,
    run_sparse_one_batch,
    sha256_file,
)


LOCKED_RANKS = (8, 16, 32, 64)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite B3 artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_basis_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite POD basis: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _load_train_snapshots(context, device: torch.device) -> tuple[torch.Tensor, list]:
    entries = context.data.split_manifest["splits"]["train"]
    data_root = Path(context.data.split_manifest["data_root"])
    height = int(context.config["data"]["height"])
    width = int(context.config["data"]["width"])
    mean = float(context.data.normalization_mean)
    std = float(context.data.normalization_std)
    chunks = []
    sources = []
    for entry in entries:
        relative_path = Path(str(entry["relative_path"]))
        if not relative_path.parts or relative_path.parts[0].lower() != "train":
            raise PermissionError(
                f"POD basis source is not in frozen train split: {relative_path}"
            )
        path = (data_root / relative_path).resolve()
        array = _load_mat_array(str(path), variable=context.config["data"]["variable"])
        if tuple(array.shape[-2:]) != (height, width):
            raise ValueError(f"Unexpected POD source shape for {path}: {array.shape}")
        tensor = torch.from_numpy(array.copy()).float().reshape(-1, height * width)
        chunks.append((tensor - mean) / std)
        sources.append(
            {
                "source_id": int(entry["source_id"]),
                "name": str(entry["name"]),
                "relative_path": str(entry["relative_path"]),
                "sha256": str(entry["sha256"]),
                "frames": int(array.shape[0]),
            }
        )
    snapshots = torch.cat(chunks, dim=0).to(device)
    return snapshots, sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        default=Path("config/sparse_experiments/b3_pod_frozen.formal.json"),
        nargs="?",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
        help="Frozen B1 data_split.json; only train and val are verified or loaded.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    context = prepare_sparse_experiment(
        args.config,
        project_root=Path.cwd(),
        device=args.device,
        loader_splits=("train", "val"),
        frozen_split_manifest=args.split_manifest,
    )
    if set(context.data.loaders) != {"train", "val"}:
        raise RuntimeError("B3 rank selection must construct train and val loaders only")
    ranks = tuple(int(value) for value in context.config["model"]["pod_ranks"])
    if ranks != LOCKED_RANKS:
        raise ValueError(f"B3 formal ranks are locked to {LOCKED_RANKS}, got {ranks}")

    device = torch.device(context.config["runtime"]["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    basis_started = time.perf_counter()
    snapshots, train_sources = _load_train_snapshots(context, device)
    mean, modes, singular_values = fit_pod_basis_from_snapshots(
        snapshots,
        max_rank=max(ranks),
        seed=int(context.config["runtime"]["seed"]),
        niter=int(context.config["model"].get("pod_pca_niter", 4)),
    )
    basis_seconds = time.perf_counter() - basis_started
    basis_peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    snapshot_count = int(snapshots.shape[0])
    del snapshots
    if device.type == "cuda":
        torch.cuda.empty_cache()

    basis_path = Path(context.config["pipeline"]["pod_basis_path"])
    if basis_path.parent.resolve() != context.run_dir.resolve():
        raise ValueError("Formal POD basis must be stored inside its unique run directory")
    split_hash = sha256_file(context.run_dir / "data_split.json")
    basis_payload = {
        "schema_version": 1,
        "spatial_mean": mean.detach().cpu(),
        "spatial_modes": modes.detach().cpu(),
        "singular_values": singular_values.detach().cpu(),
        "metadata": {
            "algorithm": "torch.pca_lowrank",
            "centered": True,
            "pca_niter": int(context.config["model"].get("pod_pca_niter", 4)),
            "max_rank": max(ranks),
            "height": int(context.config["data"]["height"]),
            "width": int(context.config["data"]["width"]),
            "snapshot_count": snapshot_count,
            "normalization_mean": context.data.normalization_mean,
            "normalization_std": context.data.normalization_std,
            "seed": int(context.config["runtime"]["seed"]),
            "split_sha256": split_hash,
            "train_sources_sha256": _canonical_hash(train_sources),
            "train_sources": train_sources,
            "val_sources_used": [],
            "test_sources_used": [],
        },
    }
    _save_basis_exclusive(basis_path, basis_payload)
    basis_hash = sha256_file(basis_path)
    _write_json_exclusive(
        context.run_dir / "pod_basis_provenance.json",
        {
            "basis_path": str(basis_path),
            "basis_sha256": basis_hash,
            "basis_seconds": basis_seconds,
            "basis_peak_memory_bytes": basis_peak_memory,
            "snapshot_count": snapshot_count,
            "split_sha256": split_hash,
            "train_sources_sha256": _canonical_hash(train_sources),
            "train_source_count": len(train_sources),
            "val_sources_used": [],
            "test_sources_used": [],
        },
    )

    smoke_config = copy.deepcopy(context.config)
    smoke_config["model"]["pod_rank"] = min(ranks)
    smoke_pipeline = build_sparse_pipeline(smoke_config, device)
    smoke_batch = move_sparse_batch_to_device(
        next(iter(context.data.loaders["val"])), device
    )
    smoke_result = run_sparse_one_batch(
        "B3", smoke_pipeline, smoke_batch, rollout_steps=30
    )
    if tuple(smoke_result["history_reconstruction"].shape[1:]) != (60, 64, 64):
        raise RuntimeError("B3 real-batch reconstruction smoke shape failed")
    if tuple(smoke_result["forecast"].shape[1:]) != (30, 64, 64):
        raise RuntimeError("B3 real-batch forecast smoke shape failed")
    if not smoke_pipeline.rfno_gradients_are_none():
        raise RuntimeError("B3 real-batch smoke found RFNO gradients")
    del smoke_pipeline, smoke_result, smoke_batch

    rows = []
    for rank in ranks:
        rank_config = copy.deepcopy(context.config)
        rank_config["model"]["pod_rank"] = rank
        pipeline = build_sparse_pipeline(rank_config, device)
        if not all(not parameter.requires_grad for parameter in pipeline.rfno.parameters()):
            raise RuntimeError("B3 RFNO must remain frozen")
        rank_payloads = {}
        for split in ("train", "val"):
            metrics = evaluate_sparse_loader(
                pipeline,
                context.data.loaders[split],
                normalization_mean=context.data.normalization_mean,
                normalization_std=context.data.normalization_std,
                rollout_steps=int(context.config["data"]["rollout_steps"]),
                forecast_horizons=context.config["evaluation"]["forecast_horizons"],
                frame_interval=float(context.config["data"]["dt"]),
                device=device,
                max_batches=None,
                source_entries=context.data.split_manifest["splits"][split],
            )
            payload = {
                "experiment_id": "B3",
                "split": split,
                "rank": rank,
                "basis_path": str(basis_path),
                "basis_sha256": basis_hash,
                "split_sha256": split_hash,
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
                        not parameter.requires_grad
                        for parameter in pipeline.rfno.parameters()
                    ),
                    "rfno_gradients_none": pipeline.rfno_gradients_are_none(),
                },
                "metrics": metrics,
            }
            _write_json_exclusive(
                context.run_dir
                / "evaluation"
                / f"rank_{rank:04d}_{split}_metrics.json",
                payload,
            )
            rank_payloads[split] = payload
        val_metrics = rank_payloads["val"]["metrics"]
        train_metrics = rank_payloads["train"]["metrics"]
        rows.append(
            {
                "rank": rank,
                "train_history_missing_nrmse": train_metrics[
                    "history_reconstruction"
                ]["missing_region"]["nrmse"],
                "train_forecast_300_nrmse": train_metrics["forecast_300"]["nrmse"],
                "val_history_missing_nrmse": val_metrics["history_reconstruction"][
                    "missing_region"
                ]["nrmse"],
                "val_forecast_300_nrmse": val_metrics["forecast_300"]["nrmse"],
                "val_forecast_300_correlation": val_metrics["forecast_300"][
                    "correlation"
                ],
                "train_inference_seconds": train_metrics["elapsed_seconds"],
                "val_inference_seconds": val_metrics["elapsed_seconds"],
            }
        )
        del pipeline
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected = min(rows, key=lambda row: float(row["val_forecast_300_nrmse"]))
    selection = {
        "experiment_id": "B3",
        "selection_split": "val",
        "primary_metric": "forecast_300_nrmse",
        "primary_mode": "min",
        "locked_ranks": list(ranks),
        "selected_rank": int(selected["rank"]),
        "selected_val_forecast_300_nrmse": selected["val_forecast_300_nrmse"],
        "basis_path": str(basis_path),
        "basis_sha256": basis_hash,
        "split_sha256": split_hash,
        "test_accessed": False,
        "rows": rows,
    }
    _write_json_exclusive(
        context.run_dir / "evaluation" / "rank_selection.json", selection
    )
    print(
        json.dumps(
            {
                "run_dir": str(context.run_dir),
                "basis_sha256": basis_hash,
                "basis_seconds": basis_seconds,
                "snapshot_count": snapshot_count,
                "selected_rank": selection["selected_rank"],
                "selected_val_forecast_300_nrmse": selection[
                    "selected_val_forecast_300_nrmse"
                ],
                "rows": rows,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
