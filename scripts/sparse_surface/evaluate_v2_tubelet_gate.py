#!/usr/bin/env python3
"""Evaluate the seed-42 V2 Tubelet-MAE gate on validation only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from neuralop.evaluation.sparse_surface import SparseHistoryMetricAccumulator
from neuralop.training.sparse_experiment_runner import (
    build_sparse_dataloaders,
    build_sparse_pipeline,
    load_frozen_data_split,
    load_resolved_sparse_config,
    move_sparse_batch_to_device,
)


RUNS = {
    "V1": {
        "config": "config/sparse_experiments/v1_vit_mae_sparse_finetune_seed42.formal.json",
        "downstream": "runs/sparse/v1_vit_mae_pretrained_sparse_finetune_seed42_30ep_formal_20260729_v1",
        "pretraining": "runs/sparse/v1_spatial_mae_pretrain_patch75_seed42_formal_20260728_v1",
        "mask_ratio": 0.75,
        "tokenization": "spatial_patch",
    },
    "V2-75": {
        "config": "config/sparse_experiments/v2_tubelet_mae_sparse_finetune_seed42.formal.json",
        "downstream": "runs/sparse/v2_tubelet_mae_pretrained_sparse_finetune_patch75_seed42_30ep_formal_20260729_v1",
        "pretraining": "runs/sparse/v2_tubelet_mae_pretrain_patch75_seed42_formal_20260729_v1",
        "mask_ratio": 0.75,
        "tokenization": "tubelet",
    },
    "V2-90": {
        "config": "config/sparse_experiments/v2_a1_tubelet_mae_sparse_finetune_mask90_seed42.formal.json",
        "downstream": "runs/sparse/v2_a1_tubelet_mae_pretrained_sparse_finetune_patch90_seed42_30ep_formal_20260729_v1",
        "pretraining": "runs/sparse/v2_a1_tubelet_mae_pretrain_patch90_seed42_formal_20260729_v1",
        "mask_ratio": 0.90,
        "tokenization": "tubelet",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def summary(run_dir: Path) -> dict[str, Any]:
    paths = list(run_dir.glob("training_summary_epoch*.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one training summary in {run_dir}")
    return read_json(paths[0])


def all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    return True


def stability(run_dir: Path, training: Mapping[str, Any]) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in (run_dir / "logs" / "epochs.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    overflow_batches = sum(
        float(record["train"].get("amp_overflow_batches", 0.0))
        for record in records
    )
    stable = (
        training.get("safety_stop_reason") is None
        and all_finite(records)
        and int(training["completed_epochs"]) > 0
        and int(training["last_epoch"]) == int(records[-1]["epoch"])
    )
    return {
        "stable": stable,
        "finite_all_epoch_records": all_finite(records),
        "safety_stop_reason": training.get("safety_stop_reason"),
        "amp_overflow_batches_total": overflow_batches,
    }


def load_checkpoint(
    pipeline: torch.nn.Module, path: Path, device: torch.device
) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    pipeline.load_state_dict(checkpoint["model_state_dict"], strict=True)
    pipeline.eval()
    return checkpoint


@torch.no_grad()
def evaluate(
    root: Path, output: Path, label: str, spec: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    run_dir = root / str(spec["downstream"])
    pretrain_dir = root / str(spec["pretraining"])
    train_summary = summary(run_dir)
    pretrain_summary = summary(pretrain_dir)
    if (
        int(train_summary["start_epoch"]) != 1
        or int(train_summary["last_epoch"]) != 30
        or int(train_summary["global_step"]) != 18_330
        or bool(train_summary["stopped_early"])
    ):
        raise RuntimeError(f"Invalid downstream budget: {label}")

    config = load_resolved_sparse_config(
        root / str(spec["config"]),
        project_root=root,
        run_dir=output / f"_resolve_{label.lower()}",
        device=str(device),
    )
    split = load_frozen_data_split(
        run_dir / "data_split.json", config, verify_splits=("val",)
    )
    data = build_sparse_dataloaders(config, split, splits=("val",))
    if set(data.loaders) != {"val"}:
        raise RuntimeError("Non-validation loader was constructed")

    pipeline = build_sparse_pipeline(config, device)
    checkpoint = load_checkpoint(
        pipeline, run_dir / "checkpoints" / "best.pt", device
    )
    reconstructor = pipeline.reconstructor
    rfno_frozen = all(not parameter.requires_grad for parameter in pipeline.rfno.parameters())
    rfno_gradients_none = all(parameter.grad is None for parameter in pipeline.rfno.parameters())
    optimizer_ids = {
        int(parameter_id)
        for group in checkpoint["optimizer_state_dict"]["param_groups"]
        for parameter_id in group["params"]
    }
    reconstructor_tensors = sum(
        1 for parameter in reconstructor.parameters() if parameter.requires_grad
    )
    optimizer_excludes_rfno = (
        len(checkpoint["optimizer_state_dict"]["param_groups"]) == 1
        and len(optimizer_ids) == reconstructor_tensors
    )
    if not (rfno_frozen and rfno_gradients_none and optimizer_excludes_rfno):
        raise RuntimeError(f"Frozen RFNO contract failed: {label}")

    metrics = SparseHistoryMetricAccumulator(
        normalization_mean=data.normalization_mean,
        normalization_std=data.normalization_std,
    )
    samples = 0
    inference_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for raw_batch in data.loaders["val"]:
        batch = move_sparse_batch_to_device(raw_batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            reconstruction = reconstructor(batch["x_obs"], batch["obs_mask"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - started
        if tuple(reconstruction.shape[1:]) != (60, 64, 64):
            raise RuntimeError("Invalid reconstruction shape")
        if not torch.isfinite(reconstruction).all():
            raise FloatingPointError("Non-finite reconstruction")
        metrics.update(
            reconstruction,
            batch["x_full"],
            batch["obs_mask"],
            source_ids=batch["source_id"],
        )
        samples += int(reconstruction.shape[0])
    result = metrics.compute()
    if samples != 777 or int(result["source_count"]) != 7:
        raise RuntimeError("Frozen validation split mismatch")

    downstream_stability = stability(run_dir, train_summary)
    pretraining_stability = stability(pretrain_dir, pretrain_summary)
    return {
        "label": label,
        "seed": 42,
        "tokenization": spec["tokenization"],
        "pretraining_mask_ratio": spec["mask_ratio"],
        "pretraining_epochs": int(pretrain_summary["last_epoch"]),
        "downstream_epochs": int(train_summary["last_epoch"]),
        "missing_nrmse": float(result["missing_region"]["nrmse"]),
        "missing_correlation": float(result["missing_region"]["correlation"]),
        "full_ssp": float(result["full"]["ssp"]),
        "full_gradient_nrmse": float(result["full"]["gradient_nrmse"]),
        "reconstructor_parameters": sum(p.numel() for p in reconstructor.parameters()),
        "pipeline_parameters": sum(p.numel() for p in pipeline.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in pipeline.parameters() if p.requires_grad
        ),
        "pretraining_seconds": float(pretrain_summary["elapsed_seconds"]),
        "pretraining_peak_memory_mb": float(
            pretrain_summary["peak_memory_allocated_mb"]
        ),
        "downstream_seconds": float(train_summary["elapsed_seconds"]),
        "downstream_peak_memory_mb": float(
            train_summary["peak_memory_allocated_mb"]
        ),
        "inference_seconds": inference_seconds,
        "inference_ms_per_sample": 1000.0 * inference_seconds / samples,
        "evaluation_peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024.0**2
            if device.type == "cuda"
            else 0.0
        ),
        "pretraining_stability": pretraining_stability,
        "downstream_stability": downstream_stability,
        "rfno_frozen": rfno_frozen,
        "rfno_gradients_none": rfno_gradients_none,
        "optimizer_excludes_rfno": optimizer_excludes_rfno,
        "per_source": result["per_source"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)

    rows = [
        evaluate(root, output, label, spec, device)
        for label, spec in RUNS.items()
    ]
    v1 = rows[0]
    comparisons = []
    for row in rows[1:]:
        competitive = (
            row["missing_nrmse"] < v1["missing_nrmse"]
            and row["pretraining_stability"]["stable"]
            and row["downstream_stability"]["stable"]
        )
        comparisons.append(
            {
                "label": row["label"],
                "missing_nrmse_minus_v1": row["missing_nrmse"]
                - v1["missing_nrmse"],
                "stable": bool(
                    row["pretraining_stability"]["stable"]
                    and row["downstream_stability"]["stable"]
                ),
                "competitive_with_v1": competitive,
            }
        )
    extend_seeds = any(row["competitive_with_v1"] for row in comparisons)
    tokenization = {
        "V1-75": {
            "token_shape": "60x4x4_spatial_patch_time_as_channels",
            "total_tokens": 256,
            "visible_tokens": 64,
        },
        "V2-75": {
            "token_shape": "10x4x4_tubelet",
            "total_tokens": 1536,
            "visible_tokens": 384,
        },
        "V2-90": {
            "token_shape": "10x4x4_tubelet",
            "total_tokens": 1536,
            "visible_tokens": 154,
        },
        "v2_90_vs_v2_75_only_pretraining_change": "mae_mask_ratio:0.75->0.90",
    }
    payload = {
        "experiment_id": "V2,V2-A1",
        "split": "val",
        "test_accessed": False,
        "tokenization": tokenization,
        "results": rows,
        "gate_comparisons": comparisons,
        "extend_seed43_44": extend_seeds,
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fields = [
        key
        for key, value in rows[0].items()
        if not isinstance(value, (dict, list))
    ]
    with (output / "summary.csv").open(
        "x", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
