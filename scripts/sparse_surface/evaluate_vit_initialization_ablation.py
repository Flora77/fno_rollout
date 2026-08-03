#!/usr/bin/env python3
"""Evaluate the paired V0/V1 spatial-ViT initialization ablation on validation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import torch
from scipy.stats import t as student_t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuralop.evaluation.sparse_surface import SparseHistoryMetricAccumulator
from neuralop.training.sparse_experiment_runner import (
    build_sparse_dataloaders,
    build_sparse_pipeline,
    load_frozen_data_split,
    load_resolved_sparse_config,
    move_sparse_batch_to_device,
)


SEEDS = (42, 43, 44)
VARIANTS = ("V0", "V1")
METRIC_DIRECTIONS = {
    "missing_nrmse": "lower",
    "missing_correlation": "higher",
    "full_ssp": "lower",
    "full_gradient_nrmse": "lower",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _config_path(root: Path, variant: str, seed: int) -> Path:
    config_root = root / "config" / "sparse_experiments"
    if variant == "V0":
        if seed == 42:
            return config_root / "v0_vit_random_sparse_finetune.example.json"
        return config_root / f"v0_vit_random_sparse_finetune_seed{seed}.formal.json"
    return config_root / f"v1_vit_mae_sparse_finetune_seed{seed}.formal.json"


def _run_dir(root: Path, variant: str, seed: int) -> Path:
    prefix = (
        "v0_vit_random_sparse_finetune"
        if variant == "V0"
        else "v1_vit_mae_pretrained_sparse_finetune"
    )
    return (
        root
        / "runs"
        / "sparse"
        / f"{prefix}_seed{seed}_30ep_formal_20260729_v1"
    )


def _flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        "missing_nrmse": float(metrics["missing_region"]["nrmse"]),
        "missing_correlation": float(metrics["missing_region"]["correlation"]),
        "full_ssp": float(metrics["full"]["ssp"]),
        "full_gradient_nrmse": float(metrics["full"]["gradient_nrmse"]),
    }


def _load_best_pipeline(
    config: Mapping[str, Any], run_dir: Path, device: torch.device
) -> tuple[torch.nn.Module, Mapping[str, Any]]:
    pipeline = build_sparse_pipeline(config, device)
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise KeyError(f"Missing model_state_dict: {checkpoint_path}")
    pipeline.load_state_dict(state, strict=True)
    pipeline.eval()
    return pipeline, checkpoint


@torch.no_grad()
def _evaluate_one(
    root: Path,
    output_dir: Path,
    *,
    variant: str,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config_path = _config_path(root, variant, seed)
    run_dir = _run_dir(root, variant, seed)
    summaries = list(run_dir.glob("training_summary_epoch*.json"))
    if len(summaries) != 1:
        raise ValueError(f"Expected one training summary: {run_dir}")
    training = _read_json(summaries[0])
    if (
        int(training["start_epoch"]) != 1
        or int(training["completed_epochs"]) != 30
        or int(training["last_epoch"]) != 30
        or int(training["global_step"]) != 18_330
        or bool(training["stopped_early"])
        or training["safety_stop_reason"] is not None
    ):
        raise ValueError(f"Run is not a fresh complete 30-epoch run: {run_dir}")

    config = load_resolved_sparse_config(
        config_path,
        project_root=root,
        run_dir=output_dir / f"_resolve_{variant.lower()}_{seed}",
        device=str(device),
    )
    split = load_frozen_data_split(
        run_dir / "data_split.json", config, verify_splits=("val",)
    )
    bundle = build_sparse_dataloaders(config, split, splits=("val",))
    if set(bundle.loaders) != {"val"}:
        raise RuntimeError("Validation evaluation constructed a non-val loader")

    pipeline, checkpoint = _load_best_pipeline(config, run_dir, device)
    rfno_frozen = all(
        not parameter.requires_grad for parameter in pipeline.rfno.parameters()
    )
    rfno_gradients_none = all(
        parameter.grad is None for parameter in pipeline.rfno.parameters()
    )
    reconstructor_parameter_count = sum(
        1 for parameter in pipeline.reconstructor.parameters() if parameter.requires_grad
    )
    optimizer_state = checkpoint.get("optimizer_state_dict", {})
    optimizer_groups = optimizer_state.get("param_groups", [])
    optimizer_parameter_count = sum(
        len(group.get("params", [])) for group in optimizer_groups
    )
    optimizer_excludes_rfno = (
        len(optimizer_groups) == 1
        and optimizer_parameter_count == reconstructor_parameter_count
    )
    if not (rfno_frozen and rfno_gradients_none and optimizer_excludes_rfno):
        raise RuntimeError(f"RFNO/optimizer freeze contract failed: {run_dir}")

    accumulator = SparseHistoryMetricAccumulator(
        normalization_mean=bundle.normalization_mean,
        normalization_std=bundle.normalization_std,
    )
    evaluated_samples = 0
    inference_seconds = 0.0
    evaluation_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for raw_batch in bundle.loaders["val"]:
        batch = move_sparse_batch_to_device(raw_batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            reconstruction = pipeline.reconstructor(
                batch["x_obs"], batch["obs_mask"]
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - started
        if tuple(reconstruction.shape[1:]) != (60, 64, 64):
            raise RuntimeError("Unexpected reconstruction shape")
        if not torch.isfinite(reconstruction).all():
            raise FloatingPointError("Non-finite validation reconstruction")
        accumulator.update(
            reconstruction,
            batch["x_full"],
            batch["obs_mask"],
            source_ids=batch["source_id"],
        )
        evaluated_samples += int(reconstruction.shape[0])
    metrics = accumulator.compute()
    total_evaluation_seconds = time.perf_counter() - evaluation_started
    eval_peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda"
        else 0.0
    )
    if int(metrics["source_count"]) != 7 or evaluated_samples != 777:
        raise RuntimeError("Validation source/sample count differs from frozen split")

    source_entries = split["splits"]["val"]
    per_mat_rows: list[dict[str, Any]] = []
    per_source_payload: dict[str, Any] = {}
    for source_id, source_metrics in sorted(
        metrics["per_source"].items(), key=lambda item: int(item[0])
    ):
        source = source_entries[int(source_id)]
        flattened = _flatten_metrics(source_metrics)
        per_mat_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "source_id": int(source_id),
                "mat_name": source["name"],
                "relative_path": source["relative_path"],
                **flattened,
            }
        )
        per_source_payload[source_id] = {
            "source": source,
            **source_metrics,
        }

    best_path = run_dir / "checkpoints" / "best.pt"
    last_path = run_dir / "checkpoints" / "last.pt"
    initialization = config["pipeline"].get("reconstructor_checkpoint")
    summary_row: dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "actual_epochs": int(training["last_epoch"]),
        "best_epoch": int(training["best_epoch"]),
        "training_seconds": float(training["elapsed_seconds"]),
        "training_peak_memory_mb": float(training["peak_memory_allocated_mb"]),
        "evaluated_samples": evaluated_samples,
        "inference_seconds": inference_seconds,
        "seconds_per_sample": inference_seconds / evaluated_samples,
        "total_evaluation_seconds": total_evaluation_seconds,
        "evaluation_peak_memory_mb": eval_peak_memory_mb,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": _sha256(best_path),
        "last_checkpoint": str(last_path.resolve()),
        "last_checkpoint_sha256": _sha256(last_path),
        "initialization_checkpoint": (
            "" if initialization is None else str(Path(initialization).resolve())
        ),
        "initialization_checkpoint_sha256": (
            "" if initialization is None else _sha256(Path(initialization))
        ),
        "rfno_frozen": rfno_frozen,
        "rfno_gradients_none": rfno_gradients_none,
        "optimizer_excludes_rfno": optimizer_excludes_rfno,
        **_flatten_metrics(metrics),
    }

    evaluation_payload = {
        "experiment_id": "V0-vs-V1-initialization",
        "variant": variant,
        "seed": seed,
        "split": "val",
        "test_accessed": False,
        "summary": summary_row,
        "metrics": {
            **metrics,
            "per_source": per_source_payload,
        },
    }
    evaluation_path = run_dir / "evaluation" / "reconstruction_val_metrics.json"
    if evaluation_path.exists():
        raise FileExistsError(evaluation_path)
    evaluation_path.write_text(
        json.dumps(evaluation_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    learning_rows = []
    epochs = [
        json.loads(line)
        for line in (run_dir / "logs" / "epochs.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if len(epochs) != 30:
        raise RuntimeError("Learning curve does not contain 30 epochs")
    for epoch in epochs:
        learning_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "epoch": int(epoch["epoch"]),
                "global_step": int(epoch["global_step"]),
                "learning_rate": float(epoch["learning_rates"][0]),
                "train_total": float(epoch["train"]["total"]),
                "val_total": float(epoch["val"]["total"]),
                "val_missing_nrmse": float(
                    epoch["val"]["reconstruction_missing_nrmse"]
                ),
                "val_missing_correlation": float(
                    epoch["val"]["reconstruction_missing_correlation"]
                ),
                "val_full_ssp": float(
                    epoch["val"]["reconstruction_full_spectrum_ssp"]
                ),
                "val_full_gradient_nrmse": float(
                    epoch["val"]["reconstruction_full_gradient_nrmse"]
                ),
            }
        )
    return summary_row, per_mat_rows, learning_rows


def _aggregate(
    summary_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_names = list(METRIC_DIRECTIONS)
    aggregate_rows = []
    for variant in VARIANTS:
        rows = [row for row in summary_rows if row["variant"] == variant]
        aggregate: dict[str, Any] = {"variant": variant, "n_seeds": len(rows)}
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_sample_std"] = statistics.stdev(values)
        for metric in (
            "training_seconds",
            "training_peak_memory_mb",
            "seconds_per_sample",
        ):
            values = [float(row[metric]) for row in rows]
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_sample_std"] = statistics.stdev(values)
        aggregate_rows.append(aggregate)

    paired_rows = []
    for seed in SEEDS:
        v0 = next(
            row
            for row in summary_rows
            if row["variant"] == "V0" and row["seed"] == seed
        )
        v1 = next(
            row
            for row in summary_rows
            if row["variant"] == "V1" and row["seed"] == seed
        )
        paired = {"seed": seed, "difference": "V1_minus_V0"}
        for metric in metric_names:
            paired[metric] = float(v1[metric]) - float(v0[metric])
        paired_rows.append(paired)

    critical = float(student_t.ppf(0.975, df=len(SEEDS) - 1))
    ci_rows = []
    for metric, direction in METRIC_DIRECTIONS.items():
        differences = [float(row[metric]) for row in paired_rows]
        mean = statistics.fmean(differences)
        sample_std = statistics.stdev(differences)
        half_width = critical * sample_std / math.sqrt(len(differences))
        low, high = mean - half_width, mean + half_width
        significant = low > 0.0 or high < 0.0
        favors_v1 = significant and (
            (direction == "lower" and high < 0.0)
            or (direction == "higher" and low > 0.0)
        )
        ci_rows.append(
            {
                "metric": metric,
                "direction": direction,
                "n_pairs": len(differences),
                "v1_minus_v0_mean": mean,
                "sample_std": sample_std,
                "student_t_df": len(differences) - 1,
                "student_t_critical_95": critical,
                "ci95_low": low,
                "ci95_high": high,
                "significant_95": significant,
                "favors_v1": favors_v1,
            }
        )
    return aggregate_rows, paired_rows, ci_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    device = torch.device(args.device)

    summary_rows: list[dict[str, Any]] = []
    per_mat_rows: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for variant in VARIANTS:
            summary, per_mat, learning = _evaluate_one(
                root,
                output_dir,
                variant=variant,
                seed=seed,
                device=device,
            )
            summary_rows.append(summary)
            per_mat_rows.extend(per_mat)
            learning_rows.extend(learning)
            del summary, per_mat, learning
            if device.type == "cuda":
                torch.cuda.empty_cache()

    aggregate_rows, paired_rows, ci_rows = _aggregate(summary_rows)
    _write_csv(output_dir / "summary_by_seed.csv", summary_rows)
    _write_csv(output_dir / "summary_three_seed.csv", aggregate_rows)
    _write_csv(output_dir / "paired_seed_differences.csv", paired_rows)
    _write_csv(output_dir / "paired_student_t_ci95.csv", ci_rows)
    _write_csv(output_dir / "per_mat_metrics.csv", per_mat_rows)
    _write_csv(output_dir / "learning_curves.csv", learning_rows)

    figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for variant in VARIANTS:
        for seed in SEEDS:
            rows = [
                row
                for row in learning_rows
                if row["variant"] == variant and row["seed"] == seed
            ]
            axis.plot(
                [row["epoch"] for row in rows],
                [row["val_missing_nrmse"] for row in rows],
                label=f"{variant} seed{seed}",
                alpha=0.85,
            )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation missing-region NRMSE")
    axis.set_title("V0 vs V1 spatial Sparse ViT initialization")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.savefig(output_dir / "learning_curves.png", dpi=170)
    plt.close(figure)

    primary = next(row for row in ci_rows if row["metric"] == "missing_nrmse")
    recommendation = (
        "recommend_V1_pretrained_initialization"
        if primary["favors_v1"]
        else "no_significant_initialization_recommendation"
    )
    manifest = {
        "experiment_id": "V0-vs-V1-initialization",
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "test_accessed": False,
        "n_runs": len(summary_rows),
        "n_per_mat_rows": len(per_mat_rows),
        "n_learning_curve_rows": len(learning_rows),
        "recommendation": recommendation,
        "summary_by_seed": summary_rows,
        "aggregate": aggregate_rows,
        "paired_student_t_ci95": ci_rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
