#!/usr/bin/env python3
"""Build a validation-only reconstruction scorecard from existing artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

from scipy.stats import t as student_t


SEEDS = (42, 43, 44)
METRICS = (
    "missing_region_nrmse",
    "missing_region_corr",
    "full_field_ssp",
    "full_field_gradient_nrmse",
)
DIRECTIONS = {
    "missing_region_nrmse": "lower",
    "missing_region_corr": "higher",
    "full_field_ssp": "lower",
    "full_field_gradient_nrmse": "lower",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def training_summary(run_dir: Path) -> dict[str, Any]:
    paths = list(run_dir.glob("training_summary_epoch*.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one training summary: {run_dir}")
    return read_json(paths[0])


def flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        "missing_region_nrmse": float(metrics["missing_region"]["nrmse"]),
        "missing_region_corr": float(metrics["missing_region"]["correlation"]),
        "full_field_ssp": float(metrics["full"]["ssp"]),
        "full_field_gradient_nrmse": float(
            metrics["full"]["gradient_nrmse"]
        ),
    }


def flatten_legacy(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        "missing_region_nrmse": float(
            metrics["missing_region"]["nrmse"]
        ),
        "missing_region_corr": float(
            metrics["missing_region"]["correlation"]
        ),
        "full_field_ssp": float(metrics["ssp"]),
        "full_field_gradient_nrmse": float(metrics["gradient_nrmse"]),
    }


def split_sources(root: Path) -> dict[int, str]:
    split = read_json(root / "data/splits/sea_surface_bimodal_v1.json")
    return {
        int(item["source_id"]): str(item["name"])
        for item in split["splits"]["val"]
    }


def vit_rows(root: Path, method: str, parameter_source: Mapping[str, Any]) -> list[dict[str, Any]]:
    prefix = (
        "v0_vit_random_sparse_finetune"
        if method == "V0"
        else "v1_vit_mae_pretrained_sparse_finetune"
    )
    sources = split_sources(root)
    rows = []
    for seed in SEEDS:
        run_dir = (
            root
            / "runs/sparse"
            / f"{prefix}_seed{seed}_30ep_formal_20260729_v1"
        )
        evaluation = read_json(
            run_dir / "evaluation/reconstruction_val_metrics.json"
        )
        train = training_summary(run_dir)
        pretrain_seconds = 0.0
        pretrain_peak = 0.0
        if method == "V1":
            pretrain = training_summary(
                root
                / "runs/sparse"
                / f"v1_spatial_mae_pretrain_patch75_seed{seed}_formal_20260728_v1"
            )
            pretrain_seconds = float(pretrain["elapsed_seconds"])
            pretrain_peak = float(pretrain["peak_memory_allocated_mb"])
        per_source = {
            int(source_id): flatten_metrics(source_metrics)
            for source_id, source_metrics in evaluation["metrics"][
                "per_source"
            ].items()
        }
        rows.append(
            {
                "method": method,
                "seed": seed,
                **flatten_metrics(evaluation["metrics"]),
                "per_source": per_source,
                "source_names": sources,
                "reconstructor_parameters": int(
                    parameter_source["reconstructor_parameters"]
                ),
                "pipeline_parameters": int(
                    parameter_source["pipeline_parameters"]
                ),
                "trainable_parameters": int(
                    parameter_source["trainable_parameters"]
                ),
                "pretraining_seconds": pretrain_seconds,
                "downstream_seconds": float(train["elapsed_seconds"]),
                "total_training_seconds": pretrain_seconds
                + float(train["elapsed_seconds"]),
                "pretraining_peak_memory_mb": pretrain_peak,
                "downstream_peak_memory_mb": float(
                    train["peak_memory_allocated_mb"]
                ),
                "peak_training_memory_mb": max(
                    pretrain_peak,
                    float(train["peak_memory_allocated_mb"]),
                ),
                "inference_seconds_per_sample": float(
                    evaluation["summary"]["seconds_per_sample"]
                ),
                "inference_scope": "reconstructor_only",
            }
        )
    return rows


def v2_rows(root: Path, gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = split_sources(root)
    rows = []
    for result in gate["results"]:
        if result["label"] not in {"V2-75", "V2-90"}:
            continue
        per_source = {
            int(source_id): flatten_metrics(source_metrics)
            for source_id, source_metrics in result["per_source"].items()
        }
        rows.append(
            {
                "method": result["label"],
                "seed": 42,
                "missing_region_nrmse": float(result["missing_nrmse"]),
                "missing_region_corr": float(result["missing_correlation"]),
                "full_field_ssp": float(result["full_ssp"]),
                "full_field_gradient_nrmse": float(
                    result["full_gradient_nrmse"]
                ),
                "per_source": per_source,
                "source_names": sources,
                "reconstructor_parameters": int(
                    result["reconstructor_parameters"]
                ),
                "pipeline_parameters": int(result["pipeline_parameters"]),
                "trainable_parameters": int(result["trainable_parameters"]),
                "pretraining_seconds": float(
                    result["pretraining_seconds"]
                ),
                "downstream_seconds": float(result["downstream_seconds"]),
                "total_training_seconds": float(
                    result["pretraining_seconds"]
                )
                + float(result["downstream_seconds"]),
                "pretraining_peak_memory_mb": float(
                    result["pretraining_peak_memory_mb"]
                ),
                "downstream_peak_memory_mb": float(
                    result["downstream_peak_memory_mb"]
                ),
                "peak_training_memory_mb": max(
                    float(result["pretraining_peak_memory_mb"]),
                    float(result["downstream_peak_memory_mb"]),
                ),
                "inference_seconds_per_sample": float(
                    result["inference_ms_per_sample"]
                )
                / 1000.0,
                "inference_scope": "reconstructor_only",
            }
        )
    return rows


def legacy_rows(
    root: Path,
    method: str,
    run_specs: Mapping[int, tuple[str, str]],
    pretrain_specs: Mapping[int, str] | None = None,
) -> list[dict[str, Any]]:
    sources = split_sources(root)
    rows = []
    for seed, (train_rel, eval_rel) in run_specs.items():
        train_dir = root / train_rel
        evaluation = read_json(root / eval_rel)
        train = training_summary(train_dir)
        pretrain_seconds = 0.0
        pretrain_peak = 0.0
        if pretrain_specs is not None:
            pretrain = training_summary(root / pretrain_specs[seed])
            pretrain_seconds = float(pretrain["elapsed_seconds"])
            pretrain_peak = float(pretrain["peak_memory_allocated_mb"])
        history = evaluation["metrics"]["history_reconstruction"]
        per_source = {
            int(source_id): flatten_legacy(
                source_metrics["history_reconstruction"]
            )
            for source_id, source_metrics in evaluation["metrics"][
                "per_source"
            ].items()
        }
        rows.append(
            {
                "method": method,
                "seed": seed,
                **flatten_legacy(history),
                "per_source": per_source,
                "source_names": sources,
                "reconstructor_parameters": int(
                    evaluation["parameters"]["trainable"]
                ),
                "pipeline_parameters": int(
                    evaluation["parameters"]["total"]
                ),
                "trainable_parameters": int(
                    evaluation["parameters"]["trainable"]
                ),
                "pretraining_seconds": pretrain_seconds,
                "downstream_seconds": float(train["elapsed_seconds"]),
                "total_training_seconds": pretrain_seconds
                + float(train["elapsed_seconds"]),
                "pretraining_peak_memory_mb": pretrain_peak,
                "downstream_peak_memory_mb": float(
                    train["peak_memory_allocated_mb"]
                ),
                "peak_training_memory_mb": max(
                    pretrain_peak,
                    float(train["peak_memory_allocated_mb"]),
                ),
                "inference_seconds_per_sample": float(
                    evaluation["metrics"]["seconds_per_sample"]
                ),
                "inference_scope": "reconstructor_plus_rfno_300_rollout",
            }
        )
    return rows


def aggregate(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "row_type": "method_summary",
        "method": method,
        "comparator": "",
        "seed": "",
        "n_seeds": len(rows),
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        result[metric] = statistics.fmean(values)
        result[f"{metric}_sample_std"] = (
            statistics.stdev(values) if len(values) >= 2 else ""
        )
    for field in (
        "reconstructor_parameters",
        "pipeline_parameters",
        "trainable_parameters",
        "pretraining_seconds",
        "downstream_seconds",
        "total_training_seconds",
        "peak_training_memory_mb",
        "inference_seconds_per_sample",
    ):
        values = [float(row[field]) for row in rows]
        result[field] = statistics.fmean(values)
        result[f"{field}_sample_std"] = (
            statistics.stdev(values) if len(values) >= 2 else ""
        )
    result["inference_scope"] = rows[0]["inference_scope"]

    for metric, direction in DIRECTIONS.items():
        candidates = [
            (
                float(source_metrics[metric]),
                int(row["seed"]),
                int(source_id),
                row["source_names"][int(source_id)],
            )
            for row in rows
            for source_id, source_metrics in row["per_source"].items()
        ]
        worst = (
            max(candidates, key=lambda item: item[0])
            if direction == "lower"
            else min(candidates, key=lambda item: item[0])
        )
        result[f"worst_per_mat_{metric}"] = worst[0]
        result[f"worst_per_mat_{metric}_seed"] = worst[1]
        result[f"worst_per_mat_{metric}_source_id"] = worst[2]
        result[f"worst_per_mat_{metric}_mat"] = worst[3]
    return result


def comparison(
    label: str,
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    ci: bool,
) -> dict[str, Any]:
    common = sorted(
        set(int(row["seed"]) for row in left)
        & set(int(row["seed"]) for row in right)
    )
    result: dict[str, Any] = {
        "row_type": "paired_difference",
        "method": label,
        "comparator": "left_minus_right",
        "seed": ",".join(map(str, common)),
        "n_seeds": len(common),
        "inference_scope": "",
    }
    for metric in METRICS:
        differences = []
        for seed in common:
            left_row = next(row for row in left if row["seed"] == seed)
            right_row = next(row for row in right if row["seed"] == seed)
            differences.append(float(left_row[metric]) - float(right_row[metric]))
        mean = statistics.fmean(differences)
        result[metric] = mean
        result[f"{metric}_sample_std"] = (
            statistics.stdev(differences) if len(differences) >= 2 else ""
        )
        if ci and len(differences) >= 2:
            std = statistics.stdev(differences)
            critical = float(student_t.ppf(0.975, len(differences) - 1))
            half_width = critical * std / math.sqrt(len(differences))
            low, high = mean - half_width, mean + half_width
            result[f"{metric}_ci95_low"] = low
            result[f"{metric}_ci95_high"] = high
            result[f"{metric}_significant_95"] = low > 0.0 or high < 0.0
        else:
            result[f"{metric}_ci95_low"] = ""
            result[f"{metric}_ci95_high"] = ""
            result[f"{metric}_significant_95"] = ""
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    gate = read_json(
        root / "results/v2_tubelet_seed42_gate_20260729_v1/manifest.json"
    )
    parameter_source = next(
        row for row in gate["results"] if row["label"] == "V1"
    )
    rows_by_method = {
        "V0": vit_rows(root, "V0", parameter_source),
        "V1": vit_rows(root, "V1", parameter_source),
        "V2-75": v2_rows(root, gate)[:1],
        "V2-90": v2_rows(root, gate)[1:],
        "B4": legacy_rows(
            root,
            "B4",
            {
                42: (
                    "runs/sparse/b4_mask_unet_frozen_fixed_points_r05_mask_00006_seed42_formal_20260721_v2",
                    "runs/sparse/b4_mask_unet_frozen_fixed_points_r05_mask_00006_seed42_formal_20260721_v2/evaluation/val_metrics.json",
                ),
                43: (
                    "runs/sparse/finalist_b4_fixed_r05_seed43_mask_00006_formal_20260722_v1",
                    "runs/sparse/finalist_b4_fixed_r05_seed43_mask_00006_formal_20260722_v1/evaluation/val_metrics.json",
                ),
                44: (
                    "runs/sparse/finalist_b4_fixed_r05_seed44_mask_00006_formal_20260722_v1",
                    "runs/sparse/finalist_b4_fixed_r05_seed44_mask_00006_formal_20260722_v1/evaluation/val_metrics.json",
                ),
            },
        ),
        "H1-A4": legacy_rows(
            root,
            "H1-A4",
            {
                seed: (
                    f"runs/sparse/h1_a4_pretrained_seed{seed}_30ep_fixed_r05_mask_00006",
                    f"runs/sparse/h1_a4_pretrained_seed{seed}_val300_20260728_v1/evaluation/val_metrics.json",
                )
                for seed in SEEDS
            },
            {
                42: "runs/sparse/h1_a1_partialconv_unet_frozen_fixed_points_r05_mask_00006_seed42",
                43: "runs/sparse/h1_a4_pretrain_partialconv3_unet_skip_fixed_points_r05_mask_00006_seed43",
                44: "runs/sparse/h1_a4_pretrain_partialconv3_unet_skip_fixed_points_r05_mask_00006_seed44",
            },
        ),
    }
    summaries = [
        aggregate(method, rows) for method, rows in rows_by_method.items()
    ]
    ranking = sorted(
        summaries, key=lambda row: float(row["missing_region_nrmse"])
    )
    for rank, row in enumerate(ranking, start=1):
        row["missing_region_nrmse_rank"] = rank

    comparisons = [
        comparison(
            "V1_minus_V0",
            rows_by_method["V1"],
            rows_by_method["V0"],
            ci=True,
        ),
        comparison(
            "V2-75_minus_V1",
            rows_by_method["V2-75"],
            rows_by_method["V1"],
            ci=False,
        ),
        comparison(
            "V2-90_minus_V2-75",
            rows_by_method["V2-90"],
            rows_by_method["V2-75"],
            ci=False,
        ),
    ]
    scorecard = summaries + comparisons
    fields: list[str] = []
    for row in scorecard:
        for key in row:
            if key not in fields:
                fields.append(key)
    output_csv = output / "reconstruction_val_scorecard.csv"
    with output_csv.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scorecard)

    best_vit = min(
        (row for row in summaries if row["method"].startswith("V")),
        key=lambda row: float(row["missing_region_nrmse"]),
    )
    payload = {
        "experiment_id": "vit-mae-reconstruction-val-scorecard",
        "category": "evaluation_only",
        "split": "val",
        "test_accessed": False,
        "composite_score_constructed": False,
        "metric_scope": {
            "missing_region_nrmse": "missing_region",
            "missing_region_corr": "missing_region",
            "full_field_ssp": "full_field_existing_evaluator",
            "full_field_gradient_nrmse": "full_field_existing_evaluator",
        },
        "ranking_by_missing_region_nrmse": [
            {
                "rank": row["missing_region_nrmse_rank"],
                "method": row["method"],
                "mean": row["missing_region_nrmse"],
                "sample_std": row["missing_region_nrmse_sample_std"],
                "n_seeds": row["n_seeds"],
            }
            for row in ranking
        ],
        "comparisons": comparisons,
        "best_vit": best_vit["method"],
        "allow_best_vit_into_v3": best_vit["method"] == "V1",
        "scorecard_csv": str(output_csv),
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
