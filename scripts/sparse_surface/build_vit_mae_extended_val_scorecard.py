#!/usr/bin/env python3
"""Build the validation-only extended ViT-MAE scorecard from saved artifacts."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

from scipy.stats import t as student_t


SEEDS = (42, 43, 44)
HORIZONS = (30, 60, 120, 180, 240, 300)
RECON_METRICS = (
    "reconstruction_missing_nrmse",
    "reconstruction_missing_corr",
    "reconstruction_full_ssp",
    "reconstruction_full_gradient_nrmse",
)
FORECAST_METRICS = tuple(
    f"forecast_{horizon}_{metric}"
    for horizon in HORIZONS
    for metric in ("nrmse", "corr", "ssp", "gradient_nrmse")
)
METRICS = RECON_METRICS + FORECAST_METRICS
DIRECTIONS = {
    **{
        metric: ("higher" if metric.endswith("_corr") else "lower")
        for metric in METRICS
    }
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


def split_sources(root: Path) -> dict[int, str]:
    split = read_json(root / "data/splits/sea_surface_bimodal_v1.json")
    return {
        int(item["source_id"]): str(item["name"])
        for item in split["splits"]["val"]
    }


def flatten_history(history: Mapping[str, Any]) -> dict[str, float]:
    return {
        "reconstruction_missing_nrmse": float(
            history["missing_region"]["nrmse"]
        ),
        "reconstruction_missing_corr": float(
            history["missing_region"]["correlation"]
        ),
        "reconstruction_full_ssp": float(history["ssp"]),
        "reconstruction_full_gradient_nrmse": float(
            history["gradient_nrmse"]
        ),
    }


def flatten_reconstruction_only(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        "reconstruction_missing_nrmse": float(
            metrics["missing_region"]["nrmse"]
        ),
        "reconstruction_missing_corr": float(
            metrics["missing_region"]["correlation"]
        ),
        "reconstruction_full_ssp": float(metrics["full"]["ssp"]),
        "reconstruction_full_gradient_nrmse": float(
            metrics["full"]["gradient_nrmse"]
        ),
    }


def flatten_val300(metrics: Mapping[str, Any]) -> dict[str, float]:
    result = flatten_history(metrics["history_reconstruction"])
    for horizon in HORIZONS:
        forecast = metrics[f"forecast_{horizon}"]
        result.update(
            {
                f"forecast_{horizon}_nrmse": float(forecast["nrmse"]),
                f"forecast_{horizon}_corr": float(forecast["correlation"]),
                f"forecast_{horizon}_ssp": float(forecast["ssp"]),
                f"forecast_{horizon}_gradient_nrmse": float(
                    forecast["gradient_nrmse"]
                ),
            }
        )
    return result


def validate_val300(evaluation: Mapping[str, Any], label: str) -> None:
    metrics = evaluation["metrics"]
    if (
        evaluation.get("split") != "val"
        or int(metrics["rollout_steps"]) != 300
        or int(metrics["evaluated_samples"]) != 777
        or int(metrics["source_count"]) != 7
    ):
        raise RuntimeError(f"Incomplete val-300 artifact: {label}")


def val300_rows(
    root: Path,
    method: str,
    train_dirs: Mapping[int, str],
    eval_dirs: Mapping[int, str],
    *,
    pretrain_dirs: Mapping[int, str] | None = None,
    cumulative_prefix_dirs: Mapping[int, tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    sources = split_sources(root)
    rows = []
    for seed in SEEDS:
        train_dir = root / train_dirs[seed]
        evaluation = read_json(root / eval_dirs[seed] / "evaluation/val_metrics.json")
        validate_val300(evaluation, f"{method}/{seed}")
        train = training_summary(train_dir)
        if (
            int(train["completed_epochs"]) < 1
            or int(train["last_epoch"]) < 1
            or train["safety_stop_reason"] is not None
        ):
            raise RuntimeError(f"Incomplete training artifact: {method}/{seed}")

        prefix_seconds = 0.0
        prefix_peaks: list[float] = []
        prefix_dirs = (
            cumulative_prefix_dirs[seed]
            if cumulative_prefix_dirs is not None
            else ()
        )
        for relative in prefix_dirs:
            prefix = training_summary(root / relative)
            prefix_seconds += float(prefix["elapsed_seconds"])
            prefix_peaks.append(float(prefix["peak_memory_allocated_mb"]))
        if pretrain_dirs is not None:
            pretrain = training_summary(root / pretrain_dirs[seed])
            prefix_seconds += float(pretrain["elapsed_seconds"])
            prefix_peaks.append(float(pretrain["peak_memory_allocated_mb"]))

        metrics = evaluation["metrics"]
        per_source = {
            int(source_id): flatten_val300(source_metrics)
            for source_id, source_metrics in metrics["per_source"].items()
        }
        stage_peak = float(train["peak_memory_allocated_mb"])
        rows.append(
            {
                "method": method,
                "seed": seed,
                **flatten_val300(metrics),
                "per_source": per_source,
                "source_names": sources,
                "total_parameters": int(evaluation["parameters"]["total"]),
                "trainable_parameters": int(
                    evaluation["parameters"]["trainable"]
                ),
                "training_stage_seconds": float(train["elapsed_seconds"]),
                "cumulative_training_seconds": prefix_seconds
                + float(train["elapsed_seconds"]),
                "peak_training_memory_mb": max([stage_peak, *prefix_peaks]),
                "inference_seconds": float(metrics["elapsed_seconds"]),
                "inference_seconds_per_sample": float(
                    metrics["seconds_per_sample"]
                ),
                "inference_scope": "reconstructor_plus_rfno_300_rollout",
            }
        )
    return rows


def vit_v0_rows(
    root: Path, parameter_source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    sources = split_sources(root)
    rows = []
    for seed in SEEDS:
        run_dir = (
            root
            / "runs/sparse"
            / f"v0_vit_random_sparse_finetune_seed{seed}_30ep_formal_20260729_v1"
        )
        evaluation = read_json(
            run_dir / "evaluation/reconstruction_val_metrics.json"
        )
        train = training_summary(run_dir)
        rows.append(
            {
                "method": "V0",
                "seed": seed,
                **flatten_reconstruction_only(evaluation["metrics"]),
                "per_source": {
                    int(source_id): flatten_reconstruction_only(source_metrics)
                    for source_id, source_metrics in evaluation["metrics"][
                        "per_source"
                    ].items()
                },
                "source_names": sources,
                "total_parameters": int(parameter_source["pipeline_parameters"]),
                "trainable_parameters": int(
                    parameter_source["trainable_parameters"]
                ),
                "training_stage_seconds": float(train["elapsed_seconds"]),
                "cumulative_training_seconds": float(train["elapsed_seconds"]),
                "peak_training_memory_mb": float(
                    train["peak_memory_allocated_mb"]
                ),
                "inference_seconds": float(
                    evaluation["summary"]["inference_seconds"]
                ),
                "inference_seconds_per_sample": float(
                    evaluation["summary"]["seconds_per_sample"]
                ),
                "inference_scope": "reconstructor_only",
            }
        )
    return rows


def overlay_v1_reconstruction_only(
    root: Path, rows: list[dict[str, Any]]
) -> None:
    """Keep the established V1/V0 reconstruction evaluator for strict pairing."""
    for row in rows:
        seed = int(row["seed"])
        path = (
            root
            / "runs/sparse"
            / (
                f"v1_vit_mae_pretrained_sparse_finetune_seed{seed}_"
                "30ep_formal_20260729_v1"
            )
            / "evaluation/reconstruction_val_metrics.json"
        )
        evaluation = read_json(path)
        row.update(flatten_reconstruction_only(evaluation["metrics"]))
        for source_id, source_metrics in evaluation["metrics"][
            "per_source"
        ].items():
            row["per_source"][int(source_id)].update(
                flatten_reconstruction_only(source_metrics)
            )


def v2_rows(root: Path, gate: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    sources = split_sources(root)
    result: dict[str, list[dict[str, Any]]] = {}
    for item in gate["results"]:
        method = str(item["label"])
        if method not in {"V2-75", "V2-90"}:
            continue
        result[method] = [
            {
                "method": method,
                "seed": 42,
                "reconstruction_missing_nrmse": float(item["missing_nrmse"]),
                "reconstruction_missing_corr": float(
                    item["missing_correlation"]
                ),
                "reconstruction_full_ssp": float(item["full_ssp"]),
                "reconstruction_full_gradient_nrmse": float(
                    item["full_gradient_nrmse"]
                ),
                "per_source": {
                    int(source_id): flatten_reconstruction_only(source_metrics)
                    for source_id, source_metrics in item["per_source"].items()
                },
                "source_names": sources,
                "total_parameters": int(item["pipeline_parameters"]),
                "trainable_parameters": int(item["trainable_parameters"]),
                "training_stage_seconds": float(item["downstream_seconds"]),
                "cumulative_training_seconds": float(
                    item["pretraining_seconds"]
                )
                + float(item["downstream_seconds"]),
                "peak_training_memory_mb": max(
                    float(item["pretraining_peak_memory_mb"]),
                    float(item["downstream_peak_memory_mb"]),
                ),
                "inference_seconds": float(item["inference_seconds"]),
                "inference_seconds_per_sample": float(
                    item["inference_ms_per_sample"]
                )
                / 1000.0,
                "inference_scope": "reconstructor_only",
            }
        ]
    return result


def aggregate(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "row_type": "method_summary",
        "method": method,
        "comparison": "",
        "seeds": ",".join(str(row["seed"]) for row in rows),
        "n_seeds": len(rows),
        "forecast_n_seeds": sum(
            "forecast_300_nrmse" in row for row in rows
        ),
        "inference_scope": rows[0]["inference_scope"],
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if metric in row]
        result[metric] = statistics.fmean(values) if values else ""
        result[f"{metric}_sample_std"] = (
            statistics.stdev(values) if len(values) >= 2 else ""
        )
        if not values:
            continue
        candidates = [
            (
                float(source_metrics[metric]),
                int(row["seed"]),
                int(source_id),
                row["source_names"][int(source_id)],
            )
            for row in rows
            for source_id, source_metrics in row["per_source"].items()
            if metric in source_metrics
        ]
        worst = (
            min(candidates, key=lambda item: item[0])
            if DIRECTIONS[metric] == "higher"
            else max(candidates, key=lambda item: item[0])
        )
        result[f"worst_per_mat_{metric}"] = worst[0]
        result[f"worst_per_mat_{metric}_seed"] = worst[1]
        result[f"worst_per_mat_{metric}_mat"] = worst[3]

    for field in (
        "total_parameters",
        "trainable_parameters",
        "training_stage_seconds",
        "cumulative_training_seconds",
        "peak_training_memory_mb",
        "inference_seconds",
        "inference_seconds_per_sample",
    ):
        values = [float(row[field]) for row in rows]
        result[field] = statistics.fmean(values)
        result[f"{field}_sample_std"] = (
            statistics.stdev(values) if len(values) >= 2 else ""
        )
    return result


def paired_effect(
    label: str,
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    metrics: tuple[str, ...] = METRICS,
) -> dict[str, Any]:
    common = sorted(
        set(int(row["seed"]) for row in left)
        & set(int(row["seed"]) for row in right)
    )
    result: dict[str, Any] = {
        "row_type": "paired_factor_effect",
        "method": label,
        "comparison": "left_minus_right",
        "seeds": ",".join(map(str, common)),
        "n_seeds": len(common),
        "forecast_n_seeds": "",
        "inference_scope": "",
    }
    for metric in metrics:
        differences = []
        for seed in common:
            left_row = next(row for row in left if row["seed"] == seed)
            right_row = next(row for row in right if row["seed"] == seed)
            if metric not in left_row or metric not in right_row:
                continue
            differences.append(float(left_row[metric]) - float(right_row[metric]))
        if not differences:
            continue
        mean = statistics.fmean(differences)
        result[metric] = mean
        result[f"{metric}_sample_std"] = (
            statistics.stdev(differences) if len(differences) >= 2 else ""
        )
        if len(differences) >= 2:
            std = statistics.stdev(differences)
            critical = float(student_t.ppf(0.975, len(differences) - 1))
            margin = critical * std / math.sqrt(len(differences))
            low, high = mean - margin, mean + margin
            result[f"{metric}_ci95_low"] = low
            result[f"{metric}_ci95_high"] = high
            result[f"{metric}_significant_95"] = low > 0.0 or high < 0.0
        else:
            result[f"{metric}_ci95_low"] = ""
            result[f"{metric}_ci95_high"] = ""
            result[f"{metric}_significant_95"] = ""
    return result


def rank_summaries(summaries: list[dict[str, Any]]) -> None:
    ranking_metrics = (
        "reconstruction_missing_nrmse",
        "forecast_30_nrmse",
        "forecast_300_nrmse",
    )
    for metric in ranking_metrics:
        eligible = [
            row
            for row in summaries
            if row.get(metric, "") != "" and int(row["n_seeds"]) == 3
        ]
        for rank, row in enumerate(
            sorted(eligible, key=lambda item: float(item[metric])), start=1
        ):
            row[f"{metric}_rank_3seed"] = rank
        available = [row for row in summaries if row.get(metric, "") != ""]
        for rank, row in enumerate(
            sorted(available, key=lambda item: float(item[metric])), start=1
        ):
            row[f"{metric}_rank_available"] = rank


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    v2 = v2_rows(root, gate)

    v1_train = {
        seed: (
            f"runs/sparse/v1_vit_mae_pretrained_sparse_finetune_seed{seed}_"
            "30ep_formal_20260729_v1"
        )
        for seed in SEEDS
    }
    v1_eval = {
        seed: (
            f"runs/sparse/v1_vit_mae_pretrained_sparse_finetune_seed{seed}_"
            "val300_20260729_v1"
        )
        for seed in SEEDS
    }
    v1_pretrain = {
        seed: (
            f"runs/sparse/v1_spatial_mae_pretrain_patch75_seed{seed}_"
            "formal_20260728_v1"
        )
        for seed in SEEDS
    }
    v1_val300_rows = val300_rows(
        root,
        "V1",
        v1_train,
        v1_eval,
        pretrain_dirs=v1_pretrain,
    )
    v1_scorecard_rows = copy.deepcopy(v1_val300_rows)
    overlay_v1_reconstruction_only(root, v1_scorecard_rows)
    rows_by_method = {
        "B4": val300_rows(
            root,
            "B4",
            {
                42: "runs/sparse/b4_mask_unet_frozen_fixed_points_r05_mask_00006_seed42_formal_20260721_v2",
                43: "runs/sparse/finalist_b4_fixed_r05_seed43_mask_00006_formal_20260722_v1",
                44: "runs/sparse/finalist_b4_fixed_r05_seed44_mask_00006_formal_20260722_v1",
            },
            {
                42: "runs/sparse/b4_mask_unet_frozen_fixed_points_r05_mask_00006_seed42_formal_20260721_v2",
                43: "runs/sparse/finalist_b4_fixed_r05_seed43_mask_00006_formal_20260722_v1",
                44: "runs/sparse/finalist_b4_fixed_r05_seed44_mask_00006_formal_20260722_v1",
            },
        ),
        "H1-A4": val300_rows(
            root,
            "H1-A4",
            {
                seed: (
                    f"runs/sparse/h1_a4_pretrained_seed{seed}_"
                    "30ep_fixed_r05_mask_00006"
                )
                for seed in SEEDS
            },
            {
                seed: (
                    f"runs/sparse/h1_a4_pretrained_seed{seed}_"
                    "val300_20260728_v1"
                )
                for seed in SEEDS
            },
            pretrain_dirs={
                42: "runs/sparse/h1_a1_partialconv_unet_frozen_fixed_points_r05_mask_00006_seed42",
                43: "runs/sparse/h1_a4_pretrain_partialconv3_unet_skip_fixed_points_r05_mask_00006_seed43",
                44: "runs/sparse/h1_a4_pretrain_partialconv3_unet_skip_fixed_points_r05_mask_00006_seed44",
            },
        ),
        "V0": vit_v0_rows(root, parameter_source),
        "V1": v1_scorecard_rows,
        "V2-75": v2["V2-75"],
        "V2-90": v2["V2-90"],
        "V3": val300_rows(
            root,
            "V3",
            {
                seed: (
                    f"runs/sparse/v3_vit_mae_frozen_rfno_30f_seed{seed}_"
                    "formal_20260729_v1"
                )
                for seed in SEEDS
            },
            {
                seed: (
                    f"runs/sparse/v3_vit_mae_frozen_rfno_30f_seed{seed}_"
                    "val300_20260729_v1"
                )
                for seed in SEEDS
            },
            cumulative_prefix_dirs={
                seed: (v1_pretrain[seed], v1_train[seed]) for seed in SEEDS
            },
        ),
    }

    h1_a5 = val300_rows(
        root,
        "H1-A5",
        {
            seed: (
                f"runs/sparse/h1_a5_random_init_seed{seed}_"
                "30ep_fixed_r05_mask_00006"
            )
            for seed in SEEDS
        },
        {
            seed: (
                f"runs/sparse/h1_a5_random_init_seed{seed}_"
                "val300_20260728_v1"
            )
            for seed in SEEDS
        },
    )

    summaries = [
        aggregate(method, rows) for method, rows in rows_by_method.items()
    ]
    rank_summaries(summaries)
    effects = [
        paired_effect(
            "H1-A4_minus_H1-A5",
            rows_by_method["H1-A4"],
            h1_a5,
        ),
        paired_effect(
            "V1_minus_V0",
            rows_by_method["V1"],
            rows_by_method["V0"],
            metrics=RECON_METRICS,
        ),
        paired_effect(
            "V2-75_minus_V1",
            rows_by_method["V2-75"],
            rows_by_method["V1"],
            metrics=RECON_METRICS,
        ),
        paired_effect(
            "V2-90_minus_V2-75",
            rows_by_method["V2-90"],
            rows_by_method["V2-75"],
            metrics=RECON_METRICS,
        ),
        paired_effect(
            "V3_minus_V1",
            rows_by_method["V3"],
            v1_val300_rows,
        ),
    ]

    output_csv = output / "vit_mae_extended_val_scorecard.csv"
    write_csv(output_csv, summaries + effects)
    v4_gate = read_json(
        root
        / "results/v4_d1_vit_mae_joint_rfno_3seed_val300_20260729_v1"
        / "gate_summary.json"
    )
    manifest = {
        "experiment_id": "vit-mae-extended-val-scorecard",
        "category": "evaluation_only",
        "split": "val",
        "test_accessed_this_run": False,
        "existing_test_results_used": False,
        "composite_score_constructed": False,
        "methods_ranked": list(rows_by_method),
        "v4_included": bool(v4_gate["gate_pass"]),
        "v4_exclusion_reason": (
            "" if v4_gate["gate_pass"] else v4_gate["blocking_items"]
        ),
        "metric_scope": {
            "reconstruction_nrmse_corr": "missing_region",
            "reconstruction_ssp_gradient_nrmse": "full_field",
            "forecast_metrics": "full_field",
        },
        "ranking_policy": {
            "primary": "separate fixed-layout ranks for reconstruction, F30, and F300 NRMSE",
            "three_seed_rank": "requires n_seeds=3",
            "available_rank": "descriptive only; includes seed42-only V2",
            "composite_score": None,
        },
        "strict_paired_effects": [row["method"] for row in effects],
        "paper_claims_supported": [
            (
                "Under the fixed-points 5% mask_00006 validation protocol, B4 "
                "ranks first for reconstruction, F30, and F300 NRMSE among "
                "methods with three seeds."
            ),
            (
                "Spatial MAE pretraining (V1 versus V0) improves all four "
                "reported reconstruction metrics with paired 95% confidence "
                "intervals excluding zero."
            ),
            (
                "Frozen-RFNO forecast-aware training (V3 versus its V1 "
                "initialization) improves reconstruction and every reported "
                "F30-F300 metric with paired 95% confidence intervals "
                "excluding zero."
            ),
            (
                "For H1-A4 versus H1-A5, only reconstruction gradient NRMSE "
                "has a paired 95% confidence interval excluding zero."
            ),
            (
                "At seed42 only, V2-75 is worse than V1 and V2-90 is worse "
                "than V2-75 on all four reconstruction metrics; these are "
                "descriptive effects without confidence intervals."
            ),
        ],
        "paper_claims_not_supported": [
            (
                "No new-method test-set superiority or test-set "
                "generalization claim; existing test results were not used."
            ),
            (
                "No three-seed or forecast claim for V2, and no forecast "
                "claim for V0, because the required saved artifacts do not "
                "exist."
            ),
            (
                "No broad causal claim from cross-architecture rankings; "
                "only the listed strict paired factors support causal "
                "interpretation."
            ),
            (
                "No robustness claim beyond fixed-points 5%, mask_00006, "
                "the frozen split, and the recorded normalization."
            ),
            (
                "No V4 joint-training or long-curriculum benefit claim "
                "because V4-D1 failed its gate and V4-D2 was not run."
            ),
            (
                "Reconstructor-only inference times for V0/V2 are not "
                "comparable with 300-frame pipeline inference times."
            ),
        ],
        "scorecard_csv": str(output_csv),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
