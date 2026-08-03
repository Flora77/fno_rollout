#!/usr/bin/env python3
"""Summarize strictly paired V1-minus-V0 validation-only 300-frame evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

from scipy.stats import t as student_t


SEEDS = (42, 43, 44)
VARIANTS = ("V0", "V1")
HORIZONS = (30, 60, 120, 180, 240, 300)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(path)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def eval_dir(root: Path, variant: str, seed: int) -> Path:
    method = (
        "v0_vit_random_sparse_finetune"
        if variant == "V0"
        else "v1_vit_mae_pretrained_sparse_finetune"
    )
    return (
        root
        / "runs"
        / "sparse"
        / f"{method}_seed{seed}_val300_20260729_v1"
    )


def flatten(metrics: Mapping[str, Any]) -> dict[str, float]:
    history = metrics["history_reconstruction"]
    values = {
        "reconstruction_missing_nrmse": float(
            history["missing_region"]["nrmse"]
        ),
        "reconstruction_missing_corr": float(
            history["missing_region"]["correlation"]
        ),
        "reconstruction_full_nrmse": float(history["nrmse"]),
        "reconstruction_full_corr": float(history["correlation"]),
        "reconstruction_full_ssp": float(history["ssp"]),
        "reconstruction_full_gradient_nrmse": float(
            history["gradient_nrmse"]
        ),
    }
    for horizon in HORIZONS:
        item = metrics[f"forecast_{horizon}"]
        values.update(
            {
                f"forecast_{horizon}_nrmse": float(item["nrmse"]),
                f"forecast_{horizon}_corr": float(item["correlation"]),
                f"forecast_{horizon}_ssp": float(item["ssp"]),
                f"forecast_{horizon}_gradient_nrmse": float(
                    item["gradient_nrmse"]
                ),
            }
        )
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    per_mat_rows: list[dict[str, Any]] = []
    growth_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    split_hashes: set[str] = set()

    for seed in SEEDS:
        for variant in VARIANTS:
            directory = eval_dir(root, variant, seed)
            metrics_path = directory / "evaluation" / "val_metrics.json"
            payload = read_json(metrics_path)
            metrics = payload["metrics"]
            if (
                payload["split"] != "val"
                or int(metrics["evaluated_samples"]) != 777
                or int(metrics["source_count"]) != 7
                or int(metrics["rollout_steps"]) != 300
            ):
                raise RuntimeError(f"Incomplete val-300: {variant}/seed{seed}")
            if not all(payload["freeze_checks"].values()):
                raise RuntimeError(f"RFNO freeze check failed: {variant}/seed{seed}")

            split_path = directory / "data_split.json"
            split_hash = sha256(split_path)
            split_hashes.add(split_hash)
            row = {
                "variant": variant,
                "seed": seed,
                "parameters_total": int(payload["parameters"]["total"]),
                "parameters_trainable": int(payload["parameters"]["trainable"]),
                "inference_seconds": float(metrics["elapsed_seconds"]),
                "seconds_per_sample": float(metrics["seconds_per_sample"]),
                "inference_peak_memory_mb": "",
                "inference_memory_status": "not_recorded_by_formal_evaluator",
                **flatten(metrics),
            }
            rows.append(row)

            source_metrics = metrics["per_source"]
            for source_id, item in sorted(
                source_metrics.items(), key=lambda pair: int(pair[0])
            ):
                per_mat_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "source_id": int(source_id),
                        "mat_name": item["source"]["name"],
                        "relative_path": item["source"]["relative_path"],
                        **flatten(item),
                    }
                )
                for point in item["error_growth_curve"]:
                    growth_rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "scope": "per_mat",
                            "source_id": int(source_id),
                            "mat_name": item["source"]["name"],
                            **point,
                        }
                    )
            for point in metrics["error_growth_curve"]:
                growth_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "scope": "aggregate",
                        "source_id": "",
                        "mat_name": "",
                        **point,
                    }
                )

            provenance_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "eval_dir": str(directory),
                    "checkpoint": payload["evaluated_checkpoint"]["path"],
                    "checkpoint_sha256": payload["evaluated_checkpoint"]["sha256"],
                    "val_metrics": str(metrics_path),
                    "val_metrics_sha256": sha256(metrics_path),
                    "split_sha256": split_hash,
                    "scientific_config_sha256": payload[
                        "scientific_config_sha256"
                    ],
                    "rfno_requires_grad_false": payload["freeze_checks"][
                        "rfno_requires_grad_false"
                    ],
                    "rfno_gradients_none": payload["freeze_checks"][
                        "rfno_gradients_none"
                    ],
                }
            )

    if len(split_hashes) != 1:
        raise RuntimeError(f"V0/V1 split mismatch: {sorted(split_hashes)}")

    metric_names = [
        key
        for key in rows[0]
        if key.startswith("reconstruction_") or key.startswith("forecast_")
    ]
    aggregate_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        aggregate: dict[str, Any] = {"variant": variant, "n_seeds": len(selected)}
        for metric in metric_names + ["inference_seconds", "seconds_per_sample"]:
            samples = [float(row[metric]) for row in selected]
            aggregate[f"{metric}_mean"] = statistics.fmean(samples)
            aggregate[f"{metric}_sample_std"] = statistics.stdev(samples)
        aggregate_rows.append(aggregate)

    paired_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        v0 = next(
            row for row in rows if row["variant"] == "V0" and row["seed"] == seed
        )
        v1 = next(
            row for row in rows if row["variant"] == "V1" and row["seed"] == seed
        )
        paired_rows.append(
            {
                "seed": seed,
                "difference": "V1_minus_V0",
                **{
                    metric: float(v1[metric]) - float(v0[metric])
                    for metric in metric_names
                },
            }
        )

    critical = float(student_t.ppf(0.975, len(SEEDS) - 1))
    ci_rows: list[dict[str, Any]] = []
    for metric in metric_names:
        samples = [float(row[metric]) for row in paired_rows]
        mean = statistics.fmean(samples)
        std = statistics.stdev(samples)
        half_width = critical * std / math.sqrt(len(samples))
        low, high = mean - half_width, mean + half_width
        direction = "higher" if metric.endswith("_corr") else "lower"
        significant = low > 0.0 or high < 0.0
        favors_v1 = significant and (
            (direction == "lower" and high < 0.0)
            or (direction == "higher" and low > 0.0)
        )
        ci_rows.append(
            {
                "metric": metric,
                "difference": "V1_minus_V0",
                "paired_mean": mean,
                "paired_sample_std": std,
                "student_t_df": len(SEEDS) - 1,
                "ci95_low": low,
                "ci95_high": high,
                "significant_95": significant,
                "favors_v1": favors_v1,
            }
        )

    worst_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            selected = [
                row
                for row in per_mat_rows
                if row["variant"] == variant and row["seed"] == seed
            ]
            for metric in metric_names:
                if metric.endswith("_corr"):
                    worst = min(selected, key=lambda item: float(item[metric]))
                else:
                    worst = max(selected, key=lambda item: float(item[metric]))
                worst_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "metric": metric,
                        "worst_value": worst[metric],
                        "source_id": worst["source_id"],
                        "mat_name": worst["mat_name"],
                    }
                )

    write_csv(output / "summary_by_seed.csv", rows)
    write_csv(output / "summary_three_seed.csv", aggregate_rows)
    write_csv(output / "paired_v1_minus_v0_by_seed.csv", paired_rows)
    write_csv(output / "paired_v1_minus_v0_ci95.csv", ci_rows)
    write_csv(output / "per_mat_metrics.csv", per_mat_rows)
    write_csv(output / "per_mat_worst.csv", worst_rows)
    write_csv(output / "error_growth_curves.csv", growth_rows)
    write_csv(output / "provenance.csv", provenance_rows)

    manifest = {
        "experiment_id": "MIN-V0-random-init-val300",
        "category": "evaluation_only",
        "split": "val",
        "test_accessed": False,
        "test_loader_constructed": False,
        "teacher_forcing": False,
        "future_context_updates": False,
        "difference": "V1_minus_V0",
        "seeds": list(SEEDS),
        "split_sha256": next(iter(split_hashes)),
        "inference_memory_status": (
            "not_recorded_by_formal_evaluator; no value was inferred or substituted"
        ),
        "paired_ci95": ci_rows,
        "artifacts": [
            "summary_by_seed.csv",
            "summary_three_seed.csv",
            "paired_v1_minus_v0_by_seed.csv",
            "paired_v1_minus_v0_ci95.csv",
            "per_mat_metrics.csv",
            "per_mat_worst.csv",
            "error_growth_curves.csv",
            "provenance.csv",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
