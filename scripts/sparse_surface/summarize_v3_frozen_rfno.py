#!/usr/bin/env python3
"""Summarize paired V1-initialization versus V3 frozen-RFNO validation results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

import torch
from scipy.stats import t as student_t


SEEDS = (42, 43, 44)
VARIANTS = ("V1_init", "V3")
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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def train_dir(root: Path, variant: str, seed: int) -> Path:
    if variant == "V1_init":
        name = (
            f"v1_vit_mae_pretrained_sparse_finetune_seed{seed}_"
            "30ep_formal_20260729_v1"
        )
    else:
        name = f"v3_vit_mae_frozen_rfno_30f_seed{seed}_formal_20260729_v1"
    return root / "runs/sparse" / name


def eval_dir(root: Path, variant: str, seed: int) -> Path:
    if variant == "V1_init":
        name = (
            f"v1_vit_mae_pretrained_sparse_finetune_seed{seed}_"
            "val300_20260729_v1"
        )
    else:
        name = f"v3_vit_mae_frozen_rfno_30f_seed{seed}_val300_20260729_v1"
    return root / "runs/sparse" / name


def training_summary(path: Path) -> dict[str, Any]:
    candidates = list(path.glob("training_summary_epoch*.json"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one training summary: {path}")
    return read_json(candidates[0])


def flatten(metrics: Mapping[str, Any]) -> dict[str, float]:
    history = metrics["history_reconstruction"]
    result = {
        "reconstruction_full_nrmse": float(history["nrmse"]),
        "reconstruction_full_corr": float(history["correlation"]),
        "reconstruction_full_ssp": float(history["ssp"]),
        "reconstruction_full_gradient_nrmse": float(
            history["gradient_nrmse"]
        ),
        "reconstruction_missing_nrmse": float(
            history["missing_region"]["nrmse"]
        ),
        "reconstruction_missing_corr": float(
            history["missing_region"]["correlation"]
        ),
        "observation_nrmse": float(history["observed_region"]["nrmse"]),
    }
    for horizon in HORIZONS:
        values = metrics[f"forecast_{horizon}"]
        result.update(
            {
                f"forecast_{horizon}_nrmse": float(values["nrmse"]),
                f"forecast_{horizon}_corr": float(values["correlation"]),
                f"forecast_{horizon}_ssp": float(values["ssp"]),
                f"forecast_{horizon}_gradient_nrmse": float(
                    values["gradient_nrmse"]
                ),
            }
        )
    return result


def checkpoint_contract(
    v1_checkpoint: Path, best_path: Path, last_path: Path
) -> dict[str, Any]:
    try:
        initial = torch.load(v1_checkpoint, map_location="cpu", weights_only=False)
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        last = torch.load(last_path, map_location="cpu", weights_only=False)
    except TypeError:
        initial = torch.load(v1_checkpoint, map_location="cpu")
        best = torch.load(best_path, map_location="cpu")
        last = torch.load(last_path, map_location="cpu")
    rfno_keys = [
        key for key in best["model_state_dict"] if key.startswith("rfno.")
    ]
    unchanged = all(
        torch.equal(best["model_state_dict"][key], initial["model_state_dict"][key])
        and torch.equal(last["model_state_dict"][key], initial["model_state_dict"][key])
        for key in rfno_keys
    )
    required = {
        "schema_version",
        "scientific_config_sha256",
        "epoch",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "curriculum_state",
        "data_loader_generator_state",
        "rng_state",
        "complete_training_state",
        "metrics",
    }
    return {
        "rfno_tensor_count": len(rfno_keys),
        "rfno_unchanged_in_best_and_last": unchanged,
        "last_has_complete_resume_state": required.issubset(last)
        and bool(last["complete_training_state"]),
        "last_epoch": int(last["epoch"]),
        "last_global_step": int(last["global_step"]),
        "last_active_rollout_steps": int(
            last["curriculum_state"]["active_rollout_steps"]
        ),
        "optimizer_group_count": len(
            last["optimizer_state_dict"]["param_groups"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    summary_rows: list[dict[str, Any]] = []
    per_mat_rows: list[dict[str, Any]] = []
    error_growth_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    contracts: dict[int, dict[str, Any]] = {}

    for seed in SEEDS:
        for variant in VARIANTS:
            training_path = train_dir(root, variant, seed)
            evaluation_path = eval_dir(root, variant, seed)
            training = training_summary(training_path)
            evaluation_file = evaluation_path / "evaluation/val_metrics.json"
            evaluation = read_json(evaluation_file)
            metrics = evaluation["metrics"]
            if (
                evaluation["split"] != "val"
                or int(metrics["rollout_steps"]) != 300
                or int(metrics["evaluated_samples"]) != 777
                or int(metrics["source_count"]) != 7
            ):
                raise RuntimeError(f"Incomplete val-300: {variant}/{seed}")
            freeze = evaluation["freeze_checks"]
            if not (
                freeze["rfno_requires_grad_false"]
                and freeze["rfno_gradients_none"]
            ):
                raise RuntimeError(f"Freeze check failed: {variant}/{seed}")
            if (
                int(training["start_epoch"]) != 1
                or int(training["last_epoch"]) != 30
                or bool(training["stopped_early"])
                or training["safety_stop_reason"] is not None
            ):
                raise RuntimeError(f"Training contract failed: {variant}/{seed}")
            if variant == "V3" and int(
                training["last_record"]["active_rollout_steps"]
            ) != 30:
                raise RuntimeError(f"V3 rollout contract failed: {seed}")

            best_path = training_path / "checkpoints/best.pt"
            row = {
                "variant": variant,
                "seed": seed,
                "best_epoch": int(training["best_epoch"]),
                "training_seconds": float(training["elapsed_seconds"]),
                "peak_memory_mb": float(
                    training["peak_memory_allocated_mb"]
                ),
                "inference_seconds": float(metrics["elapsed_seconds"]),
                "seconds_per_sample": float(metrics["seconds_per_sample"]),
                "parameters_total": int(evaluation["parameters"]["total"]),
                "parameters_trainable": int(
                    evaluation["parameters"]["trainable"]
                ),
                "checkpoint_sha256": sha256(best_path),
                **flatten(metrics),
            }
            summary_rows.append(row)

            for source_id, source_metrics in sorted(
                metrics["per_source"].items(), key=lambda item: int(item[0])
            ):
                source = source_metrics["source"]
                per_mat_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "source_id": int(source_id),
                        "mat_name": source["name"],
                        "relative_path": source["relative_path"],
                        **flatten(source_metrics),
                    }
                )
                for point in source_metrics["error_growth_curve"]:
                    error_growth_rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "scope": "per_mat",
                            "source_id": int(source_id),
                            "mat_name": source["name"],
                            **point,
                        }
                    )
            for point in metrics["error_growth_curve"]:
                error_growth_rows.append(
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
                    "train_dir": str(training_path),
                    "eval_dir": str(evaluation_path),
                    "checkpoint": str(best_path),
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "val_metrics_sha256": sha256(evaluation_file),
                }
            )

        v1_checkpoint = train_dir(root, "V1_init", seed) / "checkpoints/best.pt"
        v3_training = train_dir(root, "V3", seed)
        contracts[seed] = checkpoint_contract(
            v1_checkpoint,
            v3_training / "checkpoints/best.pt",
            v3_training / "checkpoints/last.pt",
        )

    metric_names = [
        key
        for key, value in summary_rows[0].items()
        if isinstance(value, (int, float))
        and key
        not in {
            "seed",
            "best_epoch",
            "parameters_total",
            "parameters_trainable",
        }
    ]
    aggregate_rows = []
    for variant in VARIANTS:
        candidates = [
            row for row in summary_rows if row["variant"] == variant
        ]
        aggregate: dict[str, Any] = {
            "variant": variant,
            "n_seeds": len(candidates),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in candidates]
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_sample_std"] = statistics.stdev(values)
        aggregate_rows.append(aggregate)

    paired_rows = []
    for seed in SEEDS:
        v1 = next(
            row
            for row in summary_rows
            if row["variant"] == "V1_init" and row["seed"] == seed
        )
        v3 = next(
            row
            for row in summary_rows
            if row["variant"] == "V3" and row["seed"] == seed
        )
        paired_rows.append(
            {
                "seed": seed,
                "difference": "V3_minus_V1_init",
                **{
                    metric: float(v3[metric]) - float(v1[metric])
                    for metric in metric_names
                },
            }
        )

    critical = float(student_t.ppf(0.975, len(SEEDS) - 1))
    ci_rows = []
    for metric in metric_names:
        values = [float(row[metric]) for row in paired_rows]
        mean = statistics.fmean(values)
        std = statistics.stdev(values)
        margin = critical * std / math.sqrt(len(values))
        ci_rows.append(
            {
                "metric": metric,
                "difference": "V3_minus_V1_init",
                "paired_mean": mean,
                "paired_sample_std": std,
                "paired_t_ci95_low": mean - margin,
                "paired_t_ci95_high": mean + margin,
                "ci_excludes_zero": mean - margin > 0.0
                or mean + margin < 0.0,
                "n_pairs": len(values),
            }
        )

    f30_differences = [
        float(row["forecast_30_nrmse"]) for row in paired_rows
    ]
    contract_pass = all(
        item["rfno_unchanged_in_best_and_last"]
        and item["last_has_complete_resume_state"]
        and item["last_epoch"] == 30
        and item["last_global_step"] == 18_330
        and item["last_active_rollout_steps"] == 30
        and item["optimizer_group_count"] == 1
        for item in contracts.values()
    )
    allow_v4 = contract_pass and all(value < 0.0 for value in f30_differences)

    write_csv(output / "summary_by_seed.csv", summary_rows)
    write_csv(output / "summary_three_seed.csv", aggregate_rows)
    write_csv(output / "paired_v3_minus_v1_by_seed.csv", paired_rows)
    write_csv(output / "paired_v3_minus_v1_ci95.csv", ci_rows)
    write_csv(output / "per_mat_metrics.csv", per_mat_rows)
    write_csv(output / "error_growth_curves.csv", error_growth_rows)
    write_csv(output / "provenance.csv", provenance_rows)
    manifest = {
        "experiment_id": "V3",
        "category": "learned_reconstruction_frozen",
        "seeds": list(SEEDS),
        "difference": "V3_minus_V1_init",
        "split": "val",
        "test_accessed": False,
        "teacher_forcing": False,
        "model_input_fields": ["x_obs", "obs_mask"],
        "contracts": contracts,
        "paired_ci95": ci_rows,
        "v4_gate": {
            "criterion": (
                "all three seeds improve forecast_30_nrmse and all frozen/checkpoint "
                "contracts pass"
            ),
            "forecast_30_nrmse_differences": f30_differences,
            "contract_pass": contract_pass,
            "allow_v4": allow_v4,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["v4_gate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
