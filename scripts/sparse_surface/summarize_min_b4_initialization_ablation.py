#!/usr/bin/env python3
"""Summarize the strict B4 masked-pretrained versus random initialization ablation."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

import torch
from scipy.stats import t as student_t

from neuralop.training.sparse_experiment_runner import (
    build_sparse_pipeline,
    load_resolved_sparse_config,
)


SEEDS = (42, 43, 44)
VARIANTS = ("random", "pretrained")
HORIZONS = (30, 60, 120, 180, 240, 300)
REQUIRED_CHECKPOINT_KEYS = {
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
    "rng_state",
    "data_loader_generator_state",
    "curriculum_state",
    "complete_training_state",
}


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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def config_path(root: Path, variant: str, seed: int) -> Path:
    return (
        root
        / "config"
        / "sparse_experiments"
        / f"min_b4_init_ablation_{variant}_seed{seed}.formal.json"
    )


def train_dir(root: Path, variant: str, seed: int) -> Path:
    return (
        root
        / "runs"
        / "sparse"
        / f"min_b4_init_ablation_{variant}_seed{seed}_30ep_formal_20260729_v1"
    )


def eval_dir(root: Path, variant: str, seed: int) -> Path:
    return (
        root
        / "runs"
        / "sparse"
        / f"min_b4_init_ablation_{variant}_seed{seed}_val300_20260729_v1"
    )


def pretrain_dir(root: Path, seed: int) -> Path:
    return (
        root
        / "runs"
        / "sparse"
        / f"min_b4_mask_unet_masked_pretrain_seed{seed}_formal_20260729_v1"
    )


def training_summary(directory: Path) -> dict[str, Any]:
    candidates = list(directory.glob("training_summary_epoch*.json"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one training summary: {directory}")
    return read_json(candidates[0])


def pair_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    value.pop("experiment_name")
    value["pipeline"].pop("reconstructor_checkpoint", None)
    return value


def flatten(metrics: Mapping[str, Any]) -> dict[str, float]:
    history = metrics["history_reconstruction"]
    result = {
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
        result.update(
            {
                f"forecast_{horizon}_nrmse": float(item["nrmse"]),
                f"forecast_{horizon}_corr": float(item["correlation"]),
                f"forecast_{horizon}_ssp": float(item["ssp"]),
                f"forecast_{horizon}_gradient_nrmse": float(
                    item["gradient_nrmse"]
                ),
            }
        )
    return result


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
    curve_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    split_hashes: set[str] = set()

    for seed in SEEDS:
        configs = {
            variant: read_json(config_path(root, variant, seed))
            for variant in VARIANTS
        }
        if pair_contract(configs["random"]) != pair_contract(
            configs["pretrained"]
        ):
            raise RuntimeError(f"Pair config mismatch: seed{seed}")
        if "reconstructor_checkpoint" in configs["random"]["pipeline"]:
            raise RuntimeError(f"Random run loads checkpoint: seed{seed}")
        initialization = configs["pretrained"]["pipeline"].get(
            "reconstructor_checkpoint"
        )
        if not initialization:
            raise RuntimeError(f"Pretrained run lacks initialization: seed{seed}")

        pretrain = training_summary(pretrain_dir(root, seed))
        for variant in VARIANTS:
            training_directory = train_dir(root, variant, seed)
            evaluation_directory = eval_dir(root, variant, seed)
            training = training_summary(training_directory)
            if (
                training["run_kind"] != "formal"
                or int(training["start_epoch"]) != 1
                or int(training["completed_epochs"]) != 30
                or int(training["last_epoch"]) != 30
                or int(training["global_step"]) != 18_330
                or bool(training["stopped_early"])
                or training["safety_stop_reason"] is not None
                or training["continuation"] is not None
            ):
                raise RuntimeError(f"Fresh 30-epoch contract failed: {variant}/{seed}")

            metrics_path = evaluation_directory / "evaluation/val_metrics.json"
            evaluation = read_json(metrics_path)
            metrics = evaluation["metrics"]
            if (
                evaluation["split"] != "val"
                or evaluation["evaluation_kind"] != "formal"
                or int(metrics["evaluated_samples"]) != 777
                or int(metrics["rollout_steps"]) != 300
                or int(metrics["source_count"]) != 7
                or not all(evaluation["freeze_checks"].values())
            ):
                raise RuntimeError(f"Val/freeze contract failed: {variant}/{seed}")

            best = training_directory / "checkpoints/best.pt"
            last = training_directory / "checkpoints/last.pt"
            if sha256(best) != evaluation["evaluated_checkpoint"]["sha256"]:
                raise RuntimeError(f"Evaluated checkpoint mismatch: {variant}/{seed}")
            try:
                best_state = torch.load(best, map_location="cpu", weights_only=False)
                last_state = torch.load(last, map_location="cpu", weights_only=False)
            except TypeError:
                best_state = torch.load(best, map_location="cpu")
                last_state = torch.load(last, map_location="cpu")
            if not (
                REQUIRED_CHECKPOINT_KEYS.issubset(best_state)
                and REQUIRED_CHECKPOINT_KEYS.issubset(last_state)
                and bool(best_state["complete_training_state"])
                and bool(last_state["complete_training_state"])
            ):
                raise RuntimeError(f"Incomplete checkpoint: {variant}/{seed}")

            resolved = load_resolved_sparse_config(
                config_path(root, variant, seed),
                project_root=root,
                run_dir=root / "runs/sparse" / f"_audit_{variant}_{seed}",
                device="cpu",
            )
            initial_pipeline = build_sparse_pipeline(
                resolved, torch.device("cpu")
            )
            initial_state = initial_pipeline.state_dict()
            rfno_keys = [
                key for key in initial_state if key.startswith("rfno.")
            ]
            rfno_unchanged = all(
                torch.equal(best_state["model_state_dict"][key], initial_state[key])
                and torch.equal(last_state["model_state_dict"][key], initial_state[key])
                for key in rfno_keys
            )
            if not rfno_unchanged:
                raise RuntimeError(f"RFNO state changed: {variant}/{seed}")

            split_hash = sha256(training_directory / "data_split.json")
            split_hashes.add(split_hash)
            cumulative_seconds = float(training["elapsed_seconds"])
            cumulative_peak = float(training["peak_memory_allocated_mb"])
            pretrain_seconds = 0.0
            if variant == "pretrained":
                pretrain_seconds = float(pretrain["elapsed_seconds"])
                cumulative_seconds += pretrain_seconds
                cumulative_peak = max(
                    cumulative_peak, float(pretrain["peak_memory_allocated_mb"])
                )
            row = {
                "variant": variant,
                "seed": seed,
                "best_epoch": int(training["best_epoch"]),
                "downstream_seconds": float(training["elapsed_seconds"]),
                "pretraining_seconds": pretrain_seconds,
                "cumulative_training_seconds": cumulative_seconds,
                "peak_memory_allocated_mb": float(
                    training["peak_memory_allocated_mb"]
                ),
                "cumulative_peak_memory_mb": cumulative_peak,
                "inference_seconds": float(metrics["elapsed_seconds"]),
                "seconds_per_sample": float(metrics["seconds_per_sample"]),
                "total_parameters": int(evaluation["parameters"]["total"]),
                "trainable_parameters": int(
                    evaluation["parameters"]["trainable"]
                ),
                **flatten(metrics),
            }
            rows.append(row)
            checkpoint_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "initialization_checkpoint": (
                        "" if variant == "random" else str(Path(initialization).resolve())
                    ),
                    "initialization_sha256": (
                        "" if variant == "random" else sha256(Path(initialization))
                    ),
                    "best_checkpoint": str(best),
                    "best_sha256": sha256(best),
                    "last_checkpoint": str(last),
                    "last_sha256": sha256(last),
                    "best_epoch": int(training["best_epoch"]),
                    "last_epoch": int(training["last_epoch"]),
                    "global_step": int(training["global_step"]),
                    "rfno_unchanged": rfno_unchanged,
                    "best_complete_resume_state": bool(
                        best_state["complete_training_state"]
                    ),
                    "last_complete_resume_state": bool(
                        last_state["complete_training_state"]
                    ),
                    "metrics_sha256": sha256(metrics_path),
                    "split_sha256": split_hash,
                }
            )

            epochs = [
                json.loads(line)
                for line in (training_directory / "logs/epochs.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            if len(epochs) != 30:
                raise RuntimeError(f"Incomplete learning curve: {variant}/{seed}")
            for item in epochs:
                curve_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "epoch": int(item["epoch"]),
                        "global_step": int(item["global_step"]),
                        "learning_rate": float(item["learning_rates"][0]),
                        "train_total": float(item["train"]["total"]),
                        "val_total": float(item["val"]["total"]),
                        "val_missing_nrmse": float(
                            item["val"]["reconstruction_missing_nrmse"]
                        ),
                    }
                )

            for source_id, source_metrics in sorted(
                metrics["per_source"].items(), key=lambda pair: int(pair[0])
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
                    growth_rows.append(
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

    if len(split_hashes) != 1:
        raise RuntimeError("Split hashes differ across paired runs")

    metric_names = [
        key
        for key in rows[0]
        if key.startswith("reconstruction_") or key.startswith("forecast_")
    ]
    aggregate_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        aggregate: dict[str, Any] = {"variant": variant, "n_seeds": len(selected)}
        for metric in metric_names + [
            "downstream_seconds",
            "pretraining_seconds",
            "cumulative_training_seconds",
            "peak_memory_allocated_mb",
            "cumulative_peak_memory_mb",
            "inference_seconds",
            "seconds_per_sample",
        ]:
            values = [float(row[metric]) for row in selected]
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_sample_std"] = statistics.stdev(values)
        aggregate_rows.append(aggregate)

    paired_seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        random_row = next(
            row for row in rows if row["variant"] == "random" and row["seed"] == seed
        )
        pretrained_row = next(
            row
            for row in rows
            if row["variant"] == "pretrained" and row["seed"] == seed
        )
        paired_seed_rows.append(
            {
                "seed": seed,
                "difference": "pretrained_minus_random",
                **{
                    metric: float(pretrained_row[metric]) - float(random_row[metric])
                    for metric in metric_names
                },
            }
        )

    critical = float(student_t.ppf(0.975, len(SEEDS) - 1))
    ci_rows: list[dict[str, Any]] = []
    for metric in metric_names:
        values = [float(row[metric]) for row in paired_seed_rows]
        mean = statistics.fmean(values)
        std = statistics.stdev(values)
        half_width = critical * std / math.sqrt(len(values))
        low, high = mean - half_width, mean + half_width
        direction = "higher" if metric.endswith("_corr") else "lower"
        significant = low > 0.0 or high < 0.0
        favors_pretrained = significant and (
            (direction == "lower" and high < 0.0)
            or (direction == "higher" and low > 0.0)
        )
        ci_rows.append(
            {
                "metric": metric,
                "difference": "pretrained_minus_random",
                "paired_mean": mean,
                "paired_sample_std": std,
                "ci95_low": low,
                "ci95_high": high,
                "student_t_df": len(SEEDS) - 1,
                "significant_95": significant,
                "favors_pretrained": favors_pretrained,
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
                worst = (
                    min(selected, key=lambda item: float(item[metric]))
                    if metric.endswith("_corr")
                    else max(selected, key=lambda item: float(item[metric]))
                )
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
    write_csv(output / "paired_by_seed.csv", paired_seed_rows)
    write_csv(output / "paired_ci95.csv", ci_rows)
    write_csv(output / "checkpoint_manifest.csv", checkpoint_rows)
    write_csv(output / "learning_curves.csv", curve_rows)
    write_csv(output / "per_mat_metrics.csv", per_mat_rows)
    write_csv(output / "per_mat_worst.csv", worst_rows)
    write_csv(output / "error_growth_curves.csv", growth_rows)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "MIN-B4-pretraining-initialization-ablation",
                "category": "learned_reconstruction_frozen",
                "seeds": list(SEEDS),
                "difference": "pretrained_minus_random",
                "confidence_interval": "paired Student-t, two-sided 95%, df=2",
                "split": "val",
                "split_sha256": next(iter(split_hashes)),
                "test_accessed": False,
                "test_loader_constructed": False,
                "teacher_forcing": False,
                "future_context_updates": False,
                "train_epochs_per_run": 30,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
