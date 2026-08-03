#!/usr/bin/env python3
"""Summarize the strict three-seed H1-A4 versus H1-A5 experiment."""

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

from scipy.stats import t as student_t


SEEDS = (42, 43, 44)
VARIANTS = ("H1-A4", "H1-A5")
HORIZONS = (30, 60, 120, 180, 240, 300)
T_CRITICAL_DF2_95 = 4.302652729911275


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _config_path(config_root: Path, variant: str, seed: int) -> Path:
    if variant == "H1-A4":
        return config_root / "h1_a4" / f"pretrained_seed{seed}.formal.json"
    return config_root / "h1_a5" / f"random_init_seed{seed}.formal.json"


def _train_dir(run_root: Path, variant: str, seed: int) -> Path:
    prefix = "h1_a4_pretrained" if variant == "H1-A4" else "h1_a5_random_init"
    return run_root / f"{prefix}_seed{seed}_30ep_fixed_r05_mask_00006"


def _eval_dir(
    run_root: Path, variant: str, seed: int, evaluation_suffix: str
) -> Path:
    prefix = "h1_a4_pretrained" if variant == "H1-A4" else "h1_a5_random_init"
    return run_root / f"{prefix}_seed{seed}_{evaluation_suffix}"


def _flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    result = {
        "reconstruction_missing_nrmse": float(
            metrics["history_reconstruction"]["missing_region"]["nrmse"]
        )
    }
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


def _pair_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(config))
    payload.pop("experiment_id")
    payload.pop("experiment_name")
    payload["pipeline"].pop("reconstructor_checkpoint", None)
    return payload


def summarize(
    run_root: Path,
    config_root: Path,
    output_dir: Path,
    evaluation_suffix: str,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    by_seed: list[dict[str, Any]] = []
    per_mat: list[dict[str, Any]] = []
    learning_curves: list[dict[str, Any]] = []
    checkpoint_manifest: list[dict[str, Any]] = []

    for seed in SEEDS:
        pair_configs = {
            variant: _read_json(_config_path(config_root, variant, seed))
            for variant in VARIANTS
        }
        pretrained_checkpoint = pair_configs["H1-A4"]["pipeline"].get(
            "reconstructor_checkpoint"
        )
        if not pretrained_checkpoint:
            raise ValueError(f"H1-A4 seed {seed} lacks reconstructor_checkpoint")
        if "reconstructor_checkpoint" in pair_configs["H1-A5"]["pipeline"]:
            raise ValueError(f"H1-A5 seed {seed} unexpectedly loads a checkpoint")
        if _pair_payload(pair_configs["H1-A4"]) != _pair_payload(
            pair_configs["H1-A5"]
        ):
            raise ValueError(f"Scientific pair mismatch for seed {seed}")

        for variant in VARIANTS:
            train_dir = _train_dir(run_root, variant, seed)
            summaries = list(train_dir.glob("training_summary_epoch*.json"))
            if len(summaries) != 1:
                raise ValueError(f"Expected one formal training summary in {train_dir}")
            training = _read_json(summaries[0])
            if (
                int(training["start_epoch"]) != 1
                or int(training["completed_epochs"]) != 30
                or int(training["last_epoch"]) != 30
                or bool(training["stopped_early"])
                or training.get("continuation") is not None
                or training.get("run_kind") != "formal"
            ):
                raise ValueError(f"Not a fresh complete 30-epoch run: {train_dir}")

            evaluation = _read_json(
                _eval_dir(run_root, variant, seed, evaluation_suffix)
                / "evaluation"
                / "val_metrics.json"
            )
            metrics = evaluation["metrics"]
            if (
                evaluation.get("evaluation_kind") != "formal"
                or int(metrics["rollout_steps"]) != 300
                or int(metrics["evaluated_samples"]) != 777
            ):
                raise ValueError(f"Incomplete formal val-300: {variant}/{seed}")
            freeze = evaluation["freeze_checks"]
            if not (
                freeze["rfno_requires_grad_false"]
                and freeze["rfno_gradients_none"]
            ):
                raise ValueError(f"RFNO freeze check failed: {variant}/{seed}")

            checkpoint = train_dir / "checkpoints" / "best.pt"
            checkpoint_hash = _sha256(checkpoint)
            if checkpoint_hash != evaluation["evaluated_checkpoint"]["sha256"]:
                raise ValueError(f"Evaluated checkpoint mismatch: {variant}/{seed}")

            row: dict[str, Any] = {
                "variant": variant,
                "seed": seed,
                "best_epoch": int(training["best_epoch"]),
                "training_seconds": float(training["elapsed_seconds"]),
                "peak_memory_allocated_mb": float(
                    training["peak_memory_allocated_mb"]
                ),
                "peak_memory_reserved_mb": float(
                    training["peak_memory_reserved_mb"]
                ),
                "inference_seconds": float(metrics["elapsed_seconds"]),
                "seconds_per_sample": float(metrics["seconds_per_sample"]),
                "total_parameters": int(evaluation["parameters"]["total"]),
                "trainable_parameters": int(evaluation["parameters"]["trainable"]),
            }
            row.update(_flatten_metrics(metrics))
            by_seed.append(row)
            checkpoint_manifest.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "best_epoch": int(training["best_epoch"]),
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": checkpoint_hash,
                    "scientific_config_sha256": evaluation[
                        "scientific_config_sha256"
                    ],
                }
            )

            for source_id, source_metrics in metrics["per_source"].items():
                source = source_metrics["source"]
                source_row: dict[str, Any] = {
                    "variant": variant,
                    "seed": seed,
                    "source_id": int(source_id),
                    "mat_name": source["name"],
                    "relative_path": source["relative_path"],
                }
                source_row.update(_flatten_metrics(source_metrics))
                per_mat.append(source_row)

            epochs = _read_jsonl(train_dir / "logs" / "epochs.jsonl")
            if len(epochs) != 30:
                raise ValueError(f"Expected 30 learning-curve rows: {train_dir}")
            for epoch in epochs:
                curve_row: dict[str, Any] = {
                    "variant": variant,
                    "seed": seed,
                    "epoch": int(epoch["epoch"]),
                    "global_step": int(epoch["global_step"]),
                    "learning_rate": float(epoch["learning_rates"][0]),
                    "best_metric": float(epoch["best_metric"]),
                    "best_epoch": int(epoch["best_epoch"]),
                }
                for split in ("train", "val"):
                    for key, value in epoch[split].items():
                        if isinstance(value, (int, float)):
                            curve_row[f"{split}_{key}"] = value
                learning_curves.append(curve_row)

    metric_names = [
        key
        for key, value in by_seed[0].items()
        if isinstance(value, (int, float)) and key not in {"seed", "best_epoch"}
    ]
    aggregate: list[dict[str, Any]] = []
    for variant in VARIANTS:
        candidates = [row for row in by_seed if row["variant"] == variant]
        for metric in metric_names:
            values = [float(row[metric]) for row in candidates]
            aggregate.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "mean": statistics.fmean(values),
                    "sample_std": statistics.stdev(values),
                    "n_seeds": len(values),
                }
            )

    paired: list[dict[str, Any]] = []
    for metric in metric_names:
        differences = []
        for seed in SEEDS:
            a4 = next(
                row
                for row in by_seed
                if row["variant"] == "H1-A4" and row["seed"] == seed
            )
            a5 = next(
                row
                for row in by_seed
                if row["variant"] == "H1-A5" and row["seed"] == seed
            )
            differences.append(float(a4[metric]) - float(a5[metric]))
        mean = statistics.fmean(differences)
        sample_std = statistics.stdev(differences)
        standard_error = sample_std / math.sqrt(len(differences))
        margin = T_CRITICAL_DF2_95 * standard_error
        low, high = mean - margin, mean + margin
        if standard_error == 0.0:
            p_value = 1.0 if mean == 0.0 else 0.0
        else:
            statistic = mean / standard_error
            p_value = float(2.0 * student_t.sf(abs(statistic), df=2))
        paired.append(
            {
                "metric": metric,
                "difference": "H1-A4_minus_H1-A5",
                "paired_mean": mean,
                "paired_sample_std": sample_std,
                "paired_t_ci95_low": low,
                "paired_t_ci95_high": high,
                "paired_t_p_value": p_value,
                "ci_excludes_zero": low > 0.0 or high < 0.0,
                "n_pairs": len(differences),
            }
        )

    _write_csv(output_dir / "summary_by_seed.csv", by_seed)
    _write_csv(output_dir / "summary_three_seed.csv", aggregate)
    _write_csv(output_dir / "paired_ci95.csv", paired)
    _write_csv(output_dir / "per_mat_metrics.csv", per_mat)
    _write_csv(output_dir / "learning_curves.csv", learning_curves)
    _write_csv(output_dir / "checkpoint_manifest.csv", checkpoint_manifest)
    manifest = {
        "experiment_ids": list(VARIANTS),
        "seeds": list(SEEDS),
        "difference_definition": "H1-A4_minus_H1-A5",
        "confidence_interval": "paired Student-t, df=2, two-sided 95%",
        "evaluation_suffix": evaluation_suffix,
        "train_epochs_per_run": 30,
        "rollout_steps": 300,
        "evaluated_samples_per_run": 777,
        "test_accessed": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/sparse"))
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("config/sparse_experiments"),
    )
    parser.add_argument(
        "--evaluation-suffix",
        default="val300_20260728_v1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(
        args.run_root,
        args.config_root,
        args.output_dir,
        args.evaluation_suffix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
