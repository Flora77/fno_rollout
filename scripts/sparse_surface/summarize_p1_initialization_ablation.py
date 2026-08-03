#!/usr/bin/env python3
"""Summarize the paired three-seed P1 initialization ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


HORIZONS = (30, 60, 120, 180, 240, 300)
VARIANTS = ("pretrained", "random_init")


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


def _flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    history = metrics["history_reconstruction"]
    result = {
        "reconstruction_full_nrmse": float(history["nrmse"]),
        "reconstruction_full_corr": float(history["correlation"]),
        "reconstruction_full_ssp": float(history["ssp"]),
        "reconstruction_full_gradient_nrmse": float(history["gradient_nrmse"]),
        "reconstruction_missing_nrmse": float(history["missing_region"]["nrmse"]),
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


def _training_paths(run_root: Path, variant: str, seed: int) -> tuple[Path, Path]:
    stem = (
        f"p1_init_ablation_{variant}_seed{seed}_30ep_"
        "fixed_r05_mask_00006_20260722_v1"
    )
    train_dir = run_root / stem
    eval_dir = run_root / stem.replace("_30ep_", "_val300_")
    return train_dir, eval_dir


def _numeric_items(prefix: str, values: Mapping[str, Any]) -> dict[str, float]:
    return {
        f"{prefix}_{key}": float(value)
        for key, value in values.items()
        if isinstance(value, (int, float))
    }


def summarize(run_root: Path, output_dir: Path, seeds: Iterable[int]) -> None:
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for sample standard deviation")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    summary_rows: list[dict[str, Any]] = []
    per_mat_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []

    for seed in seeds:
        for variant in VARIANTS:
            train_dir, eval_dir = _training_paths(run_root, variant, seed)
            summaries = list(train_dir.glob("training_summary_epoch*.json"))
            if len(summaries) != 1:
                raise ValueError(f"Expected one training summary in {train_dir}")
            training = _read_json(summaries[0])
            if (
                int(training["start_epoch"]) != 1
                or int(training["completed_epochs"]) != 30
                or int(training["last_epoch"]) != 30
                or bool(training["stopped_early"])
            ):
                raise ValueError(f"Run is not a fresh complete 30-epoch run: {train_dir}")

            metrics_path = eval_dir / "evaluation" / "val_metrics.json"
            evaluation = _read_json(metrics_path)
            metrics = evaluation["metrics"]
            if int(metrics["rollout_steps"]) != 300:
                raise ValueError(f"Incomplete rollout in {metrics_path}")
            freeze = evaluation["freeze_checks"]
            if not (
                freeze["rfno_requires_grad_false"] and freeze["rfno_gradients_none"]
            ):
                raise ValueError(f"RFNO freeze check failed in {metrics_path}")

            checkpoint = train_dir / "checkpoints" / "best.pt"
            config = _read_json(train_dir / "config.resolved.json")
            initialization = config["pipeline"].get("reconstructor_checkpoint")
            row: dict[str, Any] = {
                "variant": variant,
                "seed": seed,
                "best_epoch": int(training["best_epoch"]),
                "best_val_forecast_30_nrmse": float(training["best_metric"]),
                "training_seconds": float(training["elapsed_seconds"]),
                "peak_memory_mb": float(training["peak_memory_allocated_mb"]),
                "inference_seconds": float(metrics["elapsed_seconds"]),
                "seconds_per_sample": float(metrics["seconds_per_sample"]),
                "evaluated_samples": int(metrics["evaluated_samples"]),
                "downstream_checkpoint_sha256": _sha256(checkpoint),
                "initialization_checkpoint_sha256": (
                    _sha256(Path(initialization)) if initialization else ""
                ),
            }
            row.update(_flatten_metrics(metrics))
            summary_rows.append(row)

            for source_id, source_metrics in sorted(
                metrics["per_source"].items(), key=lambda item: int(item[0])
            ):
                source = source_metrics["source"]
                source_row: dict[str, Any] = {
                    "variant": variant,
                    "seed": seed,
                    "source_id": int(source_id),
                    "mat_name": source["name"],
                    "relative_path": source["relative_path"],
                }
                source_row.update(_flatten_metrics(source_metrics))
                per_mat_rows.append(source_row)
                for point in source_metrics["error_growth_curve"]:
                    curve_rows.append(
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
                curve_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "scope": "aggregate",
                        "source_id": "",
                        "mat_name": "",
                        **point,
                    }
                )

            log_path = train_dir / "logs" / "epochs.jsonl"
            epochs = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(epochs) != 30:
                raise ValueError(f"Expected 30 learning-curve rows in {log_path}")
            for epoch in epochs:
                learning_row: dict[str, Any] = {
                    "variant": variant,
                    "seed": seed,
                    "epoch": int(epoch["epoch"]),
                    "global_step": int(epoch["global_step"]),
                    "learning_rate": float(epoch["learning_rates"][0]),
                }
                learning_row.update(_numeric_items("train", epoch["train"]))
                learning_row.update(_numeric_items("val", epoch["val"]))
                learning_rows.append(learning_row)

            provenance_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "train_dir": str(train_dir.resolve()),
                    "eval_dir": str(eval_dir.resolve()),
                    "scientific_config_sha256": evaluation[
                        "scientific_config_sha256"
                    ],
                    "downstream_checkpoint": str(checkpoint.resolve()),
                    "downstream_checkpoint_sha256": row[
                        "downstream_checkpoint_sha256"
                    ],
                    "initialization_checkpoint": initialization or "",
                    "initialization_checkpoint_sha256": row[
                        "initialization_checkpoint_sha256"
                    ],
                    "val_metrics_sha256": _sha256(metrics_path),
                }
            )

    metric_names = [
        key
        for key, value in summary_rows[0].items()
        if isinstance(value, (int, float))
        and key not in {"seed", "best_epoch", "evaluated_samples"}
    ]
    aggregate_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        candidates = [row for row in summary_rows if row["variant"] == variant]
        aggregate: dict[str, Any] = {"variant": variant, "n_seeds": len(candidates)}
        for metric in metric_names:
            values = [float(row[metric]) for row in candidates]
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_sample_std"] = statistics.stdev(values)
        aggregate_rows.append(aggregate)

    paired_rows: list[dict[str, Any]] = []
    for seed in seeds:
        pretrained = next(
            row
            for row in summary_rows
            if row["variant"] == "pretrained" and row["seed"] == seed
        )
        random_init = next(
            row
            for row in summary_rows
            if row["variant"] == "random_init" and row["seed"] == seed
        )
        paired = {"seed": seed, "difference": "pretrained_minus_random_init"}
        for metric in metric_names:
            paired[metric] = float(pretrained[metric]) - float(random_init[metric])
        paired_rows.append(paired)

    _write_csv(output_dir / "summary_by_seed.csv", summary_rows)
    _write_csv(output_dir / "summary_three_seed.csv", aggregate_rows)
    _write_csv(output_dir / "paired_seed_differences.csv", paired_rows)
    _write_csv(output_dir / "per_mat_metrics.csv", per_mat_rows)
    _write_csv(output_dir / "error_growth_curves.csv", curve_rows)
    _write_csv(output_dir / "learning_curves.csv", learning_rows)
    _write_csv(output_dir / "provenance.csv", provenance_rows)

    manifest = {
        "experiment_id": "P1-ablation-pretraining-initialization",
        "seeds": list(seeds),
        "variants": list(VARIANTS),
        "n_runs": len(summary_rows),
        "n_per_mat_rows": len(per_mat_rows),
        "n_learning_curve_rows": len(learning_rows),
        "n_error_growth_rows": len(curve_rows),
        "test_accessed": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/sparse"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    summarize(args.run_root, args.output_dir, args.seeds)
    print(f"wrote P1 initialization ablation summary to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
