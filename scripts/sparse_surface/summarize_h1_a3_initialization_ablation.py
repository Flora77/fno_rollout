#!/usr/bin/env python3
"""Summarize the strict three-seed H1-A3 initialization ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping


SEEDS = (42, 43, 44)
VARIANTS = ("pretrained", "random_init")
HORIZONS = (30, 60, 120, 180, 240, 300)
# Student-t 97.5th percentile with df=2 for a paired three-seed interval.
T_CRITICAL_DF2_95 = 4.302652729911275


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


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


def _flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    history = metrics["history_reconstruction"]
    result = {
        "reconstruction_missing_nrmse": float(
            history["missing_region"]["nrmse"]
        )
    }
    for horizon in HORIZONS:
        result[f"forecast_{horizon}_nrmse"] = float(
            metrics[f"forecast_{horizon}"]["nrmse"]
        )
    forecast_300 = metrics["forecast_300"]
    result.update(
        {
            "forecast_300_corr": float(forecast_300["correlation"]),
            "forecast_300_ssp": float(forecast_300["ssp"]),
            "forecast_300_gradient_nrmse": float(
                forecast_300["gradient_nrmse"]
            ),
        }
    )
    return result


def _train_dir(run_root: Path, variant: str, seed: int) -> Path:
    return (
        run_root
        / f"h1_a3_{variant}_seed{seed}_30ep_fixed_r05_mask_00006"
    )


def _eval_dir(run_root: Path, variant: str, seed: int) -> Path:
    return (
        run_root
        / f"h1_a3_{variant}_seed{seed}_val300_fixed_r05_mask_00006"
    )


def summarize(run_root: Path, config_root: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        pair_configs: dict[str, dict[str, Any]] = {}
        for variant in VARIANTS:
            config_path = config_root / f"{variant}_seed{seed}.formal.json"
            pair_configs[variant] = _read_json(config_path)
            train_dir = _train_dir(run_root, variant, seed)
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
                raise ValueError(f"Not a fresh complete 30-epoch run: {train_dir}")

            evaluation = _read_json(
                _eval_dir(run_root, variant, seed)
                / "evaluation"
                / "val_metrics.json"
            )
            metrics = evaluation["metrics"]
            if (
                int(metrics["rollout_steps"]) != 300
                or int(metrics["evaluated_samples"]) != 777
            ):
                raise ValueError(f"Incomplete val-300 evaluation: {variant}/{seed}")
            freeze = evaluation["freeze_checks"]
            if not (
                freeze["rfno_requires_grad_false"]
                and freeze["rfno_gradients_none"]
            ):
                raise ValueError(f"RFNO freeze check failed: {variant}/{seed}")

            row: dict[str, Any] = {
                "variant": variant,
                "seed": seed,
                "best_epoch": int(training["best_epoch"]),
                "training_seconds": float(training["elapsed_seconds"]),
                "inference_seconds": float(metrics["elapsed_seconds"]),
            }
            row.update(_flatten_metrics(metrics))
            rows.append(row)

        pretrained = pair_configs["pretrained"]
        random_init = pair_configs["random_init"]
        checkpoint = pretrained["pipeline"].pop("reconstructor_checkpoint", None)
        if not checkpoint:
            raise ValueError(f"Pretrained seed {seed} lacks initialization checkpoint")
        if "reconstructor_checkpoint" in random_init["pipeline"]:
            raise ValueError(f"Random-init seed {seed} unexpectedly loads a checkpoint")
        pretrained.pop("experiment_name")
        random_init.pop("experiment_name")
        if pretrained != random_init:
            raise ValueError(f"Scientific pair mismatch for seed {seed}")

    metric_names = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float))
        and key not in {"seed", "best_epoch"}
    ]
    aggregate: list[dict[str, Any]] = []
    for variant in VARIANTS:
        candidates = [row for row in rows if row["variant"] == variant]
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
            pretrained = next(
                row
                for row in rows
                if row["variant"] == "pretrained" and row["seed"] == seed
            )
            random_init = next(
                row
                for row in rows
                if row["variant"] == "random_init" and row["seed"] == seed
            )
            differences.append(
                float(pretrained[metric]) - float(random_init[metric])
            )
        mean = statistics.fmean(differences)
        sample_std = statistics.stdev(differences)
        margin = T_CRITICAL_DF2_95 * sample_std / math.sqrt(len(differences))
        low, high = mean - margin, mean + margin
        paired.append(
            {
                "metric": metric,
                "difference": "pretrained_minus_random_init",
                "paired_mean": mean,
                "paired_sample_std": sample_std,
                "paired_t_ci95_low": low,
                "paired_t_ci95_high": high,
                "ci_excludes_zero": low > 0.0 or high < 0.0,
                "n_pairs": len(differences),
            }
        )

    _write_csv(output_dir / "summary_by_seed.csv", rows)
    _write_csv(output_dir / "summary_three_seed.csv", aggregate)
    _write_csv(output_dir / "paired_ci95.csv", paired)
    manifest = {
        "experiment_id": "H1-A3",
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "difference_definition": "pretrained_minus_random_init",
        "confidence_interval": "paired Student-t, df=2, two-sided 95%",
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
        default=Path("config/sparse_experiments/h1_a3"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run_root, args.config_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
