#!/usr/bin/env python3
"""Summarize only the locked nonlinear B5 v2 GNO/GINO matrix."""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


T_CRITICAL_DF2_95 = 4.302652729911275
HORIZONS = ("history", 30, 60, 120, 180, 240, 300)
METRICS = ("nrmse", "correlation", "ssp", "gradient_nrmse")


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def _metric_block(payload: dict[str, Any], horizon: str | int) -> dict[str, float]:
    metrics = payload["metrics"]
    if horizon == "history":
        history = metrics["history_reconstruction"]
        return {
            "nrmse": float(history["missing_region"]["nrmse"]),
            "correlation": float(history["missing_region"]["correlation"]),
            "ssp": float(history["ssp"]),
            "gradient_nrmse": float(history["gradient_nrmse"]),
        }
    forecast = metrics[f"forecast_{horizon}"]
    return {key: float(forecast[key]) for key in METRICS}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _without_run_metadata(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result.pop("experiment_name", None)
    result["runtime"].pop("seed", None)
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    out = root / "results" / "min_b5_gno_gino_v2_three_seed_20260729_v1"
    status = json.loads((out / "matrix_status.json").read_text(encoding="utf-8"))
    restore_audit = json.loads(
        (out / "checkpoint_restore_audit.json").read_text(encoding="utf-8")
    )
    if len(status) != 6 or not restore_audit["passed"]:
        raise RuntimeError("The six-run matrix or checkpoint audit is incomplete")
    if any("b5_v2_" not in item["train_dir"] for item in status):
        raise RuntimeError("A non-v2/legacy run entered the matrix")

    configs = {
        (method, seed): json.loads(
            (
                root
                / "config"
                / "sparse_experiments"
                / f"b5_v2_{method}_frozen_seed{seed}.formal.json"
            ).read_text(encoding="utf-8")
        )
        for method in ("gno", "gino")
        for seed in (42, 43, 44)
    }
    within_method_locked = all(
        _without_run_metadata(configs[(method, 42)])
        == _without_run_metadata(configs[(method, seed)])
        for method in ("gno", "gino")
        for seed in (43, 44)
    )
    shared_sections_locked = all(
        configs[("gno", seed)][section] == configs[("gino", seed)][section]
        for seed in (42, 43, 44)
        for section in ("data", "observation", "training", "evaluation", "runtime")
    )
    shared_rfno_locked = all(
        {
            key: value
            for key, value in configs[("gno", seed)]["pipeline"].items()
            if key != "reconstructor"
        }
        == {
            key: value
            for key, value in configs[("gino", seed)]["pipeline"].items()
            if key != "reconstructor"
        }
        for seed in (42, 43, 44)
    )
    if not within_method_locked or not shared_sections_locked or not shared_rfno_locked:
        raise RuntimeError("Locked six-config protocol comparison failed")

    per_seed: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    source_values: dict[tuple[str, str, str], list[float]] = {}
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    restore_by_key = {
        (item["method"], int(item["seed"])): item
        for item in restore_audit["runs"]
    }

    for item in status:
        method = str(item["method"])
        seed = int(item["seed"])
        metrics_path = Path(item["metrics"])
        training_path = Path(item["training_summary"])
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        training = json.loads(training_path.read_text(encoding="utf-8"))
        loaded[(method, seed)] = metrics_payload
        if (
            metrics_payload["metrics"]["evaluated_samples"] != 777
            or metrics_payload["metrics"]["rollout_steps"] != 300
            or not metrics_payload["freeze_checks"]["rfno_requires_grad_false"]
            or not metrics_payload["freeze_checks"]["rfno_gradients_none"]
        ):
            raise RuntimeError(f"Formal validation contract failed for {method} seed{seed}")

        for horizon in HORIZONS:
            block = _metric_block(metrics_payload, horizon)
            per_seed.append(
                {
                    "method": method,
                    "seed": seed,
                    "horizon": "history_missing" if horizon == "history" else f"F{horizon}",
                    **block,
                }
            )

        per_source = metrics_payload["metrics"]["per_source"]
        entries = per_source.values() if isinstance(per_source, dict) else per_source
        for entry in entries:
            name = str(entry["source"]["name"])
            source_values.setdefault((method, "history_missing_nrmse", name), []).append(
                float(entry["history_reconstruction"]["missing_region"]["nrmse"])
            )
            source_values.setdefault((method, "forecast_300_nrmse", name), []).append(
                float(entry["forecast_300"]["nrmse"])
            )

        best_audit = restore_by_key[(method, seed)]["restored"]["best"]
        evaluated = metrics_payload["evaluated_checkpoint"]
        if evaluated["sha256"] != best_audit["sha256"]:
            raise RuntimeError(f"Evaluation/best hash mismatch for {method} seed{seed}")
        checkpoints.append(
            {
                "method": method,
                "seed": seed,
                "best_epoch": int(metrics_payload["checkpoint_restore"]["epoch"]),
                "selection_metric": float(
                    metrics_payload["checkpoint_restore"]["metrics"][
                        "val.reconstruction_missing_nrmse"
                    ]
                ),
                "path": str(Path(evaluated["path"])),
                "sha256": str(evaluated["sha256"]),
            }
        )
        resources.append(
            {
                "method": method,
                "seed": seed,
                "actual_epochs": int(training["last_epoch"]),
                "training_seconds": float(training["elapsed_seconds"]),
                "peak_memory_mb": float(training["peak_memory_allocated_mb"]),
                "total_parameters": int(metrics_payload["parameters"]["total"]),
                "trainable_parameters": int(metrics_payload["parameters"]["trainable"]),
                "checkpoint_size_mb": float(best_audit["size_bytes"]) / (1024.0**2),
                "val300_seconds": float(metrics_payload["metrics"]["elapsed_seconds"]),
                "seconds_per_sample": float(
                    metrics_payload["metrics"]["seconds_per_sample"]
                ),
            }
        )

    aggregate: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        label = "history_missing" if horizon == "history" else f"F{horizon}"
        for metric in METRICS:
            for method in ("gno", "gino"):
                values = [
                    _metric_block(loaded[(method, seed)], horizon)[metric]
                    for seed in (42, 43, 44)
                ]
                mean, sd = _mean_sd(values)
                aggregate.append(
                    {
                        "method": method,
                        "horizon": label,
                        "metric": metric,
                        "mean": mean,
                        "sample_sd": sd,
                    }
                )
            differences = [
                _metric_block(loaded[("gino", seed)], horizon)[metric]
                - _metric_block(loaded[("gno", seed)], horizon)[metric]
                for seed in (42, 43, 44)
            ]
            mean, sd = _mean_sd(differences)
            half_width = T_CRITICAL_DF2_95 * sd / math.sqrt(3.0)
            paired.append(
                {
                    "contrast": "GINO-GNO",
                    "horizon": label,
                    "metric": metric,
                    "mean_difference": mean,
                    "sample_sd": sd,
                    "ci95_low": mean - half_width,
                    "ci95_high": mean + half_width,
                    "n": 3,
                }
            )

    worst: list[dict[str, Any]] = []
    for method in ("gno", "gino"):
        for metric in ("history_missing_nrmse", "forecast_300_nrmse"):
            candidates = []
            for (candidate_method, candidate_metric, name), values in source_values.items():
                if candidate_method == method and candidate_metric == metric:
                    mean, sd = _mean_sd(values)
                    candidates.append((mean, name, sd))
            mean, name, sd = max(candidates)
            worst.append(
                {
                    "method": method,
                    "metric": metric,
                    "mat": name,
                    "mean": mean,
                    "sample_sd": sd,
                }
            )

    resource_summary: list[dict[str, Any]] = []
    for method in ("gno", "gino"):
        rows = [row for row in resources if row["method"] == method]
        summary: dict[str, Any] = {
            "method": method,
            "total_parameters": rows[0]["total_parameters"],
            "trainable_parameters": rows[0]["trainable_parameters"],
        }
        for key in (
            "actual_epochs",
            "training_seconds",
            "peak_memory_mb",
            "checkpoint_size_mb",
            "val300_seconds",
            "seconds_per_sample",
        ):
            mean, sd = _mean_sd([float(row[key]) for row in rows])
            summary[f"{key}_mean"] = mean
            summary[f"{key}_sample_sd"] = sd
        resource_summary.append(summary)

    payload = {
        "experiment_id": "MIN-B5-GNO-GINO-v2-three-seed-rerun",
        "legacy_scalar_seed42_excluded": True,
        "test_accessed": False,
        "checkpoint_restore_audit_passed": True,
        "protocol_audit": {
            "within_method_only_seed_and_metadata_differ": within_method_locked,
            "cross_method_shared_sections_identical": shared_sections_locked,
            "cross_method_rfno_pipeline_identical": shared_rfno_locked,
            "observation_seed_values": sorted(
                {
                    int(config["observation"]["seed"])
                    for config in configs.values()
                }
            ),
            "b0_checkpoint_sha256": _sha256(
                root
                / "checkpoints"
                / "hparam_sensitivity"
                / "mode_m28x28_h32_lp64.pt"
            ),
            "coordinate_operator_sha256": _sha256(
                root
                / "neuralop"
                / "models"
                / "reconstructors"
                / "coordinate_operator.py"
            ),
            "sparse_runner_sha256": _sha256(
                root / "neuralop" / "training" / "sparse_experiment_runner.py"
            ),
        },
        "aggregate": aggregate,
        "paired_effects": paired,
        "worst_per_mat": worst,
        "checkpoints": checkpoints,
        "resource_summary": resource_summary,
        "failures": [],
    }
    (out / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(out / "per_seed_metrics.csv", per_seed)
    _write_csv(out / "aggregate_metrics.csv", aggregate)
    _write_csv(out / "paired_effects.csv", paired)
    _write_csv(out / "worst_per_mat.csv", worst)
    _write_csv(out / "checkpoints.csv", checkpoints)
    _write_csv(out / "resources.csv", resources)
    _write_csv(out / "resource_summary.csv", resource_summary)
    print(
        json.dumps(
            {
                "runs": len(status),
                "checkpoint_restore_audit_passed": True,
                "protocol_audit": payload["protocol_audit"],
                "failures": payload["failures"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
