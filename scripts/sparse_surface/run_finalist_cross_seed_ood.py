#!/usr/bin/env python3
"""Run the locked 3-seed x 3-mask finalist OOD validation matrix.

Only the validation split is evaluated. Existing seed-42 artifacts are reused;
missing seed-43/44 cells are evaluated from their locked best checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


HORIZONS = (30, 60, 120, 180, 240, 300)
SEEDS = (42, 43, 44)
MASKS = ((0, "mask_00021"), (1, "mask_00022"), (2, "mask_00023"))
METRICS = (
    "recon_missing_nrmse",
    *(f"f{h}_{field}" for h in HORIZONS for field in ("nrmse", "corr", "ssp", "gradient_nrmse")),
    "error_growth_mean_nrmse",
    "error_growth_final_nrmse",
)
LOWER_IS_BETTER = {metric for metric in METRICS if not metric.endswith("_corr")}


METHODS = {
    "b4": {
        "experiment_id": "B4",
        "method": "Mask U-Net + frozen RFNO",
        "base_configs": {
            replicate: f"config/sparse_experiments/finalist_robustness/b4_ood_random_r05_rep{replicate}.json"
            for replicate, _ in MASKS
        },
        "training_runs": {
            42: "runs/sparse/b4_mask_unet_frozen_fixed_points_r05_mask_00006_seed42_formal_20260721_v2",
            43: "runs/sparse/finalist_b4_fixed_r05_seed43_mask_00006_formal_20260722_v1",
            44: "runs/sparse/finalist_b4_fixed_r05_seed44_mask_00006_formal_20260722_v1",
        },
        "existing_ood_runs": {
            (42, replicate): f"runs/sparse/finalist_ood_b4_random_r05_rep{replicate}_{mask_id}_from_fixed_seed42_val_20260722_v1"
            for replicate, mask_id in MASKS
        },
    },
    "p1": {
        "experiment_id": "P1",
        "method": "PartialConv-MAE no-observation-consistency + frozen RFNO",
        "base_configs": {
            replicate: f"config/sparse_experiments/finalist_robustness/p1_ood_random_r05_rep{replicate}.json"
            for replicate, _ in MASKS
        },
        "training_runs": {
            42: "runs/sparse/p1_partialconv_mae_frozen_no_obs_consistency_fixed_points_r05_mask_00006_seed42_formal_20260721_v1",
            43: "runs/sparse/finalist_p1_fixed_r05_seed43_mask_00006_formal_20260722_v1",
            44: "runs/sparse/finalist_p1_fixed_r05_seed44_mask_00006_formal_20260722_v1",
        },
        "existing_ood_runs": {
            (42, replicate): f"runs/sparse/finalist_ood_p1_random_r05_rep{replicate}_{mask_id}_from_fixed_seed42_val_20260722_v1"
            for replicate, mask_id in MASKS
        },
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/finalist_cross_seed_ood_20260722"),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--preflight", action="store_true")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_fields(metrics: Mapping[str, Any]) -> dict[str, float]:
    result = {
        "recon_missing_nrmse": float(metrics["history_reconstruction"]["missing_region"]["nrmse"]),
    }
    for horizon in HORIZONS:
        forecast = metrics[f"forecast_{horizon}"]
        result.update(
            {
                f"f{horizon}_nrmse": float(forecast["nrmse"]),
                f"f{horizon}_corr": float(forecast["correlation"]),
                f"f{horizon}_ssp": float(forecast["ssp"]),
                f"f{horizon}_gradient_nrmse": float(forecast["gradient_nrmse"]),
            }
        )
    curve = metrics["error_growth_curve"]
    result["error_growth_mean_nrmse"] = statistics.fmean(float(point["nrmse"]) for point in curve)
    result["error_growth_final_nrmse"] = float(curve[-1]["nrmse"])
    return result


def _validate_metrics(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("split") != "val":
        raise PermissionError(f"non-validation artifact rejected: {path}")
    if payload.get("evaluation_kind") != "formal":
        raise ValueError(f"partial evaluation rejected: {path}")
    metrics = payload["metrics"]
    if int(metrics["rollout_steps"]) != 300 or int(metrics["evaluated_samples"]) <= 0:
        raise ValueError(f"incomplete 300-frame validation artifact: {path}")
    if not payload["freeze_checks"]["rfno_requires_grad_false"]:
        raise ValueError(f"RFNO is not frozen: {path}")
    return payload


def _new_run_name(candidate: str, seed: int, replicate: int, mask_id: str) -> str:
    return (
        f"finalist_cross_ood_{candidate}_random_r05_trainseed{seed}_"
        f"rep{replicate}_{mask_id}_val_20260722_v1"
    )


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _preflight(project_root: Path) -> list[dict[str, Any]]:
    plan = []
    split_hashes = set()
    for candidate, spec in METHODS.items():
        for seed in SEEDS:
            training_run = project_root / spec["training_runs"][seed]
            checkpoint = training_run / "checkpoints" / "best.pt"
            fixed_metrics = training_run / "evaluation" / "val_metrics.json"
            split_manifest = training_run / "data_split.json"
            for required in (checkpoint, fixed_metrics, split_manifest):
                if not required.is_file():
                    raise FileNotFoundError(required)
            _validate_metrics(fixed_metrics)
            split_hashes.add(_sha256(split_manifest))
            for replicate, mask_id in MASKS:
                base_config = project_root / spec["base_configs"][replicate]
                if not base_config.is_file():
                    raise FileNotFoundError(base_config)
                existing = spec["existing_ood_runs"].get((seed, replicate))
                run_name = Path(existing).name if existing else _new_run_name(candidate, seed, replicate, mask_id)
                run_dir = project_root / "runs" / "sparse" / run_name
                metrics_path = run_dir / "evaluation" / "val_metrics.json"
                if metrics_path.is_file():
                    _validate_metrics(metrics_path)
                elif run_dir.exists():
                    raise FileExistsError(f"incomplete existing run directory: {run_dir}")
                plan.append(
                    {
                        "candidate": candidate,
                        "method": spec["method"],
                        "experiment_id": spec["experiment_id"],
                        "train_seed": seed,
                        "eval_seed": seed,
                        "mask_replicate": replicate,
                        "mask_id": mask_id,
                        "base_config": base_config,
                        "training_run": training_run,
                        "checkpoint": checkpoint,
                        "fixed_metrics": fixed_metrics,
                        "split_manifest": split_manifest,
                        "run_name": run_name,
                        "run_dir": run_dir,
                        "metrics_path": metrics_path,
                        "needs_evaluation": not metrics_path.is_file(),
                    }
                )
    if len(split_hashes) != 1:
        raise ValueError(f"training seeds do not share one frozen split: {sorted(split_hashes)}")
    if len(plan) != 18:
        raise AssertionError(f"expected 18 cross cells, got {len(plan)}")
    return plan


def _evaluate_missing(
    plan: list[dict[str, Any]],
    *,
    project_root: Path,
    output_dir: Path,
    python: Path,
) -> None:
    config_dir = output_dir / "configs"
    log_dir = output_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)
    for cell in plan:
        if not cell["needs_evaluation"]:
            continue
        config = _load_json(cell["base_config"])
        config["experiment_name"] = cell["run_name"]
        config["runtime"]["seed"] = cell["eval_seed"]
        # The manifest was generated once with seed 42; mask_id selects the
        # realized mask. Keep that manifest seed fixed while varying the
        # checkpoint/training seed through runtime.seed.
        if config["observation"]["mask_id"] != cell["mask_id"]:
            raise ValueError(f"base config mask mismatch: {cell['base_config']}")
        config_path = config_dir / f"{cell['run_name']}.json"
        _write_exclusive_json(config_path, config)
        command = [
            str(python),
            str(project_root / "scripts" / "sparse_surface" / "run_sparse_experiment.py"),
            str(config_path),
            "--project-root",
            str(project_root),
            "--run-dir",
            str(cell["run_dir"]),
            "--split-manifest",
            str(cell["split_manifest"]),
            "--evaluation-checkpoint",
            str(cell["checkpoint"]),
            "--evaluate-split",
            "val",
            "--allow-validation-protocol-mismatch",
        ]
        log_path = log_dir / f"{cell['run_name']}.log"
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"evaluation failed ({completed.returncode}); inspect {log_path}")
        _validate_metrics(cell["metrics_path"])


def _summarize(plan: list[dict[str, Any]], output_dir: Path) -> None:
    run_rows: list[dict[str, Any]] = []
    per_mat_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    matrix_entries = []
    for cell in plan:
        payload = _validate_metrics(cell["metrics_path"])
        fixed_payload = _validate_metrics(cell["fixed_metrics"])
        values = _metric_fields(payload["metrics"])
        fixed_values = _metric_fields(fixed_payload["metrics"])
        identity = {
            "candidate": cell["candidate"],
            "method": cell["method"],
            "train_seed": cell["train_seed"],
            "eval_seed": cell["eval_seed"],
            "mask_replicate": cell["mask_replicate"],
            "mask_id": cell["mask_id"],
        }
        row = {
            **identity,
            **values,
            **{f"fixed_{metric}": fixed_values[metric] for metric in METRICS},
            **{
                f"random_over_fixed_{metric}_ratio": values[metric] / fixed_values[metric]
                for metric in METRICS
            },
            "evaluated_samples": int(payload["metrics"]["evaluated_samples"]),
            "source_count": int(payload["metrics"]["source_count"]),
            "inference_time_seconds": float(payload["metrics"]["elapsed_seconds"]),
            "checkpoint_sha256": payload["evaluated_checkpoint"]["sha256"],
            "metrics_path": str(cell["metrics_path"].resolve()),
        }
        if not all(
            math.isfinite(float(value))
            for value in row.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ):
            raise FloatingPointError(f"non-finite run summary: {cell['metrics_path']}")
        run_rows.append(row)

        for source_metrics in payload["metrics"]["per_source"].values():
            source = source_metrics["source"]
            per_mat_rows.append(
                {
                    **identity,
                    "source_name": source["name"],
                    "source_relative_path": source["relative_path"],
                    **_metric_fields(source_metrics),
                }
            )
        for point in payload["metrics"]["error_growth_curve"]:
            curve_rows.append(
                {
                    **identity,
                    "frame": int(point["frame"]),
                    "time_seconds": float(point["time_seconds"]),
                    "rmse": float(point["rmse"]),
                    "nrmse": float(point["nrmse"]),
                    "correlation": float(point["correlation"]),
                }
            )
        matrix_entries.append(
            {
                **identity,
                "checkpoint_path": str(cell["checkpoint"].resolve()),
                "checkpoint_sha256": payload["evaluated_checkpoint"]["sha256"],
                "split_manifest_path": str(cell["split_manifest"].resolve()),
                "split_sha256": _sha256(cell["split_manifest"]),
                "metrics_path": str(cell["metrics_path"].resolve()),
            }
        )

    summary_rows = []
    for candidate, spec in METHODS.items():
        method_runs = [row for row in run_rows if row["candidate"] == candidate]
        method_sources = [row for row in per_mat_rows if row["candidate"] == candidate]
        summary: dict[str, Any] = {
            "candidate": candidate,
            "method": spec["method"],
            "n_cross_evaluations": len(method_runs),
            "n_training_seeds": len({row["train_seed"] for row in method_runs}),
            "n_mask_replicates": len({row["mask_replicate"] for row in method_runs}),
            "n_val_mat": len({row["source_name"] for row in method_sources}),
            "n_per_mat_rows": len(method_sources),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in method_runs]
            ratios = [float(row[f"random_over_fixed_{metric}_ratio"]) for row in method_runs]
            worst = (
                max(method_sources, key=lambda row: float(row[metric]))
                if metric in LOWER_IS_BETTER
                else min(method_sources, key=lambda row: float(row[metric]))
            )
            summary.update(
                {
                    f"{metric}_mean": statistics.fmean(values),
                    f"{metric}_sample_std": statistics.stdev(values),
                    f"{metric}_worst_per_mat": float(worst[metric]),
                    f"{metric}_worst_train_seed": worst["train_seed"],
                    f"{metric}_worst_mask_id": worst["mask_id"],
                    f"{metric}_worst_source_name": worst["source_name"],
                    f"{metric}_random_over_fixed_ratio_mean": statistics.fmean(ratios),
                    f"{metric}_random_over_fixed_ratio_sample_std": statistics.stdev(ratios),
                    f"{metric}_random_over_fixed_percent_change_mean": 100.0
                    * (statistics.fmean(ratios) - 1.0),
                }
            )
        summary["inference_time_seconds_mean"] = statistics.fmean(
            float(row["inference_time_seconds"]) for row in method_runs
        )
        summary["inference_time_seconds_sample_std"] = statistics.stdev(
            float(row["inference_time_seconds"]) for row in method_runs
        )
        summary_rows.append(summary)

    _write_csv(output_dir / "candidate_cross_seed_ood_summary.csv", summary_rows)
    _write_csv(output_dir / "candidate_cross_seed_ood_runs.csv", run_rows)
    _write_csv(output_dir / "candidate_cross_seed_ood_per_mat.csv", per_mat_rows)
    _write_csv(output_dir / "candidate_cross_seed_ood_error_growth.csv", curve_rows)
    _write_exclusive_json(
        output_dir / "cross_seed_ood_matrix.json",
        {
            "schema_version": 1,
            "experiment_id": "finalist-cross-seed-ood",
            "split_policy": "validation only; model weights locked",
            "entries": matrix_entries,
        },
    )


def main() -> int:
    args = _parser().parse_args()
    project_root = args.project_root.resolve()
    plan = _preflight(project_root)
    missing = sum(bool(cell["needs_evaluation"]) for cell in plan)
    if args.preflight:
        print(json.dumps({"cells": len(plan), "reused": len(plan) - missing, "to_evaluate": missing}))
        return 0
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    _evaluate_missing(
        plan,
        project_root=project_root,
        output_dir=output_dir,
        python=args.python.resolve(),
    )
    _summarize(plan, output_dir)
    print(f"completed 18 validation cells ({18 - missing} reused, {missing} evaluated) -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
