#!/usr/bin/env python3
"""Aggregate finalist validation runs without touching the test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


HORIZONS = (30, 60, 120, 180, 240, 300)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_training_summary(run_dir: Path) -> Mapping[str, Any]:
    paths = sorted(run_dir.glob("training_summary_epoch*.json"))
    if not paths:
        raise FileNotFoundError(f"No formal training summary in {run_dir}")
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def _metric_fields(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    history = metrics["history_reconstruction"]
    missing = history["missing_region"]
    observed = history["observed_region"]
    result: Dict[str, Any] = {
        "recon_full_rmse": history["rmse"],
        "recon_full_nrmse": history["nrmse"],
        "recon_full_corr": history["correlation"],
        "recon_full_ssp": history["ssp"],
        "recon_full_gradient_nrmse": history["gradient_nrmse"],
        "recon_missing_rmse": missing["rmse"],
        "recon_missing_nrmse": missing["nrmse"],
        "recon_missing_corr": missing["correlation"],
        "recon_observed_rmse": observed["rmse"],
        "recon_observed_nrmse": observed["nrmse"],
        "recon_observed_corr": observed["correlation"],
    }
    for horizon in HORIZONS:
        forecast = metrics[f"forecast_{horizon}"]
        prefix = f"f{horizon}"
        result.update(
            {
                f"{prefix}_rmse": forecast["rmse"],
                f"{prefix}_nrmse": forecast["nrmse"],
                f"{prefix}_corr": forecast["correlation"],
                f"{prefix}_ssp": forecast["ssp"],
                f"{prefix}_gradient_rmse": forecast["gradient_rmse"],
                f"{prefix}_gradient_nrmse": forecast["gradient_nrmse"],
            }
        )
    curve = metrics["error_growth_curve"]
    result["error_growth_mean_nrmse"] = sum(
        float(point["nrmse"]) for point in curve
    ) / len(curve)
    result["error_growth_final_nrmse"] = curve[-1]["nrmse"]
    result["error_growth_final_corr"] = curve[-1]["correlation"]
    return result


def _identity(entry: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: entry[key]
        for key in (
            "candidate",
            "method",
            "layers",
            "distribution",
            "mask_type",
            "observation_rate",
            "mask_id",
            "mask_replicate",
            "train_seed",
            "eval_seed",
        )
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    matrix_path: Path,
    output_dir: Path,
    *,
    project_root: Path,
) -> Dict[str, Any]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if "test" in str(matrix.get("split_policy", "")).lower() and "forbidden" not in str(
        matrix.get("split_policy", "")
    ).lower():
        raise PermissionError("Finalist robustness aggregation must be validation-only")

    summary_rows = []
    per_mat_rows = []
    curve_rows = []
    for entry in matrix["entries"]:
        metrics_path = (project_root / entry["metrics_path"]).resolve()
        training_run_dir = (project_root / entry["training_run_dir"]).resolve()
        checkpoint_path = (project_root / entry["checkpoint_path"]).resolve()
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("split") != "val":
            raise PermissionError(f"Non-validation metrics rejected: {metrics_path}")
        metrics = payload["metrics"]
        if int(metrics["rollout_steps"]) != 300:
            raise ValueError(f"Expected full 300-frame rollout: {metrics_path}")
        if int(metrics["evaluated_samples"]) <= 0:
            raise ValueError(f"No validation samples: {metrics_path}")

        training = _latest_training_summary(training_run_dir)
        identity = _identity(entry)
        row = {
            **identity,
            **_metric_fields(metrics),
            "evaluated_samples": metrics["evaluated_samples"],
            "evaluated_batches": metrics["evaluated_batches"],
            "source_count": metrics["source_count"],
            "parameters_total": payload["parameters"]["total"],
            "parameters_trainable": payload["parameters"]["trainable"],
            "training_time_seconds": training["elapsed_seconds"],
            "best_epoch": training["best_epoch"],
            "last_epoch": training["last_epoch"],
            "stopped_early": training["stopped_early"],
            "safety_stop_reason": training.get("safety_stop_reason"),
            "peak_memory_allocated_mb": training["peak_memory_allocated_mb"],
            "inference_time_seconds": metrics["elapsed_seconds"],
            "seconds_per_sample": metrics["seconds_per_sample"],
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "split_sha256": _sha256(training_run_dir / "data_split.json"),
            "evaluation_config_sha256": payload["scientific_config_sha256"],
            "metrics_path": str(metrics_path),
        }
        numeric_values = [
            float(value)
            for value in row.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if not all(math.isfinite(value) for value in numeric_values):
            raise FloatingPointError(f"Non-finite summary value: {metrics_path}")
        summary_rows.append(row)

        for source_id, source_metrics in sorted(
            metrics["per_source"].items(), key=lambda item: int(item[0])
        ):
            source = source_metrics["source"]
            per_mat_rows.append(
                {
                    **identity,
                    "source_id": source_id,
                    "source_name": source["name"],
                    "source_relative_path": source["relative_path"],
                    **_metric_fields(source_metrics),
                }
            )
        for point in metrics["error_growth_curve"]:
            curve_rows.append(
                {
                    **identity,
                    "frame": point["frame"],
                    "time_seconds": point["time_seconds"],
                    "rmse": point["rmse"],
                    "nrmse": point["nrmse"],
                    "correlation": point["correlation"],
                }
            )

    _write_csv(output_dir / "candidate_val_summary.csv", summary_rows)
    _write_csv(output_dir / "candidate_val_per_mat.csv", per_mat_rows)
    _write_csv(output_dir / "candidate_val_error_growth.csv", curve_rows)
    return {
        "runs": len(summary_rows),
        "per_mat_rows": len(per_mat_rows),
        "error_growth_rows": len(curve_rows),
        "output_dir": str(output_dir.resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = summarize(
        args.matrix.resolve(),
        args.output_dir.resolve(),
        project_root=args.project_root.resolve(),
    )
    print(
        "aggregated "
        f"{result['runs']} val runs, {result['per_mat_rows']} per-MAT rows, "
        f"{result['error_growth_rows']} error-growth rows -> {result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
