#!/usr/bin/env python3
"""Export the single locked test execution without running model inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping


HORIZONS = (30, 60, 120, 180, 240, 300)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _flatten(metrics: Mapping[str, Any]) -> dict[str, float]:
    history = metrics["history_reconstruction"]
    result = {
        "reconstruction_full_nrmse": float(history["nrmse"]),
        "reconstruction_full_corr": float(history["correlation"]),
        "reconstruction_full_ssp": float(history["ssp"]),
        "reconstruction_full_gradient_nrmse": float(history["gradient_nrmse"]),
        "reconstruction_missing_nrmse": float(history["missing_region"]["nrmse"]),
        "observation_consistency_nrmse": float(
            history["observed_region"]["nrmse"]
        ),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("config/sparse_experiments/final_test_locked_20260723_v2.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final_test_locked_20260723_v2"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")

    manifest = _read_json(args.manifest)
    project_root = args.manifest.resolve().parents[2]
    comparison_rows: list[dict[str, Any]] = []
    combined_per_mat: list[dict[str, Any]] = []
    combined_growth: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []

    args.output_dir.mkdir(parents=True)
    for method in manifest["methods"]:
        method_id = str(method["method_id"])
        label = "p1_pretrained" if method_id == "P1-pretrained" else method_id.lower()
        run_dir = project_root / method["test_run_dir"]
        source_path = run_dir / "evaluation" / "test_metrics.json"
        payload = _read_json(source_path)
        metrics = payload["metrics"]
        if (
            payload["evaluation_kind"] != "formal"
            or int(metrics["rollout_steps"]) != 300
            or int(metrics["source_count"]) != 6
            or not payload["freeze_checks"]["rfno_requires_grad_false"]
            or not payload["freeze_checks"]["rfno_gradients_none"]
        ):
            raise ValueError(f"Incomplete or invalid locked test artifact: {source_path}")
        if payload["evaluated_checkpoint"]["sha256"] != method["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint mismatch for {method_id}")

        method_dir = args.output_dir / label
        method_dir.mkdir()
        shutil.copy2(source_path, method_dir / "test_metrics.json")

        comparison = {
            "method_id": method_id,
            "method": method["method"],
            "locked_role": method["role"],
            "evaluated_samples": int(metrics["evaluated_samples"]),
            "evaluated_batches": int(metrics["evaluated_batches"]),
            "source_count": int(metrics["source_count"]),
            "inference_seconds": float(metrics["elapsed_seconds"]),
            "seconds_per_sample": float(metrics["seconds_per_sample"]),
            "parameters_total": int(payload["parameters"]["total"]),
            "parameters_trainable": int(payload["parameters"]["trainable"]),
            "checkpoint_sha256": method["checkpoint_sha256"],
        }
        comparison.update(_flatten(metrics))
        comparison_rows.append(comparison)

        method_per_mat: list[dict[str, Any]] = []
        method_growth: list[dict[str, Any]] = []
        for source_id, source_metrics in sorted(
            metrics["per_source"].items(), key=lambda item: int(item[0])
        ):
            source = source_metrics["source"]
            row = {
                "method_id": method_id,
                "source_id": int(source_id),
                "mat_name": source["name"],
                "relative_path": source["relative_path"],
                **_flatten(source_metrics),
            }
            method_per_mat.append(row)
            combined_per_mat.append(row)
            for point in source_metrics["error_growth_curve"]:
                growth = {
                    "method_id": method_id,
                    "scope": "per_mat",
                    "source_id": int(source_id),
                    "mat_name": source["name"],
                    **point,
                }
                method_growth.append(growth)
                combined_growth.append(growth)
        for point in metrics["error_growth_curve"]:
            growth = {
                "method_id": method_id,
                "scope": "aggregate",
                "source_id": "",
                "mat_name": "",
                **point,
            }
            method_growth.append(growth)
            combined_growth.append(growth)

        _write_csv(method_dir / "per_mat_metrics.csv", method_per_mat)
        _write_csv(method_dir / "error_growth_curve.csv", method_growth)
        execution_rows.append(
            {
                "method_id": method_id,
                "source_test_metrics": str(source_path.resolve()),
                "source_test_metrics_sha256": _sha256(source_path),
                "exported_test_metrics": str(
                    (method_dir / "test_metrics.json").resolve()
                ),
                "checkpoint_sha256": method["checkpoint_sha256"],
                "completed": True,
            }
        )

    _write_csv(args.output_dir / "final_comparison.csv", comparison_rows)
    _write_csv(args.output_dir / "per_mat_metrics.csv", combined_per_mat)
    _write_csv(args.output_dir / "error_growth_curves.csv", combined_growth)
    _write_csv(args.output_dir / "execution_provenance.csv", execution_rows)
    summary = {
        "experiment_id": "final-locked-test",
        "completed": True,
        "methods": [row["method_id"] for row in comparison_rows],
        "evaluated_samples_per_method": {
            row["method_id"]: row["evaluated_samples"] for row in comparison_rows
        },
        "test_runs_executed": len(comparison_rows),
        "reruns": 0,
        "training_executed": False,
        "checkpoint_reselection_performed": False,
        "manifest_sha256": _sha256(args.manifest),
        "failures": [],
    }
    (args.output_dir / "final_test_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"exported locked test artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
