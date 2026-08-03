"""Summarize the locked random-init 100-epoch test matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs/sparse"
OUT = ROOT / "results/no_pretrain_100ep_test_matrix_20260731_v1"
HORIZONS = (30, 60, 120, 180, 240, 300)
METHODS = {
    "mask_unet_random": "no_pretrain100_mask_unet_random_fixed_r05_mask_00006_seed{seed}_20260731_v1",
    "st_d0_random": "no_pretrain100_st_d0_random_fixed_r05_mask_00006_seed{seed}_20260731_v1",
    "partialconv_light_random": "no_pretrain100_partialconv_light_random_fixed_r05_mask_00006_seed{seed}_20260731_v1",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_summary(run_dir: Path) -> dict:
    candidates = sorted(run_dir.glob("training_summary_epoch*.json"))
    if not candidates:
        raise FileNotFoundError(f"No training summary: {run_dir}")
    return read_json(candidates[-1])


def learned_rows() -> list[dict]:
    rows: list[dict] = []
    for method, template in METHODS.items():
        for seed in (42, 43, 44):
            name = template.format(seed=seed)
            train_dir = RUN_ROOT / name
            test_dir = RUN_ROOT / f"{name}_locked_test300"
            test = read_json(test_dir / "evaluation/test_metrics.json")
            metrics = test["metrics"]
            checkpoint = train_dir / "checkpoints/best.pt"
            summary = training_summary(train_dir)
            row = {
                "method": method,
                "seed": seed,
                "reconstruction_missing_nrmse": metrics["history_reconstruction"]["missing_region"]["nrmse"],
                "reconstruction_correlation": metrics["history_reconstruction"]["missing_region"]["correlation"],
                "reconstruction_ssp": metrics["history_reconstruction"]["ssp"],
                "reconstruction_gradient_nrmse": metrics["history_reconstruction"]["gradient_nrmse"],
                "completed_epochs": summary["completed_epochs"],
                "best_epoch": summary["best_epoch"],
                "stopped_early": summary["stopped_early"],
                "training_seconds": summary["elapsed_seconds"],
                "peak_memory_allocated_mb": summary["peak_memory_allocated_mb"],
                "test_seconds": metrics["elapsed_seconds"],
                "evaluated_samples": metrics["evaluated_samples"],
                "best_checkpoint": str(checkpoint.resolve()),
                "best_checkpoint_sha256": sha256(checkpoint),
            }
            for horizon in HORIZONS:
                block = metrics[f"forecast_{horizon}"]
                row[f"forecast_{horizon}_nrmse"] = block["nrmse"]
                row[f"forecast_{horizon}_correlation"] = block["correlation"]
                row[f"forecast_{horizon}_ssp"] = block["ssp"]
                row[f"forecast_{horizon}_gradient_nrmse"] = block["gradient_nrmse"]
            if int(row["evaluated_samples"]) != 666:
                raise ValueError(f"Incomplete test evaluation: {test_dir}")
            rows.append(row)
    return rows


def bilinear_row() -> dict:
    path = RUN_ROOT / "no_pretrain100_bilinear_frozen_fixed_r05_mask_00006_test300_20260731_v1/evaluation/test_metrics.json"
    test = read_json(path)
    metrics = test["metrics"]
    row = {
        "method": "bilinear",
        "seed": "deterministic",
        "reconstruction_missing_nrmse": metrics["history_reconstruction"]["missing_region"]["nrmse"],
        "reconstruction_correlation": metrics["history_reconstruction"]["missing_region"]["correlation"],
        "reconstruction_ssp": metrics["history_reconstruction"]["ssp"],
        "reconstruction_gradient_nrmse": metrics["history_reconstruction"]["gradient_nrmse"],
        "completed_epochs": 0,
        "best_epoch": "",
        "stopped_early": "",
        "training_seconds": 0.0,
        "peak_memory_allocated_mb": "",
        "test_seconds": metrics["elapsed_seconds"],
        "evaluated_samples": metrics["evaluated_samples"],
        "best_checkpoint": "",
        "best_checkpoint_sha256": "",
    }
    for horizon in HORIZONS:
        block = metrics[f"forecast_{horizon}"]
        row[f"forecast_{horizon}_nrmse"] = block["nrmse"]
        row[f"forecast_{horizon}_correlation"] = block["correlation"]
        row[f"forecast_{horizon}_ssp"] = block["ssp"]
        row[f"forecast_{horizon}_gradient_nrmse"] = block["gradient_nrmse"]
    if int(row["evaluated_samples"]) != 666:
        raise ValueError("Incomplete bilinear test evaluation")
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    keys = ["reconstruction_missing_nrmse"] + [f"forecast_{h}_nrmse" for h in HORIZONS]
    output: list[dict] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        record = {"method": method, "n_seeds": len(selected)}
        for key in keys:
            values = [float(row[key]) for row in selected]
            record[f"{key}_mean"] = statistics.mean(values)
            record[f"{key}_sample_sd"] = statistics.stdev(values)
        record["training_seconds_mean"] = statistics.mean(float(row["training_seconds"]) for row in selected)
        record["peak_memory_allocated_mb_mean"] = statistics.mean(float(row["peak_memory_allocated_mb"]) for row in selected)
        output.append(record)
    bilinear = next(row for row in rows if row["method"] == "bilinear")
    record = {"method": "bilinear", "n_seeds": 1}
    for key in keys:
        record[f"{key}_mean"] = bilinear[key]
        record[f"{key}_sample_sd"] = ""
    record["training_seconds_mean"] = 0.0
    record["peak_memory_allocated_mb_mean"] = ""
    output.append(record)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = learned_rows()
    rows.append(bilinear_row())
    write_csv(OUT / "test_scorecard_per_seed.csv", rows)
    summary = aggregate(rows)
    write_csv(OUT / "test_scorecard_summary.csv", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
