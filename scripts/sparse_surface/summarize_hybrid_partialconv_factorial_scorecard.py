#!/usr/bin/env python3
"""Build the validation-only hybrid PartialConv factorial scorecard."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

from scipy.stats import t as student_t


SEEDS = (42, 43, 44)
HORIZONS = (30, 60, 120, 180, 240, 300)
T_CRITICAL_DF2_95 = 4.302652729911275

METHOD_LABELS = {
    "B4": "Mask U-Net + frozen RFNO",
    "H1": "1-level PartialConv U-Net + skip",
    "H1-A1": "3-level PartialConv U-Net + skip, reconstruction pretraining only",
    "H1-A2": "1-level PartialConv U-Net, no skip",
    "H1-A3-pretrained": "1-level PartialConv U-Net + skip, pretrained downstream",
    "H1-A3-random-init": "1-level PartialConv U-Net + skip, random-init downstream",
    "H1-A4": "3-level PartialConv U-Net + skip, pretrained downstream",
    "H1-A5": "3-level PartialConv U-Net + skip, random-init downstream",
    "H1-A6": "3-level PartialConv U-Net, no skip",
}

EVALUATION_DIRS = {
    "H1": {
        42: "h1_hybrid_pconv1_unet_frozen_fixed_points_r05_mask_00006_seed42_val300_20260727_v1",
    },
    "H1-A1": {
        42: "h1_a1_partialconv_unet_frozen_fixed_points_r05_mask_00006_seed42_val300_20260727_v1",
        43: "h1_a4_pretrain_partialconv3_unet_skip_fixed_points_r05_mask_00006_seed43_val300_20260728_v1",
        44: "h1_a4_pretrain_partialconv3_unet_skip_fixed_points_r05_mask_00006_seed44_val300_20260728_v1",
    },
    "H1-A2": {
        42: "h1_a2_no_skip_frozen_fixed_points_r05_mask_00006_seed42_val300_20260728_v1",
    },
    "H1-A3-pretrained": {
        seed: f"h1_a3_pretrained_seed{seed}_val300_fixed_r05_mask_00006"
        for seed in SEEDS
    },
    "H1-A3-random-init": {
        seed: f"h1_a3_random_init_seed{seed}_val300_fixed_r05_mask_00006"
        for seed in SEEDS
    },
    "H1-A4": {
        seed: f"h1_a4_pretrained_seed{seed}_val300_20260728_v1"
        for seed in SEEDS
    },
    "H1-A5": {
        seed: f"h1_a5_random_init_seed{seed}_val300_20260728_v1"
        for seed in SEEDS
    },
    "H1-A6": {
        42: "h1_a6_no_skip_partialconv3_unet_frozen_seed42_val300_20260728_v1",
    },
}

PAIR_EFFECTS = (
    ("pretraining_multilevel", "H1-A4", "H1-A5", True),
    ("partialconv_depth_pretrained", "H1-A4", "H1-A3-pretrained", True),
    ("partialconv_depth_random_init", "H1-A5", "H1-A3-random-init", True),
    ("skip_connections_multilevel", "H1-A1", "H1-A6", True),
    ("mask_unet_vs_best_hybrid", "B4", "H1-A4", False),
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
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


def _metric_values(metrics: Mapping[str, Any]) -> dict[str, float]:
    values = {
        "reconstruction_missing_nrmse": float(
            metrics["history_reconstruction"]["missing_region"]["nrmse"]
        )
    }
    for horizon in HORIZONS:
        forecast = metrics[f"forecast_{horizon}"]
        values.update(
            {
                f"f{horizon}_nrmse": float(forecast["nrmse"]),
                f"f{horizon}_corr": float(forecast["correlation"]),
                f"f{horizon}_ssp": float(forecast["ssp"]),
                f"f{horizon}_gradient_nrmse": float(
                    forecast["gradient_nrmse"]
                ),
            }
        )
    return values


def _b4_rows(summary_path: Path) -> dict[int, dict[str, Any]]:
    with summary_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["candidate"] == "b4"
            and row["mask_type"] == "fixed_points"
            and math.isclose(float(row["observation_rate"]), 0.05)
            and row["mask_id"] == "mask_00006"
        ]
    if len(rows) != 3:
        raise ValueError(f"Expected three fixed-5% B4 rows, got {len(rows)}")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row["train_seed"])
        values = {
            "reconstruction_missing_nrmse": float(row["recon_missing_nrmse"])
        }
        for horizon in HORIZONS:
            values.update(
                {
                    f"f{horizon}_nrmse": float(row[f"f{horizon}_nrmse"]),
                    f"f{horizon}_corr": float(row[f"f{horizon}_corr"]),
                    f"f{horizon}_ssp": float(row[f"f{horizon}_ssp"]),
                    f"f{horizon}_gradient_nrmse": float(
                        row[f"f{horizon}_gradient_nrmse"]
                    ),
                }
            )
        result[seed] = {
            "metrics": values,
            "checkpoint_sha256": row["checkpoint_sha256"],
            "metrics_path": row["metrics_path"],
        }
    return result


def _load_methods(
    run_root: Path, b4_summary: Path
) -> dict[str, dict[int, dict[str, Any]]]:
    methods = {"B4": _b4_rows(b4_summary)}
    for method_id, seed_dirs in EVALUATION_DIRS.items():
        methods[method_id] = {}
        for seed, directory in seed_dirs.items():
            path = run_root / directory / "evaluation" / "val_metrics.json"
            payload = _read_json(path)
            metrics = payload["metrics"]
            if (
                payload.get("split") != "val"
                or payload.get("evaluation_kind") != "formal"
                or int(metrics["rollout_steps"]) != 300
                or int(metrics["evaluated_samples"]) != 777
            ):
                raise ValueError(f"Incomplete formal val result: {method_id}/{seed}")
            freeze = payload["freeze_checks"]
            if not (
                freeze["rfno_requires_grad_false"]
                and freeze["rfno_gradients_none"]
            ):
                raise ValueError(f"RFNO freeze check failed: {method_id}/{seed}")
            methods[method_id][seed] = {
                "metrics": _metric_values(metrics),
                "checkpoint_sha256": payload["evaluated_checkpoint"]["sha256"],
                "metrics_path": str(path.resolve()),
            }
    return methods


def _scorecard_rows(
    methods: Mapping[str, Mapping[int, Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id, seed_rows in methods.items():
        row: dict[str, Any] = {
            "method_id": method_id,
            "method": METHOD_LABELS[method_id],
            "n_seeds": len(seed_rows),
            "uncertainty_basis": (
                "sample_std_across_seeds"
                if len(seed_rows) >= 3
                else "single_seed_descriptive_only"
            ),
            "seeds": ";".join(str(seed) for seed in sorted(seed_rows)),
        }
        metric_names = list(next(iter(seed_rows.values()))["metrics"])
        for metric in metric_names:
            values = [
                float(seed_rows[seed]["metrics"][metric])
                for seed in sorted(seed_rows)
            ]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_sample_std"] = (
                statistics.stdev(values) if len(values) >= 2 else ""
            )
        row["checkpoint_sha256_by_seed"] = ";".join(
            f"{seed}:{seed_rows[seed]['checkpoint_sha256']}"
            for seed in sorted(seed_rows)
        )
        row["metrics_path_by_seed"] = ";".join(
            f"{seed}:{seed_rows[seed]['metrics_path']}"
            for seed in sorted(seed_rows)
        )
        rows.append(row)
    rows.sort(key=lambda row: float(row["f300_nrmse_mean"]))
    for rank, row in enumerate(rows, start=1):
        row["fixed_layout_f300_nrmse_rank"] = rank
    return rows


def _paired_effect_rows(
    methods: Mapping[str, Mapping[int, Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for effect_id, left, right, strict_factor_isolation in PAIR_EFFECTS:
        common_seeds = sorted(set(methods[left]) & set(methods[right]))
        metric_names = list(next(iter(methods[left].values()))["metrics"])
        for metric in metric_names:
            differences = [
                float(methods[left][seed]["metrics"][metric])
                - float(methods[right][seed]["metrics"][metric])
                for seed in common_seeds
            ]
            mean = statistics.fmean(differences)
            row: dict[str, Any] = {
                "effect_id": effect_id,
                "left_method": left,
                "right_method": right,
                "difference": f"{left}_minus_{right}",
                "metric": metric,
                "n_pairs": len(differences),
                "paired_mean": mean,
                "strict_factor_isolation": strict_factor_isolation,
            }
            if len(differences) >= 3:
                sample_std = statistics.stdev(differences)
                standard_error = sample_std / math.sqrt(len(differences))
                margin = T_CRITICAL_DF2_95 * standard_error
                low, high = mean - margin, mean + margin
                if standard_error == 0.0:
                    p_value = 1.0 if mean == 0.0 else 0.0
                else:
                    p_value = float(
                        2.0
                        * student_t.sf(
                            abs(mean / standard_error),
                            df=len(differences) - 1,
                        )
                    )
                row.update(
                    {
                        "paired_sample_std": sample_std,
                        "paired_t_ci95_low": low,
                        "paired_t_ci95_high": high,
                        "paired_t_p_value": p_value,
                        "ci_excludes_zero": low > 0.0 or high < 0.0,
                        "evidence_status": (
                            "paired_three_seed"
                            if strict_factor_isolation
                            else "paired_performance_comparison_noncausal"
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "paired_sample_std": "",
                        "paired_t_ci95_low": "",
                        "paired_t_ci95_high": "",
                        "paired_t_p_value": "",
                        "ci_excludes_zero": "",
                        "evidence_status": "single_seed_no_inference",
                    }
                )
            rows.append(row)
    return rows


def summarize(
    run_root: Path,
    b4_summary: Path,
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    methods = _load_methods(run_root, b4_summary)
    scorecard = _scorecard_rows(methods)
    effects = _paired_effect_rows(methods)
    _write_csv(output_dir / "hybrid_partialconv_factorial_scorecard.csv", scorecard)
    _write_csv(output_dir / "paired_factor_effects.csv", effects)
    manifest = {
        "experiment_id": "hybrid-partialconv-factorial-scorecard",
        "split": "val",
        "mask_id": "mask_00006",
        "observation_rate": 0.05,
        "rollout_steps": 300,
        "ranking_metric": "f300_nrmse_mean",
        "composite_score_used": False,
        "test_accessed": False,
        "single_seed_methods": [
            row["method_id"] for row in scorecard if int(row["n_seeds"]) == 1
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/sparse"))
    parser.add_argument(
        "--b4-summary",
        type=Path,
        default=Path(
            "results/finalist_robustness_20260722/candidate_val_summary.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run_root, args.b4_summary, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
