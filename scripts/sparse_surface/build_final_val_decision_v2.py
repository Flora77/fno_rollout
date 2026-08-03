#!/usr/bin/env python3
"""Build the final validation decision v2 from existing validation artifacts only."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


HORIZONS = (30, 60, 120, 180, 240, 300)
FORECAST_FIELDS = ("nrmse", "corr", "ssp", "gradient_nrmse")
METRICS = (
    "recon_missing_nrmse",
    *(f"f{horizon}_{field}" for horizon in HORIZONS for field in FORECAST_FIELDS),
)
HIGHER_IS_BETTER = {f"f{horizon}_corr" for horizon in HORIZONS}
METHODS = (
    ("B4", "Mask U-Net + frozen RFNO"),
    ("P1-pretrained", "PartialConv-MAE pretrained + frozen RFNO"),
    ("P1-random-init", "PartialConv-MAE random-init + frozen RFNO"),
    ("P2", "PartialConv-MAE + joint RFNO"),
    ("B1", "Bilinear + frozen RFNO"),
    ("B3", "Train-only POD rank64 + frozen RFNO"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = abs(float(differences.mean()))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        values.append(abs(float((differences * np.asarray(signs)).mean())))
    return float(np.mean(np.asarray(values) >= observed - 1.0e-15))


def _old_fixed_row(scorecard: pd.DataFrame, method_id: str) -> dict:
    selected = scorecard[
        (scorecard["scenario"] == "fixed_points_5pct")
        & (scorecard["method_id"] == method_id)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one fixed-layout row for {method_id}")
    source = selected.iloc[0]
    row = {
        "scenario": "fixed_points_5pct",
        "method_id": method_id,
        "method": dict(METHODS)[method_id],
        "availability": "available",
        "n_runs": int(source["n_runs"]),
        "n_seeds": int(source["n_seeds"]),
        "uncertainty_basis": source["uncertainty_basis"],
        "source_note": source.get("source_note", ""),
    }
    for metric in METRICS:
        row[f"{metric}_mean"] = source.get(f"{metric}_mean", np.nan)
        row[f"{metric}_sample_std"] = source.get(
            f"{metric}_sample_std", np.nan
        )
    return row


def _new_p1_row(summary: pd.DataFrame, variant: str) -> dict:
    selected = summary[summary["variant"] == variant]
    if len(selected) != 1:
        raise ValueError(f"Expected one aggregate P1 row for {variant}")
    source = selected.iloc[0]
    method_id = "P1-pretrained" if variant == "pretrained" else "P1-random-init"
    row = {
        "scenario": "fixed_points_5pct",
        "method_id": method_id,
        "method": dict(METHODS)[method_id],
        "availability": "available",
        "n_runs": int(source["n_seeds"]),
        "n_seeds": int(source["n_seeds"]),
        "uncertainty_basis": "sample_std_across_paired_train_seeds",
        "source_note": "strict 30-epoch downstream initialization ablation",
    }
    mappings = {"recon_missing_nrmse": "reconstruction_missing_nrmse"}
    mappings.update(
        {
            f"f{horizon}_{field}": f"forecast_{horizon}_{field}"
            for horizon in HORIZONS
            for field in FORECAST_FIELDS
        }
    )
    for target, source_name in mappings.items():
        row[f"{target}_mean"] = float(source[f"{source_name}_mean"])
        row[f"{target}_sample_std"] = float(
            source[f"{source_name}_sample_std"]
        )
    return row


def _paired_initialization_ci(by_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mappings = {"recon_missing_nrmse": "reconstruction_missing_nrmse"}
    mappings.update(
        {
            f"f{horizon}_{field}": f"forecast_{horizon}_{field}"
            for horizon in HORIZONS
            for field in FORECAST_FIELDS
        }
    )
    for metric, source_name in mappings.items():
        pretrained = (
            by_seed[by_seed["variant"] == "pretrained"]
            .sort_values("seed")[source_name]
            .to_numpy(dtype=float)
        )
        random_init = (
            by_seed[by_seed["variant"] == "random_init"]
            .sort_values("seed")[source_name]
            .to_numpy(dtype=float)
        )
        if len(pretrained) != 3 or len(random_init) != 3:
            raise ValueError("Initialization comparison requires three paired seeds")
        differences = pretrained - random_init
        mean = float(differences.mean())
        sample_std = float(differences.std(ddof=1))
        margin = float(stats.t.ppf(0.975, df=2) * sample_std / math.sqrt(3))
        higher_is_better = metric in HIGHER_IS_BETTER
        favored = differences > 0.0 if higher_is_better else differences < 0.0
        rows.append(
            {
                "metric": metric,
                "direction": (
                    "higher_is_better" if higher_is_better else "lower_is_better"
                ),
                "n_paired_seeds": 3,
                "pretrained_mean": float(pretrained.mean()),
                "random_init_mean": float(random_init.mean()),
                "paired_delta_pretrained_minus_random_mean": mean,
                "paired_delta_sample_std": sample_std,
                "paired_t_ci95_low": mean - margin,
                "paired_t_ci95_high": mean + margin,
                "paired_t_pvalue_two_sided": float(
                    stats.ttest_rel(pretrained, random_init).pvalue
                ),
                "exact_sign_flip_pvalue_two_sided": _exact_sign_flip_pvalue(
                    differences
                ),
                "ci_excludes_zero": bool(mean - margin > 0.0 or mean + margin < 0.0),
                "all_seeds_favor_pretrained": bool(np.all(favored)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-scorecard",
        type=Path,
        default=Path(
            "results/final_method_val_scorecard_20260722_v2/method_val_scorecard.csv"
        ),
    )
    parser.add_argument(
        "--initialization-dir",
        type=Path,
        default=Path("results/p1_init_ablation_3seed_val300_20260722"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final_val_decision_v2_20260723"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory exists: {args.output_dir}")

    old = pd.read_csv(args.old_scorecard)
    p1_summary = pd.read_csv(args.initialization_dir / "summary_three_seed.csv")
    p1_by_seed = pd.read_csv(args.initialization_dir / "summary_by_seed.csv")

    fixed_rows = [
        _old_fixed_row(old, "B4"),
        _new_p1_row(p1_summary, "pretrained"),
        _new_p1_row(p1_summary, "random_init"),
        _old_fixed_row(old, "P2"),
        _old_fixed_row(old, "B1"),
        _old_fixed_row(old, "B3"),
    ]
    fixed = pd.DataFrame(fixed_rows).sort_values("f300_nrmse_mean").reset_index(
        drop=True
    )
    fixed["f300_nrmse_rank"] = np.arange(1, len(fixed) + 1)
    fixed["rank_basis"] = "F300 NRMSE only; no composite score"

    b4_ood_source = old[
        (old["scenario"] == "random_points_5pct_ood")
        & (old["method_id"] == "B4")
    ]
    if len(b4_ood_source) != 1:
        raise ValueError("Expected one archived B4 OOD row")
    b4_ood = b4_ood_source.iloc[0]
    ood_rows = []
    for method_id, method in METHODS:
        row = {
            "scenario": "random_points_5pct_ood",
            "method_id": method_id,
            "method": method,
            "availability": "available" if method_id == "B4" else "unavailable",
            "ood_rank": np.nan,
            "rank_status": "not_ranked_fewer_than_two_exact_comparable_methods",
            "source_note": (
                "three random-mask replicates from the existing B4 checkpoint"
                if method_id == "B4"
                else "no existing OOD artifact for this exact locked method"
            ),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = (
                b4_ood.get(f"{metric}_mean", np.nan)
                if method_id == "B4"
                else np.nan
            )
            row[f"{metric}_sample_std"] = (
                b4_ood.get(f"{metric}_sample_std", np.nan)
                if method_id == "B4"
                else np.nan
            )
        ood_rows.append(row)
    ood = pd.DataFrame(ood_rows)

    scorecard = pd.concat((fixed, ood), ignore_index=True, sort=False)
    paired = _paired_initialization_ci(p1_by_seed)

    locked_checkpoint = Path(
        "runs/sparse/b4_mask_unet_frozen_fixed_points_r05_mask_00006_seed42_"
        "formal_20260721_v2/checkpoints/best.pt"
    ).resolve()
    expected_hash = "1a8de232140778fb2275caa9d759eeb5200f1778e490cf3b504898eae74964fc"
    if not locked_checkpoint.is_file() or _sha256(locked_checkpoint) != expected_hash:
        raise ValueError("Previously locked canonical B4 checkpoint is missing or changed")

    args.output_dir.mkdir(parents=True)
    scorecard.to_csv(
        args.output_dir / "method_val_scorecard_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fixed.to_csv(
        args.output_dir / "fixed_layout_ranking_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ood.to_csv(
        args.output_dir / "random_points_ood_ranking_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        args.output_dir / "pretraining_initialization_paired_ci95.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lock = {
        "experiment_id": "final-val-decision-v2",
        "primary_scenario": "fixed_points_5pct_mask_00006",
        "ranking_metric": "forecast_300_nrmse",
        "composite_score_used": False,
        "locked_method_id": "B4",
        "locked_method": dict(METHODS)["B4"],
        "locked_checkpoint_path": str(locked_checkpoint),
        "locked_checkpoint_sha256": expected_hash,
        "checkpoint_policy": (
            "retain the previously locked canonical seed42 checkpoint; do not select "
            "the luckiest seed from the three-seed validation sweep"
        ),
        "fixed_layout_rank": 1,
        "random_points_ood_decision": (
            "not rankable: fewer than two exact locked methods have existing OOD artifacts"
        ),
        "excluded_ood_evidence": (
            "historical P1-no-observation-consistency checkpoints use a different "
            "training protocol and are not relabeled as P1-pretrained"
        ),
        "test_accessed": False,
    }
    (args.output_dir / "final_method_lock.json").write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "experiment_id": "final-val-decision-v2",
        "source_scorecard": str(args.old_scorecard.resolve()),
        "source_initialization_dir": str(args.initialization_dir.resolve()),
        "rows": int(len(scorecard)),
        "fixed_methods": int(len(fixed)),
        "exact_ood_methods": int((ood["availability"] == "available").sum()),
        "test_accessed": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote final validation decision v2 to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
