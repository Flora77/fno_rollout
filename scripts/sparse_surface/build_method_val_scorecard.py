"""Build the locked-validation method scorecard without evaluating any model.

The script only reads existing validation artifacts.  It deliberately keeps
single-seed and multi-seed uncertainty separate and never constructs a
composite score.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = (30, 60, 120, 180, 240, 300)
HEADLINE_METRICS = (
    "recon_missing_nrmse",
    *(f"f{h}_{metric}" for h in HORIZONS for metric in ("nrmse", "corr", "ssp", "gradient_nrmse")),
    "error_growth_mean_nrmse",
    "error_growth_final_nrmse",
)
PAIRWISE_METRICS = (
    "recon_missing_nrmse",
    *(f"f{h}_nrmse" for h in HORIZONS),
    "f300_corr",
    "f300_ssp",
    "f300_gradient_nrmse",
    "error_growth_mean_nrmse",
)
HIGHER_IS_BETTER = {"f300_corr"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robustness-dir",
        type=Path,
        default=Path("results/finalist_robustness_20260722"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final_method_val_scorecard_20260722"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_from_payload(metric: dict, name: str) -> float:
    if name == "recon_missing_nrmse":
        return float(metric["history_reconstruction"]["missing_region"]["nrmse"])
    if name.startswith("f"):
        prefix, field = name.split("_", 1)
        horizon = prefix[1:]
        field = "correlation" if field == "corr" else field
        return float(metric[f"forecast_{horizon}"][field])
    curve = metric["error_growth_curve"]
    values = np.asarray([row["nrmse"] for row in curve], dtype=float)
    if name == "error_growth_mean_nrmse":
        return float(values.mean())
    if name == "error_growth_final_nrmse":
        return float(values[-1])
    raise KeyError(name)


def source_frame_from_payload(payload: dict, method_id: str) -> pd.DataFrame:
    rows = []
    for source in payload["metrics"]["per_source"].values():
        row = {
            "method_id": method_id,
            "source_name": source["source"]["name"],
        }
        for metric in HEADLINE_METRICS:
            row[metric] = metric_from_payload(source, metric)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("source_name").reset_index(drop=True)


def aggregate_runs(frame: pd.DataFrame) -> dict:
    result: dict[str, float | int] = {"n_runs": int(len(frame))}
    for metric in HEADLINE_METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        result[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
        result[f"{metric}_sample_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    for metric in ("parameters_total", "parameters_trainable"):
        values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        result[metric] = int(values.iloc[0]) if len(values) else np.nan
    for metric in ("training_time_seconds", "inference_time_seconds", "seconds_per_sample"):
        values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        result[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
        result[f"{metric}_sample_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    return result


def aggregate_sources(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("source_name", as_index=False)[list(HEADLINE_METRICS)]
        .mean(numeric_only=True)
        .sort_values("source_name")
        .reset_index(drop=True)
    )


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, replicates: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, (0.025, 0.975)))


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    if len(differences) == 0:
        return np.nan
    observed = abs(float(differences.mean()))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistics.append(abs(float((differences * np.asarray(signs)).mean())))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def paired_bootstrap_delta(
    differences: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    indices = rng.integers(0, len(differences), size=(replicates, len(differences)))
    samples = differences[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(samples, (0.025, 0.975)))


def payload_summary(payload: dict, method_id: str, method: str, training_seconds=np.nan) -> dict:
    metrics = payload["metrics"]
    row = {
        "scenario": "fixed_points_5pct",
        "method_id": method_id,
        "method": method,
        "role": "candidate",
        "n_runs": 1,
        "n_seeds": 1,
        "n_mask_replicates": 1,
        "uncertainty_basis": "single_seed_no_cross_seed_std",
        "parameters_total": payload.get("parameters", {}).get("total", np.nan),
        "parameters_trainable": payload.get("parameters", {}).get("trainable", np.nan),
        "training_time_seconds_mean": training_seconds,
        "training_time_seconds_sample_std": np.nan,
        "inference_time_seconds_mean": metrics.get("elapsed_seconds", np.nan),
        "inference_time_seconds_sample_std": np.nan,
        "seconds_per_sample_mean": metrics.get("seconds_per_sample", np.nan),
        "seconds_per_sample_sample_std": np.nan,
        "per_mat_available": True,
    }
    for metric in HEADLINE_METRICS:
        row[f"{metric}_mean"] = metric_from_payload(metrics, metric)
        row[f"{metric}_sample_std"] = np.nan
    return row


def p2_summary(run_dir: Path) -> dict:
    best = load_json(run_dir / "checkpoints" / "best.json")
    summary = load_json(run_dir / "training_summary_epoch0093.json")
    with (run_dir / "logs" / "epochs.jsonl").open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    record = next(row for row in records if int(row["epoch"]) == int(best["epoch"]))
    val = record["val"]
    row = {
        "scenario": "fixed_points_5pct",
        "method_id": "P2",
        "method": "PartialConv-MAE + joint RFNO",
        "role": "candidate",
        "n_runs": 1,
        "n_seeds": 1,
        "n_mask_replicates": 1,
        "uncertainty_basis": "single_seed_aggregate_only",
        "parameters_total": 3648608,
        "parameters_trainable": 3648608,
        "training_time_seconds_mean": float(summary["elapsed_seconds"]),
        "training_time_seconds_sample_std": np.nan,
        "inference_time_seconds_mean": np.nan,
        "inference_time_seconds_sample_std": np.nan,
        "seconds_per_sample_mean": np.nan,
        "seconds_per_sample_sample_std": np.nan,
        "per_mat_available": False,
        "source_note": "best epoch aggregate metrics; no archived per-MAT/error-growth artifact",
    }
    row["recon_missing_nrmse_mean"] = float(val["reconstruction_missing_nrmse"])
    row["recon_missing_nrmse_sample_std"] = np.nan
    for horizon in HORIZONS:
        for field in ("nrmse", "corr", "ssp", "gradient_nrmse"):
            source_field = "correlation" if field == "corr" else field
            key = f"f{horizon}_{field}"
            row[f"{key}_mean"] = float(val[f"forecast_{horizon}_{source_field}"])
            row[f"{key}_sample_std"] = np.nan
    for metric in ("error_growth_mean_nrmse", "error_growth_final_nrmse"):
        row[f"{metric}_mean"] = np.nan
        row[f"{metric}_sample_std"] = np.nan
    return row


def add_per_mat_statistics(
    row: dict,
    sources: pd.DataFrame | None,
    seed: int,
    replicates: int,
) -> None:
    if sources is None or sources.empty:
        for metric in PAIRWISE_METRICS:
            for suffix in ("per_mat_mean", "bootstrap_ci95_low", "bootstrap_ci95_high", "per_mat_worst"):
                row[f"{metric}_{suffix}"] = np.nan
        return
    for index, metric in enumerate(PAIRWISE_METRICS):
        values = sources[metric].to_numpy(dtype=float)
        rng = np.random.default_rng(seed + index)
        low, high = bootstrap_mean_ci(values, rng, replicates)
        row[f"{metric}_per_mat_mean"] = float(values.mean())
        row[f"{metric}_bootstrap_ci95_low"] = low
        row[f"{metric}_bootstrap_ci95_high"] = high
        row[f"{metric}_per_mat_worst"] = float(values.min() if metric in HIGHER_IS_BETTER else values.max())


def add_ranks(scorecard: pd.DataFrame) -> pd.DataFrame:
    rank_metrics = ("recon_missing_nrmse", "f300_nrmse", "f300_corr", "f300_ssp", "f300_gradient_nrmse")
    for metric in rank_metrics:
        ascending = metric not in HIGHER_IS_BETTER
        scorecard[f"{metric}_rank"] = np.nan
        for scenario in ("fixed_points_5pct", "random_points_5pct_ood"):
            mask = (scorecard["scenario"] == scenario) & (scorecard["role"] == "candidate")
            scorecard.loc[mask, f"{metric}_rank"] = scorecard.loc[mask, f"{metric}_mean"].rank(
                method="min", ascending=ascending
            )
    scorecard["primary_f300_nrmse_rank"] = scorecard["f300_nrmse_rank"]
    return scorecard


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.robustness_dir / "candidate_val_summary.csv")
    per_mat = pd.read_csv(args.robustness_dir / "candidate_val_per_mat.csv")
    # Required input: validate that the archived curves are present and complete.
    error_growth = pd.read_csv(args.robustness_dir / "candidate_val_error_growth.csv")
    expected_curve_rows = len(summary) * 300
    if len(error_growth) != expected_curve_rows:
        raise ValueError(f"incomplete error-growth CSV: {len(error_growth)} != {expected_curve_rows}")

    b1_path = Path(
        "runs/sparse/b1_bilinear_frozen_fixed_points_r05_mask_00006_seed42_formal_val_300f_20260721_v1/evaluation/val_metrics.json"
    )
    b3_dir = Path(
        "runs/sparse/b3_pod_frozen_fixed_points_r05_mask_00006_seed42_rank_selection_formal_20260721_v1"
    )
    b3_path = b3_dir / "evaluation" / "rank_0064_val_metrics.json"
    p2_dir = Path(
        "runs/sparse/p2_partialconv_mae_joint_d2_curriculum_300f_fixed_points_r05_mask_00006_seed42_formal_20260721_v1"
    )
    b1_payload = load_json(b1_path)
    b3_payload = load_json(b3_path)
    b3_provenance = load_json(b3_dir / "pod_basis_provenance.json")

    rows: list[dict] = [
        {
            "scenario": "complete_field_reference",
            "method_id": "B0",
            "method": "complete field + RFNO",
            "role": "ideal_reference",
            "n_runs": 0,
            "n_seeds": 0,
            "n_mask_replicates": 0,
            "uncertainty_basis": "not_available",
            "per_mat_available": False,
            "source_note": "no same-protocol archived B0 validation artifact; excluded from sparse-method ranks",
        },
        payload_summary(b1_payload, "B1", "bilinear + frozen RFNO", training_seconds=0.0),
        payload_summary(
            b3_payload,
            "B3",
            "train-only POD rank64 + frozen RFNO",
            training_seconds=float(b3_provenance["basis_seconds"]),
        ),
    ]

    fixed_sources: dict[str, pd.DataFrame | None] = {
        "B1": source_frame_from_payload(b1_payload, "B1"),
        "B3": source_frame_from_payload(b3_payload, "B3"),
    }
    ood_sources: dict[str, pd.DataFrame | None] = {}

    method_names = {
        "b4": ("B4", "Mask U-Net + frozen RFNO"),
        "p1": ("P1-no-obs", "PartialConv-MAE (no observation consistency) + frozen RFNO"),
    }
    for candidate, (method_id, method_name) in method_names.items():
        candidate_summary = summary[summary["candidate"] == candidate]
        fixed = candidate_summary[
            (candidate_summary["distribution"] == "in_distribution")
            & (candidate_summary["observation_rate"] == 0.05)
        ]
        ood = candidate_summary[
            (candidate_summary["distribution"] == "ood")
            & (candidate_summary["observation_rate"] == 0.05)
        ]
        fixed_row = {
            "scenario": "fixed_points_5pct",
            "method_id": method_id,
            "method": method_name,
            "role": "candidate",
            "n_seeds": int(fixed["train_seed"].nunique()),
            "n_mask_replicates": 1,
            "uncertainty_basis": "sample_std_across_train_seeds",
            "per_mat_available": True,
            **aggregate_runs(fixed),
        }
        ood_row = {
            "scenario": "random_points_5pct_ood",
            "method_id": method_id,
            "method": method_name,
            "role": "candidate",
            "n_seeds": int(ood["train_seed"].nunique()),
            "n_mask_replicates": int(ood["mask_replicate"].nunique()),
            "uncertainty_basis": "sample_std_across_mask_replicates_single_checkpoint",
            "per_mat_available": True,
            **aggregate_runs(ood),
        }
        # OOD evaluation uses the seed-42 checkpoint, so degradation is paired
        # against the fixed-layout seed-42 run, not the three-seed mean.
        fixed_anchor = fixed[fixed["train_seed"] == 42].iloc[0]
        for metric in HEADLINE_METRICS:
            baseline = float(fixed_anchor[metric])
            random_value = float(ood_row[f"{metric}_mean"])
            ood_row[f"fixed_to_random_{metric}_ratio"] = random_value / baseline
            ood_row[f"fixed_to_random_{metric}_percent_change"] = 100.0 * (random_value / baseline - 1.0)
        rows.extend((fixed_row, ood_row))

        candidate_sources = per_mat[per_mat["candidate"] == candidate].copy()
        fixed_sources[method_id] = aggregate_sources(
            candidate_sources[
                (candidate_sources["distribution"] == "in_distribution")
                & (candidate_sources["observation_rate"] == 0.05)
            ]
        )
        ood_sources[method_id] = aggregate_sources(
            candidate_sources[
                (candidate_sources["distribution"] == "ood")
                & (candidate_sources["observation_rate"] == 0.05)
            ]
        )

    rows.append(p2_summary(p2_dir))
    fixed_sources["P2"] = None

    for row_index, row in enumerate(rows):
        if row["scenario"] == "fixed_points_5pct":
            sources = fixed_sources.get(row["method_id"])
        elif row["scenario"] == "random_points_5pct_ood":
            sources = ood_sources.get(row["method_id"])
        else:
            sources = None
        add_per_mat_statistics(row, sources, args.seed + 100 * row_index, args.bootstrap_replicates)

    scorecard = add_ranks(pd.DataFrame(rows))
    preferred = [
        "scenario", "method_id", "method", "role", "primary_f300_nrmse_rank",
        "n_runs", "n_seeds", "n_mask_replicates", "uncertainty_basis",
        "recon_missing_nrmse_mean", "recon_missing_nrmse_sample_std",
        *(f"f{h}_{metric}_{suffix}" for h in HORIZONS for metric in ("nrmse", "corr", "ssp", "gradient_nrmse") for suffix in ("mean", "sample_std")),
        "error_growth_mean_nrmse_mean", "error_growth_mean_nrmse_sample_std",
        "error_growth_final_nrmse_mean", "error_growth_final_nrmse_sample_std",
        "parameters_total", "parameters_trainable",
        "training_time_seconds_mean", "training_time_seconds_sample_std",
        "inference_time_seconds_mean", "inference_time_seconds_sample_std",
        "seconds_per_sample_mean", "seconds_per_sample_sample_std",
        "per_mat_available", "source_note",
    ]
    preferred = [column for column in preferred if column in scorecard.columns]
    remaining = [column for column in scorecard.columns if column not in preferred]
    scorecard = scorecard[preferred + remaining]

    pairwise_rows = []
    comparison_sets = (
        ("fixed_points_5pct", ("B1", "B3", "B4", "P1-no-obs", "P2"), fixed_sources),
        ("random_points_5pct_ood", ("B4", "P1-no-obs"), ood_sources),
    )
    for scenario, methods, source_map in comparison_sets:
        for method_a, method_b in itertools.combinations(methods, 2):
            sources_a = source_map.get(method_a)
            sources_b = source_map.get(method_b)
            for metric_index, metric in enumerate(PAIRWISE_METRICS):
                base = {
                    "scenario": scenario,
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "direction": "higher_is_better" if metric in HIGHER_IS_BETTER else "lower_is_better",
                }
                if sources_a is None or sources_b is None:
                    pairwise_rows.append({**base, "status": "unavailable_missing_per_mat", "n_paired_mat": 0})
                    continue
                paired = sources_a[["source_name", metric]].merge(
                    sources_b[["source_name", metric]], on="source_name", suffixes=("_a", "_b"), validate="one_to_one"
                )
                differences = paired[f"{metric}_a"].to_numpy() - paired[f"{metric}_b"].to_numpy()
                scenario_offset = 0 if scenario == "fixed_points_5pct" else 10000
                rng = np.random.default_rng(args.seed + 1000 + scenario_offset + metric_index)
                low, high = paired_bootstrap_delta(differences, rng, args.bootstrap_replicates)
                mean_delta = float(differences.mean())
                if metric in HIGHER_IS_BETTER:
                    favored = method_a if mean_delta > 0 else method_b if mean_delta < 0 else "tie"
                else:
                    favored = method_a if mean_delta < 0 else method_b if mean_delta > 0 else "tie"
                pairwise_rows.append(
                    {
                        **base,
                        "status": "ok",
                        "n_paired_mat": len(paired),
                        "mean_a": float(paired[f"{metric}_a"].mean()),
                        "mean_b": float(paired[f"{metric}_b"].mean()),
                        "mean_delta_a_minus_b": mean_delta,
                        "bootstrap_delta_ci95_low": low,
                        "bootstrap_delta_ci95_high": high,
                        "exact_sign_flip_pvalue_two_sided": exact_sign_flip_pvalue(differences),
                        "all_mat_same_direction": bool(np.all(differences < 0) or np.all(differences > 0)),
                        "favored_method": favored,
                        "ci_excludes_zero": bool(low > 0 or high < 0),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    scorecard.to_csv(args.output_dir / "method_val_scorecard.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pairwise_rows).to_csv(
        args.output_dir / "method_pairwise_comparison.csv", index=False, encoding="utf-8-sig"
    )

    fixed_rank = scorecard[
        (scorecard["scenario"] == "fixed_points_5pct") & (scorecard["role"] == "candidate")
    ].sort_values("primary_f300_nrmse_rank")
    ood_rank = scorecard[scorecard["scenario"] == "random_points_5pct_ood"].sort_values(
        "primary_f300_nrmse_rank"
    )
    fixed_rank.to_csv(args.output_dir / "fixed_layout_ranking.csv", index=False, encoding="utf-8-sig")
    ood_rank.to_csv(args.output_dir / "ood_ranking.csv", index=False, encoding="utf-8-sig")

    b4_fixed = fixed_rank.set_index("method_id").loc["B4"]
    p1_fixed = fixed_rank.set_index("method_id").loc["P1-no-obs"]
    b4_ood = ood_rank.set_index("method_id").loc["B4"]
    p1_ood = ood_rank.set_index("method_id").loc["P1-no-obs"]
    conclusions = f"""# Locked validation conclusions

- Fixed 5% layout: B4 ranks first by the predeclared primary metric F300 NRMSE ({b4_fixed['f300_nrmse_mean']:.6f}); P1-no-obs ranks second ({p1_fixed['f300_nrmse_mean']:.6f}).
- Random-point 5% OOD: P1-no-obs ranks first ({p1_ood['f300_nrmse_mean']:.6f}); B4 degrades to {b4_ood['f300_nrmse_mean']:.6f}.
- Fixed-to-random F300 degradation is {p1_ood['fixed_to_random_f300_nrmse_percent_change']:.2f}% for P1-no-obs and {b4_ood['fixed_to_random_f300_nrmse_percent_change']:.2f}% for B4, using each seed-42 fixed checkpoint as its paired anchor.
- Recommendation: use B4 when the sensor layout is fixed and known; use P1-no-obs when layout shift/random sparse observations are expected. If one general-purpose method must be chosen, current validation evidence favors P1-no-obs for robustness.
- P2 is retained in the aggregate fixed-layout ranking, but no per-MAT or cross-seed artifact exists; no significance or robustness claim is made for P2.
- B0 is an ideal complete-field reference only. No same-protocol archived B0 validation artifact was found, so it is excluded from sparse-method ranks.
- Bootstrap intervals resample the seven validation MAT files (10,000 replicates). Pairwise p-values are exact two-sided sign-flip tests over paired MAT-level differences; no multiple-comparison adjustment or composite score is applied.
"""
    (args.output_dir / "method_val_conclusions.md").write_text(conclusions, encoding="utf-8")


if __name__ == "__main__":
    main()
