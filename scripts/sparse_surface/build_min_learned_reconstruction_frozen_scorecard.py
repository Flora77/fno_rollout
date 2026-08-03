"""Build the MIN learned-reconstruction frozen validation scorecard.

This script only reads archived validation/training metadata.  It does not load
datasets or checkpoints and therefore cannot construct a test loader.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/min_learned_reconstruction_frozen_scorecard_20260730_v1"
SEEDS = (42, 43, 44)
HORIZONS = (30, 60, 120, 180, 240, 300)
T_CRIT_DF2 = 4.302652729911275

P1 = ROOT / "results/p1_init_ablation_3seed_val300_20260722"
B4 = ROOT / "results/min_b4_pretraining_initialization_ablation_20260729_v1"
H1 = ROOT / "results/h1_a4_a5_initialization_ablation_20260728"
VIT = ROOT / "results/min_v0_random_init_val300_20260729_v1"
B5 = ROOT / "results/min_b5_gno_gino_v2_three_seed_20260729_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latest_summary(run_dir: Path) -> dict:
    files = sorted(run_dir.glob("training_summary*.json"))
    if not files:
        raise FileNotFoundError(f"no training summary in {run_dir}")
    return read_json(files[-1])


def avg_sd(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values)


def f(value: str | float | int | None) -> float:
    if value in ("", None):
        return math.nan
    return float(value)


def compact_seed_values(values: dict[int, str | float | int]) -> str:
    return ";".join(f"seed{seed}={values[seed]}" for seed in SEEDS)


def metric_columns(row: dict[str, str]) -> dict[str, float]:
    result = {
        "reconstruction_missing_nrmse": f(row["reconstruction_missing_nrmse"]),
    }
    for horizon in HORIZONS:
        result[f"forecast_{horizon}_nrmse"] = f(
            row[f"forecast_{horizon}_nrmse"]
        )
    return result


def standard_rows(
    source: Path, variant: str, method_id: str
) -> list[dict]:
    rows = [
        row
        for row in read_csv(source / "summary_by_seed.csv")
        if row["variant"] == variant
    ]
    if sorted(int(row["seed"]) for row in rows) != list(SEEDS):
        raise RuntimeError(f"{method_id}: expected seeds {SEEDS}")
    return [
        {
            "method_id": method_id,
            "seed": int(row["seed"]),
            **metric_columns(row),
            "_source": row,
        }
        for row in rows
    ]


def b5_rows(method: str, method_id: str) -> list[dict]:
    source = read_csv(B5 / "per_seed_metrics.csv")
    rows = []
    for seed in SEEDS:
        selected = {
            row["horizon"]: row
            for row in source
            if row["method"] == method and int(row["seed"]) == seed
        }
        item = {
            "method_id": method_id,
            "seed": seed,
            "reconstruction_missing_nrmse": f(
                selected["history_missing"]["nrmse"]
            ),
        }
        for horizon in HORIZONS:
            item[f"forecast_{horizon}_nrmse"] = f(
                selected[f"F{horizon}"]["nrmse"]
            )
        rows.append(item)
    return rows


METHOD_ROWS = {
    "partialconv_light_random": standard_rows(P1, "random_init", "partialconv_light_random"),
    "partialconv_light_pretrained": standard_rows(
        P1, "pretrained", "partialconv_light_pretrained"
    ),
    "mask_unet_random": standard_rows(B4, "random", "mask_unet_random"),
    "mask_unet_pretrained": standard_rows(
        B4, "pretrained", "mask_unet_pretrained"
    ),
    "mask_unet_partialconv_h1_a5": standard_rows(
        H1, "H1-A5", "mask_unet_partialconv_h1_a5"
    ),
    "mask_unet_partialconv_h1_a4": standard_rows(
        H1, "H1-A4", "mask_unet_partialconv_h1_a4"
    ),
    "spatial_vit_v0": standard_rows(VIT, "V0", "spatial_vit_v0"),
    "spatial_vit_v1": standard_rows(VIT, "V1", "spatial_vit_v1"),
    "explicit_gno_v2": b5_rows("gno", "explicit_gno_v2"),
    "latent_gino_v2": b5_rows("gino", "latent_gino_v2"),
}

DISPLAY = {
    "partialconv_light_random": "PartialConv轻量解码器 random",
    "partialconv_light_pretrained": "PartialConv轻量解码器 masked-pretrained",
    "mask_unet_random": "纯Mask U-Net random",
    "mask_unet_pretrained": "纯Mask U-Net masked-pretrained",
    "mask_unet_partialconv_h1_a5": "Mask U-Net+PartialConv H1-A5",
    "mask_unet_partialconv_h1_a4": "Mask U-Net+PartialConv H1-A4",
    "spatial_vit_v0": "Spatial ViT V0",
    "spatial_vit_v1": "Spatial ViT V1",
    "explicit_gno_v2": "Explicit GNO v2",
    "latent_gino_v2": "Latent GINO v2",
}


def per_mat_worst(
    source_dir: Path, variant_key: str, variant: str
) -> dict[str, tuple[str, float, float]]:
    rows = [
        row
        for row in read_csv(source_dir / "per_mat_metrics.csv")
        if row[variant_key] == variant
    ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["mat_name"]].append(row)
    result = {}
    for metric in ("reconstruction_missing_nrmse", "forecast_300_nrmse"):
        candidates = []
        for mat_name, mat_rows in grouped.items():
            values = [f(row[metric]) for row in mat_rows]
            if len(values) != 3:
                raise RuntimeError(
                    f"{source_dir.name}/{variant}/{mat_name}: not three seeds"
                )
            candidates.append((mat_name, mean(values), stdev(values)))
        result[metric] = max(candidates, key=lambda item: item[1])
    return result


WORST = {
    "partialconv_light_random": per_mat_worst(P1, "variant", "random_init"),
    "partialconv_light_pretrained": per_mat_worst(P1, "variant", "pretrained"),
    "mask_unet_random": per_mat_worst(B4, "variant", "random"),
    "mask_unet_pretrained": per_mat_worst(B4, "variant", "pretrained"),
    "mask_unet_partialconv_h1_a5": per_mat_worst(H1, "variant", "H1-A5"),
    "mask_unet_partialconv_h1_a4": per_mat_worst(H1, "variant", "H1-A4"),
    "spatial_vit_v0": per_mat_worst(VIT, "variant", "V0"),
    "spatial_vit_v1": per_mat_worst(VIT, "variant", "V1"),
}
for row in read_csv(B5 / "worst_per_mat.csv"):
    method_id = "explicit_gno_v2" if row["method"] == "gno" else "latent_gino_v2"
    metric = (
        "reconstruction_missing_nrmse"
        if row["metric"] == "history_missing_nrmse"
        else "forecast_300_nrmse"
    )
    WORST.setdefault(method_id, {})[metric] = (
        row["mat"],
        f(row["mean"]),
        f(row["sample_sd"]),
    )


def checkpoint_maps() -> dict[str, dict[int, tuple[str, str]]]:
    result: dict[str, dict[int, tuple[str, str]]] = defaultdict(dict)
    for row in read_csv(P1 / "provenance.csv"):
        method_id = (
            "partialconv_light_pretrained"
            if row["variant"] == "pretrained"
            else "partialconv_light_random"
        )
        result[method_id][int(row["seed"])] = (
            row["downstream_checkpoint"],
            row["downstream_checkpoint_sha256"],
        )
    for row in read_csv(B4 / "checkpoint_manifest.csv"):
        method_id = (
            "mask_unet_pretrained"
            if row["variant"] == "pretrained"
            else "mask_unet_random"
        )
        result[method_id][int(row["seed"])] = (
            row["best_checkpoint"],
            row["best_sha256"],
        )
    for row in read_csv(H1 / "checkpoint_manifest.csv"):
        method_id = (
            "mask_unet_partialconv_h1_a4"
            if row["variant"] == "H1-A4"
            else "mask_unet_partialconv_h1_a5"
        )
        result[method_id][int(row["seed"])] = (
            row["checkpoint"],
            row["checkpoint_sha256"],
        )
    for row in read_csv(VIT / "provenance.csv"):
        method_id = "spatial_vit_v1" if row["variant"] == "V1" else "spatial_vit_v0"
        result[method_id][int(row["seed"])] = (
            row["checkpoint"],
            row["checkpoint_sha256"],
        )
    for row in read_csv(B5 / "checkpoints.csv"):
        method_id = (
            "explicit_gno_v2" if row["method"] == "gno" else "latent_gino_v2"
        )
        result[method_id][int(row["seed"])] = (row["path"], row["sha256"])
    return result


CHECKPOINTS = checkpoint_maps()


def p1_pretraining() -> dict[int, dict]:
    dirs = {
        42: ROOT
        / "runs/sparse/p1_partialconv_mae_frozen_fixed_points_r05_mask_00006_seed42_formal_pretrain_20260721_v1",
        43: ROOT
        / "runs/sparse/p1_partialconv_mae_frozen_fixed_points_r05_mask_00006_seed43_formal_pretrain_completion_20260722_v1",
        44: ROOT
        / "runs/sparse/p1_partialconv_mae_frozen_fixed_points_r05_mask_00006_seed44_formal_pretrain_completion_20260722_v1",
    }
    return {seed: latest_summary(path) for seed, path in dirs.items()}


def h1_pretraining() -> dict[int, dict]:
    dirs = {
        42: ROOT
        / "runs/sparse/h1_a1_partialconv_unet_frozen_fixed_points_r05_mask_00006_seed42",
        43: ROOT
        / "runs/sparse/h1_a4_pretrain_partialconv3_unet_skip_fixed_points_r05_mask_00006_seed43",
        44: ROOT
        / "runs/sparse/h1_a4_pretrain_partialconv3_unet_skip_fixed_points_r05_mask_00006_seed44",
    }
    return {seed: latest_summary(path) for seed, path in dirs.items()}


def vit_training(method_id: str) -> tuple[dict[int, dict], dict[int, dict] | None]:
    prefix = (
        "v1_vit_mae_pretrained_sparse_finetune"
        if method_id == "spatial_vit_v1"
        else "v0_vit_random_sparse_finetune"
    )
    downstream = {
        seed: latest_summary(
            ROOT
            / f"runs/sparse/{prefix}_seed{seed}_30ep_formal_20260729_v1"
        )
        for seed in SEEDS
    }
    pretraining = None
    if method_id == "spatial_vit_v1":
        pretraining = {
            seed: latest_summary(
                ROOT
                / f"runs/sparse/v1_spatial_mae_pretrain_patch75_seed{seed}_formal_20260728_v1"
            )
            for seed in SEEDS
        }
    return downstream, pretraining


def parameter_from_val_json(eval_dir: str) -> tuple[int, int]:
    payload = read_json(Path(eval_dir) / "evaluation/val_metrics.json")
    return int(payload["parameters"]["total"]), int(payload["parameters"]["trainable"])


def resources() -> dict[str, dict]:
    result: dict[str, dict] = {}
    p1_pre = p1_pretraining()
    for method_id, variant in (
        ("partialconv_light_random", "random_init"),
        ("partialconv_light_pretrained", "pretrained"),
    ):
        source = [
            row
            for row in read_csv(P1 / "summary_by_seed.csv")
            if row["variant"] == variant
        ]
        provenance = {
            int(row["seed"]): row
            for row in read_csv(P1 / "provenance.csv")
            if row["variant"] == variant
        }
        total, trainable = parameter_from_val_json(provenance[42]["eval_dir"])
        pre = [
            f(p1_pre[seed]["elapsed_seconds"])
            if method_id.endswith("pretrained")
            else 0.0
            for seed in SEEDS
        ]
        downstream = [f(row["training_seconds"]) for row in source]
        pre_peak = [
            f(p1_pre[seed]["peak_memory_allocated_mb"])
            if method_id.endswith("pretrained")
            else 0.0
            for seed in SEEDS
        ]
        down_peak = [f(row["peak_memory_mb"]) for row in source]
        result[method_id] = {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "pretraining_seconds": pre,
            "downstream_seconds": downstream,
            "cumulative_seconds": [a + b for a, b in zip(pre, downstream)],
            "peak_memory_mb": [max(a, b) for a, b in zip(pre_peak, down_peak)],
            "reconstruction_inference_seconds": None,
            "val300_inference_seconds": [f(row["inference_seconds"]) for row in source],
        }

    for method_id, variant in (
        ("mask_unet_random", "random"),
        ("mask_unet_pretrained", "pretrained"),
    ):
        source = [
            row
            for row in read_csv(B4 / "summary_by_seed.csv")
            if row["variant"] == variant
        ]
        result[method_id] = {
            "total_parameters": int(source[0]["total_parameters"]),
            "trainable_parameters": int(source[0]["trainable_parameters"]),
            "pretraining_seconds": [f(row["pretraining_seconds"]) for row in source],
            "downstream_seconds": [f(row["downstream_seconds"]) for row in source],
            "cumulative_seconds": [
                f(row["cumulative_training_seconds"]) for row in source
            ],
            "peak_memory_mb": [
                f(row["cumulative_peak_memory_mb"]) for row in source
            ],
            "reconstruction_inference_seconds": None,
            "val300_inference_seconds": [f(row["inference_seconds"]) for row in source],
        }

    h1_pre = h1_pretraining()
    for method_id, variant in (
        ("mask_unet_partialconv_h1_a5", "H1-A5"),
        ("mask_unet_partialconv_h1_a4", "H1-A4"),
    ):
        source = [
            row
            for row in read_csv(H1 / "summary_by_seed.csv")
            if row["variant"] == variant
        ]
        pre = [
            f(h1_pre[seed]["elapsed_seconds"]) if variant == "H1-A4" else 0.0
            for seed in SEEDS
        ]
        downstream = [f(row["training_seconds"]) for row in source]
        pre_peak = [
            f(h1_pre[seed]["peak_memory_allocated_mb"])
            if variant == "H1-A4"
            else 0.0
            for seed in SEEDS
        ]
        down_peak = [f(row["peak_memory_allocated_mb"]) for row in source]
        result[method_id] = {
            "total_parameters": int(source[0]["total_parameters"]),
            "trainable_parameters": int(source[0]["trainable_parameters"]),
            "pretraining_seconds": pre,
            "downstream_seconds": downstream,
            "cumulative_seconds": [a + b for a, b in zip(pre, downstream)],
            "peak_memory_mb": [max(a, b) for a, b in zip(pre_peak, down_peak)],
            "reconstruction_inference_seconds": None,
            "val300_inference_seconds": [f(row["inference_seconds"]) for row in source],
        }

    vit_source = read_csv(VIT / "summary_by_seed.csv")
    for method_id, variant in (("spatial_vit_v0", "V0"), ("spatial_vit_v1", "V1")):
        downstream_meta, pre_meta = vit_training(method_id)
        source = [row for row in vit_source if row["variant"] == variant]
        provenance = {
            int(row["seed"]): row
            for row in read_csv(VIT / "provenance.csv")
            if row["variant"] == variant
        }
        total, trainable = parameter_from_val_json(provenance[42]["eval_dir"])
        pre = [
            f(pre_meta[seed]["elapsed_seconds"]) if pre_meta else 0.0
            for seed in SEEDS
        ]
        downstream = [
            f(downstream_meta[seed]["elapsed_seconds"]) for seed in SEEDS
        ]
        pre_peak = [
            f(pre_meta[seed]["peak_memory_allocated_mb"]) if pre_meta else 0.0
            for seed in SEEDS
        ]
        down_peak = [
            f(downstream_meta[seed]["peak_memory_allocated_mb"]) for seed in SEEDS
        ]
        recon_seconds = []
        prefix = (
            "v1_vit_mae_pretrained_sparse_finetune"
            if variant == "V1"
            else "v0_vit_random_sparse_finetune"
        )
        for seed in SEEDS:
            evaluation = read_json(
                ROOT
                / f"runs/sparse/{prefix}_seed{seed}_30ep_formal_20260729_v1/evaluation/reconstruction_val_metrics.json"
            )
            recon_seconds.append(f(evaluation["summary"]["inference_seconds"]))
        result[method_id] = {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "pretraining_seconds": pre,
            "downstream_seconds": downstream,
            "cumulative_seconds": [a + b for a, b in zip(pre, downstream)],
            "peak_memory_mb": [max(a, b) for a, b in zip(pre_peak, down_peak)],
            "reconstruction_inference_seconds": recon_seconds,
            "val300_inference_seconds": [f(row["inference_seconds"]) for row in source],
        }

    b5_resources = read_csv(B5 / "resources.csv")
    for method_id, method in (("explicit_gno_v2", "gno"), ("latent_gino_v2", "gino")):
        source = [row for row in b5_resources if row["method"] == method]
        downstream = [f(row["training_seconds"]) for row in source]
        result[method_id] = {
            "total_parameters": int(source[0]["total_parameters"]),
            "trainable_parameters": int(source[0]["trainable_parameters"]),
            "pretraining_seconds": [0.0, 0.0, 0.0],
            "downstream_seconds": downstream,
            "cumulative_seconds": downstream,
            "peak_memory_mb": [f(row["peak_memory_mb"]) for row in source],
            "reconstruction_inference_seconds": None,
            "val300_inference_seconds": [f(row["val300_seconds"]) for row in source],
        }
    return result


RESOURCES = resources()


def aggregate_scorecard() -> list[dict]:
    rows = []
    metric_names = ["reconstruction_missing_nrmse"] + [
        f"forecast_{horizon}_nrmse" for horizon in HORIZONS
    ]
    for method_id, seed_rows in METHOD_ROWS.items():
        result = {
            "row_role": "sparse_method",
            "method_id": method_id,
            "method": DISPLAY[method_id],
            "n_seeds": 3,
            "seeds": "42;43;44",
        }
        for metric in metric_names:
            m, sd = avg_sd([row[metric] for row in seed_rows])
            result[f"{metric}_mean"] = m
            result[f"{metric}_sample_sd"] = sd
        growth = [
            row["forecast_300_nrmse"] - row["forecast_30_nrmse"]
            for row in seed_rows
        ]
        result["error_growth_F300_minus_F30_mean"] = mean(growth)
        result["error_growth_F300_minus_F30_sample_sd"] = stdev(growth)
        for metric, prefix in (
            ("reconstruction_missing_nrmse", "worst_per_mat_reconstruction"),
            ("forecast_300_nrmse", "worst_per_mat_F300"),
        ):
            mat, value, sd = WORST[method_id][metric]
            result[f"{prefix}_mat"] = mat
            result[f"{prefix}_nrmse_mean"] = value
            result[f"{prefix}_nrmse_sample_sd"] = sd
        resource = RESOURCES[method_id]
        result["total_parameters"] = resource["total_parameters"]
        result["trainable_parameters"] = resource["trainable_parameters"]
        for key in (
            "pretraining_seconds",
            "downstream_seconds",
            "cumulative_seconds",
            "peak_memory_mb",
            "val300_inference_seconds",
        ):
            result[f"{key}_mean"], result[f"{key}_sample_sd"] = avg_sd(
                resource[key]
            )
        recon = resource["reconstruction_inference_seconds"]
        if recon is None:
            result["reconstruction_inference_seconds_mean"] = ""
            result["reconstruction_inference_seconds_sample_sd"] = ""
            result["reconstruction_inference_status"] = "not_archived"
        else:
            (
                result["reconstruction_inference_seconds_mean"],
                result["reconstruction_inference_seconds_sample_sd"],
            ) = avg_sd(recon)
            result["reconstruction_inference_status"] = "archived"
        result["best_checkpoint_paths"] = compact_seed_values(
            {seed: CHECKPOINTS[method_id][seed][0] for seed in SEEDS}
        )
        result["best_checkpoint_sha256"] = compact_seed_values(
            {seed: CHECKPOINTS[method_id][seed][1] for seed in SEEDS}
        )
        result["rank_eligible"] = True
        result["notes"] = (
            "100-epoch best-recipe solution score; not a uniform-budget architecture effect"
            if method_id in {"explicit_gno_v2", "latent_gino_v2"}
            else ""
        )
        rows.append(result)
    blank = {key: "" for key in rows[0]}
    blank.update(
        {
            "row_role": "complete_field_reference",
            "method_id": "b0_full_field_reference",
            "method": "B0完整场上参考",
            "n_seeds": 0,
            "rank_eligible": False,
            "notes": (
                "No same-protocol archived B0 val-300 artifact was found; "
                "reference retained without fabricated metrics and excluded from sparse ranks."
            ),
        }
    )
    rows.append(blank)
    return rows


SCORECARD = aggregate_scorecard()


def rank_rows(metric: str) -> list[dict]:
    eligible = [row for row in SCORECARD if row["rank_eligible"] is True]
    eligible.sort(key=lambda row: float(row[f"{metric}_mean"]))
    rows = []
    for rank, row in enumerate(eligible, start=1):
        item = {
            "rank": rank,
            "method_id": row["method_id"],
            "method": row["method"],
            "n_seeds": 3,
            f"{metric}_mean": row[f"{metric}_mean"],
            f"{metric}_sample_sd": row[f"{metric}_sample_sd"],
        }
        if metric == "forecast_300_nrmse":
            for horizon in (30, 60, 120, 180, 240):
                item[f"forecast_{horizon}_nrmse_mean"] = row[
                    f"forecast_{horizon}_nrmse_mean"
                ]
                item[f"forecast_{horizon}_nrmse_sample_sd"] = row[
                    f"forecast_{horizon}_nrmse_sample_sd"
                ]
            item["error_growth_F300_minus_F30_mean"] = row[
                "error_growth_F300_minus_F30_mean"
            ]
            item["worst_per_mat_F300_mat"] = row["worst_per_mat_F300_mat"]
            item["worst_per_mat_F300_nrmse_mean"] = row[
                "worst_per_mat_F300_nrmse_mean"
            ]
        else:
            item["worst_per_mat_reconstruction_mat"] = row[
                "worst_per_mat_reconstruction_mat"
            ]
            item["worst_per_mat_reconstruction_nrmse_mean"] = row[
                "worst_per_mat_reconstruction_nrmse_mean"
            ]
        rows.append(item)
    return rows


PAIRS = (
    (
        "PartialConv pretrained-random",
        "partialconv_light_pretrained",
        "partialconv_light_random",
    ),
    (
        "Mask U-Net pretrained-random",
        "mask_unet_pretrained",
        "mask_unet_random",
    ),
    (
        "H1-A4-H1-A5",
        "mask_unet_partialconv_h1_a4",
        "mask_unet_partialconv_h1_a5",
    ),
    ("V1-V0", "spatial_vit_v1", "spatial_vit_v0"),
    ("GINO-GNO", "latent_gino_v2", "explicit_gno_v2"),
)


def paired_effects() -> list[dict]:
    rows = []
    metrics = ["reconstruction_missing_nrmse"] + [
        f"forecast_{horizon}_nrmse" for horizon in HORIZONS
    ]
    for contrast, first, second in PAIRS:
        first_by_seed = {row["seed"]: row for row in METHOD_ROWS[first]}
        second_by_seed = {row["seed"]: row for row in METHOD_ROWS[second]}
        for metric in metrics:
            differences = {
                seed: first_by_seed[seed][metric] - second_by_seed[seed][metric]
                for seed in SEEDS
            }
            values = list(differences.values())
            m = mean(values)
            sd = stdev(values)
            margin = T_CRIT_DF2 * sd / math.sqrt(3)
            low, high = m - margin, m + margin
            rows.append(
                {
                    "contrast": contrast,
                    "direction": f"{DISPLAY[first]} minus {DISPLAY[second]}",
                    "metric": metric,
                    "seed42_difference": differences[42],
                    "seed43_difference": differences[43],
                    "seed44_difference": differences[44],
                    "mean_difference": m,
                    "sample_sd": sd,
                    "student_t_df": 2,
                    "ci95_low": low,
                    "ci95_high": high,
                    "significant_95": low > 0 or high < 0,
                    "interpretation": (
                        "first lower error"
                        if high < 0
                        else "first higher error"
                        if low > 0
                        else "not significant"
                    ),
                }
            )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    recon_rank = rank_rows("reconstruction_missing_nrmse")
    forecast_rank = rank_rows("forecast_300_nrmse")
    effects = paired_effects()
    write_csv(OUT / "learned_reconstruction_val_scorecard.csv", SCORECARD)
    write_csv(OUT / "paired_factor_effects.csv", effects)
    write_csv(OUT / "reconstruction_rank.csv", recon_rank)
    write_csv(OUT / "forecast_F300_rank.csv", forecast_rank)

    significant_primary = [
        row
        for row in effects
        if row["metric"]
        in {"reconstruction_missing_nrmse", "forecast_300_nrmse"}
        and row["significant_95"]
    ]
    manifest = {
        "experiment_id": "MIN-learned-reconstruction-frozen-scorecard",
        "category": "evaluation_only",
        "split": "val",
        "test_accessed": False,
        "test_metrics_read": False,
        "test_loader_constructed": False,
        "composite_score_constructed": False,
        "seeds": list(SEEDS),
        "scope": "fixed-point 5%, mask_00006",
        "b0_status": (
            "reference-only; no same-protocol archived B0 val-300 artifact; "
            "excluded from sparse rankings"
        ),
        "gno_gino_budget_note": (
            "GNO/GINO use their archived 100-epoch best recipes; their rows are "
            "solution scores, not uniform-budget pure architecture effects."
        ),
        "error_growth_definition": "forecast_300_nrmse - forecast_30_nrmse",
        "paired_ci": "two-sided Student-t 95%, n=3, df=2",
        "significant_primary_effects": significant_primary,
        "artifacts": [
            "learned_reconstruction_val_scorecard.csv",
            "paired_factor_effects.csv",
            "reconstruction_rank.csv",
            "forecast_F300_rank.csv",
            "conclusions.md",
        ],
    }
    with (OUT / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    conclusions = """# MIN learned-reconstruction frozen validation conclusions

## Supported

- Under the current validation split, fixed-point 5% observations and mask_00006,
  the pure Mask U-Net with masked pretraining ranks first for both missing-region
  reconstruction NRMSE and frozen-RFNO F300 NRMSE.
- PartialConv-light and Spatial ViT each show a statistically significant benefit
  from masked/MAE pretraining for both reconstruction and F300 NRMSE.
- Pure Mask U-Net masked pretraining has a statistically significant F300 benefit,
  while its reconstruction-NRMSE confidence interval crosses zero.

## Not supported

- H1-A4 versus H1-A5 does not establish a significant pretraining benefit with
  n=3 because both primary confidence intervals cross zero.
- GINO versus GNO does not establish a significant difference on either primary
  endpoint; GNO/GINO also cannot be used here as a uniform-budget architecture
  effect because they use 100-epoch best recipes.
- These validation results do not support claims about unseen masks, sensor
  densities, random layouts, other splits, or test-set generalization.
- B0 remains a conceptual complete-field reference because no same-protocol
  archived B0 val-300 metric artifact was found; no value was inferred.
"""
    (OUT / "conclusions.md").write_text(conclusions, encoding="utf-8")


if __name__ == "__main__":
    main()
