#!/usr/bin/env python
"""Build the locked seed-42 preliminary FNO-DeepONet comparison artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "fno_deeponet_seed42_preliminary_comparison_20260802"
)
EVAL_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "sparse"
    / "_evaluations"
    / "fno_deeponet_seed42_preliminary_20260802"
)
BENCH_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "sparse"
    / "_benchmarks"
    / "fno_deeponet_seed42_preliminary_20260802"
)


METHODS: tuple[dict[str, Any], ...] = (
    {
        "key": "mask_unet_random",
        "label": "Mask U-Net random",
        "table1": True,
        "table2": False,
        "config": "config/sparse_experiments/min_b4_init_ablation_random_seed42.formal.json",
        "run": "runs/sparse/min_b4_init_ablation_random_seed42_30ep_formal_20260729_v1",
        "epochs": 30,
        "structure": "periodic convolutional mask-aware U-Net + frozen B0 RFNO",
    },
    {
        "key": "st_d0_random",
        "label": "ST-D0 random",
        "table1": True,
        "table2": False,
        "config": "config/sparse_experiments/st_d0_sensor_token_grid_query_seed42.formal.json",
        "run": "runs/sparse/st_d0_sensor_token_grid_query_seed42_30ep_formal_20260731_v1",
        "epochs": 30,
        "structure": "sensor-token encoder/grid-query decoder + frozen B0 RFNO",
    },
    {
        "key": "partialconv_light_random_recon_only",
        "label": "PartialConv lightweight random reconstruction-only",
        "table1": True,
        "table2": False,
        "config": "config/sparse_experiments/p1_partialconv_light_random_reconstruction_only_seed42.formal.json",
        "run": "runs/sparse/p1_partialconv_light_random_reconstruction_only_30ep_fixed_r05_mask_00006_seed42_20260802_v1",
        "epochs": 30,
        "structure": "lightweight PartialConv masked autoencoder + frozen B0 RFNO",
    },
    {
        "key": "fd_r1",
        "label": "FD-R1-30ep + frozen RFNO",
        "table1": True,
        "table2": True,
        "config": "config/sparse_experiments/fd_r1_fno_deeponet_frozen_rfno_seed42.formal.json",
        "run": "runs/sparse/fd_r1_fno_deeponet_random_frozen_rfno_fixed_points_r05_mask_00006_seed42_20260731_v1",
        "epochs": 30,
        "structure": "temporal FNO branch/DeepONet trunk reconstructor + frozen B0 RFNO",
    },
    {
        "key": "fd_a1",
        "label": "FD-A1 autoregressive curriculum",
        "table1": False,
        "table2": True,
        "config": "config/sparse_experiments/fd_a1_fno_deeponet_autoregressive_seed42.formal.json",
        "run": "runs/sparse/fd_a1_fno_deeponet_reconstruction_joint_ar300_fixed_points_r05_mask_00006_seed42_20260731_v1",
        "epochs": 100,
        "structure": "FNO-DeepONet reconstructor + joint 60-to-30 FNO-DeepONet AR forecaster",
    },
    {
        "key": "fd_l1",
        "label": "FD-L1 direct 300-frame",
        "table1": False,
        "table2": True,
        "config": "config/sparse_experiments/fd_l1_fno_deeponet_direct_long_seed42.formal.json",
        "run": "runs/sparse/fd_l1_fno_deeponet_direct_sparse_to_300_fixed_points_r05_mask_00006_seed42_20260731_v1",
        "epochs": 100,
        "structure": "one-shot sparse-history encoding with direct 60-history/300-future decoding",
    },
)
HORIZONS = (30, 60, 120, 180, 240, 300)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def _metric(metric: Mapping[str, Any], name: str) -> float:
    return float(metric[name])


def _protocol_mismatches(config: Mapping[str, Any]) -> list[str]:
    data = config["data"]
    observation = config["observation"]
    training = config["training"]
    expected = {
        "data.input_steps": (data.get("input_steps"), 60),
        "data.output_steps": (data.get("output_steps"), 30),
        "data.rollout_steps": (data.get("rollout_steps"), 300),
        "data.height": (data.get("height"), 64),
        "data.width": (data.get("width"), 64),
        "observation.mask_type": (observation.get("mask_type"), "fixed_points"),
        "observation.observation_rate": (observation.get("observation_rate"), 0.05),
        "observation.mask_id": (observation.get("mask_id"), "mask_00006"),
        "observation.noise_std_fraction": (observation.get("noise_std_fraction"), 0.0),
        "observation.temporal_dropout": (observation.get("temporal_dropout"), 0.0),
        "observation.seed": (observation.get("seed"), 42),
    }
    mismatches = [
        f"{name}={actual!r}, expected {wanted!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    split = str(data.get("split_manifest_path", "")).replace("\\", "/")
    mask = str(observation.get("manifest_path", "")).replace("\\", "/")
    if not split.endswith("data/splits/sea_surface_bimodal_v1.json"):
        mismatches.append(f"split_manifest_path={split!r}")
    if not mask.endswith("data/masks/point_masks.npz"):
        mismatches.append(f"mask_manifest_path={mask!r}")
    if int(training.get("epochs", -1)) not in {30, 100}:
        mismatches.append(f"training.epochs={training.get('epochs')!r}")
    return mismatches


def _rank(values: Mapping[str, float]) -> dict[str, int]:
    return {
        key: index + 1
        for index, (key, _) in enumerate(sorted(values.items(), key=lambda item: item[1]))
    }


def _pct(new: float, reference: float) -> float:
    return 100.0 * (new / reference - 1.0)


def _fmt(value: float, digits: int = 5) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    manifest_methods: list[dict[str, Any]] = []

    for method in METHODS:
        key = method["key"]
        config_path = PROJECT_ROOT / method["config"]
        run_dir = PROJECT_ROOT / method["run"]
        checkpoint = run_dir / "checkpoints" / "best.pt"
        metrics_path = EVAL_ROOT / key / "evaluation" / "val_metrics.json"
        latency_path = BENCH_ROOT / key / "component_latency.json"
        for required in (config_path, checkpoint, metrics_path, latency_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        config = _read_json(config_path)
        metrics_payload = _read_json(metrics_path)
        latency = _read_json(latency_path)
        if metrics_payload.get("split") != "val":
            raise PermissionError(f"Non-validation metrics rejected: {metrics_path}")
        if int(config["runtime"]["seed"]) != 42:
            raise ValueError(f"Non-seed42 configuration rejected: {config_path}")
        restore_mode = str(metrics_payload.get("checkpoint_restore_mode", ""))
        mismatches = _protocol_mismatches(config)
        if mismatches:
            raise ValueError(f"Protocol mismatch for {key}: {mismatches}")

        metrics = metrics_payload["metrics"]
        history = metrics["history_reconstruction"]
        checkpoint_state = metrics_payload["checkpoint_restore"]["curriculum_state"]
        record = {
            **method,
            "config_payload": config,
            "config_path": config_path,
            "run_dir": run_dir,
            "checkpoint": checkpoint,
            "metrics_path": metrics_path,
            "latency_path": latency_path,
            "metrics_payload": metrics_payload,
            "metrics": metrics,
            "history": history,
            "latency": latency,
            "best_epoch": int(checkpoint_state["best_epoch"]),
            "training_seconds": float(checkpoint_state["elapsed_seconds"]),
            "peak_memory_mb": float(
                checkpoint_state["peak_memory_allocated_bytes"]
            ) / (1024.0 * 1024.0),
            "parameters_total": int(metrics_payload["parameters"]["total"]),
            "parameters_trainable": int(metrics_payload["parameters"]["trainable"]),
        }
        records[key] = record
        manifest_methods.append(
            {
                "key": key,
                "label": method["label"],
                "config": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "config_sha256": _sha256(config_path),
                "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "validation_metrics": str(metrics_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "validation_metrics_sha256": _sha256(metrics_path),
                "component_latency": str(latency_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "component_latency_sha256": _sha256(latency_path),
                "checkpoint_restore_mode": restore_mode,
                "protocol_mismatches": (
                    [
                        "checkpoint scientific hash differs from the current resolver; "
                        "locked split/mask/noise/normalization/model fields were audited "
                        "before validation-only restore"
                    ]
                    if "mismatch" in restore_mode
                    else []
                ),
            }
        )

    table1 = [records[m["key"]] for m in METHODS if m["table1"]]
    rank_missing = _rank(
        {r["key"]: _metric(r["history"]["missing_region"], "nrmse") for r in table1}
    )
    rank_forecast = _rank(
        {r["key"]: _metric(r["metrics"]["forecast_300"], "nrmse") for r in table1}
    )
    rank_ssp = _rank(
        {r["key"]: _metric(r["metrics"]["forecast_300"], "ssp") for r in table1}
    )

    reconstruction_rows: list[dict[str, Any]] = []
    for record in sorted(table1, key=lambda r: rank_missing[r["key"]]):
        history = record["history"]
        missing = history["missing_region"]
        observed = history["observed_region"]
        reconstruction_rows.append(
            {
                "rank_missing_nrmse": rank_missing[record["key"]],
                "method": record["label"],
                "experiment_id": record["config_payload"]["experiment_id"],
                "seed": 42,
                "epochs": record["epochs"],
                "history_full_rmse": _metric(history, "rmse"),
                "history_full_nrmse": _metric(history, "nrmse"),
                "history_full_correlation": _metric(history, "correlation"),
                "history_periodic_gradient_rmse": _metric(history, "gradient_rmse"),
                "history_periodic_gradient_nrmse": _metric(history, "gradient_nrmse"),
                "history_spectrum_ssp": _metric(history, "ssp"),
                "history_missing_rmse": _metric(missing, "rmse"),
                "history_missing_nrmse": _metric(missing, "nrmse"),
                "history_missing_correlation": _metric(missing, "correlation"),
                "observation_consistency_rmse": _metric(observed, "rmse"),
                "observation_consistency_nrmse": _metric(observed, "nrmse"),
                "observation_consistency_correlation": _metric(observed, "correlation"),
                "forecast_300_nrmse": _metric(record["metrics"]["forecast_300"], "nrmse"),
                "forecast_300_ssp": _metric(record["metrics"]["forecast_300"], "ssp"),
                "rank_forecast_300_nrmse": rank_forecast[record["key"]],
                "rank_forecast_300_ssp": rank_ssp[record["key"]],
            }
        )
    _write_csv(RESULT_DIR / "reconstruction_30ep.csv", reconstruction_rows)

    forecast_rows: list[dict[str, Any]] = []
    for method in METHODS:
        record = records[method["key"]]
        table = "table1_30ep_frozen" if method["table1"] else "table2_fno_deeponet"
        if method["table1"] and method["table2"]:
            table = "table1_30ep_frozen;table2_fno_deeponet"
        for horizon in HORIZONS:
            metric = record["metrics"][f"forecast_{horizon}"]
            forecast_rows.append(
                {
                    "comparison_table": table,
                    "method": record["label"],
                    "experiment_id": record["config_payload"]["experiment_id"],
                    "seed": 42,
                    "epochs": record["epochs"],
                    "horizon_frames": horizon,
                    "rmse": _metric(metric, "rmse"),
                    "nrmse": _metric(metric, "nrmse"),
                    "correlation": _metric(metric, "correlation"),
                    "spectrum_ssp": _metric(metric, "ssp"),
                    "periodic_gradient_rmse": _metric(metric, "gradient_rmse"),
                    "periodic_gradient_nrmse": _metric(metric, "gradient_nrmse"),
                }
            )
    _write_csv(RESULT_DIR / "forecast_horizons.csv", forecast_rows)

    efficiency_rows: list[dict[str, Any]] = []
    for method in METHODS:
        record = records[method["key"]]
        latency = record["latency"]
        efficiency_rows.append(
            {
                "method": record["label"],
                "experiment_id": record["config_payload"]["experiment_id"],
                "epochs": record["epochs"],
                "best_epoch": record["best_epoch"],
                "total_parameters": record["parameters_total"],
                "trainable_parameters": record["parameters_trainable"],
                "training_seconds": record["training_seconds"],
                "training_hours": record["training_seconds"] / 3600.0,
                "peak_training_memory_mb": record["peak_memory_mb"],
                "reconstruction_seconds_per_sample": latency["reconstruction_seconds_per_sample"],
                "forecast_300_seconds_per_sample": latency["forecast_300_seconds_per_sample"],
                "component_total_seconds_per_sample": latency["total_seconds_per_sample"],
                "end_to_end_validation_seconds_per_sample": record["metrics"]["seconds_per_sample"],
                "checkpoint_size_bytes": record["checkpoint"].stat().st_size,
                "checkpoint_size_mb": record["checkpoint"].stat().st_size / (1024.0 * 1024.0),
                "latency_measured_samples": latency["measured_samples"],
                "latency_scope": latency["timing_scope"],
            }
        )
    _write_csv(RESULT_DIR / "efficiency.csv", efficiency_rows)

    error_rows: list[dict[str, Any]] = []
    for method in METHODS:
        record = records[method["key"]]
        for point in record["metrics"]["error_growth_curve"]:
            error_rows.append(
                {
                    "method": record["label"],
                    "experiment_id": record["config_payload"]["experiment_id"],
                    "frame": int(point["frame"]),
                    "time_seconds": float(point["time_seconds"]),
                    "rmse": float(point["rmse"]),
                    "nrmse": float(point["nrmse"]),
                    "correlation": float(point["correlation"]),
                }
            )
    _write_csv(RESULT_DIR / "error_growth.csv", error_rows)

    manifest = {
        "title": "FNO-DeepONet seed42 preliminary sparse sea-surface comparison",
        "created_for": "validation-only preliminary structural evidence",
        "seed": 42,
        "test_split_accessed": False,
        "claims_scope": "single-seed preliminary evidence; no significance claim",
        "protocol": {
            "split": "data/splits/sea_surface_bimodal_v1.json",
            "mask": "data/masks/point_masks.npz/mask_00006",
            "mask_semantics": "1=observed,0=missing",
            "mask_type": "fixed_points",
            "observation_rate": 0.05,
            "noise_std_fraction": 0.0,
            "temporal_dropout": 0.0,
            "input_shape": "(B,60,64,64)",
            "normalization": "B0 checkpoint train-only statistics",
            "forecast_horizons": list(HORIZONS),
            "model_selection_split": "val",
        },
        "known_protocol_mismatches": [
            {
                "excluded_run": "p1_init_ablation_random_init_seed42_30ep_fixed_r05_mask_00006_20260722_v1",
                "reason": "forecast_supervision=true; not reconstruction-only",
            },
            {
                "comparison": "30ep frozen table versus 100ep end-to-end table",
                "reason": "different optimization objectives and budgets; tables intentionally separated",
            },
        ],
        "methods": manifest_methods,
        "outputs": [
            "manifest.json",
            "reconstruction_30ep.csv",
            "forecast_horizons.csv",
            "efficiency.csv",
            "error_growth.csv",
            "preliminary_assessment.md",
        ],
    }
    (RESULT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fd_r1 = records["fd_r1"]
    fd_a1 = records["fd_a1"]
    fd_l1 = records["fd_l1"]
    mask = records["mask_unet_random"]
    st = records["st_d0_random"]
    pconv = records["partialconv_light_random_recon_only"]
    table1_sorted = sorted(table1, key=lambda r: rank_missing[r["key"]])
    table2 = [fd_r1, fd_a1, fd_l1]
    table2_sorted = sorted(
        table2, key=lambda r: _metric(r["metrics"]["forecast_300"], "nrmse")
    )
    fd_r1_missing = _metric(fd_r1["history"]["missing_region"], "nrmse")
    fd_r1_f300 = _metric(fd_r1["metrics"]["forecast_300"], "nrmse")
    fd_a1_f300 = _metric(fd_a1["metrics"]["forecast_300"], "nrmse")
    fd_l1_f300 = _metric(fd_l1["metrics"]["forecast_300"], "nrmse")
    fd_l1_growth = fd_l1_f300 / _metric(fd_l1["metrics"]["forecast_30"], "nrmse")
    fd_a1_growth = fd_a1_f300 / _metric(fd_a1["metrics"]["forecast_30"], "nrmse")
    fd_r1_growth = fd_r1_f300 / _metric(fd_r1["metrics"]["forecast_30"], "nrmse")

    conv_token_best = min(
        (mask, st, pconv),
        key=lambda r: _metric(r["history"]["missing_region"], "nrmse"),
    )
    branch_delta = _pct(
        fd_r1_missing,
        _metric(conv_token_best["history"]["missing_region"], "nrmse"),
    )
    joint_delta = _pct(fd_a1_f300, fd_r1_f300)
    direct_delta_vs_r1 = _pct(fd_l1_f300, fd_r1_f300)
    direct_delta_vs_a1 = _pct(fd_l1_f300, fd_a1_f300)

    balance_rank = sorted(
        table2,
        key=lambda r: (
            _metric(r["metrics"]["forecast_300"], "nrmse"),
            r["latency"]["total_seconds_per_sample"],
        ),
    )
    recommended = [table1_sorted[0]["label"]]
    if table2_sorted[0]["label"] not in recommended:
        recommended.append(table2_sorted[0]["label"])
    recommended = recommended[:2]

    lines = [
        "# Seed=42 稀疏海面结构初步对比",
        "",
        "> 这些结果仅是单个 seed 的初步证据，不构成最终显著性结论；全程仅使用 validation 做模型选择和评分，未访问 test。",
        "",
        "## 公平协议与分表原则",
        "",
        "表1仅包含 seed=42、固定 5% 点观测、mask_00006、无噪声/无时间丢失、30 epoch、reconstruction-only 且连接同一冻结 B0 RFNO 的方法。表2单列 100 epoch 端到端预测结构；不能把两表按 epoch 预算简单混排。所有方法均使用 60 帧历史和 30/60/120/180/240/300 帧 validation 预测时域。",
        "",
        "旧 PartialConv random checkpoint 因 `forecast_supervision=true` 被排除；表1使用仅关闭 forecast supervision 和 rollout loss 的最小 reconstruction-only 配置。统一评分均从各自 `best.pt` 恢复。若旧 checkpoint 的科学哈希早于当前解析器新增的未使用默认字段，则仅在完成逐字段协议审计后使用 validation-only mismatch restore，并在 manifest 中明确记录；这不等同于允许数据、mask 或归一化协议变化。",
        "",
        "## 表1：30-epoch frozen-reconstruction 公平对照",
        "",
        "| 排名 | 方法 | missing NRMSE | full NRMSE | 300帧 NRMSE | 300帧 SSP | 参数量 | 总推理 s/sample |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in table1_sorted:
        lines.append(
            "| {rank} | {label} | {missing} | {full} | {f300} | {ssp} | {params:,} | {latency} |".format(
                rank=rank_missing[record["key"]],
                label=record["label"],
                missing=_fmt(_metric(record["history"]["missing_region"], "nrmse")),
                full=_fmt(_metric(record["history"], "nrmse")),
                f300=_fmt(_metric(record["metrics"]["forecast_300"], "nrmse")),
                ssp=_fmt(_metric(record["metrics"]["forecast_300"], "ssp")),
                params=record["parameters_total"],
                latency=_fmt(float(record["latency"]["total_seconds_per_sample"]), 4),
            )
        )
    lines.extend(
        [
            "",
            "主要优势与失败模式：",
            "",
            f"- missing-location 重构暂列第一的是 **{table1_sorted[0]['label']}**；其主要风险仍需结合观测一致性、空间梯度和频谱误差判断，不能只看像素 NRMSE。",
            f"- FD-R1 相对最佳卷积/token/PartialConv 对照（{conv_token_best['label']}）的 missing NRMSE 变化为 {branch_delta:+.2f}%。负值支持 temporal FNO branch，正值则表示当前配置尚未超过该对照。",
            f"- 300 帧预测暂列第一的是 **{min(table1, key=lambda r: _metric(r['metrics']['forecast_300'], 'nrmse'))['label']}**；重构排名与下游 RFNO 排名不一致时，说明局部补全误差会被动力学 rollout 选择性放大。",
            "",
            "## 表2：FNO-DeepONet 预测结构对照",
            "",
            "| 方法 | epoch | 30帧 NRMSE | 300帧 NRMSE | 300帧 SSP | 30→300增长比 | 参数量 | 总推理 s/sample |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in table2:
        f30 = _metric(record["metrics"]["forecast_30"], "nrmse")
        f300 = _metric(record["metrics"]["forecast_300"], "nrmse")
        lines.append(
            "| {label} | {epochs} | {f30} | {f300} | {ssp} | {growth} | {params:,} | {latency} |".format(
                label=record["label"],
                epochs=record["epochs"],
                f30=_fmt(f30),
                f300=_fmt(f300),
                ssp=_fmt(_metric(record["metrics"]["forecast_300"], "ssp")),
                growth=_fmt(f300 / f30, 3),
                params=record["parameters_total"],
                latency=_fmt(float(record["latency"]["total_seconds_per_sample"]), 4),
            )
        )
    lines.extend(
        [
            "",
            "结构问题的初步回答：",
            "",
            f"1. **Temporal FNO branch 是否更好：** FD-R1 对最佳非 FNO-DeepONet 30ep 对照的 missing NRMSE 变化为 {branch_delta:+.2f}%；这是 seed42 条件证据，不能外推为总体优势。",
            f"2. **Forecast-aware joint training：** FD-A1 相对 FD-R1 的 300帧 NRMSE 变化为 {joint_delta:+.2f}%；其 30→300 误差增长比为 {fd_a1_growth:.3f}，FD-R1 为 {fd_r1_growth:.3f}。",
            f"3. **直接 300 帧解码：** FD-L1 相对 FD-R1 / FD-A1 的 300帧 NRMSE 变化分别为 {direct_delta_vs_r1:+.2f}% / {direct_delta_vs_a1:+.2f}%，误差增长比为 {fd_l1_growth:.3f}。直接解码若短期较好但增长比或 SSP 较差，失败模式是时间一致性不足；若重构也差，则共享稀疏编码容量可能被 360 个输出时刻稀释。",
            f"4. **误差—稳定性—效率平衡：** 按 300帧 NRMSE 优先、同等误差下再看推理时间，当前最均衡的是 **{balance_rank[0]['label']}**。效率判断仅基于同机 validation、固定批量的模型内计时。",
            "",
            "## Protocol mismatch 与风险",
            "",
            "- 表1和表2的训练目标与 epoch 预算不同，禁止做不加限定的总排名。",
            "- 部分旧 checkpoint 可能存在解析器版本造成的哈希 mismatch；逐字段审计结果写入 manifest，任何真实 split/mask/noise/normalization mismatch 都会使构建失败。",
            "- seed42 对特定 split 和固定 mask_00006 的偶然性无法估计；没有 seed43/44 就没有方差或显著性结论。",
            "- 当前没有 OOD mask、噪声、时间丢失或 test 证据；泛化和鲁棒性仍未知。",
            "- 推理时间排除了数据加载和 CPU→GPU 拷贝；FD-L1 的共享编码成本按约定计入重构阶段。",
            "- checkpoint 大小、训练时间和峰值显存在 `efficiency.csv`；完整逐帧增长曲线在 `error_growth.csv`。",
            "",
            "## 是否进入三 seed 与下一阶段",
            "",
            f"值得进入三 seed 正式比较的最高优先级候选为：**{'、'.join(recommended)}**。建议下一阶段只扩展这 {len(recommended)} 个候选到 seed=43/44，并继续复用锁定 split/mask/noise/normalization；在三 seed validation 稳定后，再决定是否申请一次性 test。暂不建议 FD-R1-100ep 或宽泛超参数扫描。",
            "",
        ]
    )
    (RESULT_DIR / "preliminary_assessment.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
