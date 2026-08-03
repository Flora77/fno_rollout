#!/usr/bin/env python
"""Finish the missing seed-42 comparator, unified validation, latency and report."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from neuralop.training.sparse_experiment_runner import (
    load_resolved_sparse_config,
    scientific_config_sha256,
)


METHODS = (
    (
        "mask_unet_random",
        "config/sparse_experiments/min_b4_init_ablation_random_seed42.formal.json",
        "runs/sparse/min_b4_init_ablation_random_seed42_30ep_formal_20260729_v1",
    ),
    (
        "st_d0_random",
        "config/sparse_experiments/st_d0_sensor_token_grid_query_seed42.formal.json",
        "runs/sparse/st_d0_sensor_token_grid_query_seed42_30ep_formal_20260731_v1",
    ),
    (
        "partialconv_light_random_recon_only",
        "config/sparse_experiments/p1_partialconv_light_random_reconstruction_only_seed42.formal.json",
        "runs/sparse/p1_partialconv_light_random_reconstruction_only_30ep_fixed_r05_mask_00006_seed42_20260802_v1",
    ),
    (
        "fd_r1",
        "config/sparse_experiments/fd_r1_fno_deeponet_frozen_rfno_seed42.formal.json",
        "runs/sparse/fd_r1_fno_deeponet_random_frozen_rfno_fixed_points_r05_mask_00006_seed42_20260731_v1",
    ),
    (
        "fd_a1",
        "config/sparse_experiments/fd_a1_fno_deeponet_autoregressive_seed42.formal.json",
        "runs/sparse/fd_a1_fno_deeponet_reconstruction_joint_ar300_fixed_points_r05_mask_00006_seed42_20260731_v1",
    ),
    (
        "fd_l1",
        "config/sparse_experiments/fd_l1_fno_deeponet_direct_long_seed42.formal.json",
        "runs/sparse/fd_l1_fno_deeponet_direct_sparse_to_300_fixed_points_r05_mask_00006_seed42_20260731_v1",
    ),
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _checkpoint_hash(checkpoint: Path) -> str:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return str(payload.get("scientific_config_sha256", ""))


def _current_hash(config: Path, root: Path, run_dir: Path) -> str:
    resolved = load_resolved_sparse_config(
        config,
        project_root=root,
        run_dir=run_dir,
        device="cuda",
    )
    return scientific_config_sha256(resolved)


def _audit_locked_protocol(config: Path) -> None:
    payload = json.loads(config.read_text(encoding="utf-8"))
    data = payload["data"]
    observation = payload["observation"]
    required = {
        "input_steps": 60,
        "output_steps": 30,
        "rollout_steps": 300,
        "height": 64,
        "width": 64,
    }
    for key, expected in required.items():
        if data.get(key) != expected:
            raise ValueError(f"{config}: data.{key} != {expected}")
    observation_required = {
        "mask_type": "fixed_points",
        "observation_rate": 0.05,
        "mask_id": "mask_00006",
        "noise_std_fraction": 0.0,
        "temporal_dropout": 0.0,
        "seed": 42,
    }
    for key, expected in observation_required.items():
        if observation.get(key) != expected:
            raise ValueError(f"{config}: observation.{key} != {expected!r}")
    split = str(data["split_manifest_path"]).replace("\\", "/")
    mask = str(observation["manifest_path"]).replace("\\", "/")
    if not split.endswith("data/splits/sea_surface_bimodal_v1.json"):
        raise ValueError(f"Unexpected split: {split}")
    if not mask.endswith("data/masks/point_masks.npz"):
        raise ValueError(f"Unexpected mask manifest: {mask}")
    if int(payload["runtime"]["seed"]) != 42:
        raise ValueError(f"{config}: runtime.seed != 42")


def _run(
    command: list[str],
    *,
    root: Path,
    log_path: Path,
    state: dict[str, Any],
    job: str,
) -> None:
    state["status"] = "running"
    state["active_job"] = job
    state["jobs"].append(
        {"job": job, "status": "running", "log": str(log_path), "started": time.time()}
    )
    _atomic_json(state["state_path"], {k: v for k, v in state.items() if k != "state_path"})
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        completed = subprocess.run(
            command,
            cwd=str(root),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record = state["jobs"][-1]
    record["return_code"] = int(completed.returncode)
    record["finished"] = time.time()
    record["status"] = "complete" if completed.returncode == 0 else "failed"
    state["active_job"] = None
    if completed.returncode:
        state["status"] = "failed"
        _atomic_json(state["state_path"], {k: v for k, v in state.items() if k != "state_path"})
        raise subprocess.CalledProcessError(completed.returncode, command)
    _atomic_json(state["state_path"], {k: v for k, v in state.items() if k != "state_path"})


def main() -> int:
    root = Path.cwd().resolve()
    launch_dir = root / "runs/sparse/_launchers/fno_deeponet_seed42_completion_20260802"
    state_path = launch_dir / "completion_status.json"
    if state_path.exists():
        raise FileExistsError(f"Refusing duplicate completion launcher: {state_path}")
    matrix_status = root / "runs/sparse/_launchers/fno_deeponet_seed42_20260802/matrix_status.json"
    matrix = json.loads(matrix_status.read_text(encoding="utf-8"))
    if matrix.get("status") != "complete":
        raise RuntimeError(f"FNO-DeepONet matrix is not complete: {matrix.get('status')}")

    state: dict[str, Any] = {
        "launcher_pid": os.getpid(),
        "python": sys.executable,
        "started": time.time(),
        "status": "running",
        "active_job": None,
        "jobs": [],
        "state_path": state_path,
    }
    _atomic_json(state_path, {k: v for k, v in state.items() if k != "state_path"})
    for _, config_rel, _ in METHODS:
        _audit_locked_protocol(root / config_rel)

    p1_config = root / METHODS[2][1]
    p1_run = root / METHODS[2][2]
    p1_best = p1_run / "checkpoints/best.pt"
    if not p1_best.is_file():
        command = [
            sys.executable,
            str(root / "scripts/sparse_surface/train_sparse_experiment.py"),
            str(p1_config),
            "--project-root",
            str(root),
            "--device",
            "cuda",
        ]
        p1_last = p1_run / "checkpoints/last.pt"
        if p1_last.is_file():
            command.extend(["--resume", str(p1_last)])
        _run(
            command,
            root=root,
            log_path=launch_dir / "01_partialconv_reconstruction_only_train.log",
            state=state,
            job="train_partialconv_reconstruction_only",
        )

    eval_root = root / "runs/sparse/_evaluations/fno_deeponet_seed42_preliminary_20260802"
    bench_root = root / "runs/sparse/_benchmarks/fno_deeponet_seed42_preliminary_20260802"
    for index, (key, config_rel, run_rel) in enumerate(METHODS, start=2):
        config = root / config_rel
        checkpoint = root / run_rel / "checkpoints/best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        evaluation_run = eval_root / key
        metrics = evaluation_run / "evaluation/val_metrics.json"
        hash_mismatch = _checkpoint_hash(checkpoint) != _current_hash(
            config, root, evaluation_run
        )
        if not metrics.is_file():
            command = [
                sys.executable,
                str(root / "scripts/sparse_surface/run_sparse_experiment.py"),
                str(config),
                "--project-root",
                str(root),
                "--run-dir",
                str(evaluation_run),
                "--device",
                "cuda",
                "--evaluate-split",
                "val",
                "--evaluation-checkpoint",
                str(checkpoint),
                "--rollout-steps",
                "300",
            ]
            if hash_mismatch:
                command.append("--allow-validation-protocol-mismatch")
            _run(
                command,
                root=root,
                log_path=launch_dir / f"{index:02d}_{key}_val300.log",
                state=state,
                job=f"evaluate_{key}_val300",
            )

        benchmark_run = bench_root / key
        latency = benchmark_run / "component_latency.json"
        if not latency.is_file():
            command = [
                sys.executable,
                str(root / "scripts/sparse_surface/benchmark_sparse_component_latency.py"),
                str(config),
                str(checkpoint),
                "--project-root",
                str(root),
                "--run-dir",
                str(benchmark_run),
                "--device",
                "cuda",
                "--rollout-steps",
                "300",
                "--max-batches",
                "30",
                "--warmup-batches",
                "2",
            ]
            if hash_mismatch:
                command.append("--allow-config-mismatch")
            _run(
                command,
                root=root,
                log_path=launch_dir / f"{index:02d}_{key}_latency.log",
                state=state,
                job=f"benchmark_{key}_latency",
            )

    _run(
        [
            sys.executable,
            str(root / "scripts/sparse_surface/build_fno_deeponet_seed42_preliminary_comparison.py"),
        ],
        root=root,
        log_path=launch_dir / "99_build_report.log",
        state=state,
        job="build_preliminary_report",
    )
    state["status"] = "complete"
    state["finished"] = time.time()
    _atomic_json(state_path, {k: v for k, v in state.items() if k != "state_path"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
