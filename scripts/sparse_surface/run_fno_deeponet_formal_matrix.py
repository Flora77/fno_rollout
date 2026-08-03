#!/usr/bin/env python3
"""Run the three formal FNO-DeepONet experiments sequentially and quietly."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict


EXPERIMENTS = (
    (
        "FD-R1",
        "config/sparse_experiments/"
        "fd_r1_fno_deeponet_frozen_rfno_seed42.formal.json",
    ),
    (
        "FD-A1",
        "config/sparse_experiments/"
        "fd_a1_fno_deeponet_autoregressive_seed42.formal.json",
    ),
    (
        "FD-L1",
        "config/sparse_experiments/"
        "fd_l1_fno_deeponet_direct_long_seed42.formal.json",
    ),
)


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--launch-dir",
        type=Path,
        default=Path("runs/sparse/_launchers/fno_deeponet_seed42_20260802"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.project_root.resolve()
    launch_dir = args.launch_dir
    if not launch_dir.is_absolute():
        launch_dir = (root / launch_dir).resolve()
    state_path = launch_dir / "matrix_status.json"
    if state_path.exists():
        raise FileExistsError(
            f"Launcher state already exists; refusing an accidental duplicate: {state_path}"
        )

    state: Dict[str, Any] = {
        "launcher_pid": os.getpid(),
        "python": sys.executable,
        "project_root": str(root),
        "device": str(args.device),
        "started_at_unix": time.time(),
        "status": "running",
        "active_experiment": None,
        "experiments": [
            {"experiment_id": experiment_id, "config": config, "status": "pending"}
            for experiment_id, config in EXPERIMENTS
        ],
    }
    _atomic_json(state_path, state)

    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    for index, (experiment_id, relative_config) in enumerate(EXPERIMENTS):
        config_path = (root / relative_config).resolve()
        log_path = launch_dir / f"{index + 1:02d}_{experiment_id.lower()}.log"
        record = state["experiments"][index]
        record["status"] = "running"
        record["started_at_unix"] = time.time()
        record["log_path"] = str(log_path)
        state["active_experiment"] = experiment_id
        _atomic_json(state_path, state)

        command = [
            sys.executable,
            str(root / "scripts/sparse_surface/train_sparse_experiment.py"),
            str(config_path),
            "--project-root",
            str(root),
            "--device",
            str(args.device),
        ]
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
        record["return_code"] = int(completed.returncode)
        record["finished_at_unix"] = time.time()
        record["status"] = "complete" if completed.returncode == 0 else "failed"
        state["active_experiment"] = None
        if completed.returncode != 0:
            state["status"] = "failed"
            state["finished_at_unix"] = time.time()
            _atomic_json(state_path, state)
            return int(completed.returncode)
        _atomic_json(state_path, state)

    state["status"] = "complete"
    state["finished_at_unix"] = time.time()
    _atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
