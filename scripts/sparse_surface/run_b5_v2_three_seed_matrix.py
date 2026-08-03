#!/usr/bin/env python3
"""Run the locked nonlinear GNO/GINO training and validation matrix sequentially."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


RUNS = (
    ("gno", 42),
    ("gno", 43),
    ("gno", 44),
    ("gino", 42),
    ("gino", 43),
    ("gino", 44),
)


def _run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return int(process.returncode)


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    python = Path(sys.executable).resolve()
    split = project_root / "data" / "splits" / "sea_surface_bimodal_v1.json"
    artifact_root = (
        project_root / "results" / "min_b5_gno_gino_v2_three_seed_20260729_v1"
    )
    status_path = artifact_root / "matrix_status.json"
    status: list[dict[str, object]] = []

    for method, seed in RUNS:
        label = f"b5_v2_{method}_seed{seed}"
        config = (
            project_root
            / "config"
            / "sparse_experiments"
            / f"b5_v2_{method}_frozen_seed{seed}.formal.json"
        )
        train_dir = (
            project_root
            / "runs"
            / "sparse"
            / f"{label}_formal_20260729_v1"
        )
        summaries = sorted(train_dir.glob("training_summary_epoch*.json"))
        train_returncode = 0
        if not summaries:
            if train_dir.exists():
                train_returncode = 98
            else:
                train_returncode = _run(
                    [
                        str(python),
                        str(
                            project_root
                            / "scripts"
                            / "sparse_surface"
                            / "train_sparse_experiment.py"
                        ),
                        str(config),
                        "--project-root",
                        str(project_root),
                        "--run-dir",
                        str(train_dir),
                        "--split-manifest",
                        str(split),
                        "--device",
                        "cuda",
                    ],
                    artifact_root / "orchestrator_logs" / f"{label}_train.log",
                )
            summaries = sorted(train_dir.glob("training_summary_epoch*.json"))

        best = train_dir / "checkpoints" / "best.pt"
        eval_dir = (
            project_root
            / "runs"
            / "sparse"
            / f"{label}_val300_20260729_v1"
        )
        metrics = eval_dir / "evaluation" / "val_metrics.json"
        eval_returncode: int | None = None
        if train_returncode == 0 and summaries and best.is_file():
            if metrics.is_file():
                eval_returncode = 0
            elif eval_dir.exists():
                eval_returncode = 97
            else:
                eval_returncode = _run(
                    [
                        str(python),
                        str(
                            project_root
                            / "scripts"
                            / "sparse_surface"
                            / "run_sparse_experiment.py"
                        ),
                        str(config),
                        "--project-root",
                        str(project_root),
                        "--run-dir",
                        str(eval_dir),
                        "--split-manifest",
                        str(split),
                        "--evaluation-checkpoint",
                        str(best),
                        "--device",
                        "cuda",
                        "--evaluate-split",
                        "val",
                    ],
                    artifact_root / "orchestrator_logs" / f"{label}_val300.log",
                )

        status.append(
            {
                "method": method,
                "seed": seed,
                "config": str(config),
                "train_dir": str(train_dir),
                "train_returncode": train_returncode,
                "training_summary": str(summaries[-1]) if summaries else None,
                "best_checkpoint": str(best) if best.is_file() else None,
                "eval_dir": str(eval_dir),
                "eval_returncode": eval_returncode,
                "metrics": str(metrics) if metrics.is_file() else None,
            }
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps(status, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    failed = [
        item
        for item in status
        if item["train_returncode"] != 0 or item["eval_returncode"] != 0
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
