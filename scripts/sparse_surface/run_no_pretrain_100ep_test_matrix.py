"""Run the locked no-pretraining 100-epoch reconstruction/test matrix.

All learned reconstructors are trained from random initialization using train/val
only.  Test evaluation starts only after every learned run has produced a formal
best.pt and last.pt, preventing test results from influencing later training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config/sparse_experiments/no_pretrain_100ep_test"
RESULT_DIR = ROOT / "results/no_pretrain_100ep_test_matrix_20260731_v1"
TRAIN_ENTRY = ROOT / "scripts/sparse_surface/train_sparse_experiment.py"
EVAL_ENTRY = ROOT / "scripts/sparse_surface/run_sparse_experiment.py"

LEARNED_CONFIGS = (
    "mask_unet_random_seed42.json",
    "mask_unet_random_seed43.json",
    "mask_unet_random_seed44.json",
    "st_d0_random_seed42.json",
    "st_d0_random_seed43.json",
    "st_d0_random_seed44.json",
    "partialconv_light_random_seed42.json",
    "partialconv_light_random_seed43.json",
    "partialconv_light_random_seed44.json",
)
BILINEAR_CONFIG = CONFIG_DIR / "bilinear_test_reference.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required authorization for the locked test phase.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Config root must be an object: {path}")
    return payload


def write_state(state: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    target = RESULT_DIR / "matrix_state.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(target)


def run_logged(command: list[str], log_name: str) -> int:
    log_dir = RESULT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / log_name).open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n[{datetime.now(timezone.utc).isoformat()}] "
            + subprocess.list2cmdline(command)
            + "\n"
        )
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
        stream.write(f"\nreturncode={completed.returncode}\n")
        return int(completed.returncode)


def learned_run_dir(config: dict[str, Any]) -> Path:
    return ROOT / "runs/sparse" / str(config["experiment_name"])


def train_all(state: dict[str, Any]) -> bool:
    all_complete = True
    for filename in LEARNED_CONFIGS:
        config_path = CONFIG_DIR / filename
        config = load_config(config_path)
        run_dir = learned_run_dir(config)
        best = run_dir / "checkpoints/best.pt"
        last = run_dir / "checkpoints/last.pt"
        record = state["training"].setdefault(filename, {})
        record.update(
            {
                "config": str(config_path),
                "run_dir": str(run_dir),
                "best_checkpoint": str(best),
                "last_checkpoint": str(last),
            }
        )
        if best.is_file() and last.is_file():
            record["status"] = "complete"
            write_state(state)
            continue

        command = [
            sys.executable,
            str(TRAIN_ENTRY),
            str(config_path),
            "--project-root",
            str(ROOT),
        ]
        if run_dir.exists():
            if not last.is_file():
                record["status"] = "failed_existing_run_without_last"
                all_complete = False
                write_state(state)
                continue
            command.extend(
                ["--run-dir", str(run_dir), "--resume", str(last)]
            )
        returncode = run_logged(command, f"train_{config_path.stem}.log")
        if returncode == 0 and best.is_file() and last.is_file():
            record["status"] = "complete"
        else:
            record["status"] = f"failed_returncode_{returncode}"
            all_complete = False
        write_state(state)
    return all_complete


def evaluate_learned_test(
    state: dict[str, Any], filename: str, frozen_split: Path
) -> bool:
    config_path = CONFIG_DIR / filename
    config = load_config(config_path)
    train_dir = learned_run_dir(config)
    best = train_dir / "checkpoints/best.pt"
    test_dir = (
        ROOT
        / "runs/sparse"
        / f"{config['experiment_name']}_locked_test300"
    )
    metrics = test_dir / "evaluation/test_metrics.json"
    record = state["test"].setdefault(filename, {})
    record.update(
        {
            "checkpoint": str(best),
            "run_dir": str(test_dir),
            "metrics": str(metrics),
        }
    )
    if metrics.is_file():
        record["status"] = "complete_existing"
        write_state(state)
        return True
    if test_dir.exists():
        record["status"] = "failed_existing_incomplete_test_dir"
        write_state(state)
        return False
    command = [
        sys.executable,
        str(EVAL_ENTRY),
        str(config_path),
        "--project-root",
        str(ROOT),
        "--run-dir",
        str(test_dir),
        "--split-manifest",
        str(frozen_split),
        "--evaluation-checkpoint",
        str(best),
        "--evaluate-split",
        "test",
        "--allow-test",
        "--device",
        "cuda",
    ]
    returncode = run_logged(command, f"test_{config_path.stem}.log")
    if returncode == 0 and metrics.is_file():
        record["status"] = "complete"
        write_state(state)
        return True
    record["status"] = f"failed_returncode_{returncode}"
    write_state(state)
    return False


def evaluate_bilinear_test(
    state: dict[str, Any], frozen_split: Path
) -> bool:
    config = load_config(BILINEAR_CONFIG)
    test_dir = ROOT / "runs/sparse" / str(config["experiment_name"])
    metrics = test_dir / "evaluation/test_metrics.json"
    record = state["test"].setdefault("bilinear_test_reference.json", {})
    record.update({"run_dir": str(test_dir), "metrics": str(metrics)})
    if metrics.is_file():
        record["status"] = "complete_existing"
        write_state(state)
        return True
    if test_dir.exists():
        record["status"] = "failed_existing_incomplete_test_dir"
        write_state(state)
        return False
    command = [
        sys.executable,
        str(EVAL_ENTRY),
        str(BILINEAR_CONFIG),
        "--project-root",
        str(ROOT),
        "--run-dir",
        str(test_dir),
        "--split-manifest",
        str(frozen_split),
        "--evaluate-split",
        "test",
        "--allow-test",
        "--device",
        "cuda",
    ]
    returncode = run_logged(command, "test_bilinear_reference.log")
    if returncode == 0 and metrics.is_file():
        record["status"] = "complete"
        write_state(state)
        return True
    record["status"] = f"failed_returncode_{returncode}"
    write_state(state)
    return False


def main() -> int:
    args = parse_args()
    state: dict[str, Any] = {
        "experiment": "NO-PRETRAIN-100EP-TEST",
        "test_authorized": bool(args.allow_test),
        "training": {},
        "test": {},
        "test_policy": (
            "Test begins only after all nine learned train/val runs have "
            "formal best.pt and last.pt."
        ),
    }
    write_state(state)
    if not train_all(state):
        state["status"] = "training_incomplete_test_not_accessed"
        write_state(state)
        return 2
    if not args.allow_test:
        state["status"] = "training_complete_test_not_authorized"
        write_state(state)
        return 3

    first_config = load_config(CONFIG_DIR / LEARNED_CONFIGS[0])
    frozen_split = learned_run_dir(first_config) / "data_split.json"
    if not frozen_split.is_file():
        state["status"] = "missing_frozen_split_test_not_accessed"
        write_state(state)
        return 4

    all_test_complete = True
    for filename in LEARNED_CONFIGS:
        all_test_complete &= evaluate_learned_test(
            state, filename, frozen_split
        )
    all_test_complete &= evaluate_bilinear_test(state, frozen_split)
    state["status"] = (
        "complete" if all_test_complete else "test_incomplete"
    )
    write_state(state)
    return 0 if all_test_complete else 5


if __name__ == "__main__":
    raise SystemExit(main())
