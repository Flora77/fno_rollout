#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch runner for G1-G9 comparison/ablation experiments.

Place this file in the NEURALOPERATOR project root, then run for example:

    python batch_run_g1_g9_ablation_comparison.py --splits val test
    python batch_run_g1_g9_ablation_comparison.py --only G1 G5 G9 --splits val test
    python batch_run_g1_g9_ablation_comparison.py --skip_train --splits val 
    python batch_run_g1_g9_ablation_comparison.py --skip_existing --splits val test
    python batch_run_g1_g9_ablation_comparison.py --only G10 G11 --dry_run
    python batch_run_g1_g9_ablation_comparison.py --dry_run
    python batch_run_g1_g9_ablation_comparison.py
    python batch_run_g1_g9_ablation_comparison.py --only G1 G2 G3 G4 --skip_train --dry_run

Default experiment list
-----------------------
G1  FNO     + curriculum rollout    + no added loss
G2  FNO     + no curriculum rollout + no added loss
G3  MSFNO   + curriculum rollout    + no added loss
G4  MSFNO   + no curriculum rollout + no added loss
G5  FNO     + curriculum rollout    + added loss
G6  FNO     + direct 60 -> 300 prediction, no external rollout training
G7  MSFNO   + direct 60 -> 300 prediction, no external rollout training
G8  FNO     + direct 60 -> 60 prediction, no external rollout training
G9  MSFNO   + direct 60 -> 60 prediction, no external rollout training
G10 ConvLSTM+ no curriculum rollout + no added loss
G11 ConvLSTM+ direct 60 -> 300 prediction, no external rollout training

Notes
-----
1) FNO/MSFNO uses:
       scripts/msfno_rno_log/train_sea_surface_rollout_rno_msfno.py
       scripts/msfno_rno_log/validate_sea_surface_rollout_rno_msfno.py
       config/sea_surface_rollout_config_rno_msfno.py

2) ConvLSTM uses:
       scripts/convlstm_log/train_sea_surface_rollout_convlstm.py
       scripts/convlstm_log/validate_sea_surface_rollout_convlstm.py
       config/sea_surface_rollout_config_convlstm.py

3) Some switches, especially use_segment_weighting and the ConvLSTM curriculum
   fields, are not exposed by all training CLIs. This runner temporarily patches
   the corresponding config file before each experiment and restores the original
   config files at the end, even if an error occurs.

4) For G6/G7/G9, output_steps=300 and rollout_steps=300. Therefore validation
   calls the model once to produce 300 frames. ConvLSTM still has its internal
   decoder recurrence, but there is no external chunk-by-chunk rollout loop.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


CURRICULUM_STEPS = (30, 60, 120, 180, 240, 300)
CURRICULUM_BOUNDARIES = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6)

NO_CURRICULUM_STEPS = (30,)
NO_CURRICULUM_BOUNDARIES = (0.0,)

DIRECT_STEPS = (300,)
DIRECT_BOUNDARIES = (0.0,)


@dataclass(frozen=True)
class ExperimentSpec:
    gid: str
    title: str
    family: str                         # "rno_msfno" or "convlstm"
    model_arch: str                      # "fno", "msfno", or "convlstm"
    experiment_name: str
    input_steps: int
    output_steps: int
    rollout_steps: int
    rollout_train_steps: Tuple[int, ...]
    rollout_curriculum_boundaries: Tuple[float, ...]
    use_long_rollout_curriculum: bool
    use_segment_weighting: bool
    use_spatial_gradient_loss: bool
    config_relpath: str
    train_relpath: str
    validate_relpath: str
    extra_config_updates: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScriptLayout:
    rno_config: str = "config/sea_surface_rollout_config_rno_msfno.py"
    rno_train: str = "scripts/msfno_rno_log/train_sea_surface_rollout_rno_msfno.py"
    rno_validate: str = "scripts/msfno_rno_log/validate_sea_surface_rollout_rno_msfno.py"
    conv_config: str = "config/sea_surface_rollout_config_convlstm.py"
    conv_train: str = "scripts/convlstm_log/train_sea_surface_rollout_convlstm.py"
    conv_validate: str = "scripts/convlstm_log/validate_sea_surface_rollout_convlstm.py"


def find_project_root(start: Path) -> Path:
    """Find a project root containing config/ and scripts/."""
    start = start.resolve()
    for parent in [start.parent, *start.parents]:
        if (parent / "config").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError(
        "Could not find project root. Put this script under the NEURALOPERATOR "
        "project root, or under a subfolder of that root."
    )


def as_python_literal(value: Any) -> str:
    """Return a compact Python literal for dataclass config replacement."""
    if isinstance(value, tuple):
        if len(value) == 1:
            return f"({as_python_literal(value[0])},)"
        return "(" + ", ".join(as_python_literal(v) for v in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(as_python_literal(v) for v in value) + "]"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return repr(value)


def patch_dataclass_config_text(text: str, updates: Dict[str, Any], config_path: Path) -> str:
    """
    Replace dataclass default assignments while preserving type annotations/comments.

    Example:
        rollout_train_steps: tuple = (30, 60)  # comment
    becomes:
        rollout_train_steps: tuple = (30,)  # comment
    """
    out = text
    for key, value in updates.items():
        literal = as_python_literal(value)
        pattern = re.compile(
            rf"^(\s*{re.escape(key)}\s*:\s*[^=\n]+\s*=\s*)([^#\n]*?)(\s*(?:#.*)?$)",
            flags=re.MULTILINE,
        )

        def repl(match: re.Match) -> str:
            return f"{match.group(1)}{literal}{match.group(3)}"

        out, n = pattern.subn(repl, out, count=1)
        if n == 0:
            raise KeyError(f"Could not find config field '{key}' in {config_path}")
    return out


def write_patched_config(config_path: Path, original_text: str, updates: Dict[str, Any]) -> None:
    patched = patch_dataclass_config_text(original_text, updates, config_path)
    config_path.write_text(patched, encoding="utf-8")


def tuple_arg(values: Sequence[Any]) -> str:
    return ",".join(str(v) for v in values)


def build_experiments(layout: ScriptLayout) -> List[ExperimentSpec]:
    rno_common = dict(
        family="rno_msfno",
        config_relpath=layout.rno_config,
        train_relpath=layout.rno_train,
        validate_relpath=layout.rno_validate,
    )
    conv_common = dict(
        family="convlstm",
        model_arch="convlstm",
        config_relpath=layout.conv_config,
        train_relpath=layout.conv_train,
        validate_relpath=layout.conv_validate,
    )

    return [
        ExperimentSpec(
            gid="G1",
            title="FNO + curriculum rollout + no added loss",
            model_arch="fno",
            experiment_name="G1_fno_curriculum_rollout300_no_added_loss",
            input_steps=60,
            output_steps=30,
            rollout_steps=300,
            rollout_train_steps=CURRICULUM_STEPS,
            rollout_curriculum_boundaries=CURRICULUM_BOUNDARIES,
            use_long_rollout_curriculum=True,
            use_segment_weighting=False,
            use_spatial_gradient_loss=False,
            **rno_common,
        ),
        ExperimentSpec(
            gid="G2",
            title="FNO + no curriculum rollout + no added loss",
            model_arch="fno",
            experiment_name="G2_fno_no_curriculum_rollout300_no_added_loss",
            input_steps=60,
            output_steps=30,
            rollout_steps=300,
            rollout_train_steps=NO_CURRICULUM_STEPS,
            rollout_curriculum_boundaries=NO_CURRICULUM_BOUNDARIES,
            use_long_rollout_curriculum=True,
            use_segment_weighting=False,
            use_spatial_gradient_loss=False,
            **rno_common,
        ),
        ExperimentSpec(
            gid="G3",
            title="MSFNO + curriculum rollout + no added loss",
            model_arch="msfno",
            experiment_name="G3_msfno_curriculum_rollout300_no_added_loss",
            input_steps=60,
            output_steps=30,
            rollout_steps=300,
            rollout_train_steps=CURRICULUM_STEPS,
            rollout_curriculum_boundaries=CURRICULUM_BOUNDARIES,
            use_long_rollout_curriculum=True,
            use_segment_weighting=False,
            use_spatial_gradient_loss=False,
            **rno_common,
        ),
        ExperimentSpec(
            gid="G4",
            title="MSFNO + no curriculum rollout + no added loss",
            model_arch="msfno",
            experiment_name="G4_msfno_no_curriculum_rollout300_no_added_loss",
            input_steps=60,
            output_steps=30,
            rollout_steps=300,
            rollout_train_steps=NO_CURRICULUM_STEPS,
            rollout_curriculum_boundaries=NO_CURRICULUM_BOUNDARIES,
            use_long_rollout_curriculum=True,
            use_segment_weighting=False,
            use_spatial_gradient_loss=False,
            **rno_common,
        ),
        ExperimentSpec(
            gid="G5",
            title="FNO + curriculum rollout + added loss",
            model_arch="fno",
            experiment_name="G5_fno_curriculum_rollout300_added_loss",
            input_steps=60,
            output_steps=30,
            rollout_steps=300,
            rollout_train_steps=CURRICULUM_STEPS,
            rollout_curriculum_boundaries=CURRICULUM_BOUNDARIES,
            use_long_rollout_curriculum=True,
            use_segment_weighting=True,
            use_spatial_gradient_loss=True,
            **rno_common,
        ),
        ExperimentSpec(
            gid="G6",
            title="FNO direct 60->300, no external rollout training, no added loss",
            model_arch="fno",
            experiment_name="G6_fno_direct60to300_no_rollout_no_added_loss",
            input_steps=60,
            output_steps=300,
            rollout_steps=300,
            rollout_train_steps=DIRECT_STEPS,
            rollout_curriculum_boundaries=DIRECT_BOUNDARIES,
            use_long_rollout_curriculum=False,
            use_segment_weighting=False,
            use_spatial_gradient_loss=False,
            **rno_common,
        ),
        ExperimentSpec(
            gid="G7",
            title="MSFNO direct 60->300, no external rollout training, no added loss",
            model_arch="msfno",
            experiment_name="G7_msfno_direct60to300_no_rollout_no_added_loss",
            input_steps=60,
            output_steps=300,
            rollout_steps=300,
            rollout_train_steps=DIRECT_STEPS,
            rollout_curriculum_boundaries=DIRECT_BOUNDARIES,
            use_long_rollout_curriculum=False,
            use_segment_weighting=False,
            use_spatial_gradient_loss=False,
            **rno_common,
        ),

        ExperimentSpec(
            gid="G8",
            title="FNO direct 60->60, no external rollout training, no added loss",
            model_arch="fno",
            experiment_name="G8_fno_direct60to60_no_rollout_no_added_loss",
            input_steps=60,
            output_steps=60,
            rollout_steps=60,
            rollout_train_steps=DIRECT_STEPS,
            rollout_curriculum_boundaries=DIRECT_BOUNDARIES,
            use_long_rollout_curriculum=False,
            use_segment_weighting=False,
            use_spatial_gradient_loss=False,
            **rno_common,
        ),
        ExperimentSpec(
            gid="G9",
            title="MSFNO direct 60->60, no external rollout training, no added loss",
            model_arch="msfno",
            experiment_name="G9_msfno_direct60to60_no_rollout_no_added_loss",
            input_steps=60,
            output_steps=60,
            rollout_steps=60,
            rollout_train_steps=DIRECT_STEPS,
            rollout_curriculum_boundaries=DIRECT_BOUNDARIES,
            use_long_rollout_curriculum=False,
            use_segment_weighting=False,
            use_spatial_gradient_loss=False,
            **rno_common,
        ),



        ExperimentSpec(
            gid="G10",
            title="ConvLSTM + no curriculum rollout + no added loss",
            experiment_name="G10_convlstm_no_curriculum_rollout300_no_added_loss",
            input_steps=60,
            output_steps=30,
            rollout_steps=300,
            rollout_train_steps=NO_CURRICULUM_STEPS,
            rollout_curriculum_boundaries=NO_CURRICULUM_BOUNDARIES,
            use_long_rollout_curriculum=True,
            use_segment_weighting=False,
            use_spatial_gradient_loss=False,
            **conv_common,
        ),
        ExperimentSpec(
            gid="G11",
            title="ConvLSTM direct 60->300, no external rollout training, no added loss",
            experiment_name="G11_convlstm_direct60to300_no_rollout_no_added_loss",
            input_steps=60,
            output_steps=300,
            rollout_steps=300,
            rollout_train_steps=DIRECT_STEPS,
            rollout_curriculum_boundaries=DIRECT_BOUNDARIES,
            use_long_rollout_curriculum=False,
            use_segment_weighting=False,
            use_spatial_gradient_loss=False,
            **conv_common,
        ),
    ]


def build_config_updates(exp: ExperimentSpec, checkpoint_dir: str, data_root: Optional[str]) -> Dict[str, Any]:
    updates: Dict[str, Any] = {
        "experiment_name": exp.experiment_name,
        "checkpoint_dir": checkpoint_dir,
        "input_steps": int(exp.input_steps),
        "output_steps": int(exp.output_steps),
        "rollout_steps": int(exp.rollout_steps),
        "use_long_rollout_curriculum": bool(exp.use_long_rollout_curriculum),
        "rollout_train_steps": tuple(exp.rollout_train_steps),
        "rollout_curriculum_boundaries": tuple(exp.rollout_curriculum_boundaries),
        "use_segment_weighting": bool(exp.use_segment_weighting),
        "use_spatial_gradient_loss": bool(exp.use_spatial_gradient_loss),
        **exp.extra_config_updates,
    }
    if data_root is not None:
        updates["data_root"] = data_root
    if exp.family == "rno_msfno":
        updates["model_arch"] = exp.model_arch
    return updates


def build_train_command(
    exp: ExperimentSpec,
    root: Path,
    checkpoint_dir: str,
    data_root: Optional[str],
    n_epochs: Optional[int],
    batch_size: Optional[int],
    val_batch_size: Optional[int],
    test_batch_size: Optional[int],
    learning_rate: Optional[float],
) -> List[str]:
    cmd = [sys.executable, str(root / exp.train_relpath)]

    # Common CLI accepted by both training scripts.
    cmd += ["--experiment_name", exp.experiment_name]
    cmd += ["--checkpoint_dir", checkpoint_dir]
    if data_root is not None:
        cmd += ["--data_root", data_root]
    if n_epochs is not None:
        cmd += ["--n_epochs", str(n_epochs)]
    if batch_size is not None:
        cmd += ["--batch_size", str(batch_size)]
    if val_batch_size is not None:
        cmd += ["--val_batch_size", str(val_batch_size)]
    if learning_rate is not None:
        cmd += ["--learning_rate", str(learning_rate)]
    cmd += ["--input_steps", str(exp.input_steps)]
    cmd += ["--output_steps", str(exp.output_steps)]
    cmd += ["--rollout_steps", str(exp.rollout_steps)]

    # Extra CLI available in the FNO/MSFNO script. ConvLSTM's missing switches are
    # controlled by temporary config patching.
    if exp.family == "rno_msfno":
        cmd += ["--model_arch", exp.model_arch]
        if test_batch_size is not None:
            cmd += ["--test_batch_size", str(test_batch_size)]
        cmd += ["--use_long_rollout_curriculum", "1" if exp.use_long_rollout_curriculum else "0"]
        cmd += ["--rollout_train_steps", tuple_arg(exp.rollout_train_steps)]
        cmd += ["--rollout_curriculum_boundaries", tuple_arg(exp.rollout_curriculum_boundaries)]
        cmd += ["--use_spatial_gradient_loss", "1" if exp.use_spatial_gradient_loss else "0"]

    return cmd


def build_validate_command(
    exp: ExperimentSpec,
    root: Path,
    split: str,
    checkpoint_dir: str,
    data_root: Optional[str],
) -> List[str]:
    cmd = [
        sys.executable,
        str(root / exp.validate_relpath),
        split,
        "--experiment_name",
        exp.experiment_name,
        "--checkpoint_dir",
        checkpoint_dir,
    ]
    if data_root is not None:
        cmd += ["--data_root", data_root]
    return cmd


def run_command(cmd: List[str], cwd: Path, log_path: Path, dry_run: bool, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    print(f"\n[RUN] {printable}")
    print(f"[LOG] {log_path}")
    if dry_run:
        log_path.write_text(printable + "\n", encoding="utf-8")
        return 0

    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="") as f:
        f.write(printable + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, env=env)
    elapsed = time.perf_counter() - t0
    print(f"[DONE] returncode={proc.returncode}, elapsed={elapsed:.1f}s")
    return int(proc.returncode)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_union_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_filtered_summaries(checkpoint_dir: Path, experiments: Sequence[ExperimentSpec]) -> Path:
    exp_names = {e.experiment_name for e in experiments}
    summary_files = [
        ("train", checkpoint_dir / "ablation_train_summary.csv"),
        ("val", checkpoint_dir / "ablation_val_summary.csv"),
        ("test", checkpoint_dir / "ablation_test_summary.csv"),
    ]
    rows: List[Dict[str, Any]] = []
    for summary_type, csv_path in summary_files:
        for row in read_csv_rows(csv_path):
            if row.get("experiment_name") in exp_names:
                row = dict(row)
                row["summary_type"] = summary_type
                rows.append(row)
    out_path = checkpoint_dir / "G1_G9_filtered_summary.csv"
    write_union_csv(out_path, rows)
    return out_path


def write_manifest(checkpoint_dir: Path, experiments: Sequence[ExperimentSpec], args: argparse.Namespace) -> Path:
    manifest = {
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_dir": str(checkpoint_dir),
        "splits": list(args.splits),
        "experiments": [
            {
                "gid": e.gid,
                "title": e.title,
                "family": e.family,
                "model_arch": e.model_arch,
                "experiment_name": e.experiment_name,
                "input_steps": e.input_steps,
                "output_steps": e.output_steps,
                "rollout_steps": e.rollout_steps,
                "rollout_train_steps": list(e.rollout_train_steps),
                "rollout_curriculum_boundaries": list(e.rollout_curriculum_boundaries),
                "use_long_rollout_curriculum": e.use_long_rollout_curriculum,
                "use_segment_weighting": e.use_segment_weighting,
                "use_spatial_gradient_loss": e.use_spatial_gradient_loss,
                "config_relpath": e.config_relpath,
                "train_relpath": e.train_relpath,
                "validate_relpath": e.validate_relpath,
            }
            for e in experiments
        ],
    }
    out = checkpoint_dir / "G1_G9_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run G1-G9 FNO/MSFNO/ConvLSTM experiments.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/G1_G9_ablation_comparison",
                        help="Directory used by train/validate scripts for checkpoints and summary CSVs.")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Optional data root override, e.g. ./data/bimodal.")
    parser.add_argument("--splits", nargs="+", default=["val"], choices=["val", "test"],
                        help="Validation/test splits to run after each training. Default: val.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Optional experiment ids to run, e.g. --only G1 G5 G9.")
    parser.add_argument("--skip_train", action="store_true",
                        help="Only run validation/test using existing checkpoints.")
    parser.add_argument("--skip_validate", action="store_true",
                        help="Only run training, no validation/test.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip training if <checkpoint_dir>/<experiment_name>.pt already exists.")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Continue running later experiments when one command fails.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print/write commands without executing them.")
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--test_batch_size", type=int, default=None,
                        help="Only forwarded to the FNO/MSFNO training script because ConvLSTM's CLI may not expose it.")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--cuda_visible_devices", type=str, default=None,
                        help="Optional CUDA_VISIBLE_DEVICES value, e.g. 0.")

    # Optional layout overrides, useful if folder names change later.
    parser.add_argument("--rno_config", type=str, default=ScriptLayout.rno_config)
    parser.add_argument("--rno_train", type=str, default=ScriptLayout.rno_train)
    parser.add_argument("--rno_validate", type=str, default=ScriptLayout.rno_validate)
    parser.add_argument("--conv_config", type=str, default=ScriptLayout.conv_config)
    parser.add_argument("--conv_train", type=str, default=ScriptLayout.conv_train)
    parser.add_argument("--conv_validate", type=str, default=ScriptLayout.conv_validate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_project_root(Path(__file__))
    checkpoint_dir = (root / args.checkpoint_dir).resolve() if not Path(args.checkpoint_dir).is_absolute() else Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    batch_log_dir = checkpoint_dir / "batch_logs"
    batch_log_dir.mkdir(parents=True, exist_ok=True)

    layout = ScriptLayout(
        rno_config=args.rno_config,
        rno_train=args.rno_train,
        rno_validate=args.rno_validate,
        conv_config=args.conv_config,
        conv_train=args.conv_train,
        conv_validate=args.conv_validate,
    )
    experiments = build_experiments(layout)
    if args.only:
        wanted = {x.upper() for x in args.only}
        experiments = [e for e in experiments if e.gid.upper() in wanted]
        missing = wanted - {e.gid.upper() for e in experiments}
        if missing:
            raise ValueError(f"Unknown experiment id(s): {sorted(missing)}")

    required_relpaths = sorted(
        {e.config_relpath for e in experiments}
        | {e.train_relpath for e in experiments}
        | {e.validate_relpath for e in experiments}
    )
    for rel in required_relpaths:
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    config_paths = sorted({root / e.config_relpath for e in experiments})
    original_configs = {p: p.read_text(encoding="utf-8") for p in config_paths}

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    manifest_path = write_manifest(checkpoint_dir, experiments, args)
    print(f"[MANIFEST] {manifest_path}")
    print(f"[ROOT] {root}")
    print(f"[CHECKPOINT_DIR] {checkpoint_dir}")

    failures: List[Tuple[str, str, int]] = []

    try:
        for exp in experiments:
            print("\n" + "=" * 100)
            print(f"{exp.gid}. {exp.title}")
            print(f"experiment_name = {exp.experiment_name}")
            print("=" * 100)

            config_path = root / exp.config_relpath
            updates = build_config_updates(exp, str(checkpoint_dir), args.data_root)
            write_patched_config(config_path, original_configs[config_path], updates)
            print(f"[CONFIG PATCHED] {config_path}")
            print(json.dumps(updates, ensure_ascii=False, default=str, indent=2))

            ckpt_path = checkpoint_dir / f"{exp.experiment_name}.pt"
            if args.skip_train or (args.skip_existing and ckpt_path.exists()):
                if args.skip_train:
                    print(f"[SKIP TRAIN] --skip_train set: {ckpt_path}")
                else:
                    print(f"[SKIP TRAIN] checkpoint exists: {ckpt_path}")
            else:
                train_cmd = build_train_command(
                    exp=exp,
                    root=root,
                    checkpoint_dir=str(checkpoint_dir),
                    data_root=args.data_root,
                    n_epochs=args.n_epochs,
                    batch_size=args.batch_size,
                    val_batch_size=args.val_batch_size,
                    test_batch_size=args.test_batch_size,
                    learning_rate=args.learning_rate,
                )
                ret = run_command(
                    train_cmd,
                    cwd=root,
                    log_path=batch_log_dir / f"{exp.gid}_{exp.experiment_name}_train.log",
                    dry_run=args.dry_run,
                    env=env,
                )
                if ret != 0:
                    failures.append((exp.gid, "train", ret))
                    if not args.continue_on_error:
                        raise RuntimeError(f"{exp.gid} train failed with return code {ret}")

            if not args.skip_validate:
                for split in args.splits:
                    val_cmd = build_validate_command(
                        exp=exp,
                        root=root,
                        split=split,
                        checkpoint_dir=str(checkpoint_dir),
                        data_root=args.data_root,
                    )
                    ret = run_command(
                        val_cmd,
                        cwd=root,
                        log_path=batch_log_dir / f"{exp.gid}_{exp.experiment_name}_{split}.log",
                        dry_run=args.dry_run,
                        env=env,
                    )
                    if ret != 0:
                        failures.append((exp.gid, split, ret))
                        if not args.continue_on_error:
                            raise RuntimeError(f"{exp.gid} {split} failed with return code {ret}")

        summary_path = collect_filtered_summaries(checkpoint_dir, experiments)
        print(f"\n[SUMMARY] {summary_path}")

    finally:
        for path, text in original_configs.items():
            path.write_text(text, encoding="utf-8")
        print("[CONFIG RESTORED] Original config files were restored.")

    if failures:
        print("\n[FAILED COMMANDS]")
        for gid, stage, ret in failures:
            print(f"  {gid} {stage}: returncode={ret}")
        sys.exit(1)

    print("\nAll requested experiments finished.")


if __name__ == "__main__":
    main()
