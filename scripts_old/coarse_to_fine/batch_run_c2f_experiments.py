#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch runner for Coarse-to-fine sea-surface experiments.

功能：
1. 逐个实验临时修改 config/sea_surface_rollout_config.py 中的 dataclass 默认参数；
2. 运行训练脚本；
3. 运行验证脚本，可选 val/test；
4. 每个实验结束后自动进入下一组；
5. 所有实验结束后恢复原始 config 文件。

典型用法（在 neuraloperator 工程根目录运行）：

python scripts\coarse_to_fine\batch_run_c2f_experiments.py ^
  --config config\sea_surface_rollout_config_coarse_to_fine.py ^
  --train scripts\coarse_to_fine\train_sea_surface_rollout_coarse_to_fine.py ^
  --validate scripts\coarse_to_fine\validate_sea_surface_rollout_coarse_to_fine.py ^
  --splits val


如果想 val 和 test 都跑：

python scripts/coarse_to_fine/batch_run_c2f_experiments.py --splits val test

只跑某几类实验：

python scripts/coarse_to_fine/batch_run_c2f_experiments.py --only coarse_full240 residual_scale_scan

给所有实验额外覆盖参数：

python scripts/coarse_to_fine/batch_run_c2f_experiments.py --set batch_size=8 --set learning_rate=1e-4

注意：
- 本脚本会修改 config 文件，但会先备份，结束后默认自动恢复。
- train/validate 脚本本身无需支持 argparse，因为它们仍然从 config 文件读取参数。
"""

from __future__ import annotations

import argparse
import ast
import copy
import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# =============================================================================
# Experiment definitions
# =============================================================================

def default_experiments(coarse_ckpt: str) -> List[Dict[str, Any]]:
    """Return the experiment list.

    coarse_ckpt is injected into downstream experiments. By default, it points to
    the checkpoint produced by the first experiment:
        ./checkpoints/c2f_coarse_full240_uniform.pt
    """
    return [
        # ---------------------------------------------------------------------
        # (1) 重新训练 coarse-only，但改成 full240 训练
        # ---------------------------------------------------------------------
        {
            "group": "coarse_full240",
            "name": "c2f_coarse_full240_uniform",
            "description": "Coarse-only FNO, train full 240-step target directly.",
            "overrides": {
                "experiment_name": "c2f_coarse_full240_uniform",
                "c2f_train_stage": "coarse_only",
                "c2f_pretrained_coarse_path": "",
                "use_long_rollout_curriculum": False,
                "rollout_train_steps": (240,),
                "rollout_curriculum_boundaries": (0.0,),
                "rollout_detach_context": True,
                "c2f_coarse_target_mode": "uniform",
                "c2f_temporal_upsample_mode": "linear",
                "learning_rate": 2e-4,
                "c2f_coarse_loss_weight": 1.0,
                "c2f_base_loss_weight": 0.1,
                "c2f_refiner_type": "unet",
                "c2f_refiner_residual_scale": 1.0,
            },
        },

        # ---------------------------------------------------------------------
        # (2) joint U-Net + rollout_detach_context=True
        # ---------------------------------------------------------------------
        {
            "group": "joint_detach",
            "name": "c2f_joint_unet_detach_true",
            "description": "Joint FNO+U-Net finetune with detach_context=True.",
            "overrides": {
                "experiment_name": "c2f_joint_unet_detach_true",
                "c2f_train_stage": "joint",
                "c2f_pretrained_coarse_path": coarse_ckpt,
                "c2f_refiner_type": "unet",
                "c2f_refiner_base_channels": 64,
                "c2f_refiner_depth": 3,
                "c2f_refiner_residual_scale": 1.0,
                "c2f_coarse_loss_weight": 0.2,
                "c2f_base_loss_weight": 0.1,
                "use_long_rollout_curriculum": True,
                "rollout_train_steps": (20, 40, 80, 120, 160, 240),
                "rollout_curriculum_boundaries": (0.0, 0.10, 0.20, 0.30, 0.45, 0.60),
                "rollout_detach_context": True,
                "c2f_coarse_target_mode": "uniform",
                "c2f_temporal_upsample_mode": "linear",
                "learning_rate": 2e-4,
            },
        },

        # ---------------------------------------------------------------------
        # (3) joint U-Net + coarse loss weight 扫描
        # ---------------------------------------------------------------------
        *[
            {
                "group": "coarse_loss_scan",
                "name": f"c2f_joint_unet_coarsew{str(w).replace('.', 'p')}",
                "description": f"Joint U-Net, coarse loss weight = {w}.",
                "overrides": {
                    "experiment_name": f"c2f_joint_unet_coarsew{str(w).replace('.', 'p')}",
                    "c2f_train_stage": "joint",
                    "c2f_pretrained_coarse_path": coarse_ckpt,
                    "c2f_refiner_type": "unet",
                    "c2f_refiner_base_channels": 64,
                    "c2f_refiner_depth": 3,
                    "c2f_refiner_residual_scale": 1.0,
                    "c2f_coarse_loss_weight": float(w),
                    "c2f_base_loss_weight": 0.1,
                    "use_long_rollout_curriculum": True,
                    "rollout_train_steps": (20, 40, 80, 120, 160, 240),
                    "rollout_curriculum_boundaries": (0.0, 0.10, 0.20, 0.30, 0.45, 0.60),
                    "rollout_detach_context": True,
                    "c2f_coarse_target_mode": "uniform",
                    "c2f_temporal_upsample_mode": "linear",
                    "learning_rate": 2e-4,
                },
            }
            for w in (0.1, 0.2, 0.5)
        ],

        # ---------------------------------------------------------------------
        # (4) joint U-Net + residual scale 扫描
        # ---------------------------------------------------------------------
        *[
            {
                "group": "residual_scale_scan",
                "name": f"c2f_joint_unet_scale{str(s).replace('.', 'p')}",
                "description": f"Joint U-Net, refiner residual scale = {s}.",
                "overrides": {
                    "experiment_name": f"c2f_joint_unet_scale{str(s).replace('.', 'p')}",
                    "c2f_train_stage": "joint",
                    "c2f_pretrained_coarse_path": coarse_ckpt,
                    "c2f_refiner_type": "unet",
                    "c2f_refiner_base_channels": 64,
                    "c2f_refiner_depth": 3,
                    "c2f_refiner_residual_scale": float(s),
                    "c2f_coarse_loss_weight": 0.2,
                    "c2f_base_loss_weight": 0.1,
                    "use_long_rollout_curriculum": True,
                    "rollout_train_steps": (20, 40, 80, 120, 160, 240),
                    "rollout_curriculum_boundaries": (0.0, 0.10, 0.20, 0.30, 0.45, 0.60),
                    "rollout_detach_context": True,
                    "c2f_coarse_target_mode": "uniform",
                    "c2f_temporal_upsample_mode": "linear",
                    "learning_rate": 2e-4,
                },
            }
            for s in (0.5, 0.8, 1.0, 1.2)
        ],

        # ---------------------------------------------------------------------
        # (5) CNN vs U-Net refiner 的公平对比
        # 固定 coarse checkpoint，只训练 refiner，除了 refiner_type 外尽量一致。
        # ---------------------------------------------------------------------
        *[
            {
                "group": "refiner_fair",
                "name": f"c2f_refiner_only_{rtype}_fair",
                "description": f"Fair comparison: frozen coarse + {rtype.upper()} refiner.",
                "overrides": {
                    "experiment_name": f"c2f_refiner_only_{rtype}_fair",
                    "c2f_train_stage": "refiner_only",
                    "c2f_pretrained_coarse_path": coarse_ckpt,
                    "c2f_freeze_coarse_in_refiner_only": True,
                    "c2f_refiner_type": rtype,
                    "c2f_refiner_base_channels": 64,
                    "c2f_refiner_depth": 3,
                    "c2f_refiner_residual_scale": 1.0,
                    "c2f_coarse_loss_weight": 0.0,
                    "c2f_base_loss_weight": 0.0,
                    "use_long_rollout_curriculum": True,
                    "rollout_train_steps": (20, 40, 80, 120, 160, 240),
                    "rollout_curriculum_boundaries": (0.0, 0.10, 0.20, 0.30, 0.45, 0.60),
                    "rollout_detach_context": True,
                    "c2f_coarse_target_mode": "uniform",
                    "c2f_temporal_upsample_mode": "linear",
                    "learning_rate": 2e-4,
                },
            }
            for rtype in ("cnn", "unet")
        ],
    ]


# =============================================================================
# Config patching
# =============================================================================

def parse_value(raw: str) -> Any:
    """Parse CLI --set values.

    Examples:
        batch_size=8 -> int
        learning_rate=1e-4 -> float
        use_x=True -> bool
        rollout_train_steps=(240,) -> tuple
        experiment_name=my_exp -> string
    """
    text = raw.strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None

    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def parse_set_items(items: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got: {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in --set item: {item!r}")
        out[key] = parse_value(raw_value)
    return out


def format_config_value(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, tuple):
        if len(value) == 1:
            return f"({format_config_value(value[0])},)"
        return "(" + ", ".join(format_config_value(v) for v in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(format_config_value(v) for v in value) + "]"
    return repr(value)


def patch_config_text(text: str, overrides: Dict[str, Any]) -> str:
    """Patch dataclass default assignments in config text.

    It matches lines like:
        learning_rate: float = 5e-4
        rollout_train_steps: tuple = (20, 40, ...)
    and replaces only the default value.
    """
    patched = text
    missing: List[str] = []

    for key, value in overrides.items():
        value_text = format_config_value(value)

        # Preserve indentation, type annotation, and trailing comment.
        # Stop value capture before a trailing inline comment if present.
        pattern = re.compile(
            rf"^(\s*{re.escape(key)}\s*:\s*[^=\n]+?=\s*)([^#\n]*?)(\s*(?:#.*)?$)",
            flags=re.MULTILINE,
        )

        def repl(match: re.Match[str]) -> str:
            return f"{match.group(1)}{value_text}{match.group(3)}"

        patched_new, n = pattern.subn(repl, patched, count=1)
        if n == 0:
            missing.append(key)
        patched = patched_new

    if missing:
        raise KeyError(
            "The following config keys were not found in the config file: "
            + ", ".join(missing)
        )
    return patched


def apply_config(config_path: Path, overrides: Dict[str, Any]) -> None:
    text = config_path.read_text(encoding="utf-8")
    patched = patch_config_text(text, overrides)
    config_path.write_text(patched, encoding="utf-8")


# =============================================================================
# Running commands
# =============================================================================

def run_command(
    cmd: Sequence[str],
    cwd: Path,
    log_path: Path,
    dry_run: bool = False,
) -> int:
    print("    CMD:", " ".join(str(c) for c in cmd))
    print("    LOG:", str(log_path))
    if dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as f:
        f.write("# Command: " + " ".join(str(c) for c in cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        return proc.wait()


def checkpoint_path_from_overrides(overrides: Dict[str, Any]) -> str:
    ckpt_dir = str(overrides.get("checkpoint_dir", "./checkpoints"))
    exp_name = str(overrides["experiment_name"])
    return os.path.join(ckpt_dir, f"{exp_name}.pt")


def should_include(exp: Dict[str, Any], only: Sequence[str], skip: Sequence[str]) -> bool:
    if only:
        names = set(only)
        if exp["group"] not in names and exp["name"] not in names:
            return False
    if skip:
        names = set(skip)
        if exp["group"] in names or exp["name"] in names:
            return False
    return True


def summarize_experiments(experiments: Sequence[Dict[str, Any]]) -> None:
    print("\nSelected experiments:")
    for i, exp in enumerate(experiments, start=1):
        print(f"  {i:02d}. [{exp['group']}] {exp['name']} - {exp.get('description', '')}")
    print("")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch train/validate Coarse-to-fine experiments by patching a config file."
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory. Commands are executed from this directory.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sea_surface_rollout_config_coarse_to_fine.py",
        help="Path to sea_surface_rollout_config.py.",
    )
    parser.add_argument(
        "--train",
        type=str,
        default="scripts/coarse_to_fine/train_sea_surface_rollout_coarse_to_fine.py",
        help="Path to training script.",
    )
    parser.add_argument(
        "--validate",
        type=str,
        default="scripts/coarse_to_fine/validate_sea_surface_rollout_coarse_to_fine.py",
        help="Path to validation script.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use. Default: current Python.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val"],
        choices=["val", "test"],
        help="Validation splits to run after each training.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help=(
            "Run only selected groups or experiment names. "
            "Groups: coarse_full240, joint_detach, coarse_loss_scan, residual_scale_scan, refiner_fair."
        ),
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Skip selected groups or experiment names.",
    )
    parser.add_argument(
        "--set",
        dest="global_sets",
        action="append",
        default=[],
        help="Extra config override applied to all experiments, in KEY=VALUE form. Can be repeated.",
    )
    parser.add_argument(
        "--coarse-ckpt",
        type=str,
        default="./checkpoints/c2f_coarse_full240_uniform.pt",
        help=(
            "Pretrained coarse checkpoint path used by downstream experiments. "
            "Default is the checkpoint produced by experiment (1)."
        ),
    )
    parser.add_argument(
        "--skip-train-if-checkpoint-exists",
        action="store_true",
        help="Skip training for an experiment when its checkpoint already exists.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation; do not train.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print patched config values and commands without running anything.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining experiments when one train/validate command fails.",
    )
    parser.add_argument(
        "--keep-last-config",
        action="store_true",
        help="Do not restore original config at the end. Useful for debugging only.",
    )
    parser.add_argument(
        "--batch-log-dir",
        type=str,
        default="./checkpoints/batch_logs",
        help="Directory for captured stdout/stderr logs from this runner.",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = (project_root / args.config).resolve()
    train_script = (project_root / args.train).resolve()
    validate_script = (project_root / args.validate).resolve()
    batch_log_dir = (project_root / args.batch_log_dir).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not train_script.exists():
        raise FileNotFoundError(f"Train script not found: {train_script}")
    if not validate_script.exists():
        raise FileNotFoundError(f"Validate script not found: {validate_script}")

    global_overrides = parse_set_items(args.global_sets)

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = config_path.with_suffix(config_path.suffix + f".bak_batch_{timestamp}")
    original_text = config_path.read_text(encoding="utf-8")
    backup_path.write_text(original_text, encoding="utf-8")

    print(f"Project root : {project_root}")
    print(f"Config file  : {config_path}")
    print(f"Backup file  : {backup_path}")
    print(f"Train script : {train_script}")
    print(f"Val script   : {validate_script}")

    all_exps = default_experiments(coarse_ckpt=args.coarse_ckpt)
    experiments = [
        exp for exp in all_exps
        if should_include(exp, only=args.only, skip=args.skip)
    ]
    summarize_experiments(experiments)

    overall_status = 0

    try:
        for exp_index, exp in enumerate(experiments, start=1):
            exp_name = str(exp["name"])
            overrides = copy.deepcopy(exp["overrides"])
            overrides.update(global_overrides)

            ckpt_path = project_root / checkpoint_path_from_overrides(overrides)

            print("=" * 100)
            print(f"[{exp_index}/{len(experiments)}] {exp_name}")
            print(exp.get("description", ""))
            print("Overrides:")
            for k in sorted(overrides):
                print(f"  {k} = {overrides[k]!r}")
            print(f"Expected checkpoint: {ckpt_path}")

            apply_config(config_path, overrides)

            exp_log_dir = batch_log_dir / f"{timestamp}_{exp_name}"
            exp_log_dir.mkdir(parents=True, exist_ok=True)

            # Training
            if args.validate_only:
                print("  Skipping training because --validate-only was set.")
            elif args.skip_train_if_checkpoint_exists and ckpt_path.exists():
                print("  Skipping training because checkpoint exists.")
            else:
                code = run_command(
                    [args.python, str(train_script)],
                    cwd=project_root,
                    log_path=exp_log_dir / "train_stdout.log",
                    dry_run=args.dry_run,
                )
                if code != 0:
                    print(f"  Training failed for {exp_name} with return code {code}.")
                    overall_status = code or 1
                    if not args.continue_on_error:
                        return overall_status
                    continue

            # Validation
            for split in args.splits:
                code = run_command(
                    [args.python, str(validate_script), split],
                    cwd=project_root,
                    log_path=exp_log_dir / f"validate_{split}_stdout.log",
                    dry_run=args.dry_run,
                )
                if code != 0:
                    print(f"  Validation split={split} failed for {exp_name} with return code {code}.")
                    overall_status = code or 1
                    if not args.continue_on_error:
                        return overall_status

        return overall_status

    finally:
        if args.keep_last_config:
            print(f"Keeping last patched config: {config_path}")
            print(f"Original config backup remains at: {backup_path}")
        else:
            config_path.write_text(original_text, encoding="utf-8")
            print(f"Restored original config from backup: {backup_path}")


if __name__ == "__main__":
    raise SystemExit(main())
