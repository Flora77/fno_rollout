#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for FNO + U-Net Decoder ablation experiments.

默认实验：
  1) spatial_gradient_loss_weight 扫描: 0.02 / 0.05 / 0.10，固定 residual_scale=1.0
  2) residual_scale 围绕 1.0 扫描: 0.75 / 1.0 / 1.25，固定 grad_weight=0.05

用法示例：
  python scripts/run_fno_unet_ablation_batch.py --only all
  python scripts/run_fno_unet_ablation_batch.py --only grad
  python scripts/run_fno_unet_ablation_batch.py --only residual
  python scripts/run_fno_unet_ablation_batch.py --dry-run
  python scripts/run_fno_unet_ablation_batch.py --run-test
  python scripts/run_fno_unet_ablation_batch.py --set n_epochs=80 --set batch_size=8

脚本逻辑：
  - 自动备份 config/sea_surface_rollout_config_fno_unet.py
  - 每个实验运行前临时修改 dataclass 中的默认参数
  - 依次执行 train 和 validate(val)，可选 validate(test)
  - 全部结束后默认恢复原始 config 文件

注意：
  训练/验证脚本本身不需要改。validate 脚本加载 checkpoint 前需要根据 config.experiment_name
  找到对应 .pt，所以每个实验验证前必须保持 config 中 experiment_name 与训练时一致。
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclasses.dataclass(frozen=True)
class Experiment:
    name: str
    group: str
    overrides: Dict[str, Any]


# =============================================================================
# 直接在这里设置实验表
# =============================================================================
# 说明：
# - scale=1.0, grad_weight=0.05 是当前最优基准，两个扫描共用这一组，避免重复训练。
# - 如果你想完整列出 residual_scale=1.0，也可以把 baseline 视为 residual scan 的中间点。
EXPERIMENTS: List[Experiment] = [
    # spatial_gradient_loss_weight 扫描，固定 residual_scale = 1.0
    Experiment(
        name="fno_unet_gradw002_scale1p00",
        group="grad",
        overrides={
            "fno_unet_residual_scale": 1.0,
            "use_spatial_gradient_loss": True,
            "spatial_gradient_loss_weight": 0.02,
        },
    ),
    Experiment(
        name="fno_unet_gradw005_scale1p00_baseline",
        group="grad,residual",
        overrides={
            "fno_unet_residual_scale": 1.0,
            "use_spatial_gradient_loss": True,
            "spatial_gradient_loss_weight": 0.05,
        },
    ),
    Experiment(
        name="fno_unet_gradw010_scale1p00",
        group="grad",
        overrides={
            "fno_unet_residual_scale": 1.0,
            "use_spatial_gradient_loss": True,
            "spatial_gradient_loss_weight": 0.10,
        },
    ),

    # residual_scale 围绕 1.0 扫描，固定 spatial_gradient_loss_weight = 0.05
    Experiment(
        name="fno_unet_res075_gradw005",
        group="residual",
        overrides={
            "fno_unet_residual_scale": 0.75,
            "use_spatial_gradient_loss": True,
            "spatial_gradient_loss_weight": 0.05,
        },
    ),
    Experiment(
        name="fno_unet_res125_gradw005",
        group="residual",
        overrides={
            "fno_unet_residual_scale": 1.25,
            "use_spatial_gradient_loss": True,
            "spatial_gradient_loss_weight": 0.05,
        },
    ),
]


# 所有实验共同固定的参数。这里尽量只固定和 FNO+U-Net Decoder 实验相关的关键项。
COMMON_OVERRIDES: Dict[str, Any] = {
    "model_arch": "fno_unet_decoder",
    "fno_unet_base_channels": 32,
    "fno_unet_depth": 3,
    "fno_unet_decoder_dropout": 0.0,
    "fno_unet_use_context": True,
    "fno_unet_use_residual": True,

    # 仍然使用你当前日志中的 rollout curriculum，保证扫描实验只比较目标参数。
    "use_long_rollout_curriculum": True,
    "rollout_train_steps": (20, 40, 80, 120, 160, 240),
    "rollout_curriculum_boundaries": (0.0, 0.10, 0.20, 0.30, 0.45, 0.60),
    "rollout_steps": 240,
    "rollout_detach_context": False,

    # 保持训练稳定。若想节省时间，可在命令行加：--set early_stop_patience=20
    "early_stop_patience": 100,
}


# =============================================================================
# 工具函数
# =============================================================================
def parse_scalar_or_literal(text: str) -> Any:
    """Parse command-line values like True, 0.05, '(20,40)', './data/bimodal'."""
    s = text.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() == "none":
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def parse_key_value(items: Optional[Iterable[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not items:
        return out
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set value: {item!r}. Expected key=value.")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Invalid --set key in: {item!r}")
        out[k] = parse_scalar_or_literal(v)
    return out


def py_repr(value: Any) -> str:
    """Represent values as Python source literals for config dataclass defaults."""
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return repr(value)


def patch_config_text(text: str, overrides: Dict[str, Any]) -> str:
    """Patch annotated dataclass defaults, preserving inline comments when possible."""
    new_text = text
    for key, value in overrides.items():
        replacement_value = py_repr(value)
        # 匹配形如：    key: type = value  # optional comment
        pattern = re.compile(
            rf"^(?P<prefix>\s*{re.escape(key)}\s*:\s*[^=\n]+?=\s*)(?P<value>.*?)(?P<comment>\s*(?:#.*)?)$",
            flags=re.MULTILINE,
        )

        def repl(m: re.Match) -> str:
            comment = m.group("comment") or ""
            # 如果旧值末尾已经包含注释，非贪婪匹配有时会把空格吃进 value，这里统一格式化。
            return f"{m.group('prefix')}{replacement_value}{comment}"

        new_text, n = pattern.subn(repl, new_text, count=1)
        if n == 0:
            raise KeyError(
                f"Config field {key!r} not found. Please check the config file or remove this override."
            )
    return new_text


def patch_config_file(config_path: Path, overrides: Dict[str, Any]) -> None:
    text = config_path.read_text(encoding="utf-8")
    new_text = patch_config_text(text, overrides)
    config_path.write_text(new_text, encoding="utf-8")


def run_command(cmd: List[str], cwd: Path, log_file: Path, dry_run: bool = False) -> int:
    cmd_str = " ".join(cmd)
    header = f"\n{'=' * 100}\n$ {cmd_str}\nCWD: {cwd}\nTIME: {_dt.datetime.now().isoformat(timespec='seconds')}\n{'=' * 100}\n"
    print(header, flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(header)

    if dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    with log_file.open("a", encoding="utf-8") as f:
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
    return int(proc.wait())


def select_experiments(which: str) -> List[Experiment]:
    which = which.lower()
    if which == "all":
        return EXPERIMENTS
    if which in {"grad", "residual"}:
        return [e for e in EXPERIMENTS if which in {g.strip() for g in e.group.split(",")}]
    raise ValueError(f"Unsupported --only={which!r}")


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch train/validate FNO + U-Net Decoder ablation experiments.")
    p.add_argument("--project-root", type=str, default=".", help="Project root containing config/, scripts/, neuralop/.")
    p.add_argument("--config", type=str, default="config/sea_surface_rollout_config_fno_unet.py")
    p.add_argument("--train-script", type=str, default="scripts/train_sea_surface_rollout_fno_unet_decoder.py")
    p.add_argument("--validate-script", type=str, default="scripts/validate_sea_surface_rollout_per_mat_fno_unet_decoder.py")
    p.add_argument("--python", type=str, default=sys.executable, help="Python executable used to launch train/validate.")

    p.add_argument("--only", choices=["all", "grad", "residual"], default="all", help="Run all experiments or one group.")
    p.add_argument("--start-index", type=int, default=0, help="Start from this index after filtering experiments.")
    p.add_argument("--max-experiments", type=int, default=None, help="Run at most this many experiments after start-index.")

    p.add_argument("--skip-train", action="store_true", help="Only run validation for existing checkpoints.")
    p.add_argument("--skip-val", action="store_true", help="Skip validation on val split.")
    p.add_argument("--run-test", action="store_true", help="Also validate on test split after val.")
    p.add_argument("--dry-run", action="store_true", help="Only print commands and patched overrides; do not run train/validate.")
    p.add_argument("--continue-on-error", action="store_true", help="Continue next experiment if one command fails.")
    p.add_argument("--no-restore-config", action="store_true", help="Do not restore the original config at the end.")

    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a config parameter for all experiments. Examples: "
            "--set n_epochs=80 --set batch_size=8 --set data_root='./data/bimodal'"
        ),
    )
    return p


# =============================================================================
# 主流程
# =============================================================================
def main() -> None:
    args = make_argparser().parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = (project_root / args.config).resolve()
    train_script = (project_root / args.train_script).resolve()
    validate_script = (project_root / args.validate_script).resolve()

    for pth, label in [(config_path, "config"), (train_script, "train script"), (validate_script, "validate script")]:
        if not pth.exists():
            raise FileNotFoundError(f"Cannot find {label}: {pth}")

    global_overrides = parse_key_value(args.set)

    selected = select_experiments(args.only)
    selected = selected[int(args.start_index):]
    if args.max_experiments is not None:
        selected = selected[: int(args.max_experiments)]
    if not selected:
        print("No experiments selected.")
        return

    batch_stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_log = project_root / "checkpoints" / "batch_logs" / f"fno_unet_ablation_batch_{batch_stamp}.log"
    backup_path = config_path.with_suffix(config_path.suffix + f".batch_backup_{batch_stamp}")

    original_text = config_path.read_text(encoding="utf-8")
    backup_path.write_text(original_text, encoding="utf-8")

    print(f"Project root : {project_root}")
    print(f"Config file  : {config_path}")
    print(f"Backup file  : {backup_path}")
    print(f"Batch log    : {batch_log}")
    print("Selected experiments:")
    for i, exp in enumerate(selected):
        print(f"  [{i}] {exp.name} | group={exp.group} | overrides={exp.overrides}")

    failed: List[str] = []
    try:
        for i, exp in enumerate(selected):
            overrides: Dict[str, Any] = {}
            overrides.update(COMMON_OVERRIDES)
            overrides.update(exp.overrides)
            overrides["experiment_name"] = exp.name
            overrides.update(global_overrides)

            print(f"\n\n########## Experiment {i + 1}/{len(selected)}: {exp.name} ##########")
            print("Config overrides:")
            for k, v in overrides.items():
                print(f"  {k} = {v!r}")

            # 每次从原始 config 开始 patch，避免上一个实验残留参数。
            config_path.write_text(original_text, encoding="utf-8")
            patch_config_file(config_path, overrides)

            commands: List[List[str]] = []
            if not args.skip_train:
                commands.append([args.python, str(train_script)])
            if not args.skip_val:
                commands.append([args.python, str(validate_script), "val"])
            if args.run_test:
                commands.append([args.python, str(validate_script), "test"])

            for cmd in commands:
                ret = run_command(cmd, cwd=project_root, log_file=batch_log, dry_run=args.dry_run)
                if ret != 0:
                    msg = f"Experiment {exp.name} failed: command returned {ret}: {' '.join(cmd)}"
                    print("\nERROR:", msg)
                    failed.append(msg)
                    if not args.continue_on_error:
                        raise RuntimeError(msg)
                    break

    finally:
        if args.no_restore_config:
            print(f"\nConfig not restored because --no-restore-config was set: {config_path}")
        else:
            config_path.write_text(original_text, encoding="utf-8")
            print(f"\nRestored original config: {config_path}")
            print(f"Backup remains at: {backup_path}")

    if failed:
        print("\nSome experiments failed:")
        for msg in failed:
            print(" -", msg)
        raise SystemExit(1)

    print("\nAll selected experiments completed.")
    print(f"Batch log: {batch_log}")


if __name__ == "__main__":
    main()
