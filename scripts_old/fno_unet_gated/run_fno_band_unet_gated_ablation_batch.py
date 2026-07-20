#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch ablation runner for FNO + Fourier-band U-Net + gated residual model.

Recommended location:
    neuraloperator/scripts/run_fno_band_unet_gated_ablation_batch.py

Default target files:
    config/sea_surface_rollout_config_fno_band_unet_gated.py
    scripts/train_sea_surface_rollout_fno_band_unet_gated.py
    scripts/validate_sea_surface_rollout_per_mat_fno_band_unet_gated.py

Examples:
    # Show experiments and commands only
    python scripts/fno_unet_gated/run_fno_band_unet_gated_ablation_batch.py --dry-run

    # Run the most important structure ablations
    python scripts/fno_unet_gated/run_fno_band_unet_gated_ablation_batch.py --only core
    python scripts/fno_unet_gated/run_fno_band_unet_gated_ablation_batch.py --only structure


    # Run all ablations and validate on val
    python scripts/fno_unet_gated/run_fno_band_unet_gated_ablation_batch.py --only all

    # Run training + val + test
    python scripts/fno_unet_gated/run_fno_band_unet_gated_ablation_batch.py --only core --run-test

    # Skip experiments whose checkpoint already exists
    python scripts/fno_unet_gated/run_fno_band_unet_gated_ablation_batch.py --only all --skip-existing 
    python scripts/fno_unet_gated/run_fno_band_unet_gated_ablation_batch.py --only structure --skip-existing


    # Override common settings for all experiments
    python scripts/fno_unet_gated/run_fno_band_unet_gated_ablation_batch.py --only core --set n_epochs=80 --set early_stop_patience=20
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


# =============================================================================
# Project path discovery
# =============================================================================

THIS_FILE = Path(__file__).resolve()
if THIS_FILE.parent.name == "scripts":
    PROJECT_ROOT = THIS_FILE.parent.parent
else:
    PROJECT_ROOT = THIS_FILE.parent.parent.parent

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "sea_surface_rollout_config_fno_band_unet_gated.py"
DEFAULT_TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "fno_unet_gated" / "train_sea_surface_rollout_fno_band_unet_gated.py"
DEFAULT_VALIDATE_SCRIPT = PROJECT_ROOT / "scripts" / "fno_unet_gated" / "validate_sea_surface_rollout_fno_band_unet_gated.py"


# =============================================================================
# Experiment definitions
# =============================================================================

@dataclass(frozen=True)
class Experiment:
    name: str
    group: str
    params: Dict[str, Any]
    note: str = ""


# Common best settings from current log:
#   grad weight = 0.05
#   residual_scale = 1.0
#   gated residual enabled
#   learnable_gate band fusion
#   periodic padding
COMMON_BEST: Dict[str, Any] = {
    "model_arch": "fno_band_unet_gated_decoder",
    "fno_unet_residual_scale": 1.0,
    "fno_unet_use_residual": True,
    "fno_unet_use_context": True,
    "fno_unet_use_gated_residual": True,
    "fno_unet_gate_bias_init": 0.0,
    "fno_unet_band_branch_base_channels": 16,
    "fno_unet_band_cutoffs": (0.33, 0.67),
    "fno_unet_band_transition_width": 0.04,
    "fno_unet_band_fusion": "learnable_gate",
    "fno_unet_padding_mode": "periodic",
    "use_spatial_gradient_loss": True,
    "spatial_gradient_loss_weight": 0.05,
    "use_segment_weighting": True,
    "segment_weight_type": "linear",
    "segment_weight_min": 1.0,
    "segment_weight_max": 2.5,
    "normalize_segment_weights": True,
    "use_within_chunk_temporal_weighting": False,
    "rollout_train_steps": ( 30, 60, 120, 180, 240, 300),
    "rollout_curriculum_boundaries": (0.0, 0.10, 0.20, 0.30, 0.45, 0.60),
    "rollout_steps": 300,
    "rollout_detach_context": False,
    # 100 is consistent with existing log. With patience=20, most runs still reach the late plateau,
    # but you can override by --set early_stop_patience=20.
    "n_epochs": 100,
    "early_stop_patience": 100,
}


EXPERIMENTS: List[Experiment] = [
    # -------------------------------------------------------------------------
    # Baseline reproducibility
    # -------------------------------------------------------------------------
    Experiment(
        name="bandgated_base_rerun",
        group="core",
        params={**COMMON_BEST},
        note="Re-run current best model to check reproducibility.",
    ),

    # -------------------------------------------------------------------------
    # Structure ablations: quantify each architectural contribution
    # -------------------------------------------------------------------------
    Experiment(
        name="ab_no_gated_residual",
        group="core",
        params={**COMMON_BEST, "fno_unet_use_gated_residual": False},
        note="Ablate gated residual: pred = coarse + residual_scale * residual.",
    ),
    Experiment(
        name="ab_gate_bias_neg1",
        group="structure",
        params={**COMMON_BEST, "fno_unet_gate_bias_init": -1.0},
        note="More conservative residual gate initialization: sigmoid(-1)=0.269.",
    ),
    Experiment(
        name="ab_gate_bias_pos1",
        group="structure",
        params={**COMMON_BEST, "fno_unet_gate_bias_init": 1.0},
        note="More aggressive residual gate initialization: sigmoid(1)=0.731.",
    ),
    Experiment(
        name="ab_padding_zero",
        group="core",
        params={**COMMON_BEST, "fno_unet_padding_mode": "zeros"},
        note="Ablate periodic padding; compare with zero padding.",
    ),
    Experiment(
        name="ab_band_fusion_sum",
        group="core",
        params={**COMMON_BEST, "fno_unet_band_fusion": "sum"},
        note="Ablate learnable band gate; directly sum low/mid/high residuals.",
    ),
    Experiment(
        name="ab_band_fusion_scalar",
        group="structure",
        params={**COMMON_BEST, "fno_unet_band_fusion": "learnable_scalar"},
        note="Use global learnable band weights instead of spatial-temporal band gate.",
    ),
    Experiment(
        name="ab_band_fusion_mean",
        group="structure",
        params={**COMMON_BEST, "fno_unet_band_fusion": "mean"},
        note="Average low/mid/high residuals.",
    ),
    Experiment(
        name="ab_band_base8",
        group="core",
        params={**COMMON_BEST, "fno_unet_band_branch_base_channels": 8},
        note="Reduce each band branch capacity; test whether current model is over-parameterized.",
    ),
    Experiment(
        name="ab_band_base24",
        group="structure",
        params={**COMMON_BEST, "fno_unet_band_branch_base_channels": 24},
        note="Increase band branch capacity; test whether current model is under-parameterized.",
    ),
    Experiment(
        name="ab_band_cutoffs_025_060",
        group="structure",
        params={**COMMON_BEST, "fno_unet_band_cutoffs": (0.25, 0.60)},
        note="Move more content into high band; useful if high-k spectrum is underpredicted.",
    ),
    Experiment(
        name="ab_band_cutoffs_040_075",
        group="structure",
        params={**COMMON_BEST, "fno_unet_band_cutoffs": (0.40, 0.75)},
        note="Move more content into low/mid bands; useful if high band is too noisy.",
    )

]


GROUPS = {
    "core": {"core"},
    "structure": {"core", "structure"},
    "all": {"core", "structure"},
}


# =============================================================================
# Config patching helpers
# =============================================================================

def py_repr(value: Any) -> str:
    """Return a stable Python literal for config assignment."""
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, tuple):
        inner = ", ".join(py_repr(v) for v in value)
        if len(value) == 1:
            inner += ","
        return f"({inner})"
    if isinstance(value, list):
        return "[" + ", ".join(py_repr(v) for v in value) + "]"
    return repr(value)


def parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Override must be KEY=VALUE, got: {text}")
    key, value_text = text.split("=", 1)
    key = key.strip()
    value_text = value_text.strip()
    if not key:
        raise ValueError(f"Empty override key in: {text}")
    try:
        value = ast.literal_eval(value_text)
    except Exception:
        lowered = value_text.lower()
        if lowered == "true":
            value = True
        elif lowered == "false":
            value = False
        else:
            value = value_text
    return key, value


def patch_config_text(original: str, params: Dict[str, Any]) -> str:
    text = original
    missing: List[str] = []

    for key, value in params.items():
        new_value = py_repr(value)

        # Match dataclass lines like:
        #   fno_unet_residual_scale: float = 1.0
        #   rollout_train_steps: tuple = (20, 40, ...)
        pattern_typed = re.compile(
            rf"^(\s*{re.escape(key)}\s*:\s*[^=\n]+?=\s*)(.*?)(\s*(?:#.*)?)$",
            flags=re.MULTILINE,
        )
        text_new, n = pattern_typed.subn(rf"\g<1>{new_value}\g<3>", text, count=1)

        if n == 0:
            # Match untyped assignment lines:
            #   rollout_train_steps = (...)
            pattern_plain = re.compile(
                rf"^(\s*{re.escape(key)}\s*=\s*)(.*?)(\s*(?:#.*)?)$",
                flags=re.MULTILINE,
            )
            text_new, n = pattern_plain.subn(rf"\g<1>{new_value}\g<3>", text, count=1)

        if n == 0:
            missing.append(key)
        else:
            text = text_new

    if missing:
        raise KeyError(
            "The following keys were not found in config file: "
            + ", ".join(missing)
            + "\nCheck whether your config file contains these dataclass fields."
        )

    return text


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# =============================================================================
# Running helpers
# =============================================================================

def run_cmd(cmd: List[str], cwd: Path, dry_run: bool) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def select_experiments(only: str, names: Iterable[str] | None = None) -> List[Experiment]:
    if names:
        wanted = set(names)
        selected = [e for e in EXPERIMENTS if e.name in wanted]
        missing = sorted(wanted - {e.name for e in selected})
        if missing:
            raise ValueError("Unknown experiment names: " + ", ".join(missing))
        return selected

    allowed_groups = GROUPS.get(only)
    if allowed_groups is None:
        raise ValueError(f"Unsupported --only={only}. Choices: {sorted(GROUPS)}")
    return [e for e in EXPERIMENTS if e.group in allowed_groups]


def checkpoint_exists(config_text: str, experiment_name: str, project_root: Path) -> bool:
    # Lightweight default assumption matching config property:
    # checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}.pt")
    m = re.search(r'^\s*checkpoint_dir\s*:\s*[^=]+=\s*([\'"].*?[\'"])', config_text, re.M)
    if m:
        try:
            checkpoint_dir = ast.literal_eval(m.group(1))
        except Exception:
            checkpoint_dir = "./checkpoints/fno_band_unet_gated_ablation"
    else:
        checkpoint_dir = "./checkpoints/fno_band_unet_gated_ablation"

    ckpt = project_root / checkpoint_dir / f"{experiment_name}.pt"
    return ckpt.exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        default="core",
        choices=sorted(GROUPS),
        help=(
            "core: baseline + most important ablations; "
            "structure: all structure ablations; "
            "all: everything."
        ),
    )
    parser.add_argument(
        "--exp",
        nargs="*",
        default=None,
        help="Run selected experiment names only. Overrides --only.",
    )
    parser.add_argument("--run-test", action="store_true", help="Run test validation after val.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and config changes only.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if checkpoints/<experiment_name>.pt exists.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config value for all experiments, e.g. --set n_epochs=80 --set early_stop_patience=20",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--validate-script", type=Path, default=DEFAULT_VALIDATE_SCRIPT)
    args = parser.parse_args()

    config_path = args.config.resolve()
    train_script = args.train_script.resolve()
    validate_script = args.validate_script.resolve()

    for p in [config_path, train_script, validate_script]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    selected = select_experiments(args.only, args.exp)
    overrides = dict(parse_override(x) for x in args.overrides)

    original_config = read_text(config_path)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = config_path.with_suffix(config_path.suffix + f".bak_{timestamp}")

    print("=" * 88)
    print("Batch FNO-band-UNet-gated ablation runner")
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Config           : {config_path}")
    print(f"Train script     : {train_script}")
    print(f"Validate script  : {validate_script}")
    print(f"Selected group   : {args.only}")
    print(f"Run test         : {args.run_test}")
    print(f"Dry run          : {args.dry_run}")
    print(f"Skip existing    : {args.skip_existing}")
    if overrides:
        print(f"Global overrides : {overrides}")
    print("=" * 88)

    print("\nExperiments:")
    for i, exp in enumerate(selected, 1):
        print(f"{i:02d}. {exp.name:32s} [{exp.group}] {exp.note}")

    if args.dry_run:
        print("\nDry-run mode: no file will be modified and no command will be executed.")

    if not args.dry_run:
        shutil.copy2(config_path, backup_path)
        print(f"\nBacked up original config to: {backup_path}")

    try:
        for i, exp in enumerate(selected, 1):
            params = dict(exp.params)
            params.update(overrides)
            params["experiment_name"] = exp.name

            print("\n" + "=" * 88)
            print(f"[{i}/{len(selected)}] Experiment: {exp.name}")
            print(f"Note: {exp.note}")
            print("Changed params:")
            for k in sorted(params):
                if k == "experiment_name" or k in exp.params or k in overrides:
                    print(f"  {k} = {py_repr(params[k])}")

            if args.skip_existing and checkpoint_exists(original_config, exp.name, PROJECT_ROOT):
                print(f"Skip existing checkpoint for experiment: {exp.name}")
                continue

            patched = patch_config_text(original_config, params)

            if not args.dry_run:
                write_text(config_path, patched)

            run_cmd([sys.executable, str(train_script)], cwd=PROJECT_ROOT, dry_run=args.dry_run)
            run_cmd([sys.executable, str(validate_script), "val"], cwd=PROJECT_ROOT, dry_run=args.dry_run)

            if args.run_test:
                run_cmd([sys.executable, str(validate_script), "test"], cwd=PROJECT_ROOT, dry_run=args.dry_run)

    finally:
        if not args.dry_run:
            write_text(config_path, original_config)
            print("\nRestored original config.")
            print(f"Backup kept at: {backup_path}")


if __name__ == "__main__":
    main()
