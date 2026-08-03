"""
Batch runner for ablation experiments using the best hyperparameter setting.

Best setting from hyperparameter sensitivity analysis:
    n_modes = (28, 28)
    hidden_channels = 32
    lifting_channels = projection_channels = 64
    n_layers = 4

This script runs A2/A4/A5/A6 by default because A1 has already been trained,
and A3/A7 will be added later.

Why this script patches the config file:
    The existing train script may only support command-line overrides for basic
    model hyperparameters. Ablation flags such as rollout_train_steps,
    use_spatial_gradient_loss, and use_segment_weighting may not be accepted as
    CLI arguments. Therefore, this batch script temporarily patches the config
    dataclass file before each training run, launches train/validate, and restores
    the original config file at the end.

Recommended location:
    NEURALOPERATOR/scripts/fno_unet_log/batch_ablation_best_hparam.py

Typical usage from project root:
    python scripts/fno_unet_log/batch_ablation_best_hparam.py

Dry run:
    python scripts/fno_unet_log/batch_ablation_best_hparam.py --dry_run
    python scripts/fno_unet_log/batch_ablation_best_hparam.py --include_A1 --skip_existing --dry_run
"""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class AblationExperiment:
    ablation_id: str
    title: str
    description: str
    overrides: Dict[str, Any] = field(default_factory=dict)

    @property
    def experiment_name(self) -> str:
        return self.ablation_id + "_" + re.sub(r"[^A-Za-z0-9]+", "_", self.title).strip("_").lower()


def default_ablation_experiments(include_a1: bool = False) -> List[AblationExperiment]:
    """Return ablation experiments to run.

    A1 is optional because the user has already completed it.
    A3 and A7 are intentionally excluded and can be added later.
    """
    curriculum_overrides = {
        "use_long_rollout_curriculum": True,
        "rollout_train_steps": (30, 60, 120, 180, 240, 300),
        "rollout_curriculum_boundaries": (0.0, 0.1, 0.2, 0.3, 0.45, 0.6),
        "use_spatial_gradient_loss": True,
        "use_segment_weighting": True,
    }

    rows: List[AblationExperiment] = []
    if include_a1:
        rows.append(
            AblationExperiment(
                "A1",
                "curriculum_rollout",
                "Baseline: curriculum rollout + spatial gradient loss + segment weighting.",
                dict(curriculum_overrides),
            )
        )

    rows += [

        AblationExperiment(
            "A7",
            "fno_direct60to60",
            "fno direct60to60 no rollout.",
            {
                **curriculum_overrides,
                "use_segment_weighting": False,
                "input_steps": 60,
                "output_steps": 60,
                "rollout_steps": 60,
                "rollout_train_steps": (60,),
                "rollout_curriculum_boundaries": (0.0,),
                "use_long_rollout_curriculum":  False,
                "use_spatial_gradient_loss": True,
            },
        ),

        AblationExperiment(
            "A2",
            "no_curriculum",
            "Remove curriculum schedule; train directly with full 300-frame rollout.",
            {
                **curriculum_overrides,
                "use_long_rollout_curriculum": True,
                "rollout_train_steps": (300,),
                "rollout_curriculum_boundaries": (0.0,),
            },
        ),
        AblationExperiment(
            "A4",
            "no_curriculum_rollout",
            "Only train with one 30-frame chunk; validation/test still use 300-frame autoregressive rollout.",
            {
                **curriculum_overrides,
                "use_long_rollout_curriculum": True,
                "rollout_train_steps": (30,),
                "rollout_curriculum_boundaries": (0.0,),
            },
        ),
        AblationExperiment(
            "A5",
            "no_spatial_gradient_loss",
            "Remove spatial gradient loss.",
            {
                **curriculum_overrides,
                "use_spatial_gradient_loss": False,
            },
        ),
        AblationExperiment(
            "A6",
            "no_segment_weighting",
            "Remove segment weighting in rollout loss.",
            {
                **curriculum_overrides,
                "use_segment_weighting": False,
            },
        ),


    ]
    return rows


def py_literal(value: Any) -> str:
    """Convert Python values to code literals used in the config file."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, tuple):
        if len(value) == 1:
            return f"({repr(value[0])},)"
        return "(" + ", ".join(repr(v) for v in value) + ")"
    if isinstance(value, list):
        return "(" + ", ".join(repr(v) for v in value) + ")"
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def patch_config_text(original_text: str, overrides: Dict[str, Any]) -> str:
    """Patch dataclass assignment lines like `name: type = value`.

    Raises ValueError if any requested key is not found to avoid silently running
    with wrong settings.
    """
    text = original_text
    missing: List[str] = []
    for key, value in overrides.items():
        literal = py_literal(value)
        # Match indentation + field name + annotation up to '='.
        # Preserve annotation, replace only the assigned value.
        pattern = re.compile(rf"^(\s*{re.escape(key)}\s*:\s*[^=\n]+?=\s*).*$", re.MULTILINE)
        if not pattern.search(text):
            missing.append(key)
            continue
        text = pattern.sub(rf"\g<1>{literal}", text, count=1)
    if missing:
        raise ValueError(f"Could not find these config fields to patch: {missing}")
    return text


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def shell_join(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_command(cmd: Sequence[str], cwd: Path | None, dry_run: bool) -> Tuple[int, float]:
    print("\n" + "=" * 100)
    print(shell_join(cmd))
    print("=" * 100)
    if dry_run:
        return 0, 0.0
    start = time.perf_counter()
    completed = subprocess.run(list(cmd), cwd=str(cwd) if cwd is not None else None)
    elapsed = time.perf_counter() - start
    return int(completed.returncode), float(elapsed)


def append_status_csv(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_grid_csv(path: Path, experiments: Iterable[AblationExperiment], base_overrides: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "experiment_name",
        "ablation_id",
        "title",
        "description",
        "n_modes",
        "hidden_channels",
        "lifting_channels",
        "projection_channels",
        "n_layers",
        "use_long_rollout_curriculum",
        "rollout_train_steps",
        "rollout_curriculum_boundaries",
        "use_spatial_gradient_loss",
        "use_segment_weighting",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exp in experiments:
            row = {**base_overrides, **exp.overrides}
            writer.writerow({
                "experiment_name": exp.experiment_name,
                "ablation_id": exp.ablation_id,
                "title": exp.title,
                "description": exp.description,
                "n_modes": str(tuple(row["n_modes"])),
                "hidden_channels": int(row["hidden_channels"]),
                "lifting_channels": int(row["lifting_channels"]),
                "projection_channels": int(row["projection_channels"]),
                "n_layers": int(row["n_layers"]),
                "use_long_rollout_curriculum": bool(row["use_long_rollout_curriculum"]),
                "rollout_train_steps": str(tuple(row["rollout_train_steps"])),
                "rollout_curriculum_boundaries": str(tuple(row["rollout_curriculum_boundaries"])),
                "use_spatial_gradient_loss": bool(row["use_spatial_gradient_loss"]),
                "use_segment_weighting": bool(row["use_segment_weighting"]),
            })


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    # Expected location: project_root/scripts/fno_unet_log/this_file.py
    project_root = script_dir.parents[1] if len(script_dir.parents) >= 2 else script_dir

    parser = argparse.ArgumentParser(description="Run ablation experiments using best FNO-U-Net hyperparameters.")
    parser.add_argument("--train_script", type=str, default=str(script_dir / "train_sea_surface_rollout_fno_unet_metrics.py"),
                        help="Path to the training script.")
    parser.add_argument("--validate_script", type=str, default=str(script_dir / "validate_sea_surface_rollout_fno_unet_metrics.py"),
                        help="Path to the validation/testing script. Per-mat metrics version is recommended.")
    parser.add_argument("--config_path", type=str, default=str(project_root / "config" / "sea_surface_rollout_config_fno_unet_aunet_gated.py"),
                        help="Path to the config dataclass file that will be temporarily patched.")
    parser.add_argument("--checkpoint_dir", type=str, default=str(project_root / "checkpoints" / "ablation_best_hparam"),
                        help="Directory for checkpoints, logs, mat files and CSV summaries.")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python executable used to launch child scripts.")
    parser.add_argument("--cwd", type=str, default=str(project_root),
                        help="Working directory for child processes. Default: project root.")

    # Best hyperparameters from sensitivity analysis.
    parser.add_argument("--n_modes", type=int, nargs=2, default=(28, 28), metavar=("KX", "KY"))
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--lifting_channels", type=int, default=64)
    parser.add_argument("--projection_channels", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)

    parser.add_argument("--model_arch", type=str, default="fno_unet_gated_decoder",
                        choices=["fno", "tfno", "fno_unet_gated_decoder", "fno_aunet_gated_decoder"])
    parser.add_argument("--use_gated_residual", type=int, default=0,
                        help="0 for final no-gate model, 1 for gated residual.")
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)

    parser.add_argument("--data_root", type=str, default=None,
                        help="Optional dataset root. If provided, it is patched into the config file.")
    parser.add_argument("--splits", type=str, nargs="+", default=["val"], choices=["val", "test"],
                        help="Run validation/test after training. Default: val.")
    parser.add_argument("--include_A1", action="store_true", help="Also train A1 baseline. Default skips A1.")
    parser.add_argument("--skip_train", action="store_true", help="Only run validation/test for existing checkpoints.")
    parser.add_argument("--skip_validate", action="store_true", help="Only train; do not run validation/test.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip training when checkpoint already exists.")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Continue remaining experiments if one command fails.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands and config patches without running.")
    parser.add_argument("--train_extra", type=str, default="",
                        help="Extra arguments appended to every train command.")
    parser.add_argument("--validate_extra", type=str, default="",
                        help="Extra arguments appended to every validate command.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    train_script = Path(args.train_script).resolve()
    validate_script = Path(args.validate_script).resolve()
    config_path = Path(args.config_path).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    cwd = Path(args.cwd).resolve()

    ensure_dir(checkpoint_dir)

    base_overrides: Dict[str, Any] = {
        "n_modes": tuple(int(v) for v in args.n_modes),
        "hidden_channels": int(args.hidden_channels),
        "lifting_channels": int(args.lifting_channels),
        "projection_channels": int(args.projection_channels),
        "n_layers": int(args.n_layers),
        "fno_unet_use_gated_residual": bool(args.use_gated_residual),
    }
    if args.data_root is not None:
        base_overrides["data_root"] = str(Path(args.data_root).resolve())

    experiments = default_ablation_experiments(include_a1=bool(args.include_A1))

    grid_csv = checkpoint_dir / "ablation_best_hparam_grid.csv"
    status_csv = checkpoint_dir / "ablation_best_hparam_run_status.csv"
    write_grid_csv(grid_csv, experiments, base_overrides)
    print(f"Ablation grid saved to: {grid_csv}")
    print(f"Number of experiments: {len(experiments)}")
    print(f"Best hyperparameters: n_modes={tuple(args.n_modes)}, H={args.hidden_channels}, L/P={args.lifting_channels}/{args.projection_channels}, L={args.n_layers}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    original_config_text = config_path.read_text(encoding="utf-8")

    train_extra = shlex.split(args.train_extra) if args.train_extra else []
    validate_extra = shlex.split(args.validate_extra) if args.validate_extra else []

    try:
        for idx, exp in enumerate(experiments, start=1):
            exp_name = exp.experiment_name
            checkpoint_path = checkpoint_dir / f"{exp_name}.pt"
            exp_overrides = {**base_overrides, **exp.overrides}

            print("\n" + "#" * 100)
            print(f"[{idx}/{len(experiments)}] {exp_name}")
            print(exp.description)
            for k in sorted(exp_overrides):
                print(f"  {k}: {exp_overrides[k]}")
            print("#" * 100)

            patched_config = patch_config_text(original_config_text, exp_overrides)
            if not args.dry_run:
                config_path.write_text(patched_config, encoding="utf-8")
            else:
                print(f"[dry-run] Would patch config: {config_path}")

            train_rc = None
            train_time = 0.0
            val_rcs: Dict[str, int] = {}
            val_times: Dict[str, float] = {}

            if not args.skip_train:
                if args.skip_existing and checkpoint_path.exists():
                    print(f"Skip training because checkpoint already exists: {checkpoint_path}")
                    train_rc = 0
                else:
                    train_cmd = [
                        args.python, str(train_script),
                        "--experiment_name", exp_name,
                        "--checkpoint_dir", str(checkpoint_dir),
                        "--model_arch", str(args.model_arch),
                        "--use_gated_residual", str(int(args.use_gated_residual)),
                        "--n_modes", str(args.n_modes[0]), str(args.n_modes[1]),
                        "--hidden_channels", str(args.hidden_channels),
                        "--lifting_channels", str(args.lifting_channels),
                        "--projection_channels", str(args.projection_channels),
                    ]
                    if args.n_epochs is not None:
                        train_cmd += ["--n_epochs", str(args.n_epochs)]
                    if args.batch_size is not None:
                        train_cmd += ["--batch_size", str(args.batch_size)]
                    if args.learning_rate is not None:
                        train_cmd += ["--learning_rate", str(args.learning_rate)]
                    train_cmd += train_extra
                    train_rc, train_time = run_command(train_cmd, cwd=cwd, dry_run=bool(args.dry_run))
                    if train_rc != 0 and not args.continue_on_error:
                        raise SystemExit(f"Training failed for {exp_name}, return code={train_rc}")

            if not args.skip_validate:
                for split in args.splits:
                    val_cmd = [
                        args.python, str(validate_script), split,
                        "--experiment_name", exp_name,
                        "--checkpoint_dir", str(checkpoint_dir),
                    ] + validate_extra
                    rc, elapsed = run_command(val_cmd, cwd=cwd, dry_run=bool(args.dry_run))
                    val_rcs[split] = rc
                    val_times[split] = elapsed
                    if rc != 0 and not args.continue_on_error:
                        raise SystemExit(f"Validation failed for {exp_name} split={split}, return code={rc}")

            append_status_csv(status_csv, {
                "experiment_name": exp_name,
                "ablation_id": exp.ablation_id,
                "title": exp.title,
                "train_return_code": train_rc if train_rc is not None else "",
                "train_elapsed_sec": f"{train_time:.3f}",
                "val_return_codes": str(val_rcs),
                "val_elapsed_sec": str({k: round(v, 3) for k, v in val_times.items()}),
                "checkpoint_path": str(checkpoint_path),
            })

    finally:
        if not args.dry_run:
            config_path.write_text(original_config_text, encoding="utf-8")
            print(f"\nRestored original config file: {config_path}")

    print("\nAll requested ablation experiments finished.")
    print(f"Grid CSV: {grid_csv}")
    print(f"Status CSV: {status_csv}")
    print(f"Summary CSVs are written by train/validate scripts under: {checkpoint_dir}")


if __name__ == "__main__":
    main()
