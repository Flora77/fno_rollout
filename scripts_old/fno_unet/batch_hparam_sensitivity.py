"""
Batch runner for FNO-U-Net key hyperparameter sensitivity analysis.

It launches training and validation experiments by passing command-line
config overrides to:
  - train_sea_surface_rollout_fno_unet_metrics.py
  - validate_sea_surface_rollout_fno_unet_metrics.py

The studied hyperparameters are:
  n_modes, hidden_channels, lifting_channels, projection_channels.

Default grid follows the table:
  mode       H   L/P
  (32,32)   32  64
  (28,28)   32  64
  (24,24)   32  64
  (20,20)   32  64
  (16,16)   32  64
  (32,32)   64  64
  (32,32)   48  64
  (32,32)   32  64
  (32,32)   16  64
  (32,32)   32  128
  (32,32)   32  96
  (32,32)   32  64
  (32,32)   32  32

Note: (32,32), H=32, L/P=64 is repeated in the above table as the common
baseline for different one-factor sweeps. By default this script trains this
baseline only once. Use --allow_duplicates to run all repeated rows separately.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class Experiment:
    group: str
    n_modes: Tuple[int, int]
    hidden_channels: int
    lp_channels: int

    @property
    def experiment_name(self) -> str:
        kx, ky = self.n_modes
        return f"{self.group}_m{kx}x{ky}_h{self.hidden_channels}_lp{self.lp_channels}"

    @property
    def key(self) -> Tuple[Tuple[int, int], int, int]:
        return self.n_modes, self.hidden_channels, self.lp_channels


def default_experiments(allow_duplicates: bool = False) -> List[Experiment]:
    """Return the experiment list for one-factor sensitivity analysis."""
    rows = [
        # Fourier mode sweep, baseline: (32,32), H=32, L/P=64
        Experiment("mode", (32, 32), 32, 64),
        Experiment("mode", (28, 28), 32, 64),
        Experiment("mode", (24, 24), 32, 64),
        Experiment("mode", (20, 20), 32, 64),
        Experiment("mode", (16, 16), 32, 64),
        # hidden channel sweep, baseline repeated in user's table
        Experiment("hidden", (32, 32), 64, 64),
        Experiment("hidden", (32, 32), 48, 64),
        Experiment("hidden", (32, 32), 32, 64),
        Experiment("hidden", (32, 32), 16, 64),
        # lifting/projection channel sweep, baseline repeated in user's table
        Experiment("lp", (32, 32), 32, 128),
        Experiment("lp", (32, 32), 32, 96),
        Experiment("lp", (32, 32), 32, 64),
        Experiment("lp", (32, 32), 32, 32),
    ]

    if allow_duplicates:
        return rows

    # Deduplicate the common baseline so it is trained only once.
    seen = set()
    unique: List[Experiment] = []
    for exp in rows:
        if exp.key in seen:
            continue
        seen.add(exp.key)
        unique.append(exp)
    return unique


def shell_join(cmd: Sequence[str]) -> str:
    """Readable command string, compatible with Windows and Linux logs."""
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_grid_csv(path: Path, experiments: Iterable[Experiment]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment_name", "group", "n_modes", "hidden_channels", "lifting_channels", "projection_channels"],
        )
        writer.writeheader()
        for exp in experiments:
            writer.writerow({
                "experiment_name": exp.experiment_name,
                "group": exp.group,
                "n_modes": str(tuple(exp.n_modes)),
                "hidden_channels": exp.hidden_channels,
                "lifting_channels": exp.lp_channels,
                "projection_channels": exp.lp_channels,
            })


def append_status_csv(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run FNO-U-Net hyperparameter sensitivity experiments.")

    parser.add_argument("--train_script", type=str, default=str(script_dir / "train_sea_surface_rollout_fno_unet_metrics.py"),
                        help="Path to the training script with CLI config overrides.")
    parser.add_argument("--validate_script", type=str, default=str(script_dir / "validate_sea_surface_rollout_fno_unet_metrics.py"),
                        help="Path to the validation/testing script.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/hparam_sensitivity",
                        help="Directory for checkpoints and summary CSV files.")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python executable used to launch child scripts.")
    parser.add_argument("--cwd", type=str, default=None,
                        help="Working directory. Default: parent directory of train_script.")

    parser.add_argument("--model_arch", type=str, default="fno_unet_gated_decoder",
                        choices=["fno", "tfno", "fno_unet_gated_decoder", "fno_aunet_gated_decoder"],
                        help="Model architecture passed to the training script.")
    parser.add_argument("--n_epochs", type=int, default=None, help="Override n_epochs for all experiments.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size for all experiments.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning_rate for all experiments.")
    parser.add_argument("--use_gated_residual", type=int, default=0,
                        help="0 for final no-gate model, 1 for gated residual.")

    parser.add_argument("--splits", type=str, nargs="+", default=["val"], choices=["val", "test"],
                        help="Run validation/test after training. Default: val.")
    parser.add_argument("--skip_train", action="store_true", help="Only run validation/test for existing checkpoints.")
    parser.add_argument("--skip_validate", action="store_true", help="Only train; do not run validation/test.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip training when checkpoint <checkpoint_dir>/<experiment_name>.pt already exists.")
    parser.add_argument("--allow_duplicates", action="store_true",
                        help="Run repeated baseline rows as separate experiments instead of deduplicating.")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Continue remaining experiments if one command fails.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")

    parser.add_argument("--train_extra", type=str, default="",
                        help="Extra arguments appended to every train command, e.g. '--num_workers 4'.")
    parser.add_argument("--validate_extra", type=str, default="",
                        help="Extra arguments appended to every validate command.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    train_script = Path(args.train_script).resolve()
    validate_script = Path(args.validate_script).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    cwd = Path(args.cwd).resolve() if args.cwd else train_script.parent

    ensure_dir(checkpoint_dir)
    experiments = default_experiments(allow_duplicates=bool(args.allow_duplicates))

    grid_csv = checkpoint_dir / "hparam_sensitivity_grid.csv"
    status_csv = checkpoint_dir / "hparam_sensitivity_run_status.csv"
    write_grid_csv(grid_csv, experiments)
    print(f"Experiment grid saved to: {grid_csv}")
    print(f"Number of experiments: {len(experiments)}")

    train_extra = shlex.split(args.train_extra) if args.train_extra else []
    validate_extra = shlex.split(args.validate_extra) if args.validate_extra else []

    for idx, exp in enumerate(experiments, start=1):
        exp_name = exp.experiment_name
        checkpoint_path = checkpoint_dir / f"{exp_name}.pt"
        print("\n" + "#" * 100)
        print(f"[{idx}/{len(experiments)}] {exp_name}")
        print(f"n_modes={exp.n_modes}, hidden_channels={exp.hidden_channels}, L/P={exp.lp_channels}")
        print("#" * 100)

        train_rc = None
        train_time = 0.0
        val_rcs = {}
        val_times = {}

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
                    "--n_modes", str(exp.n_modes[0]), str(exp.n_modes[1]),
                    "--hidden_channels", str(exp.hidden_channels),
                    "--lifting_channels", str(exp.lp_channels),
                    "--projection_channels", str(exp.lp_channels),
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
                val_rc, val_time = run_command(val_cmd, cwd=cwd, dry_run=bool(args.dry_run))
                val_rcs[split] = val_rc
                val_times[split] = val_time
                if val_rc != 0 and not args.continue_on_error:
                    raise SystemExit(f"Validation/test failed for {exp_name}, split={split}, return code={val_rc}")

        append_status_csv(status_csv, {
            "experiment_name": exp_name,
            "group": exp.group,
            "n_modes": str(tuple(exp.n_modes)),
            "hidden_channels": exp.hidden_channels,
            "lifting_channels": exp.lp_channels,
            "projection_channels": exp.lp_channels,
            "checkpoint_path": str(checkpoint_path),
            "train_return_code": train_rc if train_rc is not None else "",
            "train_wall_time_sec": train_time,
            "validate_return_codes": str(val_rcs),
            "validate_wall_times_sec": str(val_times),
        })

    print("\nAll requested experiments finished.")
    print(f"Grid CSV:   {grid_csv}")
    print(f"Status CSV: {status_csv}")
    print(f"Training summary CSV should be in: {checkpoint_dir / 'ablation_train_summary.csv'}")
    print(f"Validation summary CSV should be in: {checkpoint_dir / 'ablation_val_summary.csv'}")
    print(f"Test summary CSV should be in: {checkpoint_dir / 'ablation_test_summary.csv'}")


if __name__ == "__main__":
    main()
