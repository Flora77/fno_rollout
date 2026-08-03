"""Run the four fixed-budget B5 residual-graph ablations sequentially."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


VARIANTS = (
    ("baseline", "baseline_h64_r64_g3_ref3_seed42.json"),
    ("small", "small_h32_r32_g3_ref3_seed42.json"),
    ("graph2", "graph2_h64_r64_g2_ref3_seed42.json"),
    ("no_refinement", "no_refinement_h64_r64_g3_ref0_seed42.json"),
)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    python = Path(sys.executable).resolve()
    split = root / "data" / "splits" / "sea_surface_bimodal_v1.json"
    config_root = root / "config" / "sparse_experiments" / "b5_v3_ablation"
    artifact_root = root / "results" / "b5_gno_residual_graph_v3_20260731_v1"
    status_path = artifact_root / "ablation_status.json"
    status: list[dict[str, object]] = []

    for variant, config_name in VARIANTS:
        config = config_root / config_name
        raw = json.loads(config.read_text(encoding="utf-8"))
        run_dir = root / "runs" / "sparse" / str(raw["experiment_name"])
        summaries = sorted(run_dir.glob("training_summary_epoch*.json"))
        returncode = 0
        log_path = artifact_root / "ablation_logs" / f"{variant}.log"
        if not summaries:
            resume_checkpoint = run_dir / "checkpoints" / "last.pt"
            if run_dir.exists() and not resume_checkpoint.is_file():
                returncode = 98
            else:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    str(python),
                    str(
                        root
                        / "scripts"
                        / "sparse_surface"
                        / "train_sparse_experiment.py"
                    ),
                    str(config),
                    "--project-root",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cuda",
                ]
                if resume_checkpoint.is_file():
                    command.extend(["--resume", str(resume_checkpoint)])
                else:
                    command.extend(["--split-manifest", str(split)])
                # A launcher/environment failure may leave a log without ever
                # creating the run directory. Replace that stale log, while
                # appending when a valid last.pt makes the run resumable.
                log_mode = "a" if resume_checkpoint.is_file() else "w"
                with log_path.open(log_mode, encoding="utf-8") as log:
                    process = subprocess.run(
                        command,
                        cwd=root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                    returncode = int(process.returncode)
            summaries = sorted(run_dir.glob("training_summary_epoch*.json"))
        best_json = run_dir / "checkpoints" / "best.json"
        status.append(
            {
                "variant": variant,
                "config": str(config),
                "run_dir": str(run_dir),
                "returncode": returncode,
                "training_summary": str(summaries[-1]) if summaries else None,
                "best_json": str(best_json) if best_json.is_file() else None,
                "test_accessed": False,
            }
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps(status, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if returncode != 0:
            return returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
