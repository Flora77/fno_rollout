"""Train residual-graph-v3 GNO seeds 42/43/44 sequentially and resumably."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    config_root = root / "config" / "sparse_experiments" / "b5_v3_formal"
    split = root / "data" / "splits" / "sea_surface_bimodal_v1.json"
    artifact_root = root / "results" / "b5_gno_residual_graph_v3_20260802_v1"
    artifact_root.mkdir(parents=True, exist_ok=True)
    status_path = artifact_root / "training_status.json"
    status: list[dict[str, object]] = []

    for seed in (42, 43, 44):
        config = config_root / f"seed{seed}.json"
        raw = json.loads(config.read_text(encoding="utf-8"))
        run_dir = root / "runs" / "sparse" / raw["experiment_name"]
        summaries = sorted(run_dir.glob("training_summary_epoch*.json"))
        returncode = 0
        if not summaries:
            last = run_dir / "checkpoints" / "last.pt"
            if run_dir.exists() and not last.is_file():
                returncode = 98
            else:
                command = [
                    sys.executable,
                    str(root / "scripts" / "sparse_surface" / "train_sparse_experiment.py"),
                    str(config),
                    "--project-root", str(root),
                    "--run-dir", str(run_dir),
                    "--device", "cuda",
                ]
                if last.is_file():
                    command.extend(["--resume", str(last)])
                else:
                    command.extend(["--split-manifest", str(split)])
                log_path = artifact_root / f"seed{seed}_train.log"
                with log_path.open("a" if last.is_file() else "w", encoding="utf-8") as log:
                    returncode = subprocess.run(
                        command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                        text=True, check=False,
                    ).returncode
            summaries = sorted(run_dir.glob("training_summary_epoch*.json"))
        best = run_dir / "checkpoints" / "best.json"
        status.append({
            "seed": seed, "returncode": returncode,
            "run_dir": str(run_dir),
            "training_summary": str(summaries[-1]) if summaries else None,
            "best_json": str(best) if best.is_file() else None,
            "test_accessed": False,
        })
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        if returncode != 0:
            return returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
