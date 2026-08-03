import csv
import json
from pathlib import Path

import torch

from scripts.sparse_surface.summarize_finalist_robustness import HORIZONS, summarize


def _region(value):
    return {
        "rmse": value,
        "nrmse": value,
        "correlation": 1.0 - value,
        "ssp": value,
        "gradient_rmse": value,
        "gradient_nrmse": value,
    }


def test_summarize_finalist_robustness_writes_three_validation_tables(tmp_path):
    run = tmp_path / "run"
    (run / "evaluation").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    torch.save({"model_state_dict": {}}, run / "checkpoints" / "best.pt")
    (run / "data_split.json").write_text("{}", encoding="utf-8")
    training = {
        "elapsed_seconds": 12.0,
        "best_epoch": 2,
        "last_epoch": 3,
        "stopped_early": True,
        "safety_stop_reason": None,
        "peak_memory_allocated_mb": 4.0,
    }
    (run / "training_summary_epoch0003.json").write_text(
        json.dumps(training), encoding="utf-8"
    )

    history = {
        **_region(0.1),
        "missing_region": _region(0.2),
        "observed_region": _region(0.05),
    }
    curve = [
        {
            "frame": float(index),
            "time_seconds": index * 0.25,
            "rmse": 0.1,
            "nrmse": 0.1,
            "correlation": 0.9,
        }
        for index in range(1, 301)
    ]
    source_metrics = {
        "history_reconstruction": history,
        "error_growth_curve": curve,
        "source": {
            "name": "sample.mat",
            "relative_path": "val/sample.mat",
        },
    }
    metrics = {
        "history_reconstruction": history,
        "error_growth_curve": curve,
        "per_source": {"0": source_metrics},
        "evaluated_samples": 2,
        "evaluated_batches": 1,
        "source_count": 1,
        "rollout_steps": 300,
        "elapsed_seconds": 1.5,
        "seconds_per_sample": 0.75,
    }
    for horizon in HORIZONS:
        metrics[f"forecast_{horizon}"] = _region(0.1)
        source_metrics[f"forecast_{horizon}"] = _region(0.1)
    payload = {
        "split": "val",
        "scientific_config_sha256": "config",
        "parameters": {"total": 10, "trainable": 5},
        "metrics": metrics,
    }
    (run / "evaluation" / "val_metrics.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    matrix = {
        "split_policy": "frozen val only; test forbidden",
        "entries": [
            {
                "candidate": "candidate",
                "method": "method",
                "layers": "rate_sweep",
                "distribution": "in_distribution",
                "mask_type": "fixed_points",
                "observation_rate": 0.05,
                "mask_id": "mask",
                "mask_replicate": 0,
                "train_seed": 42,
                "eval_seed": 42,
                "metrics_path": "run/evaluation/val_metrics.json",
                "training_run_dir": "run",
                "checkpoint_path": "run/checkpoints/best.pt",
            }
        ],
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    output = tmp_path / "output"

    result = summarize(matrix_path, output, project_root=tmp_path)

    assert result["runs"] == 1
    assert result["per_mat_rows"] == 1
    assert result["error_growth_rows"] == 300
    with (output / "candidate_val_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["candidate"] == "candidate"
    assert float(row["f300_nrmse"]) == 0.1
