import copy
import os
from typing import Any, Dict, Optional

import yaml


DEFAULT_CFG: Dict[str, Any] = {
    "project": {
        "experiment_name": "sea_surface_rollout_lightning",
        "output_root": "./outputs",
        "checkpoint_dirname": "checkpoints",
        "plot_dirname": "plots",
        "log_dirname": "logs",
        "summary_dirname": "summaries",
        "resolved_config_name": "resolved_config.yaml",
    },
    "seed": 42,
    "data": {
        "data_root": "./data/sea_surface_simple",
        "variable": "height",
        "input_steps": 40,
        "output_steps": 20,
        "stride": 4,
        "rollout_stride": 4,
        "normalize": True,
        "batch_size": 8,
        "val_batch_size": 8,
        "test_batch_size": 8,
        "num_workers": 4,
        "pin_memory": True,
    },
    "model": {
        "arch": "fno",
        "n_modes": [28, 28],
        "hidden_channels": 32,
        "lifting_channels": 64,
        "projection_channels": 64,
        "n_layers": 4,
    },
    "optim": {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "scheduler": {
            "enabled": True,
            "type": "plateau",
            "step_size": 20,
            "gamma": 0.5,
            "t_max": 80,
            "eta_min": 1e-6,
            "patience": 5,
            "factor": 0.5,
            "min_lr": 1e-6,
        },
    },
    "rollout": {
        "use_long_rollout_curriculum": False,
        "rollout_train_steps": [20, 40, 80, 120, 160, 240],
        "rollout_curriculum_boundaries": [0.0, 0.10, 0.25, 0.45, 0.65, 0.80],
        "rollout_steps": 240,
        "detach_context": False,
        "use_segment_weighting": False,
        "segment_weight_type": "linear",
        "segment_weight_min": 1.0,
        "segment_weight_max": 2.5,
        "segment_weight_power": 2.0,
        "normalize_segment_weights": True,
        "use_within_chunk_temporal_weighting": False,
        "chunk_time_weight_type": "linear",
        "chunk_time_weight_min": 1.0,
        "chunk_time_weight_max": 2.0,
        "chunk_time_weight_power": 2.0,
        "normalize_chunk_time_weights": True,
    },
    "evaluation": {
        "dt": 0.25,
        "num_full_samples_to_save": 3,
        "num_trace_points": 5,
        "spectral_high_k_ratio": 0.67,
        "spectral_band_split_ratios": [0.33, 0.67, 0.85],
        "plot_num_samples": 3,
        "plot_future_steps": [19, 59, 119, 239],
        "denormalize_for_plot": True,
        "save_val_mat": True,
        "save_test_mat": True,
        "save_plots": True,
        "save_distribution_plots": True,
        "export_mat_name_template": "{experiment_name}_{split}_rollout.mat",
    },
    "logging": {
        "tensorboard": {
            "enabled": True,
            "name": None,
        },
        "wandb": {
            "enabled": False,
            "project": "sea_surface",
            "entity": None,
            "name": None,
            "save_dir": None,
            "offline": False,
            "tags": [],
            "notes": None,
        },
        "csv": {
            "train_summary_csv": "ablation_train_summary.csv",
            "val_summary_csv": "ablation_val_summary.csv",
            "test_summary_csv": "ablation_test_summary.csv",
        },
    },
    "trainer": {
        "accelerator": "auto",
        "devices": 1,
        "precision": "32-true",
        "max_epochs": 80,
        "deterministic": False,
        "benchmark": False,
        "log_every_n_steps": 10,
        "num_sanity_val_steps": 0,
        "reload_dataloaders_every_n_epochs": 1,
        "val_check_interval": 1.0,
        "check_val_every_n_epoch": 1,
    },
    "ckpt": {
        "monitor": "val_rollout_rmse",
        "mode": "min",
        "save_top_k": 1,
        "save_last": True,
        "filename": "best-epoch{epoch:03d}-valrmse{val_rollout_rmse:.6f}",
        "resume_from": None,
        "early_stopping": {
            "enabled": True,
            "patience": 10,
            "min_delta": 0.0,
        },
    },
    "runtime": {
        "validate_after_fit": False,
    },
}


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base



def _normalize_cfg_types(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg["model"]["n_modes"] = tuple(int(v) for v in cfg["model"]["n_modes"])
    cfg["rollout"]["rollout_train_steps"] = [int(v) for v in cfg["rollout"]["rollout_train_steps"]]
    cfg["rollout"]["rollout_curriculum_boundaries"] = [float(v) for v in cfg["rollout"]["rollout_curriculum_boundaries"]]
    cfg["evaluation"]["spectral_band_split_ratios"] = [float(v) for v in cfg["evaluation"]["spectral_band_split_ratios"]]
    cfg["evaluation"]["plot_future_steps"] = [int(v) for v in cfg["evaluation"]["plot_future_steps"]]
    return cfg



def prepare_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    experiment_name = cfg["project"]["experiment_name"]
    output_root = os.path.abspath(cfg["project"]["output_root"])
    root_dir = os.path.join(output_root, experiment_name)
    checkpoint_dir = os.path.join(root_dir, cfg["project"]["checkpoint_dirname"])
    plot_dir = os.path.join(root_dir, cfg["project"]["plot_dirname"])
    log_dir = os.path.join(root_dir, cfg["project"]["log_dirname"])
    summary_dir = os.path.join(root_dir, cfg["project"]["summary_dirname"])
    resolved_config_path = os.path.join(root_dir, cfg["project"]["resolved_config_name"])

    cfg["paths"] = {
        "root_dir": root_dir,
        "checkpoint_dir": checkpoint_dir,
        "plot_dir": plot_dir,
        "log_dir": log_dir,
        "summary_dir": summary_dir,
        "resolved_config_path": resolved_config_path,
        "val_summary_path": os.path.join(summary_dir, cfg["logging"]["csv"]["val_summary_csv"]),
        "test_summary_path": os.path.join(summary_dir, cfg["logging"]["csv"]["test_summary_csv"]),
        "train_summary_path": os.path.join(summary_dir, cfg["logging"]["csv"]["train_summary_csv"]),
    }
    return cfg



def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CFG)
    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = deep_update(cfg, user_cfg)
    if overrides:
        cfg = deep_update(cfg, overrides)
    cfg = _normalize_cfg_types(cfg)
    cfg = prepare_paths(cfg)
    return cfg



def ensure_project_dirs(cfg: Dict[str, Any]) -> None:
    for key in ["root_dir", "checkpoint_dir", "plot_dir", "log_dir", "summary_dir"]:
        os.makedirs(cfg["paths"][key], exist_ok=True)



def save_resolved_config(cfg: Dict[str, Any]) -> None:
    ensure_project_dirs(cfg)
    with open(cfg["paths"]["resolved_config_path"], "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)



def resolve_checkpoint_path(cfg: Dict[str, Any], explicit_ckpt: Optional[str] = None) -> str:
    if explicit_ckpt:
        return explicit_ckpt
    resume_from = cfg["ckpt"].get("resume_from")
    if resume_from:
        return resume_from
    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    ckpts = [os.path.join(checkpoint_dir, x) for x in os.listdir(checkpoint_dir) if x.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under: {checkpoint_dir}")
    best_like = [p for p in ckpts if os.path.basename(p) != "last.ckpt"]
    pool = best_like if best_like else ckpts
    pool.sort(key=os.path.getmtime, reverse=True)
    return pool[0]
