import os
from typing import Dict, List, Tuple

import scipy.io as sio
import torch

from .logging_utils import append_csv_row, finalize_loggers, log_image_to_loggers, log_metrics_to_loggers
from .plotting import save_rollout_plots
from .rollout import detailed_evaluate_rollout



def run_export_evaluation(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    cfg: Dict,
    data_mean: float,
    data_std: float,
    split: str,
    loggers: List = None,
) -> Tuple[Dict, List[str], str]:
    loggers = loggers or []
    result = detailed_evaluate_rollout(model, loader, device, cfg, data_mean, data_std, split)
    plot_dir = os.path.join(cfg["paths"]["plot_dir"], split)
    os.makedirs(plot_dir, exist_ok=True)
    mat_name = cfg["evaluation"]["export_mat_name_template"].format(experiment_name=cfg["project"]["experiment_name"], split=split)
    mat_path = os.path.join(cfg["paths"]["root_dir"], mat_name)
    if (split == "val" and cfg["evaluation"]["save_val_mat"]) or (split == "test" and cfg["evaluation"]["save_test_mat"]):
        sio.savemat(mat_path, result)
    image_paths: List[str] = []
    if cfg["evaluation"]["save_plots"]:
        image_paths = save_rollout_plots(result, plot_dir, cfg)

    metrics = {
        f"{split}/rmse": float(result["rmse"]),
        f"{split}/rel_l2": float(result["rel_l2"]),
        f"{split}/final_step_rmse": float(result["final_step_rmse"]),
        f"{split}/last_chunk_rmse": float(result["last_chunk_rmse"]),
        f"{split}/spectral_rel_l2_global": float(result["spectral_rel_l2_global"]),
        f"{split}/spectral_rel_l2_high": float(result["spectral_rel_l2_high"]),
        f"{split}/std_mae": float(result["std_mae"]),
        f"{split}/hs_mae": float(result["hs_mae"]),
        f"{split}/rms_slope_mae": float(result["rms_slope_mae"]),
    }
    if loggers:
        log_metrics_to_loggers(loggers, metrics)
        for image_path in image_paths:
            name = os.path.splitext(os.path.basename(image_path))[0]
            log_image_to_loggers(loggers, f"{split}/{name}", image_path)
        finalize_loggers(loggers)

    summary_row = {
        "experiment_name": cfg["project"]["experiment_name"],
        "split": split,
        "rmse": float(result["rmse"]),
        "rel_l2": float(result["rel_l2"]),
        "final_step_rmse": float(result["final_step_rmse"]),
        "last_chunk_rmse": float(result["last_chunk_rmse"]),
        "spectral_rel_l2_global": float(result["spectral_rel_l2_global"]),
        "spectral_rel_l2_high": float(result["spectral_rel_l2_high"]),
        "std_mae": float(result["std_mae"]),
        "hs_mae": float(result["hs_mae"]),
        "rms_slope_mae": float(result["rms_slope_mae"]),
        "mat_path": mat_path,
        "plot_dir": plot_dir,
    }
    csv_path = cfg["paths"]["val_summary_path"] if split == "val" else cfg["paths"]["test_summary_path"]
    append_csv_row(csv_path, summary_row)
    return result, image_paths, mat_path
