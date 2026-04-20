import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def _ensure_4d(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[None, ...]
    return x



def _ensure_3d(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[None, ...]
    return x



def _ensure_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, ...]
    return x



def _parse_string_array(raw):
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, np.ndarray):
        return [str(raw)]
    out = []
    for item in np.ravel(raw):
        if isinstance(item, bytes):
            out.append(item.decode("utf-8", errors="ignore"))
        elif isinstance(item, np.ndarray) and item.size == 1:
            out.append(str(item.item()))
        else:
            out.append(str(item))
    return out



def load_rollout_mat(mat_path: str) -> Dict:
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    return {
        "x_plot": _ensure_4d(mat["x_plot"]),
        "y_true_plot": _ensure_4d(mat["y_true_plot"]),
        "y_pred_plot": _ensure_4d(mat["y_pred_plot"]),
        "stepwise_rmse": np.asarray(mat["stepwise_rmse"], dtype=np.float32).reshape(-1),
        "chunk_rmse": np.asarray(mat["chunk_rmse"], dtype=np.float32).reshape(-1),
        "chunk_bounds": _ensure_2d(np.asarray(mat["chunk_bounds"], dtype=np.int32)),
        "rmse": float(np.asarray(mat["rmse"], dtype=np.float32).squeeze()),
        "rel_l2": float(np.asarray(mat["rel_l2"], dtype=np.float32).squeeze()),
        "final_step_rmse": float(np.asarray(mat["final_step_rmse"], dtype=np.float32).squeeze()),
        "last_chunk_rmse": float(np.asarray(mat["last_chunk_rmse"], dtype=np.float32).squeeze()),
        "radial_k": np.asarray(mat["radial_k"], dtype=np.float32).reshape(-1),
        "global_spectrum_true": np.asarray(mat["global_spectrum_true"], dtype=np.float32).reshape(-1),
        "global_spectrum_pred": np.asarray(mat["global_spectrum_pred"], dtype=np.float32).reshape(-1),
        "spectral_rel_l2_global": float(np.asarray(mat["spectral_rel_l2_global"], dtype=np.float32).squeeze()),
        "spectral_rel_l2_high": float(np.asarray(mat["spectral_rel_l2_high"], dtype=np.float32).squeeze()),
        "high_k_start_idx": int(np.asarray(mat["high_k_start_idx"], dtype=np.int32).squeeze()),
        "step_stat_true_std": np.asarray(mat["step_stat_true_std"], dtype=np.float32).reshape(-1),
        "step_stat_pred_std": np.asarray(mat["step_stat_pred_std"], dtype=np.float32).reshape(-1),
        "step_stat_true_hs": np.asarray(mat["step_stat_true_hs"], dtype=np.float32).reshape(-1),
        "step_stat_pred_hs": np.asarray(mat["step_stat_pred_hs"], dtype=np.float32).reshape(-1),
        "step_stat_true_rms_slope": np.asarray(mat["step_stat_true_rms_slope"], dtype=np.float32).reshape(-1),
        "step_stat_pred_rms_slope": np.asarray(mat["step_stat_pred_rms_slope"], dtype=np.float32).reshape(-1),
        "std_mae": float(np.asarray(mat["std_mae"], dtype=np.float32).squeeze()),
        "hs_mae": float(np.asarray(mat["hs_mae"], dtype=np.float32).squeeze()),
        "rms_slope_mae": float(np.asarray(mat["rms_slope_mae"], dtype=np.float32).squeeze()),
        "point_true_all": _ensure_3d(mat["point_true_all"]),
        "point_pred_all": _ensure_3d(mat["point_pred_all"]),
        "point_names": _parse_string_array(mat.get("point_names", None)),
        "point_coords": _ensure_2d(np.asarray(mat.get("point_coords", np.empty((0, 2))), dtype=np.int32)),
        "data_mean": float(np.asarray(mat["data_mean"], dtype=np.float32).squeeze()),
        "data_std": float(np.asarray(mat["data_std"], dtype=np.float32).squeeze()),
        "normalize": bool(int(np.asarray(mat["normalize"]).squeeze())),
        "dt": float(np.asarray(mat["dt"], dtype=np.float32).squeeze()),
        "saved_sample_ids": np.asarray(mat.get("saved_sample_ids", np.arange(0)), dtype=np.int32).reshape(-1),
        "sample_rmse": np.asarray(mat.get("sample_rmse", np.empty((0,))), dtype=np.float32).reshape(-1),
        "sample_rel_l2": np.asarray(mat.get("sample_rel_l2", np.empty((0,))), dtype=np.float32).reshape(-1),
        "sample_final_step_rmse": np.asarray(mat.get("sample_final_step_rmse", np.empty((0,))), dtype=np.float32).reshape(-1),
    }



def maybe_denormalize_field(x: np.ndarray, mean: float, std: float, enabled: bool) -> np.ndarray:
    return x * std + mean if enabled else x



def maybe_denormalize_error(x: np.ndarray, std: float, enabled: bool) -> np.ndarray:
    return x * std if enabled else x



def compute_frame_metrics(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    err = pred - true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    rel_l2 = float(np.linalg.norm(err.ravel()) / (np.linalg.norm(true.ravel()) + eps))
    return {"rmse": rmse, "mae": mae, "bias": bias, "rel_l2": rel_l2}



def compute_trace_metrics(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    err = pred - true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    rel_l2 = float(np.linalg.norm(err) / (np.linalg.norm(true) + eps))
    return {"rmse": rmse, "mae": mae, "bias": bias, "rel_l2": rel_l2}



def append_metrics_lines(lines, title: str, metrics: Dict[str, float], indent: str = "  ") -> None:
    lines.append(title)
    lines.append(f"{indent}RMSE   : {metrics['rmse']:.6e}")
    lines.append(f"{indent}MAE    : {metrics['mae']:.6e}")
    lines.append(f"{indent}Bias   : {metrics['bias']:.6e}")
    lines.append(f"{indent}RelL2  : {metrics['rel_l2']:.6e}")



def save_metrics_txt(lines, save_dir: str, filename: str = "plot_metrics.txt") -> None:
    path = os.path.join(save_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")



def plot_metric_text(data: Dict, save_dir: str) -> str:
    rmse_phys = float(maybe_denormalize_error(data["rmse"], data["data_std"], data["normalize"]))
    final_step_rmse_phys = float(maybe_denormalize_error(data["final_step_rmse"], data["data_std"], data["normalize"]))
    last_chunk_rmse_phys = float(maybe_denormalize_error(data["last_chunk_rmse"], data["data_std"], data["normalize"]))

    plt.figure(figsize=(9, 6))
    plt.axis("off")
    lines = [
        "Rollout summary",
        f"RMSE: {rmse_phys:.6f} m",
        f"RelL2: {data['rel_l2']:.6f}",
        f"Final-step RMSE: {final_step_rmse_phys:.6f} m",
        f"Last-chunk RMSE: {last_chunk_rmse_phys:.6f} m",
        f"Global spectral RelL2: {data['spectral_rel_l2_global']:.6f}",
        f"High-k spectral RelL2: {data['spectral_rel_l2_high']:.6f}",
        f"STD MAE: {data['std_mae']:.6f}",
        f"Hs MAE: {data['hs_mae']:.6f}",
        f"RMS-slope MAE: {data['rms_slope_mae']:.6f}",
    ]
    plt.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=11)
    plt.tight_layout()
    out = os.path.join(save_dir, "metrics_summary.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out



def plot_stepwise_rmse(data: Dict, save_dir: str) -> str:
    t = np.arange(1, len(data["stepwise_rmse"]) + 1) * data["dt"]
    stepwise_rmse_phys = maybe_denormalize_error(data["stepwise_rmse"], data["data_std"], data["normalize"])
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, stepwise_rmse_phys, marker="o")
    plt.xlabel("Prediction time [s]")
    plt.ylabel("RMSE [m]")
    plt.title("Stepwise rollout RMSE")
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(save_dir, "stepwise_rmse.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out



def plot_chunk_rmse(data: Dict, save_dir: str) -> str:
    labels = [f"{int(s)}-{int(e)}" for s, e in data["chunk_bounds"]]
    chunk_rmse_phys = maybe_denormalize_error(data["chunk_rmse"], data["data_std"], data["normalize"])
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(len(labels)), chunk_rmse_phys, marker="o")
    plt.xticks(np.arange(len(labels)), labels, rotation=30)
    plt.xlabel("Chunk [steps]")
    plt.ylabel("RMSE [m]")
    plt.title("Chunk RMSE")
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(save_dir, "chunk_rmse.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out



def plot_global_spectrum(data: Dict, save_dir: str) -> str:
    k = data["radial_k"]
    true_spec = np.maximum(data["global_spectrum_true"], 1e-12)
    pred_spec = np.maximum(data["global_spectrum_pred"], 1e-12)
    plt.figure(figsize=(8, 5))
    plt.semilogy(k, true_spec, label="True")
    plt.semilogy(k, pred_spec, label="Pred", linestyle="--")
    hk = data["high_k_start_idx"]
    if 0 < hk < len(k):
        plt.axvline(float(k[hk]), linestyle=":", linewidth=1.2)
    plt.xlabel("Radial wavenumber bin")
    plt.ylabel("Spectral energy [m²]")
    plt.title("Global radial spectrum")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(save_dir, "global_radial_spectrum.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out



def plot_wave_stats(data: Dict, save_dir: str) -> str:
    t = np.arange(1, len(data["step_stat_true_std"]) + 1) * data["dt"]
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9), sharex=True)
    axes[0].plot(t, data["step_stat_true_std"], label="True std")
    axes[0].plot(t, data["step_stat_pred_std"], label="Pred std", linestyle="--")
    axes[0].set_ylabel("std [m]")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(t, data["step_stat_true_hs"], label="True Hs")
    axes[1].plot(t, data["step_stat_pred_hs"], label="Pred Hs", linestyle="--")
    axes[1].set_ylabel("Hs [m]")
    axes[1].grid(True)
    axes[1].legend()
    axes[2].plot(t, data["step_stat_true_rms_slope"], label="True rms_slope")
    axes[2].plot(t, data["step_stat_pred_rms_slope"], label="Pred rms_slope", linestyle="--")
    axes[2].set_xlabel("Prediction time [s]")
    axes[2].set_ylabel("rms slope")
    axes[2].grid(True)
    axes[2].legend()
    fig.tight_layout()
    out = os.path.join(save_dir, "wave_statistics.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out



def plot_point_traces(data: Dict, save_dir: str, cfg: Dict, metrics_lines: List[str]) -> List[str]:
    outputs: List[str] = []
    if data["point_true_all"].size == 0:
        return outputs
    point_true_all = maybe_denormalize_field(data["point_true_all"], data["data_mean"], data["data_std"], data["normalize"])
    point_pred_all = maybe_denormalize_field(data["point_pred_all"], data["data_mean"], data["data_std"], data["normalize"])
    n_samples, n_steps, n_points = point_true_all.shape
    num_samples = min(int(cfg["evaluation"]["plot_num_samples"]), n_samples)
    t = np.arange(1, n_steps + 1) * data["dt"]
    plot_point_indices = list(range(n_points))
    metrics_lines.append("=" * 80)
    metrics_lines.append("[Point Traces] per-sample, per-point metrics in physical units")
    for sample_idx in range(num_samples):
        sample_id = int(data["saved_sample_ids"][sample_idx]) if sample_idx < len(data["saved_sample_ids"]) else sample_idx
        fig, axes = plt.subplots(len(plot_point_indices), 1, figsize=(10, 2.8 * len(plot_point_indices)), sharex=True, squeeze=False)
        axes = axes[:, 0]
        metrics_lines.append(f"  sample_index={sample_idx}, saved_sample_id={sample_id}")
        for ax, p_idx in zip(axes, plot_point_indices):
            true_trace = point_true_all[sample_idx, :, p_idx]
            pred_trace = point_pred_all[sample_idx, :, p_idx]
            name = data["point_names"][p_idx] if p_idx < len(data["point_names"]) else f"p{p_idx}"
            if p_idx < len(data["point_coords"]):
                coord = data["point_coords"][p_idx]
                label_name = f"{name} ({int(coord[0])}, {int(coord[1])})" if np.size(coord) >= 2 else name
            else:
                label_name = name
            ax.plot(t, true_trace, label="True")
            ax.plot(t, pred_trace, label="Pred", linestyle="--")
            ax.set_ylabel(f"{label_name}\nheight [m]")
            ax.grid(True)
            ax.legend()
            point_metrics = compute_trace_metrics(pred_trace, true_trace)
            append_metrics_lines(metrics_lines, title=f"    point={label_name}", metrics=point_metrics, indent="      ")
        axes[-1].set_xlabel("Prediction time [s]")
        fig.suptitle(f"Point traces of sample {sample_idx} (saved_sample_id={sample_id})", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out = os.path.join(save_dir, f"sample_{sample_idx:02d}_point_traces.png")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
        metrics_lines.append("")
    return outputs



def plot_sample_fields(data: Dict, save_dir: str, cfg: Dict, metrics_lines: List[str]) -> List[str]:
    outputs: List[str] = []
    x_plot = maybe_denormalize_field(data["x_plot"], data["data_mean"], data["data_std"], data["normalize"])
    y_true_plot = maybe_denormalize_field(data["y_true_plot"], data["data_mean"], data["data_std"], data["normalize"])
    y_pred_plot = maybe_denormalize_field(data["y_pred_plot"], data["data_mean"], data["data_std"], data["normalize"])
    if x_plot.size == 0:
        return outputs
    num_samples = min(int(cfg["evaluation"]["plot_num_samples"]), x_plot.shape[0])
    steps = [t for t in cfg["evaluation"]["plot_future_steps"] if t < y_true_plot.shape[1]]
    for sample_idx in range(num_samples):
        sample_id = int(data["saved_sample_ids"][sample_idx]) if sample_idx < len(data["saved_sample_ids"]) else sample_idx
        ncols = len(steps) + 1
        fig, axes = plt.subplots(3, ncols, figsize=(4.2 * ncols, 10.5), squeeze=False, constrained_layout=True)
        x_last = x_plot[sample_idx, -1]
        input_vmin = float(np.min(x_last))
        input_vmax = float(np.max(x_last))
        im_input = axes[0, 0].imshow(x_last, origin="lower", vmin=input_vmin, vmax=input_vmax)
        axes[0, 0].set_title("Input last")
        fig.colorbar(im_input, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Surface height [m]")
        axes[1, 0].axis("off")
        axes[2, 0].axis("off")
        field_vmin = min(float(np.min(y_true_plot[sample_idx, steps])), float(np.min(y_pred_plot[sample_idx, steps])))
        field_vmax = max(float(np.max(y_true_plot[sample_idx, steps])), float(np.max(y_pred_plot[sample_idx, steps])))
        err_abs = max(float(np.max(np.abs(y_pred_plot[sample_idx, t_idx] - y_true_plot[sample_idx, t_idx]))) for t_idx in steps)
        row0_ims, row1_ims, row2_ims = [], [], []
        metrics_lines.append("=" * 80)
        metrics_lines.append(f"[Sample Fields] sample_index={sample_idx}, saved_sample_id={sample_id}")
        for j, t_idx in enumerate(steps, start=1):
            true_frame = y_true_plot[sample_idx, t_idx]
            pred_frame = y_pred_plot[sample_idx, t_idx]
            err_frame = pred_frame - true_frame
            im_true = axes[0, j].imshow(true_frame, origin="lower", vmin=field_vmin, vmax=field_vmax)
            axes[0, j].set_title(f"True t+{t_idx + 1}")
            row0_ims.append(im_true)
            im_pred = axes[1, j].imshow(pred_frame, origin="lower", vmin=field_vmin, vmax=field_vmax)
            axes[1, j].set_title(f"Pred t+{t_idx + 1}")
            row1_ims.append(im_pred)
            im_err = axes[2, j].imshow(err_frame, origin="lower", vmin=-err_abs, vmax=err_abs)
            axes[2, j].set_title(f"Err t+{t_idx + 1}")
            row2_ims.append(im_err)
            frame_metrics = compute_frame_metrics(pred_frame, true_frame)
            append_metrics_lines(metrics_lines, title=f"  time_step={t_idx + 1} (time={data['dt'] * (t_idx + 1):.3f}s)", metrics=frame_metrics, indent="    ")
        if row0_ims:
            fig.colorbar(row0_ims[-1], ax=list(axes[0, 1:]), fraction=0.02, pad=0.02, label="Surface height [m]")
        if row1_ims:
            fig.colorbar(row1_ims[-1], ax=list(axes[1, 1:]), fraction=0.02, pad=0.02, label="Surface height [m]")
        if row2_ims:
            fig.colorbar(row2_ims[-1], ax=list(axes[2, 1:]), fraction=0.02, pad=0.02, label="Prediction error [m]")
        out = os.path.join(save_dir, f"sample_{sample_idx:02d}_fields.png")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
        metrics_lines.append("")
    return outputs



def plot_metric_distributions(data: Dict, save_dir: str) -> str:
    sample_rmse = maybe_denormalize_error(data["sample_rmse"], data["data_std"], data["normalize"])
    sample_final = maybe_denormalize_error(data["sample_final_step_rmse"], data["data_std"], data["normalize"])
    sample_rel = data["sample_rel_l2"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    if sample_rmse.size > 0:
        axes[0].hist(sample_rmse, bins=min(30, max(10, sample_rmse.size // 2)), alpha=0.8)
    axes[0].set_title("Sample RMSE distribution")
    axes[0].set_xlabel("RMSE [m]")
    axes[0].grid(True)
    if sample_rel.size > 0:
        axes[1].hist(sample_rel, bins=min(30, max(10, sample_rel.size // 2)), alpha=0.8)
    axes[1].set_title("Sample RelL2 distribution")
    axes[1].set_xlabel("RelL2")
    axes[1].grid(True)
    if sample_final.size > 0:
        axes[2].hist(sample_final, bins=min(30, max(10, sample_final.size // 2)), alpha=0.8)
    axes[2].set_title("Sample final-step RMSE distribution")
    axes[2].set_xlabel("Final-step RMSE [m]")
    axes[2].grid(True)
    fig.tight_layout()
    out = os.path.join(save_dir, "sample_metric_distributions.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out



def plot_wave_stat_scatter(data: Dict, save_dir: str) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    stats = [
        (data["step_stat_true_std"], data["step_stat_pred_std"], "std [m]"),
        (data["step_stat_true_hs"], data["step_stat_pred_hs"], "Hs [m]"),
        (data["step_stat_true_rms_slope"], data["step_stat_pred_rms_slope"], "rms slope"),
    ]
    for ax, (true_v, pred_v, title) in zip(axes, stats):
        ax.scatter(true_v, pred_v, s=10, alpha=0.8)
        lo = min(float(np.min(true_v)), float(np.min(pred_v)))
        hi = max(float(np.max(true_v)), float(np.max(pred_v)))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
        ax.set_xlabel(f"True {title}")
        ax.set_ylabel(f"Pred {title}")
        ax.set_title(title)
        ax.grid(True)
    fig.tight_layout()
    out = os.path.join(save_dir, "wave_stat_scatter.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out



def save_rollout_plots(data: Dict, save_dir: str, cfg: Dict) -> List[str]:
    ensure_dir(save_dir)
    metrics_lines: List[str] = []
    rmse_phys = float(maybe_denormalize_error(data["rmse"], data["data_std"], data["normalize"]))
    final_step_rmse_phys = float(maybe_denormalize_error(data["final_step_rmse"], data["data_std"], data["normalize"]))
    last_chunk_rmse_phys = float(maybe_denormalize_error(data["last_chunk_rmse"], data["data_std"], data["normalize"]))
    metrics_lines += [
        f"Split: {data['split']}",
        f"RMSE: {rmse_phys:.6e} m",
        f"RelL2: {data['rel_l2']:.6e}",
        f"Final-step RMSE: {final_step_rmse_phys:.6e} m",
        f"Last-chunk RMSE: {last_chunk_rmse_phys:.6e} m",
        f"Global spectral RelL2: {data['spectral_rel_l2_global']:.6e}",
        f"High-k spectral RelL2: {data['spectral_rel_l2_high']:.6e}",
        f"STD MAE: {data['std_mae']:.6e}",
        f"Hs MAE: {data['hs_mae']:.6e}",
        f"RMS-slope MAE: {data['rms_slope_mae']:.6e}",
        "",
    ]
    outputs = [
        plot_metric_text(data, save_dir),
        plot_stepwise_rmse(data, save_dir),
        plot_chunk_rmse(data, save_dir),
        plot_global_spectrum(data, save_dir),
        plot_wave_stats(data, save_dir),
    ]
    outputs.extend(plot_point_traces(data, save_dir, cfg, metrics_lines))
    outputs.extend(plot_sample_fields(data, save_dir, cfg, metrics_lines))
    if cfg["evaluation"].get("save_distribution_plots", True):
        outputs.append(plot_metric_distributions(data, save_dir))
        outputs.append(plot_wave_stat_scatter(data, save_dir))
    save_metrics_txt(metrics_lines, save_dir, filename="plot_metrics.txt")
    return outputs
