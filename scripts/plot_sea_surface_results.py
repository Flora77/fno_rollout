import os
import sys
from typing import Tuple,List
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from config.sea_surface_rollout_config import SeaSurfaceSimpleConfig

def load_prediction_mat(mat_path: str):
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"MAT file not found at {mat_path}")
    mat = sio.loadmat(mat_path)
    required_keys = ["x","y_true","y_pred"]
    for key in required_keys:
        if key not in mat:
            raise KeyError(f"Key '{key}' not found in MAT file at {mat_path}")
    x = np.asarray(mat["x"], dtype=np.float32)
    y_true = np.asarray(mat["y_true"], dtype=np.float32)
    y_pred = np.asarray(mat["y_pred"], dtype=np.float32)
    stepwise_rmse = np.asarray(mat["stepwise_rmse"], dtype=np.float32).squeeze() if "stepwise_rmse" in mat else None
    mse = np.asarray(mat["mse"], dtype=np.float32).squeeze() if "mse" in mat else None
    rmse = np.asarray(mat["rmse"], dtype=np.float32).squeeze() if "rmse" in mat else None
    rel_l2 = np.asarray(mat["rel_l2"], dtype=np.float32).squeeze() if "rel_l2" in mat else None
    if x.ndim != 4 or y_true.ndim != 4 or y_pred.ndim != 4:
        raise ValueError(
            f"Expected x/y_true/y_pred to have shape (N,T,H,W), "
            f"but got x={x.shape}, y_true={y_true.shape}, y_pred={y_pred.shape}"
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    return x, y_true, y_pred, stepwise_rmse, mse, rmse, rel_l2

def ensure_dir(path:str):
    os.makedirs(path, exist_ok=True)

def get_prediction_mat_path(config: SeaSurfaceSimpleConfig) -> str:
    return os.path.join(config.checkpoint_dir, f"{config.experiment_name}_pdt.mat")

def compute_global_vmin_vmax(arrays: List[np.ndarray]) -> Tuple[float,float]:
    vals = [float(np.min(a)) for a in arrays] + [float(np.max(a)) for a in arrays]
    return min(vals), max(vals)

def get_default_point_rcs(H: int, W: int) -> List[Tuple[int, int]]:
    return [
        (H // 4, (3 * W) // 4),
        (H // 2, W // 2),
        ((3 * H) // 4, W // 4),
    ]

def plot_single_sample(
    x : np.ndarray,
    y_true : np.ndarray,
    y_pred : np.ndarray,
    sample_idx : int,
    future_steps_to_show : int,
    save_dir : str,
    cmap_field: str = "viridis",
    cmap_error: str = "bwr",
    interpolation: str = "bilinear",
):
    input_steps, H, W = x.shape
    output_steps = y_true.shape[0]
    x_last = x[-1]
    valid_steps = [t for t in future_steps_to_show if 0 <= t < output_steps]
    if len(valid_steps) == 0:
        raise ValueError(f"No valid future steps to show. Got future_steps_to_show={future_steps_to_show} with output_steps={output_steps}" )
    ncols = len(valid_steps) + 1
    nrows = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows),squeeze=False)
    im0 =axes[0,0].imshow(x_last, origin='lower', cmap=cmap_field, interpolation=interpolation)
    axes[0,0].set_title("Input last frame")
    plt.colorbar(im0, ax=axes[0,0],fraction=0.046, pad=0.04)
    for r in range(1,nrows):
        axes[r,0].axis('off')
    true_pred_arrays = []
    error_arrays = []
    for t in valid_steps:
        true_pred_arrays.append((y_true[t], y_pred[t]))
        error_arrays.append(y_true[t] - y_pred[t])
    filed_vmin, filed_vmax = compute_global_vmin_vmax(true_pred_arrays)
    err_abs_max = max(float(np.max(np.abs(e))) for e in error_arrays)
    err_vmin, err_vmax = -err_abs_max, err_abs_max

    for j, t in enumerate(valid_steps,start=1):
        true_frame = y_true[t]
        pred_frame = y_pred[t]
        error_frame = true_frame - pred_frame
        abs_error_frame = np.abs(error_frame)
        im_true = axes[0,j].imshow(true_frame, origin='lower', vmin=filed_vmin, vmax=filed_vmax, cmap=cmap_field, interpolation=interpolation)
        axes[0,j].set_title(f"True t+{t+1}")
        plt.colorbar(im_true, ax=axes[0,j],fraction=0.046, pad=0.04)
        im_pred = axes[1,j].imshow(pred_frame, origin='lower', vmin=filed_vmin, vmax=filed_vmax, cmap=cmap_field, interpolation=interpolation)
        axes[1,j].set_title(f"Pred t+{t+1}")
        plt.colorbar(im_pred, ax=axes[1,j],fraction=0.046, pad=0.04)        
        im_err = axes[2,j].imshow(error_frame, origin='lower', vmin=err_vmin, vmax=err_vmax, cmap=cmap_error, interpolation=interpolation)
        axes[2,j].set_title(f"Error t+{t+1}\n(pred - true)")
        plt.colorbar(im_err, ax=axes[2,j],fraction=0.046, pad=0.04)
        im_abs_err = axes[3,j].imshow(abs_error_frame, origin='lower',cmap = 'magma', interpolation=interpolation)
        axes[3,j].set_title(f"Abs Error t+{t+1}")
        plt.colorbar(im_abs_err, ax=axes[3,j],fraction=0.046, pad=0.04)    
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
            
    fig.suptitle(f"Validation Sample #{sample_idx}",fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(save_dir, f"sample_{sample_idx:04d}.png")
    fig.savefig(save_path,dpi=150,bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot for sample {sample_idx} at {save_path}")

def plot_stepwise_rmse(stepwise_rmse: np.ndarray,save_dir:str):
    if stepwise_rmse is None:
        print("No stepwise_rmse data to plot.")
        return
    plt.figure(figsize=(7,4))
    plt.plot(np.arange(1, len(stepwise_rmse)+1), stepwise_rmse, marker='o')
    plt.xlabel("Prediction step")
    plt.ylabel("RMSE")
    plt.title("Stepwise RMSE on Validation Set")
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "stepwise_rmse.png")
    plt.savefig(save_path,dpi=150,bbox_inches='tight')
    plt.close()
    print(f"Saved stepwise RMSE plot at {save_path}")

def plot_metric_text(mse, rmse, rel_l2, stepwise_rmse, save_dir:str):
    plt.figure(figsize=(6,3))
    plt.axis('off')
    lines = [
        "Validation Metrics",
        f"MSE    : {mse:.6f}" if mse is not None else "MSE    : N/A",
        f"RMSE   : {rmse:.6f}" if rmse is not None else "RMSE   : N/A",
        f"Rel L2 : {rel_l2:.6f}" if rel_l2 is not None else "Rel L2 : N/A",
        f"Stepwise RMSE: {', '.join(f'{v:.4f}' for v in stepwise_rmse)}" if stepwise_rmse is not None else "Stepwise RMSE: N/A"
    ]
    text = "\n".join(lines)
    plt.text(0.05, 0.9, text, fontsize=12, va="top")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "metrics_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics summary plot to: {save_path}")    


def plot_point_timeseries(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_idx: int,
    point_rcs:List[Tuple[int,int]],
    save_dir:str,
):
    T,H,W = y_true.shape
    t_axis = np.arange(1,T+1)
    n_points = len(point_rcs)
    fig,axes = plt.subplots(nrows=n_points, ncols=1, figsize=(8,3*n_points), squeeze=False)
    all_series = []
    point_metrics = []
    for rr,cc in point_rcs:
        true_series = y_true[:,rr,cc]
        pred_series = y_pred[:,rr,cc]
        err_series = pred_series - true_series
        rmse_point = np.sqrt(np.mean(err_series**2))
        all_series.append((true_series, pred_series))
        point_metrics.append((rr,cc,true_series,pred_series,rmse_point))
    series_vmin, series_vmax = compute_global_vmin_vmax(all_series)
    for i, (rr,cc,true_series,pred_series,rmse_point) in enumerate(point_metrics):
        ax = axes[i,0]
        ax.plot(t_axis, true_series, label="True", marker='o',linewidth=2,markersize=3)
        ax.plot(t_axis, pred_series, label="Pred", linewidth=2,linestyle='--')
        ax.set_title(f"Point{i+1}: (r={rr}, c={cc}) | RMSE= {rmse_point:.4f}")
        ax.set_xlabel("Prediction step")
        ax.set_ylabel("Sea Surface Height")
        ax.set_ylim(series_vmin, series_vmax)
        ax.grid(True,alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle(f"Time Series at Selected Points for Sample #{sample_idx}", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"sample_{sample_idx}_point_timeseries.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved point timeseries plot for sample {sample_idx} at {save_path}")



def main():
    config = SeaSurfaceSimpleConfig()
    pred_mat_path = get_prediction_mat_path(config)
    output_dir = os.path.join(config.checkpoint_dir, f"{config.experiment_name}_plots")
    ensure_dir(output_dir)
    print("=" * 80)
    print("Sea Surface Simple Plotting")
    print("=" * 80)
    print(f"Prediction mat : {pred_mat_path}")
    print(f"Output dir     : {output_dir}")
    print("=" * 80)

    x_all, y_true_all, y_pred_all, stepwise_rmse, mse, rmse, rel_l2 = load_prediction_mat(pred_mat_path)
    H, W = y_true_all.shape[2], y_true_all.shape[3]
    point_rcs = get_default_point_rcs(H, W)

    print(f"x_all shape: {x_all.shape}")
    print(f"y_true_all shape: {y_true_all.shape}")
    print(f"y_pred_all shape: {y_pred_all.shape}")
    plot_metric_text(mse, rmse, rel_l2, stepwise_rmse, output_dir)
    plot_stepwise_rmse(stepwise_rmse, output_dir)
    num_samples = x_all.shape[0]
    sample_indices = [0, min(1, num_samples-1), min(2, num_samples-1)]
    sample_indices = sorted(list(set(sample_indices)))  
    future_steps_to_show = [1, 9, 19]
    for idx in sample_indices:
        plot_single_sample(
            x=x_all[idx],
            y_true=y_true_all[idx],
            y_pred=y_pred_all[idx],
            sample_idx = idx,
            future_steps_to_show = future_steps_to_show,
            save_dir = output_dir,
        )
        plot_point_timeseries(
            y_true=y_true_all[idx],
            y_pred=y_pred_all[idx],
            sample_idx=idx,
            point_rcs=point_rcs,
            save_dir=output_dir,
        )
    print("=" * 80)
    print("Plotting finished.")
    print("=" * 80)
if __name__ == "__main__":
    main()    