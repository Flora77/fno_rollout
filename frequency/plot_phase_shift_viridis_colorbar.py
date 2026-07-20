# -*- coding: utf-8 -*-
"""
Plot one-row Hovmöller diagrams of spatial phase error for 6 experiments.

Data path:
    E:\hanqi\python\neuraloperator\frequency\sample9

Variables in each MAT file:
    y_true_plot : [Ns, T, Nx, Ny]
    y_pred_plot : [Ns, T, Nx, Ny]

Task:
    - Select the same sample from all 6 MAT files
    - Extract the Hovmöller error field along a fixed transect y = y_idx
    - Plot 6 error Hovmöller diagrams in one row
    - Use the same symmetric colorbar range for all subplots
    - Default colormap is viridis, consistent with the spatial error-field figure
    
Example command:
    python plot_phase_shift_viridis_colorbar.py --sample_index 5 --cbar_step 0.2
    python plot_phase_shift_viridis_colorbar.py --sample_index 0 --y_idx 22
    python plot_phase_shift_viridis_colorbar.py --sample_index 0 --vmax 1.10 --cbar_step 0.10
    python plot_phase_shift_viridis_colorbar.py --sample_index 5 --cmap viridis --vmax 1.10 --cbar_step 0.10

    
Error definition:
    e(x,t) = eta_pred(x,y0,t) - eta_true(x,y0,t)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# =============================================================================
# User settings
# =============================================================================

DEFAULT_DATA_DIR = r"E:\hanqi\python\neuraloperator\frequency\sample9"

DT = 0.25
LX = 281.0
NX = 64
NY = 64

# 6 experiments
EXPERIMENTS = [
    {
        "tag": "(a)",
        "label": "G2 FNO",
        "file": "G2_fno_no_curriculum_rollout300_no_added_loss_val_rollout.mat",        

    },
    {
        "tag": "(b)",
        "label": "G3 MSFNO",
        "file": "G4_msfno_no_curriculum_rollout300_no_added_loss_val_rollout.mat",
    },
    {
        "tag": "(c)",
        "label": "G4 FNO+CR",
        "file": "G1_fno_curriculum_rollout300_no_added_loss_val_rollout.mat",
    },
    {
        "tag": "(d)",
        "label": "G5 MSFNO+CR",
        "file": "G3_msfno_curriculum_rollout300_no_added_loss_val_rollout.mat",        

    },
    {
        "tag": "(e)",
        "label": "Ours",
        "file": "mode_m28x28_h32_lp64_val_rollout.mat",
    },
]


# =============================================================================
# Plot style
# =============================================================================

def setup_matplotlib():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 0.75,
        "figure.dpi": 150,
        "savefig.dpi": 1000,
        "mathtext.fontset": "stix",
        "savefig.bbox": "tight",
    })


# =============================================================================
# IO
# =============================================================================

def load_mat_var(mat_path: Path, var_name: str) -> np.ndarray:
    if not mat_path.exists():
        raise FileNotFoundError(f"File not found: {mat_path}")

    data = sio.loadmat(mat_path)
    if var_name not in data:
        keys = [k for k in data.keys() if not k.startswith("__")]
        raise KeyError(f"{var_name} not found in {mat_path.name}. Available keys: {keys}")

    arr = np.asarray(data[var_name]).squeeze()
    return arr


def load_one_sample(mat_path: Path, sample_index: int):
    y_true_all = load_mat_var(mat_path, "y_true_plot")
    y_pred_all = load_mat_var(mat_path, "y_pred_plot")

    if y_true_all.ndim != 4 or y_pred_all.ndim != 4:
        raise ValueError(
            f"{mat_path.name}: y_true_plot / y_pred_plot should be 4D, "
            f"got {y_true_all.shape} and {y_pred_all.shape}"
        )

    if y_true_all.shape != y_pred_all.shape:
        raise ValueError(
            f"{mat_path.name}: y_true_plot and y_pred_plot shape mismatch: "
            f"{y_true_all.shape} vs {y_pred_all.shape}"
        )

    n_samples = y_true_all.shape[0]
    if not (0 <= sample_index < n_samples):
        raise IndexError(
            f"sample_index={sample_index} out of range for {mat_path.name}, "
            f"valid: 0~{n_samples - 1}"
        )

    y_true = y_true_all[sample_index].astype(np.float64)   # [T, Nx, Ny]
    y_pred = y_pred_all[sample_index].astype(np.float64)   # [T, Nx, Ny]
    return y_true, y_pred


# =============================================================================
# Helpers
# =============================================================================

def grid_index_to_physical_x(nx: int = NX, lx: float = LX) -> np.ndarray:
    return np.linspace(0.0, lx, nx)


def round_up_to_step(value: float, step: float) -> float:
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    value = max(float(value), float(step))
    return float(math.ceil(value / step) * step)


def compute_common_vmax(error_xt_list: list[np.ndarray], percentile: float = 99.0, step: float = 0.05) -> float:
    all_vals = np.concatenate([np.abs(arr).ravel() for arr in error_xt_list])
    vmax_raw = float(np.percentile(all_vals, percentile))
    return round_up_to_step(vmax_raw, step)


def make_symmetric_ticks(vmax: float, step: float = 0.05) -> np.ndarray:
    vmax = round_up_to_step(vmax, step)
    n = int(round(vmax / step))
    ticks = np.arange(-n, n + 1, dtype=float) * step
    ticks[np.isclose(ticks, 0.0)] = 0.0
    return ticks


# =============================================================================
# Main plotting
# =============================================================================

def plot_hovmoller_error_row(
    sample_index: int,
    y_idx: int,
    data_dir: Path,
    out_dir: Path,
    percentile: float = 99.0,
    cbar_step: float = 0.10,
    vmax: float | None = None,
    cmap: str = "viridis",
    interpolation: str = "bicubic",
):
    t = np.arange(300, dtype=np.float64) * DT
    x_phys = grid_index_to_physical_x()

    error_xt_list = []
    labels = []

    ref_true = None

    for exp in EXPERIMENTS:
        mat_path = data_dir / exp["file"]
        y_true, y_pred = load_one_sample(mat_path, sample_index)

        if ref_true is None:
            ref_true = y_true.copy()
        else:
            max_abs_diff = float(np.max(np.abs(ref_true - y_true)))
            if max_abs_diff > 1e-8:
                print(
                    f"Warning: y_true_plot in {mat_path.name} is not identical to the first file. "
                    f"max_abs_diff={max_abs_diff:.3e}"
                )

        # Hovmöller error along y = y_idx
        # shape: [T, Nx]
        error_xt = y_pred[:, :, y_idx] - y_true[:, :, y_idx]

        error_xt_list.append(error_xt)
        labels.append(f"{exp['tag']} {exp['label']}")

    if vmax is None:
        vmax = compute_common_vmax(error_xt_list, percentile=percentile, step=cbar_step)
    else:
        vmax = round_up_to_step(vmax, cbar_step)

    vmin = -vmax

    n_exp = len(error_xt_list)
    fig, axes = plt.subplots(
        1, n_exp,
        figsize=(2.8 * n_exp, 3.6),
        sharey=True,
        constrained_layout=False
    )
    if n_exp == 1:
        axes = [axes]

    extent = [x_phys[0], x_phys[-1], t[0], t[-1]]
    im = None

    for i, ax in enumerate(axes):
        err_xt = error_xt_list[i]

        # Use the same color standard as the spatial error-field figure:
        # a common symmetric range for all subplots and a viridis-style map
        # where lower errors are purple, near-zero values are green/teal,
        # and higher positive errors are yellow.
        im = ax.imshow(
            err_xt,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
        )

        ax.set_title(labels[i], pad=8)

        # 论文风格：x 轴少量刻度
        ax.set_xticks([0, 70, 140, 210])

        if i == 0:
            ax.set_ylabel("$t$ / s")
            ax.set_yticks([0, 15, 30, 45, 60, 75])
        else:
            ax.tick_params(labelleft=False)

        ax.set_xlabel("x / m")
        ax.tick_params(direction="out", length=3, width=0.7)

    # fig.suptitle(
    #     rf"Hovmöller diagrams of spatial phase error",
    #     fontsize=11.0,
    #     y=0.98,
    # )

    # 手动留出 colorbar 空间
    fig.subplots_adjust(
        left=0.055,
        right=0.90,
        bottom=0.20,
        top=0.82,
        wspace=0.05,
    )

    # Shared colorbar
    cax = fig.add_axes([0.915, 0.20, 0.015, 0.58])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"Prediction error / m")

    cbar_ticks = make_symmetric_ticks(vmax, step=cbar_step)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar.ax.tick_params(length=3, width=0.7)

    out_png = out_dir / f"sample{sample_index + 1:02d}_hovmoller_error_row_y{y_idx}.png"
    out_pdf = out_dir / f"sample{sample_index + 1:02d}_hovmoller_error_row_y{y_idx}.pdf"
    fig.savefig(out_png, dpi=1000)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Common colorbar range: [{vmin:.4f}, {vmax:.4f}] m")
    print("Colorbar ticks:", ", ".join(f"{v:.2f}" for v in cbar_ticks))


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the 6 MAT files."
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="0-based sample index."
    )
    parser.add_argument(
        "--y_idx",
        type=int,
        default=32,
        help="Grid y-index of the horizontal transect for Hovmöller diagram."
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile used for automatic common colorbar range."
    )
    parser.add_argument(
        "--cbar_step",
        type=float,
        default=0.10,
        help="Shared colorbar tick interval. Use 0.10 to match the spatial error-field figure style."
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Manual symmetric colorbar limit. For the spatial error-field style, use --vmax 1.10."
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap for the error value. Use 'viridis' to match the spatial error-field figure."
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="bicubic",
        help="Interpolation method for imshow."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_matplotlib()

    if not (0 <= args.y_idx < NY):
        raise ValueError(f"y_idx must be in [0, {NY - 1}], got {args.y_idx}")

    if args.cbar_step <= 0:
        raise ValueError(f"cbar_step must be positive, got {args.cbar_step}")

    if args.vmax is not None and args.vmax <= 0:
        raise ValueError(f"vmax must be positive, got {args.vmax}")

    data_dir = Path(args.data_dir)
    out_dir = data_dir / "figures_hovmoller_error_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_hovmoller_error_row(
        sample_index=args.sample_index,
        y_idx=args.y_idx,
        data_dir=data_dir,
        out_dir=out_dir,
        percentile=args.percentile,
        cbar_step=args.cbar_step,
        vmax=args.vmax,
        cmap=args.cmap,
        interpolation=args.interpolation,
    )


if __name__ == "__main__":
    main()