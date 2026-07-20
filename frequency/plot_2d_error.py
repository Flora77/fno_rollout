# -*- coding: utf-8 -*-
"""
Paper-style 2D signed error fields for 5 experiments.

Layout:
    rows -> time snapshots: 15.0 s, 37.5 s, 75.0 s
    cols -> experiments: G1, G2, G3, G4, Ours

Each column:
    - small paper-style title with (a)(b)(c)(d)(e)
    - top-left metrics box: Hs, NRMSE, SSP, Corr

Error:
    e(x,y,t) = y_pred(x,y,t) - y_true(x,y,t)

Data:
    y_true_plot, y_pred_plot: [9, 300, 64, 64]

Example command:
    python plot_2d_error.py --sample_index 5
    python plot_2d_error.py --sample_index 0 --vmax 0.20        
    python plot_2d_error.py --sample_index 0 --percentile 99.5


"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.random import f
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


# =============================================================================
# User settings
# =============================================================================

DEFAULT_DATA_DIR = r"E:\hanqi\python\neuraloperator\frequency\sample9"

DT = 0.25
LX = 281.0
LY = 281.0
EPS = 1e-12

TIME_STEPS = [60, 150, 300]
TIME_INDICES = [59, 149, 299]
TIME_SECONDS = [step * DT for step in TIME_STEPS]
X_TICKS = [0, 70, 140, 210, 280]
Y_TICKS = [0, 70, 140, 210, 280]
CBAR_STEP = 0.1

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
        "label": "G6 Ours",
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
    if arr.ndim != 4:
        raise ValueError(f"{mat_path.name}: {var_name} should be [sample,time,x,y], got {arr.shape}")

    return arr.astype(np.float64)


def get_one_sample_true_pred(mat_path: Path, sample_index: int):
    y_true = load_mat_var(mat_path, "y_true_plot")
    y_pred = load_mat_var(mat_path, "y_pred_plot")

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch in {mat_path.name}: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )

    n_samples = y_true.shape[0]
    if not (0 <= sample_index < n_samples):
        raise IndexError(f"sample_index={sample_index} out of range, valid: 0~{n_samples-1}")

    return y_true[sample_index], y_pred[sample_index]


# =============================================================================
# Metrics
# =============================================================================

def compute_hs_sample(y_true: np.ndarray) -> float:
    return float(4.0 * np.std(y_true))


def compute_frame_nrmse(y_true: np.ndarray, y_pred: np.ndarray, hs_ref: float) -> np.ndarray:
    diff = y_pred - y_true
    rmse_t = np.sqrt(np.mean(diff ** 2, axis=(1, 2)))
    return rmse_t / max(hs_ref, EPS)


def compute_frame_corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    t_len = y_true.shape[0]
    corr = np.empty(t_len, dtype=np.float64)

    for t in range(t_len):
        a = y_true[t].reshape(-1)
        b = y_pred[t].reshape(-1)
        a0 = a - np.mean(a)
        b0 = b - np.mean(b)
        denom = np.sqrt(np.sum(a0 ** 2) * np.sum(b0 ** 2)) + EPS
        corr[t] = np.sum(a0 * b0) / denom

    return corr


def compute_frame_ssp(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    fft_true = np.fft.rfft2(y_true, axes=(-2, -1), norm="ortho")
    fft_pred = np.fft.rfft2(y_pred, axes=(-2, -1), norm="ortho")

    diff_norm = np.sqrt(np.sum(np.abs(fft_pred - fft_true) ** 2, axis=(1, 2)))
    true_norm = np.sqrt(np.sum(np.abs(fft_true) ** 2, axis=(1, 2)))
    pred_norm = np.sqrt(np.sum(np.abs(fft_pred) ** 2, axis=(1, 2)))

    return diff_norm / (true_norm + pred_norm + EPS)


def compute_sample_metrics(y_true: np.ndarray, y_pred: np.ndarray, hs_theory: float | None = None) -> dict:
    hs_sample = compute_hs_sample(y_true)
    hs_show = hs_sample if hs_theory is None else float(hs_theory)

    nrmse_curve = compute_frame_nrmse(y_true, y_pred, hs_sample)
    ssp_curve = compute_frame_ssp(y_true, y_pred)
    corr_curve = compute_frame_corr(y_true, y_pred)

    return {
        "Hs_show": hs_show,
        "Hs_sample": hs_sample,
        "NRMSE_mean": float(np.mean(nrmse_curve)),
        "SSP_mean": float(np.mean(ssp_curve)),
        "Corr_mean": float(np.mean(corr_curve)),
    }


# =============================================================================
# Data collection
# =============================================================================

def collect_experiment_results(
    data_dir: Path,
    sample_index: int,
    experiments: list[dict],
    time_indices: list[int],
    hs_theory: float | None = None,
):
    results = []
    ref_true = None

    for exp in experiments:
        mat_path = data_dir / exp["file"]
        y_true, y_pred = get_one_sample_true_pred(mat_path, sample_index)

        if ref_true is None:
            ref_true = y_true.copy()
        else:
            max_abs_diff = float(np.max(np.abs(ref_true - y_true)))
            if max_abs_diff > 1e-8:
                print(
                    f"Warning: y_true_plot in {mat_path.name} differs from first file. "
                    f"max_abs_diff={max_abs_diff:.3e}"
                )

        metrics = compute_sample_metrics(y_true, y_pred, hs_theory=hs_theory)
        error_fields = [y_pred[idx] - y_true[idx] for idx in time_indices]

        results.append({
            "tag": exp["tag"],
            "label": exp["label"],
            "file": exp["file"],
            "metrics": metrics,
            "error_fields": error_fields,
        })

    return results


# =============================================================================
# Helpers
# =============================================================================

def compute_symmetric_vmax(results, percentile=99.0, step=0.05):
    vals = []
    for item in results:
        for err in item["error_fields"]:
            vals.append(np.abs(err).ravel())
    vals = np.concatenate(vals)

    vmax_raw = float(np.percentile(vals, percentile))
    vmax_raw = max(vmax_raw, step)

    # 向上取整到 step 的整数倍，例如 0.132 -> 0.15
    vmax = np.ceil(vmax_raw / step) * step
    return float(vmax)


def make_cbar_formatter(vmax: float):
    if vmax >= 0.1:
        return FormatStrFormatter("%.2f")
    elif vmax >= 0.01:
        return FormatStrFormatter("%.3f")
    else:
        return FormatStrFormatter("%.4f")

def make_symmetric_cbar_ticks(vmax: float, step: float = 0.05):
    n = int(round(vmax / step))
    ticks = np.arange(-n, n + 1, dtype=float) * step
    ticks[np.isclose(ticks, 0.0)] = 0.0
    return ticks

# =============================================================================
# Plot
# =============================================================================

def plot_error_fields(
    results,
    sample_index: int,
    out_dir: Path,
    vmax: float | None = None,
    percentile: float = 99.0,
):
    nrows = 3
    ncols = 5

    if len(results) != ncols:
        raise ValueError(f"Expected {ncols} experiments, got {len(results)}")

    if vmax is None:
        vmax = compute_symmetric_vmax(results, percentile=percentile, step=CBAR_STEP)

    vmin = -vmax
    cmap = "viridis"

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(13.8, 8.0),
        constrained_layout=False,
    )

    extent = [0.0, LX, 0.0, LY]
    im = None

    for col, item in enumerate(results):
        metrics = item["metrics"]

        for row in range(nrows):
            ax = axes[row, col]
            err = item["error_fields"][row]

            im = ax.imshow(
                err.T,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="bicubic",
                aspect="equal",
            )

            # 稀疏坐标刻度
            ax.set_xticks(X_TICKS)
            ax.set_yticks(Y_TICKS)

            # 每列第一行：标题 + 指标框
            if row == 0:
                ax.set_title(
                    f"{item['tag']} {item['label']}",
                    pad=12,
                    fontweight="normal",
                )

                # NRMSE 改成百分比数值，但不显示 %
                nrmse_show = metrics["NRMSE_mean"] * 100.0

                # text = (
                #     # rf"$H_s={metrics['Hs_show']:.2f}\ \mathrm{{m}}$" "\n"
                #     # rf"NRMSE$={nrmse_show:.2f}$" "\n"
                #     # rf"SSP$={metrics['SSP_mean']:.3f}$" "\n"
                #     # rf"Corr$={metrics['Corr_mean']:.3f}$"
                # )
                # ax.text(
                #     0.03, 0.97, text,
                #     transform=ax.transAxes,
                #     ha="left", va="top",
                #     fontsize=7.8,
                #     bbox=dict(
                #         boxstyle="round,pad=0.22",
                #         facecolor="white",
                #         edgecolor="0.65",
                #         alpha=0.86
                #     ),
                # )

            # 行标签：只显示时间
            if col == 0:
                ax.set_ylabel("y / m")
                ax.text(
                    -0.30, 0.50,
                    rf"$t={TIME_SECONDS[row]:.1f}\ \mathrm{{s}}$",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    rotation=90,
                    fontsize=12,
                )
            else:
                ax.set_yticklabels([])

            if row == nrows - 1:
                ax.set_xlabel("x / m")
            else:
                ax.set_xticklabels([])

            ax.tick_params(direction="out", length=3, width=0.7)

    # fig.suptitle(
    #     "Spatial error fields for a representative sample",
    #     fontsize=11.5,
    #     y=0.955,
    # )

    fig.subplots_adjust(
        left=0.075,
        right=0.885,
        bottom=0.085,
        top=0.885,
        wspace=0.08,
        hspace=0.10,
    )

    # 固定 colorbar 位置
    cax = fig.add_axes([0.900, 0.145, 0.020, 0.66])
    cbar = fig.colorbar(im, cax=cax)

    cbar.set_label(r"Prediction error / m")

    # 固定成 0.05 间隔的对称刻度
    cbar_ticks = make_symmetric_cbar_ticks(vmax, step=CBAR_STEP)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar.ax.tick_params(length=3, width=0.7)

    out_png = out_dir / f"sample{sample_index + 1:02d}_error_fields_paper_v2.png"
    out_pdf = out_dir / f"sample{sample_index + 1:02d}_error_fields_paper_v2.pdf"
    out_svg = out_dir / f"sample{sample_index + 1:02d}_error_fields_paper_v2.svg"
    fig.savefig(out_png, dpi=1000)
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Colorbar range: [{vmin:.6f}, {vmax:.6f}] m")
    print(f"Colorbar ticks: {cbar_ticks}")

# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the MAT files."
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="0-based sample index."
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Manual symmetric colorbar limit. If not set, auto-computed."
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile used to auto-compute vmax. Default: 99.0"
    )
    parser.add_argument(
        "--hs_theory",
        type=float,
        default=None,
        help="Optional theoretical Hs shown in the metric box. If not set, use Hs=4*std(y_true)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_matplotlib()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / "figures_error_fields_paper_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = collect_experiment_results(
        data_dir=data_dir,
        sample_index=args.sample_index,
        experiments=EXPERIMENTS,
        time_indices=TIME_INDICES,
        hs_theory=args.hs_theory,
    )

    plot_error_fields(
        results=results,
        sample_index=args.sample_index,
        out_dir=out_dir,
        vmax=args.vmax,
        percentile=args.percentile,
    )


if __name__ == "__main__":
    main()