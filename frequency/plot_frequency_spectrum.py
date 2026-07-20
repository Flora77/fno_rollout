# -*- coding: utf-8 -*-
"""
Plot time-history metrics and frequency-domain 1D spectra for sample9 experiments.

Input .mat variables:
    y_true_plot: [num_samples, time, nx, ny]
    y_pred_plot: [num_samples, time, nx, ny]

Default data shape:
    [9, 300, 64, 64]
    dt = 0.25 s
    domain = 281 m x 281 m

Usage:
    python plot_frequency_spectrum.py
    python plot_frequency_spectrum.py --sample_index 0 --fmax 0.55
    python plot_frequency_spectrum.py --sample_index 0 --fmin 0.05 --fmax 0.55 --sfmin 1e-4 --sfmax 1e1
    python plot_frequency_spectrum.py --sample_index 0 --fmin 0.05 --fmax 0.40 --sfmin 1e-4 --sfmax 1e1
    python plot_frequency_spectrum.py --sample_index 3 --fmin 0.05 --fmax 0.40 --sfmin 5e-3 --sfmax 2e1
    python plot_frequency_spectrum.py --sample_index 5 --fmin 0.05 --fmax 0.40 --sfmin 1e-2 --sfmax 2e1  
    python plot_frequency_spectrum.py --sample_index 8 --fmin 0.05 --fmax 0.40 --sfmin 1e-3 --sfmax 3e1  

Specify sample:
    python plot_frequency_spectrum.py --sample_index 4
Specify folder:
    python plot_frequency_spectrum.py --data_dir E:\hanqi\python\neuraloperator\frequency\sample9
"""

from __future__ import annotations

import argparse
from ast import arg
import csv
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


# =============================================================================
# User settings
# =============================================================================

DEFAULT_DATA_DIR = r"E:\hanqi\python\neuraloperator\frequency\sample9"

DT = 0.25               # s
DOMAIN_X = 281.0        # m
DOMAIN_Y = 281.0        # m
EPS = 1e-12

# 六组实验文件。文件名需要和 sample9 文件夹内保持一致。
EXPERIMENTS = [

    {
        "label": "G2 FNO",
        "file": "G2_fno_no_curriculum_rollout300_no_added_loss_val_rollout.mat",
    },
    {
        "label": "G3 MSFNO",
        "file": "G4_msfno_no_curriculum_rollout300_no_added_loss_val_rollout.mat",
    },
    {
        "label": "G4 FNO+CR",
        "file": "G1_fno_curriculum_rollout300_no_added_loss_val_rollout.mat",
    },
    {
        "label": "G5 MSFNO+CR",
        "file": "G3_msfno_curriculum_rollout300_no_added_loss_val_rollout.mat",
    },

    {
        "label": "G6 Ours",
        "file": "mode_m28x28_h32_lp64_val_rollout.mat",
    },
]


# =============================================================================
# Basic tools
# =============================================================================

def setup_matplotlib() -> None:
    """Paper-style matplotlib settings."""
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.6,
        "figure.dpi": 150,
        "savefig.dpi": 1000,
        "savefig.bbox": "tight",
        "mathtext.fontset": "stix",
    })


def load_mat_array(mat_path: Path, var_name: str) -> np.ndarray:
    """Load one array from .mat and convert to float64."""
    if not mat_path.exists():
        raise FileNotFoundError(f"File not found: {mat_path}")

    mat = sio.loadmat(mat_path)
    if var_name not in mat:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"{var_name} not found in {mat_path.name}. Available keys: {keys}")

    arr = np.asarray(mat[var_name])
    arr = np.squeeze(arr)

    if arr.ndim != 4:
        raise ValueError(
            f"{mat_path.name}:{var_name} should be [sample,time,x,y], "
            f"but got shape {arr.shape}"
        )

    return arr.astype(np.float64)


def get_sample_fields(mat_path: Path, sample_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return y_true and y_pred of one selected sample, both [T,H,W]."""
    y_true_all = load_mat_array(mat_path, "y_true_plot")
    y_pred_all = load_mat_array(mat_path, "y_pred_plot")

    if y_true_all.shape != y_pred_all.shape:
        raise ValueError(
            f"Shape mismatch in {mat_path.name}: "
            f"y_true_plot={y_true_all.shape}, y_pred_plot={y_pred_all.shape}"
        )

    n_samples = y_true_all.shape[0]
    if sample_index < 0 or sample_index >= n_samples:
        raise IndexError(
            f"sample_index={sample_index} is out of range. "
            f"This file has {n_samples} samples, valid range: 0~{n_samples - 1}"
        )

    return y_true_all[sample_index], y_pred_all[sample_index]


# =============================================================================
# Metrics
# =============================================================================

def compute_hs_ref(y_true: np.ndarray) -> float:
    """
    Significant wave height of the selected true sample.

    Here Hs is computed from the full selected sample [T,H,W]:
        Hs = 4 * std(eta_true)
    """
    return float(4.0 * np.std(y_true))


def compute_frame_nrmse(y_true: np.ndarray, y_pred: np.ndarray, hs_ref: float) -> np.ndarray:
    """NRMSE(t) = spatial RMSE(t) / Hs_ref."""
    diff = y_pred - y_true
    rmse_t = np.sqrt(np.mean(diff ** 2, axis=(1, 2)))
    return rmse_t / max(hs_ref, EPS)


def compute_frame_corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Corr(t) between predicted and true spatial fields."""
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
    """
    SSP(t) based on spatial rfft2 at each time step.

    SSP(t) = ||FFT(pred_t)-FFT(true_t)|| /
             (||FFT(true_t)|| + ||FFT(pred_t)|| + eps)
    """
    fft_true = np.fft.rfft2(y_true, axes=(-2, -1), norm="ortho")
    fft_pred = np.fft.rfft2(y_pred, axes=(-2, -1), norm="ortho")

    diff_norm = np.sqrt(np.sum(np.abs(fft_pred - fft_true) ** 2, axis=(1, 2)))
    true_norm = np.sqrt(np.sum(np.abs(fft_true) ** 2, axis=(1, 2)))
    pred_norm = np.sqrt(np.sum(np.abs(fft_pred) ** 2, axis=(1, 2)))

    return diff_norm / (true_norm + pred_norm + EPS)


def compute_all_metric_curves(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray | float]:
    """Compute time-history NRMSE, SSP, and Corr for one sample."""
    hs_ref = compute_hs_ref(y_true)
    nrmse = compute_frame_nrmse(y_true, y_pred, hs_ref)
    ssp = compute_frame_ssp(y_true, y_pred)
    corr = compute_frame_corr(y_true, y_pred)

    return {
        "hs_ref": hs_ref,
        "nrmse": nrmse,
        "ssp": ssp,
        "corr": corr,
        "nrmse_mean": float(np.mean(nrmse)),
        "ssp_mean": float(np.mean(ssp)),
        "corr_mean": float(np.mean(corr)),
    }


# =============================================================================
# Frequency spectrum
# =============================================================================

def temporal_omnidirectional_spectrum(
    eta: np.ndarray,
    dt: float,
    use_window: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-dimensional frequency spectrum S(f) from eta(t,x,y).

    Procedure:
      1. remove temporal mean at each grid point;
      2. compute one-sided temporal FFT along time;
      3. convert to PSD with unit approximately m^2/Hz;
      4. average PSD over all spatial grid points.

    eta shape:
        [T,H,W]

    Return:
        freq: [Nf]
        S_f : [Nf], spatially averaged one-sided spectrum
    """
    eta = np.asarray(eta, dtype=np.float64)
    if eta.ndim != 3:
        raise ValueError(f"eta should be [T,H,W], got {eta.shape}")

    n_time = eta.shape[0]
    x = eta - np.mean(eta, axis=0, keepdims=True)

    if use_window:
        window = np.hanning(n_time).astype(np.float64)
        # U corrects the window energy loss.
        u = np.sum(window ** 2) / n_time
        x = x * window[:, None, None]
    else:
        u = 1.0

    fft = np.fft.rfft(x, axis=0)
    freq = np.fft.rfftfreq(n_time, d=dt)

    # Periodogram scaling: integrate S(f) df approximately equals variance.
    psd = (dt / (n_time * u)) * np.abs(fft) ** 2

    # One-sided correction: double positive non-DC and non-Nyquist bins.
    if n_time % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0

    s_f = np.mean(psd, axis=(1, 2))
    return freq, s_f


# =============================================================================
# Plotting
# =============================================================================

def plot_metric_curves(
    time: np.ndarray,
    results: list[dict],
    out_dir: Path,
    sample_index: int,
) -> None:
    """Plot NRMSE, SSP, Corr time histories in one 3-panel figure."""
    fig, axes = plt.subplots(
        3, 1,
        figsize=(7.2, 7.0),
        sharex=True,
        constrained_layout=True,
    )

    metric_info = [
        ("nrmse", "NRMSE", "NRMSE"),
        ("ssp", "SSP", "SSP"),
        ("corr", "Corr", "Correlation"),
    ]

    for ax, (key, ylabel, title) in zip(axes, metric_info):
        for item in results:
            curve = item[key]
            mean_val = item[f"{key}_mean"]
            label = f"{item['label']} ({mean_val:.4f})"
            ax.plot(time, curve, label=label)

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

    axes[-1].set_xlabel("Time / s")

    # Put one shared legend outside the figure.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.045),
        ncol=2,
        frameon=False,
    )

    out_png = out_dir / f"sample{sample_index + 1:02d}_metric_time_histories.png"
    fig.savefig(out_png)
    plt.close(fig)

    print(f"Saved: {out_png}")


def plot_frequency_spectra(
    spectra: list[dict],
    out_dir: Path,
    sample_index: int,
    f_min_plot: float | None = None,
    f_max_plot: float | None = None,
    sf_min_plot: float | None = None,
    sf_max_plot: float | None = None,
) -> None:
    """Plot true and predicted one-dimensional frequency spectra."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)

    # True spectrum is the same for all files. Plot it only once.
    freq = spectra[0]["freq"]
    s_true = spectra[0]["s_true"]

    # Avoid log(0) display issue.
    s_true_plot = np.maximum(s_true, EPS)

    ax.semilogy(
        freq,
        s_true_plot,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Ref",
        zorder=10,
    )

    for item in spectra:
        s_pred_plot = np.maximum(item["s_pred"], EPS)
        ax.semilogy(
            item["freq"],
            s_pred_plot,
            linewidth=1.5,
            label=item["label"],
        )

    ax.set_xlabel("$f$ / Hz")
    ax.set_ylabel(r"$F(f)$ / m$^2$ Hz$^{-1}$")
    # ax.set_title("Frequency-domain Omnidirectional Spectrum")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)

    if f_max_plot is None:
        f_max_plot = float(np.max(freq))
    ax.set_xlim(f_min_plot, f_max_plot)
    if sf_min_plot is not None or sf_max_plot is not None:
        ax.set_ylim(sf_min_plot, sf_max_plot)


    ax.legend(frameon=False, ncol=2)

    out_png = out_dir / f"sample{sample_index + 1:02d}_frequency_spectrum.png"
    out_pdf = out_dir / f"sample{sample_index + 1:02d}_frequency_spectrum.pdf"
    out_svg = out_dir / f"sample{sample_index + 1:02d}_frequency_spectrum.svg"
    fig.savefig(out_png, dpi=1000)
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"Saved: {out_png}")


def save_metric_mean_csv(results: list[dict], out_dir: Path, sample_index: int) -> None:
    """Save time-averaged metrics to CSV."""
    csv_path = out_dir / f"sample{sample_index + 1:02d}_metric_means.csv"

    fieldnames = [
        "label",
        "filename",
        "sample_index_0based",
        "sample_id_1based",
        "Hs_ref_true",
        "NRMSE_mean",
        "SSP_mean",
        "Corr_mean",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in results:
            writer.writerow({
                "label": item["label"],
                "filename": item["file"],
                "sample_index_0based": sample_index,
                "sample_id_1based": sample_index + 1,
                "Hs_ref_true": f"{item['hs_ref']:.8f}",
                "NRMSE_mean": f"{item['nrmse_mean']:.8f}",
                "SSP_mean": f"{item['ssp_mean']:.8f}",
                "Corr_mean": f"{item['corr_mean']:.8f}",
            })

    print(f"Saved: {csv_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Folder containing the six .mat files.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="0-based sample index. Example: 0 means the first sample; 4 means the fifth sample.",
    )
    parser.add_argument(
        "--no_window",
        action="store_true",
        help="Do not use Hann window when computing temporal frequency spectrum.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Maximum frequency shown in the spectrum plot. Default: Nyquist frequency.",
    )

    parser.add_argument(
        "--fmin",
        type=float,
        default=0.0,
        help="Minimum frequency shown in the spectrum plot. Default: 0.0 Hz.",
    )

    parser.add_argument(
        "--sfmin",
        type=float,
        default=1e-4,
        help="Minimum y-axis value of S(f) in spectrum plot. Example: 1e-4.",
    )

    parser.add_argument(
        "--sfmax",
        type=float,
        default=None,
        help="Maximum y-axis value of S(f) in spectrum plot. Example: 1e1.",
    )



    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_matplotlib()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / "figures_sample_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_index = int(args.sample_index)
    use_window = not bool(args.no_window)

    metric_results = []
    spectra_results = []

    ref_true = None
    time = None

    for exp in EXPERIMENTS:
        mat_path = data_dir / exp["file"]
        y_true, y_pred = get_sample_fields(mat_path, sample_index)

        if ref_true is None:
            ref_true = y_true.copy()
        else:
            max_abs_diff = float(np.max(np.abs(ref_true - y_true)))
            if max_abs_diff > 1e-8:
                print(
                    f"Warning: y_true_plot in {mat_path.name} is not exactly the same as the first file. "
                    f"max_abs_diff={max_abs_diff:.3e}"
                )

        if time is None:
            n_time = y_true.shape[0]
            time = np.arange(n_time, dtype=np.float64) * DT

        metrics = compute_all_metric_curves(y_true, y_pred)
        metrics.update({
            "label": exp["label"],
            "file": exp["file"],
        })
        metric_results.append(metrics)

        freq, s_true = temporal_omnidirectional_spectrum(
            y_true,
            dt=DT,
            use_window=use_window,
        )
        _, s_pred = temporal_omnidirectional_spectrum(
            y_pred,
            dt=DT,
            use_window=use_window,
        )
        spectra_results.append({
            "label": exp["label"],
            "file": exp["file"],
            "freq": freq,
            "s_true": s_true,
            "s_pred": s_pred,
        })

    plot_metric_curves(time, metric_results, out_dir, sample_index)
    plot_frequency_spectra(
        spectra_results,
        out_dir,
        sample_index,
        f_min_plot=args.fmin,
        f_max_plot=args.fmax,
        sf_min_plot=args.sfmin,
        sf_max_plot=args.sfmax,
    )
    save_metric_mean_csv(metric_results, out_dir, sample_index)

    print("\nTime-averaged metrics:")
    print("-" * 90)
    print(f"{'Experiment':30s} {'NRMSE':>12s} {'SSP':>12s} {'Corr':>12s} {'Hs_ref':>12s}")
    print("-" * 90)
    for item in metric_results:
        print(
            f"{item['label']:30s} "
            f"{item['nrmse_mean']:12.6f} "
            f"{item['ssp_mean']:12.6f} "
            f"{item['corr_mean']:12.6f} "
            f"{item['hs_ref']:12.6f}"
        )


if __name__ == "__main__":
    main()