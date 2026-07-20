# -*- coding: utf-8 -*-
"""
Plot local crest zoom-in curves for one selected point and one selected sample.

For each available MAT file:
    - use the same selected sample
    - use one selected point from point_true_all / point_pred_all
    - plot True and Pred local crest zoom-in curve
    - annotate Delta t_c and Delta eta_c

MAT variables:
    point_true_all : [Ns, T, Np]
    point_pred_all : [Ns, T, Np]
    point_coords   : [Np, 2]

Default data directory:
    E:\hanqi\python\neuraloperator\frequency\sample9

Usage example:
    python plot_point_trace.py --sample_index 5 --point_index 0
    python plot_point_trace.py --sample_index 5 --point_index 0 --dominant_search_t0 20 --dominant_search_t1 60  
    python plot_point_trace.py --sample_index 5 --point_index 0 --zoom_half_width_sec 8
    python plot_point_trace.py --sample_index 5 --point_index 2 --zoom_half_width_sec 8
    
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


# =============================================================================
# Basic settings
# =============================================================================

DEFAULT_DATA_DIR = r"E:\hanqi\python\neuraloperator\frequency\sample9"
DT = 0.25
EPS = 1e-12

# Missing files are skipped automatically.
EXPERIMENTS = [
    {
        "tag": "(a)",
        "label": "G1 FNO+CR",
        "file": "G1_fno_curriculum_rollout300_no_added_loss_val_rollout.mat",
        "color": "#1f77b4",
    },
    {
        "tag": "(b)",
        "label": "G2 FNO",
        "file": "G2_fno_no_curriculum_rollout300_no_added_loss_val_rollout.mat",
        "color": "#ff7f0e",
    },
    {
        "tag": "(c)",
        "label": "G3 MSFNO+CR",
        "file": "G3_msfno_curriculum_rollout300_no_added_loss_val_rollout.mat",
        "color": "#2ca02c",
    },
    {
        "tag": "(d)",
        "label": "G4 MSFNO",
        "file": "G4_msfno_no_curriculum_rollout300_no_added_loss_val_rollout.mat",
        "color": "#d62728",
    },
    {
        "tag": "(e)",
        "label": "Ours",
        "file": "mode_m28x28_h32_lp64_val_rollout.mat",
        "color": "#9467bd",
    },
]


# =============================================================================
# Plot style
# =============================================================================

def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.85,
        "lines.linewidth": 1.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "mathtext.fontset": "stix",
        "savefig.bbox": "tight",
    })


# =============================================================================
# IO
# =============================================================================

def load_mat_var(mat_path: Path, var_name: str) -> np.ndarray:
    if not mat_path.exists():
        raise FileNotFoundError(f"File not found: {mat_path}")

    mat = sio.loadmat(mat_path)
    if var_name not in mat:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"{var_name} not found in {mat_path.name}. Available keys: {keys}")

    return np.asarray(mat[var_name]).squeeze()


def load_point_data(mat_path: Path, sample_index: int, point_index: int) -> dict:
    point_true_all = load_mat_var(mat_path, "point_true_all")
    point_pred_all = load_mat_var(mat_path, "point_pred_all")
    point_coords = load_mat_var(mat_path, "point_coords")

    if point_true_all.ndim != 3 or point_pred_all.ndim != 3:
        raise ValueError(
            f"{mat_path.name}: point_true_all and point_pred_all should be [Ns,T,Np], "
            f"got {point_true_all.shape} and {point_pred_all.shape}"
        )

    if point_true_all.shape != point_pred_all.shape:
        raise ValueError(
            f"{mat_path.name}: shape mismatch, "
            f"point_true_all={point_true_all.shape}, point_pred_all={point_pred_all.shape}"
        )

    n_samples, n_time, n_points = point_true_all.shape

    if not (0 <= sample_index < n_samples):
        raise IndexError(
            f"{mat_path.name}: sample_index={sample_index} out of range, valid 0~{n_samples - 1}"
        )

    if not (0 <= point_index < n_points):
        raise IndexError(
            f"{mat_path.name}: point_index={point_index} out of range, valid 0~{n_points - 1}"
        )

    if point_coords.ndim != 2 or point_coords.shape[0] < n_points:
        raise ValueError(f"{mat_path.name}: invalid point_coords shape: {point_coords.shape}")

    return {
        "true": point_true_all[sample_index, :, point_index].astype(np.float64),
        "pred": point_pred_all[sample_index, :, point_index].astype(np.float64),
        "coord": tuple(point_coords[point_index].astype(int).tolist()),
        "n_time": n_time,
    }


# =============================================================================
# Crest detection
# =============================================================================

def get_index_range_from_time(t: np.ndarray, time_range: tuple[float, float] | None) -> tuple[int, int]:
    if time_range is None:
        return 0, len(t) - 1

    t0, t1 = time_range
    i0 = int(np.searchsorted(t, t0, side="left"))
    i1 = int(np.searchsorted(t, t1, side="right")) - 1
    i0 = max(0, i0)
    i1 = min(len(t) - 1, i1)

    if i1 <= i0:
        raise ValueError(f"Invalid time_range={time_range}, got index range [{i0}, {i1}]")

    return i0, i1


def find_true_crest(series_true: np.ndarray, t: np.ndarray, search_range=None) -> tuple[int, float, float]:
    """
    True crest is defined as the maximum of the true signal in the search window.
    """
    i0, i1 = get_index_range_from_time(t, search_range)
    idx_local = int(np.argmax(series_true[i0:i1 + 1]))
    idx = i0 + idx_local
    return idx, float(t[idx]), float(series_true[idx])


def find_pred_crest_near_true(
    series_pred: np.ndarray,
    t: np.ndarray,
    t_true: float,
    half_window_sec: float,
) -> tuple[int, float, float]:
    """
    Predicted crest is matched by searching the local maximum around the true crest time.
    """
    search_range = (t_true - half_window_sec, t_true + half_window_sec)
    i0, i1 = get_index_range_from_time(t, search_range)
    idx_local = int(np.argmax(series_pred[i0:i1 + 1]))
    idx = i0 + idx_local
    return idx, float(t[idx]), float(series_pred[idx])

def find_crest_in_same_range(series: np.ndarray, t: np.ndarray, search_range) -> tuple[int, float, float]:
    """
    Find crest as the maximum of a signal within the same prescribed search range.

    This is used to make true and predicted crests searched in exactly the same window:
        t_c = argmax_{t in W_c} eta(t)
    """
    i0, i1 = get_index_range_from_time(t, search_range)
    idx_local = int(np.argmax(series[i0:i1 + 1]))
    idx = i0 + idx_local
    return idx, float(t[idx]), float(series[idx])

# =============================================================================
# Data loading
# =============================================================================

def load_available_experiments(
    data_dir: Path,
    sample_index: int,
    point_index: int,
) -> tuple[list[dict], np.ndarray, tuple[int, int]]:
    """
    Load all available MAT files. Missing files are skipped.
    """
    loaded = []
    ref_true = None
    ref_coord = None
    n_time_ref = None

    for exp in EXPERIMENTS:
        mat_path = data_dir / exp["file"]

        if not mat_path.exists():
            print(f"Skip missing file: {mat_path.name}")
            continue

        try:
            item = load_point_data(mat_path, sample_index, point_index)
        except Exception as exc:
            print(f"Skip invalid file {mat_path.name}: {exc}")
            continue

        if ref_true is None:
            ref_true = item["true"].copy()
            ref_coord = item["coord"]
            n_time_ref = item["n_time"]
        else:
            max_diff = float(np.max(np.abs(ref_true - item["true"])))
            if max_diff > 1e-8:
                print(
                    f"Warning: point_true_all in {mat_path.name} differs from the first file. "
                    f"max_abs_diff={max_diff:.3e}"
                )

            if item["coord"] != ref_coord:
                print(
                    f"Warning: point_coords in {mat_path.name} differs from the first file: "
                    f"{item['coord']} vs {ref_coord}"
                )

            if item["n_time"] != n_time_ref:
                raise ValueError(
                    f"{mat_path.name}: time length differs from the first file: "
                    f"{item['n_time']} vs {n_time_ref}"
                )

        loaded.append({
            "tag": exp["tag"],
            "label": exp["label"],
            "file": exp["file"],
            "color": exp["color"],
            "pred": item["pred"],
        })

    if ref_true is None or len(loaded) == 0:
        raise RuntimeError("No valid MAT files were loaded. Please check data_dir and EXPERIMENTS.")

    return loaded, ref_true, ref_coord


# =============================================================================
# Plot
# =============================================================================

def compute_common_zoom_ylim(
    true_series: np.ndarray,
    loaded: list[dict],
    t: np.ndarray,
    zoom_range: tuple[float, float],
    margin_ratio: float = 0.22,
) -> tuple[float, float]:
    mask = (t >= zoom_range[0]) & (t <= zoom_range[1])
    vals = [true_series[mask]]

    for item in loaded:
        vals.append(item["pred"][mask])

    vals = np.concatenate(vals)
    ymin = float(np.min(vals))
    ymax = float(np.max(vals))
    yrange = max(ymax - ymin, 1e-6)

    return ymin - margin_ratio * yrange, ymax + margin_ratio * yrange


def plot_one_point_local_crest_column(
    loaded: list[dict],
    true_series: np.ndarray,
    point_coord: tuple[int, int],
    sample_index: int,
    point_index: int,
    out_dir: Path,
    dominant_search_range=None,
    crest_half_window_sec: float = 5.0,
    zoom_half_width_sec: float = 6.0,
) -> None:
    n_time = len(true_series)
    t = np.arange(n_time, dtype=np.float64) * DT

    if dominant_search_range is not None:
        # True and Pred will use the same user-specified crest search window.
        idx_true, t_true, eta_true = find_crest_in_same_range(
            true_series,
            t,
            search_range=dominant_search_range,
        )
    else:
        # Keep the original behavior if no search window is provided.
        idx_true, t_true, eta_true = find_true_crest(
            true_series,
            t,
            search_range=None,
        )

    zoom_t0 = max(t[0], t_true - zoom_half_width_sec)
    zoom_t1 = min(t[-1], t_true + zoom_half_width_sec)
    zoom_mask = (t >= zoom_t0) & (t <= zoom_t1)
    zoom_ylim = compute_common_zoom_ylim(true_series, loaded, t, (zoom_t0, zoom_t1))

    n_exp = len(loaded)

    fig_height = max(2.15 * n_exp, 3.0)
    fig, axes = plt.subplots(
        n_exp,
        1,
        figsize=(8.4, fig_height),
        sharex=True,
        constrained_layout=False,
    )

    if n_exp == 1:
        axes = [axes]

    csv_rows = []

    for row, item in enumerate(loaded):
        ax = axes[row]
        pred_series = item["pred"]
        color = item["color"]

        if dominant_search_range is not None:
            # Use exactly the same search window as the true crest.
            idx_pred, t_pred, eta_pred = find_crest_in_same_range(
                pred_series,
                t,
                search_range=dominant_search_range,
            )
        else:
            # Keep the original behavior if no common search window is provided.
            idx_pred, t_pred, eta_pred = find_pred_crest_near_true(
                pred_series,
                t,
                t_true=t_true,
                half_window_sec=crest_half_window_sec,
            )

        delta_t = t_pred - t_true
        delta_eta = eta_pred - eta_true

        csv_rows.append({
            "sample_index_0based": sample_index,
            "sample_id_1based": sample_index + 1,
            "point_index_0based": point_index,
            "point_id_1based": point_index + 1,
            "point_coord": point_coord,
            "experiment": item["label"],
            "t_true_crest_sec": f"{t_true:.6f}",
            "eta_true_crest_m": f"{eta_true:.8f}",
            "t_pred_crest_sec": f"{t_pred:.6f}",
            "eta_pred_crest_m": f"{eta_pred:.8f}",
            "delta_t_sec": f"{delta_t:.6f}",
            "delta_eta_m": f"{delta_eta:.8f}",
            "crest_ratio_pred_over_true": f"{eta_pred / (eta_true + EPS):.8f}",
            "source_file": item["file"],
        })

        # Curves
        ax.plot(
            t[zoom_mask],
            true_series[zoom_mask],
            color="black",
            linewidth=2.0,
            linestyle="-",
            label="True",
            zorder=4,
        )
        ax.plot(
            t[zoom_mask],
            pred_series[zoom_mask],
            color=color,
            linewidth=1.8,
            linestyle="-",
            label="Pred",
            zorder=3,
        )

        # Crest markers
        ax.plot(t_true, eta_true, "o", color="black", markersize=5, zorder=6)
        ax.plot(t_pred, eta_pred, "^", color=color, markersize=6, zorder=6)

        # Vertical reference lines
        ax.axvline(t_true, color="black", linestyle="--", linewidth=0.8, alpha=0.70, zorder=1)
        ax.axvline(t_pred, color=color, linestyle=":", linewidth=0.9, alpha=0.85, zorder=1)

        ax.set_xlim(zoom_t0, zoom_t1)
        ax.set_ylim(*zoom_ylim)
        ax.set_ylabel(r"$\eta$ / m")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        ax.tick_params(direction="out", length=3, width=0.7)

        # Subplot title
        ax.set_title(
            f"{item['tag']} {item['label']}",
            loc="left",
            pad=4,
            fontweight="normal",
        )

        # Small curve legend inside but away from left title
        ax.legend(
            loc="upper right",
            frameon=False,
            handlelength=2.0,
            borderaxespad=0.2,
        )

        # Metric textbox
        text = (
            rf"$t^{{true}}={t_true:.2f}\,$s, "
            rf"$t^{{pred}}={t_pred:.2f}\,$s" "\n"
            rf"$\eta^{{true}}={eta_true:.3f}\,$m, "
            rf"$\eta^{{pred}}={eta_pred:.3f}\,$m"
        )
        ax.text(
            0.02,
            0.94,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="0.65",
                alpha=0.88,
            ),
        )

        # ---------------------------------------------------------------------
        # Arrow 1: Delta t_c, horizontal double arrow near the top
        # ---------------------------------------------------------------------
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin
        y_arrow_t = ymax - 0.12 * yrange

        x0 = min(t_true, t_pred)
        x1 = max(t_true, t_pred)

        # If the two times are too close, draw a small visible arrow around t_true.
        if abs(x1 - x0) < 1e-5:
            x0 = t_true - 0.03 * (zoom_t1 - zoom_t0)
            x1 = t_true + 0.03 * (zoom_t1 - zoom_t0)

        ax.annotate(
            "",
            xy=(x0, y_arrow_t),
            xytext=(x1, y_arrow_t),
            arrowprops=dict(arrowstyle="<->", color=color, lw=1.1),
            zorder=8,
        )
        ax.text(
            0.5 * (x0 + x1),
            y_arrow_t + 0.035 * yrange,
            rf"$\Delta t_c={delta_t:+.2f}\,$s",
            color=color,
            ha="center",
            va="bottom",
            fontsize=8.0,
            bbox=dict(
                boxstyle="round,pad=0.16",
                facecolor="white",
                edgecolor="none",
                alpha=0.80,
            ),
            zorder=8,
        )

        # ---------------------------------------------------------------------
        # Arrow 2: Delta eta_c, vertical double arrow near the predicted crest
        # ---------------------------------------------------------------------
        x_arrow_eta = min(
            max(t_pred + 0.06 * (zoom_t1 - zoom_t0), zoom_t0 + 0.05 * (zoom_t1 - zoom_t0)),
            zoom_t1 - 0.06 * (zoom_t1 - zoom_t0),
        )

        y0 = min(eta_true, eta_pred)
        y1 = max(eta_true, eta_pred)

        # If the two crest heights are too close, draw a small visible arrow.
        if abs(y1 - y0) < 1e-8:
            y0 = eta_true - 0.025 * yrange
            y1 = eta_true + 0.025 * yrange

        ax.annotate(
            "",
            xy=(x_arrow_eta, y0),
            xytext=(x_arrow_eta, y1),
            arrowprops=dict(arrowstyle="<->", color=color, lw=1.1),
            zorder=8,
        )
        ax.text(
            x_arrow_eta + 0.025 * (zoom_t1 - zoom_t0),
            0.5 * (y0 + y1),
            rf"$\Delta \eta={delta_eta:+.3f}\,$m",
            color=color,
            ha="left",
            va="center",
            fontsize=8.0,
            bbox=dict(
                boxstyle="round,pad=0.16",
                facecolor="white",
                edgecolor="none",
                alpha=0.80,
            ),
            zorder=8,
        )

    axes[-1].set_xlabel("Time / s")

    # fig.suptitle(
    #     f"Local crest comparison at P{point_index + 1} {point_coord} "
    #     f"for Sample {sample_index + 1}",
    #     fontsize=11.5,
    #     y=0.985,
    # )

    fig.subplots_adjust(
        left=0.105,
        right=0.975,
        bottom=0.065,
        top=0.93,
        hspace=0.42,
    )

    out_png = out_dir / (
        f"sample{sample_index + 1:02d}_point{point_index + 1:02d}_local_crest_column.png"
    )

    fig.savefig(out_png)
    plt.close(fig)
    print(f"Saved: {out_png}")
    csv_path = out_dir / (
        f"sample{sample_index + 1:02d}_point{point_index + 1:02d}_crest_metrics.csv"
    )

    fieldnames = [
        "sample_index_0based",
        "sample_id_1based",
        "point_index_0based",
        "point_id_1based",
        "point_coord",
        "experiment",
        "t_true_crest_sec",
        "eta_true_crest_m",
        "t_pred_crest_sec",
        "eta_pred_crest_m",
        "delta_t_sec",
        "delta_eta_m",
        "crest_ratio_pred_over_true",
        "source_file",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved: {csv_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one-point local crest zoom-in figures for multiple experiments."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing MAT files.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="0-based sample index. Example: 0 means the first sample.",
    )
    parser.add_argument(
        "--point_index",
        type=int,
        default=0,
        help="0-based point index among 5 points. Example: 0~4.",
    )
    parser.add_argument(
        "--dominant_search_t0",
        type=float,
        default=None,
        help="Optional start time in seconds for searching the true dominant crest.",
    )
    parser.add_argument(
        "--dominant_search_t1",
        type=float,
        default=None,
        help="Optional end time in seconds for searching the true dominant crest.",
    )
    parser.add_argument(
        "--crest_half_window_sec",
        type=float,
        default=5.0,
        help="Half window in seconds for matching predicted crest around true crest.",
    )
    parser.add_argument(
        "--zoom_half_width_sec",
        type=float,
        default=6.0,
        help="Half width in seconds of the local zoom window around true crest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_matplotlib()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / "figures_one_point_local_crest_column"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_index < 0:
        raise ValueError(f"sample_index must be non-negative, got {args.sample_index}")

    if args.point_index < 0:
        raise ValueError(f"point_index must be non-negative, got {args.point_index}")

    if args.crest_half_window_sec <= 0:
        raise ValueError(f"crest_half_window_sec must be positive, got {args.crest_half_window_sec}")

    if args.zoom_half_width_sec <= 0:
        raise ValueError(f"zoom_half_width_sec must be positive, got {args.zoom_half_width_sec}")

    if args.dominant_search_t0 is not None and args.dominant_search_t1 is not None:
        dominant_search_range = (args.dominant_search_t0, args.dominant_search_t1)
    else:
        dominant_search_range = None

    loaded, true_series, point_coord = load_available_experiments(
        data_dir=data_dir,
        sample_index=int(args.sample_index),
        point_index=int(args.point_index),
    )

    plot_one_point_local_crest_column(
        loaded=loaded,
        true_series=true_series,
        point_coord=point_coord,
        sample_index=int(args.sample_index),
        point_index=int(args.point_index),
        out_dir=out_dir,
        dominant_search_range=dominant_search_range,
        crest_half_window_sec=float(args.crest_half_window_sec),
        zoom_half_width_sec=float(args.zoom_half_width_sec),
    )


if __name__ == "__main__":
    main()