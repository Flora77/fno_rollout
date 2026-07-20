# -*- coding: utf-8 -*-
"""
Plot local crest zoom-in curves for one selected point and one selected sample.

This version detects multiple local crests in the zoom window, e.g. when
--zoom_half_width_sec 8 covers about three wave crests.

For each available MAT file:
    - use the same selected sample
    - use one selected point from point_true_all / point_pred_all
    - plot True and Pred local crest zoom-in curve
    - detect multiple true local crests within the zoom window
    - match one predicted local crest around each true crest
    - draw long vertical dashed lines for Delta t_c
    - draw long horizontal dashed lines for Delta eta
    - keep compact Delta t_c and Delta eta text in the upper-left box

MAT variables:
    point_true_all : [Ns, T, Np]
    point_pred_all : [Ns, T, Np]
    point_coords   : [Np, 2]

Default data directory:
    E:\hanqi\python\neuraloperator\frequency\sample9

Usage examples:
    python plot_point_trace_grid_multi_crest.py --sample_index 5 --point_index 0 --zoom_half_width_sec 8
    python plot_point_trace_grid_multi_crest.py --sample_index 5 --point_index 2 --zoom_half_width_sec 8 --num_crests 3
    python plot_point_trace_grid_multi_crest.py --sample_index 5 --point_index 2 --dominant_search_t0 46 --dominant_search_t1 55 --zoom_half_width_sec 8 --num_crests 3
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
        "label": "G2 FNO",
        "file": "G2_fno_no_curriculum_rollout300_no_added_loss_val_rollout.mat",
        "color": "#ff7f0e",
    },
    {
        "tag": "(b)",
        "label": "G3 MSFNO",
        "file": "G4_msfno_no_curriculum_rollout300_no_added_loss_val_rollout.mat",
        "color": "#d62728",
    },
    {
        "tag": "(c)",
        "label": "G4 FNO+CR",
        "file": "G1_fno_curriculum_rollout300_no_added_loss_val_rollout.mat",
        "color": "#1f77b4",
    },

    {
        "tag": "(d)",
        "label": "G5 MSFNO+CR",
        "file": "G3_msfno_curriculum_rollout300_no_added_loss_val_rollout.mat",
        "color": "#2ca02c",
    },

    {
        "tag": "(e)",
        "label": "G6 Ours",
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
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 0.85,
        "lines.linewidth": 1.5,
        "figure.dpi": 150,
        "savefig.dpi": 1000,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
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
    True reference crest for centering the zoom window.
    """
    i0, i1 = get_index_range_from_time(t, search_range)
    idx_local = int(np.argmax(series_true[i0:i1 + 1]))
    idx = i0 + idx_local
    return idx, float(t[idx]), float(series_true[idx])


def simple_find_peaks(series: np.ndarray, min_distance_samples: int = 1) -> np.ndarray:
    """
    Lightweight local-maxima detector without scipy.signal dependency.

    A peak satisfies:
        series[i] >= series[i-1] and series[i] > series[i+1]
    Then a greedy non-maximum suppression keeps peaks with a minimum distance.
    """
    series = np.asarray(series)
    n = len(series)
    if n < 3:
        return np.array([], dtype=int)

    candidates = []
    for i in range(1, n - 1):
        if series[i] >= series[i - 1] and series[i] > series[i + 1]:
            candidates.append(i)

    if len(candidates) == 0:
        return np.array([], dtype=int)

    candidates = np.asarray(candidates, dtype=int)

    if min_distance_samples <= 1:
        return candidates

    # Greedy selection from highest to lowest peak.
    order = candidates[np.argsort(series[candidates])[::-1]]
    selected = []
    for idx in order:
        if all(abs(idx - j) >= min_distance_samples for j in selected):
            selected.append(int(idx))

    return np.asarray(sorted(selected), dtype=int)


def detect_true_local_crests(
    true_series: np.ndarray,
    t: np.ndarray,
    zoom_range: tuple[float, float],
    num_crests: int = 3,
    min_distance_sec: float = 2.0,
    prominence: float | None = None,
) -> list[dict]:
    """
    Detect multiple true local crests inside the zoom window.

    If more than num_crests are found, keep the highest num_crests and sort them by time.
    If fewer are found, use all available peaks.
    """
    i0, i1 = get_index_range_from_time(t, zoom_range)
    segment = true_series[i0:i1 + 1]
    min_dist = max(1, int(round(min_distance_sec / DT)))

    local_peak_idx = simple_find_peaks(segment, min_distance_samples=min_dist)

    # Optional prominence-like filtering using neighboring troughs in a small window.
    if prominence is not None and prominence > 0 and len(local_peak_idx) > 0:
        keep = []
        half_win = max(2, min_dist // 2)
        for p in local_peak_idx:
            left = max(0, p - half_win)
            right = min(len(segment) - 1, p + half_win)
            local_min = min(float(np.min(segment[left:p + 1])), float(np.min(segment[p:right + 1])))
            if float(segment[p] - local_min) >= prominence:
                keep.append(int(p))
        local_peak_idx = np.asarray(keep, dtype=int)

    if len(local_peak_idx) == 0:
        # Fallback: use the largest point in the window.
        local_peak_idx = np.asarray([int(np.argmax(segment))], dtype=int)

    global_peak_idx = i0 + local_peak_idx

    if len(global_peak_idx) > num_crests:
        # Keep the most energetic/highest crests.
        order_height = np.argsort(true_series[global_peak_idx])[::-1]
        global_peak_idx = global_peak_idx[order_height[:num_crests]]

    global_peak_idx = np.asarray(sorted(global_peak_idx), dtype=int)

    crests = []
    for k, idx in enumerate(global_peak_idx, start=1):
        crests.append({
            "crest_id": k,
            "idx_true": int(idx),
            "t_true": float(t[idx]),
            "eta_true": float(true_series[idx]),
        })

    return crests


def build_match_windows(
    true_crests: list[dict],
    zoom_range: tuple[float, float],
    match_half_width_sec: float = 2.0,
) -> list[tuple[float, float]]:
    """
    Build one matching window for each true crest.

    The window is centered at the true crest. It is clipped by:
        - half distance to neighboring true crests
        - zoom window boundary
        - match_half_width_sec
    This reduces the chance of matching a predicted crest from a neighboring wave.
    """
    if len(true_crests) == 0:
        return []

    t_crests = np.array([c["t_true"] for c in true_crests], dtype=np.float64)
    windows = []

    for i, tc in enumerate(t_crests):
        if i == 0:
            left_mid = zoom_range[0]
        else:
            left_mid = 0.5 * (t_crests[i - 1] + tc)

        if i == len(t_crests) - 1:
            right_mid = zoom_range[1]
        else:
            right_mid = 0.5 * (tc + t_crests[i + 1])

        left = max(zoom_range[0], left_mid, tc - match_half_width_sec)
        right = min(zoom_range[1], right_mid, tc + match_half_width_sec)

        if right <= left:
            left = max(zoom_range[0], tc - match_half_width_sec)
            right = min(zoom_range[1], tc + match_half_width_sec)

        windows.append((float(left), float(right)))

    return windows


def find_pred_crest_in_window(
    pred_series: np.ndarray,
    t: np.ndarray,
    search_range: tuple[float, float],
) -> tuple[int, float, float]:
    i0, i1 = get_index_range_from_time(t, search_range)
    idx_local = int(np.argmax(pred_series[i0:i1 + 1]))
    idx = i0 + idx_local
    return idx, float(t[idx]), float(pred_series[idx])


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
# Plot helpers
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


def format_multi_crest_text(crest_metrics: list[dict]) -> str:
    """
    Compact upper-left label. Example:
        C1: Δt=+0.00s, Δη=-0.12m
        C2: Δt=+0.25s, Δη=+0.08m
    """
    lines = []
    for m in crest_metrics:
        lines.append(
            rf"C{m['crest_id']}: "
            rf"$\Delta t={m['delta_t']:+.2f}\,$s, "
            rf"$\Delta\eta={m['delta_eta']:+.3f}\,$m"
        )
    return "\n".join(lines)


# =============================================================================
# Plot
# =============================================================================

def plot_one_point_multi_crest_grid(
    loaded: list[dict],
    true_series: np.ndarray,
    point_coord: tuple[int, int],
    sample_index: int,
    point_index: int,
    out_dir: Path,
    dominant_search_range=None,
    zoom_half_width_sec: float = 8.0,
    num_crests: int = 3,
    crest_min_distance_sec: float = 2.0,
    crest_prominence: float | None = None,
    match_half_width_sec: float = 2.0,
    layout: str = "grid",
    ncols: int = 3,
) -> None:
    n_time = len(true_series)
    t = np.arange(n_time, dtype=np.float64) * DT

    # The dominant/reference crest is used only to center the zoom window.
    _, t_center, _ = find_true_crest(true_series, t, search_range=dominant_search_range)

    zoom_t0 = max(t[0], t_center - zoom_half_width_sec)
    zoom_t1 = min(t[-1], t_center + zoom_half_width_sec)
    zoom_range = (zoom_t0, zoom_t1)
    zoom_mask = (t >= zoom_t0) & (t <= zoom_t1)
    zoom_ylim = compute_common_zoom_ylim(true_series, loaded, t, zoom_range)

    true_crests = detect_true_local_crests(
        true_series=true_series,
        t=t,
        zoom_range=zoom_range,
        num_crests=num_crests,
        min_distance_sec=crest_min_distance_sec,
        prominence=crest_prominence,
    )
    match_windows = build_match_windows(
        true_crests=true_crests,
        zoom_range=zoom_range,
        match_half_width_sec=match_half_width_sec,
    )

    n_exp = len(loaded)
    if n_exp <= 0:
        raise ValueError("No experiments to plot.")

    layout = str(layout).lower()
    if layout == "column":
        ncols_eff = 1
        nrows = n_exp
        figsize = (8.4, max(2.15 * n_exp, 3.0))
    else:
        ncols_eff = max(1, int(ncols))
        ncols_eff = min(ncols_eff, n_exp)
        nrows = int(np.ceil(n_exp / ncols_eff))
        figsize = (4.65 * ncols_eff, 2.45 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols_eff,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes_arr = np.asarray(axes, dtype=object).reshape(-1)

    csv_rows = []

    for row, item in enumerate(loaded):
        ax = axes_arr[row]
        pred_series = item["pred"]
        color = item["color"]

        # Curves
        ax.plot(
            t[zoom_mask],
            true_series[zoom_mask],
            color="black",
            linewidth=2.0,
            linestyle="-",
            label="Ref",
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

        crest_metrics = []

        for crest, win in zip(true_crests, match_windows):
            cid = crest["crest_id"]
            idx_true = crest["idx_true"]
            t_true = crest["t_true"]
            eta_true = crest["eta_true"]

            idx_pred, t_pred, eta_pred = find_pred_crest_in_window(
                pred_series=pred_series,
                t=t,
                search_range=win,
            )

            delta_t = t_pred - t_true
            delta_eta = eta_pred - eta_true

            crest_metrics.append({
                "crest_id": cid,
                "t_true": t_true,
                "eta_true": eta_true,
                "t_pred": t_pred,
                "eta_pred": eta_pred,
                "delta_t": delta_t,
                "delta_eta": delta_eta,
                "match_t0": win[0],
                "match_t1": win[1],
            })

            csv_rows.append({
                "sample_index_0based": sample_index,
                "sample_id_1based": sample_index + 1,
                "point_index_0based": point_index,
                "point_id_1based": point_index + 1,
                "point_coord": point_coord,
                "experiment": item["label"],
                "crest_id": cid,
                "match_t0_sec": f"{win[0]:.6f}",
                "match_t1_sec": f"{win[1]:.6f}",
                "t_true_crest_sec": f"{t_true:.6f}",
                "eta_true_crest_m": f"{eta_true:.8f}",
                "t_pred_crest_sec": f"{t_pred:.6f}",
                "eta_pred_crest_m": f"{eta_pred:.8f}",
                "delta_t_sec": f"{delta_t:.6f}",
                "delta_eta_m": f"{delta_eta:.8f}",
                "crest_ratio_pred_over_true": f"{eta_pred / (eta_true + EPS):.8f}",
                "source_file": item["file"],
            })

            # Crest markers.
            ax.plot(t_true, eta_true, "o", color="black", markersize=4.6, zorder=8)
            ax.plot(t_pred, eta_pred, "^", color=color, markersize=5.2, zorder=8)

            # Delta t_c: long vertical dashed lines.
            ax.axvline(
                t_true,
                color="black",
                linestyle="--",
                linewidth=0.75,
                alpha=0.42,
                zorder=1,
            )
            if abs(delta_t) > 1e-10:
                ax.axvline(
                    t_pred,
                    color=color,
                    linestyle="--",
                    linewidth=0.85,
                    alpha=0.58,
                    zorder=1,
                )

            # Delta eta: long horizontal dashed lines.
            ax.hlines(
                eta_true,
                xmin=zoom_t0,
                xmax=zoom_t1,
                colors="black",
                linestyles="--",
                linewidth=0.70,
                alpha=0.25,
                zorder=2,
            )
            if abs(delta_eta) > 1e-10:
                ax.hlines(
                    eta_pred,
                    xmin=zoom_t0,
                    xmax=zoom_t1,
                    colors=color,
                    linestyles="--",
                    linewidth=0.78,
                    alpha=0.42,
                    zorder=2,
                )

        # Optional dominant-search window markers for transparency.
        if dominant_search_range is not None:
            ax.axvline(dominant_search_range[0], color="0.70", linestyle=":", linewidth=0.75, alpha=0.55, zorder=0)
            ax.axvline(dominant_search_range[1], color="0.70", linestyle=":", linewidth=0.75, alpha=0.55, zorder=0)

        ax.set_xlim(zoom_t0, zoom_t1)
        ax.set_ylim(*zoom_ylim)
        ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.28)
        ax.tick_params(direction="out", length=3, width=0.7)

        ax.set_title(
            f"{item['tag']} {item['label']}",
            loc="left",
            pad=4,
            fontweight="normal",
        )

        ax.legend(
            loc="upper right",
            frameon=False,
            handlelength=2.0,
            borderaxespad=0.2,
        )

        # Compact text box: only Delta t_c and Delta eta for each detected crest.
        text = format_multi_crest_text(crest_metrics)
        ax.text(
            0.02,
            0.93,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            linespacing=1.15,
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="white",
                edgecolor="0.65",
                alpha=0.86,
            ),
            zorder=20,
        )

    # Hide unused panels.
    for k in range(n_exp, len(axes_arr)):
        axes_arr[k].axis("off")

    # Axis labels: only left column and bottom row.
    for i, ax in enumerate(axes_arr[:n_exp]):
        grid_row = i // ncols_eff
        grid_col = i % ncols_eff

        if grid_col == 0:
            ax.set_ylabel(r"$\eta$ / m")
        else:
            ax.tick_params(labelleft=False)

        if grid_row == nrows - 1:
            ax.set_xlabel(r"$t$ / s")
        else:
            ax.tick_params(labelbottom=False)

    if layout == "column":
        fig.subplots_adjust(
            left=0.105,
            right=0.975,
            bottom=0.065,
            top=0.93,
            hspace=0.42,
        )
        suffix = "column"
    else:
        fig.subplots_adjust(
            left=0.065,
            right=0.985,
            bottom=0.095,
            top=0.89,
            wspace=0.03,
            hspace=0.15,
        )
        suffix = f"grid_{nrows}x{ncols_eff}"

    out_png = out_dir / (
        f"sample{sample_index + 1:02d}_point{point_index + 1:02d}_multi_crest_{suffix}.png"
    )
    out_pdf = out_png.with_suffix(".pdf")
    out_svg = out_png.with_suffix(".svg")
    fig.savefig(out_png, dpi=1000)
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"Saved: {out_png}")
    csv_path = out_dir / (
        f"sample{sample_index + 1:02d}_point{point_index + 1:02d}_multi_crest_metrics.csv"
    )

    fieldnames = [
        "sample_index_0based",
        "sample_id_1based",
        "point_index_0based",
        "point_id_1based",
        "point_coord",
        "experiment",
        "crest_id",
        "match_t0_sec",
        "match_t1_sec",
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
        description="Plot multi-crest local zoom-in figures for multiple experiments."
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
        help="Optional start time in seconds for searching the central/reference crest.",
    )
    parser.add_argument(
        "--dominant_search_t1",
        type=float,
        default=None,
        help="Optional end time in seconds for searching the central/reference crest.",
    )
    parser.add_argument(
        "--zoom_half_width_sec",
        type=float,
        default=8.0,
        help="Half width in seconds of the local zoom window around the central true crest.",
    )
    parser.add_argument(
        "--num_crests",
        type=int,
        default=3,
        help="Number of true local crests to detect in the zoom window.",
    )
    parser.add_argument(
        "--crest_min_distance_sec",
        type=float,
        default=2.0,
        help="Minimum time distance between detected true crests.",
    )
    parser.add_argument(
        "--crest_prominence",
        type=float,
        default=None,
        help="Optional minimum local prominence for true crest detection. Default: None.",
    )
    parser.add_argument(
        "--match_half_width_sec",
        type=float,
        default=2.0,
        help="Maximum half-width of each predicted-crest matching window around each true crest.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="grid",
        choices=["grid", "column"],
        help="Figure layout. 'grid' is recommended for papers; 'column' uses one column.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of columns when layout='grid'. For 5 experiments, ncols=3 gives a compact 2x3 layout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_matplotlib()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / "figures_one_point_multi_crest"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_index < 0:
        raise ValueError(f"sample_index must be non-negative, got {args.sample_index}")

    if args.point_index < 0:
        raise ValueError(f"point_index must be non-negative, got {args.point_index}")

    if args.zoom_half_width_sec <= 0:
        raise ValueError(f"zoom_half_width_sec must be positive, got {args.zoom_half_width_sec}")

    if args.num_crests <= 0:
        raise ValueError(f"num_crests must be positive, got {args.num_crests}")

    if args.crest_min_distance_sec <= 0:
        raise ValueError(f"crest_min_distance_sec must be positive, got {args.crest_min_distance_sec}")

    if args.match_half_width_sec <= 0:
        raise ValueError(f"match_half_width_sec must be positive, got {args.match_half_width_sec}")

    if args.ncols <= 0:
        raise ValueError(f"ncols must be positive, got {args.ncols}")

    if args.dominant_search_t0 is not None and args.dominant_search_t1 is not None:
        dominant_search_range = (args.dominant_search_t0, args.dominant_search_t1)
    else:
        dominant_search_range = None

    loaded, true_series, point_coord = load_available_experiments(
        data_dir=data_dir,
        sample_index=int(args.sample_index),
        point_index=int(args.point_index),
    )

    plot_one_point_multi_crest_grid(
        loaded=loaded,
        true_series=true_series,
        point_coord=point_coord,
        sample_index=int(args.sample_index),
        point_index=int(args.point_index),
        out_dir=out_dir,
        dominant_search_range=dominant_search_range,
        zoom_half_width_sec=float(args.zoom_half_width_sec),
        num_crests=int(args.num_crests),
        crest_min_distance_sec=float(args.crest_min_distance_sec),
        crest_prominence=args.crest_prominence,
        match_half_width_sec=float(args.match_half_width_sec),
        layout=str(args.layout),
        ncols=int(args.ncols),
    )


if __name__ == "__main__":
    main()
