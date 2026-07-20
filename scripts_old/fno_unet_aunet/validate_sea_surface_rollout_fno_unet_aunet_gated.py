import csv
import argparse
import inspect
import math
import os
import time
from typing import Any, Dict, List, Sequence, Tuple
import sys
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

from pathlib import Path

def _find_project_root(start_file: Path) -> Path:
    """Find project root containing both neuralop/ and config/."""
    start = start_file.resolve()
    for parent in [start.parent, *start.parents]:
        if (parent / "neuralop").exists() and (parent / "config").exists():
            return parent
    # Fallback compatible with the original scripts placed under scripts/*/.
    return start.parent.parent.parent

project_root = _find_project_root(Path(__file__))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from neuralop.data.datasets.sea_surface_simple import SeaSurfaceSimpleDataset
from config.sea_surface_rollout_config_fno_unet_aunet_gated import SeaSurfaceRolloutConfig

try:
    from neuralop.models.fno import FNO, TFNO
    from neuralop.models.fno_unet_aunet_gated_decoder import FNOGlobalUNetGatedDecoder
except ImportError as e:
    raise ImportError("Please install neuralop before running this script.") from e


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_csv_row(csv_path: str, row: Dict[str, Any]) -> None:
    """
    Append one row to CSV safely.

    Compatible with both normal validation and timing validation.
    It can:
      1) create a new CSV;
      2) add new columns to an existing CSV header;
      3) repair malformed old rows where csv.DictReader creates key None.
    """
    ensure_dir(os.path.dirname(csv_path) or ".")
    row = {str(k): v for k, v in dict(row).items()}

    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
        return

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fieldnames = [str(k) for k in (reader.fieldnames or []) if k is not None]
        old_rows = list(reader)

    new_keys = [k for k in row.keys() if k not in old_fieldnames]
    fieldnames = old_fieldnames + new_keys

    cleaned_old_rows = []
    for old_row in old_rows:
        old_row = dict(old_row)

        extra_values = old_row.pop(None, [])
        if extra_values is None:
            extra_values = []
        if not isinstance(extra_values, list):
            extra_values = [extra_values]

        cleaned = {k: old_row.get(k, "") for k in old_fieldnames}

        for k, v in zip(new_keys, extra_values):
            cleaned[k] = v
        for k in new_keys:
            cleaned.setdefault(k, "")

        cleaned_old_rows.append(cleaned)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for old_row in cleaned_old_rows:
            writer.writerow(old_row)
        writer.writerow(row)

def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA before/after timing because GPU kernels are asynchronous."""
    if isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _filter_model_kwargs(model_cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(model_cls)
    return {k: v for k, v in kwargs.items() if v is not None and k in sig.parameters}


def _resolve_decoder_type(config: SeaSurfaceRolloutConfig) -> str:
    arch = str(config.model_arch).lower()
    if arch in {"fno_aunet_gated_decoder", "fno_attention_unet_gated_decoder"}:
        return "aunet"
    if arch in {"fno_unet_gated_decoder", "fno_global_unet_gated_decoder"}:
        return "unet"
    return str(getattr(config, "fno_unet_refiner_type", "unet")).lower()


def _optional_int(value):
    if value is None:
        return None
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return None
    return value_int if value_int > 0 else None


def build_model(config: SeaSurfaceRolloutConfig) -> torch.nn.Module:
    arch = str(config.model_arch).lower()

    if arch in {
        "fno_unet_gated_decoder",
        "fno_aunet_gated_decoder",
        "fno_attention_unet_gated_decoder",
        "fno_global_unet_gated_decoder",
    }:
        return FNOGlobalUNetGatedDecoder(
            n_modes=tuple(config.n_modes),
            hidden_channels=int(config.hidden_channels),
            in_channels=int(config.input_steps),
            out_channels=int(config.output_steps),
            n_layers=int(config.n_layers),
            lifting_channels=int(config.lifting_channels),
            projection_channels=int(config.projection_channels),
            fno_arch=str(getattr(config, "fno_unet_fno_arch", "fno")),
            decoder_type=_resolve_decoder_type(config),
            decoder_base_channels=int(getattr(config, "fno_unet_base_channels", 32)),
            unet_depth=int(getattr(config, "fno_unet_depth", 3)),
            unet_dropout=float(getattr(config, "fno_unet_decoder_dropout", 0.0)),
            use_context=bool(getattr(config, "fno_unet_use_context", True)),
            use_residual=bool(getattr(config, "fno_unet_use_residual", True)),
            residual_scale=float(getattr(config, "fno_unet_residual_scale", 1.0)),
            use_gated_residual=bool(getattr(config, "fno_unet_use_gated_residual", True)),
            gate_hidden_channels=int(getattr(config, "fno_unet_gate_hidden_channels", 32)),
            gate_bias_init=float(getattr(config, "fno_unet_gate_bias_init", 0.0)),
            attention_inter_channels=_optional_int(getattr(config, "fno_aunet_attention_inter_channels", None)),
            padding_mode=str(getattr(config, "fno_unet_padding_mode", "periodic")),
        )

    if arch == "fno":
        model_cls = FNO
    elif arch == "tfno":
        model_cls = TFNO
    else:
        raise ValueError(f"Unsupported model_arch: {config.model_arch}")

    kwargs = dict(
        n_modes=tuple(config.n_modes),
        hidden_channels=int(config.hidden_channels),
        in_channels=int(config.input_steps),
        out_channels=int(config.output_steps),
        n_layers=int(config.n_layers),
        lifting_channels=int(config.lifting_channels),
        projection_channels=int(config.projection_channels),
    )
    return model_cls(**_filter_model_kwargs(model_cls, kwargs))


def restore_config_from_checkpoint(config: SeaSurfaceRolloutConfig, checkpoint: Dict[str, Any]) -> SeaSurfaceRolloutConfig:
    ckpt_cfg = checkpoint.get("config", {})
    if isinstance(ckpt_cfg, dict):
        for k, v in ckpt_cfg.items():
            if hasattr(config, k):
                if k == "n_modes" and isinstance(v, list):
                    v = tuple(v)
                setattr(config, k, v)
    return config


def build_rollout_loader(config: SeaSurfaceRolloutConfig, split: str):
    train_dir = os.path.join(config.data_root, "train")
    eval_dir = os.path.join(config.data_root, split)
    train_norm_dataset = SeaSurfaceSimpleDataset(
        data_dir=train_dir,
        variable=config.variable,
        input_steps=int(config.input_steps),
        output_steps=int(config.output_steps),
        stride=int(config.stride),
        normalize=bool(config.normalize),
    )
    eval_dataset = SeaSurfaceSimpleDataset(
        data_dir=eval_dir,
        variable=config.variable,
        input_steps=int(config.input_steps),
        output_steps=int(config.rollout_steps),
        stride=int(config.rollout_stride),
        normalize=bool(config.normalize),
        mean=train_norm_dataset.mean if config.normalize else None,
        std=train_norm_dataset.std if config.normalize else None,
    )
    batch_size = int(config.val_batch_size if split == "val" else config.test_batch_size)
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory),
    )
    return train_norm_dataset, eval_dataset, loader


@torch.no_grad()
def rollout_predict(model: torch.nn.Module, x_init: torch.Tensor, rollout_steps: int, input_steps: int, one_shot_steps: int) -> torch.Tensor:
    model.eval()     
    context = x_init.clone()
    preds = []
    generated = 0
    while generated < rollout_steps:
        pred_chunk = model(context)
        take = min(one_shot_steps, rollout_steps - generated, pred_chunk.shape[1])
        pred_use = pred_chunk[:, :take]
        preds.append(pred_use)
        generated += take
        context = torch.cat([context, pred_use], dim=1)[:, -input_steps:]
    return torch.cat(preds, dim=1)


def compute_sample_rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    return torch.norm(pred_flat - target_flat, dim=1) / (torch.norm(target_flat, dim=1) + 1e-12)


def build_default_points(h: int, w: int, num_points: int) -> List[Tuple[str, int, int]]:
    cy, cx = h // 2, w // 2
    points = [("center", cy, cx)]
    if num_points >= 5:
        dy = max(1, h // 6)
        dx = max(1, w // 6)
        points += [
            ("up", max(0, cy - dy), cx),
            ("down", min(h - 1, cy + dy), cx),
            ("left", cy, max(0, cx - dx)),
            ("right", cy, min(w - 1, cx + dx)),
        ]
    return points[:num_points]


def extract_point_traces(x: torch.Tensor, points: Sequence[Tuple[str, int, int]]) -> torch.Tensor:
    traces = []
    for _, yy, xx in points:
        traces.append(x[:, :, yy, xx])
    return torch.stack(traces, dim=-1)


def build_radial_index(h: int, w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ky = np.fft.fftfreq(h) * h
    kx = np.fft.rfftfreq(w) * w
    grid_y, grid_x = np.meshgrid(ky, kx, indexing="ij")
    kr = np.sqrt(grid_x ** 2 + grid_y ** 2)
    radial_idx = np.rint(kr).astype(np.int64)
    radial_idx_flat = radial_idx.reshape(-1)
    n_bins = int(radial_idx_flat.max()) + 1
    bin_count = np.bincount(radial_idx_flat, minlength=n_bins).astype(np.float64)
    radial_k = np.arange(n_bins, dtype=np.float32)
    return radial_k, radial_idx_flat, bin_count


def compute_batch_stepwise_radial_spectrum(fields: np.ndarray, radial_idx_flat: np.ndarray, bin_count: np.ndarray) -> np.ndarray:
    fft = np.fft.rfft2(fields, axes=(-2, -1), norm="ortho")
    power = np.abs(fft) ** 2
    power_sum = power.sum(axis=0)
    t_out = power_sum.shape[0]
    out = np.zeros((t_out, len(bin_count)), dtype=np.float64)
    for t in range(t_out):
        out[t] = np.bincount(radial_idx_flat, weights=power_sum[t].reshape(-1), minlength=len(bin_count)) / np.maximum(bin_count, 1.0)
    out /= max(fields.shape[0], 1)
    return out


def get_band_ratio_list(config: SeaSurfaceRolloutConfig) -> List[float]:
    vals = sorted(float(r) for r in config.spectral_band_split_ratios)
    clipped = []
    prev = 0.0
    for r in vals:
        r = min(max(r, max(0.05, prev + 1e-3)), 0.98)
        clipped.append(r)
        prev = r
    return clipped


def compute_band_edges(n_bins: int, config: SeaSurfaceRolloutConfig) -> List[int]:
    ratios = get_band_ratio_list(config)
    n_bands = len(ratios) + 1
    edges = [0]
    prev = 0
    for i, r in enumerate(ratios):
        remaining = n_bands - (i + 1)
        candidate = int(math.floor(n_bins * r))
        candidate = max(candidate, prev + 1)
        candidate = min(candidate, n_bins - remaining)
        edges.append(candidate)
        prev = candidate
    edges.append(n_bins)
    return edges


def init_stat_accumulator(num_steps: int) -> Dict[str, np.ndarray]:
    zeros = lambda: np.zeros(num_steps, dtype=np.float64)
    return {
        "sum1": zeros(),
        "sum2": zeros(),
        "count": zeros(),
        "frame_max_sum": zeros(),
        "frame_min_sum": zeros(),
        "frame_count": zeros(),
        "slope_sq_sum": zeros(),
        "slope_count": zeros(),
    }


def update_stat_accumulator(acc: Dict[str, np.ndarray], fields: np.ndarray) -> None:
    bsz, _, h, w = fields.shape
    flat = fields.reshape(bsz, fields.shape[1], -1).astype(np.float64)
    acc["sum1"] += flat.sum(axis=(0, 2))
    acc["sum2"] += (flat ** 2).sum(axis=(0, 2))
    acc["count"] += float(bsz * flat.shape[-1])
    acc["frame_max_sum"] += flat.max(axis=-1).sum(axis=0)
    acc["frame_min_sum"] += flat.min(axis=-1).sum(axis=0)
    acc["frame_count"] += float(bsz)
    dx = fields[..., :, 1:] - fields[..., :, :-1]
    dy = fields[..., 1:, :] - fields[..., :-1, :]
    acc["slope_sq_sum"] += (dx.astype(np.float64) ** 2).sum(axis=(0, 2, 3))
    acc["slope_sq_sum"] += (dy.astype(np.float64) ** 2).sum(axis=(0, 2, 3))
    acc["slope_count"] += float(bsz * (h * (w - 1) + (h - 1) * w))


def finalize_stats(acc: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    eps = 1e-12
    count = np.maximum(acc["count"], eps)
    mean = acc["sum1"] / count
    ex2 = acc["sum2"] / count
    var = np.maximum(ex2 - mean ** 2, eps)
    std = np.sqrt(var)
    rms = np.sqrt(np.maximum(ex2, eps))
    hs = 4.0 * std
    crest = acc["frame_max_sum"] / np.maximum(acc["frame_count"], eps)
    trough = acc["frame_min_sum"] / np.maximum(acc["frame_count"], eps)
    rms_slope = np.sqrt(acc["slope_sq_sum"] / np.maximum(acc["slope_count"], eps))
    return {
        "std": std.astype(np.float32),
        "rms": rms.astype(np.float32),
        "significant_height": hs.astype(np.float32),
        "crest": crest.astype(np.float32),
        "trough": trough.astype(np.float32),
        "rms_slope": rms_slope.astype(np.float32),
    }


@torch.no_grad()
def evaluate_rollout(model: torch.nn.Module, loader: DataLoader, device: torch.device, config: SeaSurfaceRolloutConfig, data_mean: float, data_std: float, split: str) -> Dict[str, Any]:
    model.eval()

    rollout_steps = int(config.rollout_steps)
    one_shot_steps = int(config.output_steps)
    input_steps = int(config.input_steps)

    total_sqerr = 0.0
    total_count = 0
    total_rel_l2 = 0.0
    total_samples = 0

    step_sqerr = np.zeros(rollout_steps, dtype=np.float64)
    step_count = np.zeros(rollout_steps, dtype=np.int64)

    num_chunks = int(math.ceil(rollout_steps / one_shot_steps))
    chunk_sqerr = np.zeros(num_chunks, dtype=np.float64)
    chunk_count = np.zeros(num_chunks, dtype=np.int64)
    chunk_bounds = [(i * one_shot_steps + 1, min((i + 1) * one_shot_steps, rollout_steps)) for i in range(num_chunks)]

    # 只保存“每个 .mat 文件的第一个有效样本”用于后续逐样本绘图。
    # 说明：SeaSurfaceSimpleDataset.index_map 中每个元素为 (file_id, start_idx)，
    # DataLoader 在 shuffle=False 时会按 index_map 顺序取样，因此某个 file_id 第一次出现
    # 就是该 .mat 文件的第一个样本。
    saved_x = []
    saved_y_true = []
    saved_y_pred = []
    saved_sample_ids = []
    saved_file_ids = []
    saved_start_indices = []
    saved_file_id_set = set()

    # point trace 也只保存上述逐 .mat 文件第一个样本，而不是所有样本。
    point_true_all = []
    point_pred_all = []
    point_specs = None

    eval_dataset = getattr(loader, "dataset", None)
    index_map = getattr(eval_dataset, "index_map", None)

    radial_k = None
    radial_idx_flat = None
    bin_count = None
    step_spectrum_true_sum = None
    step_spectrum_pred_sum = None
    step_spectrum_count = 0.0

    true_stats_acc = init_stat_accumulator(rollout_steps)
    pred_stats_acc = init_stat_accumulator(rollout_steps)
    sample_cursor = 0

    # Timing: only count model rollout inference from the input context to 240-step prediction.
    # Data loading, host-to-device transfer, metric calculation, CPU conversion, and .mat saving are excluded.
    rollout_infer_time_batch_list = []
    rollout_infer_time_total_sec = 0.0
    rollout_infer_num_samples = 0
    rollout_infer_num_batches = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        bsz = int(x.shape[0])

        sync_if_cuda(device)
        t0 = time.perf_counter()
        pred = rollout_predict(model, x, rollout_steps, input_steps, one_shot_steps)
        sync_if_cuda(device)
        infer_dt = time.perf_counter() - t0

        rollout_infer_time_batch_list.append(float(infer_dt))
        rollout_infer_time_total_sec += float(infer_dt)
        rollout_infer_num_samples += bsz
        rollout_infer_num_batches += 1

        bsz, _, h, w = pred.shape
        if point_specs is None:
            point_specs = build_default_points(h, w, int(config.evaluation_num_trace_points))
        if radial_k is None:
            radial_k, radial_idx_flat, bin_count = build_radial_index(h, w)
            step_spectrum_true_sum = np.zeros((rollout_steps, len(radial_k)), dtype=np.float64)
            step_spectrum_pred_sum = np.zeros((rollout_steps, len(radial_k)), dtype=np.float64)

        sq_err = (pred - y) ** 2
        total_sqerr += sq_err.sum().item()
        total_count += sq_err.numel()

        rel_l2_batch = compute_sample_rel_l2(pred, y)
        total_rel_l2 += rel_l2_batch.sum().item()
        total_samples += bsz

        for t in range(rollout_steps):
            step_sqerr[t] += sq_err[:, t].sum().item()
            step_count[t] += sq_err[:, t].numel()
        for i in range(num_chunks):
            s = i * one_shot_steps
            e = min((i + 1) * one_shot_steps, rollout_steps)
            chunk_sqerr[i] += sq_err[:, s:e].sum().item()
            chunk_count[i] += sq_err[:, s:e].numel()

        point_true_batch = extract_point_traces(y, point_specs).cpu().numpy().astype(np.float32)
        point_pred_batch = extract_point_traces(pred, point_specs).cpu().numpy().astype(np.float32)

        for local_idx in range(bsz):
            global_sample_idx = sample_cursor + local_idx
            if index_map is not None and global_sample_idx < len(index_map):
                file_id, start_idx = index_map[global_sample_idx]
            else:
                # 兜底：如果后续换成没有 index_map 的数据集，就退化为逐全局样本保存。
                file_id, start_idx = global_sample_idx, -1

            file_id = int(file_id)
            start_idx = int(start_idx)
            if file_id in saved_file_id_set:
                continue

            saved_file_id_set.add(file_id)
            saved_x.append(x[local_idx:local_idx + 1].cpu().numpy().astype(np.float32))
            saved_y_true.append(y[local_idx:local_idx + 1].cpu().numpy().astype(np.float32))
            saved_y_pred.append(pred[local_idx:local_idx + 1].cpu().numpy().astype(np.float32))
            saved_sample_ids.append(global_sample_idx)
            saved_file_ids.append(file_id)
            saved_start_indices.append(start_idx)
            point_true_all.append(point_true_batch[local_idx:local_idx + 1])
            point_pred_all.append(point_pred_batch[local_idx:local_idx + 1])

        sample_cursor += bsz

        y_np = y.cpu().numpy().astype(np.float32)
        pred_np = pred.cpu().numpy().astype(np.float32)
        if config.normalize:
            y_phy = y_np * data_std + data_mean
            pred_phy = pred_np * data_std + data_mean
        else:
            y_phy = y_np
            pred_phy = pred_np

        step_spectrum_true_sum += compute_batch_stepwise_radial_spectrum(y_phy, radial_idx_flat, bin_count) * bsz
        step_spectrum_pred_sum += compute_batch_stepwise_radial_spectrum(pred_phy, radial_idx_flat, bin_count) * bsz
        step_spectrum_count += float(bsz)
        update_stat_accumulator(true_stats_acc, y_phy)
        update_stat_accumulator(pred_stats_acc, pred_phy)

    rmse = float(np.sqrt(total_sqerr / max(total_count, 1)))
    rel_l2 = float(total_rel_l2 / max(total_samples, 1))
    stepwise_rmse = np.sqrt(step_sqerr / np.maximum(step_count, 1)).astype(np.float32)
    chunk_rmse = np.sqrt(chunk_sqerr / np.maximum(chunk_count, 1)).astype(np.float32)
    final_step_rmse = float(stepwise_rmse[-1])
    last_chunk_rmse = float(chunk_rmse[-1])

    rollout_infer_time_batch_arr = np.asarray(rollout_infer_time_batch_list, dtype=np.float64)
    rollout_infer_time_per_batch_sec = (
        float(np.mean(rollout_infer_time_batch_arr))
        if rollout_infer_time_batch_arr.size > 0 else float("nan")
    )
    rollout_infer_time_per_sample_sec = (
        float(rollout_infer_time_total_sec / rollout_infer_num_samples)
        if rollout_infer_num_samples > 0 else float("nan")
    )

    step_spectrum_true = (step_spectrum_true_sum / max(step_spectrum_count, 1.0)).astype(np.float32)
    step_spectrum_pred = (step_spectrum_pred_sum / max(step_spectrum_count, 1.0)).astype(np.float32)
    global_spectrum_true = step_spectrum_true.mean(axis=0).astype(np.float32)
    global_spectrum_pred = step_spectrum_pred.mean(axis=0).astype(np.float32)

    high_k_start_idx = int(max(1, math.floor((len(radial_k) - 1) * float(config.spectral_high_k_ratio))))
    high_mask = radial_k >= float(high_k_start_idx)
    spectral_rel_l2_global = float(np.linalg.norm(global_spectrum_pred - global_spectrum_true) / (np.linalg.norm(global_spectrum_true) + 1e-12))
    spectral_rel_l2_high = float(np.linalg.norm(global_spectrum_pred[high_mask] - global_spectrum_true[high_mask]) / (np.linalg.norm(global_spectrum_true[high_mask]) + 1e-12)) if np.any(high_mask) else 0.0

    band_edges = compute_band_edges(len(radial_k), config)
    band_energy_true = np.stack([step_spectrum_true[:, band_edges[i]:band_edges[i + 1]].sum(axis=1) for i in range(len(band_edges) - 1)], axis=1).astype(np.float32)
    band_energy_pred = np.stack([step_spectrum_pred[:, band_edges[i]:band_edges[i + 1]].sum(axis=1) for i in range(len(band_edges) - 1)], axis=1).astype(np.float32)

    true_stats = finalize_stats(true_stats_acc)
    pred_stats = finalize_stats(pred_stats_acc)
    std_mae = float(np.mean(np.abs(pred_stats["std"] - true_stats["std"])))
    hs_mae = float(np.mean(np.abs(pred_stats["significant_height"] - true_stats["significant_height"])))
    rms_slope_mae = float(np.mean(np.abs(pred_stats["rms_slope"] - true_stats["rms_slope"])))

    point_names = np.asarray([name for name, _, _ in point_specs], dtype=object)
    point_coords = np.asarray([[yy, xx] for _, yy, xx in point_specs], dtype=np.int32)

    result = {
        "split": split,
        "rmse": np.array(rmse, dtype=np.float32),
        "rel_l2": np.array(rel_l2, dtype=np.float32),
        "final_step_rmse": np.array(final_step_rmse, dtype=np.float32),
        "last_chunk_rmse": np.array(last_chunk_rmse, dtype=np.float32),
        "stepwise_rmse": stepwise_rmse,
        "chunk_rmse": chunk_rmse,
        "chunk_bounds": np.asarray(chunk_bounds, dtype=np.int32),
        "radial_k": radial_k.astype(np.float32),
        "global_spectrum_true": global_spectrum_true,
        "global_spectrum_pred": global_spectrum_pred,
        "step_spectrum_true": step_spectrum_true,
        "step_spectrum_pred": step_spectrum_pred,
        "spectral_rel_l2_global": np.array(spectral_rel_l2_global, dtype=np.float32),
        "spectral_rel_l2_high": np.array(spectral_rel_l2_high, dtype=np.float32),
        "high_k_start_idx": np.array(high_k_start_idx, dtype=np.int32),
        "band_energy_true": band_energy_true,
        "band_energy_pred": band_energy_pred,
        "step_stat_true_std": true_stats["std"],
        "step_stat_pred_std": pred_stats["std"],
        "step_stat_true_hs": true_stats["significant_height"],
        "step_stat_pred_hs": pred_stats["significant_height"],
        "step_stat_true_rms_slope": true_stats["rms_slope"],
        "step_stat_pred_rms_slope": pred_stats["rms_slope"],
        "std_mae": np.array(std_mae, dtype=np.float32),
        "hs_mae": np.array(hs_mae, dtype=np.float32),
        "rms_slope_mae": np.array(rms_slope_mae, dtype=np.float32),
        "x_plot": np.concatenate(saved_x, axis=0) if saved_x else np.empty((0, input_steps, 0, 0), dtype=np.float32),
        "y_true_plot": np.concatenate(saved_y_true, axis=0) if saved_y_true else np.empty((0, rollout_steps, 0, 0), dtype=np.float32),
        "y_pred_plot": np.concatenate(saved_y_pred, axis=0) if saved_y_pred else np.empty((0, rollout_steps, 0, 0), dtype=np.float32),
        "saved_sample_ids": np.asarray(saved_sample_ids, dtype=np.int32),
        "saved_file_ids": np.asarray(saved_file_ids, dtype=np.int32),
        "saved_start_indices": np.asarray(saved_start_indices, dtype=np.int32),
        "num_saved_plot_samples": np.array(len(saved_sample_ids), dtype=np.int32),
        "point_true_all": np.concatenate(point_true_all, axis=0) if point_true_all else np.empty((0, rollout_steps, 0), dtype=np.float32),
        "point_pred_all": np.concatenate(point_pred_all, axis=0) if point_pred_all else np.empty((0, rollout_steps, 0), dtype=np.float32),
        "point_names": point_names,
        "point_coords": point_coords,
        "data_mean": np.array(float(data_mean), dtype=np.float32),
        "data_std": np.array(float(data_std), dtype=np.float32),
        "normalize": np.array(int(config.normalize), dtype=np.int32),
        "dt": np.array(float(config.dt), dtype=np.float32),
        "rollout_steps": np.array(int(config.rollout_steps), dtype=np.int32),
        "one_shot_steps": np.array(int(config.output_steps), dtype=np.int32),
        "rollout_infer_time_total_sec": np.array(float(rollout_infer_time_total_sec), dtype=np.float64),
        "rollout_infer_time_per_sample_sec": np.array(float(rollout_infer_time_per_sample_sec), dtype=np.float64),
        "rollout_infer_time_per_batch_sec": np.array(float(rollout_infer_time_per_batch_sec), dtype=np.float64),
        "rollout_infer_time_batch_list_sec": rollout_infer_time_batch_arr.astype(np.float64),
        "rollout_infer_num_samples": np.array(int(rollout_infer_num_samples), dtype=np.int32),
        "rollout_infer_num_batches": np.array(int(rollout_infer_num_batches), dtype=np.int32),
        "rollout_infer_steps": np.array(int(rollout_steps), dtype=np.int32),
        "experiment_name": np.asarray(config.experiment_name, dtype=object),
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Validate/test FNO + single U-Net/AU-Net gated residual rollout model")
    parser.add_argument("split", nargs="?", default="val", choices=["val", "test"])
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    return parser.parse_args()


def main(split: str = "val", experiment_name: str = None, checkpoint_dir: str = None, checkpoint_path: str = None) -> None:
    config = SeaSurfaceRolloutConfig()
    if experiment_name is not None:
        config.experiment_name = experiment_name
    if checkpoint_dir is not None:
        config.checkpoint_dir = checkpoint_dir
    ckpt_path = checkpoint_path if checkpoint_path is not None else config.checkpoint_path

    device = get_device(config.device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = restore_config_from_checkpoint(config, checkpoint)
    model = build_model(config).to(device)
    state_dict = checkpoint["model_state_dict"]
    if isinstance(state_dict, dict) and "_metadata" in state_dict:
        state_dict = {k: v for k, v in state_dict.items() if k != "_metadata"}
    model.load_state_dict(state_dict, strict=True)

    train_norm_dataset, _, loader = build_rollout_loader(config, split=split)
    result = evaluate_rollout(model, loader, device, config, train_norm_dataset.mean, train_norm_dataset.std, split)

    if split == "val" and config.save_val_mat:
        sio.savemat(config.val_mat_path, result)
        print(f"Saved val mat to: {config.val_mat_path}")
    if split == "test" and config.save_test_mat:
        sio.savemat(config.test_mat_path, result)
        print(f"Saved test mat to: {config.test_mat_path}")

    summary_row = {
        "experiment_name": config.experiment_name,
        "model_arch": config.model_arch,
        "fno_unet_refiner_type": _resolve_decoder_type(config),
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
        "rollout_infer_time_total_sec": float(result["rollout_infer_time_total_sec"]),
        "rollout_infer_time_per_sample_sec": float(result["rollout_infer_time_per_sample_sec"]),
        "rollout_infer_time_per_batch_sec": float(result["rollout_infer_time_per_batch_sec"]),
        "rollout_infer_num_samples": int(result["rollout_infer_num_samples"]),
        "rollout_infer_num_batches": int(result["rollout_infer_num_batches"]),
        "rollout_infer_steps": int(result["rollout_infer_steps"]),
    }
    print(
        f"Rollout inference time ({split}) | "
        f"total={summary_row['rollout_infer_time_total_sec']:.6f}s | "
        f"per_sample={summary_row['rollout_infer_time_per_sample_sec']:.6f}s | "
        f"per_batch={summary_row['rollout_infer_time_per_batch_sec']:.6f}s | "
        f"num_samples={summary_row['rollout_infer_num_samples']} | "
        f"num_batches={summary_row['rollout_infer_num_batches']} | "
        f"rollout_steps={summary_row['rollout_infer_steps']}"
    )
    csv_path = config.val_summary_path if split == "val" else config.test_summary_path
    append_csv_row(csv_path, summary_row)
    print(f"Appended summary to: {csv_path}")


if __name__ == "__main__":
    # Usage examples:
    #   python validate_sea_surface_rollout_fno_unet_aunet_gated.py val --experiment_name fno_single_unet_gated
    #   python validate_sea_surface_rollout_fno_unet_aunet_gated.py test --experiment_name fno_single_aunet_gated
    args = parse_args()
    main(
        split=args.split,
        experiment_name=args.experiment_name,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_path=args.checkpoint_path,
    )
