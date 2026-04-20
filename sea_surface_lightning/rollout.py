import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader



def parse_rollout_train_steps(cfg: Dict) -> List[int]:
    rollout_cfg = cfg["rollout"]
    if not rollout_cfg["use_long_rollout_curriculum"]:
        return [int(rollout_cfg["rollout_steps"])]
    steps = sorted({int(s) for s in rollout_cfg["rollout_train_steps"] if int(s) > 0})
    return steps if steps else [int(rollout_cfg["rollout_steps"])]



def parse_curriculum_boundaries(cfg: Dict, num_stages: int) -> List[float]:
    boundaries = [float(v) for v in cfg["rollout"]["rollout_curriculum_boundaries"]]
    if len(boundaries) != num_stages:
        if num_stages == 1:
            return [0.0]
        return [i / num_stages for i in range(num_stages)]
    boundaries = sorted(boundaries)
    boundaries[0] = 0.0
    return boundaries



def select_rollout_steps_for_epoch(epoch: int, max_epochs: int, rollout_steps_list: Sequence[int], boundaries: Sequence[float]) -> int:
    if len(rollout_steps_list) == 1:
        return int(rollout_steps_list[0])
    progress = 0.0 if max_epochs <= 1 else float(epoch - 1) / float(max(max_epochs - 1, 1))
    idx = 0
    for i, boundary in enumerate(boundaries):
        if progress >= boundary:
            idx = i
    idx = min(idx, len(rollout_steps_list) - 1)
    return int(rollout_steps_list[idx])



def build_time_weights(num_steps: int, cfg: Dict, device: torch.device) -> torch.Tensor:
    rollout_cfg = cfg["rollout"]
    if (not rollout_cfg["use_within_chunk_temporal_weighting"]) or str(rollout_cfg["chunk_time_weight_type"]).lower() == "none":
        w = torch.ones(num_steps, device=device, dtype=torch.float32)
    else:
        weight_type = str(rollout_cfg["chunk_time_weight_type"]).lower()
        if weight_type == "linear":
            w = torch.linspace(float(rollout_cfg["chunk_time_weight_min"]), float(rollout_cfg["chunk_time_weight_max"]), steps=num_steps, device=device)
        elif weight_type == "power":
            x = torch.linspace(0.0, 1.0, steps=num_steps, device=device)
            w = float(rollout_cfg["chunk_time_weight_min"]) + (float(rollout_cfg["chunk_time_weight_max"]) - float(rollout_cfg["chunk_time_weight_min"])) * (x ** float(rollout_cfg["chunk_time_weight_power"]))
        elif weight_type == "exp":
            min_w = max(float(rollout_cfg["chunk_time_weight_min"]), 1e-8)
            max_w = max(float(rollout_cfg["chunk_time_weight_max"]), 1e-8)
            growth = math.log(max_w / min_w) / max(num_steps - 1, 1)
            idx = torch.arange(num_steps, device=device, dtype=torch.float32)
            w = min_w * torch.exp(growth * idx)
        else:
            raise ValueError(f"Unsupported chunk_time_weight_type: {rollout_cfg['chunk_time_weight_type']}")
    if rollout_cfg["normalize_chunk_time_weights"]:
        w = w / w.mean().clamp(min=1e-8)
    return w.view(1, num_steps, 1, 1)



def build_segment_weights(num_segments: int, cfg: Dict, device: torch.device) -> torch.Tensor:
    rollout_cfg = cfg["rollout"]
    if (not rollout_cfg["use_segment_weighting"]) or str(rollout_cfg["segment_weight_type"]).lower() == "none":
        w = torch.ones(num_segments, device=device, dtype=torch.float32)
    else:
        min_w = float(rollout_cfg["segment_weight_min"])
        max_w = float(rollout_cfg["segment_weight_max"])
        weight_type = str(rollout_cfg["segment_weight_type"]).lower()
        if weight_type == "linear":
            w = torch.linspace(min_w, max_w, steps=num_segments, device=device)
        elif weight_type == "power":
            x = torch.linspace(0.0, 1.0, steps=num_segments, device=device)
            w = min_w + (max_w - min_w) * (x ** float(rollout_cfg["segment_weight_power"]))
        elif weight_type == "exp":
            min_w = max(min_w, 1e-8)
            max_w = max(max_w, 1e-8)
            growth = math.log(max_w / min_w) / max(num_segments - 1, 1)
            idx = torch.arange(num_segments, device=device, dtype=torch.float32)
            w = min_w * torch.exp(growth * idx)
        else:
            raise ValueError(f"Unsupported segment_weight_type: {rollout_cfg['segment_weight_type']}")
    if rollout_cfg["normalize_segment_weights"]:
        w = w / w.mean().clamp(min=1e-8)
    return w



def rollout_forward_train(model: nn.Module, x_init: torch.Tensor, input_steps: int, one_shot_steps: int, rollout_steps: int, detach_context: bool) -> Tuple[List[torch.Tensor], List[int]]:
    current_x = x_init
    preds: List[torch.Tensor] = []
    lengths: List[int] = []
    generated = 0
    while generated < rollout_steps:
        pred_full = model(current_x)
        take = min(one_shot_steps, rollout_steps - generated, pred_full.shape[1])
        pred_use = pred_full[:, :take]
        preds.append(pred_use)
        lengths.append(int(take))
        generated += take
        context_chunk = pred_use.detach() if detach_context else pred_use
        current_x = torch.cat([current_x, context_chunk], dim=1)[:, -input_steps:]
    return preds, lengths



def get_rollout_targets(y: torch.Tensor, lengths: Sequence[int]) -> List[torch.Tensor]:
    targets: List[torch.Tensor] = []
    start = 0
    for take in lengths:
        end = start + int(take)
        targets.append(y[:, start:end])
        start = end
    return targets



def compute_weighted_mse(pred: torch.Tensor, target: torch.Tensor, time_weights: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2) * time_weights).mean()



def compute_rollout_loss(preds: List[torch.Tensor], targets: List[torch.Tensor], segment_weights: torch.Tensor, cfg: Dict, time_weight_cache: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = preds[0].device
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    sum_weights = segment_weights.sum().clamp(min=1e-8)
    mse_values: List[float] = []

    for i, (pred_i, target_i) in enumerate(zip(preds, targets)):
        take = int(pred_i.shape[1])
        if take not in time_weight_cache:
            time_weight_cache[take] = build_time_weights(take, cfg, device)
        loss_i = compute_weighted_mse(pred_i.float(), target_i.float(), time_weight_cache[take])
        total_loss = total_loss + segment_weights[i] * loss_i
        mse_values.append(float(loss_i.detach().item()))

    total_loss = total_loss / sum_weights
    return total_loss, {"loss": float(total_loss.detach().item()), "mse": float(np.mean(mse_values))}



def rollout_predict(model: nn.Module, x_init: torch.Tensor, rollout_steps: int, input_steps: int, one_shot_steps: int) -> torch.Tensor:
    context = x_init.clone()
    preds: List[torch.Tensor] = []
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



def get_band_ratio_list(cfg: Dict) -> List[float]:
    vals = sorted(float(r) for r in cfg["evaluation"]["spectral_band_split_ratios"])
    clipped: List[float] = []
    prev = 0.0
    for r in vals:
        r = min(max(r, max(0.05, prev + 1e-3)), 0.98)
        clipped.append(r)
        prev = r
    return clipped



def compute_band_edges(n_bins: int, cfg: Dict) -> List[int]:
    ratios = get_band_ratio_list(cfg)
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



def detailed_evaluate_rollout(model: nn.Module, loader: DataLoader, device: torch.device, cfg: Dict, data_mean: float, data_std: float, split: str) -> Dict[str, np.ndarray]:
    model.eval()

    rollout_steps = int(cfg["rollout"]["rollout_steps"])
    one_shot_steps = int(cfg["data"]["output_steps"])
    input_steps = int(cfg["data"]["input_steps"])

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

    saved_x = []
    saved_y_true = []
    saved_y_pred = []
    saved_sample_ids = []
    point_true_all = []
    point_pred_all = []
    point_specs = None

    radial_k = None
    radial_idx_flat = None
    bin_count = None
    step_spectrum_true_sum = None
    step_spectrum_pred_sum = None
    step_spectrum_count = 0.0

    true_stats_acc = init_stat_accumulator(rollout_steps)
    pred_stats_acc = init_stat_accumulator(rollout_steps)
    sample_cursor = 0
    sample_rmse_list = []
    sample_rel_l2_list = []
    sample_final_step_rmse_list = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = rollout_predict(model, x, rollout_steps, input_steps, one_shot_steps)

            bsz, _, h, w = pred.shape
            if point_specs is None:
                point_specs = build_default_points(h, w, int(cfg["evaluation"]["num_trace_points"]))
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

            sample_rmse = torch.sqrt(torch.mean(sq_err.view(bsz, -1), dim=1))
            sample_final_rmse = torch.sqrt(torch.mean(sq_err[:, -1].reshape(bsz, -1), dim=1))
            sample_rmse_list.append(sample_rmse.cpu().numpy().astype(np.float32))
            sample_rel_l2_list.append(rel_l2_batch.cpu().numpy().astype(np.float32))
            sample_final_step_rmse_list.append(sample_final_rmse.cpu().numpy().astype(np.float32))

            for t in range(rollout_steps):
                step_sqerr[t] += sq_err[:, t].sum().item()
                step_count[t] += sq_err[:, t].numel()
            for i in range(num_chunks):
                start = i * one_shot_steps
                end = min((i + 1) * one_shot_steps, rollout_steps)
                chunk_sqerr[i] += sq_err[:, start:end].sum().item()
                chunk_count[i] += sq_err[:, start:end].numel()

            point_true_all.append(extract_point_traces(y, point_specs).cpu().numpy().astype(np.float32))
            point_pred_all.append(extract_point_traces(pred, point_specs).cpu().numpy().astype(np.float32))

            remaining = max(0, int(cfg["evaluation"]["num_full_samples_to_save"]) - len(saved_sample_ids))
            if remaining > 0:
                take = min(remaining, bsz)
                saved_x.append(x[:take].cpu().numpy().astype(np.float32))
                saved_y_true.append(y[:take].cpu().numpy().astype(np.float32))
                saved_y_pred.append(pred[:take].cpu().numpy().astype(np.float32))
                saved_sample_ids.extend(list(range(sample_cursor, sample_cursor + take)))
            sample_cursor += bsz

            y_np = y.cpu().numpy().astype(np.float32)
            pred_np = pred.cpu().numpy().astype(np.float32)
            if cfg["data"]["normalize"]:
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

    step_spectrum_true = (step_spectrum_true_sum / max(step_spectrum_count, 1.0)).astype(np.float32)
    step_spectrum_pred = (step_spectrum_pred_sum / max(step_spectrum_count, 1.0)).astype(np.float32)
    global_spectrum_true = step_spectrum_true.mean(axis=0).astype(np.float32)
    global_spectrum_pred = step_spectrum_pred.mean(axis=0).astype(np.float32)

    high_k_start_idx = int(max(1, math.floor((len(radial_k) - 1) * float(cfg["evaluation"]["spectral_high_k_ratio"]))))
    high_mask = radial_k >= float(high_k_start_idx)
    spectral_rel_l2_global = float(np.linalg.norm(global_spectrum_pred - global_spectrum_true) / (np.linalg.norm(global_spectrum_true) + 1e-12))
    spectral_rel_l2_high = float(np.linalg.norm(global_spectrum_pred[high_mask] - global_spectrum_true[high_mask]) / (np.linalg.norm(global_spectrum_true[high_mask]) + 1e-12)) if np.any(high_mask) else 0.0

    band_edges = compute_band_edges(len(radial_k), cfg)
    band_energy_true = np.stack([step_spectrum_true[:, band_edges[i] : band_edges[i + 1]].sum(axis=1) for i in range(len(band_edges) - 1)], axis=1).astype(np.float32)
    band_energy_pred = np.stack([step_spectrum_pred[:, band_edges[i] : band_edges[i + 1]].sum(axis=1) for i in range(len(band_edges) - 1)], axis=1).astype(np.float32)

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
        "point_true_all": np.concatenate(point_true_all, axis=0) if point_true_all else np.empty((0, rollout_steps, 0), dtype=np.float32),
        "point_pred_all": np.concatenate(point_pred_all, axis=0) if point_pred_all else np.empty((0, rollout_steps, 0), dtype=np.float32),
        "point_names": point_names,
        "point_coords": point_coords,
        "data_mean": np.array(float(data_mean), dtype=np.float32),
        "data_std": np.array(float(data_std), dtype=np.float32),
        "normalize": np.array(int(cfg["data"]["normalize"]), dtype=np.int32),
        "dt": np.array(float(cfg["evaluation"]["dt"]), dtype=np.float32),
        "rollout_steps": np.array(int(cfg["rollout"]["rollout_steps"]), dtype=np.int32),
        "one_shot_steps": np.array(int(cfg["data"]["output_steps"]), dtype=np.int32),
        "experiment_name": np.asarray(cfg["project"]["experiment_name"], dtype=object),
        "sample_rmse": np.concatenate(sample_rmse_list, axis=0) if sample_rmse_list else np.empty((0,), dtype=np.float32),
        "sample_rel_l2": np.concatenate(sample_rel_l2_list, axis=0) if sample_rel_l2_list else np.empty((0,), dtype=np.float32),
        "sample_final_step_rmse": np.concatenate(sample_final_step_rmse_list, axis=0) if sample_final_step_rmse_list else np.empty((0,), dtype=np.float32),
    }
    return result
