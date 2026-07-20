import csv
import inspect
import json
import logging
import math
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
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
from config.sea_surface_rollout_config_fno_unet import SeaSurfaceRolloutConfig
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

try:
    from neuralop.models.fno import FNO, TFNO
    from neuralop.models.fno_unet_decoder import FNOUNetDecoder
except ImportError as e:
    raise ImportError("Please install neuralop before running this script.") from e


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def build_scheduler(optimizer, config):
    if not getattr(config, "use_lr_scheduler", False):
        return None
    scheduler_type = str(getattr(config, "lr_scheduler_type", "cosine")).lower()
    if scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=int(config.lr_scheduler_step_size),
            gamma=float(config.lr_scheduler_gamma),
        )
    if scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=int(config.lr_scheduler_t_max),
            eta_min=float(config.lr_scheduler_eta_min),
        )
    if scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.lr_scheduler_factor),
            patience=int(config.lr_scheduler_patience),
            min_lr=float(config.lr_scheduler_min_lr),
        )
    raise ValueError(f"Unsupported lr_scheduler_type: {config.lr_scheduler_type}")

def get_current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])

def setup_logger(config: SeaSurfaceRolloutConfig) -> logging.Logger:
    ensure_dir(config.log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config.log_dir, f"{config.experiment_name}_{timestamp}.log")

    logger = logging.getLogger(config.experiment_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("Logging to %s", log_path)
    return logger


def append_csv_row(csv_path: str, row: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(csv_path) or ".")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _filter_model_kwargs(model_cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(model_cls)
    return {k: v for k, v in kwargs.items() if v is not None and k in sig.parameters}


def build_model(config: SeaSurfaceRolloutConfig) -> nn.Module:
    arch = str(config.model_arch).lower()

    if arch == "fno_unet_decoder":
        return FNOUNetDecoder(
            n_modes=tuple(config.n_modes),
            hidden_channels=int(config.hidden_channels),
            in_channels=int(config.input_steps),
            out_channels=int(config.output_steps),
            n_layers=int(config.n_layers),
            lifting_channels=int(config.lifting_channels),
            projection_channels=int(config.projection_channels),
            fno_arch=str(getattr(config, "fno_unet_fno_arch", "fno")),
            unet_base_channels=int(getattr(config, "fno_unet_base_channels", 32)),
            unet_depth=int(getattr(config, "fno_unet_depth", 3)),
            unet_dropout=float(getattr(config, "fno_unet_decoder_dropout", 0.0)),
            use_context=bool(getattr(config, "fno_unet_use_context", True)),
            use_residual=bool(getattr(config, "fno_unet_use_residual", True)),
            residual_scale=float(getattr(config, "fno_unet_residual_scale", 1.0)),
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


def build_dataloaders(
    config: SeaSurfaceRolloutConfig,
    train_target_steps: int,
    val_rollout_steps: int,
) -> Tuple[SeaSurfaceSimpleDataset, DataLoader, DataLoader]:
    train_dir = os.path.join(config.data_root, "train")
    val_dir = os.path.join(config.data_root, "val")

    train_dataset = SeaSurfaceSimpleDataset(
        data_dir=train_dir,
        variable=config.variable,
        input_steps=int(config.input_steps),
        output_steps=int(train_target_steps),
        stride=int(config.stride),
        normalize=bool(config.normalize),
    )
    val_dataset = SeaSurfaceSimpleDataset(
        data_dir=val_dir,
        variable=config.variable,
        input_steps=int(config.input_steps),
        output_steps=int(val_rollout_steps),
        stride=int(config.rollout_stride),
        normalize=bool(config.normalize),
        mean=train_dataset.mean if config.normalize else None,
        std=train_dataset.std if config.normalize else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.val_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory),
    )
    return train_dataset, train_loader, val_loader


def build_time_weights(num_steps: int, config: SeaSurfaceRolloutConfig, device: torch.device) -> torch.Tensor:
    if (not config.use_within_chunk_temporal_weighting) or str(config.chunk_time_weight_type).lower() == "none":
        w = torch.ones(num_steps, device=device, dtype=torch.float32)
    else:
        weight_type = str(config.chunk_time_weight_type).lower()
        if weight_type == "linear":
            w = torch.linspace(float(config.chunk_time_weight_min), float(config.chunk_time_weight_max), steps=num_steps, device=device)
        elif weight_type == "power":
            x = torch.linspace(0.0, 1.0, steps=num_steps, device=device)
            w = float(config.chunk_time_weight_min) + (float(config.chunk_time_weight_max) - float(config.chunk_time_weight_min)) * (x ** float(config.chunk_time_weight_power))
        elif weight_type == "exp":
            min_w = max(float(config.chunk_time_weight_min), 1e-8)
            max_w = max(float(config.chunk_time_weight_max), 1e-8)
            growth = math.log(max_w / min_w) / max(num_steps - 1, 1)
            idx = torch.arange(num_steps, device=device, dtype=torch.float32)
            w = min_w * torch.exp(growth * idx)
        else:
            raise ValueError(f"Unsupported chunk_time_weight_type: {config.chunk_time_weight_type}")
    if config.normalize_chunk_time_weights:
        w = w / w.mean().clamp(min=1e-8)
    return w.view(1, num_steps, 1, 1)


def build_segment_weights(num_segments: int, config: SeaSurfaceRolloutConfig, device: torch.device) -> torch.Tensor:
    if (not config.use_segment_weighting) or str(config.segment_weight_type).lower() == "none":
        w = torch.ones(num_segments, device=device, dtype=torch.float32)
    else:
        min_w = float(config.segment_weight_min)
        max_w = float(config.segment_weight_max)
        weight_type = str(config.segment_weight_type).lower()
        if weight_type == "linear":
            w = torch.linspace(min_w, max_w, steps=num_segments, device=device)
        elif weight_type == "power":
            x = torch.linspace(0.0, 1.0, steps=num_segments, device=device)
            w = min_w + (max_w - min_w) * (x ** float(config.segment_weight_power))
        elif weight_type == "exp":
            min_w = max(min_w, 1e-8)
            max_w = max(max_w, 1e-8)
            growth = math.log(max_w / min_w) / max(num_segments - 1, 1)
            idx = torch.arange(num_segments, device=device, dtype=torch.float32)
            w = min_w * torch.exp(growth * idx)
        else:
            raise ValueError(f"Unsupported segment_weight_type: {config.segment_weight_type}")
    if config.normalize_segment_weights:
        w = w / w.mean().clamp(min=1e-8)
    return w


def parse_rollout_train_steps(config: SeaSurfaceRolloutConfig) -> List[int]:
    if not config.use_long_rollout_curriculum:
        return [int(config.rollout_steps)]
    steps = sorted({int(s) for s in config.rollout_train_steps if int(s) > 0})
    return steps if steps else [int(config.rollout_steps)]


def parse_curriculum_boundaries(num_stages: int, config: SeaSurfaceRolloutConfig) -> List[float]:
    boundaries = [float(v) for v in config.rollout_curriculum_boundaries]
    if len(boundaries) != num_stages:
        if num_stages == 1:
            return [0.0]
        return [i / num_stages for i in range(num_stages)]
    boundaries = sorted(boundaries)
    boundaries[0] = 0.0
    return boundaries


def select_rollout_steps_for_epoch(epoch: int, n_epochs: int, rollout_steps_list: Sequence[int], boundaries: Sequence[float]) -> int:
    if len(rollout_steps_list) == 1:
        return int(rollout_steps_list[0])
    progress = 0.0 if n_epochs <= 1 else float(epoch - 1) / float(max(n_epochs - 1, 1))
    idx = 0
    for i, b in enumerate(boundaries):
        if progress >= b:
            idx = i
    idx = min(idx, len(rollout_steps_list) - 1)
    return int(rollout_steps_list[idx])


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
    targets = []
    start = 0
    for take in lengths:
        end = start + int(take)
        targets.append(y[:, start:end])
        start = end
    return targets


def compute_weighted_mse(pred: torch.Tensor, target: torch.Tensor, time_weights: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2) * time_weights).mean()


def compute_spatial_gradient_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE of spatial slopes. Helps reduce over-smoothing in high-wavenumber components."""
    dx_pred = pred[..., :, 1:] - pred[..., :, :-1]
    dx_tgt = target[..., :, 1:] - target[..., :, :-1]
    dy_pred = pred[..., 1:, :] - pred[..., :-1, :]
    dy_tgt = target[..., 1:, :] - target[..., :-1, :]
    return ((dx_pred - dx_tgt) ** 2).mean() + ((dy_pred - dy_tgt) ** 2).mean()


def compute_rollout_loss(preds: List[torch.Tensor], targets: List[torch.Tensor], segment_weights: torch.Tensor, config: SeaSurfaceRolloutConfig, time_weight_cache: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = preds[0].device
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    sum_weights = segment_weights.sum().clamp(min=1e-8)
    mse_values: List[float] = []
    grad_values: List[float] = []

    use_grad_loss = bool(getattr(config, "use_spatial_gradient_loss", False))
    grad_weight = float(getattr(config, "spatial_gradient_loss_weight", 0.0))

    for i, (pred_i, target_i) in enumerate(zip(preds, targets)):
        take = int(pred_i.shape[1])
        if take not in time_weight_cache:
            time_weight_cache[take] = build_time_weights(take, config, device)

        mse_i = compute_weighted_mse(pred_i.float(), target_i.float(), time_weight_cache[take])
        loss_i = mse_i

        if use_grad_loss and grad_weight > 0.0:
            grad_i = compute_spatial_gradient_mse(pred_i.float(), target_i.float())
            loss_i = loss_i + grad_weight * grad_i
            grad_values.append(float(grad_i.detach().item()))

        total_loss = total_loss + segment_weights[i] * loss_i
        mse_values.append(float(mse_i.detach().item()))

    total_loss = total_loss / sum_weights
    return total_loss, {
        "loss": float(total_loss.detach().item()),
        "mse": float(np.mean(mse_values)),
        "grad_mse": float(np.mean(grad_values)) if grad_values else 0.0,
    }


@torch.no_grad()
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


@torch.no_grad()
def evaluate_rollout(model: nn.Module, loader: DataLoader, device: torch.device, input_steps: int, one_shot_steps: int, rollout_steps: int) -> Dict[str, float]:
    model.eval()
    total_sq = 0.0
    total_count = 0
    total_rel = 0.0
    total_samples = 0
    final_sq = 0.0
    final_count = 0
    last_sq = 0.0
    last_count = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        pred = rollout_predict(model, x, rollout_steps, input_steps, one_shot_steps)
        sq = (pred.float() - y.float()) ** 2
        total_sq += sq.sum().item()
        total_count += sq.numel()
        total_samples += x.shape[0]

        pred_flat = pred.view(pred.shape[0], -1)
        y_flat = y.view(y.shape[0], -1)
        total_rel += (torch.norm(pred_flat - y_flat, dim=1) / (torch.norm(y_flat, dim=1) + 1e-12)).sum().item()

        final_sq += sq[:, -1].sum().item()
        final_count += sq[:, -1].numel()

        last_take = min(one_shot_steps, rollout_steps)
        last_sq += sq[:, -last_take:].sum().item()
        last_count += sq[:, -last_take:].numel()

    rollout_mse = total_sq / max(total_count, 1)
    final_mse = final_sq / max(final_count, 1)
    last_mse = last_sq / max(last_count, 1)
    return {
        "rollout_rmse": float(math.sqrt(rollout_mse)),
        "final_step_rmse": float(math.sqrt(final_mse)),
        "last_chunk_rmse": float(math.sqrt(last_mse)),
        "rel_l2": float(total_rel / max(total_samples, 1)),
    }

def main() -> None:
    config = SeaSurfaceRolloutConfig()
    ensure_dir(config.checkpoint_dir)
    ensure_dir(config.plot_dir)
    logger = setup_logger(config)
    set_seed(int(config.seed))
    device = get_device(config.device)

    logger.info("Experiment: %s", config.experiment_name)
    logger.info("Config: %s", json.dumps(config.__dict__, ensure_ascii=False, default=str))

    model = build_model(config).to(device)
    optimizer = AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    scheduler = build_scheduler(optimizer, config)


    rollout_steps_list = parse_rollout_train_steps(config)
    boundaries = parse_curriculum_boundaries(len(rollout_steps_list), config)

    best_metric = float("inf")
    best_epoch = 0
    epochs_without_improve = 0
    time_weight_cache: Dict[int, torch.Tensor] = {}

    # Cache Dataset/DataLoader by active_rollout_steps.
    # When curriculum learning is enabled, each different rollout length needs
    # a different training dataset because y has a different time length.
    # Reusing the cached loader avoids rebuilding Dataset/DataLoader every epoch.
    loader_cache: Dict[int, Tuple[SeaSurfaceSimpleDataset, DataLoader, DataLoader]] = {}

    for epoch in range(1, int(config.n_epochs) + 1):
        current_lr = get_current_lr(optimizer)
        active_rollout_steps = select_rollout_steps_for_epoch(epoch, int(config.n_epochs), rollout_steps_list, boundaries)
        loader_key = int(active_rollout_steps)
        if loader_key not in loader_cache:
            loader_cache[loader_key] = build_dataloaders(config, active_rollout_steps, int(config.rollout_steps))
        train_dataset, train_loader, val_loader = loader_cache[loader_key]

        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        segment_weights = build_segment_weights(int(math.ceil(active_rollout_steps / int(config.output_steps))), config, device)

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            preds, lengths = rollout_forward_train(
                model=model,
                x_init=x,
                input_steps=int(config.input_steps),
                one_shot_steps=int(config.output_steps),
                rollout_steps=int(active_rollout_steps),
                detach_context=bool(config.rollout_detach_context),
            )
            targets = get_rollout_targets(y, lengths)
            loss, _ = compute_rollout_loss(preds, targets, segment_weights[:len(preds)], config, time_weight_cache)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(config.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
            optimizer.step()

            train_loss_sum += float(loss.detach().item())
            train_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)
        val_metrics = evaluate_rollout(
            model=model,
            loader=val_loader,
            device=device,
            input_steps=int(config.input_steps),
            one_shot_steps=int(config.output_steps),
            rollout_steps=int(config.rollout_steps),
        )
        select_metric = float(val_metrics["rollout_rmse"])

        logger.info(
            "Epoch %03d/%03d| active_rollout=%d | train_loss=%.6f | val_rollout_rmse=%.6f | val_final_rmse=%.6f | val_last_chunk_rmse=%.6f | val_rel_l2=%.6f | lr=%.6f",
            epoch,
            config.n_epochs,
            active_rollout_steps,
            train_loss,
            val_metrics["rollout_rmse"],
            val_metrics["final_step_rmse"],
            val_metrics["last_chunk_rmse"],
            val_metrics["rel_l2"],
            current_lr,
        )

        if select_metric < best_metric:
            best_metric = select_metric
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "epoch": epoch,
                    "best_rollout_rmse": best_metric,
                    "train_dataset_mean": float(train_dataset.mean),
                    "train_dataset_std": float(train_dataset.std),
                },
                config.checkpoint_path,
            )
            logger.info("Saved new best checkpoint to %s", config.checkpoint_path)
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= int(config.early_stop_patience):
            logger.info("Early stopping triggered at epoch %d", epoch)
            break
        if scheduler is not None:
            scheduler_type = config.lr_scheduler_type.lower()
            if scheduler_type == "plateau":
                scheduler.step(val_metrics["rollout_rmse"])
            else:
                scheduler.step()

    

    summary_row = {
        "experiment_name": config.experiment_name,
        "model_arch": config.model_arch,
        "use_long_rollout_curriculum": int(config.use_long_rollout_curriculum),
        "use_segment_weighting": int(config.use_segment_weighting),
        "use_within_chunk_temporal_weighting": int(config.use_within_chunk_temporal_weighting),
        "best_epoch": int(best_epoch),
        "best_val_rollout_rmse": float(best_metric),
        "checkpoint_path": config.checkpoint_path,
    }
    append_csv_row(config.train_summary_path, summary_row)
    logger.info("Appended train summary to %s", config.train_summary_path)


if __name__ == "__main__":
    main()
