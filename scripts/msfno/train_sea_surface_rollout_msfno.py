import csv
import inspect
import json
import logging
import math
import os
import random
import warnings
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from neuralop.data.datasets.sea_surface_simple import SeaSurfaceSimpleDataset
from config.sea_surface_rollout_config_msfno import SeaSurfaceRolloutConfig
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

try:
    from neuralop.models.fno import FNO, TFNO
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


def get_clean_model_state_dict(model: nn.Module) -> Dict[str, Any]:
    """
    Export a checkpoint state_dict without neuraloperator/PyTorch metadata keys.

    The returned dict is safe for the original validation scripts that call:
        model.load_state_dict(checkpoint["model_state_dict"])

    It removes the non-parameter key "_metadata" and moves tensors to CPU so
    checkpoint files do not depend on the current GPU state.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Attempting to update metadata for a module with metadata already in self\.state_dict\(\).*",
            category=UserWarning,
        )
        raw_state_dict = model.state_dict()

    cleaned: Dict[str, Any] = {}
    for k, v in raw_state_dict.items():
        key = str(k)
        if key == "_metadata":
            continue
        if isinstance(v, torch.Tensor):
            cleaned[key] = v.detach().cpu().clone()
        else:
            cleaned[key] = v
    return cleaned


def save_clean_checkpoint(
    *,
    config: SeaSurfaceRolloutConfig,
    model: nn.Module,
    epoch: int,
    last_epoch: int,
    best_metric: float,
    train_dataset: SeaSurfaceSimpleDataset,
    logger: logging.Logger,
) -> None:
    """
    Save a clean .pt checkpoint immediately when validation improves.

    The file is written through a temporary path and then atomically replaced,
    which avoids leaving a half-written checkpoint if saving is interrupted.
    """
    state_dict = get_clean_model_state_dict(model)
    if "_metadata" in state_dict:
        # Defensive guard; get_clean_model_state_dict should already remove it.
        state_dict.pop("_metadata", None)

    checkpoint = {
        "model_state_dict": state_dict,
        "config": config.__dict__,
        "epoch": int(epoch),
        "last_epoch": int(last_epoch),
        "best_rollout_rmse": float(best_metric),
        "train_dataset_mean": float(train_dataset.mean) if train_dataset is not None else float("nan"),
        "train_dataset_std": float(train_dataset.std) if train_dataset is not None else float("nan"),
    }

    ensure_dir(os.path.dirname(config.checkpoint_path) or ".")
    tmp_path = str(config.checkpoint_path) + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, config.checkpoint_path)

    logger.info(
        "Saved new checkpoint to %s | best_epoch=%d | last_epoch=%d | best_val_rollout_rmse=%.6f",
        config.checkpoint_path,
        int(epoch),
        int(last_epoch),
        float(best_metric),
    )


class SinActivation(nn.Module):
    """Sine activation used by some MscaleFNO ablation settings."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def _as_tuple(value: Any, item_type=float) -> Tuple:
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace("[", "").replace("]", "").replace("(", "").replace(")", "").split(",")]
        return tuple(item_type(p) for p in parts if p)
    if isinstance(value, (list, tuple)):
        return tuple(item_type(v) for v in value)
    return (item_type(value),)


def _make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "sin":
        return SinActivation()
    if name in ("none", "identity", ""):
        return nn.Identity()
    raise ValueError(f"Unsupported msfno_conv_activation: {name}")


def _make_norm3d(name: str, channels: int) -> nn.Module:
    name = str(name).lower()
    if name == "batch":
        return nn.BatchNorm3d(channels)
    if name == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if name in ("none", "identity", ""):
        return nn.Identity()
    raise ValueError(f"Unsupported msfno_conv_norm: {name}")


class MultiScaleFNO2d(nn.Module):
    """
    Paper-style MSFNO adapted to the existing 2D sea-surface rollout code.

    Original paper idea:
        Branch i receives scaled coordinates and scaled input function:
            [c_i x, c_i a(x)]
        Multiple branch outputs are then fused by a convolutional block.

    In this codebase, the tensor is [B, T_in, H, W], where time frames are
    treated as channels by a 2D FNO. Therefore, the paper's coordinate scaling
    is implemented by explicitly concatenating scaled spatial coordinates:
        z_i(x,y) = concat(c_i * eta_{1:T_in}(x,y), c_i * x, c_i * y)
    so each FNO branch sees the same physical field through a different scale.
    """

    def __init__(
        self,
        branch_cls,
        *,
        n_modes: Tuple[int, int],
        in_channels: int,
        out_channels: int,
        n_layers: int,
        branch_hidden_channels: int,
        branch_lifting_channels: int,
        branch_projection_channels: int,
        scales: Sequence[float],
        fusion: str = "conv",
        scale_input_field: bool = True,
        add_scaled_coords: bool = True,
        scale_coordinates: bool = True,
        output_scale: bool = False,
        coord_range: Sequence[float] = (0.0, 1.0),
        branch_positional_embedding: Any = None,
        conv_hidden_channels: Sequence[int] = (32, 64, 32),
        conv_kernel_size: Sequence[int] = (3, 3, 3),
        conv_norm: str = "batch",
        conv_activation: str = "relu",
    ) -> None:
        super().__init__()
        self.scales = tuple(float(s) for s in scales)
        if len(self.scales) == 0:
            raise ValueError("msfno_scales must contain at least one scale.")

        self.fusion_type = str(fusion).lower()
        self.scale_input_field = bool(scale_input_field)
        self.add_scaled_coords = bool(add_scaled_coords)
        self.scale_coordinates = bool(scale_coordinates)
        self.output_scale = bool(output_scale)

        coord_range = tuple(float(v) for v in coord_range)
        if len(coord_range) == 1:
            coord_range = (0.0, coord_range[0])
        if len(coord_range) != 2:
            raise ValueError("msfno_coord_range must be a tuple/list of length 2, e.g. (0.0, 1.0).")
        self.coord_min = float(coord_range[0])
        self.coord_max = float(coord_range[1])

        branch_in_channels = int(in_channels) + (2 if self.add_scaled_coords else 0)
        branch_kwargs = dict(
            n_modes=tuple(n_modes),
            hidden_channels=int(branch_hidden_channels),
            in_channels=int(branch_in_channels),
            out_channels=int(out_channels),
            n_layers=int(n_layers),
            lifting_channels=int(branch_lifting_channels),
            projection_channels=int(branch_projection_channels),
            # If the installed neuraloperator version supports this argument,
            # setting it to None avoids adding another unscaled internal grid.
            positional_embedding=branch_positional_embedding,
        )
        branch_kwargs = _filter_model_kwargs(branch_cls, branch_kwargs)
        self.branches = nn.ModuleList([branch_cls(**branch_kwargs) for _ in self.scales])

        if self.fusion_type == "conv":
            k = tuple(int(v) for v in conv_kernel_size)
            if len(k) == 1:
                k = (k[0], k[0], k[0])
            if len(k) != 3:
                raise ValueError("msfno_conv_kernel_size must be an int or a tuple/list of length 3.")
            padding = tuple(v // 2 for v in k)

            layers: List[nn.Module] = []
            in_ch = len(self.scales)
            for out_ch in [int(v) for v in conv_hidden_channels]:
                layers.append(nn.Conv3d(in_ch, out_ch, kernel_size=k, padding=padding))
                norm_layer = _make_norm3d(conv_norm, out_ch)
                if not isinstance(norm_layer, nn.Identity):
                    layers.append(norm_layer)
                act_layer = _make_activation(conv_activation)
                if not isinstance(act_layer, nn.Identity):
                    layers.append(act_layer)
                in_ch = out_ch
            layers.append(nn.Conv3d(in_ch, 1, kernel_size=1))
            self.fusion = nn.Sequential(*layers)

        elif self.fusion_type == "weighted_sum":
            self.branch_logits = nn.Parameter(torch.zeros(len(self.scales), dtype=torch.float32))
            self.fusion = nn.Identity()

        elif self.fusion_type == "mean":
            self.fusion = nn.Identity()

        else:
            raise ValueError("msfno_fusion must be one of: conv, weighted_sum, mean.")

    def _scaled_coord_channels(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        bsz, _, h, w = x.shape
        yy = torch.linspace(self.coord_min, self.coord_max, steps=h+1, device=x.device, dtype=x.dtype)[:-1]
        xx = torch.linspace(self.coord_min, self.coord_max, steps=w+1, device=x.device, dtype=x.dtype)[:-1]
        try:
            grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        except TypeError:  # for older PyTorch
            grid_y, grid_x = torch.meshgrid(yy, xx)
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(bsz, -1, -1, -1)
        if self.scale_coordinates:
            coords = coords * float(scale)
        return coords

    def _run_branch(self, branch: nn.Module, x: torch.Tensor, scale: float) -> torch.Tensor:
        # Paper-style input-function scaling: a(x) -> c_i a(x)
        field = x * float(scale) if self.scale_input_field else x

        # Paper-style coordinate scaling: x -> c_i x.
        # In the current 2D-FNO implementation, x,y coordinates are appended as two extra channels.
        if self.add_scaled_coords:
            coords = self._scaled_coord_channels(x, scale)
            branch_input = torch.cat([field, coords], dim=1)
        else:
            branch_input = field

        y = branch(branch_input)
        if isinstance(y, (tuple, list)):
            y = y[0]

        # Normally keep False when using conv fusion; the Conv block learns branch weighting/filtering.
        if self.output_scale:
            y = y * float(scale)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [
            self._run_branch(branch, x, scale)
            for branch, scale in zip(self.branches, self.scales)
        ]

        if self.fusion_type == "conv":
            # [B, N_branch, T_out, H, W] -> [B, 1, T_out, H, W] -> [B, T_out, H, W]
            stacked = torch.stack(branch_outputs, dim=1)
            return self.fusion(stacked).squeeze(1)

        if self.fusion_type == "weighted_sum":
            weights = torch.softmax(self.branch_logits, dim=0)
            out = torch.zeros_like(branch_outputs[0])
            for w, y in zip(weights, branch_outputs):
                out = out + w * y
            return out

        return torch.stack(branch_outputs, dim=0).mean(dim=0)

def _get_branch_channels(config: SeaSurfaceRolloutConfig) -> Tuple[int, int, int]:
    width_factor = float(getattr(config, "msfno_branch_width_factor", 0.5))
    hidden = int(getattr(config, "msfno_branch_hidden_channels", 0) or max(1, round(int(config.hidden_channels) * width_factor)))
    lifting = int(getattr(config, "msfno_branch_lifting_channels", 0) or max(hidden, round(int(config.lifting_channels) * width_factor)))
    projection = int(getattr(config, "msfno_branch_projection_channels", 0) or max(hidden, round(int(config.projection_channels) * width_factor)))
    return hidden, lifting, projection


def build_model(config: SeaSurfaceRolloutConfig) -> nn.Module:
    arch = str(config.model_arch).lower()

    if arch == "fno":
        model_cls = FNO
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

    if arch == "tfno":
        model_cls = TFNO
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

    if arch in ("msfno", "msfno2d", "mscale_fno", "mscalefno"):
        branch_arch = str(getattr(config, "msfno_branch_arch", "fno")).lower()
        if branch_arch == "fno":
            branch_cls = FNO
        elif branch_arch == "tfno":
            branch_cls = TFNO
        else:
            raise ValueError(f"Unsupported msfno_branch_arch: {branch_arch}")

        branch_hidden, branch_lifting, branch_projection = _get_branch_channels(config)
        return MultiScaleFNO2d(
            branch_cls,
            n_modes=tuple(config.n_modes),
            in_channels=int(config.input_steps),
            out_channels=int(config.output_steps),
            n_layers=int(config.n_layers),
            branch_hidden_channels=branch_hidden,
            branch_lifting_channels=branch_lifting,
            branch_projection_channels=branch_projection,
            scales=_as_tuple(getattr(config, "msfno_scales", (0.5, 1.0, 2.0, 4.0)), float),
            fusion=str(getattr(config, "msfno_fusion", "conv")),
            scale_input_field=bool(getattr(config, "msfno_scale_input_field", getattr(config, "msfno_scale_input", True))),
            add_scaled_coords=bool(getattr(config, "msfno_add_scaled_coords", True)),
            scale_coordinates=bool(getattr(config, "msfno_scale_coordinates", True)),
            output_scale=bool(getattr(config, "msfno_output_scale", False)),
            coord_range=_as_tuple(getattr(config, "msfno_coord_range", (0.0, 1.0)), float),
            branch_positional_embedding=getattr(config, "msfno_branch_positional_embedding", None),
            conv_hidden_channels=_as_tuple(getattr(config, "msfno_conv_hidden_channels", (32, 64, 32)), int),
            conv_kernel_size=_as_tuple(getattr(config, "msfno_conv_kernel_size", (3, 3, 3)), int),
            conv_norm=str(getattr(config, "msfno_conv_norm", "batch")),
            conv_activation=str(getattr(config, "msfno_conv_activation", "relu")),
        )

    raise ValueError(f"Unsupported model_arch: {config.model_arch}")


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


def compute_rollout_loss(preds: List[torch.Tensor], targets: List[torch.Tensor], segment_weights: torch.Tensor, config: SeaSurfaceRolloutConfig, time_weight_cache: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = preds[0].device
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    sum_weights = segment_weights.sum().clamp(min=1e-8)
    mse_values: List[float] = []

    for i, (pred_i, target_i) in enumerate(zip(preds, targets)):
        take = int(pred_i.shape[1])
        if take not in time_weight_cache:
            time_weight_cache[take] = build_time_weights(take, config, device)
        loss_i = compute_weighted_mse(pred_i.float(), target_i.float(), time_weight_cache[take])
        total_loss = total_loss + segment_weights[i] * loss_i
        mse_values.append(float(loss_i.detach().item()))

    total_loss = total_loss / sum_weights
    return total_loss, {"loss": float(total_loss.detach().item()), "mse": float(np.mean(mse_values))}


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
        x = batch["x"].to(device)
        y = batch["y"].to(device)
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
    last_epoch = 0
    epochs_without_improve = 0
    time_weight_cache: Dict[int, torch.Tensor] = {}

    # Cache Dataset/DataLoader objects. The training dataset target length only changes
    # when active_rollout_steps switches in the rollout curriculum, so rebuilding them
    # every epoch is unnecessary and can be very slow for .mat-based datasets.
    cached_active_rollout_steps = None
    train_dataset = None
    train_loader = None
    val_loader = None

    for epoch in range(1, int(config.n_epochs) + 1):
        last_epoch = epoch
        current_lr = get_current_lr(optimizer)
        active_rollout_steps = select_rollout_steps_for_epoch(epoch, int(config.n_epochs), rollout_steps_list, boundaries)
        if cached_active_rollout_steps != active_rollout_steps:
            train_dataset, train_loader, val_loader = build_dataloaders(config, active_rollout_steps, int(config.rollout_steps))
            cached_active_rollout_steps = active_rollout_steps

        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        segment_weights = build_segment_weights(int(math.ceil(active_rollout_steps / int(config.output_steps))), config, device)

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
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
            save_clean_checkpoint(
                config=config,
                model=model,
                epoch=best_epoch,
                last_epoch=last_epoch,
                best_metric=best_metric,
                train_dataset=train_dataset,
                logger=logger,
            )
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

    

    if best_epoch == 0:
        # Fallback for unusual cases, for example n_epochs <= 0. In normal training,
        # the first epoch improves because best_metric starts at +inf, so a clean
        # checkpoint has already been saved inside the training loop.
        best_epoch = int(last_epoch)
        best_metric = float("nan") if last_epoch == 0 else float(best_metric)
        save_clean_checkpoint(
            config=config,
            model=model,
            epoch=best_epoch,
            last_epoch=last_epoch,
            best_metric=best_metric,
            train_dataset=train_dataset,
            logger=logger,
        )

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
