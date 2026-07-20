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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from neuralop.data.datasets.sea_surface_simple import SeaSurfaceSimpleDataset
from config.sea_surface_rollout_config_coarse_to_fine import SeaSurfaceRolloutConfig
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


def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return a plain state_dict that can be safely passed to load_state_dict.

    Some checkpoints may contain PyTorch's internal ``_metadata`` as a normal key.
    That key is not a model parameter/buffer, so strict loading reports it as an
    unexpected key. This helper also removes common wrapper prefixes.
    """
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict to be a dict, got {type(state_dict)}")

    cleaned = {}
    for k, v in state_dict.items():
        k = str(k)
        if k == "_metadata" or k.endswith("._metadata"):
            continue
        cleaned[k] = v

    def _strip_prefix_if_all_keys_have(prefix: str, sd: Dict[str, Any]) -> Dict[str, Any]:
        tensor_keys = [k for k in sd.keys() if k != "_metadata"]
        if tensor_keys and all(k.startswith(prefix) for k in tensor_keys):
            return {k[len(prefix):]: v for k, v in sd.items()}
        return sd

    cleaned = _strip_prefix_if_all_keys_have("module.", cleaned)
    cleaned = _strip_prefix_if_all_keys_have("_orig_mod.", cleaned)
    return cleaned


def load_model_state_dict_safely(model: nn.Module, state_dict: Dict[str, Any], strict: bool = True):
    """Load a possibly dirty checkpoint state_dict.

    strict=True is preserved after removing harmless metadata keys: if there are
    real missing/unexpected model weights, this function still raises an error.
    """
    cleaned = clean_state_dict(state_dict)
    incompatible = model.load_state_dict(cleaned, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if strict and (missing or unexpected):
        msg = []
        if missing:
            msg.append(f"Missing key(s) in state_dict: {missing}")
        if unexpected:
            msg.append(f"Unexpected key(s) in state_dict: {unexpected}")
        raise RuntimeError("Error(s) in loading state_dict for " + model.__class__.__name__ + ":\n\t" + "\n\t".join(msg))
    return incompatible


# =============================================================================
# Model blocks
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.insert(2, nn.Dropout2d(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCNNRefiner(nn.Module):
    """A light chunk-wise CNN refiner. Time is treated as channels."""
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 64, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        depth = max(int(depth), 1)
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for _ in range(depth - 1):
            layers += [
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.GELU(),
            ]
            if dropout > 0:
                layers.append(nn.Dropout2d(float(dropout)))
        layers.append(nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetRefiner(nn.Module):
    """2D U-Net refiner. Time frames are channels, so input is B x C x H x W."""
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 64, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        self.depth = max(int(depth), 1)
        chs = [int(base_channels) * (2 ** i) for i in range(self.depth)]

        self.enc_blocks = nn.ModuleList()
        prev = int(in_channels)
        for ch in chs:
            self.enc_blocks.append(ConvBlock(prev, ch, dropout=dropout))
            prev = ch

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(chs[-1], chs[-1] * 2, dropout=dropout)

        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        prev = chs[-1] * 2
        for ch in reversed(chs):
            self.up_convs.append(nn.ConvTranspose2d(prev, ch, kernel_size=2, stride=2))
            self.up_blocks.append(ConvBlock(ch * 2, ch, dropout=dropout))
            prev = ch

        self.out_conv = nn.Conv2d(chs[0], int(out_channels), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        h = x
        for block in self.enc_blocks:
            h = block(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)
        for up, block, skip in zip(self.up_convs, self.up_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h)
        return self.out_conv(h)


class CoarseToFineModel(nn.Module):
    def __init__(self, config: SeaSurfaceRolloutConfig):
        super().__init__()
        self.config = config
        arch = str(config.model_arch).lower()
        if arch == "fno":
            model_cls = FNO
        elif arch == "tfno":
            model_cls = TFNO
        else:
            raise ValueError(f"Unsupported model_arch: {config.model_arch}")

        factor = max(int(getattr(config, "c2f_coarse_time_factor", 4)), 1)
        coarse_steps = int(math.ceil(int(config.rollout_steps) / factor))
        self.coarse_steps = coarse_steps

        coarse_kwargs = dict(
            n_modes=tuple(config.n_modes),
            hidden_channels=int(config.hidden_channels),
            in_channels=int(config.input_steps),
            out_channels=int(coarse_steps),
            n_layers=int(config.n_layers),
            lifting_channels=int(config.lifting_channels),
            projection_channels=int(config.projection_channels),
        )
        self.coarse_predictor = model_cls(**_filter_model_kwargs(model_cls, coarse_kwargs))

        refiner_type = str(getattr(config, "c2f_refiner_type", "unet")).lower()
        refiner_in = int(getattr(config, "c2f_refiner_in_channels", int(config.input_steps) + int(config.output_steps)))
        refiner_out = int(getattr(config, "c2f_refiner_out_channels", int(config.output_steps)))
        refiner_base = int(getattr(config, "c2f_refiner_base_channels", 64))
        refiner_depth = int(getattr(config, "c2f_refiner_depth", 3))
        refiner_dropout = float(getattr(config, "c2f_refiner_dropout", 0.0))
        if refiner_type == "cnn":
            self.refiner = SimpleCNNRefiner(refiner_in, refiner_out, refiner_base, refiner_depth, refiner_dropout)
        elif refiner_type == "unet":
            self.refiner = UNetRefiner(refiner_in, refiner_out, refiner_base, refiner_depth, refiner_dropout)
        else:
            raise ValueError(f"Unsupported c2f_refiner_type: {config.c2f_refiner_type}")

    def temporal_upsample(self, coarse: torch.Tensor, target_steps: int, mode: str = "linear") -> torch.Tensor:
        """coarse: B x Tc x H x W -> B x target_steps x H x W."""
        if coarse.shape[1] == target_steps:
            return coarse
        bsz, tc, h, w = coarse.shape
        if tc <= 1:
            return coarse[:, :1].repeat(1, int(target_steps), 1, 1)
        x = coarse.permute(0, 2, 3, 1).reshape(bsz * h * w, 1, tc)
        if mode == "nearest":
            y = F.interpolate(x, size=int(target_steps), mode="nearest")
        else:
            y = F.interpolate(x, size=int(target_steps), mode="linear", align_corners=True)
        return y.reshape(bsz, h, w, int(target_steps)).permute(0, 3, 1, 2).contiguous()

    def forward(self, x_context: torch.Tensor, target_steps: int, use_refiner: bool = True, detach_context: bool = False) -> Dict[str, torch.Tensor]:
        factor = max(int(getattr(self.config, "c2f_coarse_time_factor", 4)), 1)
        coarse_needed = int(math.ceil(int(target_steps) / factor))
        coarse_full = self.coarse_predictor(x_context)
        coarse = coarse_full[:, :coarse_needed]
        base = self.temporal_upsample(coarse, int(target_steps), mode=str(getattr(self.config, "c2f_temporal_upsample_mode", "linear")).lower())

        if not use_refiner:
            return {"pred": base, "base": base, "coarse": coarse}

        context = x_context
        preds: List[torch.Tensor] = []
        generated = 0
        one_shot_steps = int(self.config.output_steps)
        input_steps = int(self.config.input_steps)
        residual_scale = float(getattr(self.config, "c2f_refiner_residual_scale", 1.0))
        while generated < int(target_steps):
            take = min(one_shot_steps, int(target_steps) - generated)
            base_chunk = base[:, generated:generated + take]
            if take < one_shot_steps:
                pad = one_shot_steps - take
                pad_frame = base_chunk[:, -1:].repeat(1, pad, 1, 1)
                base_refiner = torch.cat([base_chunk, pad_frame], dim=1)
            else:
                base_refiner = base_chunk
            refiner_in = torch.cat([context, base_refiner], dim=1)
            residual_full = self.refiner(refiner_in)
            residual = residual_full[:, :take] * residual_scale
            pred_chunk = base_chunk + residual
            preds.append(pred_chunk)
            generated += take
            context_chunk = pred_chunk.detach() if detach_context else pred_chunk
            context = torch.cat([context, context_chunk], dim=1)[:, -input_steps:]
        pred = torch.cat(preds, dim=1)
        return {"pred": pred, "base": base, "coarse": coarse}


def build_model(config: SeaSurfaceRolloutConfig) -> CoarseToFineModel:
    return CoarseToFineModel(config)


# =============================================================================
# Coarse targets / losses
# =============================================================================

def build_coarse_indices(target_steps: int, factor: int, device: torch.device, mode: str = "last_in_group") -> torch.Tensor:
    factor = max(int(factor), 1)
    target_steps = int(target_steps)
    coarse_steps = int(math.ceil(target_steps / factor))
    if str(mode).lower() == "uniform":
        if coarse_steps <= 1:
            idx = torch.tensor([target_steps - 1], device=device, dtype=torch.long)
        else:
            idx = torch.linspace(0, target_steps - 1, steps=coarse_steps, device=device).round().long()
    else:
        idx_vals = [min((i + 1) * factor - 1, target_steps - 1) for i in range(coarse_steps)]
        idx = torch.tensor(idx_vals, device=device, dtype=torch.long)
    return idx


def build_coarse_target(y: torch.Tensor, config: SeaSurfaceRolloutConfig) -> torch.Tensor:
    idx = build_coarse_indices(
        target_steps=y.shape[1],
        factor=int(getattr(config, "c2f_coarse_time_factor", 4)),
        device=y.device,
        mode=str(getattr(config, "c2f_coarse_target_mode", "last_in_group")),
    )
    return y.index_select(dim=1, index=idx)


def split_by_lengths(x: torch.Tensor, one_shot_steps: int) -> Tuple[List[torch.Tensor], List[int]]:
    chunks: List[torch.Tensor] = []
    lengths: List[int] = []
    start = 0
    while start < x.shape[1]:
        end = min(start + int(one_shot_steps), x.shape[1])
        chunks.append(x[:, start:end])
        lengths.append(end - start)
        start = end
    return chunks, lengths


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


def compute_weighted_mse(pred: torch.Tensor, target: torch.Tensor, time_weights: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2) * time_weights).mean()


def compute_rollout_loss_from_tensors(pred: torch.Tensor, target: torch.Tensor, segment_weights: torch.Tensor, config: SeaSurfaceRolloutConfig, time_weight_cache: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_chunks, _ = split_by_lengths(pred, int(config.output_steps))
    target_chunks, _ = split_by_lengths(target, int(config.output_steps))
    device = pred.device
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    sum_weights = segment_weights[:len(pred_chunks)].sum().clamp(min=1e-8)
    mse_values: List[float] = []
    for i, (pred_i, target_i) in enumerate(zip(pred_chunks, target_chunks)):
        take = int(pred_i.shape[1])
        if take not in time_weight_cache:
            time_weight_cache[take] = build_time_weights(take, config, device)
        loss_i = compute_weighted_mse(pred_i.float(), target_i.float(), time_weight_cache[take])
        total_loss = total_loss + segment_weights[i] * loss_i
        mse_values.append(float(loss_i.detach().item()))
    total_loss = total_loss / sum_weights
    return total_loss, {"loss": float(total_loss.detach().item()), "mse": float(np.mean(mse_values))}


# =============================================================================
# Data / curriculum
# =============================================================================

def build_dataloaders(config: SeaSurfaceRolloutConfig, train_target_steps: int, val_rollout_steps: int) -> Tuple[SeaSurfaceSimpleDataset, DataLoader, DataLoader]:
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


# =============================================================================
# Loading / freezing / evaluation
# =============================================================================

def load_pretrained_coarse_if_needed(model: CoarseToFineModel, config: SeaSurfaceRolloutConfig, device: torch.device, logger: logging.Logger) -> None:
    path = str(getattr(config, "c2f_pretrained_coarse_path", "") or "")
    if not path:
        return
    if not os.path.exists(path):
        logger.warning("c2f_pretrained_coarse_path does not exist: %s", path)
        return
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "coarse_state_dict" in checkpoint:
        model.coarse_predictor.load_state_dict(clean_state_dict(checkpoint["coarse_state_dict"]), strict=False)
        logger.info("Loaded coarse_state_dict from %s", path)
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = clean_state_dict(checkpoint["model_state_dict"])
        coarse_state = {k.replace("coarse_predictor.", ""): v for k, v in state.items() if k.startswith("coarse_predictor.")}
        model.coarse_predictor.load_state_dict(coarse_state, strict=False)
        logger.info("Loaded coarse predictor weights from model_state_dict in %s", path)
    else:
        logger.warning("Unsupported checkpoint format for coarse loading: %s", path)


def apply_train_stage_freezing(model: CoarseToFineModel, config: SeaSurfaceRolloutConfig, logger: logging.Logger) -> bool:
    stage = str(getattr(config, "c2f_train_stage", "joint")).lower()
    use_refiner = stage != "coarse_only"

    for p in model.parameters():
        p.requires_grad_(True)

    if stage == "coarse_only":
        for p in model.refiner.parameters():
            p.requires_grad_(False)
        logger.info("Training stage: coarse_only. Refiner is frozen/unused.")
    elif stage == "refiner_only":
        if bool(getattr(config, "c2f_freeze_coarse_in_refiner_only", True)):
            for p in model.coarse_predictor.parameters():
                p.requires_grad_(False)
            logger.info("Training stage: refiner_only. Coarse predictor is frozen.")
        else:
            logger.info("Training stage: refiner_only. Coarse predictor is trainable because c2f_freeze_coarse_in_refiner_only=False.")
    elif stage == "joint":
        logger.info("Training stage: joint. Coarse predictor and refiner are both trainable.")
    else:
        raise ValueError(f"Unsupported c2f_train_stage: {config.c2f_train_stage}")
    return use_refiner


@torch.no_grad()
def evaluate_rollout(model: CoarseToFineModel, loader: DataLoader, device: torch.device, config: SeaSurfaceRolloutConfig) -> Dict[str, float]:
    model.eval()
    rollout_steps = int(config.rollout_steps)
    use_refiner = str(getattr(config, "c2f_train_stage", "joint")).lower() != "coarse_only"

    total_sq = 0.0
    total_count = 0
    total_rel = 0.0
    total_samples = 0
    final_sq = 0.0
    final_count = 0
    last_sq = 0.0
    last_count = 0
    coarse_sq = 0.0
    coarse_count = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        out = model(x, target_steps=rollout_steps, use_refiner=use_refiner, detach_context=False)
        pred = out["pred"]
        base = out["base"]

        sq = (pred.float() - y.float()) ** 2
        total_sq += sq.sum().item()
        total_count += sq.numel()
        total_samples += x.shape[0]

        pred_flat = pred.reshape(pred.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        total_rel += (torch.norm(pred_flat - y_flat, dim=1) / (torch.norm(y_flat, dim=1) + 1e-12)).sum().item()

        final_sq += sq[:, -1].sum().item()
        final_count += sq[:, -1].numel()

        last_take = min(int(config.output_steps), rollout_steps)
        last_sq += sq[:, -last_take:].sum().item()
        last_count += sq[:, -last_take:].numel()

        coarse_err = (base.float() - y.float()) ** 2
        coarse_sq += coarse_err.sum().item()
        coarse_count += coarse_err.numel()

    rollout_mse = total_sq / max(total_count, 1)
    final_mse = final_sq / max(final_count, 1)
    last_mse = last_sq / max(last_count, 1)
    coarse_mse = coarse_sq / max(coarse_count, 1)
    return {
        "rollout_rmse": float(math.sqrt(rollout_mse)),
        "final_step_rmse": float(math.sqrt(final_mse)),
        "last_chunk_rmse": float(math.sqrt(last_mse)),
        "rel_l2": float(total_rel / max(total_samples, 1)),
        "coarse_base_rmse": float(math.sqrt(coarse_mse)),
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
    load_pretrained_coarse_if_needed(model, config, device, logger)
    use_refiner = apply_train_stage_freezing(model, config, logger)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Check c2f_train_stage and freezing settings.")
    optimizer = AdamW(trainable_params, lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    scheduler = build_scheduler(optimizer, config)

    rollout_steps_list = parse_rollout_train_steps(config)
    boundaries = parse_curriculum_boundaries(len(rollout_steps_list), config)

    best_metric = float("inf")
    best_epoch = 0
    epochs_without_improve = 0
    time_weight_cache: Dict[int, torch.Tensor] = {}
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
        pred_loss_sum = 0.0
        coarse_loss_sum = 0.0
        base_loss_sum = 0.0
        train_batches = 0
        segment_weights = build_segment_weights(int(math.ceil(active_rollout_steps / int(config.output_steps))), config, device)

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            out = model(
                x,
                target_steps=int(active_rollout_steps),
                use_refiner=bool(use_refiner),
                detach_context=bool(config.rollout_detach_context),
            )
            pred = out["pred"]
            base = out["base"]
            coarse = out["coarse"]
            coarse_target = build_coarse_target(y, config)
            coarse_loss = F.mse_loss(coarse.float(), coarse_target.float())
            base_loss, _ = compute_rollout_loss_from_tensors(base, y, segment_weights, config, time_weight_cache)

            stage = str(getattr(config, "c2f_train_stage", "joint")).lower()
            if stage == "coarse_only":
                loss = coarse_loss + float(getattr(config, "c2f_base_loss_weight", 0.1)) * base_loss
                pred_loss = base_loss.detach()
            else:
                pred_loss, _ = compute_rollout_loss_from_tensors(pred, y, segment_weights, config, time_weight_cache)
                if stage == "refiner_only":
                    loss = pred_loss
                else:
                    loss = pred_loss + float(getattr(config, "c2f_coarse_loss_weight", 0.2)) * coarse_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(config.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(config.grad_clip_norm))
            optimizer.step()

            train_loss_sum += float(loss.detach().item())
            pred_loss_sum += float(pred_loss.detach().item())
            coarse_loss_sum += float(coarse_loss.detach().item())
            base_loss_sum += float(base_loss.detach().item())
            train_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)
        train_pred_loss = pred_loss_sum / max(train_batches, 1)
        train_coarse_loss = coarse_loss_sum / max(train_batches, 1)
        train_base_loss = base_loss_sum / max(train_batches, 1)

        val_metrics = evaluate_rollout(model, val_loader, device, config)
        select_metric = float(val_metrics["rollout_rmse"])

        logger.info(
            "Epoch %03d/%03d | stage=%s | active_rollout=%d | train_loss=%.6f | pred_loss=%.6f | coarse_loss=%.6f | base_loss=%.6f | val_rmse=%.6f | val_base_rmse=%.6f | val_final=%.6f | val_last_chunk=%.6f | val_rel_l2=%.6f | lr=%.6f",
            epoch,
            config.n_epochs,
            getattr(config, "c2f_train_stage", "joint"),
            active_rollout_steps,
            train_loss,
            train_pred_loss,
            train_coarse_loss,
            train_base_loss,
            val_metrics["rollout_rmse"],
            val_metrics["coarse_base_rmse"],
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
                    "model_state_dict": clean_state_dict(model.state_dict()),
                    "coarse_state_dict": clean_state_dict(model.coarse_predictor.state_dict()),
                    "refiner_state_dict": clean_state_dict(model.refiner.state_dict()),
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
        "c2f_train_stage": getattr(config, "c2f_train_stage", "joint"),
        "c2f_refiner_type": getattr(config, "c2f_refiner_type", "unet"),
        "c2f_coarse_time_factor": int(getattr(config, "c2f_coarse_time_factor", 4)),
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
