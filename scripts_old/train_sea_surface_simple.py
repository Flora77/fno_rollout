import os
import sys
import math
import random
from typing import Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from config.sea_surface_simple_config import SeaSurfaceSimpleConfig
from neuralop.data.datasets.sea_surface_simple import (
    SeaSurfaceSimpleDataConfig,
    build_sea_surface_simple_dataloaders,
)
from neuralop.models.fno import FNO,TFNO

def set_seed(seed:int =42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def build_model(config: SeaSurfaceSimpleConfig) -> nn.Module:
    model_arch = config.model_arch.lower()
    common_kwargs = dict(
        n_modes=config.n_modes,
        hidden_channels=config.hidden_channels,
        in_channels=config.input_steps,
        out_channels=config.output_steps,
        n_layers=config.n_layers,
    )
    if model_arch == "fno":
        model = FNO(**common_kwargs)
    elif model_arch == "tfno":
        model = TFNO(**common_kwargs)
    else:
        raise ValueError(f"Unsupported model architecture: {config.model_arch}")
    return model
 
def build_time_weights(config: SeaSurfaceSimpleConfig, device: torch.device) -> torch.Tensor:
    n = config.output_steps
    if not config.use_time_weighted_mse or config.time_weight_type == "none":
        w = torch.ones(n, device=device, dtype=torch.float32)
    elif config.time_weight_type == "linear":
        w = torch.linspace(config.time_weight_min,
                           config.time_weight_max, 
                           steps=n, 
                           device=device,
                           dtype=torch.float32
        )
    elif config.time_weight_type == "power":
        x = torch.linspace(0, 1, steps=n, device=device,dtype=torch.float32)
        w = config.time_weight_min + (config.time_weight_max - config.time_weight_min) * (x ** config.time_weight_power)
    elif config.time_weight_type == "exp":
        if n == 1:
            w = torch.ones(1, device=device, dtype=torch.float32) * config.time_weight_min
        else:
            min_w = max(config.time_weight_min, 1e-8)
            max_w = max(config.time_weight_max, 1e-8)
            growth = math.log(max_w / min_w) / (n - 1)
            idx = torch.arange(n, device=device, dtype=torch.float32)
            w = min_w * torch.exp(growth * idx)
    elif config.time_weight_type == "late_piecewise":
        w = torch.ones(n, device=device, dtype=torch.float32)
        k0 = int(n * 0.6)
        w[k0:] = torch.linspace(config.time_weight_min, config.time_weight_max, steps=n - k0, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported time_weight_type: {config.time_weight_type}")
    if config.normalize_time_weights:
        w = w / w.mean().clamp(min=1e-8)
    return w.view(1, n, 1, 1)

        

def compute_batch_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor,
    time_weights: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    sq_err = (pred - target) ** 2
    if time_weights is None:
        loss = torch.mean(sq_err)
    else:
        if time_weights.ndim != 4 or time_weights.shape[1] != pred.shape[1]:
            raise ValueError(
                f"time_weights shape mismatch: got {time_weights.shape}, "
                f"expected (1, {pred.shape[1]}, 1, 1)"
            )
        loss = torch.mean(sq_err * time_weights)
    rmse = torch.sqrt(torch.mean(sq_err))
    return loss, rmse

def train_one_epoch(
    model : nn.Module,
    loader ,
    optimizer :torch.optim.Optimizer,
    device : torch.device,  
    time_weights: torch.Tensor = None,  
) -> Tuple[float,float]:
    model.train()
    running_mse = 0.0
    running_rmse = 0.0
    num_batches = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        pred =model(x)
        if pred.shape != y.shape:
            raise RuntimeError(
                f"Prediction shape mismatch: pred={pred.shape}, target={y.shape}. "
                f"Please verify model input/output channel convention."
            )  
        mse, rmse = compute_batch_metrics(pred, y, time_weights)
        mse.backward()
        optimizer.step()
        running_mse += mse.item()
        running_rmse += rmse.item()
        num_batches += 1
    epoch_mse = running_mse / max(num_batches, 1)
    epoch_rmse = running_rmse / max(num_batches, 1)
    return epoch_mse, epoch_rmse

@torch.no_grad()
def evaluate(
    model : nn.Module,
    loader ,
    device : torch.device,
) -> Tuple[float,float]:
    model.eval()
    running_mse = 0.0
    running_rmse = 0.0
    num_batches = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x)
        if pred.shape != y.shape:
            raise RuntimeError(
                f"Prediction shape mismatch: pred={pred.shape}, target={y.shape}. "
                f"Please verify model input/output channel convention."
            )        
        mse, rmse = compute_batch_metrics(pred, y)
        running_mse += mse.item()
        running_rmse += rmse.item()
        num_batches += 1
    epoch_mse = running_mse / max(num_batches, 1)
    epoch_rmse = running_rmse / max(num_batches, 1)
    return epoch_mse, epoch_rmse

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: SeaSurfaceSimpleConfig,
    epoch: int,
    best_val_rmse: float,
) -> None:
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    chechpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_rmse": best_val_rmse,
        "config": config.__dict__,
    }
    torch.save(chechpoint, config.checkpoint_path)

def main():
    config = SeaSurfaceSimpleConfig()
    set_seed(42)
    device =get_device(config.device)
    
    train_dir = os.path.join(config.data_root, "train")
    val_dir = os.path.join(config.data_root, "val")
    
    print("="*80)
    print("Sea Surface Simple Training")
    print("="*80)
    print(f"Train Dir       : {train_dir}")
    print(f"Val Dir         : {val_dir}")
    print(f"Variable        : {config.variable}")
    print(f"Input steps     : {config.input_steps}")
    print(f"Output steps    : {config.output_steps}")
    print(f"Stride          : {config.stride}")
    print(f"Model arch      : {config.model_arch}")
    print(f"n_modes         : {config.n_modes}")
    print(f"hidden_channels : {config.hidden_channels}")
    print(f"n_layers        : {config.n_layers}")
    print(f"Epochs          : {config.n_epochs}")
    print(f"Device          : {device}")
    print(f"Checkpoint path : {config.checkpoint_path}")
    print("=" * 80)

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Val directory not found: {val_dir}")

    data_config = SeaSurfaceSimpleDataConfig(
        train_dir=train_dir,
        val_dir=val_dir,
        variable=config.variable,
        input_steps=config.input_steps,
        output_steps=config.output_steps,
        stride=config.stride,
        normalize=config.normalize,
        batch_size=config.batch_size,
        val_batch_size=config.val_batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        patience=config.early_stop_patience,
    )

    train_loader, val_loader = build_sea_surface_simple_dataloaders(data_config)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")   
    print("Train dataset size:", len(train_loader.dataset))
    print("Val dataset size  :", len(val_loader.dataset))

    sample_batch = next(iter(train_loader))
    print(f"Sample batch - x shape: {sample_batch['x'].shape}, y shape: {sample_batch['y'].shape}") 
    print("=" * 80)
    model = build_model(config).to(device)
    time_weights = build_time_weights(config, device)
    print(f"Use time weighted MSE: {config.use_time_weighted_mse}")
    print(f"Time weight type    : {config.time_weight_type}")
    print(f"Time weights        : {time_weights.view(-1).cpu().numpy()}")
    optimizer=AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=1e-4,
        min_lr=1e-6,
    )
    best_val_rmse = math.inf
    for epoch in range(1, config.n_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        train_mse, train_rmse =train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            time_weights=time_weights,
        )
        val_mse, val_rmse =evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )
        print(
            f"[Epoch {epoch:03d}/{config.n_epochs:03d}, LR: {current_lr:.6f}] "
            f"Train weighted MSE: {train_mse:.6f}, Train RMSE: {train_rmse:.6f} | "
            f"Val MSE: {val_mse:.6f}, Val RMSE: {val_rmse:.6f}"
        )
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                best_val_rmse=best_val_rmse,
            )
            print(f"Saved best ckpt to:{config.checkpoint_path}")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= data_config.patience:
            print(f"Early stopping at epoch {epoch}")
            break
        scheduler.step(val_rmse)
    print("=" * 80)        
    print(f"Training finished. Best val_rmse = {best_val_rmse:.6f}")
    print("=" * 80)

if __name__ == "__main__":
    main()      