import os
import sys
from typing import Dict,List
import numpy as np
import torch
import scipy.io as sio
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from config.sea_surface_rollout_config import SeaSurfaceSimpleConfig
from neuralop.data.datasets.sea_surface_simple import (
    SeaSurfaceSimpleDataConfig,
    build_sea_surface_simple_dataloaders,
)

try:
    from neuralop.models.fno import FNO, TFNO
except ImportError as e:
    raise ImportError(
    print("Error importing FNO/TFNO models."
          " Make sure the neuralop package is properly installed.")
    )from e

def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def build_model(config: SeaSurfaceSimpleConfig) -> torch.nn.Module:
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

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    mse = torch.mean((pred - target) ** 2).item()
    rmse = float(np.sqrt(mse))
    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    num = torch.norm(pred_flat - target_flat, dim=1)
    den = torch.norm(target_flat, dim=1) + 1e-12
    rel_l2 = torch.mean(num / den).item()
    return {"mse": mse, "rmse": rmse, "rel_l2": rel_l2}

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    total_mse = 0.0
    total_rmse = 0.0
    total_rel_l2 = 0.0
    num_batches = 0
    all_x:List[np.ndarray] = []
    all_y_true:List[np.ndarray] = []
    all_y_pred:List[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x)
        if pred.shape != y.shape:
            raise RuntimeError(
                f"Prediction shape mismatch: pred={pred.shape}, target={y.shape}. "
                "Please verify model input/output shape convention."
            )        
        metrics = compute_metrics(pred, y)
        total_mse += metrics["mse"]
        total_rmse += metrics["rmse"]
        total_rel_l2 += metrics["rel_l2"]
        num_batches += 1
        all_x.append(x.detach().cpu().numpy())
        all_y_true.append(y.detach().cpu().numpy())
        all_y_pred.append(pred.detach().cpu().numpy())
    avg_metrics = {
        "mse": total_mse / max(num_batches, 1),
        "rmse": total_rmse / max(num_batches, 1),
        "rel_l2": total_rel_l2 / max(num_batches, 1),
    }
    x_all = np.concatenate(all_x, axis=0)
    y_true_all = np.concatenate(all_y_true, axis=0)
    y_pred_all = np.concatenate(all_y_pred, axis=0)
    return avg_metrics, x_all, y_true_all, y_pred_all

def compute_step_rmse(y_pred: np.ndarray, y_true: np.ndarray) ->np.ndarray:
    assert y_pred.shape == y_true.shape, f"y_pred_shape:{y_pred.shape}, y_true_shape:{y_true.shape}"
    n_samples, t_out, h, w = y_pred.shape
    rmse_list = []
    for t in range(t_out):
        err = y_pred[:, t] - y_true[:, t]
        mse_t = np.mean(err ** 2)
        rmse_t = np.sqrt(mse_t)
        rmse_list.append(rmse_t)
    return np.asarray(rmse_list,dtype=np.float32)

def main():
    config = SeaSurfaceSimpleConfig()
    device = get_device(config.device)
    train_dir = os.path.join(config.data_root, "train")
    val_dir = os.path.join(config.data_root, "val")
    print("="*80)
    print("Sea Surface Simple Evaluation")
    print("="*80)
    print(f"Train Dir           : {train_dir}")
    print(f"Val Dir             : {val_dir}")
    print(f"Variable            : {config.variable}")
    print(f"Input steps         : {config.input_steps}")
    print(f"Output steps        : {config.output_steps}")
    print(f"Stride              : {config.stride}")
    print(f"Model arch          : {config.model_arch}")
    print(f"n_modes             : {config.n_modes}")
    print(f"hidden_channels     : {config.hidden_channels}")
    print(f"n_layers            : {config.n_layers}")
    print(f"Epochs              : {config.n_epochs}")
    print(f"Device              : {device}")
    print(f"Checkpoint path     : {config.checkpoint_path}")
    print("=" * 80)

    if not os.path.exists(config.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {config.checkpoint_path}. ")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found at {train_dir}. ")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Val directory not found at {val_dir}. ")

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
    )

    _, val_loader = build_sea_surface_simple_dataloaders(data_config)
    print(f"Val batches  : {len(val_loader)}")
    sample_batch = next(iter(val_loader))
    print(f"Sample batch - x shape: {sample_batch['x'].shape}")
    print(f"Sample batch - y shape: {sample_batch['y'].shape}")
    model = build_model(config).to(device)
    checkpoint = torch.load(config.checkpoint_path, map_location=device,weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"model_state_dict not found in checkpoint at {config.checkpoint_path}.")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from checkpoint: {config.checkpoint_path}")

    metrics, x_all, y_true_all, y_pred_all = evaluate_model(model, val_loader, device)
    stepwise_rmse = compute_step_rmse(y_pred_all, y_true_all)
    print("=" * 80)
    print("Validation Metrics")
    print("=" * 80)
    print(f"MSE     : {metrics['mse']:.6f}")
    print(f"RMSE    : {metrics['rmse']:.6f}")
    print(f"Rel L2  : {metrics['rel_l2']:.6f}")
    print("Stepwise RMSE:")
    print(stepwise_rmse)
    print("=" * 80)    

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    save_path = os.path.join(config.checkpoint_dir, f"{config.experiment_name}_val_pdt.mat")
    sio.savemat(
        save_path,
        {
            "x": x_all.astype(np.float32),
            "y_true": y_true_all.astype(np.float32),
            "y_pred": y_pred_all.astype(np.float32),
            "stepwise_rmse": stepwise_rmse.astype(np.float32),
            "mse": np.array(metrics["mse"], dtype=np.float32),
            "rmse": np.array(metrics["rmse"], dtype=np.float32),
            "rel_l2": np.array(metrics["rel_l2"], dtype=np.float32),
        }
    )
    print(f"Saved predictions and metrics to {save_path}")
if __name__ == "__main__":
    main()


                                