import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sea_surface_lightning.config import ensure_project_dirs, load_config, resolve_checkpoint_path, save_resolved_config
from sea_surface_lightning.data_module import SeaSurfaceDataModule
from sea_surface_lightning.eval_runner import run_export_evaluation
from sea_surface_lightning.lightning_module import SeaSurfaceLitModule
from sea_surface_lightning.logging_utils import build_pl_loggers



def parse_args():
    parser = argparse.ArgumentParser(description="Detailed validation export for sea-surface rollout.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path. Defaults to latest/best-like ckpt.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    save_resolved_config(cfg)
    ckpt_path = resolve_checkpoint_path(cfg, explicit_ckpt=args.ckpt)

    data_module = SeaSurfaceDataModule(cfg)
    data_module.setup("validate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = SeaSurfaceLitModule.load_from_checkpoint(ckpt_path, cfg=cfg, map_location=device, weights_only=False, strict=False)
    module.eval()
    module.to(device)

    loggers = build_pl_loggers(cfg, stage="validate")
    _, _, mat_path = run_export_evaluation(
        model=module.model,
        loader=data_module.val_dataloader(),
        device=device,
        cfg=cfg,
        data_mean=float(data_module.train_mean),
        data_std=float(data_module.train_std),
        split="val",
        loggers=loggers,
    )
    print(f"Validation export finished. CKPT={ckpt_path}")
    print(f"Saved MAT to: {mat_path}")
    print(f"Saved plots to: {os.path.join(cfg['paths']['plot_dir'], 'val')}")


if __name__ == "__main__":
    main()
