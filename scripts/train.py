import argparse
import os
import sys

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sea_surface_lightning.config import ensure_project_dirs, load_config, resolve_checkpoint_path, save_resolved_config
from sea_surface_lightning.data_module import SeaSurfaceDataModule
from sea_surface_lightning.lightning_module import SeaSurfaceLitModule
from sea_surface_lightning.logging_utils import append_csv_row, build_pl_loggers



def parse_args():
    parser = argparse.ArgumentParser(description="Train sea-surface rollout model with PyTorch Lightning.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume from.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    save_resolved_config(cfg)
    pl.seed_everything(int(cfg["seed"]), workers=True)

    data_module = SeaSurfaceDataModule(cfg)
    model = SeaSurfaceLitModule(cfg)
    loggers = build_pl_loggers(cfg, stage="train")

    ckpt_callback = ModelCheckpoint(
        dirpath=cfg["paths"]["checkpoint_dir"],
        monitor=cfg["ckpt"]["monitor"],
        mode=cfg["ckpt"]["mode"],
        save_top_k=int(cfg["ckpt"]["save_top_k"]),
        save_last=bool(cfg["ckpt"]["save_last"]),
        filename=cfg["ckpt"]["filename"],
        auto_insert_metric_name=False,
    )
    callbacks = [ckpt_callback, LearningRateMonitor(logging_interval="epoch"), ModelSummary(max_depth=2)]
    early_cfg = cfg["ckpt"]["early_stopping"]
    if early_cfg["enabled"]:
        callbacks.append(
            EarlyStopping(
                monitor=cfg["ckpt"]["monitor"],
                mode=cfg["ckpt"]["mode"],
                patience=int(early_cfg["patience"]),
                min_delta=float(early_cfg["min_delta"]),
            )
        )

    trainer = pl.Trainer(
        default_root_dir=cfg["paths"]["root_dir"],
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        precision=cfg["trainer"]["precision"],
        max_epochs=int(cfg["trainer"]["max_epochs"]),
        deterministic=bool(cfg["trainer"]["deterministic"]),
        benchmark=bool(cfg["trainer"]["benchmark"]),
        log_every_n_steps=int(cfg["trainer"]["log_every_n_steps"]),
        num_sanity_val_steps=int(cfg["trainer"]["num_sanity_val_steps"]),
        reload_dataloaders_every_n_epochs=int(cfg["trainer"]["reload_dataloaders_every_n_epochs"]),
        val_check_interval=cfg["trainer"]["val_check_interval"],
        check_val_every_n_epoch=int(cfg["trainer"]["check_val_every_n_epoch"]),
        gradient_clip_val=float(cfg["optim"]["grad_clip_norm"]),
        logger=loggers if len(loggers) > 1 else (loggers[0] if loggers else False),
        callbacks=callbacks,
    )

    resume_path = args.resume or cfg["ckpt"].get("resume_from")
    if resume_path == "auto":
        resume_path = resolve_checkpoint_path(cfg)
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_path)

    summary_row = {
        "experiment_name": cfg["project"]["experiment_name"],
        "best_model_path": ckpt_callback.best_model_path,
        "best_model_score": float(ckpt_callback.best_model_score) if ckpt_callback.best_model_score is not None else None,
        "last_model_path": os.path.join(cfg["paths"]["checkpoint_dir"], "last.ckpt"),
        "monitor": cfg["ckpt"]["monitor"],
        "max_epochs": int(cfg["trainer"]["max_epochs"]),
    }
    append_csv_row(cfg["paths"]["train_summary_path"], summary_row)

    if cfg["runtime"].get("validate_after_fit", False) and ckpt_callback.best_model_path:
        trainer.validate(model=None, datamodule=data_module, ckpt_path=ckpt_callback.best_model_path)


if __name__ == "__main__":
    main()
