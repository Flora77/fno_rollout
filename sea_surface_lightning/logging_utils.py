import csv
import os
from typing import Dict, List, Optional

import numpy as np


import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger




def append_csv_row(csv_path: str, row: Dict) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)



def build_pl_loggers(cfg: Dict, stage: str = "train") -> List:
    loggers = []
    tb_cfg = cfg["logging"]["tensorboard"]
    if tb_cfg["enabled"]:
        tb_name = tb_cfg["name"] or cfg["project"]["experiment_name"]
        loggers.append(
            TensorBoardLogger(
                save_dir=cfg["paths"]["log_dir"],
                name=tb_name,
                version=stage,
                default_hp_metric=False,
            )
        )

    wb_cfg = cfg["logging"]["wandb"]
    if wb_cfg["enabled"]:
        save_dir = wb_cfg["save_dir"] or cfg["paths"]["log_dir"]
        loggers.append(
            WandbLogger(
                project=wb_cfg["project"],
                entity=wb_cfg["entity"],
                name=wb_cfg["name"] or f"{cfg['project']['experiment_name']}-{stage}",
                save_dir=save_dir,
                offline=bool(wb_cfg["offline"]),
                tags=list(wb_cfg.get("tags", [])),
                notes=wb_cfg.get("notes"),
                log_model=False,
            )
        )
    return loggers



def log_metrics_to_loggers(loggers: List, metrics: Dict[str, float], step: Optional[int] = None) -> None:
    for logger in loggers:
        if isinstance(logger, TensorBoardLogger):
            for key, value in metrics.items():
                logger.experiment.add_scalar(key, float(value), global_step=0 if step is None else step)
            logger.save()
        elif isinstance(logger, WandbLogger):
            logger.experiment.log(metrics, step=step)



def log_image_to_loggers(loggers: List, name: str, image_path: str, step: Optional[int] = None) -> None:
    if not os.path.exists(image_path):
        return
    img = np.asarray(__import__("PIL.Image").Image.open(image_path))
    for logger in loggers:
        if isinstance(logger, TensorBoardLogger):
            if img.ndim == 2:
                logger.experiment.add_image(name, img[None, ...], global_step=0 if step is None else step, dataformats="CHW")
            else:
                logger.experiment.add_image(name, img, global_step=0 if step is None else step, dataformats="HWC")
            logger.save()
        elif isinstance(logger, WandbLogger):
            import wandb

            logger.experiment.log({name: wandb.Image(image_path)}, step=step)



def finalize_loggers(loggers: List) -> None:
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            try:
                logger.experiment.finish()
            except Exception:
                pass
