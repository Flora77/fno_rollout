import math
from typing import Dict, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR


import lightning.pytorch as pl

from .model_factory import build_model
from .rollout import (
    build_segment_weights,
    compute_rollout_loss,
    compute_sample_rel_l2,
    get_rollout_targets,
    parse_curriculum_boundaries,
    parse_rollout_train_steps,
    rollout_forward_train,
    rollout_predict,
    select_rollout_steps_for_epoch,
)


class SeaSurfaceLitModule(pl.LightningModule):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg)
        self.time_weight_cache: Dict[int, torch.Tensor] = {}
        self._val_outputs: List[Dict[str, float]] = []
        self._curriculum_steps = parse_rollout_train_steps(cfg)
        self._curriculum_boundaries = parse_curriculum_boundaries(cfg, len(self._curriculum_steps))
        self.save_hyperparameters({"cfg": cfg})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _active_rollout_steps(self) -> int:
        epoch = int(self.current_epoch) + 1
        max_epochs = int(self.cfg["trainer"]["max_epochs"])
        return select_rollout_steps_for_epoch(epoch, max_epochs, self._curriculum_steps, self._curriculum_boundaries)

    def on_train_epoch_start(self) -> None:
        self.log("train_active_rollout_steps", float(self._active_rollout_steps()), prog_bar=True, logger=True)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["x"]
        y = batch["y"]
        active_rollout_steps = int(y.shape[1])
        one_shot_steps = int(self.cfg["data"]["output_steps"])
        input_steps = int(self.cfg["data"]["input_steps"])
        preds, lengths = rollout_forward_train(
            model=self.model,
            x_init=x,
            input_steps=input_steps,
            one_shot_steps=one_shot_steps,
            rollout_steps=active_rollout_steps,
            detach_context=bool(self.cfg["rollout"]["detach_context"]),
        )
        targets = get_rollout_targets(y, lengths)
        segment_weights = build_segment_weights(int(math.ceil(active_rollout_steps / one_shot_steps)), self.cfg, x.device)
        loss, loss_info = compute_rollout_loss(preds, targets, segment_weights[: len(preds)], self.cfg, self.time_weight_cache)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self.log("train_mse_chunks", float(loss_info["mse"]), on_step=False, on_epoch=True, prog_bar=False, batch_size=x.shape[0])
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_outputs = []

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        x = batch["x"]
        y = batch["y"]
        rollout_steps = int(y.shape[1])
        one_shot_steps = int(self.cfg["data"]["output_steps"])
        input_steps = int(self.cfg["data"]["input_steps"])
        pred = rollout_predict(self.model, x, rollout_steps, input_steps, one_shot_steps)
        sq = (pred.float() - y.float()) ** 2
        pred_flat = pred.view(pred.shape[0], -1)
        y_flat = y.view(y.shape[0], -1)
        rel_sum = (torch.norm(pred_flat - y_flat, dim=1) / (torch.norm(y_flat, dim=1) + 1e-12)).sum()
        last_take = min(one_shot_steps, rollout_steps)
        self._val_outputs.append(
            {
                "total_sq": float(sq.sum().detach().cpu().item()),
                "total_count": float(sq.numel()),
                "rel_sum": float(rel_sum.detach().cpu().item()),
                "sample_count": float(x.shape[0]),
                "final_sq": float(sq[:, -1].sum().detach().cpu().item()),
                "final_count": float(sq[:, -1].numel()),
                "last_sq": float(sq[:, -last_take:].sum().detach().cpu().item()),
                "last_count": float(sq[:, -last_take:].numel()),
            }
        )

    def on_validation_epoch_end(self) -> None:
        if not self._val_outputs:
            return
        total_sq = sum(x["total_sq"] for x in self._val_outputs)
        total_count = sum(x["total_count"] for x in self._val_outputs)
        rel_sum = sum(x["rel_sum"] for x in self._val_outputs)
        sample_count = sum(x["sample_count"] for x in self._val_outputs)
        final_sq = sum(x["final_sq"] for x in self._val_outputs)
        final_count = sum(x["final_count"] for x in self._val_outputs)
        last_sq = sum(x["last_sq"] for x in self._val_outputs)
        last_count = sum(x["last_count"] for x in self._val_outputs)
        rollout_rmse = math.sqrt(total_sq / max(total_count, 1.0))
        final_rmse = math.sqrt(final_sq / max(final_count, 1.0))
        last_rmse = math.sqrt(last_sq / max(last_count, 1.0))
        rel_l2 = rel_sum / max(sample_count, 1.0)
        self.log("val_rollout_rmse", rollout_rmse, prog_bar=True, logger=True)
        self.log("val_final_step_rmse", final_rmse, prog_bar=False, logger=True)
        self.log("val_last_chunk_rmse", last_rmse, prog_bar=False, logger=True)
        self.log("val_rel_l2", rel_l2, prog_bar=False, logger=True)
        self._val_outputs = []

    def configure_optimizers(self):
        optim_cfg = self.cfg["optim"]
        optimizer = AdamW(self.parameters(), lr=float(optim_cfg["learning_rate"]), weight_decay=float(optim_cfg["weight_decay"]))
        scheduler_cfg = optim_cfg["scheduler"]
        if not scheduler_cfg["enabled"]:
            return optimizer
        scheduler_type = str(scheduler_cfg["type"]).lower()
        if scheduler_type == "step":
            scheduler = StepLR(optimizer, step_size=int(scheduler_cfg["step_size"]), gamma=float(scheduler_cfg["gamma"]))
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=int(scheduler_cfg["t_max"]), eta_min=float(scheduler_cfg["eta_min"]))
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        if scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(scheduler_cfg["factor"]),
                patience=int(scheduler_cfg["patience"]),
                min_lr=float(scheduler_cfg["min_lr"]),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.cfg["ckpt"]["monitor"],
                    "interval": "epoch",
                },
            }
        raise ValueError(f"Unsupported scheduler type: {scheduler_cfg['type']}")
