from typing import Dict, Optional

from torch.utils.data import DataLoader

import lightning.pytorch as pl
from .datasets import SeaSurfaceSimpleDataset
from .rollout import parse_curriculum_boundaries, parse_rollout_train_steps, select_rollout_steps_for_epoch


class SeaSurfaceDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.train_mean: Optional[float] = None
        self.train_std: Optional[float] = None
        self._train_dataset_cache = {}
        self._val_dataset = None
        self._test_dataset = None
        self._curriculum_steps = parse_rollout_train_steps(cfg)
        self._curriculum_boundaries = parse_curriculum_boundaries(cfg, len(self._curriculum_steps))

    def setup(self, stage: Optional[str] = None) -> None:
        data_cfg = self.cfg["data"]
        if self.train_mean is None or self.train_std is None:
            train_stats_ds = SeaSurfaceSimpleDataset(
                data_dir=f"{data_cfg['data_root']}/train",
                variable=data_cfg["variable"],
                input_steps=int(data_cfg["input_steps"]),
                output_steps=int(data_cfg["output_steps"]),
                stride=int(data_cfg["stride"]),
                normalize=bool(data_cfg["normalize"]),
            )
            self.train_mean = float(train_stats_ds.mean)
            self.train_std = float(train_stats_ds.std)
            self._train_dataset_cache[int(data_cfg["output_steps"])] = train_stats_ds

        if stage in (None, "fit", "validate") and self._val_dataset is None:
            self._val_dataset = SeaSurfaceSimpleDataset(
                data_dir=f"{data_cfg['data_root']}/val",
                variable=data_cfg["variable"],
                input_steps=int(data_cfg["input_steps"]),
                output_steps=int(self.cfg["rollout"]["rollout_steps"]),
                stride=int(data_cfg["rollout_stride"]),
                normalize=bool(data_cfg["normalize"]),
                mean=self.train_mean if data_cfg["normalize"] else None,
                std=self.train_std if data_cfg["normalize"] else None,
            )

        if stage in (None, "test") and self._test_dataset is None:
            self._test_dataset = SeaSurfaceSimpleDataset(
                data_dir=f"{data_cfg['data_root']}/test",
                variable=data_cfg["variable"],
                input_steps=int(data_cfg["input_steps"]),
                output_steps=int(self.cfg["rollout"]["rollout_steps"]),
                stride=int(data_cfg["rollout_stride"]),
                normalize=bool(data_cfg["normalize"]),
                mean=self.train_mean if data_cfg["normalize"] else None,
                std=self.train_std if data_cfg["normalize"] else None,
            )

    def _active_rollout_steps(self) -> int:
        max_epochs = int(self.cfg["trainer"]["max_epochs"])
        epoch = int(getattr(self.trainer, "current_epoch", 0)) + 1 if self.trainer is not None else 1
        return select_rollout_steps_for_epoch(epoch, max_epochs, self._curriculum_steps, self._curriculum_boundaries)

    def train_dataloader(self) -> DataLoader:
        self.setup("fit")
        active_steps = self._active_rollout_steps()
        if active_steps not in self._train_dataset_cache:
            data_cfg = self.cfg["data"]
            self._train_dataset_cache[active_steps] = SeaSurfaceSimpleDataset(
                data_dir=f"{data_cfg['data_root']}/train",
                variable=data_cfg["variable"],
                input_steps=int(data_cfg["input_steps"]),
                output_steps=int(active_steps),
                stride=int(data_cfg["stride"]),
                normalize=bool(data_cfg["normalize"]),
                mean=self.train_mean if data_cfg["normalize"] else None,
                std=self.train_std if data_cfg["normalize"] else None,
            )
        ds = self._train_dataset_cache[active_steps]
        data_cfg = self.cfg["data"]
        return DataLoader(
            ds,
            batch_size=int(data_cfg["batch_size"]),
            shuffle=True,
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=bool(data_cfg["pin_memory"]),
        )

    def val_dataloader(self) -> DataLoader:
        self.setup("validate")
        data_cfg = self.cfg["data"]
        return DataLoader(
            self._val_dataset,
            batch_size=int(data_cfg["val_batch_size"]),
            shuffle=False,
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=bool(data_cfg["pin_memory"]),
        )

    def test_dataloader(self) -> DataLoader:
        self.setup("test")
        data_cfg = self.cfg["data"]
        return DataLoader(
            self._test_dataset,
            batch_size=int(data_cfg["test_batch_size"]),
            shuffle=False,
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=bool(data_cfg["pin_memory"]),
        )
