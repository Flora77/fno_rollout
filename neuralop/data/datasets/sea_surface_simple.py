import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import h5py


CANDIDATE_KEYS = ["h", "eta", "elevation", "height", "surface"]


def _build_candidate_keys(variable: Optional[str]) -> List[str]:
    keys: List[str] = []
    if variable is not None:
        variable = str(variable).strip()
        if variable:
            keys.append(variable)
    for key in CANDIDATE_KEYS:
        if key not in keys:
            keys.append(key)
    return keys


def _load_mat_array(mat_path: str, variable: Optional[str] = None) -> np.ndarray:
    candidate_keys = _build_candidate_keys(variable)
    arr = None
    try:
        mat = sio.loadmat(mat_path)
        for key in candidate_keys:
            if key in mat:
                arr = mat[key]
                break
    except NotImplementedError:
        with h5py.File(mat_path, "r") as mat:
            for key in candidate_keys:
                if key in mat:
                    arr = np.array(mat[key])
                    arr = arr.T.copy()
                    break

    if arr is None:
        raise KeyError(
            f"No valid variable key found in {mat_path}. Tried keys: {candidate_keys}"
        )

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (T,H,W), got shape {arr.shape} from {mat_path}")
    return arr


class SeaSurfaceSimpleDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        variable: Optional[str] = "height",
        input_steps: int = 40,
        output_steps: int = 20,
        stride: int = 4,
        normalize: bool = True,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        super().__init__()
        import glob

        self.data_dir = data_dir
        self.variable = variable
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.stride = int(stride)
        self.normalize = bool(normalize)

        mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
        if len(mat_files) == 0:
            raise ValueError(f"No .mat files found in {data_dir}")

        self.data_list: List[np.ndarray] = []
        self.index_map: List[Tuple[int, int]] = []
        total_window = self.input_steps + self.output_steps
        all_data_for_norm: List[np.ndarray] = []

        for f_idx, f_path in enumerate(mat_files):
            height = _load_mat_array(f_path, variable=variable)
            self.data_list.append(height)
            T = height.shape[0]

            if T >= total_window:
                starts = list(range(0, T - total_window + 1, self.stride))
                for s in starts:
                    self.index_map.append((f_idx, s))

            if self.normalize and mean is None:
                all_data_for_norm.append(height)

        if len(self.index_map) == 0:
            raise ValueError(
                f"No valid sequences found in {data_dir} with total_window={total_window}"
            )

        if self.normalize:
            if mean is None or std is None:
                concat_data = np.concatenate(all_data_for_norm, axis=0)
                mean = float(concat_data.mean())
                std = float(concat_data.std())
                std = max(std, 1e-6)
            self.mean = float(mean)
            self.std = float(std)
            for i in range(len(self.data_list)):
                self.data_list[i] = (self.data_list[i] - self.mean) / self.std
        else:
            self.mean = 0.0
            self.std = 1.0

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f_idx, s = self.index_map[idx]
        data = self.data_list[f_idx]
        x = data[s : s + self.input_steps]
        y = data[s + self.input_steps : s + self.input_steps + self.output_steps]

        if x.shape[0] != self.input_steps:
            raise RuntimeError(
                f"x length mismatch at idx={idx}, expected {self.input_steps}, got {x.shape[0]}"
            )
        if y.shape[0] != self.output_steps:
            raise RuntimeError(
                f"y length mismatch at idx={idx}, expected {self.output_steps}, got {y.shape[0]}"
            )

        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.from_numpy(y).float(),
        }


@dataclass
class SeaSurfaceSimpleDataConfig:
    train_dir: str = "/home/chenping/neuraloperator/data/sea_surface_simple/train"
    val_dir: str = "/home/chenping/neuraloperator/data/sea_surface_simple/val"
    test_dir: str = "/home/chenping/neuraloperator/data/sea_surface_simple/test"
    variable: Optional[str] = "height"
    input_steps: int = 40
    output_steps: int = 20
    stride: int = 4
    normalize: bool = True
    batch_size: int = 8
    val_batch_size: int = 8
    test_batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True


def build_sea_surface_simple_dataloaders(
    config: SeaSurfaceSimpleDataConfig,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = SeaSurfaceSimpleDataset(
        data_dir=config.train_dir,
        variable=config.variable,
        input_steps=config.input_steps,
        output_steps=config.output_steps,
        stride=config.stride,
        normalize=config.normalize,
    )
    val_dataset = SeaSurfaceSimpleDataset(
        data_dir=config.val_dir,
        variable=config.variable,
        input_steps=config.input_steps,
        output_steps=config.output_steps,
        stride=config.stride,
        normalize=config.normalize,
        mean=train_dataset.mean if config.normalize else None,
        std=train_dataset.std if config.normalize else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return train_loader, val_loader
