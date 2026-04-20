import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


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
        raise KeyError(f"No valid variable key found in {mat_path}. Tried keys: {candidate_keys}")

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
                for start in starts:
                    self.index_map.append((f_idx, start))

            if self.normalize and mean is None:
                all_data_for_norm.append(height)

        if len(self.index_map) == 0:
            raise ValueError(f"No valid sequences found in {data_dir} with total_window={total_window}")

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
        f_idx, start = self.index_map[idx]
        data = self.data_list[f_idx]
        x = data[start : start + self.input_steps]
        y = data[start + self.input_steps : start + self.input_steps + self.output_steps]
        if x.shape[0] != self.input_steps:
            raise RuntimeError(f"x length mismatch at idx={idx}, expected {self.input_steps}, got {x.shape[0]}")
        if y.shape[0] != self.output_steps:
            raise RuntimeError(f"y length mismatch at idx={idx}, expected {self.output_steps}, got {y.shape[0]}")
        return {"x": torch.from_numpy(x).float(), "y": torch.from_numpy(y).float()}
