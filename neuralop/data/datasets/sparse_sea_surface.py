"""Sparse-observation wrapper for sea-surface sequence datasets.

The wrapped dataset keeps its original ``{"x", "y"}`` contract.  This module
derives observations from ``x`` only and exposes ``sparse_model_inputs`` so
callers can explicitly keep supervision-only tensors away from the model.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


SUPPORTED_MASK_TYPES = frozenset(
    {"fixed_points", "random_points", "blocks", "stripes"}
)
MODEL_INPUT_KEYS = ("x_obs", "obs_mask")


@dataclass(frozen=True)
class SparseMaskMetadata:
    """Metadata for one mask stored in a sparse-observation manifest."""

    mask_id: str
    mask_type: str
    shape: Tuple[int, int, int]
    requested_observation_rate: Optional[float]
    effective_observation_rate: float
    replicate: Optional[int]
    seed: Optional[int]
    temporal_dropout: float


def _as_manifest_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        path = path.with_suffix(".json")
    if path.suffix.lower() != ".json":
        raise ValueError(
            f"Mask manifest must be a .json file (or its paired .npz), got {path}"
        )
    return path


class SparseMaskManifest:
    """Validated, in-memory access to masks referenced by a JSON manifest."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        *,
        verify_hashes: bool = True,
    ) -> None:
        self.manifest_path = _as_manifest_path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Mask manifest not found: {self.manifest_path}")

        with self.manifest_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)

        if not isinstance(document, dict):
            raise ValueError("Mask manifest root must be a JSON object")
        # Stage 1 generator writes ``entries``.  ``masks`` remains accepted for
        # compatibility with early hand-authored manifests.
        entries = document.get("entries", document.get("masks"))
        if not isinstance(entries, list) or not entries:
            raise ValueError("Mask manifest must contain a non-empty 'entries' list")

        npz_name = document.get("npz_file")
        if not isinstance(npz_name, str) or not npz_name.strip():
            raise ValueError("Mask manifest must define a non-empty 'npz_file'")
        npz_path = Path(npz_name)
        if not npz_path.is_absolute():
            npz_path = self.manifest_path.parent / npz_path
        self.npz_path = npz_path
        if not self.npz_path.is_file():
            raise FileNotFoundError(f"Mask archive not found: {self.npz_path}")

        expected_shape = self._document_shape(document)
        metadata: Dict[str, SparseMaskMetadata] = {}
        masks: Dict[str, Tensor] = {}

        with np.load(self.npz_path, allow_pickle=False) as archive:
            for raw_entry in entries:
                if not isinstance(raw_entry, dict):
                    raise ValueError("Each mask manifest entry must be a JSON object")
                mask_id = raw_entry.get("mask_id")
                if not isinstance(mask_id, str) or not mask_id:
                    raise ValueError("Each mask entry must define a non-empty 'mask_id'")
                if mask_id in metadata:
                    raise ValueError(f"Duplicate mask_id in manifest: {mask_id}")
                if mask_id not in archive.files:
                    raise KeyError(f"Mask {mask_id!r} is missing from {self.npz_path}")

                mask_type = raw_entry.get("mask_type")
                if mask_type not in SUPPORTED_MASK_TYPES:
                    raise ValueError(
                        f"Unsupported mask_type {mask_type!r} for {mask_id}; "
                        f"expected one of {sorted(SUPPORTED_MASK_TYPES)}"
                    )

                mask_array = np.asarray(archive[mask_id])
                if mask_array.ndim != 3:
                    raise ValueError(
                        f"Mask {mask_id} must have shape (T,H,W), got {mask_array.shape}"
                    )
                entry_shape = self._entry_shape(raw_entry, mask_array.shape)
                if tuple(mask_array.shape) != entry_shape:
                    raise ValueError(
                        f"Mask {mask_id} shape {mask_array.shape} does not match "
                        f"manifest entry {entry_shape}"
                    )
                if expected_shape is not None and entry_shape != expected_shape:
                    raise ValueError(
                        f"Mask {mask_id} shape {entry_shape} does not match manifest "
                        f"shape {expected_shape}"
                    )

                unique_values = np.unique(mask_array)
                if not np.isin(unique_values, (0, 1)).all():
                    raise ValueError(
                        f"Mask {mask_id} is not binary; values must use 1=observed, "
                        "0=missing"
                    )

                canonical = np.ascontiguousarray(mask_array, dtype=np.uint8)
                expected_hash = raw_entry.get("sha256")
                if verify_hashes and expected_hash is not None:
                    actual_hash = hashlib.sha256(canonical.tobytes()).hexdigest()
                    if actual_hash != expected_hash:
                        raise ValueError(
                            f"SHA256 mismatch for mask {mask_id}: expected "
                            f"{expected_hash}, got {actual_hash}"
                        )

                effective_rate = float(canonical.mean())
                declared_rate = raw_entry.get("effective_observation_rate")
                if declared_rate is not None and not math.isclose(
                    effective_rate,
                    float(declared_rate),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"Effective observation rate mismatch for mask {mask_id}: "
                        f"manifest={declared_rate}, archive={effective_rate}"
                    )
                temporal_dropout = float(raw_entry.get("temporal_dropout", 0.0))
                if not 0.0 <= temporal_dropout < 1.0:
                    raise ValueError(
                        f"temporal_dropout for {mask_id} must be in [0,1), "
                        f"got {temporal_dropout}"
                    )

                metadata[mask_id] = SparseMaskMetadata(
                    mask_id=mask_id,
                    mask_type=mask_type,
                    shape=entry_shape,
                    requested_observation_rate=self._optional_float(
                        raw_entry.get("requested_observation_rate")
                    ),
                    effective_observation_rate=effective_rate,
                    replicate=self._optional_int(raw_entry.get("replicate")),
                    seed=self._optional_int(raw_entry.get("seed")),
                    temporal_dropout=temporal_dropout,
                )
                masks[mask_id] = torch.from_numpy(canonical.astype(bool, copy=False))

        self._metadata = metadata
        self._masks = masks
        self.mask_ids = tuple(metadata)

    @staticmethod
    def _document_shape(document: Mapping[str, Any]) -> Optional[Tuple[int, int, int]]:
        keys = ("time_steps", "height", "width")
        if not all(key in document for key in keys):
            return None
        return tuple(int(document[key]) for key in keys)  # type: ignore[return-value]

    @staticmethod
    def _entry_shape(
        entry: Mapping[str, Any], fallback: Iterable[int]
    ) -> Tuple[int, int, int]:
        raw_shape = entry.get("shape", fallback)
        if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 3:
            raise ValueError(f"Invalid mask shape in manifest entry: {raw_shape!r}")
        return tuple(int(value) for value in raw_shape)  # type: ignore[return-value]

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        return None if value is None else float(value)

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        return None if value is None else int(value)

    def __len__(self) -> int:
        return len(self.mask_ids)

    def metadata(self, mask_id: str) -> SparseMaskMetadata:
        try:
            return self._metadata[mask_id]
        except KeyError as error:
            raise KeyError(f"Unknown mask_id: {mask_id}") from error

    def get_mask(self, mask_id: str) -> Tensor:
        """Return a clone so callers cannot mutate the cached deterministic mask."""

        try:
            return self._masks[mask_id].clone()
        except KeyError as error:
            raise KeyError(f"Unknown mask_id: {mask_id}") from error


class SparseSeaSurfaceDataset(Dataset):
    """Add deterministic sparse observations to an existing dense dataset.

    By default ``base_dataset`` returns ``{"x": Tensor, "y": Tensor}``.
    Reconstruction-only training can set ``include_target=False`` to require
    the base dataset's history-only interface instead.  Masks are assigned
    deterministically by cycling through ``mask_ids`` in manifest order.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        manifest_path: Union[str, Path],
        *,
        mask_ids: Optional[Sequence[str]] = None,
        fill_value: float = 0.0,
        verify_hashes: bool = True,
        include_target: bool = True,
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.manifest = SparseMaskManifest(
            manifest_path, verify_hashes=verify_hashes
        )
        selected_ids = self.manifest.mask_ids if mask_ids is None else tuple(mask_ids)
        if not selected_ids:
            raise ValueError("At least one mask_id must be selected")
        for mask_id in selected_ids:
            self.manifest.metadata(mask_id)
        self.mask_ids = tuple(selected_ids)
        self.fill_value = float(fill_value)
        self.include_target = bool(include_target)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def mask_id_for_index(self, idx: int) -> str:
        normalized_idx = self._normalize_index(idx)
        return self.mask_ids[normalized_idx % len(self.mask_ids)]

    def _normalize_index(self, idx: int) -> int:
        length = len(self)
        if idx < 0:
            idx += length
        if idx < 0 or idx >= length:
            raise IndexError(f"Dataset index out of range: {idx}")
        return idx

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = self._normalize_index(idx)
        if self.include_target:
            dense_sample = self.base_dataset[idx]
        else:
            history_loader = getattr(self.base_dataset, "get_history_only", None)
            if not callable(history_loader):
                raise TypeError(
                    "Target-free sparse training requires base_dataset.get_history_only"
                )
            dense_sample = history_loader(idx)
        if not isinstance(dense_sample, Mapping):
            raise TypeError("Wrapped dataset samples must be mappings")
        if "x" not in dense_sample:
            raise KeyError("Wrapped dataset samples must contain 'x'")
        if self.include_target and "y" not in dense_sample:
            raise KeyError("Target-enabled wrapped samples must contain 'y'")

        x_full = dense_sample["x"]
        y = dense_sample.get("y") if self.include_target else None
        if not isinstance(x_full, Tensor):
            raise TypeError("Wrapped dataset 'x' value must be torch.Tensor")
        if self.include_target and not isinstance(y, Tensor):
            raise TypeError("Wrapped dataset 'y' value must be torch.Tensor")
        if x_full.ndim != 3:
            raise ValueError(
                f"Expected dense input x with shape (T,H,W), got {tuple(x_full.shape)}"
            )

        mask_id = self.mask_id_for_index(idx)
        mask_bool = self.manifest.get_mask(mask_id)
        if tuple(mask_bool.shape) != tuple(x_full.shape):
            raise ValueError(
                f"Mask {mask_id} shape {tuple(mask_bool.shape)} does not match "
                f"x shape {tuple(x_full.shape)}"
            )

        # Only the historical input x participates in observation construction.
        # The future target y is returned untouched and is never read here.
        x_full = x_full.clone()
        obs_mask = mask_bool.to(device=x_full.device, dtype=x_full.dtype)
        fill = torch.as_tensor(
            self.fill_value, device=x_full.device, dtype=x_full.dtype
        )
        x_obs = torch.where(mask_bool.to(x_full.device), x_full, fill)

        # Preserve source-file provenance for per-MAT evaluation without exposing
        # it to the model. SeaSurfaceSimpleDataset stores (file_id, start_idx).
        source_id = dense_sample.get("source_id")
        if source_id is None:
            index_map = getattr(self.base_dataset, "index_map", None)
            if index_map is not None and idx < len(index_map):
                source_id = index_map[idx][0]
            else:
                source_id = idx

        sample = {
            "x_full": x_full,
            "x_obs": x_obs,
            "obs_mask": obs_mask,
            "mask_id": mask_id,
            "source_id": source_id,
        }
        if self.include_target:
            sample["y"] = y
        return sample


def sparse_model_inputs(sample: Mapping[str, Any]) -> Dict[str, Tensor]:
    """Extract only tensors that are allowed to enter a sparse-input model."""

    missing = [key for key in MODEL_INPUT_KEYS if key not in sample]
    if missing:
        raise KeyError(f"Sparse sample is missing model input fields: {missing}")
    inputs = {key: sample[key] for key in MODEL_INPUT_KEYS}
    if not all(isinstance(value, Tensor) for value in inputs.values()):
        raise TypeError("x_obs and obs_mask must be torch.Tensor values")
    return inputs  # type: ignore[return-value]
