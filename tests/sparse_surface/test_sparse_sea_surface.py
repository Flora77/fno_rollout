import hashlib
import json

import numpy as np
import pytest
import scipy.io as sio
import torch

from neuralop.data.datasets.sea_surface_simple import SeaSurfaceSimpleDataset
from neuralop.data.datasets.sparse_sea_surface import (
    SparseMaskManifest,
    SparseSeaSurfaceDataset,
    sparse_model_inputs,
)


def _write_dense_data(path, *, target_offset=0.0):
    path.mkdir()
    height = np.arange(8 * 4 * 4, dtype=np.float32).reshape(8, 4, 4)
    height[4:] += target_offset
    sio.savemat(path / "surface.mat", {"height": height})
    return SeaSurfaceSimpleDataset(
        str(path), input_steps=4, output_steps=4, stride=1, normalize=False
    )


def _example_masks():
    fixed = np.zeros((4, 4, 4), dtype=np.uint8)
    fixed[:, 0, 0] = 1
    fixed[:, 2, 3] = 1

    random_points = np.zeros((4, 4, 4), dtype=np.uint8)
    random_points[0, 0, 1] = 1
    random_points[1, 1, 2] = 1
    random_points[2, 2, 3] = 1
    random_points[3, 3, 0] = 1

    blocks = np.zeros((4, 4, 4), dtype=np.uint8)
    blocks[:, :2, :2] = 1

    stripes = np.zeros((4, 4, 4), dtype=np.uint8)
    stripes[:, :, ::2] = 1
    return {
        "fixed": ("fixed_points", fixed),
        "random": ("random_points", random_points),
        "blocks": ("blocks", blocks),
        "stripes": ("stripes", stripes),
    }


def _write_manifest(path):
    masks = _example_masks()
    np.savez_compressed(
        path / "test_masks.npz", **{key: value[1] for key, value in masks.items()}
    )
    entries = []
    for replicate, (mask_id, (mask_type, mask)) in enumerate(masks.items()):
        entries.append(
            {
                "mask_id": mask_id,
                "mask_type": mask_type,
                "requested_observation_rate": float(mask.mean()),
                "effective_observation_rate": float(mask.mean()),
                "replicate": replicate,
                "seed": 42 + replicate,
                "shape": list(mask.shape),
                "sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
            }
        )
    manifest = {
        "version": 1,
        "npz_file": "test_masks.npz",
        "height": 4,
        "width": 4,
        "time_steps": 4,
        "entries": entries,
    }
    manifest_path = path / "test_masks.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_wrapper_preserves_dense_contract_and_zero_fill_identity(tmp_path):
    dense = _write_dense_data(tmp_path / "dense")
    original = dense[0]
    manifest_path = _write_manifest(tmp_path)

    sparse = SparseSeaSurfaceDataset(dense, manifest_path, mask_ids=["fixed"])
    sample = sparse[0]

    assert set(dense[0]) == {"x", "y"}
    assert set(sample) == {
        "x_full",
        "x_obs",
        "obs_mask",
        "y",
        "mask_id",
        "source_id",
    }
    assert sample["source_id"] == 0
    assert sample["x_full"].shape == (4, 4, 4)
    assert sample["x_obs"].shape == (4, 4, 4)
    assert sample["obs_mask"].shape == (4, 4, 4)
    torch.testing.assert_close(sample["x_full"], original["x"])
    torch.testing.assert_close(sample["y"], original["y"])
    torch.testing.assert_close(
        sample["x_obs"], sample["x_full"] * sample["obs_mask"]
    )


def test_target_free_wrapper_uses_history_only_path(tmp_path, monkeypatch):
    dense = _write_dense_data(tmp_path / "dense")
    original = dense[0]
    manifest_path = _write_manifest(tmp_path)

    def reject_legacy_target_loading(self, idx):
        raise AssertionError("target-free training called legacy __getitem__")

    monkeypatch.setattr(
        SeaSurfaceSimpleDataset, "__getitem__", reject_legacy_target_loading
    )
    sample = SparseSeaSurfaceDataset(
        dense,
        manifest_path,
        mask_ids=["fixed"],
        include_target=False,
    )[0]

    assert set(sample) == {
        "x_full",
        "x_obs",
        "obs_mask",
        "mask_id",
        "source_id",
    }
    assert "y" not in sample
    torch.testing.assert_close(sample["x_full"], original["x"])


@pytest.mark.parametrize(
    ("mask_id", "mask_type"),
    [
        ("fixed", "fixed_points"),
        ("random", "random_points"),
        ("blocks", "blocks"),
        ("stripes", "stripes"),
    ],
)
def test_all_supported_mask_types_load(tmp_path, mask_id, mask_type):
    manifest_path = _write_manifest(tmp_path)
    manifest = SparseMaskManifest(manifest_path)

    mask = manifest.get_mask(mask_id)
    assert manifest.metadata(mask_id).mask_type == mask_type
    assert mask.shape == (4, 4, 4)
    assert mask.dtype == torch.bool
    assert set(mask.unique().tolist()).issubset({False, True})


def test_same_mask_id_is_exactly_reproducible(tmp_path):
    dense = _write_dense_data(tmp_path / "dense")
    manifest_path = _write_manifest(tmp_path)
    first = SparseSeaSurfaceDataset(dense, manifest_path, mask_ids=["random"])
    second = SparseSeaSurfaceDataset(dense, manifest_path, mask_ids=["random"])

    first_sample = first[0]
    second_sample = second[0]
    assert first_sample["mask_id"] == second_sample["mask_id"] == "random"
    assert torch.equal(first_sample["obs_mask"], second_sample["obs_mask"])
    assert torch.equal(first_sample["x_obs"], second_sample["x_obs"])


def test_fill_value_changes_only_missing_positions(tmp_path):
    dense = _write_dense_data(tmp_path / "dense")
    manifest_path = _write_manifest(tmp_path)
    zero_fill = SparseSeaSurfaceDataset(
        dense, manifest_path, mask_ids=["fixed"], fill_value=0.0
    )[0]
    other_fill = SparseSeaSurfaceDataset(
        dense, manifest_path, mask_ids=["fixed"], fill_value=-123.0
    )[0]
    observed = zero_fill["obs_mask"].bool()
    missing = ~observed

    assert torch.equal(zero_fill["x_obs"][observed], other_fill["x_obs"][observed])
    assert torch.equal(other_fill["x_obs"][observed], other_fill["x_full"][observed])
    assert torch.all(other_fill["x_obs"][missing] == -123.0)


def test_future_target_never_changes_observation_or_model_inputs(tmp_path):
    first_dense = _write_dense_data(tmp_path / "dense_a", target_offset=0.0)
    second_dense = _write_dense_data(tmp_path / "dense_b", target_offset=10000.0)
    manifest_path = _write_manifest(tmp_path)
    first = SparseSeaSurfaceDataset(first_dense, manifest_path, mask_ids=["stripes"])[0]
    second = SparseSeaSurfaceDataset(second_dense, manifest_path, mask_ids=["stripes"])[0]

    assert not torch.equal(first["y"], second["y"])
    assert torch.equal(first["x_full"], second["x_full"])
    assert torch.equal(first["x_obs"], second["x_obs"])
    assert torch.equal(first["obs_mask"], second["obs_mask"])
    assert set(sparse_model_inputs(first)) == {"x_obs", "obs_mask"}
    assert "x_full" not in sparse_model_inputs(first)
    assert "y" not in sparse_model_inputs(first)
