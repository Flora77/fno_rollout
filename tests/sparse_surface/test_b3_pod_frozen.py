from types import SimpleNamespace

import pytest
import torch
from torch import nn

from neuralop.models.reconstructors import (
    PODSpatialReconstructor,
    fit_pod_basis_from_snapshots,
)
from neuralop.models.sparse_forecast_pipeline import BilinearFrozenRFNOPipeline
from scripts.sparse_surface.run_b3_pod_rank_selection import _load_train_snapshots


class TinyRFNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(60, 30, kernel_size=1)

    def forward(self, history):
        return self.projection(history)


def _orthonormal_modes(rank=4, points=16):
    torch.manual_seed(51)
    matrix = torch.randn(points, rank)
    return torch.linalg.qr(matrix).Q.transpose(0, 1).contiguous()


def test_fit_pod_basis_is_finite_and_orthonormal():
    torch.manual_seed(52)
    snapshots = torch.randn(40, 16)

    mean, modes, singular_values = fit_pod_basis_from_snapshots(
        snapshots, max_rank=4, seed=7, niter=2
    )

    assert mean.shape == (16,)
    assert modes.shape == (4, 16)
    assert singular_values.shape == (4,)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(modes).all()
    torch.testing.assert_close(modes @ modes.T, torch.eye(4), atol=1e-5, rtol=1e-5)


def test_gappy_pod_recovers_known_subspace_and_ignores_missing_fill():
    torch.manual_seed(53)
    modes = _orthonormal_modes(rank=3)
    mean = torch.linspace(-0.2, 0.2, 16)
    coefficients = torch.randn(2 * 60, 3)
    full = (mean + coefficients @ modes).reshape(2, 60, 4, 4)
    mask = torch.zeros_like(full)
    mask[..., ::2, :] = 1.0
    zero_fill = full * mask
    other_fill = torch.where(mask.bool(), full, torch.randn_like(full) * 100.0)
    reconstructor = PODSpatialReconstructor(
        mean, modes, rank=3, height=4, width=4, ridge=0.0
    )

    first = reconstructor(zero_fill, mask)
    second = reconstructor(other_fill, mask)

    torch.testing.assert_close(first, full, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


def test_gappy_pod_all_missing_returns_train_mean():
    modes = _orthonormal_modes(rank=2)
    mean = torch.linspace(-1.0, 1.0, 16)
    reconstructor = PODSpatialReconstructor(
        mean, modes, rank=2, height=4, width=4
    )
    value = torch.randn(1, 60, 4, 4)
    mask = torch.zeros_like(value)

    output = reconstructor(value, mask)

    expected = mean.reshape(1, 1, 4, 4).expand_as(output)
    torch.testing.assert_close(output, expected)


def test_pod_frozen_pipeline_shapes_and_rfno_gradients():
    modes = _orthonormal_modes(rank=4)
    reconstructor = PODSpatialReconstructor(
        torch.zeros(16), modes, rank=4, height=4, width=4
    )
    pipeline = BilinearFrozenRFNOPipeline(TinyRFNO(), reconstructor=reconstructor)
    x_full = torch.randn(2, 60, 4, 4)
    mask = torch.zeros_like(x_full)
    mask[..., ::2, :] = 1.0

    output = pipeline(x_full * mask, mask)

    assert output["history_reconstruction"].shape == (2, 60, 4, 4)
    assert output["forecast"].shape == (2, 30, 4, 4)
    assert torch.isfinite(output["history_reconstruction"]).all()
    assert torch.isfinite(output["forecast"]).all()
    assert all(not parameter.requires_grad for parameter in pipeline.rfno.parameters())
    assert pipeline.rfno_gradients_are_none()


def test_pod_basis_loader_rejects_non_train_source_before_reading_file():
    context = SimpleNamespace(
        data=SimpleNamespace(
            split_manifest={
                "data_root": ".",
                "splits": {
                    "train": [
                        {
                            "source_id": 0,
                            "name": "leak.mat",
                            "relative_path": "val/leak.mat",
                            "sha256": "unused",
                        }
                    ]
                },
            },
            normalization_mean=0.0,
            normalization_std=1.0,
        ),
        config={"data": {"height": 4, "width": 4, "variable": "height"}},
    )

    with pytest.raises(PermissionError, match="not in frozen train split"):
        _load_train_snapshots(context, torch.device("cpu"))
