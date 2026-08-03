import math

import pytest
import torch
from torch import nn

from neuralop.evaluation.sparse_surface import (
    SparseHistoryMetricAccumulator,
    compute_b1_metrics,
)
from neuralop.models.reconstructors import BilinearSpatialReconstructor
from neuralop.models.sparse_forecast_pipeline import (
    BilinearFrozenRFNOPipeline,
    load_rfno_from_b0_checkpoint,
)


class TinyRFNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(60, 30, kernel_size=1)

    def forward(self, history):
        return self.projection(history)


class TrainableScaleReconstructor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x_obs, obs_mask):
        return self.scale * x_obs


def test_b0_checkpoint_must_contain_training_normalization(tmp_path):
    checkpoint_path = tmp_path / "missing_normalization.pt"
    torch.save({"config": {}, "model_state_dict": {}}, checkpoint_path)

    with pytest.raises(KeyError, match="training-set normalization"):
        load_rfno_from_b0_checkpoint(checkpoint_path)


def test_all_observed_reconstruction_is_exact_identity():
    history = torch.randn(2, 60, 8, 8)
    mask = torch.ones_like(history)
    reconstruction = BilinearSpatialReconstructor()(history, mask)

    assert torch.equal(reconstruction, history)


def test_interpolation_is_frame_independent_and_finite():
    x_obs = torch.zeros(1, 2, 8, 8)
    mask = torch.zeros_like(x_obs)
    mask[:, :, ::3, ::3] = 1
    x_obs[0, 0, mask[0, 0].bool()] = 1.0
    x_obs[0, 1, mask[0, 1].bool()] = 2.0

    reconstructor = BilinearSpatialReconstructor()
    reconstruction = reconstructor(x_obs, mask)
    changed = x_obs.clone()
    changed[:, 1] *= 5.0
    changed_reconstruction = reconstructor(changed, mask)

    assert torch.isfinite(reconstruction).all()
    assert torch.equal(reconstruction[:, 0], changed_reconstruction[:, 0])
    assert torch.equal(reconstruction[mask.bool()], x_obs[mask.bool()])


def test_pipeline_shapes_finiteness_and_rfno_freeze():
    rfno = TinyRFNO()
    pipeline = BilinearFrozenRFNOPipeline(rfno)
    pipeline.train()
    x_full = torch.randn(2, 60, 8, 8)
    mask = torch.zeros_like(x_full)
    mask[:, :, ::2, ::2] = 1
    outputs = pipeline(x_full * mask, mask)

    assert outputs["history_reconstruction"].shape == (2, 60, 8, 8)
    assert outputs["forecast"].shape == (2, 30, 8, 8)
    assert torch.isfinite(outputs["history_reconstruction"]).all()
    assert torch.isfinite(outputs["forecast"]).all()
    assert not pipeline.rfno.training
    assert all(not parameter.requires_grad for parameter in pipeline.rfno.parameters())
    assert pipeline.rfno_gradients_are_none()
    assert tuple(pipeline.optimizer_parameters()) == ()


def test_b1_long_rollout_is_300_frames_without_future_inputs():
    pipeline = BilinearFrozenRFNOPipeline(TinyRFNO())
    x_full = torch.randn(1, 60, 8, 8)
    mask = torch.ones_like(x_full)

    outputs = pipeline.rollout(x_full, mask, rollout_steps=300)

    assert outputs["history_reconstruction"].shape == (1, 60, 8, 8)
    assert outputs["forecast"].shape == (1, 300, 8, 8)
    assert torch.isfinite(outputs["forecast"]).all()
    assert pipeline.rfno_gradients_are_none()


def test_b1_all_missing_end_to_end_is_finite():
    pipeline = BilinearFrozenRFNOPipeline(TinyRFNO())
    x_obs = torch.zeros(1, 60, 8, 8)
    mask = torch.zeros_like(x_obs)

    outputs = pipeline(x_obs, mask)

    assert torch.equal(
        outputs["history_reconstruction"],
        torch.zeros_like(outputs["history_reconstruction"]),
    )
    assert torch.isfinite(outputs["forecast"]).all()


def test_backward_never_populates_frozen_rfno_gradients():
    reconstructor = TrainableScaleReconstructor()
    pipeline = BilinearFrozenRFNOPipeline(
        TinyRFNO(), reconstructor=reconstructor
    )
    optimizer_parameters = tuple(pipeline.optimizer_parameters())
    outputs = pipeline(torch.randn(1, 60, 8, 8), torch.ones(1, 60, 8, 8))
    outputs["forecast"].square().mean().backward()

    assert len(optimizer_parameters) == 1
    assert optimizer_parameters[0] is reconstructor.scale
    assert reconstructor.scale.grad is not None
    assert torch.isfinite(reconstructor.scale.grad)
    assert pipeline.rfno_gradients_are_none()


def test_metrics_are_separated_and_use_physical_normalization():
    x_full = torch.randn(1, 60, 8, 8)
    history = x_full + 0.1
    mask = torch.ones_like(x_full)
    mask[:, :, 1::2, 1::2] = 0
    y = torch.randn(1, 30, 8, 8)
    forecast = y + 0.2

    metrics = compute_b1_metrics(
        history,
        x_full,
        forecast,
        y,
        mask,
        source_ids=["surface_0"],
        normalization_mean=-3.99e-10,
        normalization_std=0.8006430268287659,
    )

    assert set(metrics) == {
        "history_reconstruction",
        "forecast_30",
        "error_growth_curve",
        "source_count",
        "per_source",
    }
    assert metrics["history_reconstruction"]["rmse"] == pytest.approx(
        0.1 * 0.8006430268287659, rel=1e-5
    )
    assert metrics["forecast_30"]["rmse"] == pytest.approx(
        0.2 * 0.8006430268287659, rel=1e-5
    )
    assert len(metrics["error_growth_curve"]) == 30
    assert metrics["error_growth_curve"][-1]["frame"] == 30
    assert metrics["history_reconstruction"]["missing_region"]["rmse"] == pytest.approx(
        0.1 * 0.8006430268287659, rel=1e-5
    )
    assert all(
        math.isfinite(value)
        for section in (
            metrics["history_reconstruction"],
            metrics["forecast_30"],
        )
        for value in section.values()
        if isinstance(value, (int, float))
    )


def test_metrics_are_computed_per_source_before_equal_weight_averaging():
    x_full = torch.stack(
        [torch.ones(60, 4, 4), torch.arange(60 * 4 * 4).reshape(60, 4, 4)]
    ).float()
    history = x_full + 0.1
    y = torch.stack(
        [torch.ones(30, 4, 4), torch.arange(30 * 4 * 4).reshape(30, 4, 4)]
    ).float()
    forecast = y + 0.2
    mask = torch.ones_like(x_full)

    metrics = compute_b1_metrics(
        history,
        x_full,
        forecast,
        y,
        mask,
        source_ids=["low_energy", "high_energy"],
        normalization_mean=0.0,
        normalization_std=1.0,
    )

    per_source = metrics["per_source"]
    expected = sum(
        value["forecast_30"]["nrmse"] for value in per_source.values()
    ) / 2.0
    assert metrics["forecast_30"]["nrmse"] == pytest.approx(expected)
    assert metrics["source_count"] == 2


def test_history_metrics_support_p1_missing_region_model_selection():
    x_full = torch.randn(2, 60, 4, 4)
    reconstruction = x_full + 0.2
    mask = torch.ones_like(x_full)
    mask[:, :, ::2, ::2] = 0
    accumulator = SparseHistoryMetricAccumulator(
        normalization_mean=0.0, normalization_std=2.0
    )
    accumulator.update(
        reconstruction,
        x_full,
        mask,
        source_ids=["mat_a", "mat_b"],
    )
    metrics = accumulator.compute()

    assert metrics["source_count"] == 2
    assert metrics["full"]["rmse"] == pytest.approx(0.4, rel=1e-5)
    assert metrics["missing_region"]["rmse"] == pytest.approx(0.4, rel=1e-5)
    assert metrics["observed_region"]["rmse"] == pytest.approx(0.4, rel=1e-5)
    assert math.isfinite(metrics["full"]["ssp"])
    assert math.isfinite(metrics["full"]["gradient_rmse"])
