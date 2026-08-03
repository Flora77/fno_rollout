import torch
from torch import nn

from neuralop.models.reconstructors import PartialConvMaskedAutoencoder
from neuralop.models.sparse_forecast_pipeline import PartialConvMAERFNOPipeline
from neuralop.training.sparse_coupled_forecast import (
    SparseCoupledForecastTrainer,
    SparseCoupledTrainingConfig,
)


class TinyRFNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(60, 30, kernel_size=1)

    def forward(self, history):
        return self.projection(history)


def _pipeline(coupling):
    reconstructor = PartialConvMaskedAutoencoder(
        input_steps=60,
        encoder_channels=(8,),
        decoder_channels=(8,),
    )
    return PartialConvMAERFNOPipeline(
        TinyRFNO(), reconstructor, coupling=coupling
    )


def _batch(future_steps=60):
    torch.manual_seed(21)
    x_full = torch.randn(2, 60, 8, 8)
    mask = (torch.rand_like(x_full) > 0.5).float()
    return {
        "x_full": x_full,
        "x_obs": x_full * mask,
        "obs_mask": mask,
        "y": torch.randn(2, future_steps, 8, 8),
    }


def _finite_non_none_gradients(module):
    gradients = [parameter.grad for parameter in module.parameters()]
    present = [gradient for gradient in gradients if gradient is not None]
    return bool(present) and all(torch.isfinite(gradient).all() for gradient in present)


def test_curriculum_starts_at_30_and_reaches_300():
    config = SparseCoupledTrainingConfig(coupling="joint")

    assert config.curriculum_steps() == (30, 60, 120, 180, 240, 300)
    assert config.active_rollout_steps(epoch=1, total_epochs=100) == 30
    assert config.active_rollout_steps(epoch=100, total_epochs=100) == 300


def test_joint_one_batch_updates_both_modules_with_smaller_rfno_lr():
    pipeline = _pipeline("frozen")
    config = SparseCoupledTrainingConfig(
        coupling="joint",
        reconstructor_lr=2.0e-3,
        rfno_lr=2.0e-4,
        history_gradient_weight=0.05,
        forecast_gradient_weight=0.05,
        spectrum_weight=0.01,
    )
    trainer = SparseCoupledForecastTrainer(pipeline, config)
    optimizer = trainer.build_optimizer()
    result = trainer.train_batch(_batch(), optimizer, active_rollout_steps=30)

    assert pipeline.coupling == "joint"
    assert [group["name"] for group in optimizer.param_groups] == [
        "reconstructor",
        "rfno",
    ]
    assert optimizer.param_groups[1]["lr"] < optimizer.param_groups[0]["lr"]
    assert result["history_reconstruction"].shape == (2, 60, 8, 8)
    assert result["forecast"].shape == (2, 30, 8, 8)
    assert _finite_non_none_gradients(pipeline.reconstructor)
    assert _finite_non_none_gradients(pipeline.rfno)
    assert set(result["losses"]) == {
        "hidden_reconstruction",
        "observation_consistency",
        "history_gradient",
        "rollout",
        "forecast_gradient",
        "spectrum",
        "weighted_hidden_reconstruction",
        "weighted_observation_consistency",
        "weighted_history_gradient",
        "weighted_rollout",
        "weighted_forecast_gradient",
        "weighted_spectrum",
        "total",
    }
    assert all(torch.isfinite(value) for value in result["losses"].values())


def test_short_two_chunk_rollout_train_and_validation_are_finite():
    pipeline = _pipeline("joint")
    config = SparseCoupledTrainingConfig(coupling="joint")
    trainer = SparseCoupledForecastTrainer(pipeline, config)
    optimizer = trainer.build_optimizer()
    batch = _batch(future_steps=60)

    train_result = trainer.train_batch(batch, optimizer, active_rollout_steps=60)
    validation_result = trainer.validate_batch(batch, active_rollout_steps=60)

    for result in (train_result, validation_result):
        assert result["forecast"].shape == (2, 60, 8, 8)
        assert result["active_rollout_steps"] == 60
        assert torch.isfinite(result["forecast"]).all()
        assert all(torch.isfinite(value) for value in result["losses"].values())


def test_p1_frozen_regression_excludes_rfno_and_keeps_gradients_none():
    pipeline = _pipeline("joint")
    config = SparseCoupledTrainingConfig(coupling="frozen")
    trainer = SparseCoupledForecastTrainer(pipeline, config)
    optimizer = trainer.build_optimizer()
    result = trainer.train_batch(_batch(), optimizer, active_rollout_steps=30)

    assert pipeline.coupling == "frozen"
    assert [group["name"] for group in optimizer.param_groups] == ["reconstructor"]
    assert all(not parameter.requires_grad for parameter in pipeline.rfno.parameters())
    assert pipeline.rfno_gradients_are_none()
    assert _finite_non_none_gradients(pipeline.reconstructor)
    assert result["forecast"].shape == (2, 30, 8, 8)
    assert config.forecast_supervision is False
    assert result["losses"]["rollout"].item() == 0.0
    assert result["losses"]["forecast_gradient"].item() == 0.0
    assert result["losses"]["spectrum"].item() == 0.0


def test_p2_joint_all_observed_and_all_missing_batches_are_finite():
    for mask_value in (0.0, 1.0):
        pipeline = _pipeline("joint")
        config = SparseCoupledTrainingConfig(coupling="joint")
        trainer = SparseCoupledForecastTrainer(pipeline, config)
        optimizer = trainer.build_optimizer()
        batch = _batch(future_steps=30)
        batch["obs_mask"] = torch.full_like(batch["x_full"], mask_value)
        batch["x_obs"] = batch["x_full"] * batch["obs_mask"]

        result = trainer.train_batch(batch, optimizer, active_rollout_steps=30)

        assert torch.isfinite(result["history_reconstruction"]).all()
        assert torch.isfinite(result["forecast"]).all()
        assert all(torch.isfinite(value) for value in result["losses"].values())
        assert _finite_non_none_gradients(pipeline.rfno)
