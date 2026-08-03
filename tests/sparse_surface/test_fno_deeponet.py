from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neuralop.models.fno_deeponet import (
    FNODeepONet,
    FNODeepONetGridForecaster,
    FNODeepONetReconstructor,
)
from neuralop.models.sparse_forecast_pipeline import (
    FNODeepONetDirectSparsePipeline,
    LearnedReconstructionRFNOPipeline,
)
from neuralop.training.sparse_coupled_forecast import (
    SparseCoupledForecastTrainer,
    SparseCoupledTrainingConfig,
)
from neuralop.training.sparse_experiment_runner import SUPPORTED_EXPERIMENTS


def _operator(
    *,
    input_steps: int = 4,
    height: int = 4,
    width: int = 4,
) -> FNODeepONet:
    return FNODeepONet(
        input_steps=input_steps,
        height=height,
        width=width,
        branch_width=4,
        branch_modes=3,
        branch_layers=2,
        branch_hidden=8,
        latent_dim=4,
        trunk_hidden=8,
        trunk_layers=3,
        time_fourier_bands=2,
        spatial_fourier_bands=2,
        time_max_frequency=4.0,
        query_chunk_size=32,
    )


@pytest.mark.parametrize("scenario", ("all_valid", "all_missing", "isolated", "irregular"))
def test_fno_deeponet_mask_scenarios_are_finite(scenario: str) -> None:
    torch.manual_seed(7)
    model = _operator()
    values = torch.randn(2, 4, 4, 4)
    mask = torch.zeros_like(values)
    if scenario == "all_valid":
        mask.fill_(1)
    elif scenario == "isolated":
        mask[:, :, 1, 2] = 1
    elif scenario == "irregular":
        mask[:, :, 0, 0] = 1
        mask[:, :, 1:3, 2] = 1
        mask[:, 1::2, 3, 1:4] = 1
    prediction = model(values, mask, torch.tensor([-1.0, 0.0, 0.5]))
    assert prediction.shape == (2, 3, 4, 4)
    assert torch.isfinite(prediction).all()


def test_masked_fill_values_cannot_change_fno_deeponet_output() -> None:
    torch.manual_seed(11)
    model = _operator().eval()
    mask = torch.zeros(1, 4, 4, 4)
    mask[:, :, 0, 0] = 1
    mask[:, :, 2, 3] = 1
    observed = torch.randn_like(mask)
    first = observed * mask + 17.0 * (1.0 - mask)
    second = observed * mask - 103.0 * (1.0 - mask)
    query_times = torch.tensor([-1.0, -0.25, 0.5])
    with torch.no_grad():
        first_output = model(first, mask, query_times)
        second_output = model(second, mask, query_times)
    torch.testing.assert_close(first_output, second_output, rtol=0.0, atol=0.0)


def test_reconstructor_and_autoregressive_forecaster_gradient_policies() -> None:
    torch.manual_seed(13)
    x_obs = torch.randn(1, 4, 4, 4)
    obs_mask = torch.zeros_like(x_obs)
    obs_mask[:, :, ::2, ::2] = 1

    frozen_pipeline = LearnedReconstructionRFNOPipeline(
        FNODeepONetGridForecaster(_operator(), output_steps=2),
        FNODeepONetReconstructor(_operator()),
        input_steps=4,
        output_steps=2,
        coupling="frozen",
    )
    frozen_output = frozen_pipeline.rollout(
        x_obs,
        obs_mask,
        rollout_steps=4,
    )
    assert frozen_output["history_reconstruction"].shape == x_obs.shape
    assert frozen_output["forecast"].shape == (1, 4, 4, 4)
    frozen_output["forecast"].square().mean().backward()
    assert frozen_pipeline.rfno_gradients_are_none()
    assert any(
        parameter.grad is not None
        for parameter in frozen_pipeline.reconstructor.parameters()
    )

    joint_pipeline = LearnedReconstructionRFNOPipeline(
        FNODeepONetGridForecaster(_operator(), output_steps=2),
        FNODeepONetReconstructor(_operator()),
        input_steps=4,
        output_steps=2,
        coupling="joint",
    )
    joint_output = joint_pipeline.rollout(
        x_obs,
        obs_mask,
        rollout_steps=4,
    )
    joint_output["forecast"].square().mean().backward()
    reconstructor_gradients = [
        parameter.grad
        for parameter in joint_pipeline.reconstructor.parameters()
        if parameter.grad is not None
    ]
    forecaster_gradients = [
        parameter.grad
        for parameter in joint_pipeline.rfno.parameters()
        if parameter.grad is not None
    ]
    assert reconstructor_gradients and forecaster_gradients
    assert all(torch.isfinite(gradient).all() for gradient in reconstructor_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in forecaster_gradients)
    assert any(torch.count_nonzero(gradient) for gradient in forecaster_gradients)


def test_fd_a1_one_batch_forward_backward_uses_curriculum_contract() -> None:
    torch.manual_seed(17)
    reconstructor = FNODeepONetReconstructor(
        _operator(input_steps=60, height=2, width=2)
    )
    forecaster = FNODeepONetGridForecaster(
        _operator(input_steps=60, height=2, width=2),
        output_steps=30,
    )
    pipeline = LearnedReconstructionRFNOPipeline(
        forecaster,
        reconstructor,
        input_steps=60,
        output_steps=30,
        coupling="joint",
    )
    config = SparseCoupledTrainingConfig(
        coupling="joint",
        input_steps=60,
        one_shot_steps=30,
        max_rollout_steps=300,
        use_rollout_curriculum=True,
        rollout_train_steps=(30, 60, 120, 180, 240, 300),
        rollout_curriculum_boundaries=(0.0, 0.1, 0.2, 0.3, 0.45, 0.6),
        forecast_supervision=True,
        spectrum_weight=0.0,
    )
    assert config.active_rollout_steps(1, 100) == 30
    assert config.active_rollout_steps(100, 100) == 300
    trainer = SparseCoupledForecastTrainer(pipeline, config)
    optimizer = trainer.build_optimizer()
    x_full = torch.randn(1, 60, 2, 2)
    mask = torch.zeros_like(x_full)
    mask[:, :, 0, 0] = 1
    result = trainer.train_batch(
        {
            "x_obs": x_full * mask,
            "obs_mask": mask,
            "x_full": x_full,
            "y": torch.randn(1, 300, 2, 2),
        },
        optimizer,
        active_rollout_steps=30,
    )
    assert result["history_reconstruction"].shape == (1, 60, 2, 2)
    assert result["forecast"].shape == (1, 30, 2, 2)
    assert torch.isfinite(result["losses"]["total"])


def test_fd_l1_direct_300_frame_forward_backward() -> None:
    torch.manual_seed(19)
    pipeline = FNODeepONetDirectSparsePipeline(
        _operator(input_steps=60, height=2, width=2),
        forecast_steps=300,
    )
    config = SparseCoupledTrainingConfig(
        coupling="direct",
        input_steps=60,
        one_shot_steps=30,
        max_rollout_steps=300,
        use_rollout_curriculum=False,
        forecast_supervision=True,
        spectrum_weight=0.0,
    )
    trainer = SparseCoupledForecastTrainer(pipeline, config)
    optimizer = trainer.build_optimizer()
    x_full = torch.randn(1, 60, 2, 2)
    mask = torch.zeros_like(x_full)
    mask[:, :, 1, 1] = 1
    result = trainer.train_batch(
        {
            "x_obs": x_full * mask,
            "obs_mask": mask,
            "x_full": x_full,
            "y": torch.randn(1, 300, 2, 2),
        },
        optimizer,
        active_rollout_steps=300,
    )
    assert result["history_reconstruction"].shape == (1, 60, 2, 2)
    assert result["forecast"].shape == (1, 300, 2, 2)
    assert torch.isfinite(result["losses"]["total"])
    gradients = [
        parameter.grad
        for parameter in pipeline.direct_operator.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_formal_fno_deeponet_configs_share_comparison_controls() -> None:
    root = Path(__file__).resolve().parents[2]
    names = (
        "fd_r1_fno_deeponet_frozen_rfno_seed42.formal.json",
        "fd_a1_fno_deeponet_autoregressive_seed42.formal.json",
        "fd_l1_fno_deeponet_direct_long_seed42.formal.json",
    )
    configs = [
        json.loads((root / "config" / "sparse_experiments" / name).read_text())
        for name in names
    ]
    assert {"FD-R1", "FD-A1", "FD-L1"} <= SUPPORTED_EXPERIMENTS
    controls = [
        (
            config["data"]["split_manifest_path"],
            config["observation"],
            config["runtime"]["seed"],
            config["data"]["rollout_steps"],
        )
        for config in configs
    ]
    assert controls[0] == controls[1] == controls[2]
