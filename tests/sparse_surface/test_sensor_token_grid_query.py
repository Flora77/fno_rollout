import json
from pathlib import Path

import torch

from neuralop.models.reconstructors import (
    CrossAttentionQueryBlock,
    PeriodicSensorTokenGridQueryReconstructor,
    PeriodicViTMaskedAutoencoder,
    periodic_fourier_coordinates,
)
from neuralop.training.sparse_experiment_runner import (
    SUPPORTED_EXPERIMENTS,
    load_resolved_sparse_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "sparse_experiments"
    / "st_d0_sensor_token_grid_query.example.json"
)


def _tiny_model():
    return PeriodicSensorTokenGridQueryReconstructor(
        input_steps=60,
        dimension=32,
        encoder_depth=2,
        encoder_heads=4,
        encoder_mlp_ratio=2.0,
        decoder_depth=2,
        decoder_heads=4,
        decoder_mlp_ratio=2.0,
        coordinate_bands=4,
        query_chunk_size=16,
    )


def _sensor_inputs(batch=2, sensors=5):
    values = torch.randn(batch, sensors, 60)
    coordinates = torch.rand(batch, sensors, 2)
    mask = torch.ones(batch, sensors, dtype=torch.bool)
    return values, coordinates, mask


def test_sensor_token_grid_query_is_sensor_permutation_invariant():
    torch.manual_seed(1)
    model = _tiny_model().eval()
    values, coordinates, sensor_mask = _sensor_inputs()
    permutation = torch.tensor([3, 0, 4, 1, 2])

    with torch.no_grad():
        baseline, _ = model.reconstruct_from_sensor_tokens(
            values, coordinates, sensor_mask, height=8, width=8
        )
        permuted, _ = model.reconstruct_from_sensor_tokens(
            values[:, permutation],
            coordinates[:, permutation],
            sensor_mask[:, permutation],
            height=8,
            width=8,
        )

    torch.testing.assert_close(baseline, permuted, rtol=2.0e-5, atol=2.0e-6)


def test_sensor_token_grid_query_excludes_padded_sensors():
    torch.manual_seed(2)
    model = _tiny_model().eval()
    values, coordinates, sensor_mask = _sensor_inputs(batch=1, sensors=3)
    padded_values = torch.cat((values, torch.randn(1, 2, 60) * 100.0), dim=1)
    padded_coordinates = torch.cat((coordinates, torch.randn(1, 2, 2) * 100.0), dim=1)
    padded_mask = torch.tensor([[True, True, True, False, False]])

    with torch.no_grad():
        baseline, _ = model.reconstruct_from_sensor_tokens(
            values, coordinates, sensor_mask, height=8, width=8
        )
        padded, aux = model.reconstruct_from_sensor_tokens(
            padded_values,
            padded_coordinates,
            padded_mask,
            height=8,
            width=8,
        )

    torch.testing.assert_close(baseline, padded, rtol=0.0, atol=0.0)
    assert aux["sensor_count"].tolist() == [3]


def test_sensor_and_query_coordinates_are_unit_periodic():
    model = _tiny_model().eval()
    values = torch.randn(1, 3, 60)
    coordinates = torch.tensor([[[0.0, 0.125], [0.25, 0.5], [0.75, 0.875]]])
    sensor_mask = torch.ones(1, 3, dtype=torch.bool)

    torch.testing.assert_close(
        periodic_fourier_coordinates(coordinates, 4),
        periodic_fourier_coordinates(coordinates + 1.0, 4),
        rtol=0.0,
        atol=0.0,
    )
    with torch.no_grad():
        baseline, _ = model.reconstruct_from_sensor_tokens(
            values, coordinates, sensor_mask, height=8, width=8
        )
        wrapped, _ = model.reconstruct_from_sensor_tokens(
            values, coordinates + 1.0, sensor_mask, height=8, width=8
        )
    torch.testing.assert_close(baseline, wrapped, rtol=0.0, atol=0.0)


def test_sensor_token_grid_query_mask_boundaries_and_fill_invariance():
    torch.manual_seed(3)
    model = _tiny_model().eval()
    x_full = torch.randn(1, 60, 8, 8)
    masks = {
        "all_observed": torch.ones_like(x_full),
        "all_missing": torch.zeros_like(x_full),
        "isolated": torch.zeros_like(x_full),
        "irregular": torch.zeros_like(x_full),
    }
    masks["isolated"][:, :, 2, 3] = 1.0
    masks["irregular"][:, :, ::2, 1::3] = 1.0

    with torch.no_grad():
        for name, mask in masks.items():
            fills = (
                torch.where(mask.bool(), x_full, torch.zeros_like(x_full)),
                torch.where(mask.bool(), x_full, torch.full_like(x_full, 100.0)),
                torch.where(mask.bool(), x_full, torch.randn_like(x_full) * 17.0),
            )
            outputs = [model(value, mask) for value in fills]
            assert outputs[0].shape == (1, 60, 8, 8), name
            assert torch.isfinite(outputs[0]).all(), name
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)
            torch.testing.assert_close(outputs[0], outputs[2], rtol=0.0, atol=0.0)

        _, all_observed_aux = model.forward_with_aux(
            x_full, masks["all_observed"]
        )
        _, all_missing_aux = model.forward_with_aux(
            torch.full_like(x_full, 100.0), masks["all_missing"]
        )
        _, isolated_aux = model.forward_with_aux(
            x_full * masks["isolated"], masks["isolated"]
        )

    assert all_observed_aux["sensor_count"].tolist() == [64]
    assert all_missing_aux["sensor_count"].tolist() == [0]
    assert all_missing_aux["empty_context"].tolist() == [True]
    assert isolated_aux["sensor_count"].tolist() == [1]


def test_sensor_token_grid_query_backward_is_finite():
    torch.manual_seed(4)
    model = _tiny_model()
    x_full = torch.randn(2, 60, 8, 8)
    mask = torch.zeros_like(x_full)
    mask[:, :, ::2, ::2] = 1.0
    reconstruction = model(x_full * mask, mask)
    loss = ((reconstruction - x_full).square() * (1.0 - mask)).mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert reconstruction.shape == (2, 60, 8, 8)
    assert torch.isfinite(loss)
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_sensor_token_grid_query_matches_v0_parameter_budget_and_contract():
    v0 = PeriodicViTMaskedAutoencoder()
    model = PeriodicSensorTokenGridQueryReconstructor()
    v0_parameters = sum(parameter.numel() for parameter in v0.parameters())
    parameters = sum(parameter.numel() for parameter in model.parameters())

    assert abs(parameters - v0_parameters) / v0_parameters <= 0.05
    assert model.dimension == 128
    assert len(model.sensor_encoder.layers) == 4
    assert len(model.query_decoder) == 3
    assert all(
        isinstance(block, CrossAttentionQueryBlock)
        for block in model.query_decoder
    )
    assert model.hard_observation_consistency is False


def test_st_d0_config_resolves_without_test_access(tmp_path):
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert raw["training"]["pretrain_reconstructor"] is False
    assert raw["training"]["mae_mask_ratio"] == 0.0
    assert raw["model"]["hard_observation_consistency"] is False

    resolved = load_resolved_sparse_config(
        CONFIG_PATH,
        project_root=PROJECT_ROOT,
        run_dir=tmp_path / "st_d0",
        device="cpu",
    )
    assert "ST-D0" in SUPPORTED_EXPERIMENTS
    assert resolved["experiment_id"] == "ST-D0"
