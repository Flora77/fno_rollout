import pytest
import torch
from torch import nn

from neuralop.models.reconstructors import (
    PeriodicGINOReconstructor,
    PeriodicGNOReconstructor,
)
from neuralop.models.sparse_forecast_pipeline import LearnedReconstructionRFNOPipeline
from neuralop.training.masked_reconstruction import (
    MaskedReconstructionLoss,
    MaskedReconstructionPretrainer,
)


class TinyRFNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(60, 30, 1)

    def forward(self, history):
        return self.projection(history)


def _inputs(batch=2, size=8):
    torch.manual_seed(81)
    full = torch.randn(batch, 60, size, size)
    mask = torch.zeros_like(full)
    mask[:, :, ::2, ::2] = 1.0
    return full, mask


def _models():
    return (
        PeriodicGNOReconstructor(
            input_steps=60,
            hidden_channels=4,
            radius=0.6,
            mlp_channels=(8,),
            kernel_mode="channelwise_nonlinear",
        ),
        PeriodicGINOReconstructor(
            input_steps=60,
            hidden_channels=4,
            latent_shape=(4, 4),
            radius=0.6,
            n_modes=(2, 2),
            fno_layers=1,
            mlp_channels=(8,),
            kernel_mode="channelwise_nonlinear",
            latent_positional_embedding="none",
        ),
    )


def _residual_graph_model(**overrides):
    options = {
        "input_steps": 60,
        "hidden_channels": 8,
        "radius": 0.6,
        "mlp_channels": (12,),
        "kernel_mode": "channelwise_nonlinear",
        "kernel_rank": 8,
        "architecture": "residual_graph_v3",
        "temporal_channels": 4,
        "temporal_dilations": (1, 2),
        "graph_layers": 2,
        "grid_refinement_layers": 2,
        "kernel_normalization": "neighbor_mean",
        "query_chunk_size": 16,
        "hard_observation_consistency": True,
        "raw_observation_supervision": True,
    }
    options.update(overrides)
    return PeriodicGNOReconstructor(**options)


def _residual_graph_v4_model(**overrides):
    options = {
        "input_steps": 60,
        "hidden_channels": 8,
        "radius": 0.3,
        "mlp_channels": (12,),
        "kernel_mode": "channelwise_nonlinear",
        "kernel_rank": 4,
        "architecture": "residual_graph_v4",
        "temporal_channels": 4,
        "temporal_dilations": (1, 2),
        "temporal_tokens": 2,
        "relative_fourier_frequencies": (1, 2, 4),
        "cache_geometry": True,
        "graph_layers": 2,
        "grid_refinement_layers": 2,
        "kernel_normalization": "neighbor_mean",
        "query_chunk_size": 16,
        "hard_observation_consistency": True,
        "raw_observation_supervision": True,
    }
    options.update(overrides)
    return PeriodicGNOReconstructor(**options)


@pytest.mark.parametrize("model", _models())
def test_coordinate_reconstructor_shape_finite_and_fill_invariant(model):
    full, mask = _inputs()
    zero_fill = full * mask
    other_fill = torch.where(mask.bool(), full, torch.randn_like(full) * 100.0)
    model.eval()

    with torch.no_grad():
        first = model(zero_fill, mask)
        second = model(other_fill, mask)

    assert first.shape == full.shape
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_coordinate_reconstructor_rejects_time_varying_layout():
    full, mask = _inputs(batch=1)
    mask[:, 1, 0, 0] = 0.0
    with pytest.raises(ValueError, match="time-invariant"):
        _models()[0](full * mask, mask)


@pytest.mark.parametrize("model", _models())
def test_coordinate_reconstructor_hard_observation_consistency(model):
    model.hard_observation_consistency = True
    full, mask = _inputs(batch=1)

    reconstruction = model(full * mask, mask)

    torch.testing.assert_close(
        reconstruction[mask.bool()],
        full[mask.bool()],
        rtol=0.0,
        atol=0.0,
    )


def test_nonlinear_gno_is_not_an_affine_value_interpolator():
    torch.manual_seed(82)
    model = PeriodicGNOReconstructor(
        input_steps=60,
        hidden_channels=8,
        radius=0.6,
        mlp_channels=(12,),
        kernel_mode="channelwise_nonlinear",
    )
    _, mask = _inputs(batch=1)
    first = torch.randn_like(mask) * mask
    second = torch.randn_like(mask) * mask
    zero = torch.zeros_like(mask)

    affine_residual = (
        model(first + second, mask)
        - model(first, mask)
        - model(second, mask)
        + model(zero, mask)
    )

    assert float(affine_residual.abs().max()) > 1.0e-5


def test_legacy_scalar_gno_state_dict_remains_loadable():
    original = PeriodicGNOReconstructor(
        input_steps=60,
        hidden_channels=4,
        radius=0.6,
        mlp_channels=(8,),
        kernel_mode="scalar_legacy",
    )
    restored = PeriodicGNOReconstructor(
        input_steps=60,
        hidden_channels=4,
        radius=0.6,
        mlp_channels=(8,),
        kernel_mode="scalar_legacy",
    )

    restored.load_state_dict(original.state_dict(), strict=True)


def test_residual_graph_gno_shape_finite_fill_invariant_and_full_rank():
    full, mask = _inputs(batch=1)
    model = _residual_graph_model()
    zero_fill = full * mask
    other_fill = torch.where(mask.bool(), full, torch.randn_like(full) * 100.0)

    first = model.forward_with_aux(zero_fill, mask)
    second = model.forward_with_aux(other_fill, mask)

    assert model.kernel_rank == model.hidden_channels
    assert first["raw_reconstruction"].shape == full.shape
    assert first["reconstruction"].shape == full.shape
    assert torch.isfinite(first["raw_reconstruction"]).all()
    torch.testing.assert_close(
        first["raw_reconstruction"],
        second["raw_reconstruction"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first["reconstruction"][mask.bool()],
        full[mask.bool()],
        rtol=0.0,
        atol=0.0,
    )


def test_residual_graph_gno_is_periodic_shift_equivariant():
    full, mask = _inputs(batch=1)
    # Stay below the half-domain antipodal tie where +0.5 and -0.5 are both
    # valid shortest displacements on a periodic domain.
    model = _residual_graph_model(
        radius=0.3, hard_observation_consistency=False
    ).eval()
    shift = (1, 2)

    with torch.no_grad():
        reference = model(full * mask, mask)
        shifted = model(
            torch.roll(full * mask, shifts=shift, dims=(-2, -1)),
            torch.roll(mask, shifts=shift, dims=(-2, -1)),
        )

    torch.testing.assert_close(
        shifted,
        torch.roll(reference, shifts=shift, dims=(-2, -1)),
        rtol=2.0e-5,
        atol=2.0e-5,
    )


def test_residual_graph_gno_uses_raw_output_for_observation_loss():
    full, mask = _inputs(batch=1)
    model = _residual_graph_model()
    trainer = MaskedReconstructionPretrainer(
        model,
        MaskedReconstructionLoss(
            hidden_weight=0.0,
            observation_weight=1.0,
            history_gradient_weight=0.0,
        ),
    )

    result = trainer.forward_batch(
        {"x_full": full, "x_obs": full * mask, "obs_mask": mask}
    )
    observation_loss = result["losses"]["observation_consistency"]
    observation_loss.backward()

    assert float(observation_loss) > 0.0
    torch.testing.assert_close(
        result["reconstruction"][mask.bool()],
        full[mask.bool()],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        result["raw_reconstruction"][mask.bool()],
        result["reconstruction"][mask.bool()],
    )
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.parameters()
    )


def test_residual_graph_gno_rejects_low_rank_kernel():
    with pytest.raises(ValueError, match="at least hidden_channels"):
        _residual_graph_model(kernel_rank=4)


def test_residual_graph_gno_supports_no_grid_refinement_ablation():
    full, mask = _inputs(batch=1)
    model = _residual_graph_model(grid_refinement_layers=0)

    result = model.forward_with_aux(full * mask, mask)

    assert len(model.grid_refinement) == 0
    assert result["raw_reconstruction"].shape == full.shape
    assert torch.isfinite(result["raw_reconstruction"]).all()


def test_residual_graph_v4_preserves_tokens_uses_fourier_kernel_and_cache():
    full, mask = _inputs(batch=1)
    model = _residual_graph_v4_model().eval()

    first_linear = next(
        layer for layer in model.sensor_to_grid.kernel if isinstance(layer, nn.Linear)
    )
    assert model.temporal_tokens == 2
    assert model.temporal_token_channels == 4
    assert first_linear.in_features == 2 + 4 * 3

    with torch.no_grad():
        first = model.forward_with_aux(full * mask, mask)
        first_cache = model.sensor_to_grid.geometry_cache_report()
        second = model.forward_with_aux(full * mask, mask)
        second_cache = model.sensor_to_grid.geometry_cache_report()

    assert first["raw_reconstruction"].shape == full.shape
    assert torch.isfinite(first["raw_reconstruction"]).all()
    assert first_cache["misses"] > 0
    assert second_cache["hits"] > first_cache["hits"]
    torch.testing.assert_close(
        first["raw_reconstruction"],
        second["raw_reconstruction"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first["reconstruction"][mask.bool()],
        full[mask.bool()],
        rtol=0.0,
        atol=0.0,
    )


def test_residual_graph_v4_cache_does_not_change_numerics():
    full, mask = _inputs(batch=1)
    cached = _residual_graph_v4_model(cache_geometry=True).eval()
    uncached = _residual_graph_v4_model(cache_geometry=False).eval()
    uncached.load_state_dict(cached.state_dict(), strict=True)

    with torch.no_grad():
        cached_output = cached(full * mask, mask)
        uncached_output = uncached(full * mask, mask)

    torch.testing.assert_close(cached_output, uncached_output, rtol=0.0, atol=0.0)


def test_residual_graph_v4_periodic_shift_and_backward_contract():
    full, mask = _inputs(batch=1)
    model = _residual_graph_v4_model(
        hard_observation_consistency=False, cache_geometry=False
    )
    shift = (1, 2)

    model.eval()
    with torch.no_grad():
        reference = model(full * mask, mask)
        shifted = model(
            torch.roll(full * mask, shifts=shift, dims=(-2, -1)),
            torch.roll(mask, shifts=shift, dims=(-2, -1)),
        )
    torch.testing.assert_close(
        shifted,
        torch.roll(reference, shifts=shift, dims=(-2, -1)),
        rtol=1.0e-3,
        atol=5.0e-5,
    )

    model.train()
    model(full * mask, mask).square().mean().backward()
    for prefix in ("history_token_encoder", "sensor_operator_blocks", "token_grid_fusion"):
        assert any(
            name.startswith(prefix)
            and parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and bool(torch.count_nonzero(parameter.grad))
            for name, parameter in model.named_parameters()
        )


def test_coordinate_reconstructor_mask_edge_cases():
    torch.manual_seed(83)
    model = PeriodicGNOReconstructor(
        input_steps=60,
        hidden_channels=4,
        radius=0.3,
        mlp_channels=(8,),
        kernel_mode="channelwise_nonlinear",
    )
    full = torch.randn(1, 60, 8, 8)
    masks = []
    all_valid = torch.ones_like(full)
    masks.append(all_valid)
    isolated = torch.zeros_like(full)
    isolated[:, :, 3, 5] = 1.0
    masks.append(isolated)
    irregular = torch.zeros_like(full)
    irregular[:, :, 1::3, ::2] = 1.0
    masks.append(irregular)

    for mask in masks:
        output = model(full * mask, mask)
        assert output.shape == full.shape
        assert torch.isfinite(output).all()

    with pytest.raises(ValueError, match="at least one observed sensor"):
        model(torch.zeros_like(full), torch.zeros_like(full))


@pytest.mark.parametrize("model_index", (0, 1))
@pytest.mark.parametrize("coupling", ("frozen", "joint"))
def test_coordinate_pipeline_freeze_and_joint_gradient_contract(coupling, model_index):
    full, mask = _inputs()
    pipeline = LearnedReconstructionRFNOPipeline(
        TinyRFNO(), _models()[model_index], coupling=coupling
    )
    output = pipeline(full * mask, mask)
    output["forecast"].square().mean().backward()

    reconstructor_gradients = [
        parameter.grad for parameter in pipeline.reconstructor.parameters()
    ]
    assert any(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in reconstructor_gradients
    )
    if coupling == "frozen":
        assert pipeline.rfno_gradients_are_none()
        assert all(not parameter.requires_grad for parameter in pipeline.rfno.parameters())
    else:
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and bool(torch.count_nonzero(parameter.grad))
            for parameter in pipeline.rfno.parameters()
        )


def test_coordinate_pipeline_300_frame_rollout_shape_and_finite():
    full, mask = _inputs(batch=1)
    pipeline = LearnedReconstructionRFNOPipeline(
        TinyRFNO(),
        PeriodicGNOReconstructor(
            input_steps=60,
            hidden_channels=4,
            radius=0.6,
            mlp_channels=(8,),
            kernel_mode="channelwise_nonlinear",
            kernel_rank=4,
            hard_observation_consistency=True,
        ),
        coupling="frozen",
    )

    output = pipeline.rollout(full * mask, mask, rollout_steps=300)

    assert output["history_reconstruction"].shape == full.shape
    assert output["forecast"].shape == (1, 300, 8, 8)
    assert torch.isfinite(output["history_reconstruction"]).all()
    assert torch.isfinite(output["forecast"]).all()
    assert pipeline.rfno_gradients_are_none()
