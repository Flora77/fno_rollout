import torch
from torch import nn

from neuralop.models.reconstructors import (
    PartialConvMaskedAutoencoder,
    PeriodicConv2d,
    PeriodicMaskUNet,
)
from neuralop.models.sparse_forecast_pipeline import MaskUNetFrozenRFNOPipeline
from neuralop.training.masked_reconstruction import (
    MaskedReconstructionLoss,
    MaskedReconstructionPretrainer,
)
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


def _small_reconstructor():
    return PeriodicMaskUNet(input_steps=60, channels=(4, 8))


def _two_sample_batch(size=8):
    coordinates = torch.linspace(0.0, 2.0 * torch.pi, size + 1)[:-1]
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    time = torch.linspace(0.0, 1.0, 60).view(1, 60, 1, 1)
    phases = torch.tensor([0.0, 0.7]).view(2, 1, 1, 1)
    x_full = torch.sin(xx + phases + time) + 0.5 * torch.cos(yy - time)
    mask = torch.zeros_like(x_full)
    mask[:, :, ::2, ::2] = 1.0
    mask[:, :, 1::2, 1::2] = 1.0
    return {
        "x_full": x_full,
        "x_obs": x_full * mask,
        "obs_mask": mask,
        "y": torch.full((2, 30, size, size), 1.0e6),
    }


def test_periodic_convolution_wraps_across_spatial_boundary():
    layer = PeriodicConv2d(1, 1, kernel_size=3, bias=False)
    with torch.no_grad():
        layer.conv.weight.zero_()
        layer.conv.weight[0, 0, 1, 0] = 1.0
    value = torch.zeros(1, 1, 5, 5)
    value[0, 0, 2, -1] = 3.0

    output = layer(value)

    assert output[0, 0, 2, 0].item() == 3.0


def test_mask_unet_shape_finite_and_missing_fill_invariance():
    torch.manual_seed(31)
    model = _small_reconstructor().eval()
    full = torch.randn(2, 60, 8, 8)
    mask = (torch.rand_like(full) > 0.7).float()
    zero_fill = full * mask
    arbitrary_fill = torch.where(mask.bool(), full, torch.randn_like(full) * 100.0)

    with torch.no_grad():
        first = model(zero_fill, mask)
        second = model(arbitrary_fill, mask)

    assert first.shape == full.shape
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_mask_unet_all_observed_and_all_missing_are_finite():
    torch.manual_seed(32)
    x_full = torch.randn(1, 60, 8, 8)
    for mask_value in (0.0, 1.0):
        mask = torch.full_like(x_full, mask_value)
        output = _small_reconstructor()(x_full * mask, mask)
        assert output.shape == x_full.shape
        assert torch.isfinite(output).all()


def test_default_mask_unet_parameter_count_matches_p1_scale():
    b4_parameters = sum(parameter.numel() for parameter in PeriodicMaskUNet().parameters())
    p1_parameters = sum(
        parameter.numel()
        for parameter in PartialConvMaskedAutoencoder().parameters()
    )

    assert 0.8 <= b4_parameters / p1_parameters <= 1.25


def test_b4_pipeline_outputs_and_strictly_freezes_rfno():
    torch.manual_seed(33)
    pipeline = MaskUNetFrozenRFNOPipeline(
        TinyRFNO(), _small_reconstructor(), coupling="frozen"
    )
    batch = _two_sample_batch()
    optimizer_parameters = tuple(pipeline.optimizer_parameters())
    rfno_parameter_ids = {id(parameter) for parameter in pipeline.rfno.parameters()}

    pipeline.train()
    outputs = pipeline(batch["x_obs"], batch["obs_mask"])
    outputs["forecast"].square().mean().backward()

    assert outputs["history_reconstruction"].shape == (2, 60, 8, 8)
    assert outputs["forecast"].shape == (2, 30, 8, 8)
    assert torch.isfinite(outputs["history_reconstruction"]).all()
    assert torch.isfinite(outputs["forecast"]).all()
    assert optimizer_parameters
    assert not any(id(parameter) in rfno_parameter_ids for parameter in optimizer_parameters)
    assert all(not parameter.requires_grad for parameter in pipeline.rfno.parameters())
    assert not pipeline.rfno.training
    assert pipeline.rfno_gradients_are_none()


def test_b4_reconstruction_training_does_not_require_or_use_future_target():
    torch.manual_seed(34)
    pipeline = MaskUNetFrozenRFNOPipeline(
        TinyRFNO(), _small_reconstructor(), coupling="frozen"
    )
    config = SparseCoupledTrainingConfig(coupling="frozen")
    trainer = SparseCoupledForecastTrainer(pipeline, config)
    batch = _two_sample_batch()
    without_future = {key: value for key, value in batch.items() if key != "y"}

    first = trainer.validate_batch(without_future, active_rollout_steps=30)
    changed = {**without_future, "y": torch.randn_like(batch["y"]) * 1.0e9}
    second = trainer.validate_batch(changed, active_rollout_steps=30)

    torch.testing.assert_close(first["losses"]["total"], second["losses"]["total"])
    assert first["losses"]["rollout"].item() == 0.0


def test_mask_unet_two_samples_overfit():
    torch.manual_seed(35)
    model = _small_reconstructor()
    pretrainer = MaskedReconstructionPretrainer(
        model,
        MaskedReconstructionLoss(
            hidden_weight=1.0,
            observation_weight=0.1,
            history_gradient_weight=0.0,
        ),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    batch = _two_sample_batch()

    initial = float(pretrainer.validate_batch(batch)["losses"]["total"])
    for _ in range(80):
        pretrainer.train_batch(batch, optimizer, max_grad_norm=1.0)
    final = float(pretrainer.validate_batch(batch)["losses"]["total"])

    assert final < initial * 0.35, (initial, final)
