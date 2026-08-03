import torch
from torch import nn

from neuralop.models.reconstructors.partialconv_mae import (
    PartialConvEncoder,
    PartialConvMaskedAutoencoder,
)
from neuralop.models.sparse_forecast_pipeline import (
    PartialConvMAEFrozenRFNOPipeline,
)
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
    return PartialConvMaskedAutoencoder(
        input_steps=60,
        encoder_channels=(8,),
        decoder_channels=(8,),
    )


def _smooth_two_sample_batch(size=8):
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


def test_partialconv_encoder_propagates_mask_and_downsamples():
    encoder = PartialConvEncoder(in_channels=60, channels=(8, 12))
    value = torch.randn(2, 60, 16, 16)
    mask = (torch.rand_like(value) > 0.7).float()

    latent, latent_mask, sizes = encoder(value * mask, mask)

    assert latent.shape == (2, 12, 4, 4)
    assert latent_mask.shape == (2, 1, 4, 4)
    assert sizes == ((16, 16), (8, 8), (4, 4))
    assert torch.isfinite(latent).all()


def test_mae_reconstructs_60_frames_and_ignores_missing_fill_values():
    torch.manual_seed(11)
    model = _small_reconstructor().eval()
    full = torch.randn(2, 60, 8, 8)
    mask = (torch.rand_like(full) > 0.5).float()
    zero_fill = full * mask
    other_fill = torch.where(mask.bool(), full, torch.randn_like(full) * 100.0)

    with torch.no_grad():
        zero_reconstruction = model(zero_fill, mask)
        other_reconstruction = model(other_fill, mask)

    assert zero_reconstruction.shape == (2, 60, 8, 8)
    assert torch.isfinite(zero_reconstruction).all()
    torch.testing.assert_close(
        zero_reconstruction, other_reconstruction, rtol=0.0, atol=0.0
    )


def test_pretraining_one_batch_train_and_validation_losses_are_finite():
    torch.manual_seed(12)
    model = _small_reconstructor()
    criterion = MaskedReconstructionLoss(
        hidden_weight=1.0,
        observation_weight=0.2,
        history_gradient_weight=0.05,
    )
    pretrainer = MaskedReconstructionPretrainer(model, criterion)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
    batch = _smooth_two_sample_batch()

    train_result = pretrainer.train_batch(batch, optimizer)
    validation_result = pretrainer.validate_batch(batch)

    assert train_result["reconstruction"].shape == (2, 60, 8, 8)
    assert validation_result["reconstruction"].shape == (2, 60, 8, 8)
    for result in (train_result, validation_result):
        assert set(result["losses"]) == {
            "hidden_reconstruction",
            "observation_consistency",
            "history_gradient",
            "weighted_hidden_reconstruction",
            "weighted_observation_consistency",
            "weighted_history_gradient",
            "total",
        }
        assert all(torch.isfinite(loss) for loss in result["losses"].values())


def test_frozen_pipeline_separates_outputs_and_excludes_rfno_from_optimizer():
    torch.manual_seed(13)
    reconstructor = _small_reconstructor()
    pipeline = PartialConvMAEFrozenRFNOPipeline(TinyRFNO(), reconstructor)
    pipeline.train()
    batch = _smooth_two_sample_batch()
    optimizer_parameters = tuple(pipeline.optimizer_parameters())
    rfno_parameter_ids = {id(parameter) for parameter in pipeline.rfno.parameters()}

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
    reconstructor_grads = [
        parameter.grad
        for parameter in reconstructor.parameters()
        if parameter.requires_grad
    ]
    assert any(gradient is not None for gradient in reconstructor_grads)
    assert all(
        gradient is None or torch.isfinite(gradient).all()
        for gradient in reconstructor_grads
    )


def test_two_samples_overfit_masked_reconstruction():
    torch.manual_seed(14)
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
    batch = _smooth_two_sample_batch()

    initial = float(pretrainer.validate_batch(batch)["losses"]["total"])
    for _ in range(80):
        pretrainer.train_batch(batch, optimizer)
    final = float(pretrainer.validate_batch(batch)["losses"]["total"])

    assert final < initial * 0.35, (initial, final)


def test_p1_end_to_end_all_observed_and_all_missing_are_finite():
    torch.manual_seed(15)
    x_full = torch.randn(1, 60, 8, 8)
    for mask_value in (0.0, 1.0):
        mask = torch.full_like(x_full, mask_value)
        reconstructor = _small_reconstructor()
        pipeline = PartialConvMAEFrozenRFNOPipeline(TinyRFNO(), reconstructor)
        outputs = pipeline(x_full * mask, mask)
        losses = MaskedReconstructionLoss(
            hidden_weight=1.0,
            observation_weight=0.1,
            history_gradient_weight=0.05,
        )(outputs["history_reconstruction"], x_full, mask)

        assert outputs["history_reconstruction"].shape == x_full.shape
        assert outputs["forecast"].shape == (1, 30, 8, 8)
        assert torch.isfinite(outputs["history_reconstruction"]).all()
        assert torch.isfinite(outputs["forecast"]).all()
        assert all(torch.isfinite(value) for value in losses.values())
        assert pipeline.rfno_gradients_are_none()


def test_p1_frozen_training_does_not_require_or_use_future_targets():
    torch.manual_seed(16)
    pipeline = PartialConvMAEFrozenRFNOPipeline(TinyRFNO(), _small_reconstructor())
    config = SparseCoupledTrainingConfig(coupling="frozen")
    trainer = SparseCoupledForecastTrainer(pipeline, config)
    batch = _smooth_two_sample_batch()
    without_future = {key: value for key, value in batch.items() if key != "y"}
    first = trainer.validate_batch(without_future, active_rollout_steps=30)
    changed_future = {**without_future, "y": torch.randn_like(batch["y"]) * 1.0e9}
    second = trainer.validate_batch(changed_future, active_rollout_steps=30)

    assert config.forecast_supervision is False
    torch.testing.assert_close(first["losses"]["total"], second["losses"]["total"])
    assert first["losses"]["rollout"].item() == 0.0
    assert first["losses"]["forecast_gradient"].item() == 0.0
    assert first["losses"]["spectrum"].item() == 0.0
