import torch
from torch import nn

from neuralop.layers.partial_convolution import PeriodicPartialConv2d
from neuralop.models.reconstructors import PeriodicHybridPartialConvUNet
from neuralop.models.sparse_forecast_pipeline import LearnedReconstructionRFNOPipeline


class TinyRFNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(60, 30, 1)

    def forward(self, history):
        return self.projection(history)


def _inputs(batch=2, size=8):
    torch.manual_seed(71)
    full = torch.randn(batch, 60, size, size)
    mask = torch.zeros_like(full)
    mask[:, :, ::2, ::2] = 1.0
    mask[:, :, 1::2, 1::2] = 1.0
    return full, mask


def test_first_layer_hybrid_shape_finite_and_fill_invariant():
    full, mask = _inputs()
    model = PeriodicHybridPartialConvUNet(
        input_steps=60, channels=(4, 8, 12), partialconv_levels=1
    ).eval()
    zero_fill = full * mask
    arbitrary_fill = torch.where(mask.bool(), full, torch.randn_like(full) * 100.0)

    with torch.no_grad():
        first = model(zero_fill, mask)
        second = model(arbitrary_fill, mask)

    assert first.shape == full.shape
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert sum(isinstance(m, PeriodicPartialConv2d) for m in model.modules()) == 1


def test_all_partialconv_and_no_skip_ablation_paths():
    full, mask = _inputs(batch=1)
    for levels, use_skips in ((3, True), (1, False)):
        model = PeriodicHybridPartialConvUNet(
            input_steps=60,
            channels=(4, 8, 12),
            partialconv_levels=levels,
            use_skip_connections=use_skips,
        )
        output = model(full * mask, mask)
        assert output.shape == full.shape
        assert torch.isfinite(output).all()
        assert sum(
            isinstance(module, PeriodicPartialConv2d) for module in model.modules()
        ) == levels


def test_hybrid_frozen_pipeline_keeps_rfno_grad_none():
    full, mask = _inputs()
    pipeline = LearnedReconstructionRFNOPipeline(
        TinyRFNO(),
        PeriodicHybridPartialConvUNet(input_steps=60, channels=(4, 8)),
        coupling="frozen",
    )
    output = pipeline(full * mask, mask)
    output["forecast"].square().mean().backward()
    optimizer_parameter_ids = {
        id(parameter)
        for group in pipeline.optimizer_param_groups(
            reconstructor_lr=2.0e-4,
            rfno_lr=2.0e-5,
        )
        for parameter in group["params"]
    }

    assert output["history_reconstruction"].shape == full.shape
    assert output["forecast"].shape == (2, 30, 8, 8)
    assert pipeline.rfno_gradients_are_none()
    assert all(not parameter.requires_grad for parameter in pipeline.rfno.parameters())
    assert not optimizer_parameter_ids.intersection(
        id(parameter) for parameter in pipeline.rfno.parameters()
    )
