import torch
import torch.nn.functional as F

from neuralop.layers.partial_convolution import PeriodicPartialConv2d


def test_all_valid_mask_matches_regular_circular_convolution():
    torch.manual_seed(1)
    layer = PeriodicPartialConv2d(2, 3, kernel_size=3, bias=True)
    value = torch.randn(2, 2, 8, 9)
    mask = torch.ones(2, 1, 8, 9)

    output, next_mask = layer(value, mask)
    reference = F.conv2d(
        F.pad(value, (1, 1, 1, 1), mode="circular"),
        layer.weight,
        layer.bias,
    )

    torch.testing.assert_close(output, reference)
    assert torch.equal(next_mask, torch.ones_like(next_mask))


def test_all_missing_mask_returns_zero_output_and_mask():
    torch.manual_seed(2)
    layer = PeriodicPartialConv2d(2, 4, kernel_size=3, bias=True)
    value = torch.randn(2, 2, 8, 8)
    mask = torch.zeros(2, 1, 8, 8)

    output, next_mask = layer(value, mask)

    assert torch.equal(output, torch.zeros_like(output))
    assert torch.equal(next_mask, torch.zeros_like(next_mask))


def test_isolated_observation_updates_periodic_neighborhood():
    layer = PeriodicPartialConv2d(1, 1, kernel_size=3, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    value = torch.zeros(1, 1, 8, 8)
    mask = torch.zeros_like(value)
    value[0, 0, 0, 0] = 2.0
    mask[0, 0, 0, 0] = 1.0

    output, next_mask = layer(value, mask)

    assert next_mask.sum().item() == 9
    assert next_mask[0, 0, 7, 7] == 1
    assert next_mask[0, 0, 0, 7] == 1
    assert next_mask[0, 0, 7, 0] == 1
    torch.testing.assert_close(
        output[next_mask.bool()],
        torch.full_like(output[next_mask.bool()], 18.0),
    )
    assert torch.equal(output[next_mask == 0], torch.zeros_like(output[next_mask == 0]))


def test_random_irregular_holes_produce_binary_finite_outputs():
    torch.manual_seed(3)
    layer = PeriodicPartialConv2d(3, 5, kernel_size=3, bias=True)
    value = torch.randn(2, 3, 11, 13)
    mask = (torch.rand(2, 3, 11, 13) > 0.4).float()

    output, next_mask = layer(value, mask)

    assert output.shape == (2, 5, 11, 13)
    assert next_mask.shape == (2, 1, 11, 13)
    assert torch.isfinite(output).all()
    assert set(next_mask.unique().tolist()).issubset({0.0, 1.0})


def test_missing_fill_values_cannot_change_output():
    torch.manual_seed(4)
    layer = PeriodicPartialConv2d(2, 3, kernel_size=3, bias=True)
    observed_values = torch.randn(2, 2, 9, 10)
    mask = (torch.rand(2, 1, 9, 10) > 0.5).float()
    expanded_mask = mask.expand_as(observed_values).bool()

    zero_fill = torch.where(expanded_mask, observed_values, 0.0)
    hundred_fill = torch.where(expanded_mask, observed_values, 100.0)
    random_fill = torch.where(
        expanded_mask, observed_values, torch.randn_like(observed_values) * 50.0
    )

    zero_output, zero_mask = layer(zero_fill, mask)
    hundred_output, hundred_mask = layer(hundred_fill, mask)
    random_output, random_mask = layer(random_fill, mask)

    torch.testing.assert_close(zero_output, hundred_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(zero_output, random_output, rtol=0.0, atol=0.0)
    assert torch.equal(zero_mask, hundred_mask)
    assert torch.equal(zero_mask, random_mask)


def test_backward_gradients_are_finite_and_missing_value_gradients_are_zero():
    torch.manual_seed(5)
    layer = PeriodicPartialConv2d(2, 3, kernel_size=3, bias=True)
    value = torch.randn(2, 2, 8, 8, requires_grad=True)
    mask = (torch.rand(2, 1, 8, 8) > 0.35).float()

    output, _ = layer(value, mask)
    loss = output.square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert layer.bias is not None
    assert layer.bias.grad is not None and torch.isfinite(layer.bias.grad).all()
    missing = ~mask.expand_as(value).bool()
    assert torch.equal(value.grad[missing], torch.zeros_like(value.grad[missing]))
