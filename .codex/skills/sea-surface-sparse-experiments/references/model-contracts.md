# Model and data contracts

## Mask semantics

- Use Boolean or `{0,1}` tensors.
- Define `1` as observed/valid and `0` as missing/invalid.
- Broadcast masks only through documented dimensions.
- Never infer validity from whether a value equals zero.

## Sparse dataset sample

Prefer a wrapper over `SeaSurfaceSimpleDataset` and return:

```python
{
    "x_full": Tensor[T_in, H, W],
    "x_obs": Tensor[T_in, H, W],
    "obs_mask": Tensor[T_in, H, W],
    "sensor_coords": Optional[Tensor[N, 2]],
    "sensor_values": Optional[Tensor[N, T_in]],
    "sensor_mask": Optional[Tensor[N]],
    "y": Tensor[T_out, H, W],
    "mask_id": str,
}
```

Keep `x_full` for supervised losses and evaluation only. Pass a filtered input dictionary to
the model so `x_full` cannot be consumed accidentally.

## Reconstruction interface

For fixed-grid methods:

```python
x_recon = reconstructor(x_obs, obs_mask)
# x_obs, obs_mask, x_recon: (B, 60, H, W)
```

For coordinate-aware methods:

```python
x_recon = reconstructor(sensor_coords, sensor_values, sensor_mask, output_grid)
# sensor_coords: (B, N, 2)
# sensor_values: (B, N, 60)
# x_recon: (B, 60, H, W)
```

If variable sensor counts are padded, `sensor_mask` must exclude padded entries from every
aggregation. Record coordinate normalization and neighbor-radius conventions.

## Forecast interface

```python
y_chunk = forecaster(x_history)
# x_history: (B, 60, H, W)
# y_chunk:   (B, 30, H, W)
```

Autoregressive rollout appends predictions and retains the most recent 60 frames without
using ground-truth future frames.

## Pipeline interface

Prefer structured outputs:

```python
{
    "history_reconstruction": x_recon,
    "forecast": y_pred,
    "observation_projection": observed_prediction,
    "aux": {...},
}
```

Allow `history_reconstruction=None` for direct sparse-to-future models.

## Frozen and joint modes

Frozen mode must:

- set RFNO parameters to `requires_grad_(False)`;
- exclude RFNO parameters from optimizer groups;
- decide explicitly whether RFNO runs in `eval()` or `train()` mode;
- keep RFNO gradients `None` after backward.

Joint mode must:

- set selected RFNO parameters trainable;
- include them in optimizer groups, optionally at a lower learning rate;
- verify finite gradients reach both modules;
- record whether RFNO was initialized from B0.

## Loss decomposition

Use named terms rather than an opaque total:

```text
L_total =
    lambda_hidden * L_hidden_reconstruction
  + lambda_obs    * L_observation_consistency
  + lambda_roll   * L_rollout
  + lambda_grad   * L_spatial_gradient
  + lambda_spec   * L_spectrum
  + lambda_phys   * L_physics_optional
```

Log raw and weighted values for every active term. Do not add adversarial or physics losses
until the reconstruction and forecast-only paths are validated.

## PartialConv rules

For input window `X`, binary mask `M`, kernel size `K`, and learnable weights `W`, use the
valid entries and renormalize by local support. Return zero when local support is empty and
update the output mask to valid when any local input is valid.

Use circular padding for both values and masks on periodic HOS grids. Test that arbitrary
replacement values under `M=0` cannot affect the output.
