# Repository map

## Active baseline

- Dense dataset: `neuralop/data/datasets/sea_surface_simple.py`
- RFNO model: `neuralop/models/fno_unet_aunet_gated_decoder.py`
- GINO implementation: `neuralop/models/gino.py`
- RFNO configuration: `config/sea_surface_rollout_config_fno_unet_aunet_gated.py`
- RFNO training entry: `scripts/fno_unet_log/train_sea_surface_rollout_fno_unet_metrics.py`
- RFNO validation entries:
  - `scripts/fno_unet_log/validate_sea_surface_rollout_fno_unet_metrics.py`
  - `scripts/fno_unet_log/validate_sea_surface_rollout_fno_unet_metrics_per_mat.py`

Treat `scripts_old/`, `config_old/`, and `checkpoints_old/` as historical references, not
destinations for new work.

## Current baseline contracts

- Dense sample: `{"x": (T_in,H,W), "y": (T_out,H,W)}`
- Batch history: `(B,60,H,W)`
- One predicted chunk: `(B,30,H,W)`
- Long rollout: `(B,300,H,W)`
- Spatial resolution: normally `64 x 64`
- Time interval: `0.25 s`
- Historical duration: `15 s`
- One-chunk duration: `7.5 s`
- Long forecast duration: `75 s`
- RFNO spatial padding: periodic/circular

## Preferred additions

- Sparse data wrapper: `neuralop/data/datasets/sparse_sea_surface.py`
- Reconstruction modules: `neuralop/models/reconstructors/`
- Composed model: `neuralop/models/sparse_forecast_pipeline.py`
- Sparse configuration: `config/sea_surface_sparse_experiment.py`
- Active scripts: `scripts/sparse_surface/`
- Tests: `tests/sparse_surface/`

Keep reconstruction, forecasting, loss calculation, and experiment logging separate enough
that each component can be replaced without copying the trainer.
