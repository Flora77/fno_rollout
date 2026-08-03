# Sea-surface research repository guidance

## Scope

Apply these rules to all work in this repository. Preserve unrelated user changes and
keep the established full-field RFNO experiment reproducible while adding sparse-
observation experiments.

## Scientific invariants

- Treat sea-surface tensors as `(B, T, H, W)` at public model and dataset boundaries.
- Use 60 historical frames, 30 frames per forecast chunk, and 300 frames for the
  established long rollout unless an experiment explicitly changes the horizon.
- Define observation masks as `1=observed` and `0=missing` everywhere.
- Treat zero-filling as a storage representation only; always pass an explicit mask.
- Never expose complete history fields or future targets to a sparse model input.
- Reuse the same data split, normalization statistics, mask manifest, noise realization,
  and random seeds across methods being compared.
- Report history reconstruction and future forecasting metrics separately.
- Keep the all-valid/full-field RFNO result as the upper-reference experiment.

## Implementation rules

- Preserve `neuralop/data/datasets/sea_surface_simple.py` legacy `{"x", "y"}` behavior.
  Add sparse behavior through a wrapper, adapter, or separate dataset module.
- Preserve `neuralop/models/fno_unet_aunet_gated_decoder.py` baseline behavior. Compose
  reconstructors and RFNO in a separate pipeline rather than editing the baseline in
  place when possible.
- Keep fixed-RFNO and joint-RFNO modes in one implementation controlled by configuration.
- In fixed mode, exclude RFNO parameters from the optimizer and verify that their
  gradients remain `None`.
- In joint mode, verify finite nonzero gradients reach trainable RFNO parameters.
- Use periodic/circular spatial padding for HOS-domain convolutional models unless the
  experiment explicitly models a nonperiodic observation window.
- Put active experiment code under `neuralop/`, `config/`, and `scripts/`; do not add new
  work to `scripts_old/` or `config_old/`.

## Validation

- Run shape and finite-value tests for every new model path.
- Test all-valid, all-missing, isolated-point, and irregular masks for masked operators.
- Verify that changing values at masked positions does not affect PartialConv outputs.
- Run a one-batch forward/backward smoke test before expensive training.
- Prefer a two-sample overfit test for a new learned reconstructor.
- Do not launch full GPU training unless the user explicitly requests it.
- Report commands run, results, skipped checks, and remaining risks.

## Codex workflow

Use `$sea-surface-sparse-experiments` for sparse-observation baselines, mask generation,
PartialConv-MAE-RFNO, GNO/GINO reconstruction, frozen/joint RFNO pipelines, direct sparse
forecasting, ablations, and result comparison.
