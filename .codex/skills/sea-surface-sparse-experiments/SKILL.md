---
name: sea-surface-sparse-experiments
description: Implement, modify, debug, test, and compare sparse-observation sea-surface reconstruction and long-horizon forecasting experiments in the neuraloperator repository. Use for full-field RFNO, bilinear/Kriging/POD plus RFNO, Mask U-Net, GNO/GINO, PartialConv-MAE-RFNO, frozen or jointly trained RFNO pipelines, direct sparse-to-future models, mask generation, observation-noise experiments, ablations, and validation of experiment results.
---

# Sparse Sea-Surface Experiments

Build one scientifically controlled experiment at a time while preserving the existing
full-field RFNO baseline.

## Load only the needed context

1. Read `references/repo-map.md` for every task.
2. Read `references/experiment-matrix.md` when selecting or comparing a method.
3. Read `references/model-contracts.md` before changing datasets, models, or trainers.
4. Read `references/evaluation-protocol.md` before training or evaluating.
5. Read `references/implementation-roadmap.md` when planning multiple stages.
6. Read `references/prompt-examples.md` when the user asks how to drive the workflow.
7. Inspect current repository files before assuming a documented path or API is current.

## Classify the requested experiment

Choose exactly one primary category and state it before editing:

- `full_field`: complete history enters the established RFNO.
- `deterministic_reconstruction`: bilinear, Kriging, or POD reconstructs history.
- `learned_reconstruction_frozen`: a learned reconstructor feeds a frozen RFNO.
- `learned_reconstruction_joint`: reconstructor and RFNO train together.
- `direct_sparse_forecast`: sparse observations map directly to future fields.
- `pretraining`: masked reconstruction or mask-to-predict representation learning.
- `evaluation_only`: compare checkpoints or saved predictions without model changes.

Identify the experiment ID from `references/experiment-matrix.md`. If the request mixes
multiple categories, implement the smallest prerequisite vertical slice first.

## Preserve comparability

- Keep public tensors in `(B,T,H,W)` unless a documented coordinate boundary requires a
  different representation.
- Define masks as `1=observed`, `0=missing`.
- Keep complete fields out of sparse inputs and future targets out of all inputs.
- Reuse identical split, mask manifest, observation noise, seed, normalization, rollout,
  and metric settings across comparable methods.
- Do not change the legacy dense dataset or RFNO numerical behavior in place when an
  adapter or pipeline can preserve it.
- Separate reconstruction metrics from future forecast metrics.
- Give every run a unique experiment name containing method, coupling, mask, rate, and seed.

## Use a composable pipeline

Implement the conceptual path:

```text
observation operator -> optional reconstructor -> optional RFNO -> outputs
```

Prefer a sparse dataset wrapper that retains the dense sample and exposes `x_full`,
`x_obs`, `obs_mask`, optional coordinate observations, `y`, and `mask_id`. Follow the
exact contracts in `references/model-contracts.md`.

Support coupling without duplicating trainers:

- `frozen`: exclude RFNO parameters from the optimizer and keep their gradients `None`.
- `joint`: train reconstructor and RFNO and verify finite gradients reach both.
- `direct`: bypass explicit history reconstruction.

## Implement PartialConv-MAE-RFNO

For PartialConv:

- Convolve valid entries only and renormalize by local valid count.
- Update and propagate masks at every PartialConv layer.
- Use periodic spatial padding for the HOS periodic domain.
- Test all-valid, all-missing, isolated-point, and irregular-hole masks.
- Verify masked fill values cannot influence the result.

For masked pretraining:

1. Generate realistic masks from a shared manifest.
2. Reconstruct the 60-frame history.
3. Weight reconstruction loss primarily on deliberately hidden locations.
4. Enforce consistency at observed locations.
5. Pretrain the reconstructor before attaching RFNO.
6. Add forecast-aware fine-tuning only after reconstruction smoke tests pass.
7. Do not add GAN loss unless the user explicitly requests an ablation.

## Work incrementally

1. Inspect the relevant code and state the minimal change surface.
2. Define input/output shapes, mask semantics, freeze policy, and acceptance tests.
3. Implement one vertical slice with minimal unrelated refactoring.
4. Run shape, finite-value, mask-invariance, and gradient checks.
5. Run one-batch forward/backward testing.
6. Run a two-sample overfit test for a new learned reconstructor when practical.
7. Expand to curriculum rollout only after the short path is stable.
8. Do not start expensive full training without explicit authorization.

Use bundled scripts when applicable:

- `scripts/generate_mask_manifest.py`: create deterministic shared masks.
- `scripts/validate_sparse_config.py`: validate a JSON experiment definition.
- `scripts/summarize_sparse_results.py`: aggregate repeated-seed CSV results.

## Validate completion

Check as applicable:

- One-chunk output is `(B,30,H,W)`.
- Long rollout output is `(B,300,H,W)`.
- All outputs and losses are finite.
- Dense all-valid behavior matches its reference path.
- Fixed RFNO parameters receive no gradients.
- Joint RFNO parameters receive finite gradients.
- Saved results identify method, coupling, mask type, observation rate, noise, seed,
  parameter count, checkpoint, and inference time.

## Report the result

Report the experiment category and ID, files changed, tensor contracts, commands and tests
run, tests skipped, and remaining scientific or implementation risks.
