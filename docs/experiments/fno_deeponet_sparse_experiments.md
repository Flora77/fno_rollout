# FNO-DeepONet sparse sea-surface experiments

## Paper-to-repository mapping

The implementation follows the coordinate-aware FNO-DeepONet factorization from
Zheng et al.:

- the branch consumes masked wave histories, sensor coordinates, and an explicit
  `1=observed` validity channel;
- temporal FNO layers mix all grid slots and the complete 60-frame history before
  producing one branch latent;
- the trunk Fourier-embeds query time and spatial coordinates;
- a branch/trunk inner product decodes the requested full-field values.

The paper's 100-frame gauge history and 34-step target-gauge forecast are adapted
to the repository's controlled `(B,60,64,64)` history, 30-frame chunk, and
300-frame rollout contracts. Spatial trunk frequencies use integer periodic
harmonics because the HOS field is periodic. The baseline dataset and
`FNOGlobalUNetGatedDecoder` are not modified.

## Experiment IDs

| ID | Category | Reconstruction | Forecast |
|---|---|---|---|
| `FD-R1` | learned reconstruction, frozen | FNO-DeepONet decodes 60 history frames | established frozen B0 RFNO, autoregressive evaluation to 300 |
| `FD-A1` | learned reconstruction, joint | FNO-DeepONet decodes 60 history frames | FNO-DeepONet replaces the RFNO FNO+residual module; 60-to-30 chunks use the existing 30/60/120/180/240/300 curriculum |
| `FD-L1` | direct sparse forecast | the shared branch decodes the 60-frame history for a separate reconstruction score | the same branch directly decodes all 300 future frames without autoregressive context |

All three seed-42 configurations reuse:

- `data/splits/sea_surface_bimodal_v1.json`;
- `data/masks/point_masks.npz`, `mask_00006`, fixed-points rate 0.05;
- the B0 checkpoint's train-only normalization statistics;
- zero observation noise and zero temporal dropout;
- reconstruction and forecast metrics from the shared sparse evaluator.

## Commands

Run a no-training shape and finite-value check:

```powershell
python scripts/sparse_surface/run_sparse_experiment.py `
  config/sparse_experiments/fd_r1_fno_deeponet_frozen_rfno_seed42.formal.json `
  --project-root . --device cuda --dry-run --rollout-steps 30
```

Use the corresponding `fd_a1_...json` or `fd_l1_...json` file for the other
paths. For a bounded one-batch forward/backward check:

```powershell
python scripts/sparse_surface/train_sparse_experiment.py `
  config/sparse_experiments/fd_a1_fno_deeponet_autoregressive_seed42.formal.json `
  --project-root . --device cuda --smoke `
  --max-epochs 1 --max-train-batches 1 --max-val-batches 1
```

Omit `--smoke` and the limits only when a full training run is explicitly
intended. Evaluate a trained `best.pt` with
`scripts/sparse_surface/run_sparse_experiment.py --evaluate-split val`; the
shared result contains separate `history_reconstruction` metrics and
`forecast_30/60/120/180/240/300` metrics.
