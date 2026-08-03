# Implementation roadmap

## Stage 0: freeze the reference

- Record B0 configuration, checkpoint, metrics, data split, and normalization values.
- Add no new modeling code.
- Confirm one-chunk and 300-frame evaluation still run.

## Stage 1: common observation layer

- Add a sparse wrapper around the dense dataset.
- Load masks from a deterministic manifest.
- Return both fixed-grid and optional coordinate representations.
- Test mask semantics, reproducibility, and absence of target leakage.

Exit criterion: B1 can run through the common data and evaluation pipeline.

## Stage 2: deterministic baselines

- Implement bilinear interpolation first.
- Add Kriging and POD only after B1 metrics and output saving are stable.
- Keep RFNO frozen.

Exit criterion: reconstruction and forecast metrics are logged separately for B1.

## Stage 3: learned reconstructors

- Implement a common reconstructor interface.
- Add B4 Mask U-Net.
- Add P1 PartialConv-MAE with reconstruction pretraining.
- Keep RFNO frozen while validating reconstruction.

Exit criterion: learned reconstructors overfit a tiny sample and pass mask invariance tests.

## Stage 4: forecast-aware training

- Attach the established RFNO through a pipeline.
- Compare frozen P1 with joint P2.
- Start with 30-frame loss, then reuse the established rollout curriculum to 300 frames.

Exit criterion: gradients, checkpoints, metrics, and rollout are stable in both modes.

## Stage 5: irregular sensors

- Add GNO/GINO lifting from point histories to the latent/full grid.
- Support padded variable sensor counts with an explicit sensor mask.
- Compare B5 and B6 under the same observation coordinates.

## Stage 6: scientific extensions

- Test B7 direct sparse-to-future prediction.
- Add light observation, spectrum, dispersion, or physics constraints one at a time.
- Add P3 online analysis correction only if the preceding stages are complete.

## Recommended Codex task size

Give Codex one exit criterion per task. Do not combine dataset construction, PartialConv,
MAE pretraining, RFNO joint rollout, and final experiments in one prompt.
