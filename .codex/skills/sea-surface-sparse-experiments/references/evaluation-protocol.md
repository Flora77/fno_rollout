# Evaluation protocol

## Shared data controls

- Freeze train/validation/test file lists before comparing methods.
- Derive normalization statistics from training data only.
- Generate mask manifests independently of model training.
- Reuse the same manifest IDs and noise realizations across all compared methods.
- Record code revision, configuration, model checkpoint, and manifest checksum.

## Observation scenarios

Start with these scenarios and reduce only when compute is constrained:

| Family | Suggested settings | Purpose |
|---|---|---|
| Fixed points | 1%, 2%, 5%, 10%, 20% | Fixed buoy arrays |
| Random points | 1%, 2%, 5%, 10%, 20% | Position robustness |
| Blocks/patches | 25%, 50%, 75% observed | Radar or image gaps |
| Stripes | 25%, 50%, 75% observed | Scan-line occlusion |
| Noise | 0%, 1%, 2%, 5% of training std | Measurement noise |
| Temporal dropout | 0%, 10%, 25% | Sensor/communication failure |

Use at least three seeds for final tables. Use one seed during implementation smoke tests.

## Reconstruction metrics

Report on the complete field and separately on missing locations:

- RMSE and NRMSE;
- correlation coefficient;
- spatial-gradient error;
- spectral-shape or spectral-energy error;
- crest/trough amplitude bias when relevant;
- observation-consistency error at measured locations.

## Forecast metrics

Report the established metrics at consistent horizons:

- one chunk: 30 frames / 7.5 s;
- 60, 120, 180, 240, and 300 frames;
- NRMSE, SSP, correlation, spectral errors, and slope/gradient errors;
- error growth curves rather than only final-horizon error;
- trustworthy horizon under a predefined threshold.

Define the trustworthy-horizon threshold before comparing methods. Do not tune the threshold
after seeing test results.

## Efficiency and reproducibility

Record:

- total and trainable parameter counts;
- peak training memory;
- reconstruction and forecast inference time;
- checkpoint size;
- seed and number of repeated runs;
- mean and sample standard deviation across seeds.

## Mandatory ablations for PartialConv-MAE-RFNO

1. Ordinary masked U-Net versus PartialConv network.
2. Random initialization versus masked pretraining.
3. P1 frozen RFNO versus P2 joint RFNO.
4. With versus without observation-consistency loss.
5. With versus without spectrum/gradient loss.

Keep all other factors fixed within each ablation.
