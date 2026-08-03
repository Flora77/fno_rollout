# Experiment matrix

## Controlled comparison

| ID | Method | Sparse input | Explicit reconstruction | Learned reconstruction | RFNO policy |
|---|---|---:|---:|---:|---|
| B0 | Full field + RFNO | No | No | No | Existing |
| B1 | Bilinear + RFNO | Yes | Yes | No | Frozen |
| B2 | Kriging + RFNO | Yes | Yes | No | Frozen |
| B3 | POD + RFNO | Yes | Yes | Partly | Frozen |
| B4 | Mask U-Net + RFNO | Yes | Yes | Yes | Frozen |
| B5 | GNO/GINO + RFNO | Yes | Yes/latent | Yes | Frozen |
| B6 | GNO/GINO + RFNO | Yes | Yes/latent | Yes | Joint |
| B7 | Direct sparse-to-future | Yes | No/implicit | Yes | Independent |
| P1 | PartialConv-MAE + RFNO | Yes | Yes | Yes | Frozen |
| P2 | PartialConv-MAE + RFNO | Yes | Yes | Yes | Joint |
| P3 | PartialConv-MAE + RFNO + online correction | Yes | Analysis update | Yes | Joint/recurrent |

## Meaning of the comparisons

- B0 is an ideal-observation upper reference, not a sparse baseline.
- B1 is the minimum engineering baseline and validates the complete sparse pipeline.
- B2 and B3 test whether classical spatial priors are sufficient.
- B4 tests image-inpainting reconstruction.
- B5 versus B6 isolates forecast-aware joint training.
- B7 tests whether explicit history reconstruction is necessary.
- P1 versus B4 isolates PartialConv/MAE pretraining benefits.
- P1 versus P2 isolates the value of end-to-end forecast-aware fine-tuning.
- P3 is optional and should begin only after P2 is stable.

## Required experiment factors

Use the same factor definitions across methods:

- observation rate;
- mask family and manifest ID;
- observation noise;
- sensor dropout;
- random seed;
- reconstruction supervision mode;
- RFNO coupling mode;
- rollout horizon;
- checkpoint provenance.

Do not introduce multiple new contributions in one comparison. For example, compare P1 and
P2 with identical reconstructors and masks before adding a physics loss.

## Recommended execution order

1. B0: freeze the existing reference result and metadata.
2. B1: prove the common sparse data and evaluation path.
3. B2/B3: add classical baselines if required by thesis scope.
4. B4 and P1: compare learned gridded reconstruction.
5. B5/B6: add coordinate-aware sparse-sensor experiments.
6. P2: enable joint long-rollout training.
7. B7: test direct sparse-to-future prediction.
8. P3: add sequential assimilation only if schedule permits.
