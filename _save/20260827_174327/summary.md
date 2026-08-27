# ErrorEnsemble training report

- Report: `20260827_174327`
- Created: 2026-08-27T18:44:46
- Dataset signature: `acc18a7e`
- Label / model: `yield` / `regression`
- TelePred delay: 0
- Stages: 3 (finished: 3/3)
- Planned epochs/stage: 1

## Compare ensembles en_0 .. en_m

After training `model_0 .. model_m`, the report scores these ensembles
in **y-normalized** space (same space as dataloader `y`):

- Wholesale over dataset D:
  `en_0 = model_0(D)`
  `en_k = model_0(D) + denorm_0(model_1(D)) + ... + denorm_{k-1}(model_k(D))`

For each i: load only model_i → full pass over D → add in y-space → free → next.
Never loads several models at once; never sums inside one shared batch loop.

Metrics compare `en_k` to `y` (normalized), not `y_raw`.
Invariant: loss(en_i, y) must not grow with i (retrain with --clean if it does).

- Loaded models: [0, 1, 2] (keys: ['bfb706ad', 'c89342a3', 'eec62ed6'])

| ensemble | definition | train loss | val loss | train resid MSE | val resid MSE |
|----------|------------|------------|----------|-----------------|---------------|
| `en_0` | `model_0(D)` | 1.606578 | 2.431997 | 0.803289 | 1.215999 |
| `en_1` | `model_0(D) + denorm_0(model_1(D))` | 1.601092 | 2.399681 | 0.800546 | 1.199840 |
| `en_2` | `model_0(D) + denorm_0(model_1(D)) + denorm_1(model_2(D))` | 1.581241 | 2.417483 | 0.790621 | 1.208742 |

- Plot: `ensemble_performance_vs_stage.png`
- Residual plot: `ensemble_residual_vs_stage.png`

## Per-stage training

### model_0 (`bfb706ad`)
- Finished: True
- History epochs: 1
- Final train loss: 1.915244749455083
- Final val loss: 2.2081237948807506
- Best val loss: 2.2081237948807506
- Latest epoch curve: epoch=1 batches=18653 avg=1.915244749455083 complete=True
- Validation plots: 1 full + 1 p0_30 (epochs [1])

### model_1 (`c89342a3`)
- Finished: True
- History epochs: 1
- Final train loss: 2.0099667934577523
- Final val loss: 2.396962688502502
- Best val loss: 2.396962688502502
- Latest epoch curve: epoch=1 batches=18653 avg=2.0099667934577523 complete=True
- Validation plots: 1 full + 1 p0_30 (epochs [1])

### model_2 (`eec62ed6`)
- Finished: True
- History epochs: 1
- Final train loss: 1.9940959689544044
- Final val loss: 2.401646362216466
- Best val loss: 2.401646362216466
- Latest epoch curve: epoch=1 batches=18653 avg=1.9940959689544044 complete=True
- Validation plots: 1 full + 1 p0_30 (epochs [1])

## Artifacts
- `ensemble_performance_vs_stage.png` - compare loss(en_k, y) for en_0..en_m
- `ensemble_residual_vs_stage.png` - residual y-en_k for en_0..en_m
- `training_history.png` - overlay of per-stage train/val curves
- `model_i/training_history.png` - per-stage train vs val
- `model_i/latest_epoch_loss.png` - partial/interrupted epoch curve
- `model_i/validation_eXXX.png` - full val batch curves (`--report`)
- `model_i/validation_eXXX_p0_30.png` - [0%, 30%] slice (`--report`)
