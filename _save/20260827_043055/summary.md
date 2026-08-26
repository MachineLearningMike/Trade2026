# ErrorEnsemble training report

- Report: `20260827_043055`
- Created: 2026-08-27T04:30:55
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
| `en_0` | `model_0(D)` | 1.823483 | 2.226058 | 0.911741 | 1.113029 |
| `en_1` | `model_0(D) + denorm_0(model_1(D))` | 1.819356 | 2.217079 | 0.909678 | 1.108539 |
| `en_2` | `model_0(D) + denorm_0(model_1(D)) + denorm_1(model_2(D))` | 1.816674 | 2.213089 | 0.908337 | 1.106544 |

- Plot: `ensemble_performance_vs_stage.png`
- Residual plot: `ensemble_residual_vs_stage.png`

## Per-stage training

### model_0 (`bfb706ad`)
- Finished: True
- History epochs: 1
- Final train loss: 1.91120314350756
- Final val loss: 2.225521610534925
- Best val loss: 2.225521610534925
- Latest epoch curve: epoch=1 batches=18653 avg=1.91120314350756 complete=True
- Validation plots: 1 full + 1 p0_30 (epochs [1])

### model_1 (`c89342a3`)
- Finished: True
- History epochs: 1
- Final train loss: 2.0022759657823093
- Final val loss: 2.455512171761939
- Best val loss: 2.455512171761939
- Latest epoch curve: epoch=1 batches=18653 avg=2.0022759657823093 complete=True
- Validation plots: 1 full + 1 p0_30 (epochs [1])

### model_2 (`eec62ed6`)
- Finished: True
- History epochs: 1
- Final train loss: 2.0058806787618892
- Final val loss: 2.451379407416973
- Best val loss: 2.451379407416973
- Latest epoch curve: epoch=1 batches=18653 avg=2.0058806787618892 complete=True
- Validation plots: 1 full + 1 p0_30 (epochs [1])

## Artifacts
- `ensemble_performance_vs_stage.png` - compare loss(en_k, y) for en_0..en_m
- `ensemble_residual_vs_stage.png` - residual y-en_k for en_0..en_m
- `training_history.png` - overlay of per-stage train/val curves
- `model_i/training_history.png` - per-stage train vs val
- `model_i/latest_epoch_loss.png` - partial/interrupted epoch curve
- `model_i/validation_eXXX.png` - full val batch curves (`--report`)
- `model_i/validation_eXXX_p0_30.png` - [0%, 30%] slice (`--report`)
