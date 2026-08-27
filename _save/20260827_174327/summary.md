# ErrorEnsemble training report

- Report: `20260827_174327`
- Created: 2026-08-27T17:43:27
- Dataset signature: `3a4ebf56`
- Label / model: `yield` / `regression`
- TelePred delay: 0
- Stages: 3 (finished: 0/3)
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

- No ensemble comparison (no loadable models, or eval skipped).

## Per-stage training

### model_0 (`fee9586e`)
- Finished: False
- History epochs: 0
- Final train loss: None
- Final val loss: None
- Best val loss: None
- Validation plots: 0 full + 0 p0_30 (epochs None)
- Error: checkpoint dir missing

### model_1 (`1df08984`)
- Finished: False
- History epochs: 0
- Final train loss: None
- Final val loss: None
- Best val loss: None
- Validation plots: 0 full + 0 p0_30 (epochs None)
- Error: checkpoint dir missing

### model_2 (`fa33829d`)
- Finished: False
- History epochs: 0
- Final train loss: None
- Final val loss: None
- Best val loss: None
- Validation plots: 0 full + 0 p0_30 (epochs None)
- Error: checkpoint dir missing

## Artifacts
- `ensemble_performance_vs_stage.png` - compare loss(en_k, y) for en_0..en_m
- `ensemble_residual_vs_stage.png` - residual y-en_k for en_0..en_m
- `training_history.png` - overlay of per-stage train/val curves
- `model_i/training_history.png` - per-stage train vs val
- `model_i/latest_epoch_loss.png` - partial/interrupted epoch curve
- `model_i/validation_eXXX.png` - full val batch curves (`--report`)
- `model_i/validation_eXXX_p0_30.png` - [0%, 30%] slice (`--report`)
