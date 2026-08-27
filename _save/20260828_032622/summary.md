# ErrorEnsemble training report

- Report: `20260828_032622`
- Created: 2026-08-28T03:26:24
- Dataset signature: `2c098af5`
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

- Loaded models: [0, 1, 2] (keys: ['e5ad0d83', '75eaf5ab', 'b3f42c18'])

| ensemble | definition | train loss | val loss | train resid MSE | val resid MSE |
|----------|------------|------------|----------|-----------------|---------------|
| `en_0` | `model_0(D)` | 0.973543 | 1.054746 | 0.973543 | 1.054746 |
| `en_1` | `model_0(D) + denorm_0(model_1(D))` | 0.972254 | 1.054324 | 0.972254 | 1.054324 |
| `en_2` | `model_0(D) + denorm_0(model_1(D)) + denorm_1(model_2(D))` | 0.971659 | 1.054187 | 0.971659 | 1.054187 |

- Plot: `ensemble_performance_vs_stage.png`
- Residual plot: `ensemble_residual_vs_stage.png`

## Per-stage training

### model_0 (`e5ad0d83`)
- Finished: True
- History epochs: 1
- Final train loss: 1.0135071603664736
- Final val loss: 1.0547937734484643
- Best val loss: 1.0547937734484643
- Latest epoch curve: epoch=1 batches=18643 avg=1.0135071603664736 complete=True
- Validation plots: 1 full + 1 p0_30 (epochs [1])

### model_1 (`75eaf5ab`)
- Finished: True
- History epochs: 1
- Final train loss: 1.0245755326792334
- Final val loss: 1.0833536298432904
- Best val loss: 1.0833536298432904
- Latest epoch curve: epoch=1 batches=18643 avg=1.0245755326792334 complete=True
- Validation plots: 1 full + 1 p0_30 (epochs [1])

### model_2 (`b3f42c18`)
- Finished: True
- History epochs: 1
- Final train loss: 1.0296807235387277
- Final val loss: 1.0845272660309124
- Best val loss: 1.0845272660309124
- Latest epoch curve: epoch=1 batches=18643 avg=1.0296807235387277 complete=True
- Validation plots: 1 full + 1 p0_30 (epochs [1])

## Artifacts
- `ensemble_performance_vs_stage.png` - compare loss(en_k, y) for en_0..en_m
- `ensemble_residual_vs_stage.png` - residual y-en_k for en_0..en_m
- `training_history.png` - overlay of per-stage train/val curves
- `model_i/training_history.png` - per-stage train vs val
- `model_i/latest_epoch_loss.png` - partial/interrupted epoch curve
- `model_i/validation_eXXX.png` - full val batch curves (`--report`)
- `model_i/validation_eXXX_p0_30.png` - [0%, 30%] slice (`--report`)
