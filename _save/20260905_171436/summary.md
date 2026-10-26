# ErrorEnsemble training report

- Report: `20260905_171436`
- Created: 2026-09-05T17:14:37
- Dataset signature: `5a9d1623`
- Label / model: `rise_flag` / `regression`
- TelePred delay: 0
- Stages: 3 (finished: 3/3)
- Planned epochs/stage: 2

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

- Loaded models: [0, 1, 2] (keys: ['2c1aa02a', '60477263', '2741bb7c'])

| ensemble | definition | train loss | val loss | train resid MSE | val resid MSE |
|----------|------------|------------|----------|-----------------|---------------|
| `en_0` | `model_0(D)` | 0.512424 | 0.639967 | 3.318205 | 2.824810 |
| `en_1` | `model_0(D) + denorm_0(model_1(D))` | 1.883574 | 1.225749 | 15.929825 | 6.752421 |
| `en_2` | `model_0(D) + denorm_0(model_1(D)) + denorm_1(model_2(D))` | 2.254052 | 3.041441 | 148.274185 | 171.600067 |

- Plot: `ensemble_performance_vs_stage.png`
- Residual plot: `ensemble_residual_vs_stage.png`

## Per-stage training

### model_0 (`2c1aa02a`)
- Finished: True
- History epochs: 2
- Final train loss: 0.5313840643742151
- Final val loss: 0.6367025061246053
- Best val loss: 0.6367025061246053
- Latest epoch curve: epoch=2 batches=2062 avg=0.5313840643742151 complete=True
- Validation plots: 2 full + 2 p0_30 (epochs [1, 2])

### model_1 (`60477263`)
- Finished: True
- History epochs: 2
- Final train loss: 0.23730921862370521
- Final val loss: 0.3793212206844516
- Best val loss: 0.3793212206844516
- Latest epoch curve: epoch=2 batches=2062 avg=0.23730921862370521 complete=True
- Validation plots: 2 full + 2 p0_30 (epochs [1, 2])

### model_2 (`2741bb7c`)
- Finished: True
- History epochs: 2
- Final train loss: 0.26336590612395716
- Final val loss: 0.2687973520362083
- Best val loss: 0.25119940630723664
- Latest epoch curve: epoch=2 batches=2062 avg=0.26336590612395716 complete=True
- Validation plots: 2 full + 2 p0_30 (epochs [1, 2])

## Artifacts
- `ensemble_performance_vs_stage.png` - compare loss(en_k, y) for en_0..en_m
- `ensemble_residual_vs_stage.png` - residual y-en_k for en_0..en_m
- `training_history.png` - overlay of per-stage train/val curves
- `model_i/training_history.png` - per-stage train vs val
- `model_i/latest_epoch_loss.png` - partial/interrupted epoch curve
- `model_i/validation_eXXX.png` - full val batch curves (`--report`)
- `model_i/validation_eXXX_p0_30.png` - [0%, 30%] slice (`--report`)

## Git

- Commit: `179517e`
- Message: `20260905_171436`
