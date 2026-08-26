# ErrorEnsemble training report

- Report: `20260826_210911`
- Created: 2026-08-26T21:09:11
- Dataset signature: `acc18a7e`
- Label / model: `yield` / `regression`
- TelePred delay: 0
- Stages: 3 (finished: 3/3)
- Planned epochs/stage: 2

## Compare ensembles en_0 .. en_m

After training `model_0 .. model_m`, the report scores these ensembles
in **denormalized** label space (each stage uses its saved mean/std):

- `en_0 = denorm_0(model_0)`
- `en_1 = denorm_0(model_0) + denorm_1(model_1)`
- `en_k = sum_{i=0..k} denorm_i(model_i)`

This compares **ensemble** predictions against raw `y`, not each
stage's residual-training loss.

- Loaded models: [0, 1, 2] (keys: ['bfb706ad', 'c89342a3', 'eec62ed6'])

| ensemble | definition | train loss | val loss | train resid MSE | val resid MSE |
|----------|------------|------------|----------|-----------------|---------------|
| `en_0` | `denorm_0(model_0)` | 8190.670322 | 10829.924302 | 4095.335177 | 5414.962091 |
| `en_1` | `denorm_0(model_0) + denorm_1(model_1)` | 12983.243371 | 14665.195408 | 6491.621688 | 7332.597697 |
| `en_2` | `denorm_0(model_0) + denorm_1(model_1) + denorm_2(model_2)` | 28873.039062 | 29730.902464 | 14436.519536 | 14865.451224 |

- Plot: `ensemble_performance_vs_stage.png`
- Residual plot: `ensemble_residual_vs_stage.png`

## Per-stage training

### model_0 (`bfb706ad`)
- Finished: True
- History epochs: 2
- Final train loss: 1.7717032123956005
- Final val loss: 2.2372831402267286
- Best val loss: 2.2208873702696077
- Latest epoch curve: epoch=2 batches=18653 avg=1.7717032123956005 complete=True
- Validation plots: 2 full + 2 p0_30 (epochs [1, 2])

### model_1 (`c89342a3`)
- Finished: True
- History epochs: 2
- Final train loss: 1.7801651008783035
- Final val loss: 2.2249306439586873
- Best val loss: 2.2082069696135442
- Latest epoch curve: epoch=2 batches=18653 avg=1.7801651008783035 complete=True
- Validation plots: 2 full + 2 p0_30 (epochs [1, 2])

### model_2 (`eec62ed6`)
- Finished: True
- History epochs: 2
- Final train loss: 1.7855076592441335
- Final val loss: 2.2256269018141492
- Best val loss: 2.203324854703536
- Latest epoch curve: epoch=2 batches=18653 avg=1.7855076592441335 complete=True
- Validation plots: 2 full + 2 p0_30 (epochs [1, 2])

## Artifacts
- `ensemble_performance_vs_stage.png` - compare loss(en_k, y) for en_0..en_m
- `ensemble_residual_vs_stage.png` - residual y-en_k for en_0..en_m
- `training_history.png` - overlay of per-stage train/val curves
- `model_i/training_history.png` - per-stage train vs val
- `model_i/latest_epoch_loss.png` - partial/interrupted epoch curve
- `model_i/validation_eXXX.png` - full val batch curves (`--report`)
- `model_i/validation_eXXX_p0_30.png` - [0%, 30%] slice (`--report`)
