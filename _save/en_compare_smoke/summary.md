# ErrorEnsemble training report

- Report: `en_compare_smoke`
- Created: 2026-08-26T19:24:53
- Dataset signature: `acc18a7e`
- Label / model: `yield` / `regression`
- TelePred delay: 0
- Stages: 3 (finished: 1/3)
- Planned epochs/stage: 2

## Compare ensembles en_0 .. en_m

After training `model_0 .. model_m`, the report scores these ensembles
on train/val with the label loss:

- `en_0 = model_0`
- `en_1 = model_0 + model_1`
- `en_k = sum_{i=0..k} model_i`

This compares **ensemble** predictions against `y`, not each stage's
residual-training loss.

- Loaded models: [0, 1] (keys: ['bfb706ad', 'c89342a3'])

| ensemble | definition | train loss | val loss | train resid MSE | val resid MSE |
|----------|------------|------------|----------|-----------------|---------------|
| `en_0` | `model_0` | 1.997181 | 3.078714 | 0.998590 | 1.539357 |
| `en_1` | `model_0 + model_1` | 1.925805 | 3.122229 | 0.962903 | 1.561114 |

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
- Finished: False
- History epochs: 1
- Final train loss: 1.823045553107789
- Final val loss: 2.2082069696135442
- Best val loss: 2.2082069696135442
- Latest epoch curve: epoch=1 batches=18653 avg=1.823045553107789 complete=True
- Validation plots: 1 full + 1 p0_30 (epochs [1])

### model_2 (`eec62ed6`)
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
