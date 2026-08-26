# ErrorEnsemble training report

- Report: `20260827_003658`
- Created: 2026-08-27T00:36:58
- Dataset signature: `acc18a7e`
- Label / model: `yield` / `regression`
- TelePred delay: 0
- Stages: 3 (finished: 3/3)
- Planned epochs/stage: 1

## Compare ensembles en_0 .. en_m

After training `model_0 .. model_m`, the report scores these ensembles
in **denormalized** label space (each stage uses its saved mean/std):

- Wholesale over dataset D: `en_k = model_0(D) + ... + model_k(D)`
  (each stage denormalized with its saved mean/std, then added)

For each i: load only model_i → full pass over D → add → free → next.
Never loads several models at once; never sums inside one shared batch loop.

This compares **ensemble** predictions against raw `y`, not each
stage's residual-training loss.

- Loaded models: [0, 1, 2] (keys: ['bfb706ad', 'c89342a3', 'eec62ed6'])

| ensemble | definition | train loss | val loss | train resid MSE | val resid MSE |
|----------|------------|------------|----------|-----------------|---------------|
| `en_0` | `model_0(D)` | 8260.145508 | 10771.157227 | 4130.072754 | 5385.578613 |
| `en_1` | `model_0(D) + model_1(D)` | 8201.207031 | 10674.111328 | 4100.604004 | 5337.055664 |
| `en_2` | `model_0(D) + model_1(D) + model_2(D)` | 8165.592285 | 10647.338867 | 4082.795898 | 5323.669922 |

- Plot: `ensemble_performance_vs_stage.png`
- Residual plot: `ensemble_residual_vs_stage.png`

## Per-stage training

### model_0 (`bfb706ad`)
- Finished: True
- History epochs: 1
- Final train loss: 1.9127858124159298
- Final val loss: 2.192278943010523
- Best val loss: 2.192278943010523
- Latest epoch curve: epoch=1 batches=18653 avg=1.9127858124159298 complete=True
- Validation plots: 1 full + 1 p0_30 (epochs [1])

### model_1 (`c89342a3`)
- Finished: True
- History epochs: 1
- Final train loss: 2.007241144457599
- Final val loss: 2.382763709760538
- Best val loss: 2.382763709760538
- Latest epoch curve: epoch=1 batches=18653 avg=2.007241144457599 complete=True
- Validation plots: 1 full + 1 p0_30 (epochs [1])

### model_2 (`eec62ed6`)
- Finished: True
- History epochs: 1
- Final train loss: 1.9999514976445967
- Final val loss: 2.382432418554735
- Best val loss: 2.382432418554735
- Latest epoch curve: epoch=1 batches=18653 avg=1.9999514976445967 complete=True
- Validation plots: 1 full + 1 p0_30 (epochs [1])

## Artifacts
- `ensemble_performance_vs_stage.png` - compare loss(en_k, y) for en_0..en_m
- `ensemble_residual_vs_stage.png` - residual y-en_k for en_0..en_m
- `training_history.png` - overlay of per-stage train/val curves
- `model_i/training_history.png` - per-stage train vs val
- `model_i/latest_epoch_loss.png` - partial/interrupted epoch curve
- `model_i/validation_eXXX.png` - full val batch curves (`--report`)
- `model_i/validation_eXXX_p0_30.png` - [0%, 30%] slice (`--report`)
