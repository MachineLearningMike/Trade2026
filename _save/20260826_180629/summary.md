# ErrorEnsemble training report

- Report: `20260826_180629`
- Created: 2026-08-26T18:07:57
- Dataset signature: `acc18a7e`
- Label / model: `yield` / `regression`
- TelePred delay: 0
- Stages: 3 (finished: 0/3)
- Planned epochs/stage: 2

## Stages

### model_0 (`bfb706ad`)
- Finished: False
- History epochs: 0
- Final train loss: None
- Final val loss: None
- Best val loss: None
- Latest epoch curve: epoch=1 batches=12217 avg=1.9440210957500619 complete=False
- Validation plots: 0 full + 0 p0_30 (epochs [])

### model_1 (`c89342a3`)
- Finished: False
- History epochs: 0
- Final train loss: None
- Final val loss: None
- Best val loss: None
- Validation plots: 0 full + 0 p0_30 (epochs [])

### model_2 (`eec62ed6`)
- Finished: False
- History epochs: 0
- Final train loss: None
- Final val loss: None
- Best val loss: None
- Validation plots: 0 full + 0 p0_30 (epochs None)
- Error: checkpoint dir missing

## Artifacts
- `training_history.png` - overlay of all stages (if any epoch finished)
- `model_i/training_history.png` - per-stage train vs val
- `model_i/latest_epoch_loss.png` - partial/interrupted epoch minibatch curve
- `model_i/validation_eXXX.png` - full val batch curves (`--report`)
- `model_i/validation_eXXX_p0_30.png` - [0%, 30%] slice (`--report`)
