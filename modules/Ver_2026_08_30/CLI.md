# TRAIN / INFER cheat-sheet (simplified CLI)

Remember **verbs**, not 60 flags. Details: `_cli_simplify_plan.txt`.

```text
python TRAIN.py train --model regression --label rise_flag --epochs 12
python TRAIN.py infer --label rise_flag --test 32
python TRAIN.py validate --latest --slice 0:30
python TRAIN.py ensemble --label yield --delays 0,2,5,10
python TRAIN.py error-ensemble --label yield --stages 3 --clean
python TRAIN.py plot --history
python TRAIN.py plot --loss
python TRAIN.py report              # → _save/YYYYMMDD_HHMMSS/ + git commit
python TRAIN.py report my_run
python TRAIN.py train --label yield --set patch_size=15 --set num_layers=6
```

| Verb | Meaning |
|---|---|
| `train` | one model×label cell |
| `infer` / `test` | test-set eval |
| `validate` | `--best` (default) or `--latest` |
| `ensemble` | TelePred delay bank |
| `error-ensemble` | residual stages |
| `plot` | `--history` / `--loss` / `--validation` |
| `report` | export val+history under `_save/`; bare → datetime folder + git commit |
| `search` | hparam search (`--with-delays` optional; alias: `sweep`) |
| `tools` | `compare`, `viz-trend`, `viz-features`, `yield-windows` |

Shared: `--model --label --yield-target --latent-dist --epochs --extend --clean --delay --delays --checkpoint --past --future --batch --set KEY=VAL`

Legacy flat flags still work (warning). Force with `TRAIN_CLI_LEGACY=1`.
`INFER.py` forwards to `TRAIN.py infer`.
