#!/usr/bin/env python
"""
Phase 7 golden smoke matrix (§0 COMPAT + ensemble kinds).

CPU / synthetic / 1 epoch. Proves each atomic cell trains + infers, plus
TelePred (delays 0,2) and ErrorEnsemble (n=2).

Usage::

    python test_golden_matrix.py

Exit 0 on full pass. Checkpoints under ``_golden_smoke_ckpts/`` (deleted on success).
"""
from __future__ import annotations

import math
import shutil
import sys
import traceback
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
CKPT_ROOT = ROOT / '_golden_smoke_ckpts'


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def _train_infer_cell(model_type: str, label: str, cell_dir: Path, **overrides):
    from test_phase1_smoke import _smoke_config as base_smoke_config, _build_synthetic_dataset as build_synthetic_dataset
    from train_run import train_one_run
    from trainer import generate_checkpoint_key
    from INFER import run_test
    import dataset_build
    import INFER as infer_mod

    if cell_dir.exists():
        shutil.rmtree(cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)

    cfg = base_smoke_config(
        cell_dir,
        model_type=model_type,
        label=label,
        **overrides,
    )
    dataset, total, task = build_synthetic_dataset(cfg)
    assert len(dataset) >= 4

    def _fake_get_dataset(_c, task=None):
        # Rebuild with the *caller* config (delay/stage may differ).
        ds, tot, _t = build_synthetic_dataset(_c)
        return ds, tot

    dataset_build.get_dataset = _fake_get_dataset
    infer_mod.get_dataset = _fake_get_dataset

    outcome = train_one_run(
        cfg,
        dataset,
        total,
        task=task,
        num_epochs=1.0,
        resume=False,
        probe_weights=False,
        show_banner=False,
        suppress_history_plot=True,
    )
    key = outcome.get('checkpoint_key') or generate_checkpoint_key(cfg, dataset.cache_key)
    key_dir = cell_dir / key
    assert key_dir.exists(), f'missing key dir {key_dir}'
    assert list(key_dir.glob('*.pt')), f'no .pt in {key_dir}'
    assert (key_dir / 'run_config.json').exists(), 'run_config.json missing'
    train_loss = outcome.get('final_train_loss')
    assert _finite(train_loss), f'non-finite train loss {train_loss}'

    # Infer metrics (skip viz). For vae/none supervised metrics are skipped.
    result = run_test(cfg, max_samples=8, visualize=False)
    assert int(result.get('output_dim', -1)) == int(cfg['output_dim'])
    metrics = result.get('metrics') or {}
    if label != 'none':
        assert metrics, f'empty metrics for ({model_type}, {label})'
    print(
        f'  OK atomic ({model_type}, {label}) key={key} '
        f'loss={train_loss:.4f} metrics_keys={list(metrics)[:5]}'
    )
    return cfg, key


def _telepred_bank(cell_dir: Path):
    from test_phase1_smoke import _smoke_config as base_smoke_config, _build_synthetic_dataset as build_synthetic_dataset
    from ensemble_train import run_ensemble_train

    if cell_dir.exists():
        shutil.rmtree(cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)
    cfg = base_smoke_config(
        cell_dir,
        model_type='regression',
        label='yield',
        yield_target='both',
        telepred_delays=[0, 2],
    )

    def _gd(c, task=None):
        ds, tot, _t = build_synthetic_dataset(c)
        return ds, tot

    results = run_ensemble_train(
        cfg,
        delays=[0, 2],
        get_dataset=_gd,
        num_epochs=1.0,
        write_csv=False,
        probe_weights=False,
        continue_on_error=False,
    )
    assert len(results) == 2, results
    for row in results:
        assert row.get('status') == 'ok', row
        assert row.get('checkpoint_key'), row
        assert _finite(row.get('final_train_loss')), row
    keys = [p for p in cell_dir.iterdir() if p.is_dir()]
    assert len(keys) >= 2, f'telepred expected >=2 key dirs, got {keys}'
    print(f'  OK telepred yield×regression delays=[0,2] n_keys={len(keys)}')


def _error_ensemble(cell_dir: Path):
    from test_phase1_smoke import _smoke_config as base_smoke_config, _build_synthetic_dataset as build_synthetic_dataset
    from error_ensemble import (
        count_finished_error_ensemble_stages,
        run_error_ensemble_train,
    )

    if cell_dir.exists():
        shutil.rmtree(cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)
    cfg = base_smoke_config(
        cell_dir,
        model_type='regression',
        label='yield',
        yield_target='near',
        error_ensemble_n=2,
    )
    dataset, total, task = build_synthetic_dataset(cfg)
    out = run_error_ensemble_train(
        cfg,
        dataset,
        total,
        task=task,
        n=2,
        start_stage=0,
        num_epochs=1.0,
        resume=False,
        probe_weights=False,
    )
    finished = count_finished_error_ensemble_stages(cfg, dataset)
    assert finished >= 1, f'error ensemble finished={finished} out={out}'
    print(f'  OK error_ensemble yield×regression n=2 finished={finished}')


def main() -> int:
    print('=' * 60)
    print('Phase 7 golden smoke matrix')
    print('=' * 60)
    if CKPT_ROOT.exists():
        shutil.rmtree(CKPT_ROOT)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    cells = [
        ('regression', 'trend', {'output_dim': 2}),
        ('regression', 'yield', {'yield_target': 'both'}),
        ('regression', 'latent', {'latent_distribution': 'normal'}),
        ('regression', 'rise_flag', {}),
        ('vae', 'none', {
            'emit_reconstruction': True,
            'use_reconstruction': True,
            'output_dim': 1,
        }),
    ]

    print('\n[1] Atomic COMPAT cells')
    for model_type, label, extra in cells:
        print(f'→ ({model_type}, {label})')
        _train_infer_cell(
            model_type,
            label,
            CKPT_ROOT / f'{model_type}_{label}',
            **extra,
        )

    print('\n[2] TelePred ensemble kind')
    _telepred_bank(CKPT_ROOT / 'telepred_yield')

    print('\n[3] ErrorEnsemble kind')
    _error_ensemble(CKPT_ROOT / 'error_yield')

    print('\n' + '=' * 60)
    print('Phase 7 golden matrix PASSED')
    print('=' * 60)
    shutil.rmtree(CKPT_ROOT, ignore_errors=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
