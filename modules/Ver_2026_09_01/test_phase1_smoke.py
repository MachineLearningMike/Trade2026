#!/usr/bin/env python
"""
Minimum Phase 1 end-to-end smoke (CPU, few steps).

Proves:
  - (regression, rise_flag) builds, trains at least one minibatch path
  - checkpoint is written
  - INFER run_test metrics path handles out_dim==1 (BCE / accuracy)

Usage (from this directory, torch env)::

    python test_phase1_smoke.py

Exit 0 on success. Safe to delete checkpoint dir afterward.
"""
from __future__ import annotations

import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


def _smoke_config(ckpt_dir: Path, **overrides) -> dict:
    from config import get_config
    from INFER import _resolve_task

    cfg = get_config()
    cfg.update({
        'price_feature_idx': 0,
        'past_window': 40,
        'future_window': 10,
        'step': 40,
        'telepred_delay': 0,
        'telepred_delays': [0, 2],
        'label': 'rise_flag',
        'model_type': 'regression',
        'yield_target': 'both',
        'latent_distribution': 'normal',
        'output_dim': 1,
        'patch_size': 5,
        'num_layers': 1,
        'num_heads': 5,
        'n_latent_macros': 5,  # embed_dim = (5+5)*5 = 50, divisible by 5
        'ff_dim_multiplier': 2,
        'head_hidden_dim_multiplier': 2,
        'dropout': 0.0,
        'batch_size': 4,
        'validate_batch_size': 4,
        'test_batch_size': 4,
        'epochs': 1,
        'learning_rate': 1e-3,
        'device': 'cpu',
        'validate': False,
        'resume': False,
        'clean': True,
        'use_reconstruction': False,
        'emit_reconstruction': False,
        'checkpoint_dir': str(ckpt_dir),
        'save_every_n_minutes': 60.0,
        'recent_checkpoints': 1,
        'normalization_layers': 'two-layer',
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'shuffle_split': False,
        'error_ensemble_n': 2,
    })
    if overrides:
        cfg.update(overrides)
    if str(cfg.get('model_type', '')).lower() == 'vae':
        cfg['label'] = 'none'
    _resolve_task(cfg).sync_output_dim(cfg)
    return cfg


def _build_dataset(cfg: dict):
    from data import TradingDataset
    from INFER import _resolve_task

    task = _resolve_task(cfg)
    dataset = TradingDataset(cfg, task=task)
    return dataset, dataset.total_features, task


def main() -> int:
    root = Path(__file__).resolve().parent
    ckpt_dir = root / '_phase1_smoke_ckpts'
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('Phase 1 smoke: (regression, rise_flag) CPU tiny e2e')
    print('=' * 60)

    cfg = _smoke_config(ckpt_dir)
    assert cfg['label'] == 'rise_flag'
    assert int(cfg['output_dim']) == 1

    print('\n[1/3] Building rise_flag dataset...')
    dataset, total_features, task = _build_dataset(cfg)
    assert getattr(dataset, 'label_mode') == 'rise_flag'
    assert int(dataset.output_dim) == 1
    assert len(dataset) >= 4, f'too few samples: {len(dataset)}'
    print(f'  samples={len(dataset)} train/val/test='
          f'{len(dataset.train_indices)}/{len(dataset.val_indices)}/'
          f'{len(dataset.test_indices)} output_dim={dataset.output_dim}')

    # Stub TradingDataset so train/test share the same built instance.
    import data as data_mod
    import INFER as infer_mod

    def _FakeTradingDataset(_config, task=None):
        return dataset

    data_mod.TradingDataset = _FakeTradingDataset
    infer_mod.TradingDataset = _FakeTradingDataset

    print('\n[2/3] train_one_run (1 epoch, no val, CPU)...')
    from trainer import train_one_run
    outcome = train_one_run(
        cfg,
        dataset,
        total_features,
        task=task,
        num_epochs=1.0,
        resume=False,
        extend=False,
        probe_weights=True,
        show_banner=True,
        suppress_history_plot=True,
    )
    key = outcome.get('checkpoint_key') if isinstance(outcome, dict) else None
    if not key:
        # ResumableTrainer may return status string; resolve key from disk/config.
        from trainer import generate_checkpoint_key
        key = generate_checkpoint_key(cfg, dataset.cache_key)
    key_dir = ckpt_dir / key
    if not key_dir.exists():
        print(f'FAIL: expected checkpoint dir missing: {key_dir}')
        return 1
    pts = list(key_dir.glob('*.pt'))
    if not pts:
        print(f'FAIL: no .pt under {key_dir}')
        return 1
    print(f'  checkpoint_key={key}  pt_files={len(pts)}')

    print('\n[3/3] INFER run_test metrics (rise_flag / dim 1)...')
    from INFER import run_test
    # Cap samples; no interactive viz.
    result = run_test(cfg, max_samples=8, visualize=False)
    metrics = result.get('metrics') or {}
    out_dim = int(result.get('output_dim', -1))
    if out_dim != 1:
        print(f'FAIL: output_dim={out_dim}, expected 1')
        return 1
    if 'bce_with_logits' not in metrics or 'accuracy' not in metrics:
        print(f'FAIL: rise_flag metrics missing: {metrics}')
        return 1
    bce = float(metrics['bce_with_logits'])
    acc = float(metrics['accuracy'])
    if not np.isfinite(bce) or not np.isfinite(acc):
        print(f'FAIL: non-finite metrics bce={bce} acc={acc}')
        return 1
    print(f'  bce={bce:.4f}  accuracy={acc:.4f}')

    print('\n' + '=' * 60)
    print('Phase 1 smoke PASSED')
    print(f'  checkpoints kept at: {ckpt_dir}')
    print('=' * 60)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
