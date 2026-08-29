#!/usr/bin/env python
"""
Minimum Phase 1 end-to-end smoke (CPU, synthetic, few steps).

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


def _smoke_config(ckpt_dir: Path) -> dict:
    from config import get_config
    from labels import sync_output_dim

    cfg = get_config()
    cfg.update({
        'use_binance_data': False,
        'total_synthetic_days': 1,
        'num_features': 5,
        'price_feature_idx': 0,
        'past_window': 40,
        'future_window': 10,
        'step': 40,
        'telepred_delay': 0,
        'label': 'rise_flag',
        'model_type': 'regression',
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
        'look_price': False,
        'use_reconstruction': False,
        'checkpoint_dir': str(ckpt_dir),
        'save_every_n_minutes': 60.0,
        'recent_checkpoints': 1,
        'normalize_window': True,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'shuffle_split': False,
    })
    sync_output_dim(cfg)
    return cfg


def _build_synthetic_dataset(cfg: dict):
    """Market-only synthetic TradingDataset (no Binance / FSV dependency)."""
    from data import TradingDataset, generate_synthetic_data

    T = 60 * 24 * int(cfg['total_synthetic_days'])
    F = int(cfg['num_features'])
    price_idx = int(cfg['price_feature_idx'])
    market_list = generate_synthetic_data(T, F, price_idx)
    names = [f'feature_{i}' for i in range(F)]
    dataset = TradingDataset(
        features=market_list,
        n_market_features=F,
        n_macro_features=0,
        feature_names=names,
        price_feature_idx=price_idx,
        past_window=int(cfg['past_window']),
        future_minutes=int(cfg['future_window']),
        step=int(cfg['step']),
        device=cfg['device'],
        normalize_window=bool(cfg.get('normalize_window', True)),
        runtime_config=cfg,
    )
    total = np.stack(market_list, axis=0).astype(np.float32)
    return dataset, total


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

    print('\n[1/3] Building synthetic rise_flag dataset...')
    dataset, total_features = _build_synthetic_dataset(cfg)
    assert getattr(dataset, 'label_mode') == 'rise_flag'
    assert int(dataset.output_dim) == 1
    assert len(dataset) >= 4, f'too few samples: {len(dataset)}'
    print(f'  samples={len(dataset)} train/val/test='
          f'{len(dataset.train_indices)}/{len(dataset.val_indices)}/'
          f'{len(dataset.test_indices)} output_dim={dataset.output_dim}')

    # INFER.run_test → get_dataset(config). Stub so train/test share the same
    # synthetic market-only dataset (avoids Binance/FSV and key mismatch).
    import dataset_build
    import INFER as infer_mod

    def _fake_get_dataset(_config):
        return dataset, total_features

    dataset_build.get_dataset = _fake_get_dataset
    infer_mod.get_dataset = _fake_get_dataset

    print('\n[2/3] train_one_run (1 epoch, no val, CPU)...')
    from train_run import train_one_run
    outcome = train_one_run(
        cfg,
        dataset,
        total_features,
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
