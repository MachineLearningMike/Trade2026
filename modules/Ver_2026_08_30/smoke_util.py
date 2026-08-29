# smoke_util.py
# Shared tiny synthetic configs/datasets for Phase 1 / golden matrix smokes.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def base_smoke_config(ckpt_dir: Path, **overrides: Any) -> dict:
    from config import get_config, apply_overrides
    from labels import sync_output_dim
    from compat import resolve_atomic_slot

    cfg = get_config()
    apply_overrides(
        cfg,
        {
            'use_binance_data': False,
            'total_synthetic_days': 1,
            'num_features': 5,
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
            'output_dim': 2,
            'patch_size': 5,
            'num_layers': 1,
            'num_heads': 5,
            'n_latent_macros': 5,
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
            'emit_reconstruction': False,
            'checkpoint_dir': str(ckpt_dir),
            'save_every_n_minutes': 60.0,
            'recent_checkpoints': 1,
            'normalization': 'two-layer',
            'normalize_window': True,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'shuffle_split': False,
            'error_ensemble_n': 2,
        },
    )
    apply_overrides(cfg, overrides)
    if str(cfg.get('model_type', '')).lower() == 'vae':
        resolve_atomic_slot(cfg, coerce_vae_to_none=True, warn=False)
    sync_output_dim(cfg)
    return cfg


def build_synthetic_dataset(cfg: dict):
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
