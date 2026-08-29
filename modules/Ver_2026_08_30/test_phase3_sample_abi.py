#!/usr/bin/env python
"""Phase 3 smoke: opaque sample dict ABI + label_cache.build_cache_row."""
from __future__ import annotations

import traceback

import torch


def main() -> int:
    from label_cache import build_cache_row, pack_sample_for_label
    from sample_batch import (
        as_train_tuple,
        collate_samples,
        resolve_normalization_mode,
        unpack_model_input,
        unpack_target,
    )

    assert resolve_normalization_mode({'normalization': 'two-layer'}) == 'two-layer'
    assert resolve_normalization_mode({'normalization': 'one-layer'}) == 'one-layer'
    assert resolve_normalization_mode({'normalize_window': True}) == 'two-layer'
    assert resolve_normalization_mode({'normalize_window': False}) == 'one-layer'

    mean = torch.zeros(3)
    std = torch.ones(3)
    future = torch.linspace(1, 10, 10)
    history = torch.linspace(0.5, 1.0, 20)
    y, h = build_cache_row(
        'rise_flag',
        future_prices=future,
        history_prices=history,
        current_price=torch.tensor(1.0),
        mean=mean,
        std=std,
        price_feature_idx=0,
        normalize_window=True,
        min_trend_horizon=3,
        future_window=10,
        fitness_threshold=0.75,
        horizon_search='longest_linear',
        regime_use_past=True,
        min_yield_percent=0.26,
    )
    assert y.shape == (1,)
    assert h == 0.0

    sample = pack_sample_for_label(
        'rise_flag',
        x=torch.zeros(8, 3),
        p=torch.zeros(4),
        mean=torch.zeros(3),
        std=torch.ones(3),
        shape=torch.zeros(4, 3),
        y=y,
    )
    assert set(sample) >= {'x', 'p', 'mean', 'std', 'shape', 'y'}
    batch = collate_samples([sample, sample])
    assert batch['x'].shape[0] == 2
    kwargs = unpack_model_input(batch)
    assert kwargs['x'].shape[0] == 2
    assert unpack_target(batch).shape[0] == 2
    tup = as_train_tuple(batch)
    assert len(tup) == 6

    # Dataset returns dict
    from test_phase1_smoke import _build_synthetic_dataset, _smoke_config
    from pathlib import Path
    cfg = _smoke_config(Path('_phase3_tmp_ckpts'))
    ds, _ = _build_synthetic_dataset(cfg)
    item = ds[0]
    assert isinstance(item, dict)
    assert 'y' in item and item['y'].shape[-1] == 1

    print('phase3 sample ABI smoke ok')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
