#!/usr/bin/env python
"""Phase 3 smoke: opaque datapoint ABI + Task.build_raw_y / pack / unpack."""
from __future__ import annotations

import traceback

import torch


def main() -> int:
    from data import ThreadedDataLoader, TradingDataset
    from Task_Rise import Task_Rise

    assert TradingDataset.resolve_normalization_layers({'normalization_layers': 'two-layer'}) == 'two-layer'
    assert TradingDataset.resolve_normalization_layers({'normalization_layers': 'one-layer'}) == 'one-layer'
    assert TradingDataset.resolve_normalization_layers({}) == 'two-layer'

    T, F = 40, 3
    total_x = torch.randn(T, F)
    total_p = torch.linspace(1.0, 2.0, T)
    mean = torch.zeros(F)
    std = torch.ones(F)
    t = 25
    cfg = {
        'past_window': 10,
        'future_window': 10,
        'telepred_delay': 0,
        'price_feature_idx': 0,
        'normalization_layers': 'two-layer',
        'min_yield_percent': 0.26,
    }
    task = Task_Rise
    y = task.build_raw_y(
        t, total_x, total_p, cfg, mean=mean, std=std
    )
    assert y.shape == (1,)
    assert not isinstance(y, tuple)

    # Synthetic datapoint dict matching default Task schema.
    sample = {
        'x': torch.zeros(8, 3),
        'p': torch.zeros(4),
        'mean': torch.zeros(3),
        'std': torch.ones(3),
        'momentum': torch.zeros(4, 3),
        'y': y,
        'label': task.name,
    }
    assert set(sample) >= {'x', 'p', 'mean', 'std', 'momentum', 'y', 'label'}
    batch = ThreadedDataLoader.collate_datapoints([sample, sample])
    assert batch['x'].shape[0] == 2
    kwargs, y_b = task.unpack_datapoint(batch)
    assert kwargs['x'].shape[0] == 2
    assert y_b.shape[0] == 2
    assert set(kwargs) == set(task.MODEL_INPUT_KEYS)

    from test_phase1_smoke import _build_dataset, _smoke_config
    from pathlib import Path
    cfg_ds = _smoke_config(Path('_phase3_tmp_ckpts'))
    ds, _, _ = _build_dataset(cfg_ds)
    item = ds[0]
    assert isinstance(item, dict)
    assert 'y' in item and item['y'].shape[-1] == 1
    kwargs0, y0 = ds.task.unpack_datapoint(item)
    assert set(kwargs0) == set(ds.task.MODEL_INPUT_KEYS)
    assert y0.shape[-1] == 1

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
