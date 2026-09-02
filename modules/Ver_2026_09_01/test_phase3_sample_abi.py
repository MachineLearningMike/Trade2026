#!/usr/bin/env python
"""Phase 3 smoke: opaque sample dict ABI + Task.build_raw_y."""
from __future__ import annotations

import traceback

import torch


def main() -> int:
    from data import (
        as_train_tuple,
        collate_samples,
        pack_model_sample,
        resolve_normalization_layers,
        unpack_model_input,
        unpack_target,
    )
    from task_base import get_task_class

    assert resolve_normalization_layers({'normalization_layers': 'two-layer'}) == 'two-layer'
    assert resolve_normalization_layers({'normalization_layers': 'one-layer'}) == 'one-layer'
    assert resolve_normalization_layers({}) == 'two-layer'

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
    y = get_task_class('rise_flag').build_raw_y(
        t, total_x, total_p, cfg, mean=mean, std=std
    )
    assert y.shape == (1,)
    assert not isinstance(y, tuple)

    sample = pack_model_sample(
        x=torch.zeros(8, 3),
        p=torch.zeros(4),
        mean=torch.zeros(3),
        std=torch.ones(3),
        momentum=torch.zeros(4, 3),
        y=y,
        label='rise_flag',
    )
    assert set(sample) >= {'x', 'p', 'mean', 'std', 'momentum', 'y'}
    batch = collate_samples([sample, sample])
    assert batch['x'].shape[0] == 2
    kwargs = unpack_model_input(batch)
    assert kwargs['x'].shape[0] == 2
    assert unpack_target(batch).shape[0] == 2
    tup = as_train_tuple(batch)
    assert len(tup) == 6

    from test_phase1_smoke import _build_synthetic_dataset, _smoke_config
    from pathlib import Path
    cfg_ds = _smoke_config(Path('_phase3_tmp_ckpts'))
    ds, _, _ = _build_synthetic_dataset(cfg_ds)
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
