#!/usr/bin/env python
"""Phase 6 smoke: sectioned config helpers + apply_overrides / apply_cli_namespace."""
from __future__ import annotations

import traceback


def main() -> int:
    from config import (
        CONFIG_SECTIONS,
        all_sections,
        apply_cli_namespace,
        apply_overrides,
        clone_config,
        filter_active_knobs,
        get_config,
        section_view,
    )

    cfg = get_config()
    assert 'label' in CONFIG_SECTIONS['label']
    assert 'model_type' in CONFIG_SECTIONS['model']
    secs = all_sections(cfg)
    assert set(secs) == set(CONFIG_SECTIONS)
    assert 'past_window' in section_view(cfg, 'data')

    local = clone_config(cfg)
    apply_overrides(local, {'label': 'yield', 'yield_target': 'near', 'epochs': 3})
    assert local['label'] == 'yield'
    assert local['yield_target'] == 'near'
    assert local['epochs'] == 3
    # global untouched
    assert get_config()['label'] == cfg['label']

    class Args:
        epochs = 9
        save_every = 7.0
        clean = False
        past_window = 100
        future_window = 20
        batch_size = 8
        total_days = 2
        step = 5
        max_accum_deviation = 1.0
        min_trend_horizon = 4
        min_yield_percent = 0.1
        model_type = 'regression'
        label = 'trend'
        yield_target = 'both'
        latent_distribution = 'normal'
        lr = 1e-4
        slope_weight = 2.0
        intercept_weight = 3.0
        chance_weight = 1.0
        near_weight = 1.0
        rise_flag_weight = 1.0
        recon_weight = 1.0
        vae_kl_weight = 0.2
        patch_size = 10
        num_layers = 2
        num_heads = 5
        ff_dim_multiplier = 4
        dropout = 0.1
        head_hidden_dim_multiplier = 4
        train_ratio = 0.7
        val_ratio = 0.2
        validate_every = 1
        val_batch_size = 16
        telepred_delay = 0
        use_binance = False
        symbol = 'BTCUSDT'
        no_causal = True
        no_reconstruction = True
        no_look_price = True
        no_validate = True
        no_normalize_window = False
        checkpoint_key = None

    local2 = clone_config(cfg)
    apply_cli_namespace(local2, Args())
    assert local2['epochs'] == 9
    assert local2['label'] == 'trend'
    assert local2['causal'] is False
    assert local2['emit_reconstruction'] is False
    assert local2['look_price'] is False
    assert local2['validate'] is False

    active = filter_active_knobs({'label': 'rise_flag', 'model_type': 'regression',
                                  'rise_flag_loss_weight': 2.0, 'slope_loss_weight': 9.0,
                                  'past_window': 40, 'device': 'cpu'})
    assert 'rise_flag_loss_weight' in active
    # slope is trend-private; may still appear via train/data union — filter keeps
    # data/train always. Private filter is soft documentation; ensure rise key kept.
    assert active['rise_flag_loss_weight'] == 2.0

    print('phase6 config smoke ok')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
