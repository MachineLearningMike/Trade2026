# dataset_build.py
# Assemble TradingDataset from market series + FSV macros.
# Shared by TRAIN and INFER (avoids TRAIN ↔ INFER import cycle).

import sys
from pathlib import Path

import numpy as np

from data import TradingDataset, get_binance_data, generate_synthetic_data
from task_base import get_task_class, resolve_future_window
from telepred import resolve_past_window


def project_root_dir() -> Path:
    """Repo root: ``…/Trade2026`` when this file lives in ``modules/Ver_…``."""
    from config import project_root_dir as _project_root_dir
    return _project_root_dir()


def get_dataset(config, task=None):
    """
    Build TradingDataset from Binance (or synthetic) market data + FSV macros.

    TRAIN/INFER should pass ``task=get_task_class(config['label'])`` when they
    already hold the principal task; if omitted, it is created here from config.
    """
    market_features = None
    target_feature_id = None
    market_feature_names = None

    if config.get('use_binance_data', False):
        start_s = config.get('binance_start_date')
        end_s = config.get('binance_end_date')
        if start_s and end_s and str(start_s) >= str(end_s):
            # String compare works for 'YYYY-MM-DD HH:MM:SS' layout.
            print(
                f"⚠ binance_start_date ({start_s}) must be strictly before "
                f"binance_end_date ({end_s}). Zero-length range yields T=0."
            )
            sys.exit(1)
        market_features, T, nForget, target_feature_id, market_feature_names = get_binance_data(config)
        if market_features is None or target_feature_id is None:
            print("⚠ Failed to fetch Binance data.")
            sys.exit(1)
        # get_binance_data returns a list of 1-d arrays → stack to (F_m, T)
        if isinstance(market_features, list):
            market_arr = np.stack(market_features, axis=0).astype(np.float32)
        else:
            market_arr = np.asarray(market_features, dtype=np.float32)
        if market_arr.ndim != 2 or market_arr.shape[1] == 0:
            print(
                f"⚠ Market series is empty (shape={getattr(market_arr, 'shape', None)}). "
                f"Check binance_start_date / binance_end_date and data availability."
            )
            sys.exit(1)
    else:
        T = 60 * 24 * config.get('total_synthetic_days', 30)
        F = config.get('num_features', 5)
        target_feature_id = config.get('price_feature_idx', 0)
        market_list = generate_synthetic_data(T, F, target_feature_id)
        market_arr = np.stack(market_list, axis=0).astype(np.float32)
        market_feature_names = [f'feature_{i}' for i in range(F)]
        print(f"Generated synthetic market data: F={F}, T={T}")

    # Add Financial State Vectors (FSV) to features
    from macro_03_query_value_age import get_value_age_array
    macro_features, macro_feature_names = get_value_age_array(config)
    macro_features = np.asarray(macro_features, dtype=np.float32)
    if macro_features.ndim == 1:
        # Legacy empty / corrupt cache: promote to (0, T) or fail clearly
        if macro_features.size == 0:
            macro_features = np.zeros((0, market_arr.shape[1]), dtype=np.float32)
        else:
            raise ValueError(
                f"macro_features must be 2-D (n_channels, T); got 1-D shape "
                f"{macro_features.shape}. Recompute FSV cache (delete "
                f"{config.get('data_dir', 'dataset_cache')}/macro_value_decay_*.npy)."
            )
    if macro_features.ndim != 2:
        raise ValueError(
            f"macro_features must be 2-D (n_channels, T); got shape "
            f"{macro_features.shape}"
        )

    macro_feature_names_full = []
    for name in (macro_feature_names or []):
        macro_feature_names_full.append(name + "_value")
        macro_feature_names_full.append(name + "_decay")

    if macro_features.shape[0] == 0:
        print(
            "⚠ No macro FSV channels loaded (empty feature list or FSV failure). "
            "Continuing with market features only."
        )
    elif macro_features.shape[0] != len(macro_feature_names_full):
        print(
            f"⚠ macro channel count {macro_features.shape[0]} != "
            f"name count {len(macro_feature_names_full)}; "
            f"trimming names/channels to the shorter length."
        )
        n_ch = min(macro_features.shape[0], len(macro_feature_names_full))
        macro_features = macro_features[:n_ch]
        macro_feature_names_full = macro_feature_names_full[:n_ch]

    # Align time lengths (market vs macro caches can differ slightly)
    if macro_features.shape[0] > 0:
        T = min(int(market_arr.shape[1]), int(macro_features.shape[1]))
        if market_arr.shape[1] != macro_features.shape[1]:
            print(
                f"⚠ Aligning time lengths: market T={market_arr.shape[1]}, "
                f"macro T={macro_features.shape[1]} → using T={T}"
            )
        market_arr = market_arr[:, :T]
        macro_features = macro_features[:, :T]
        total_features = np.concatenate([market_arr, macro_features], axis=0)  # (F, T)
    else:
        T = int(market_arr.shape[1])
        total_features = market_arr
    n_market_features = market_arr.shape[0]
    n_macro_features = macro_features.shape[0]

    past_bars = int(resolve_past_window(config))
    future_bars = int(resolve_future_window(config))
    need_T = past_bars + future_bars
    if T < need_T:
        print(
            f"⚠ Time length T={T} is shorter than past+future "
            f"({past_bars}+{future_bars}={need_T}). "
            f"Widen binance_start_date..binance_end_date or reduce "
            f"past_window / future_window."
        )
        sys.exit(1)

    # TradingDataset stacks list/rows on axis=1 → (T, F)
    features_for_dataset = [total_features[i] for i in range(total_features.shape[0])]

    train_ratio = config.get('train_ratio', 0.70)
    val_ratio = config.get('val_ratio', 0.20)
    test_ratio = 1.0 - train_ratio - val_ratio
    split_ratios = (train_ratio, val_ratio, test_ratio)

    if task is None:
        task = get_task_class(config.get('label', 'trend'))
    task.sync_output_dim(config)

    dataset = TradingDataset(
        features=features_for_dataset,
        n_market_features=n_market_features,
        n_macro_features=n_macro_features,
        feature_names=market_feature_names + macro_feature_names_full,
        price_feature_idx=target_feature_id,
        task=task,
        past_window=resolve_past_window(config),
        future_minutes=task.resolve_future_window(config),
        step=config['step'],
        max_accum_deviation=config['max_accum_deviation'],
        device=config['device'],
        data_dir=config.get('data_dir', 'dataset_cache'),
        split_ratios=split_ratios,
        runtime_config=config,
    )
    return dataset, total_features
