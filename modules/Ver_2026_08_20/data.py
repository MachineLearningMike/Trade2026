# data.py
# Dataset and DataLoader with per-window normalization and caching.
# Label algorithms / target mapping live in labels.py; this module
# owns series fetch, windowing, norm/label caches, and loaders.

import sys
import os
from pathlib import Path

import math
import torch
from torch.utils.data import Dataset, Subset
import threading
import queue
import random
import time
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any
import pickle
import hashlib
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from config import get_config
from datetime import datetime

from Binance import Binance
from labels import (
    LABEL_FITNESS_THRESHOLD,
    LABEL_TREND,
    LABEL_YIELD,
    LABEL_LATENT,
    LABEL_RISE_FLAG,
    VALID_LABEL_MODES,
    LABEL_FULL_DIM_TREND,
    LABEL_FULL_DIM_YIELD,
    LABEL_FULL_DIM_LATENT,
    LABEL_FULL_DIM_RISE_FLAG,
    HORIZON_SEARCH_LONGEST_LINEAR,
    HORIZON_SEARCH_SHORTEST_TO_THRESHOLD,
    VALID_HORIZON_SEARCH_MODES,
    Label_Trend,
    Label_Yield,
    get_label_class,
    label_target_slice,
    label_target_names,
    resolve_future_window,
)
from telepred import (
    signature_fields as telepred_signature_fields,
    resolve_delay as telepred_resolve_delay,
    resolve_past_window,
    valid_bot_times,
    windows_for_sample,
)

# Bump when label *implementation* changes but config params stay the same
# (e.g. yield chance now on raw prices). Used only inside the pickle — never
# hashed into generate_dataset_signature / checkpoint keys.
LABEL_CACHE_ALGO_VERSION = 6  # v6: latent target = future[0] (next bar after history)


def _normalize_label_mode(mode: str) -> str:
    """Map legacy label ids to current ones (trend / yield / latent)."""
    return get_label_class(mode).name


# ═══════════════════════════════════════════════════════════
# CONFIG-ONLY FEATURE LAYOUT + DATASET SIGNATURE
# (no Binance download / no FSV time-series materialization)
# ═══════════════════════════════════════════════════════════

def _interval_to_minutes(interval: str) -> int:
    """Parse Binance-style interval strings: 1m, 3m, 5m, 15m, 1h, 4h, 1d, ..."""
    if not interval:
        return 1
    s = str(interval).strip().lower()
    units = {'m': 1, 'h': 60, 'd': 60 * 24, 'w': 60 * 24 * 7}
    if s[-1] in units:
        return int(s[:-1]) * units[s[-1]]
    # bare integer → minutes
    return int(s)


def estimate_T_from_config(config: dict) -> int:
    """
    Nominal series length T from config alone (no Binance download).

    - Binance/FSV path: floor((end - start) / interval), same rule as
      fsv_03_query_value_age total_minutes for 1m bars.
    - Synthetic path: 60 * 24 * total_synthetic_days.

    Note: actual downloaded T can still differ slightly (gaps, fills,
    market/macro alignment). Label pickle continues to validate real T on load.
    """
    if not config.get('use_binance_data', False):
        days = int(config.get('total_synthetic_days', 30))
        return 60 * 24 * days

    start_s = config.get('binance_start_date')
    end_s = config.get('binance_end_date')
    if not start_s or not end_s:
        return 0

    # Accept "YYYY-MM-DD HH:MM:SS" or date-only
    def _parse(s: str) -> datetime:
        s = str(s).strip()
        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return datetime.fromisoformat(s)

    start = _parse(start_s)
    end = _parse(end_s)
    interval_min = _interval_to_minutes(config.get('binance_interval', '1m'))
    if interval_min <= 0:
        interval_min = 1

    total_minutes = int((end - start).total_seconds() / 60)
    if total_minutes <= 0:
        return 0
    return total_minutes // interval_min


def resolve_feature_layout(config: dict) -> Dict[str, Any]:
    """
    Derive market/macro feature names and price_feature_idx from config alone.

    Mirrors TRAIN.get_dataset naming conventions without loading any series data.
    """
    if config.get('use_binance_data', False):
        symbol_dict = config.get('binance_symbols') or {}
        market_feature_names: List[str] = []
        price_feature_idx = None
        target = config.get('prediction_target', ('UNKNOWN', 'UNKNOWN'))
        target_symbol, target_feature = target[0], target[1]

        for symbol, features in symbol_dict.items():
            for feature in features:
                if symbol == target_symbol and feature == target_feature:
                    price_feature_idx = len(market_feature_names)
                market_feature_names.append(f"{symbol}_{feature}")

        if price_feature_idx is None:
            price_feature_idx = int(config.get('price_feature_idx', 0))
    else:
        n = int(config.get('num_features', 5))
        market_feature_names = [f'feature_{i}' for i in range(n)]
        price_feature_idx = int(config.get('price_feature_idx', 0))

    # Macro base names: get_features(config) = core + extras, include/exclude filters
    from macro_dictionary import get_features
    macro_catalog = get_features(config, apply_filters=True)
    macro_base_names = list(macro_catalog.keys())
    macro_feature_names_full: List[str] = []
    for name in macro_base_names:
        macro_feature_names_full.append(name + "_value")
        macro_feature_names_full.append(name + "_decay")

    feature_names = market_feature_names + macro_feature_names_full
    return {
        'market_feature_names': market_feature_names,
        'macro_base_names': macro_base_names,
        'macro_feature_names_full': macro_feature_names_full,
        'feature_names': feature_names,
        'price_feature_idx': price_feature_idx,
        'n_market_features': len(market_feature_names),
        'n_macro_features': len(macro_feature_names_full),
        'F': len(feature_names),
    }


def generate_dataset_signature(config: dict) -> str:
    """
    Hash everything that defines label-cache / dataset identity from config alone.

    Safe to call without downloading Binance data or materializing FSV arrays.
    Used by --history, checkpoint keys, and TradingDataset label cache.
    """
    layout = resolve_feature_layout(config)
    T = estimate_T_from_config(config)

    past_window = resolve_past_window(config)
    future_window = resolve_future_window(config)

    train_ratio = config.get('train_ratio', 0.70)
    val_ratio = config.get('val_ratio', 0.20)

    key_params = {
        # --- data identity ---
        'use_binance_data': config.get('use_binance_data', False),
        'binance_datatype': config.get('binance_datatype', 'klines'),
        'binance_interval': config.get('binance_interval', '1m'),
        'binance_start_date': config.get('binance_start_date'),
        'binance_end_date': config.get('binance_end_date'),
        'binance_symbols': config.get('binance_symbols'),
        'total_synthetic_days': config.get('total_synthetic_days'),
        'macros_to_exclude': sorted(config.get('macros_to_exclude', [])),
        'macros_to_include': sorted(config.get('macros_to_include') or []),
        'prediction_target': list(config.get('prediction_target', ('UNKNOWN', 'UNKNOWN'))),
        'price_feature_idx': layout['price_feature_idx'],
        'feature_names': list(layout['feature_names']),
        'F': layout['F'],
        'T': T,  # nominal length from config (see estimate_T_from_config)
        'num_shapes': config.get('num_shapes', 4),

        # --- windowing / sampling (TelePred delta1 / delta2) ---
        'past_window': past_window,
        'future_window': future_window,
        'step': config.get('step', 1),

        # --- label algorithm ---
        # Note: algorithm implementation fixes (e.g. yield on raw prices) must
        # NOT be hashed here — they would churn dataset_signature / checkpoint
        # keys. Invalidate stale .pkl content via LABEL_CACHE_ALGO_VERSION in
        # TradingDataset._load_or_compute_labels instead.
        'label': _normalize_label_mode(config.get('label', LABEL_TREND)),
        'min_yield_percent': config.get('min_yield_percent', 0.0),
        # Model target selection for yield (chance / near / both) — must
        # distinguish single-dim chance vs near checkpoints/stats.
        'yield_target': config.get('yield_target', 'both'),
        'min_trend_horizon': config.get('min_trend_horizon', 3),
        # Density mode forces model head width (2=normal, 3=student).
        'latent_distribution': config.get('latent_distribution', 'normal'),
        'regime_use_past': config.get('regime_use_past', True),
        'fitness_threshold': config.get('fitness_threshold', LABEL_FITNESS_THRESHOLD),
        'horizon_search': config.get('horizon_search', HORIZON_SEARCH_LONGEST_LINEAR),
        'output_dim': config.get('output_dim', 3),
        'normalize_window': config.get('normalize_window', True),


        # --- train-split stats ---
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'shuffle_split': config.get('shuffle_split', False),
    }
    # TelePred delay / definition (omit d=0 and telepred_v1 for key stability).
    key_params.update(telepred_signature_fields(config))

    param_string = json.dumps(key_params, sort_keys=True, default=str)
    return hashlib.md5(param_string.encode()).hexdigest()[:8]


# ═══════════════════════════════════════════════════════════
# BINANCE DATA LOADER + KEYED CACHE
# ═══════════════════════════════════════════════════════════

# Uniform short id length for all pipeline keys (binance / labels / FSV / ckpt).
CACHE_KEY_HEX_LEN = 8

BINANCE_CACHE_SUBDIR = 'binance'
# Flat keyed files: candles_<8hex>.npy + candles_<8hex>_meta.json
BINANCE_CANDLES_PREFIX = 'candles_'
# Bump when the on-disk Binance cache contract changes (meta fields, keying,
# or payload layout). Included in the folder hash so old folders miss.
# v4: flat candles_<key>.npy (was binance/<key>/candles.npy).
BINANCE_CACHE_VERSION = 4


def _binance_key_params(config: dict) -> dict:
    """
    Config fields that determine the Binance market-series payload.

    Order of ``binance_symbols`` entries is preserved (feature layout depends
    on it). Label / split / model / window hyperparams are omitted.
    """
    symbol_dict = config.get('binance_symbols') or {}
    # List-of-pairs keeps symbol insertion order under sort_keys=True dumps.
    symbols_ordered = [
        {'symbol': str(sym), 'columns': list(cols)}
        for sym, cols in symbol_dict.items()
    ]
    pred = config.get('prediction_target', ('UNKNOWN', 'UNKNOWN'))
    return {
        'cache_version': BINANCE_CACHE_VERSION,
        'binance_datatype': config.get('binance_datatype', 'klines'),
        'binance_interval': config.get('binance_interval', '1m'),
        'binance_start_date': config.get('binance_start_date'),
        'binance_end_date': config.get('binance_end_date'),
        'binance_symbols': symbols_ordered,
        'prediction_target': list(pred) if pred is not None else None,
    }


def generate_binance_data_key(config: dict) -> str:
    """Hash ``_binance_key_params(config)`` to a short id (8-hex)."""
    param_string = json.dumps(
        _binance_key_params(config), sort_keys=True, default=str
    )
    return hashlib.md5(param_string.encode()).hexdigest()[:CACHE_KEY_HEX_LEN]


def binance_cache_root(config: dict) -> Path:
    """Return ``<data_dir>/binance/`` (flat keyed files live here)."""
    return Path(config.get('data_dir', 'dataset_cache')) / BINANCE_CACHE_SUBDIR


def binance_candles_path(config: dict, key: Optional[str] = None) -> Path:
    """``<data_dir>/binance/candles_<key>.npy``."""
    if key is None:
        key = generate_binance_data_key(config)
    return binance_cache_root(config) / f'{BINANCE_CANDLES_PREFIX}{key}.npy'


def binance_meta_path(config: dict, key: Optional[str] = None) -> Path:
    """``<data_dir>/binance/candles_<key>_meta.json``."""
    if key is None:
        key = generate_binance_data_key(config)
    return binance_cache_root(config) / f'{BINANCE_CANDLES_PREFIX}{key}_meta.json'


def _format_binance_key_params(key_params: dict) -> str:
    symbols = key_params.get('binance_symbols') or []
    sym_txt = ', '.join(
        f"{s.get('symbol')}[{','.join(s.get('columns') or [])}]"
        for s in symbols
    ) or '(none)'
    return (
        f"{key_params.get('binance_datatype')}/"
        f"{key_params.get('binance_interval')} "
        f"{key_params.get('binance_start_date')} → "
        f"{key_params.get('binance_end_date')} | {sym_txt}"
    )


def _save_binance_cache(
    candles_path: Path,
    meta_path: Path,
    key: str,
    key_params: dict,
    features: List[np.ndarray],
    T: int,
    nForget: int,
    target_feature_id: int,
    feature_names: List[str],
) -> Path:
    """Persist stacked candles + sidecar meta next to ``candles_<key>.npy``."""
    candles_path.parent.mkdir(parents=True, exist_ok=True)
    features_arr = np.stack(
        [np.asarray(f, dtype=np.float32) for f in features], axis=0
    )
    if features_arr.ndim != 2 or features_arr.shape[1] != int(T):
        raise ValueError(
            f"Binance cache features shape {features_arr.shape} incompatible "
            f"with T={T}"
        )
    tmp_candles = candles_path.with_suffix('.tmp.npy')
    tmp_meta = meta_path.with_suffix('.tmp.json')

    np.save(tmp_candles, features_arr)
    os.replace(tmp_candles, candles_path)

    # Sidecar meta is required: npy alone lacks target_feature_id, names, and
    # key_params for config-consistency checks on load.
    meta = {
        'binance_key': key,
        'cache_version': BINANCE_CACHE_VERSION,
        'key_params': key_params,
        'T': int(T),
        'F': int(features_arr.shape[0]),
        'nForget': int(nForget) if nForget is not None else 0,
        'target_feature_id': int(target_feature_id),
        'feature_names': list(feature_names),
        'payload': candles_path.name,
        'saved_at': time.time(),
    }
    with open(tmp_meta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, default=str)
    os.replace(tmp_meta, meta_path)
    return candles_path


def _load_binance_cache(candles_path: Path, meta_path: Path, key: str, key_params: dict):
    """
    Load cached Binance market data from flat keyed files.

    Returns:
        (features_list, T, nForget, target_feature_id, feature_names) or None
    """
    if not meta_path.exists() or not candles_path.exists():
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        if meta.get('binance_key') not in (None, key):
            print(
                f"⚠ Binance cache key mismatch "
                f"(meta={meta.get('binance_key')}, expected={key}); ignoring cache."
            )
            return None
        meta_params = meta.get('key_params')
        if not isinstance(meta_params, dict):
            print(
                "⚠ Binance cache missing key_params (old schema); "
                "ignoring cache."
            )
            return None
        if meta.get('cache_version') != BINANCE_CACHE_VERSION:
            print(
                f"⚠ Binance cache_version "
                f"{meta.get('cache_version')!r} != {BINANCE_CACHE_VERSION}; "
                "ignoring cache."
            )
            return None
        expected = json.dumps(key_params, sort_keys=True, default=str)
        actual = json.dumps(meta_params, sort_keys=True, default=str)
        if actual != expected:
            print(
                "⚠ Binance cache key_params mismatch; ignoring cache.\n"
                f"  meta: {_format_binance_key_params(meta_params)}\n"
                f"  want: {_format_binance_key_params(key_params)}"
            )
            return None
        features_arr = np.load(candles_path)
        if features_arr.ndim != 2:
            print(
                f"⚠ Binance cache {candles_path.name} has shape "
                f"{features_arr.shape}; expected 2-D (F, T)."
            )
            return None
        T_meta = int(meta.get('T', features_arr.shape[1]))
        F_meta = int(meta.get('F', features_arr.shape[0]))
        if features_arr.shape != (F_meta, T_meta):
            print(
                f"⚠ Binance cache shape {features_arr.shape} != meta "
                f"(F={F_meta}, T={T_meta}); ignoring cache."
            )
            return None
        feature_names = list(meta.get('feature_names') or [])
        if feature_names and len(feature_names) != features_arr.shape[0]:
            print(
                f"⚠ Binance cache feature_names length "
                f"{len(feature_names)} != F={features_arr.shape[0]}; ignoring."
            )
            return None
        if not feature_names:
            feature_names = [f'feature_{i}' for i in range(features_arr.shape[0])]
        nForget = int(meta.get('nForget', 0))
        target_feature_id = int(meta.get('target_feature_id', 0))
        features = [
            features_arr[i].astype(np.float32, copy=False)
            for i in range(features_arr.shape[0])
        ]
        return features, T_meta, nForget, target_feature_id, feature_names
    except Exception as e:
        print(f"⚠ Failed to load Binance cache {candles_path.name}: {e}")
        return None


def get_binance_data(config):
    """Get data from Binance using the configured parameters.

    Looks up ``<data_dir>/binance/candles_<key>.npy`` (+ ``_meta.json``).
    On a miss, fetches from Binance and saves under those keyed filenames.

    Returns:
        features, T, nForget, target_feature_id, feature_names
        On failure all five values are None.
    """
    key_params = _binance_key_params(config)
    key = generate_binance_data_key(config)
    candles_path = binance_candles_path(config, key)
    meta_path = binance_meta_path(config, key)

    print(f"🔑 Binance data key: {key}")
    print(f"📁 Binance cache file: {candles_path}")
    print(f"   {_format_binance_key_params(key_params)}")

    cached = _load_binance_cache(candles_path, meta_path, key, key_params)
    if cached is not None:
        features, T, nForget, target_feature_id, feature_names = cached
        print(f"✓ Binance cache hit ({candles_path.name})")
        print(f"  Number of features: {len(features)}")
        print(f"  Time points: {T}")
        print(f"  Forgotten points: {nForget}")
        print(f"  Target feature Id: {target_feature_id}")
        return features, T, nForget, target_feature_id, feature_names
    print("⚠ Binance cache miss — fetching from source.")

    if Binance is None:
        print("⚠ Binance not available.")
        return None, None, None, None, None

    try:
        bn = Binance.instantiate(connect=False)

        datatype = config['binance_datatype']
        interval = config['binance_interval']
        start_date = config['binance_start_date']
        end_date = config['binance_end_date']
        symbol_dict = config['binance_symbols']
        target = config['prediction_target']

        # Config strings are UTC wall times (naive). Binance stamps them UTC.
        start = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

        print(f"Fetching Binance data:")
        print(f"  Datatype: {datatype}")
        print(f"  Interval: {interval}")
        print(f"  Start (UTC): {start}")
        print(f"  End (UTC): {end}")
        print(f"  Symbols: {list(symbol_dict.keys())}")

        features, nForget, target_feature_id, feature_names = bn.get_combined_history_3(
            datatype, interval, start, end, symbol_dict, target
        )

        T = features[0].shape[0] if len(features) > 0 else 0
        for feature in features:
            assert feature.shape[0] == T, "All features must have the same number of time points."

        print(f"\nBinance data loaded:")
        print(f"  Number of features: {len(features)}")
        print(f"  Time points: {T}")
        print(f"  Forgotten points: {nForget}")
        print(f"  Target feature Id: {target_feature_id}")

        try:
            saved = _save_binance_cache(
                candles_path,
                meta_path,
                key,
                key_params,
                features,
                T,
                nForget,
                target_feature_id,
                feature_names,
            )
            print(f"✓ Saved Binance data → {saved}")
        except Exception as e:
            print(f"⚠ Failed to save Binance cache: {e}")

        return features, T, nForget, target_feature_id, feature_names

    except Exception as e:
        print(f"⚠ Error fetching Binance data: {e}")
        return None, None, None, None, None


def generate_synthetic_data(T: int, F: int, price_feature_idx: int = 0):
    """Generate synthetic trading data for testing."""
    features = []
    for i in range(F):
        if i == price_feature_idx:
            price = np.cumsum(np.random.randn(T) * 0.5 + 0.1)
            features.append(price.astype(np.float32))
        elif i == 1:
            volume = np.abs(np.random.randn(T) * 100 + 1000)
            features.append(volume.astype(np.float32))
        else:
            features.append(np.random.randn(T).astype(np.float32))
    return features


# ═══════════════════════════════════════════════════════════
# LABEL FIND WRAPPERS (canonical impl in labels.py)
# ═══════════════════════════════════════════════════════════

def find_trend(*args, **kwargs):
    """Delegate to Label_Trend.find (single implementation)."""
    return Label_Trend.find(*args, **kwargs)


def find_yield(
    future,
    current,
    min_yield_percent: float,
    max_hope_future_horizon: Optional[int] = None,
    yield_target: Optional[str] = None,
):
    """
    Yield label: find(p[t+1:t+1+delta2], p[t]).
    Cache path should leave yield_target=None so the full (chance, near)
    pair is stored; model uses target_slice.
    """
    return Label_Yield.find(
        future,
        current,
        min_yield_percent=min_yield_percent,
        max_hope_future_horizon=max_hope_future_horizon,
        yield_target=yield_target,
    )



def find_power_regime(
    future,
    history,
    min_horizon=3,
    max_horizon=30,
    fitness_threshold=1.0,
    exponent_min=0.2,
    exponent_max=3.0,
    exponent_steps=30,
):
    """
    Find the longest approximately power-law future segment.

    Model:
        y = b + a * x^e

    Args:
        future : Tensor (future_window,)
            Future normalized prices.

        history : Tensor (history_window,)
            Past normalized prices used only for fitness normalization.

        min_horizon : int
            Minimum candidate horizon.

        max_horizon : int
            Maximum candidate horizon.

        fitness_threshold : float
            Maximum allowable dimensionless fitness.

        exponent_min : float
            Minimum power exponent.

        exponent_max : float
            Maximum power exponent.

        exponent_steps : int
            Number of exponent candidates.

    Returns:
        horizon : int
        exponent : Tensor
        scale : Tensor
        bias : Tensor
    """

    device = future.device
    dtype = future.dtype
    eps = 1e-8

    max_horizon = min(max_horizon, future.shape[0])

    history_sigma = history.std().clamp(min=eps)

    best_h = min_horizon
    best_e = torch.tensor(1.0, device=device, dtype=dtype)
    best_a = torch.tensor(0.0, device=device, dtype=dtype)
    best_b = torch.tensor(0.0, device=device, dtype=dtype)

    exponents = torch.linspace(
        exponent_min,
        exponent_max,
        exponent_steps,
        device=device,
        dtype=dtype,
    )

    for h in range(min_horizon, max_horizon + 1):

        y = future[:h]

        x = torch.arange(
            h,
            device=device,
            dtype=dtype,
        )

        best_fitness = torch.tensor(
            float("inf"),
            device=device,
            dtype=dtype,
        )

        local_best = None

        for e in exponents:

            # avoid x=0 problems for fractional powers
            z = (x + 1.0) ** e

            z_mean = z.mean()
            y_mean = y.mean()

            dz = z - z_mean

            a = torch.sum(
                dz * (y - y_mean)
            ) / (
                torch.sum(dz * dz) + eps
            )

            b = y_mean - a * z_mean

            prediction = a * z + b

            deviation = torch.sum(
                torch.abs(y - prediction)
            )

            if deviation < best_fitness:
                best_fitness = deviation
                local_best = (e, a, b)

        fitness = best_fitness / history_sigma

        if fitness <= fitness_threshold:

            best_h = h
            best_e, best_a, best_b = local_best

        else:
            break

    return best_h, best_e, best_a, best_b

# ═══════════════════════════════════════════════════════════
# VISUALIZATION FUNCTION
# ═══════════════════════════════════════════════════════════

def _as_float(v) -> float:
    return float(v.item() if hasattr(v, 'item') else v)


def _trend_to_price_space(vec, price_mean: float, price_std: float, default_horizon: int = 0):
    """
    Map a trend vector from price-normalized space to price space.

    Accepts:
      len 3 → (horizon, slope, intercept)
      len 2 → (slope, intercept); horizon = default_horizon
    """
    if hasattr(vec, 'detach'):
        vec = vec.detach().cpu().flatten()
    n = len(vec)
    if n == 3:
        horizon = max(0, int(round(_as_float(vec[0]))))
        slope_norm = _as_float(vec[1])
        intercept_norm = _as_float(vec[2])
    elif n == 2:
        horizon = max(0, int(default_horizon))
        slope_norm = _as_float(vec[0])
        intercept_norm = _as_float(vec[1])
    else:
        raise ValueError(f"Trend vector must have length 2 or 3, got {n}")
    if price_std == 0:
        price_std = 1.0
    slope = slope_norm * price_std
    intercept = intercept_norm * price_std + price_mean
    return horizon, slope, intercept



def visualize_trend(*args, **kwargs):
    """Delegate to labels.visualize_trend (single source; uses cache, no recompute)."""
    from labels import visualize_trend as _visualize_trend
    return _visualize_trend(*args, **kwargs)




# ═══════════════════════════════════════════════════════════
# NORMALIZATION FUNCTION (PUBLIC)
# ═══════════════════════════════════════════════════════════

def normalize_window(
    window: torch.Tensor,
    compute_moments: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalize a window and optionally compute higher-order moments.
    
    Args:
        window: Input window of shape (T, F)
        compute_moments: If True, compute shape moments (skewness, kurtosis, etc.)
    
    Returns:
        normalized: Normalized window (always returned)
        mean: Mean of window (None if compute_moments=False)
        std: Standard deviation of window (None if compute_moments=False)
        shape: Higher-order moments (None if compute_moments=False)
    """
    mean = window.mean(dim=0)     # (F,)
    std = window.std(dim=0)       # (F,)
    std[std == 0] = 1
    
    normalized_x = (window - mean) / std          # (T, F)
    
    if not compute_moments:
        return normalized_x, None, None, None
        
    centered = window - mean   # (T, F)
    
    m3 = (centered ** 3).mean(dim=0)    # (T, F -> F)
    m4 = (centered ** 4).mean(dim=0)    # (T, F -> F)
    m5 = (centered ** 5).mean(dim=0)    # (T, F -> F)
    m6 = (centered ** 6).mean(dim=0)    # (T, F -> F)

    std_sq = std.clamp(min=1e-6)    # (F,)
    
    def signed_root(x, power):
        return torch.sign(x) * (torch.abs(x) ** (1/power))
    
    ms3 = signed_root(m3, 3) / std_sq   # (F,)
    ms4 = signed_root(m4, 4) / std_sq   # (F,)
    ms5 = signed_root(m5, 5) / std_sq   # (F,)
    ms6 = signed_root(m6, 6) / std_sq   # (F,)

    shape = torch.stack([ms3, ms4, ms5, ms6], dim=0)    # (4, F)
    
    return normalized_x, mean, std, shape     # (T, F), (F,), (F,), (4, F)


# ═══════════════════════════════════════════════════════════
# TRADING DATASET
# ═══════════════════════════════════════════════════════════

class TradingDataset(Dataset):
    """Trading dataset with per-window normalization and label caching."""
    
    def __init__(
        self, 
        features: List[np.ndarray],
        n_market_features: int,
        n_macro_features: int,
        feature_names: List[str],
        price_feature_idx: int,
        past_window: Optional[int] = None,
        future_minutes: Optional[int] = None,
        step: int = 1,
        max_accum_deviation: float = 12.0,
        min_trend_horizon: int = 3,
        device: str = 'cpu',
        normalize_window: bool = True,
        data_dir: Optional[str] = None,
        split_ratios: Optional[Tuple[float, float, float]] = None,
        runtime_config: Optional[dict] = None,
    ):
        # Prefer the caller's config (TRAIN / ensemble per-delay) over module defaults.
        config = dict(runtime_config) if runtime_config is not None else get_config()
        self.feature_names = feature_names
        self.n_market_features = n_market_features
        self.n_macro_features = n_macro_features
        self.price_feature_idx = price_feature_idx
        self.step = step
        self.max_accum_deviation = max_accum_deviation
        self.device = device
        self.normalize_window = normalize_window
        self.num_shapes = config['num_shapes']

        self.label_mode = _normalize_label_mode(config.get('label', LABEL_TREND))
        self.min_yield_percent = float(config.get('min_yield_percent', 0.0))

        # TelePred delta1 / delta2
        if past_window is None:
            past_window = resolve_past_window(config)
        self.past_window = int(past_window)
        if future_minutes is None:
            future_minutes = resolve_future_window(config)
        self.future_minutes = int(future_minutes)
        self.future_window = self.future_minutes

        self.min_trend_horizon = int(
            config.get('min_trend_horizon', min_trend_horizon)
        )
        # Trend search / p-vector caps use shared future_window.
        self.max_past_horizon = self.future_window
        self.model_past_len = int(self.future_window)
        self.min_horizon = self.min_trend_horizon
        self.max_horizon = self.future_window
        # Compat aliases for older call sites / viz
        self.max_trend_future_horizon = self.future_window
        self.max_hope_future_horizon = self.future_window

        self.horizon_search = config.get(
            'horizon_search', HORIZON_SEARCH_LONGEST_LINEAR
        )
        if self.horizon_search not in VALID_HORIZON_SEARCH_MODES:
            raise ValueError(
                f"Unknown horizon_search={self.horizon_search!r}; "
                f"expected one of {VALID_HORIZON_SEARCH_MODES}"
            )
        self.fitness_threshold = float(
            config.get('fitness_threshold', LABEL_FITNESS_THRESHOLD)
        )
        self.regime_use_past = bool(config.get('regime_use_past', True))
        # Yield target: both | chance | near
        if self.label_mode == LABEL_YIELD:
            from labels import resolve_yield_target, Label_Yield
            self.yield_target = resolve_yield_target(config)
            forced_dim = Label_Yield.resolve_output_dim(
                yield_target=self.yield_target
            )
            cfg_dim = int(config.get('output_dim', forced_dim))
            if cfg_dim != forced_dim:
                print(
                    f"⚠ label={LABEL_YIELD} yield_target={self.yield_target!r}: "
                    f"forcing output_dim {cfg_dim} → {forced_dim}"
                )
            self.output_dim = forced_dim
            self.latent_dim = self.output_dim
            self._label_slice = label_target_slice(
                self.label_mode,
                self.output_dim,
                yield_target=self.yield_target,
            )
        elif self.label_mode == LABEL_LATENT:
            from labels import Label_Latent, resolve_latent_distribution
            self.yield_target = 'both'
            self.latent_distribution = resolve_latent_distribution(config)
            # Model head width from density mode (2=normal, 3=student).
            forced_dim = Label_Latent.resolve_output_dim(
                config, latent_distribution=self.latent_distribution
            )
            cfg_dim = int(config.get('output_dim', forced_dim))
            if cfg_dim != forced_dim:
                print(
                    f"⚠ label={LABEL_LATENT} latent_distribution="
                    f"{self.latent_distribution!r}: forcing output_dim "
                    f"{cfg_dim} → {forced_dim}"
                )
            self.output_dim = forced_dim
            self.latent_dim = forced_dim  # model head width
            # Cache is always the scalar observation (slice [0:1]).
            self._label_slice = label_target_slice(self.label_mode, self.output_dim)
        elif self.label_mode == LABEL_RISE_FLAG:
            from labels import Label_RiseFlag
            self.yield_target = 'both'
            forced_dim = Label_RiseFlag.resolve_output_dim(config)
            cfg_dim = int(config.get('output_dim', forced_dim))
            if cfg_dim != forced_dim:
                print(
                    f"⚠ label={LABEL_RISE_FLAG}: forcing output_dim "
                    f"{cfg_dim} → {forced_dim}"
                )
            self.output_dim = forced_dim
            self.latent_dim = forced_dim
            self._label_slice = label_target_slice(self.label_mode, self.output_dim)
        else:
            self.yield_target = 'both'
            self.output_dim = int(config.get('output_dim', 3))
            if self.output_dim not in (2, 3):
                raise ValueError(
                    f"output_dim must be 2 or 3 for trend, got {self.output_dim}"
                )
            self.latent_dim = self.output_dim
            self._label_slice = label_target_slice(self.label_mode, self.output_dim)
        self.horizon_mode = config.get('horizon_mode', 'floating')
        self.weight_type = config.get('weight_type', 'exponential')
        self.weight_decay = config.get('weight_decay', 0.9)
        
        if split_ratios is None:
            train_ratio = config.get('train_ratio', 0.70)
            val_ratio = config.get('val_ratio', 0.20)
            test_ratio = 1.0 - train_ratio - val_ratio
            split_ratios = (train_ratio, val_ratio, test_ratio)
        self.split_ratios = split_ratios
        self.shuffle_split = config.get('shuffle_split', False)
        
        if data_dir is None:
            data_dir = config.get('data_dir', 'dataset_cache')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.Total = self._stack_features(features)
        self.T, self.F = self.Total.shape
        
        self.telepred_delay = telepred_resolve_delay(config)
        # Contiguous span only when delay=0; with delay, input and future are gapped.
        self.total_window = (
            self.past_window + self.telepred_delay + self.future_window
        )

        if self.future_window < self.min_trend_horizon:
            self.future_window = self.min_trend_horizon
            self.future_minutes = self.future_window
            self.max_horizon = self.future_window
            self.max_past_horizon = self.future_window
            self.max_trend_future_horizon = self.future_window
            self.max_hope_future_horizon = self.future_window
            self.model_past_len = self.future_window
            print(
                f"⚠ future_window adjusted up to min_trend_horizon "
                f"({self.min_trend_horizon})"
            )

        # Bot times t (TelePred); valid_starts = input-window starts (legacy alias).
        self.valid_t, self.valid_starts = self._get_valid_bot_times_and_starts()

        self.cache_key = self._generate_cache_key(config)
        self.cache_file = self.data_dir / f"labels_{self.cache_key}.pkl"
        
        # Load or compute labels and normalization parameters
        (
            self.label_tensor,
            self.mean_tensor,
            self.std_tensor,
            self.shape_tensor,
            self.stats,
            self.h_past_tensor,
        ) = self._load_or_compute_labels()
        
        self._create_splits()

        # Model-target stats (slice of full label vectors).
        self._label_stat_mean = self.stats['label']['mean'][self._label_slice]
        self._label_stat_std = self.stats['label']['std'][self._label_slice]
        # Binary rise_flag targets stay in {0,1} for BCE — no z-score.
        if self.label_mode == LABEL_RISE_FLAG:
            self._label_stat_mean = torch.zeros_like(self._label_stat_mean)
            self._label_stat_std = torch.ones_like(self._label_stat_std)
        
        # Cache local references for faster __getitem__
        self._mean_tensor = self.mean_tensor
        self._std_tensor = self.std_tensor
        self._shape_tensor = self.shape_tensor
        self._label_tensor = self.label_tensor
        self._h_past_tensor = self.h_past_tensor
        self._valid_t = self.valid_t
        self._valid_starts = self.valid_starts
        self._Total = self.Total
        self._F = self.F
        self._past_window = self.past_window
        self._future_window = self.future_window
        self._telepred_delay = self.telepred_delay
        self._normalize = self.normalize_window
        
        print(f"\nTradingDataset initialized:")
        print(f"  Features: {self.F}, Time steps: {self.T}")
        print(f"  Past window (delta1): {self.past_window} bars")
        print(f"  Future window (delta2): {self.future_window} bars")
        print(f"  TelePred delay d: {self.telepred_delay} "
              f"(input ends at bot_time - d)")
        print(f"  Model p length (recent past): {self.model_past_len}")
        print(f"  Step: {step}")
        print(f"  Label mode: {self.label_mode}")
        if self.label_mode == LABEL_TREND:
            print(f"  Horizon search: {self.horizon_search}")
            print(f"  Fitness threshold: {self.fitness_threshold}")
            print(f"  Regime use past: {self.regime_use_past} "
                  f"(max past/future={self.future_window})")
            print(
                f"  Min trend horizon: {self.min_trend_horizon}, "
                f"Max trend future: {self.future_window}"
            )
        elif self.label_mode == LABEL_LATENT:
            print(
                f"  latent_distribution: "
                f"{getattr(self, 'latent_distribution', config.get('latent_distribution'))}"
            )
            print(f"  model output_dim (density params): {self.output_dim}")
            print(f"  label cache dim (observation): {LABEL_FULL_DIM_LATENT}")
        elif self.label_mode == LABEL_RISE_FLAG:
            print(f"  min_yield_percent (rise threshold stub): {self.min_yield_percent}")
            print(f"  label cache dim (binary flag): {LABEL_FULL_DIM_RISE_FLAG}")
        else:
            print(f"  min_yield_percent: {self.min_yield_percent}")
            print(f"  yield_target: {self.yield_target}")
        _ldist = getattr(self, 'latent_distribution', None)
        _onames = label_target_names(
            self.label_mode,
            self.output_dim,
            yield_target=getattr(self, 'yield_target', 'both'),
            latent_distribution=_ldist,
        )
        print(f"  Model output_dim: {self.output_dim} {_onames}")
        print(f"  Horizon viz mode: {self.horizon_mode}")
        if self.horizon_mode == 'weighted':
            print(f"  Weight type: {self.weight_type}, Decay: {self.weight_decay}")
        print(f"  Max accum deviation (legacy): {max_accum_deviation}")
        print(f"  Normalize window: {self.normalize_window}")
        print(f"  Total samples: {len(self)}")
        print(f"  Cache: {'Loaded' if self.cache_file.exists() else 'Computed'}")
        print(f"  Split ratios - Train: {self.split_ratios[0]:.2f}, "
              f"Val: {self.split_ratios[1]:.2f}, Test: {self.split_ratios[2]:.2f}")
        print(f"  Split sizes - Train: {len(self.train_indices)}, "
              f"Val: {len(self.val_indices)}, Test: {len(self.test_indices)}")
        print(f"\n📊 Dataset Statistics (training set, {len(self.train_indices)} samples):")
        if self.label_mode == LABEL_TREND:
            full_names = ('horizon', 'slope', 'intercept')
        elif self.label_mode == LABEL_LATENT:
            full_names = ('y',)  # scalar observation in cache
        elif self.label_mode == LABEL_RISE_FLAG:
            full_names = ('rise_flag',)
        else:
            full_names = ('chance', 'near')

        print(f"  Full label {full_names}:")
        print(f"    Mean:  {self.stats['label']['mean'].tolist()}")
        print(f"    Std:   {self.stats['label']['std'].tolist()}")
        if self.label_mode == LABEL_LATENT:
            print(f"  Observation y_truth (normalized) {full_names}:")
            print(f"    Mean:  {self._label_stat_mean.tolist()}")
            print(f"    Std:   {self._label_stat_std.tolist()}")
            print(f"  Model head params (not label-normalized): {_onames}")
        else:
            print(
                f"  Model target "
                f"{label_target_names(self.label_mode, self.output_dim, yield_target=getattr(self, 'yield_target', 'both'))}:"
            )
            print(f"    Mean:  {self._label_stat_mean.tolist()}")
            print(f"    Std:   {self._label_stat_std.tolist()}")
        
    def _stack_features(self, features: List[np.ndarray]) -> torch.Tensor:
        shapes = [arr.shape for arr in features]
        if len(set(shapes)) != 1:
            raise ValueError(f"All features must have the same shape. Got: {shapes}")
        stacked = np.stack(features, axis=1)
        return torch.from_numpy(stacked).float()
    
    def _get_valid_bot_times_and_starts(self) -> Tuple[List[int], List[int]]:
        """TelePred bot times t and corresponding model-input window starts."""
        try:
            times = valid_bot_times(
                self.T,
                self.past_window,
                self.future_window,
                delay=self.telepred_delay,
                step=self.step,
            )
        except ValueError as e:
            raise ValueError(
                f"Time series length {self.T} is too short for "
                f"past={self.past_window}, delay={self.telepred_delay}, "
                f"future={self.future_window} ({e})"
            ) from e
        starts = [
            windows_for_sample(
                t, self.telepred_delay, self.past_window, self.future_window
            )["input"].start
            for t in times
        ]
        return times, starts
    
    def _generate_cache_key(self, config) -> str:
        """
        Dataset / label-cache identity. Config-only so callers like --history
        can recompute the same key without downloading data.
        """
        return generate_dataset_signature(config)
    
    def _normalize_window(
        self, 
        window: torch.Tensor, 
        compute_moments: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Internal wrapper for normalize_window."""
        return normalize_window(window, compute_moments)
    
    def _create_splits(self):
        num_samples = len(self.valid_starts)
        # valid_starts is already chronological. Shuffle only if requested;
        # default is a time-ordered cut (train → val → test).
        indices = list(range(num_samples))
        
        if self.shuffle_split:
            random.seed(42)
            random.shuffle(indices)
        
        train_ratio, val_ratio, test_ratio = self.split_ratios
        
        train_end = int(train_ratio * num_samples)
        val_end = train_end + int(val_ratio * num_samples)
        
        self.train_indices = indices[:train_end]
        self.val_indices = indices[train_end:val_end]
        self.test_indices = indices[val_end:]
        
        self.train_dataset = Subset(self, self.train_indices)
        self.val_dataset = Subset(self, self.val_indices)
        self.test_dataset = Subset(self, self.test_indices)
    
    def _compute_labels_and_norm_params(self):
        label_list, mean_list, std_list, shape_list = [], [], [], []
        h_past_list: List[float] = []
        
        for idx in tqdm(range(len(self)), desc="Computing labels and norm params"):
            # TelePred: x delayed by d; Y(t) anchored at bot time t.
            t = self.valid_t[idx]
            win = windows_for_sample(
                t, self.telepred_delay, self.past_window, self.future_window
            )
            x = self.Total[win["input"], :]

            if self.normalize_window:
                _, mean, std, shape = self._normalize_window(x, compute_moments=True)
                mean_list.append(mean)
                std_list.append(std)
                shape_list.append(shape)
            else:
                mean = torch.zeros(self.F)
                std = torch.ones(self.F)
                shape = torch.zeros(self.num_shapes, self.F)
                mean_list.append(mean)
                std_list.append(std)
                shape_list.append(shape)

            future_prices = self.Total[win["future"], self.price_feature_idx]
            # Label history ends at p[t] (not at t-d). Trend/latent use the window;
            # yield uses only the scalar anchor p[t].
            history_prices = self.Total[
                win["label_history"], self.price_feature_idx
            ]
            current_price = self.Total[int(win["bot_time"]), self.price_feature_idx]

            if self.label_mode == LABEL_TREND:
                # Trend labels: free OLS fitness needs a dimensionless price scale,
                # so use the same window z-score as the model input.
                if self.normalize_window:
                    price_mean = mean[self.price_feature_idx]
                    price_std = std[self.price_feature_idx]
                    if float(price_std) == 0.0:
                        price_std = 1.0
                    future_for_label = (future_prices - price_mean) / price_std
                    history_for_label = (history_prices - price_mean) / price_std
                else:
                    future_for_label = future_prices
                    history_for_label = history_prices

                best_h, best_slope, best_intercept, best_h_past = find_trend(
                    future_for_label,
                    history_for_label,
                    min_horizon=self.min_trend_horizon,
                    max_horizon=self.future_window,
                    fitness_threshold=self.fitness_threshold,
                    mode=self.horizon_search,
                    max_past_horizon=self.future_window,
                    use_past=self.regime_use_past,
                )
                # Full cache vector; model may slice to (m,b) when output_dim=2.
                # h_past cached separately for viz (not a model target).
                label = torch.tensor(
                    [best_h, best_slope, best_intercept], dtype=torch.float32
                )
                h_past_list.append(float(best_h_past))
            elif self.label_mode == LABEL_YIELD:
                # Y(t)=label(p[t+1:t+1+delta2], p[t]). MUST use raw prices —
                # window z-score makes p[t]≈0 and blows up chance.
                result = find_yield(
                    future_prices,
                    current_price,
                    min_yield_percent=self.min_yield_percent,
                    max_hope_future_horizon=self.future_window,
                    # full cache always (chance, near); model uses yield_target slice
                )
                if len(result) == 1:
                    chance, near = float(result[0]), 0.0
                else:
                    chance, near = float(result[0]), float(result[1])
                label = torch.tensor([chance, near], dtype=torch.float32)
                h_past_list.append(0.0)
            elif self.label_mode == LABEL_RISE_FLAG:
                # Same TelePred contract as Yield: raw future vs p[t].
                from labels import Label_RiseFlag
                result = Label_RiseFlag.find(
                    future_prices,
                    current_price,
                    min_yield_percent=self.min_yield_percent,
                    max_hope_future_horizon=self.future_window,
                )
                flag = float(result[0] if isinstance(result, (tuple, list)) else result)
                label = torch.tensor([flag], dtype=torch.float32)
                h_past_list.append(0.0)
            elif self.label_mode == LABEL_LATENT:
                # Realized next bar (future[0]) as one draw from the latent
                # price density. Window z-score (from history) keeps the
                # observation scale-free across regimes; find() returns
                # future[0] in that space.
                from labels import Label_Latent
                if self.normalize_window:
                    price_mean = mean[self.price_feature_idx]
                    price_std = std[self.price_feature_idx]
                    if float(price_std) == 0.0:
                        price_std = 1.0
                    future_for_label = (future_prices - price_mean) / price_std
                    history_for_label = (history_prices - price_mean) / price_std
                else:
                    future_for_label = future_prices
                    history_for_label = history_prices
                result = Label_Latent.find(future_for_label, history_for_label)
                if isinstance(result, (tuple, list)):
                    y_obs = float(result[0])
                else:
                    y_obs = float(
                        result.item() if hasattr(result, 'item') else result
                    )
                label = torch.tensor([y_obs], dtype=torch.float32)
                h_past_list.append(0.0)
            else:
                raise ValueError(f"Unknown label mode: {self.label_mode!r}")

            label_list.append(label)

        # Unnormalized labels and x's properties of all data samples - train, validation, and test
        label_tensor = torch.stack(label_list)
        mean_tensor = torch.stack(mean_list)
        std_tensor = torch.stack(std_list)
        shape_tensor = torch.stack(shape_list)
        h_past_tensor = torch.tensor(h_past_list, dtype=torch.float32)
        
        # Create splits first
        self._create_splits()
        
        # Get training indices
        train_indices = self.train_indices
        
        # Extract training data
        label_train = label_tensor[train_indices]
        mean_train = mean_tensor[train_indices]
        std_train = std_tensor[train_indices]
        shape_train = shape_tensor[train_indices]
        
        # Compute statistics from training set only
        stats = {
            'label': {
                'mean': label_train.mean(dim=0),
                'std': label_train.std(dim=0),
            },
            'mean': {
                'mean': mean_train.mean(dim=0),
                'std': mean_train.std(dim=0),
            },
            'std': {
                'mean': std_train.mean(dim=0),
                'std': std_train.std(dim=0),
            },
            'shape': {
                'mean': shape_train.mean(dim=0),
                'std': shape_train.std(dim=0),
            },
        }
        
        return (
            label_tensor, mean_tensor, std_tensor, shape_tensor,
            stats, h_past_tensor,
        )
    
    def _load_or_compute_labels(self):
        if self.cache_file.exists():
            print(f"Loading cached labels from: {self.cache_file}")
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                shape_ok = (
                    cached_data.get('num_samples') == len(self) and
                    cached_data.get('F') == self.F and
                    cached_data.get('T') == self.T and
                    cached_data.get('past_window') == self.past_window and
                    cached_data.get('future_window') == self.future_window
                )
                # Implementation-only version (not part of dataset_signature).
                # Missing key → treat as v1 (pre-raw-yield fix).
                algo_ver = int(cached_data.get('label_algo_version', 1))
                algo_ok = algo_ver == LABEL_CACHE_ALGO_VERSION

                if shape_ok and algo_ok:
                    label_tensor = cached_data['label_tensor']
                    mean_tensor = cached_data['mean_tensor']
                    std_tensor = cached_data['std_tensor']
                    shape_tensor = cached_data['shape_tensor']
                    stats = cached_data['stats']
                    h_past_tensor = cached_data.get('h_past_tensor')
                    if h_past_tensor is None:
                        h_past_tensor = torch.zeros(
                            label_tensor.shape[0], dtype=torch.float32
                        )

                    print(f"✓ Cache loaded successfully! ({label_tensor.shape[0]} samples)")
                    return (
                        label_tensor, mean_tensor, std_tensor, shape_tensor,
                        stats, h_past_tensor,
                    )

                if not shape_ok:
                    print("⚠ Cache mismatch (shapes/windows). Recomputing...")
                else:
                    print(
                        f"⚠ Cache label_algo_version={algo_ver} "
                        f"!= current {LABEL_CACHE_ALGO_VERSION}. "
                        f"Recomputing labels (dataset key unchanged)."
                    )
            except Exception as e:
                print(f"⚠ Failed to load cache: {e}. Recomputing...")
        
        print("Computing labels and normalization parameters...")
        (
            label_tensor, mean_tensor, std_tensor, shape_tensor,
            stats, h_past_tensor,
        ) = self._compute_labels_and_norm_params()
        
        try:
            cache_data = {
                'label_tensor': label_tensor,
                'mean_tensor': mean_tensor,
                'std_tensor': std_tensor,
                'shape_tensor': shape_tensor,
                'h_past_tensor': h_past_tensor,
                'stats': stats,
                'num_samples': len(self),
                'F': self.F,
                'T': self.T,
                'past_window': self.past_window,
                'future_window': self.future_window,
                'step': self.step,
                'max_accum_deviation': self.max_accum_deviation,
                'min_trend_horizon': self.min_trend_horizon,

                'fitness_threshold': self.fitness_threshold,
                'horizon_search': self.horizon_search,
                'label': self.label_mode,
                'min_yield_percent': self.min_yield_percent,
                'normalize_window': self.normalize_window,
                'price_feature_idx': self.price_feature_idx,
                'cache_key': self.cache_key,
                'label_algo_version': LABEL_CACHE_ALGO_VERSION,
                'timestamp': time.time(),
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✓ Cache saved to: {self.cache_file}")
            print(f"  Cache size: {self.cache_file.stat().st_size / 1024:.2f} KB")
        except Exception as e:
            print(f"⚠ Failed to save cache: {e}")
        
        return (
            label_tensor, mean_tensor, std_tensor, shape_tensor,
            stats, h_past_tensor,
        )
    
    def visualize_trends(
        self,
        num_samples: int = 6,
        figsize: Tuple[int, int] = (15, 10),
        interactive: bool = True,
    ):
        """
        Explicitly visualize linear trends for the dataset.

        Args:
            num_samples: Samples per page
            figsize: Figure size
            interactive: If True, Next/Quit buttons page through unlimited batches
        """
        if self.label_tensor.shape[0] == 0:
            print("⚠ No labels available for visualization.")
            return

        print("\n" + "=" * 60)
        print("VISUALIZING LINEAR TRENDS")
        if interactive:
            print("  Interactive: Next = new random page, Quit = exit")
        print("=" * 60)

        visualize_trend(
            total_tensor=self.Total,
            valid_starts=self.valid_starts,
            label_tensor=self.label_tensor,
            mean_tensor=self.mean_tensor,
            std_tensor=self.std_tensor,
            price_feature_idx=self.price_feature_idx,
            past_window=self.past_window,
            future_window=self.future_window,
            mode=self.horizon_mode,
            num_samples=num_samples,
            figsize=figsize,
            interactive=interactive,
            h_past_tensor=self.h_past_tensor,
            min_horizon=self.min_trend_horizon,
            max_horizon=self.future_window,
            max_past_horizon=self.future_window,
            fitness_threshold=self.fitness_threshold,
            horizon_search=self.horizon_search,
            regime_use_past=self.regime_use_past,
            normalize_prices=self.normalize_window,
        )

    def visualize_feature(self, fid, feature, feature_name, time_interval_minutes=1):
        """
        Visualize a specific feature over time with x-axis in days.
        
        Args:
            fid: Feature index
            feature: 1-d numpy array of the feature values over time
            time_interval_minutes: Time interval between samples (default: 1 minute)
        """
        
        mean = feature.mean()
        std = feature.std()
        feature = (feature - mean) / (std + 1e-8)
        
        # Calculate time in days
        total_minutes = len(feature) * time_interval_minutes
        total_days = total_minutes / (60 * 24)
        
        # Create x-axis in days
        x_days = np.linspace(0, total_days, len(feature))
        
        plt.figure(figsize=(15, 5))
        plt.plot(x_days, feature, label=f'Feature {fid}', color='blue')
        plt.title(f'Feature {fid}: {feature_name} Over Time', fontsize=14)
        plt.xlabel(f'Days (interval: {time_interval_minutes} min)', fontsize=12)
        plt.ylabel('Value (Z-score normalized)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        """
        Normalize a model-target label vector.

        y must already be sliced to model target size:
          trend  → (h,m,b) or (m,b)
          yield → (chance, near)
        """
        mean = self._label_stat_mean
        std = self._label_stat_std
        if y.shape[-1] != mean.shape[-1]:
            raise ValueError(
                f"y last dim {y.shape[-1]} != expected output_dim {mean.shape[-1]}"
            )
        return (y - mean.to(y.device)) / (std.to(y.device) + 1e-8)

    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse of normalize_y for label-cache target vectors (not density params)."""
        device = y.device
        label_mean = self._label_stat_mean.to(device)
        label_std = self._label_stat_std.to(device)
        if y.shape[-1] != label_mean.shape[-1]:
            raise ValueError(
                f"y last dim {y.shape[-1]} != expected output_dim {label_mean.shape[-1]}"
            )
        return y * label_std + label_mean
    
    def clear_cache(self):
        if self.cache_file.exists():
            self.cache_file.unlink()
            print(f"✓ Cache cleared: {self.cache_file}")
    
    @classmethod
    def clear_all_caches(cls, data_dir: Optional[str] = None):
        if data_dir is None:
            config = get_config()
            data_dir = config.get('data_dir', 'dataset_cache')
        cache_path = Path(data_dir)
        if cache_path.exists():
            import shutil
            shutil.rmtree(cache_path)
            print(f"✓ All caches cleared from: {cache_path}")
        else:
            print(f"✓ No cache directory found at: {cache_path}")
    
    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimized __getitem__ with cached references and view() instead of unsqueeze()."""
        t = self._valid_t[idx]
        win = windows_for_sample(
            t, self._telepred_delay, self._past_window, self._future_window
        )
        x = self._Total[win["input"], :]
        
        if self._normalize:
            mean = self._mean_tensor[idx]       # (F,)
            std = self._std_tensor[idx].clone()  # clone: do not mutate cache
            std[std == 0] = 1
            x_normalized = (x - mean) / std     # (T, F)
            shape = self._shape_tensor[idx]     # (num_shapes, F)
        else:
            x_normalized = x
            mean = torch.zeros(self._F)
            std = torch.ones(self._F)
            shape = torch.zeros(self.num_shapes, self._F)
        
        # Full label cache is always (horizon, slope, intercept); slice for model.
        y_full = self._label_tensor[idx]
        y = y_full[self._label_slice]

        # Normalize label, mean, std, and shape using dataset statistics
        y_normalized = self.normalize_y(y)

        # p: last model_past_len bars of normalized target price.
        # Length must match model.n_max_past_horizon (mode-dependent).
        recent_prices = x_normalized[-self.model_past_len:, self.price_feature_idx]
        
        # Mean: (F,)
        mean_normalized = (mean - self.stats['mean']['mean']) / (self.stats['mean']['std'] + 1e-8)
        
        # Std: (F,)
        std_normalized = (std - self.stats['std']['mean']) / (self.stats['std']['std'] + 1e-8)
        
        # Shape: (num_shapes, F)
        shape_normalized = (shape - self.stats['shape']['mean']) / (self.stats['shape']['std'] + 1e-8)
        
        return x_normalized, recent_prices, mean_normalized, std_normalized, shape_normalized, y_normalized


# ═══════════════════════════════════════════════════════════
# OPTIMIZED THREADED DATA LOADER
# ═══════════════════════════════════════════════════════════

class ThreadedDataLoader:
    """DataLoader with continuous prefetch thread (doesn't restart per batch)."""
    
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.current_epoch_indices = None
        self.current_batch_start = 0
        self.epoch = 0
        
        # Larger queue to buffer batches
        self.prefetch_queue = queue.Queue(maxsize=10)
        self.prefetch_thread = None
        self.running = False
        self.iterator = None
        self.load_start_time = None
        
    def start_epoch(self, epoch, resume_indices=None, resume_batch_start=0):
        """Start a new epoch."""
        # Stop any existing thread
        self._stop_prefetch()
        
        # Clear the queue
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except:
                break
        
        self.epoch = epoch
        
        if resume_indices is not None:
            self.current_epoch_indices = resume_indices
        else:
            self.current_epoch_indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.seed(42 + epoch)
                random.shuffle(self.current_epoch_indices)
        
        self.current_batch_start = resume_batch_start
        self.running = True
        
        # Create iterator BEFORE starting thread
        self._create_iterator()
        
        # Start prefetch thread
        self._start_prefetch()
    
    def _create_iterator(self):
        """Create iterator with safety checks."""
        if self.current_epoch_indices is None:
            self.current_epoch_indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.seed(42 + self.epoch)
                random.shuffle(self.current_epoch_indices)
        
        # Safety: ensure batch_start is within bounds
        start = self.current_batch_start * self.batch_size
        if start >= len(self.current_epoch_indices):
            print(f"⚠ Batch start {start} exceeds indices length {len(self.current_epoch_indices)}. Resetting.")
            self.current_batch_start = 0
            start = 0
        
        remaining_indices = self.current_epoch_indices[start:]
        
        if len(remaining_indices) == 0:
            print(f"⚠ No remaining indices. Resetting to full epoch.")
            self.current_batch_start = 0
            remaining_indices = self.current_epoch_indices
        
        self.iterator = self._batch_generator(remaining_indices)
    
    def _batch_generator(self, indices):
        """Generator that yields batches."""
        # dataset_get = self.dataset.__getitem__
        batch_size = self.batch_size
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            x_list, p_list, mean_list, std_list, shape_list, y_list = [], [], [], [], [], []
            
            for idx in batch_indices:
                try:
                    x, p, mean, std, shape, y = self.dataset.__getitem__(idx)
                    x_list.append(x)
                    p_list.append(p)
                    mean_list.append(mean)
                    std_list.append(std)
                    shape_list.append(shape)
                    y_list.append(y)
                except IndexError as e:
                    print(f"⚠ IndexError for idx={idx}, dataset length={len(self.dataset)}, indices length={len(self.current_epoch_indices)}")
                    raise
            
            x_batch = torch.stack(x_list)
            p_batch = torch.stack(p_list)
            mean_batch = torch.stack(mean_list)
            std_batch = torch.stack(std_list)
            shape_batch = torch.stack(shape_list)
            y_batch = torch.stack(y_list)

            # (B, T, F), (B, model_past_len), (B, F), (B, F), (B, 4, F), (B, out).
            # All normalized and ready for model input.
            yield x_batch, p_batch, mean_batch, std_batch, shape_batch, y_batch
    
    def get_total_batches_in_epoch(self):
        """Get total number of batches in current epoch."""
        if self.current_epoch_indices is None:
            return math.ceil(len(self.dataset) / self.batch_size)
        return max(1, math.ceil(len(self.current_epoch_indices) / self.batch_size))
    
    def get_state(self):
        """Get current state for resuming."""
        return {
            'epoch_indices': self.current_epoch_indices,
            'batch_start': self.current_batch_start,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'epoch': self.epoch,
        }
    
    def restore_state(self, state):
        """Restore state from checkpoint."""
        self.current_epoch_indices = state['epoch_indices']
        self.current_batch_start = state['batch_start']
        self.batch_size = state['batch_size']
        self.shuffle = state['shuffle']
        self.epoch = state.get('epoch', 0)
    
    def _start_prefetch(self):
        """Start the prefetch thread (only if not already running)."""
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            return
        
        self.load_start_time = time.time()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    
    def _prefetch_worker(self):
        """Continuous worker that prefetches ALL batches."""
        try:
            # Safety: ensure iterator exists
            if self.iterator is None:
                print("⚠ Iterator is None in prefetch worker. Creating...")
                self._create_iterator()
            
            while self.running:
                try:
                    # Safety: check if iterator exists
                    if self.iterator is None:
                        print("⚠ Iterator became None. Recreating...")
                        self._create_iterator()
                        if self.iterator is None:
                            self.prefetch_queue.put(('error', RuntimeError("Iterator is None"), 0))
                            return
                    
                    # Measure only batch assembly — NOT queue.put() blocking
                    # (put blocks when the queue is full; that is consumer backpressure,
                    # not data-loading cost, and must not be counted as load_time).
                    t_build0 = time.perf_counter()
                    batch = next(self.iterator)
                    load_time = time.perf_counter() - t_build0
                    self.prefetch_queue.put(('data', batch, load_time))
                except StopIteration:
                    break
                except Exception as e:
                    self.prefetch_queue.put(('error', e, 0))
                    return
            
            if self.running:
                self.prefetch_queue.put(('stop', None, 0))
            self.running = False
            
        except Exception as e:
            self.prefetch_queue.put(('error', e, 0))
    
    def _stop_prefetch(self):
        """Stop the prefetch thread."""
        self.running = False
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=0.5)
        self.prefetch_thread = None
        
        # Clear the queue
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except:
                break
    
    def get_next_batch(self):
        """
        Get the next batch from the prefetch queue.

        Blocks until a batch is ready, the epoch ends, or the worker errors.

        Returns:
            (batch, wait_time_s, load_time_s)
            batch is None at end of epoch.
            wait_time_s: time this call blocked on the queue (consumer stall).
            load_time_s: producer-side batch build time only (not put-wait).
        """
        wait_start = time.perf_counter()
        while True:
            try:
                status, data, load_time = self.prefetch_queue.get(timeout=0.5)
            except queue.Empty:
                if self.running or (
                    self.prefetch_thread is not None and self.prefetch_thread.is_alive()
                ):
                    continue
                wait_time = time.perf_counter() - wait_start
                return None, wait_time, 0.0
            except Exception as e:
                print(f"🔍 DEBUG: queue.get() raised: {e}")
                wait_time = time.perf_counter() - wait_start
                return None, wait_time, 0.0

            wait_time = time.perf_counter() - wait_start

            if status == 'stop':
                self.running = False
                return None, wait_time, float(load_time or 0.0)
            if status == 'error':
                raise data

            # Track resume position (batches consumed this epoch)
            self.current_batch_start += 1
            return data, wait_time, float(load_time or 0.0)
    
    def __del__(self):
        """Cleanup on deletion."""
        self._stop_prefetch()
