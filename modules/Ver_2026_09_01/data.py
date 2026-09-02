# data.py
# Dataset and DataLoader with per-window normalization and caching.
# Task APIs live in task_base / Task_*.py; this module
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
from typing import List, Tuple, Optional, Dict, Any, Mapping, Sequence, Union
import pickle
import hashlib
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from config import get_config
from datetime import datetime

from Binance import Binance
from task_base import (
    LABEL_TREND,
    get_task_class,
    resolve_future_window,
)
from telepred import (
    signature_fields as telepred_signature_fields,
    resolve_delay as telepred_resolve_delay,
    resolve_past_window,
    valid_bot_times,
    windows_for_sample,
)
# Bump when label *implementation* / cache format changes but config params
# stay the same. Used only inside the pickle — never hashed into
# generate_dataset_signature / checkpoint keys.
# v7: past-regime aux removed (future-only).
# v8: Task.build_raw_y(t, total_x, total_p, config) owns price windowing.
# v9: flat prefixed cache keys x.* / y.* / inter.* / intra.* (task not label).
# v10: inter.normalization_layers (config normalize_window removed).
LABEL_CACHE_ALGO_VERSION = 10


def _pack_label_cache(
    *,
    mean_tensor: torch.Tensor,
    std_tensor: torch.Tensor,
    momentum_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    stats: dict,
    dataset: 'TradingDataset',
    config: dict,
) -> dict:
    """
    Flat namespaced label-cache payload:

      x.*      — FeaturePrep moments (mean / std / momentum)
      y.*      — raw task targets
      inter.*  — shared dataset / window identity (task-agnostic)
      intra.*  — task-private meta from Task.cache_intra_fields(config)
    """
    task = dataset.task
    out = {
        'cache_algo_version': LABEL_CACHE_ALGO_VERSION,
        # --- x_tensors ---
        'x.mean_tensor': mean_tensor,
        'x.std_tensor': std_tensor,
        'x.momentum_tensor': momentum_tensor,
        # --- y_tensors ---
        'y.label_tensor': label_tensor,
        # --- inter_task (shared) ---
        'inter.stats': stats,
        'inter.num_samples': len(dataset),
        'inter.F': dataset.F,
        'inter.T': dataset.T,
        'inter.past_window': dataset.past_window,
        'inter.future_window': dataset.future_window,
        'inter.step': dataset.step,
        'inter.max_accum_deviation': dataset.max_accum_deviation,
        'inter.task': dataset.label_mode,
        'inter.normalization_layers': dataset.normalization_layers,
        'inter.price_feature_idx': dataset.price_feature_idx,
        'inter.cache_key': dataset.cache_key,
        'inter.timestamp': time.time(),
    }
    for key, value in task.cache_intra_fields(config).items():
        out[f'intra.{key}'] = value
    return out


def _unpack_label_cache(cached_data: dict):
    """Read v9 prefixed cache; raises KeyError if required keys missing."""
    return (
        cached_data['y.label_tensor'],
        cached_data['x.mean_tensor'],
        cached_data['x.std_tensor'],
        cached_data['x.momentum_tensor'],
        cached_data['inter.stats'],
    )


def _normalize_label_mode(mode: str) -> str:
    """Validate and return canonical task id."""
    return get_task_class(str(mode).strip().lower()).name


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
        'num_momentum': config.get('num_momentum', 4),

        # --- windowing / sampling (TelePred delta1 / delta2) ---
        'past_window': past_window,
        'future_window': future_window,
        'step': config.get('step', 1),

        # --- task id + shared FeaturePrep / head width ---
        # Task-private knobs come only via Task.signature_fields(config).
        # Implementation fixes invalidate via LABEL_CACHE_ALGO_VERSION, not here.
        'label': _normalize_label_mode(config.get('label', LABEL_TREND)),
        'output_dim': config.get('output_dim', 3),
        # FeaturePrep depth (one-layer | two-layer)
        'normalization_layers': resolve_normalization_layers(config),

        # --- train-split stats ---
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'shuffle_split': config.get('shuffle_split', False),
    }
    # TelePred delay / definition (omit d=0 and telepred_v1 for key stability).
    key_params.update(telepred_signature_fields(config))
    # Task-owned signature fragments (private knobs stay inside Task.*).
    label_mode = key_params['label']
    key_params.update(get_task_class(label_mode).signature_fields(config))

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
        task,
        past_window: Optional[int] = None,
        future_minutes: Optional[int] = None,
        step: int = 1,
        max_accum_deviation: float = 12.0,
        device: str = 'cpu',
        data_dir: Optional[str] = None,
        split_ratios: Optional[Tuple[float, float, float]] = None,
        runtime_config: Optional[dict] = None,
    ):
        # Prefer the caller's config (TRAIN / ensemble per-delay) over module defaults.
        config = dict(runtime_config) if runtime_config is not None else get_config()
        self._runtime_config = config
        self.feature_names = feature_names
        self.n_market_features = n_market_features
        self.n_macro_features = n_macro_features
        self.price_feature_idx = price_feature_idx
        self.step = step
        self.max_accum_deviation = max_accum_deviation
        self.device = device
        self.num_momentum = config['num_momentum']
        self.normalization_layers = resolve_normalization_layers(config)
        # Runtime bool for window z-score / price scaling (not a config key).
        self.normalize_window = self.normalization_layers == 'two-layer'
        config['normalization_layers'] = self.normalization_layers

        if task is None:
            raise TypeError(
                'TradingDataset requires task= (Task class from get_task_class)'
            )
        self.task = task
        self.label_mode = str(getattr(task, 'name', '') or task.__name__)
        config['label'] = self.label_mode

        # TelePred delta1 / delta2 (Task may clamp future via resolve_future_window).
        if past_window is None:
            past_window = resolve_past_window(config)
        self.past_window = int(past_window)
        if future_minutes is None:
            future_minutes = int(self.task.resolve_future_window(config))
        else:
            future_minutes = self.task.resolve_future_window({
                **config,
                'future_window': int(future_minutes),
            })
        self.future_minutes = int(future_minutes)
        self.future_window = self.future_minutes
        self.model_past_len = int(self.task.resolve_model_past_len({
            **config,
            'future_window': self.future_window,
        }))

        # Task-owned output binding.
        _bound = self.task.bind_dataset_outputs(config)
        self.yield_target = _bound.get('yield_target', 'both')
        if 'latent_distribution' in _bound:
            self.latent_distribution = _bound['latent_distribution']
        self.output_dim = int(_bound['output_dim'])
        self.latent_dim = int(_bound.get('latent_dim', self.output_dim))
        self._label_slice = _bound['label_slice']
        self._label_normalize_policy = _bound.get(
            'normalize_policy', 'zscore'
        )
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

        # Bot times t (TelePred); valid_starts = input-window starts (legacy alias).
        self.valid_t, self.valid_starts = self._get_valid_bot_times_and_starts()

        self.cache_key = self._generate_cache_key(config)
        self.cache_file = self.data_dir / f"labels_{self.cache_key}.pkl"

        # Load or compute labels and normalization parameters
        (
            self.label_tensor,
            self.mean_tensor,
            self.std_tensor,
            self.momentum_tensor,
            self.stats,
        ) = self._load_or_compute_labels()

        self._create_splits()

        # Model-target stats (slice of full label vectors). Scaling into loss
        # space is Task.normalize — data.py only dispatches.
        self._label_stat_mean = self.stats['label']['mean'][self._label_slice]
        self._label_stat_std = self.stats['label']['std'][self._label_slice]

        # Cache local references for faster __getitem__
        self._mean_tensor = self.mean_tensor
        self._std_tensor = self.std_tensor
        self._momentum_tensor = self.momentum_tensor
        self._label_tensor = self.label_tensor
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
        print(
            f"  Label y normalize policy: "
            f"{getattr(self, '_label_normalize_policy', '?')} "
            f"(Task.normalize; data.py only dispatches)"
        )
        for line in self.task.dataset_banner_lines(
            bound=_bound, config=config, dataset=self
        ):
            print(line)
        _ldist = getattr(self, 'latent_distribution', None)
        _onames = self.task.target_names(
            self.output_dim,
            yield_target=getattr(self, 'yield_target', 'both'),
            latent_distribution=_ldist,
        )
        print(f"  Model output_dim: {self.output_dim} {_onames}")
        print(f"  Horizon viz mode: {self.horizon_mode}")
        if self.horizon_mode == 'weighted':
            print(f"  Weight type: {self.weight_type}, Decay: {self.weight_decay}")
        print(f"  Max accum deviation (legacy): {max_accum_deviation}")
        print(f"  Normalization layers: {self.normalization_layers}")
        print(f"  Total samples: {len(self)}")
        print(f"  Cache: {'Loaded' if self.cache_file.exists() else 'Computed'}")
        print(f"  Split ratios - Train: {self.split_ratios[0]:.2f}, "
              f"Val: {self.split_ratios[1]:.2f}, Test: {self.split_ratios[2]:.2f}")
        print(f"  Split sizes - Train: {len(self.train_indices)}, "
              f"Val: {len(self.val_indices)}, Test: {len(self.test_indices)}")
        print(f"\n📊 Dataset Statistics (training set, {len(self.train_indices)} samples):")
        cache_dim = int(self.label_tensor.shape[-1])
        try:
            full_names = self.task.target_names(
                cache_dim,
                yield_target='both',
                latent_distribution=_ldist,
            )
        except Exception:
            full_names = tuple(f'y{i}' for i in range(cache_dim))

        print(f"  Full label {full_names}:")
        print(f"    Mean:  {self.stats['label']['mean'].tolist()}")
        print(f"    Std:   {self.stats['label']['std'].tolist()}")
        print(f"  Model target {_onames}:")
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
        label_list, mean_list, std_list, momentum_list = [], [], [], []
        task = self.task
        total_x = self.Total
        total_p = self.Total[:, self.price_feature_idx]
        # Shared window / FeaturePrep fields Task.build_raw_y reads from config.
        config = dict(self._runtime_config)
        config['past_window'] = self.past_window
        config['future_window'] = self.future_window
        config['telepred_delay'] = self.telepred_delay
        config['price_feature_idx'] = self.price_feature_idx
        config['normalization_layers'] = self.normalization_layers

        for idx in tqdm(range(len(self)), desc="Computing labels and norm params"):
            # TelePred: feature window delayed by d; Y(t) anchored at bot time t.
            t = self.valid_t[idx]
            win = windows_for_sample(
                t, self.telepred_delay, self.past_window, self.future_window
            )
            x = total_x[win['input'], :]

            if self.normalize_window:
                _, mean, std, momentum = task.normalize_x(x, compute_moments=True)
            else:
                mean = torch.zeros(self.F)
                std = torch.ones(self.F)
                momentum = torch.zeros(self.num_momentum, self.F)
            mean_list.append(mean)
            std_list.append(std)
            momentum_list.append(momentum)

            label = task.build_raw_y(
                t, total_x, total_p, config, mean=mean, std=std
            )
            label_list.append(label)

        # Unnormalized labels and x's properties of all data samples
        label_tensor = torch.stack(label_list)
        mean_tensor = torch.stack(mean_list)
        std_tensor = torch.stack(std_list)
        momentum_tensor = torch.stack(momentum_list)

        # Create splits first
        self._create_splits()

        train_indices = self.train_indices
        label_train = label_tensor[train_indices]
        mean_train = mean_tensor[train_indices]
        std_train = std_tensor[train_indices]
        momentum_train = momentum_tensor[train_indices]

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
            'momentum': {
                'mean': momentum_train.mean(dim=0),
                'std': momentum_train.std(dim=0),
            },
        }

        return (
            label_tensor, mean_tensor, std_tensor, momentum_tensor, stats,
        )

    def _load_or_compute_labels(self):
        # Label cache key = dataset_signature (NOT epochs / lr / batch_size).
        print(f"Label cache key: {self.cache_key}")
        print(f"  path: {self.cache_file}")
        if self.cache_file.exists():
            print(f"Loading cached labels from: {self.cache_file}")
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                # Implementation-only version (not part of dataset_signature).
                # Missing key → treat as v1 (pre-raw-yield / pre-prefix format).
                algo_ver = int(
                    cached_data.get(
                        'cache_algo_version',
                        cached_data.get('label_algo_version', 1),
                    )
                )
                algo_ok = algo_ver == LABEL_CACHE_ALGO_VERSION
                prefixed = (
                    'inter.num_samples' in cached_data
                    and 'y.label_tensor' in cached_data
                    and 'x.mean_tensor' in cached_data
                )
                shape_ok = prefixed and (
                    cached_data.get('inter.num_samples') == len(self) and
                    cached_data.get('inter.F') == self.F and
                    cached_data.get('inter.T') == self.T and
                    cached_data.get('inter.past_window') == self.past_window and
                    cached_data.get('inter.future_window') == self.future_window
                )

                if shape_ok and algo_ok:
                    (
                        label_tensor, mean_tensor, std_tensor,
                        momentum_tensor, stats,
                    ) = _unpack_label_cache(cached_data)

                    print(f"✓ Cache loaded successfully! ({label_tensor.shape[0]} samples)")
                    return (
                        label_tensor, mean_tensor, std_tensor, momentum_tensor,
                        stats,
                    )

                if not prefixed or not algo_ok:
                    print(
                        f"⚠ Cache format/version "
                        f"(algo={algo_ver}, prefixed={prefixed}) "
                        f"!= current {LABEL_CACHE_ALGO_VERSION}. "
                        f"Recomputing labels (dataset key unchanged)."
                    )
                else:
                    print("⚠ Cache mismatch (shapes/windows). Recomputing...")
                    print(
                        f"  cached: samples={cached_data.get('inter.num_samples')} "
                        f"F={cached_data.get('inter.F')} "
                        f"T={cached_data.get('inter.T')} "
                        f"past={cached_data.get('inter.past_window')} "
                        f"future={cached_data.get('inter.future_window')}"
                    )
                    print(
                        f"  now:    samples={len(self)} F={self.F} T={self.T} "
                        f"past={self.past_window} future={self.future_window}"
                    )
            except Exception as e:
                print(f"⚠ Failed to load cache: {e}. Recomputing...")
        else:
            print("  (no cache file yet for this dataset signature)")

        print("Computing labels and normalization parameters...")
        (
            label_tensor, mean_tensor, std_tensor, momentum_tensor, stats,
        ) = self._compute_labels_and_norm_params()

        try:
            cache_data = _pack_label_cache(
                mean_tensor=mean_tensor,
                std_tensor=std_tensor,
                momentum_tensor=momentum_tensor,
                label_tensor=label_tensor,
                stats=stats,
                dataset=self,
                config=dict(self._runtime_config),
            )
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✓ Cache saved to: {self.cache_file}")
            print(f"  Cache size: {self.cache_file.stat().st_size / 1024:.2f} KB")
        except Exception as e:
            print(f"⚠ Failed to save cache: {e}")

        return (
            label_tensor, mean_tensor, std_tensor, momentum_tensor, stats,
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

        self.task.visualize(
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
        Raw cache y → loss space using train-set label stats owned by the dataset.

        Kept (not inlined) so callers need not touch ``_label_stat_*``.
        """
        return self.task.normalize(
            y, self._label_stat_mean, self._label_stat_std
        )

    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        """Loss-space y → raw cache units via train-set label stats."""
        return self.task.denormalize(
            y, self._label_stat_mean, self._label_stat_std
        )
    
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

    def __getitem__(self, idx: int) -> dict:
        """
        Opaque sample dict.

        Keys: x, p, mean, std, momentum, y.
        FeaturePrep: two-layer (default) or one-layer from config normalization.
        """
        t = self._valid_t[idx]
        win = windows_for_sample(
            t, self._telepred_delay, self._past_window, self._future_window
        )
        x_raw = self._Total[win["input"], :]

        mode = getattr(self, 'normalization_layers', None)
        if mode is None:
            mode = resolve_normalization_layers(self._runtime_config)

        if mode == 'two-layer':
            mean = self._mean_tensor[idx]       # (F,)
            std = self._std_tensor[idx].clone()  # clone: do not mutate cache
            std[std == 0] = 1
            x_normalized = (x_raw - mean) / std     # (T, F)
            momentum = self._momentum_tensor[idx]     # (num_momentum, F)
            mean_normalized = (
                (mean - self.stats['mean']['mean'])
                / (self.stats['mean']['std'] + 1e-8)
            )
            std_normalized = (
                (std - self.stats['std']['mean'])
                / (self.stats['std']['std'] + 1e-8)
            )
            momentum_normalized = (
                (momentum - self.stats['momentum']['mean'])
                / (self.stats['momentum']['std'] + 1e-8)
            )
        else:
            # one-layer: no intra-window x rewrite; FiLM moments neutral.
            x_normalized = x_raw
            mean_normalized = torch.zeros(self._F)
            std_normalized = torch.ones(self._F)
            momentum_normalized = torch.zeros(self.num_momentum, self._F)

        y_full = self._label_tensor[idx]
        y = y_full[self._label_slice]
        y_normalized = self.normalize_y(y)

        recent_prices = x_normalized[
            -self.model_past_len:, self.price_feature_idx
        ]

        return pack_model_sample(
            x=x_normalized,
            p=recent_prices,
            mean=mean_normalized,
            std=std_normalized,
            momentum=momentum_normalized,
            y=y_normalized,
            label=self.label_mode,
        )


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
            
            samples = []
            
            for idx in batch_indices:
                try:
                    samples.append(self.dataset.__getitem__(idx))
                except IndexError as e:
                    print(f"⚠ IndexError for idx={idx}, dataset length={len(self.dataset)}, indices length={len(self.current_epoch_indices)}")
                    raise

            # Opaque batched dict: x, p, mean, std, momentum, y
            yield collate_samples(samples)
    
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
                    
                    # Metric only batch assembly — NOT queue.put() blocking
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

# ═══════════════════════════════════════════════════════════
# SAMPLE / BATCH ABI (dict only: x, p, mean, std, momentum, y)
# ═══════════════════════════════════════════════════════════

Sample = Dict[str, Any]
Batch = Dict[str, Any]

MODEL_INPUT_KEYS: Tuple[str, ...] = ('x', 'p', 'mean', 'std', 'momentum')
SAMPLE_TENSOR_KEYS: Tuple[str, ...] = MODEL_INPUT_KEYS + ('y',)


def resolve_normalization_layers(config: Optional[dict] = None) -> str:
    """
    Return config['normalization_layers']: 'two-layer' | 'one-layer'.

    Default: 'two-layer'.
    """
    cfg = config or {}
    raw = cfg.get('normalization_layers', 'two-layer')
    mode = str(raw).strip().lower().replace('_', '-')
    if mode in ('two-layer', 'twolayer', '2-layer', '2layer'):
        return 'two-layer'
    if mode in ('one-layer', 'onelayer', '1-layer', '1layer'):
        return 'one-layer'
    raise ValueError(
        f"normalization_layers must be 'one-layer' or 'two-layer', got {raw!r}"
    )


def collate_samples(samples: List[Sample]) -> Batch:
    """Stack a list of sample dicts into a batched dict."""
    if not samples:
        return {}
    for i, s in enumerate(samples):
        if not isinstance(s, Mapping):
            raise TypeError(
                f"sample[{i}] must be a dict, got {type(s).__name__}"
            )
    out: Batch = {}
    for key in SAMPLE_TENSOR_KEYS:
        vals = [s[key] for s in samples]
        if vals[0] is None:
            out[key] = None
            continue
        out[key] = torch.stack(vals, dim=0)
    for key, val in samples[0].items():
        if key not in out:
            out[key] = val
    return out


def batch_to_device(batch: Batch, device, non_blocking: bool = True) -> Batch:
    """Move tensor fields of a batch dict to device."""
    if not isinstance(batch, Mapping):
        raise TypeError(f"batch must be a dict, got {type(batch).__name__}")
    out: Batch = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


def unpack_model_input(sample_or_batch: Union[Sample, Batch]) -> Dict[str, torch.Tensor]:
    """Kwargs for forward(x, p, mean, std, momentum) from a sample/batch dict."""
    if not isinstance(sample_or_batch, Mapping):
        raise TypeError(
            f"sample/batch must be a dict, got {type(sample_or_batch).__name__}"
        )
    missing = [
        k for k in MODEL_INPUT_KEYS
        if k not in sample_or_batch or sample_or_batch[k] is None
    ]
    if missing:
        raise KeyError(
            f"model input missing keys {missing}; have {list(sample_or_batch)}"
        )
    return {k: sample_or_batch[k] for k in MODEL_INPUT_KEYS}


def unpack_target(sample_or_batch: Union[Sample, Batch]) -> torch.Tensor:
    """Supervised y tensor from a sample/batch dict."""
    if not isinstance(sample_or_batch, Mapping):
        raise TypeError(
            f"sample/batch must be a dict, got {type(sample_or_batch).__name__}"
        )
    if 'y' not in sample_or_batch:
        raise KeyError("sample/batch missing 'y'")
    return sample_or_batch['y']


def as_train_tuple(
    batch: Batch,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    (x, p, mean, std, momentum, y) view of a batch dict.

    Prefer unpack_model_input / unpack_target; kept for ErrorEnsemble call sites.
    """
    kwargs = unpack_model_input(batch)
    y = unpack_target(batch)
    return (
        kwargs['x'], kwargs['p'], kwargs['mean'], kwargs['std'],
        kwargs['momentum'], y,
    )


def pack_model_sample(
    *,
    x: torch.Tensor,
    p: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    momentum: torch.Tensor,
    y: torch.Tensor,
    **extra: Any,
) -> Sample:
    """Label-facing pack of the current transformer sample schema."""
    sample: Sample = {
        'x': x,
        'p': p,
        'mean': mean,
        'std': std,
        'momentum': momentum,
        'y': y,
    }
    sample.update(extra)
    return sample
