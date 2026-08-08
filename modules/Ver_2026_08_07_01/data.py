# data.py
# Dataset and DataLoader with per-window normalization and caching

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
from pathlib import Path
import pickle
import hashlib
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from config import get_config
from datetime import datetime

from Binance import Binance

# Default label fitness threshold (overridden by config['fitness_threshold']).
# Kept as a module constant so older call sites / cache docs stay consistent.
LABEL_FITNESS_THRESHOLD = 0.75

# Label families (config key ``label``).
LABEL_LINEAR_REGIME = 'linear_regime'       # (horizon, slope, intercept) → model slice
LABEL_SCATTERED_HOPE = 'scattered_hope'     # (chance, wait)
VALID_LABEL_MODES = (LABEL_LINEAR_REGIME, LABEL_SCATTERED_HOPE)

# Full layout for linear_regime cache / label_tensor.
LABEL_FULL_DIM_LINEAR = 3
LABEL_FULL_DIM_SCATTERED = 2

# Horizon search rules for find_linear_regime (free OLS, no f[0] anchor).
HORIZON_SEARCH_LONGEST_LINEAR = 'longest_linear'
HORIZON_SEARCH_SHORTEST_TO_THRESHOLD = 'shortest_to_threshold'
VALID_HORIZON_SEARCH_MODES = (
    HORIZON_SEARCH_LONGEST_LINEAR,
    HORIZON_SEARCH_SHORTEST_TO_THRESHOLD,
)


def label_target_slice(label_mode: str, output_dim: int) -> slice:
    """
    Map (label_mode, output_dim) → slice into stored label vectors.

    linear_regime (stored as h, m, b):
      output_dim=3 → [horizon, slope, intercept]
      output_dim=2 → [slope, intercept]
    scattered_hope (stored as chance, wait):
      always [chance, wait]  (output_dim must be 2)
    """
    if label_mode == LABEL_SCATTERED_HOPE:
        if output_dim != 2:
            raise ValueError(
                f"label={LABEL_SCATTERED_HOPE!r} requires output_dim=2, got {output_dim}"
            )
        return slice(0, 2)
    if label_mode == LABEL_LINEAR_REGIME:
        if output_dim == 3:
            return slice(0, 3)
        if output_dim == 2:
            return slice(1, 3)
        raise ValueError(f"output_dim must be 2 or 3 for linear_regime, got {output_dim}")
    raise ValueError(
        f"Unknown label mode {label_mode!r}; expected one of {VALID_LABEL_MODES}"
    )


def label_target_names(label_mode: str, output_dim: int) -> Tuple[str, ...]:
    if label_mode == LABEL_SCATTERED_HOPE:
        return ('chance', 'wait')
    if label_mode == LABEL_LINEAR_REGIME:
        if output_dim == 3:
            return ('horizon', 'slope', 'intercept')
        if output_dim == 2:
            return ('slope', 'intercept')
        raise ValueError(f"output_dim must be 2 or 3 for linear_regime, got {output_dim}")
    raise ValueError(
        f"Unknown label mode {label_mode!r}; expected one of {VALID_LABEL_MODES}"
    )


def resolve_future_window(config: dict) -> int:
    """
    Sample future_window length from the active label mode's max future horizon.

      linear_regime  → max_trend_future_horizon
      scattered_hope → max_hope_future_horizon
    """
    mode = config.get('label', LABEL_LINEAR_REGIME)
    if mode == LABEL_SCATTERED_HOPE:
        return int(config.get('max_hope_future_horizon', 60))
    # linear_regime (default) and any unknown fall back to trend horizon
    return int(config.get('max_trend_future_horizon', 50))


def resolve_model_past_len(config: dict) -> int:
    """
    Length of recent-past price vector ``p`` fed to the model head.

    Matches the active label mode's future horizon scale:
      linear_regime  → max_trend_future_horizon
      scattered_hope → max_hope_future_horizon
    """
    return resolve_future_window(config)


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
    from fsv_dictionary import get_features
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

    past_days = config.get('past_days', 7)
    future_window = resolve_future_window(config)
    max_trend_future = int(config.get('max_trend_future_horizon', future_window))

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
        'features_to_exclude': sorted(config.get('features_to_exclude', [])),
        'features_to_include': sorted(config.get('features_to_include') or []),
        'prediction_target': list(config.get('prediction_target', ('UNKNOWN', 'UNKNOWN'))),
        'price_feature_idx': layout['price_feature_idx'],
        'feature_names': list(layout['feature_names']),
        'F': layout['F'],
        'T': T,  # nominal length from config (see estimate_T_from_config)
        'num_shapes': config.get('num_shapes', 4),

        # --- windowing / sampling ---
        'past_days': past_days,
        'future_window': future_window,
        'model_past_len': resolve_model_past_len(config),
        'step': config.get('step', 1),

        # --- label algorithm ---
        'label': config.get('label', LABEL_LINEAR_REGIME),
        # hope labels always use raw price %; linear uses z-score when normalize=True
        'hope_label_price_space': 'raw',
        'linear_label_price_space': 'zscore' if config.get('normalize', True) else 'raw',
        'min_yield_percent': config.get('min_yield_percent', 0.0),
        'max_hope_future_horizon': config.get('max_hope_future_horizon', 60),
        'min_trend_horizon': config.get('min_trend_horizon', 3),
        'max_trend_future_horizon': max_trend_future,
        'regime_use_past': config.get('regime_use_past', True),
        'fitness_threshold': config.get('fitness_threshold', LABEL_FITNESS_THRESHOLD),
        'horizon_search': config.get('horizon_search', HORIZON_SEARCH_LONGEST_LINEAR),
        'output_dim': config.get('output_dim', 3),
        'normalize': config.get('normalize', True),

        # --- train-split stats ---
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'shuffle_split': config.get('shuffle_split', True),
    }

    param_string = json.dumps(key_params, sort_keys=True, default=str)
    return hashlib.md5(param_string.encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════
# BINANCE DATA LOADER
# ═══════════════════════════════════════════════════════════

def get_binance_data(config):
    """Get data from Binance using the configured parameters.

    Returns:
        features, T, nForget, target_feature_id, feature_names
        On failure all five values are None.
    """
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
        
        start = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
        
        print(f"Fetching Binance data:")
        print(f"  Datatype: {datatype}")
        print(f"  Interval: {interval}")
        print(f"  Start: {start}")
        print(f"  End: {end}")
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
# HORIZON LABELING FUNCTION - SUPPORTS ALL MODES
# ═══════════════════════════════════════════════════════════

def _ols_fit_segment(
    y: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
):
    """
    Free (unanchored) OLS fit of y on time coordinates t.

    If t is None, uses t = 0..h-1. Prefer t measured relative to the
    past/future boundary (past negative, future non-negative) so intercept
    is the level at the decision time t=0.

    Returns:
        slope, intercept, residual_l1  where residual_l1 = sum(|y - ŷ|)
        (caller may divide by a scale to make fitness dimensionless).
    """
    device = y.device
    dtype = y.dtype
    h = y.shape[0]
    if t is None:
        t = torch.arange(h, device=device, dtype=dtype)
    else:
        t = t.to(device=device, dtype=dtype)
    t_mean = t.mean()
    y_mean = y.mean()
    dt = t - t_mean
    denom = torch.sum(dt * dt)
    if denom < eps:
        slope = torch.zeros((), device=device, dtype=dtype)
        intercept = y_mean
    else:
        slope = torch.sum(dt * (y - y_mean)) / denom
        intercept = y_mean - slope * t_mean
    residual_l1 = torch.sum(torch.abs(y - (slope * t + intercept)))
    return slope, intercept, residual_l1


def _regime_segment(history: torch.Tensor, future: torch.Tensor, h_past: int, h_fut: int):
    """
    Build (y, t) for a joint past+future window.

    t is relative to the first future bar (t=0):
      past  → t = -h_past, ..., -1
      future → t = 0, ..., h_fut-1
    """
    device = future.device
    dtype = future.dtype
    parts_y = []
    parts_t = []
    if h_past > 0:
        parts_y.append(history[-h_past:].to(device=device, dtype=dtype))
        parts_t.append(
            torch.arange(-h_past, 0, device=device, dtype=dtype)
        )
    if h_fut > 0:
        parts_y.append(future[:h_fut].to(device=device, dtype=dtype))
        parts_t.append(
            torch.arange(0, h_fut, device=device, dtype=dtype)
        )
    y = torch.cat(parts_y, dim=0)
    t = torch.cat(parts_t, dim=0)
    return y, t


def find_linear_regime(
    future,
    history,
    min_horizon=3,
    max_horizon=30,
    fitness_threshold=1.0,
    mode=HORIZON_SEARCH_LONGEST_LINEAR,
    max_past_horizon=None,
    use_past=True,
):
    """
    Free OLS linear-regime search spanning recent past and nearest future.

    Time origin t=0 is the first future bar. The search window is
        history[-h_past:]  ‖  future[:h_fut]
    with times t ∈ {-h_past,...,-1, 0,...,h_fut-1}.

    Recent past and nearest future **compete** to grow the regime:
    at each step try +1 past and/or +1 future; pick the expansion that
    best preserves linearity (lowest fitness) while respecting the mode.

    Fitness is dimensionless cumulative L1 residual:
        fitness = sum |y - ŷ| / history_sigma
    history_sigma = std(history) (clamped).

    Search modes (config key ``horizon_search``):
        'longest_linear' (default)
            Grow total span while fitness <= threshold.
            Label horizon = h_fut at the last admissible window.
        'shortest_to_threshold'
            Grow until fitness >= threshold (budget exhausted).
            Label horizon = h_fut at that window (or max if never reached).

    Args:
        future : Tensor (future_window,)
        history : Tensor (history_window,)  — past prices (scale + optional past span)
        min_horizon, max_horizon : int  — bounds on **future** extent h_fut
        max_past_horizon : optional max bars of past (default = max_horizon)
        use_past : if False, future-only search (legacy behavior)
        fitness_threshold, mode : as above

    Returns:
        horizon : int   — future extent h_fut of the chosen regime
        slope, intercept : Tensor (0-d)  — joint OLS on past+future window
        h_past : int    — recent-past extent included in the joint fit
    """
    if mode not in VALID_HORIZON_SEARCH_MODES:
        raise ValueError(
            f"Unknown horizon search mode {mode!r}; "
            f"expected one of {VALID_HORIZON_SEARCH_MODES}"
        )

    device = future.device
    dtype = future.dtype
    eps = 1e-8

    history = history.to(device=device, dtype=dtype)
    future = future.to(device=device, dtype=dtype)

    max_horizon = min(int(max_horizon), int(future.shape[0]))
    min_horizon = max(2, int(min_horizon))
    if min_horizon > max_horizon:
        min_horizon = max_horizon

    if max_past_horizon is None:
        max_past_horizon = max_horizon
    max_past_horizon = min(int(max_past_horizon), int(history.shape[0]))
    if not use_past:
        max_past_horizon = 0

    history_sigma = history.std().clamp(min=eps)

    def _eval(h_past: int, h_fut: int):
        y, t = _regime_segment(history, future, h_past, h_fut)
        slope, intercept, residual_l1 = _ols_fit_segment(y, t=t, eps=eps)
        fitness = residual_l1 / history_sigma
        return slope, intercept, fitness

    # Seed: min_horizon future bars at the boundary (past empty).
    h_past, h_fut = 0, min_horizon
    slope, intercept, fitness = _eval(h_past, h_fut)
    best_h_fut = h_fut
    best_h_past = h_past
    best_slope, best_intercept = slope, intercept
    best_total = h_past + h_fut

    if mode == HORIZON_SEARCH_LONGEST_LINEAR:
        # Accept seed even if slightly over threshold (fallback).
        admissible = fitness <= fitness_threshold
        if admissible:
            best_h_fut, best_h_past = h_fut, h_past
            best_slope, best_intercept = slope, intercept
            best_total = h_past + h_fut

        while True:
            candidates = []
            if h_fut < max_horizon:
                candidates.append((h_past, h_fut + 1))
            if h_past < max_past_horizon:
                candidates.append((h_past + 1, h_fut))
            if not candidates:
                break

            # Compete: only expansions that stay under threshold;
            # prefer lower fitness, then larger total span.
            valid = []
            for hp, hf in candidates:
                s, b, fit = _eval(hp, hf)
                if fit <= fitness_threshold:
                    valid.append((fit, -(hp + hf), hp, hf, s, b))

            if not valid:
                # No legal expansion — keep last good (or seed if never good).
                if not admissible:
                    best_h_fut, best_h_past = h_fut, h_past
                    best_slope, best_intercept = slope, intercept
                break

            valid.sort()  # lowest fitness, then largest total (via -total)
            fit, neg_tot, hp, hf, s, b = valid[0]
            h_past, h_fut = hp, hf
            slope, intercept, fitness = s, b, fit
            admissible = True
            total = h_past + h_fut
            if total >= best_total:
                best_total = total
                best_h_fut, best_h_past = h_fut, h_past
                best_slope, best_intercept = slope, intercept

    else:
        # shortest_to_threshold: grow until budget exhausted.
        best_h_past = h_past
        while fitness < fitness_threshold:
            candidates = []
            if h_fut < max_horizon:
                candidates.append((h_past, h_fut + 1))
            if h_past < max_past_horizon:
                candidates.append((h_past + 1, h_fut))
            if not candidates:
                break

            # Compete: pick expansion with lower fitness (stay linear longer).
            scored = []
            for hp, hf in candidates:
                s, b, fit = _eval(hp, hf)
                scored.append((fit, hp, hf, s, b))
            scored.sort()
            fit, hp, hf, s, b = scored[0]
            h_past, h_fut = hp, hf
            slope, intercept, fitness = s, b, fit
            best_h_fut, best_h_past = h_fut, h_past
            best_slope, best_intercept = slope, intercept
            if fitness >= fitness_threshold:
                break

    return best_h_fut, best_slope, best_intercept, best_h_past


def find_scattered_hope(
    future,
    history,
    min_yield_percent: float,
    max_hope_future_horizon: Optional[int] = None,
):
    """
    Scattered-hope label: (chance, wait).

    Parallel role to find_linear_regime: both consume the same
    (future, history) price windows (typically z-scored with past-window
    moments). Characteristics:
      - linear_regime → continuity / OLS fitness → (horizon, slope, intercept)
      - scattered_hope → thresholded yield events → (chance, wait)

    Args:
        future : Tensor (future_window,)
            Future prices (same scale as history).
        history : Tensor (history_window,)
            Past prices; last bar is the reference level for yield.
        min_yield_percent : float
            Minimum yield threshold (%) to count an event.
        max_hope_future_horizon : optional int
            Cap on future look-ahead (defaults to len(future)).

    Returns:
        chance : float
        wait : float
    """
    n = int(future.shape[0])
    if max_hope_future_horizon is not None:
        n = min(n, max(0, int(max_hope_future_horizon)))

    chance = 0.0
    wait = 0.0
    if n == 0 or history.numel() == 0:
        return chance, wait

    current = history[-1]
    # Avoid division by zero on the reference level (raw or z-scored).
    eps = 1e-8
    denom = current if abs(float(current)) > eps else (
        current + eps if float(current) >= 0 else current - eps
    )

    for t in range(n):
        _yield = ((future[t] - current) / denom) * 100
        yv = float(_yield)
        if yv < min_yield_percent:
            continue
        chance += yv
        # t is minutes from decision; weight by inverse yield strength
        wait += t * 1000.0 / yv

    return float(chance), float(wait)


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


def visualize_linear_trends(
    total_tensor: torch.Tensor,
    valid_starts: List[int],
    label_tensor: torch.Tensor,
    mean_tensor: torch.Tensor,
    std_tensor: torch.Tensor,
    price_feature_idx: int,
    past_window: int,
    future_window: int,
    mode: str = 'anchored',
    num_samples: int = 6,
    figsize: Tuple[int, int] = (15, 10),
    interactive: bool = True,
    pred_tensor: Optional[torch.Tensor] = None,
    fixed_order: bool = False,
    sample_ids: Optional[List[int]] = None,
    default_pred_horizon: Optional[int] = None,
    # Regime search params (to recover h_past for label drawing)
    min_horizon: int = 3,
    max_horizon: Optional[int] = None,
    max_past_horizon: Optional[int] = None,
    fitness_threshold: float = LABEL_FITNESS_THRESHOLD,
    horizon_search: str = HORIZON_SEARCH_LONGEST_LINEAR,
    regime_use_past: bool = True,
    normalize_prices: bool = True,
):
    """
    Visualize joint past+future linear regimes.

    Plots for each panel:
      - actual prices (recent past + full future)
      - label regime line on t ∈ [-h_past, h_fut)  (joint OLS; t=0 = first future)
      - inferred trend (from pred_tensor), if provided — future span only

    label_tensor / pred_tensor are in price-normalized space:
      len 3 → (horizon, slope, intercept)
      len 2 → (slope, intercept); preds use default_pred_horizon for length.

    h_past is recomputed via find_linear_regime so the red line matches the
    competitive past/future labeling rule.

    When fixed_order=True, pages sequentially through the provided samples only
    (no random re-sampling). Otherwise interactive Next draws random batches.

    sample_ids: optional dataset indices for titles (aligned with label_tensor rows).
    Keyboard: N / Right / Space = Next, Q / Escape = Quit.
    """
    if total_tensor.is_cuda:
        total_tensor = total_tensor.cpu()
    if pred_tensor is not None and pred_tensor.is_cuda:
        pred_tensor = pred_tensor.cpu()
    if label_tensor.is_cuda:
        label_tensor = label_tensor.cpu()
    if mean_tensor.is_cuda:
        mean_tensor = mean_tensor.cpu()
    if std_tensor.is_cuda:
        std_tensor = std_tensor.cpu()

    num_available = int(label_tensor.shape[0])
    if num_available == 0:
        print("⚠ No samples available for trend visualization.")
        return

    if pred_tensor is not None and pred_tensor.shape[0] != num_available:
        raise ValueError(
            f"pred_tensor length ({pred_tensor.shape[0]}) must match "
            f"label_tensor length ({num_available})"
        )

    num_samples = min(num_samples, num_available)
    if num_samples < 1:
        num_samples = 1
    # Prefer a full 2x3 grid when possible (only for random browsing)
    if not fixed_order and num_samples % 2 != 0 and num_samples >= 6:
        num_samples = 6

    n_cols = 3 if num_samples > 1 else 1
    n_rows = max(1, (num_samples + n_cols - 1) // n_cols)
    n_pages = max(1, (num_available + num_samples - 1) // num_samples) if fixed_order else None

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    # Room for buttons at the bottom in interactive mode
    if interactive:
        fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.14, hspace=0.45, wspace=0.25)
    else:
        fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.08, hspace=0.45, wspace=0.25)

    page_num = [0]  # 1-based after first draw
    # Keep button refs alive (matplotlib GC otherwise disconnects them)
    _widgets: List[Any] = []

    def _page_indices(page: int) -> List[int]:
        """0-based row indices into label/pred tensors for this page."""
        k = min(num_samples, num_available)
        if fixed_order:
            start_i = (page - 1) * num_samples
            if start_i >= num_available:
                start_i = 0
            return list(range(start_i, min(start_i + num_samples, num_available)))
        rng = random.Random(int(time.time() * 1000) + page)
        return rng.sample(range(num_available), k)

    def _draw_page():
        page_num[0] += 1
        if fixed_order and n_pages is not None and page_num[0] > n_pages:
            page_num[0] = 1  # wrap
        indices = _page_indices(page_num[0])

        for ax in axes:
            ax.clear()
            ax.set_visible(True)

        max_h = int(max_horizon) if max_horizon is not None else int(future_window)

        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
            ax = axes[i]
            start = valid_starts[idx]
            mean = mean_tensor[idx].flatten()
            std = std_tensor[idx].flatten()

            price_mean = float(mean[price_feature_idx].item())
            price_std = float(std[price_feature_idx].item())
            if price_std == 0:
                price_std = 1.0

            h_lbl, m_lbl, b_lbl = _trend_to_price_space(
                label_tensor[idx], price_mean, price_std
            )

            y_start = start + past_window
            y_end = y_start + future_window
            history_prices = total_tensor[
                start:y_start, price_feature_idx
            ].float()
            future_prices = total_tensor[
                y_start:y_end, price_feature_idx
            ].float()

            # Recover h_past (and confirm joint fit) with the same labeling rule.
            if normalize_prices:
                hist_n = (history_prices - price_mean) / price_std
                fut_n = (future_prices - price_mean) / price_std
            else:
                hist_n, fut_n = history_prices, future_prices
            try:
                _, _, _, h_past_lbl = find_linear_regime(
                    fut_n,
                    hist_n,
                    min_horizon=min_horizon,
                    max_horizon=max_h,
                    fitness_threshold=fitness_threshold,
                    mode=horizon_search,
                    max_past_horizon=max_past_horizon,
                    use_past=regime_use_past,
                )
                h_past_lbl = int(h_past_lbl)
            except Exception:
                h_past_lbl = 0

            # Actual prices: enough past to cover regime past (min 15 for context).
            past_context = max(15, h_past_lbl)
            past_ctx = min(past_context, past_window, y_start)
            ctx_start = y_start - past_ctx
            actual_prices = total_tensor[
                ctx_start:y_end, price_feature_idx
            ].cpu().numpy()
            time_steps = np.arange(-past_ctx, len(actual_prices) - past_ctx)

            if past_ctx > 0:
                ax.plot(
                    time_steps[: past_ctx + 1],
                    actual_prices[: past_ctx + 1],
                    color='#8AB4D4',
                    linewidth=2,
                    label='Actual (past)',
                    alpha=0.75,
                )
            ax.plot(
                time_steps[past_ctx:],
                actual_prices[past_ctx:],
                'b-',
                linewidth=2,
                label='Actual (future)',
                alpha=0.75,
            )

            # Label joint regime: t = -h_past .. h_fut-1 (intercept at t=0)
            if h_lbl > 0 or h_past_lbl > 0:
                t0 = -h_past_lbl
                t1 = max(h_lbl, 1)
                t_lbl = np.arange(t0, t1)
                if len(t_lbl) > 0:
                    ax.plot(
                        t_lbl,
                        m_lbl * t_lbl + b_lbl,
                        'r-',
                        linewidth=2.5,
                        label=(
                            f'Label regime: past={h_past_lbl}, H={h_lbl}, '
                            f'm={m_lbl:.4f}, b={b_lbl:.2f}'
                        ),
                    )
                if h_past_lbl > 0:
                    ax.axvline(
                        x=-h_past_lbl, color='red', linestyle=':',
                        linewidth=1.2, alpha=0.65,
                    )
                if h_lbl > 0:
                    ax.axvline(
                        x=h_lbl - 1, color='red', linestyle='--',
                        linewidth=1.3, alpha=0.65,
                    )
                # Regime bands: slightly different tint for past vs future (match price split)
                if h_past_lbl > 0:
                    ax.axvspan(
                        -h_past_lbl, 0,
                        alpha=0.12, color='#B8D4E8',  # soft blue — past regime
                    )
                if h_lbl > 0:
                    ax.axvspan(
                        0, max(h_lbl - 1, 0),
                        alpha=0.12, color='#FFCC80',  # soft orange — future regime
                    )

            # Inferred trend (model): future side only; t=0..H-1
            if pred_tensor is not None:
                pred_h_default = (
                    default_pred_horizon
                    if default_pred_horizon is not None
                    else max_h
                )
                h_prd, m_prd, b_prd = _trend_to_price_space(
                    pred_tensor[idx],
                    price_mean,
                    price_std,
                    default_horizon=pred_h_default,
                )
                if h_prd > 0:
                    t_prd = np.arange(h_prd)
                    ax.plot(
                        t_prd, m_prd * t_prd + b_prd, 'g-', linewidth=2.5,
                        label=f'Inferred: H={h_prd}, m={m_prd:.4f}, b={b_prd:.2f}',
                    )
                    ax.axvline(
                        x=h_prd - 1, color='green', linestyle='--',
                        linewidth=1.3, alpha=0.65,
                    )
                title_line2 = (
                    f'pred H={h_prd}, m={m_prd:.4f}, b={b_prd:.2f}'
                )
            else:
                title_line2 = None

            ax.axvline(x=0, color='gray', linestyle=':', linewidth=1.2, alpha=0.5)

            display_id = sample_ids[idx] if sample_ids is not None else idx
            title_line1 = (
                f'Sample {display_id}: past={h_past_lbl}, H={h_lbl}, '
                f'm={m_lbl:.4f}, b={b_lbl:.2f}'
            )
            ax.set_title(
                title_line1 if title_line2 is None else f'{title_line1}\n{title_line2}',
                fontsize=9,
            )
            ax.set_xlabel('Time (min; t=0 = first future bar)')
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='best')

        for i in range(len(indices), len(axes)):
            axes[i].clear()
            axes[i].set_visible(False)

        page_label = f'Page {page_num[0]}'
        if fixed_order and n_pages is not None:
            page_label = f'Page {page_num[0]}/{n_pages}'
        line1 = f'Linear Regime Visualization — Mode: {mode.upper()}  |  {page_label}'
        line2_parts = []
        if regime_use_past:
            line2_parts.append('past+future compete')
        if pred_tensor is not None:
            line2_parts.append('red=label regime  green=inferred')
        if fixed_order:
            line2_parts.append('chosen samples only')
        if interactive:
            line2_parts.append('[Next] / [Quit]')
        line2 = '  |  '.join(line2_parts) if line2_parts else ''
        fig.suptitle(
            line1 if not line2 else f'{line1}\n{line2}',
            fontsize=12,
            fontweight='bold',
        )
        fig.canvas.draw_idle()
        shown_ids = (
            [sample_ids[i] for i in indices] if sample_ids is not None else list(indices)
        )
        print(f"  {page_label}: samples {shown_ids}")

    if interactive:
        ax_next = fig.add_axes([0.55, 0.02, 0.18, 0.055])
        ax_quit = fig.add_axes([0.76, 0.02, 0.18, 0.055])
        btn_next = Button(ax_next, 'Next')
        btn_quit = Button(ax_quit, 'Quit')
        _widgets.extend([btn_next, btn_quit, ax_next, ax_quit])

        def _on_next(_event):
            _draw_page()

        def _on_quit(_event):
            print("  Quit — closing trend browser.")
            plt.close(fig)

        btn_next.on_clicked(_on_next)
        btn_quit.on_clicked(_on_quit)

        def _on_key(event):
            if event.key in ('n', 'right', ' '):
                _draw_page()
            elif event.key in ('q', 'escape'):
                print("  Quit — closing trend browser.")
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', _on_key)
        if fixed_order:
            print(
                f"Trend browser ({num_available} chosen samples): "
                f"[Next] or N/Space = next page · [Quit] or Q = exit"
            )
        else:
            print("Trend browser: [Next] or N/Space = next page · [Quit] or Q = exit")

    _draw_page()
    plt.show(block=True)


def visualize_scattered_hope(
    total_tensor: torch.Tensor,
    valid_starts: List[int],
    label_tensor: torch.Tensor,
    price_feature_idx: int,
    past_window: int,
    future_window: int,
    min_yield_percent: float = 0.0,
    max_hope_future_horizon: Optional[int] = None,
    num_samples: int = 6,
    figsize: Tuple[int, int] = (15, 10),
    interactive: bool = True,
    pred_tensor: Optional[torch.Tensor] = None,
    fixed_order: bool = False,
    sample_ids: Optional[List[int]] = None,
    past_context: int = 15,
):
    """
    Visualize scattered-hope labels (chance, wait) in raw price space.

    Each panel:
      - recent past + future actual prices
      - horizontal line at reference price history[-1]
      - markers on future bars where yield % >= min_yield_percent
      - label (chance, wait); optional pred (chance, wait)
    """
    if total_tensor.is_cuda:
        total_tensor = total_tensor.cpu()
    if label_tensor.is_cuda:
        label_tensor = label_tensor.cpu()
    if pred_tensor is not None and pred_tensor.is_cuda:
        pred_tensor = pred_tensor.cpu()

    num_available = int(label_tensor.shape[0])
    if num_available == 0:
        print("⚠ No samples available for hope visualization.")
        return
    if pred_tensor is not None and pred_tensor.shape[0] != num_available:
        raise ValueError(
            f"pred_tensor length ({pred_tensor.shape[0]}) must match "
            f"label_tensor length ({num_available})"
        )

    num_samples = min(num_samples, num_available)
    if num_samples < 1:
        num_samples = 1
    if not fixed_order and num_samples % 2 != 0 and num_samples >= 6:
        num_samples = 6

    n_cols = 3 if num_samples > 1 else 1
    n_rows = max(1, (num_samples + n_cols - 1) // n_cols)
    n_pages = (
        max(1, (num_available + num_samples - 1) // num_samples)
        if fixed_order else None
    )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    if interactive:
        fig.subplots_adjust(
            left=0.07, right=0.98, top=0.88, bottom=0.14, hspace=0.50, wspace=0.25
        )
    else:
        fig.subplots_adjust(
            left=0.07, right=0.98, top=0.88, bottom=0.08, hspace=0.50, wspace=0.25
        )

    page_num = [0]
    _widgets: List[Any] = []
    max_h = (
        int(max_hope_future_horizon)
        if max_hope_future_horizon is not None
        else int(future_window)
    )
    max_h = min(max_h, int(future_window))

    def _page_indices(page: int) -> List[int]:
        k = min(num_samples, num_available)
        if fixed_order:
            start_i = (page - 1) * num_samples
            if start_i >= num_available:
                start_i = 0
            return list(range(start_i, min(start_i + num_samples, num_available)))
        rng = random.Random(int(time.time() * 1000) + page)
        return rng.sample(range(num_available), k)

    def _as_pair(vec) -> Tuple[float, float]:
        v = vec.detach().cpu().flatten() if hasattr(vec, 'detach') else vec
        return float(v[0]), float(v[1])

    def _draw_page():
        page_num[0] += 1
        if fixed_order and n_pages is not None and page_num[0] > n_pages:
            page_num[0] = 1
        indices = _page_indices(page_num[0])

        for ax in axes:
            ax.clear()
            ax.set_visible(True)

        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
            ax = axes[i]
            start = valid_starts[idx]
            y_start = start + past_window
            y_end = y_start + future_window

            history = total_tensor[start:y_start, price_feature_idx].float()
            future = total_tensor[y_start:y_end, price_feature_idx].float()
            ref = float(history[-1].item()) if history.numel() else 0.0
            eps = 1e-8
            denom = ref if abs(ref) > eps else (ref + eps if ref >= 0 else ref - eps)

            # Context: recent past + full future
            past_ctx = min(past_context, past_window, y_start)
            ctx_start = y_start - past_ctx
            actual = total_tensor[ctx_start:y_end, price_feature_idx].cpu().numpy()
            t_axis = np.arange(-past_ctx, len(actual) - past_ctx)

            if past_ctx > 0:
                ax.plot(
                    t_axis[: past_ctx + 1], actual[: past_ctx + 1],
                    color='#8AB4D4', linewidth=2, label='Actual (past)', alpha=0.8,
                )
            ax.plot(
                t_axis[past_ctx:], actual[past_ctx:],
                'b-', linewidth=2, label='Actual (future)', alpha=0.8,
            )

            # Reference price at t=0 decision
            ax.axhline(y=ref, color='gray', linestyle=':', linewidth=1.2, alpha=0.7,
                       label=f'ref={ref:.2f}')
            ax.axvline(x=0, color='gray', linestyle=':', linewidth=1.2, alpha=0.5)

            # Event markers: yield % >= threshold within hope horizon
            n_hope = min(max_h, int(future.shape[0]))
            event_t, event_p, event_y = [], [], []
            for t in range(n_hope):
                yv = float(((future[t] - ref) / denom) * 100)
                if yv >= min_yield_percent:
                    event_t.append(t)
                    event_p.append(float(future[t].item()))
                    event_y.append(yv)
            if event_t:
                ax.scatter(
                    event_t, event_p, c='orange', s=28, zorder=5,
                    label=f'hits (≥{min_yield_percent:g}%)',
                )
                # Light connector from ref to each hit
                for tt, pp in zip(event_t, event_p):
                    ax.plot([0, tt], [ref, pp], color='orange', alpha=0.25, linewidth=1.0)

            # Hope horizon band
            if n_hope > 0:
                ax.axvspan(0, max(n_hope - 1, 0), alpha=0.08, color='#FFE0B2')

            chance_lbl, wait_lbl = _as_pair(label_tensor[idx])
            title_l1 = f'chance={chance_lbl:.2f}, wait={wait_lbl:.2f}'
            title_l2 = None
            if pred_tensor is not None:
                chance_prd, wait_prd = _as_pair(pred_tensor[idx])
                title_l2 = f'pred chance={chance_prd:.2f}, wait={wait_prd:.2f}'
                # Annotate corner so pred is visible without cluttering legend
                ax.text(
                    0.02, 0.98,
                    f'pred c={chance_prd:.1f} w={wait_prd:.1f}',
                    transform=ax.transAxes, va='top', ha='left', fontsize=8,
                    color='green',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                )

            display_id = sample_ids[idx] if sample_ids is not None else idx
            head = f'Sample {display_id}: {title_l1}'
            ax.set_title(
                head if title_l2 is None else f'{head}\n{title_l2}',
                fontsize=9,
            )
            ax.set_xlabel('Time (min; t=0 = first future bar)')
            ax.set_ylabel('Price (raw)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='best')

        for i in range(len(indices), len(axes)):
            axes[i].clear()
            axes[i].set_visible(False)

        page_label = f'Page {page_num[0]}'
        if fixed_order and n_pages is not None:
            page_label = f'Page {page_num[0]}/{n_pages}'
        line1 = f'Scattered Hope — min_yield={min_yield_percent:g}%  |  {page_label}'
        line2_parts = [f'horizon={max_h}', 'orange=hits ≥ threshold']
        if pred_tensor is not None:
            line2_parts.append('green text=pred')
        if fixed_order:
            line2_parts.append('chosen samples only')
        if interactive:
            line2_parts.append('[Next] / [Quit]')
        line2 = '  |  '.join(line2_parts)
        fig.suptitle(f'{line1}\n{line2}', fontsize=12, fontweight='bold')
        fig.canvas.draw_idle()
        shown_ids = (
            [sample_ids[i] for i in indices] if sample_ids is not None else list(indices)
        )
        print(f"  {page_label}: samples {shown_ids}")

    if interactive:
        ax_next = fig.add_axes([0.55, 0.02, 0.18, 0.055])
        ax_quit = fig.add_axes([0.76, 0.02, 0.18, 0.055])
        btn_next = Button(ax_next, 'Next')
        btn_quit = Button(ax_quit, 'Quit')
        _widgets.extend([btn_next, btn_quit, ax_next, ax_quit])

        def _on_next(_event):
            _draw_page()

        def _on_quit(_event):
            print("  Quit — closing hope browser.")
            plt.close(fig)

        btn_next.on_clicked(_on_next)
        btn_quit.on_clicked(_on_quit)

        def _on_key(event):
            if event.key in ('n', 'right', ' '):
                _draw_page()
            elif event.key in ('q', 'escape'):
                print("  Quit — closing hope browser.")
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', _on_key)
        if fixed_order:
            print(
                f"Hope browser ({num_available} chosen samples): "
                f"[Next] or N/Space = next page · [Quit] or Q = exit"
            )
        else:
            print("Hope browser: [Next] or N/Space = next page · [Quit] or Q = exit")

    _draw_page()
    plt.show(block=True)


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
        past_days: int = 7,
        future_minutes: Optional[int] = None,
        step: int = 1,
        max_accum_deviation: float = 12.0,
        min_trend_horizon: int = 3,
        max_trend_future_horizon: Optional[int] = None,
        device: str = 'cpu',
        normalize: bool = True,
        data_dir: Optional[str] = None,
        force_recompute: bool = False,
        split_ratios: Optional[Tuple[float, float, float]] = None
    ):
        config = get_config()
        self.feature_names = feature_names
        self.n_market_features = n_market_features
        self.n_macro_features = n_macro_features
        self.past_days = past_days
        self.price_feature_idx = price_feature_idx
        self.step = step
        self.max_accum_deviation = max_accum_deviation
        self.device = device
        self.normalize = normalize
        self.force_recompute = force_recompute
        self.num_shapes = config['num_shapes']

        self.label_mode = config.get('label', LABEL_LINEAR_REGIME)
        if self.label_mode not in VALID_LABEL_MODES:
            raise ValueError(
                f"Unknown label={self.label_mode!r}; "
                f"expected one of {VALID_LABEL_MODES}"
            )
        self.min_yield_percent = float(config.get('min_yield_percent', 0.0))
        self.max_hope_future_horizon = int(
            config.get('max_hope_future_horizon', 60)
        )

        # future_window from label-mode max future horizon
        if future_minutes is None:
            future_minutes = resolve_future_window(config)
        self.future_minutes = int(future_minutes)

        self.min_trend_horizon = int(
            config.get('min_trend_horizon', min_trend_horizon)
        )
        mtf = config.get('max_trend_future_horizon', max_trend_future_horizon)
        if mtf is None:
            mtf = self.future_minutes
        self.max_trend_future_horizon = int(mtf)
        # Model p length + regime past max follow the active label mode's horizon.
        self.model_past_len = resolve_model_past_len(config)
        self.max_past_horizon = self.model_past_len
        # Aliases used by older call sites / viz
        self.min_horizon = self.min_trend_horizon
        self.max_horizon = self.max_trend_future_horizon

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
        self.output_dim = int(config.get('output_dim', 3))
        if self.label_mode == LABEL_SCATTERED_HOPE:
            if self.output_dim != 2:
                print(
                    f"⚠ label={LABEL_SCATTERED_HOPE}: forcing output_dim "
                    f"{self.output_dim} → 2 (chance, wait)"
                )
                self.output_dim = 2
        elif self.output_dim not in (2, 3):
            raise ValueError(
                f"output_dim must be 2 or 3 for linear_regime, got {self.output_dim}"
            )
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
        self.shuffle_split = config.get('shuffle_split', True)
        
        if data_dir is None:
            data_dir = config.get('data_dir', 'dataset_cache')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.Total = self._stack_features(features)
        self.T, self.F = self.Total.shape
        
        self.past_window = 60 * 24 * past_days
        self.future_window = self.future_minutes
        self.total_window = self.past_window + self.future_window
        
        if self.max_trend_future_horizon < self.min_trend_horizon:
            self.max_trend_future_horizon = self.min_trend_horizon
            self.max_horizon = self.max_trend_future_horizon
            print(
                f"⚠ max_trend_future_horizon adjusted to min_trend_horizon "
                f"({self.min_trend_horizon})"
            )
            # Re-sync model past length only if still on linear_regime
            if self.label_mode == LABEL_LINEAR_REGIME:
                self.model_past_len = self.max_trend_future_horizon
                self.max_past_horizon = self.model_past_len
        
        self.valid_starts = self._get_valid_starts()
        
        self.cache_key = self._generate_cache_key(config)
        self.cache_file = self.data_dir / f"labels_{self.cache_key}.pkl"
        
        # Load or compute labels and normalization parameters
        self.label_tensor, self.mean_tensor, self.std_tensor, self.shape_tensor, self.stats = self._load_or_compute_labels()
        
        self._create_splits()

        # Model-target stats (slice of full label vectors).
        self._label_stat_mean = self.stats['label']['mean'][self._label_slice]
        self._label_stat_std = self.stats['label']['std'][self._label_slice]
        
        # Cache local references for faster __getitem__
        self._mean_tensor = self.mean_tensor
        self._std_tensor = self.std_tensor
        self._shape_tensor = self.shape_tensor
        self._label_tensor = self.label_tensor
        self._valid_starts = self.valid_starts
        self._Total = self.Total
        self._F = self.F
        self._past_window = self.past_window
        self._normalize = self.normalize
        
        print(f"\nTradingDataset initialized:")
        print(f"  Features: {self.F}, Time steps: {self.T}")
        print(f"  Past window: {self.past_window} minutes ({past_days} days)")
        print(f"  Future window: {self.future_window} minutes "
              f"(from {'max_hope_future_horizon' if self.label_mode == LABEL_SCATTERED_HOPE else 'max_trend_future_horizon'})")
        print(f"  Step: {step}")
        print(f"  Label mode: {self.label_mode}")
        print(f"  Model past vector p length: {self.model_past_len}")
        if self.label_mode == LABEL_LINEAR_REGIME:
            print(f"  Horizon search: {self.horizon_search}")
            print(f"  Fitness threshold: {self.fitness_threshold}")
            print(f"  Regime use past: {self.regime_use_past} "
                  f"(max past/future trend={self.max_trend_future_horizon})")
            print(
                f"  Min trend horizon: {self.min_trend_horizon}, "
                f"Max trend future: {self.max_trend_future_horizon}"
            )
            print(f"  Label price space: z-scored (past-window μ,σ)")
        else:
            print(f"  min_yield_percent: {self.min_yield_percent}")
            print(f"  max_hope_future_horizon: {self.max_hope_future_horizon}")
            print(f"  Label price space: raw prices (true % yield)")
        print(
            f"  Model output_dim: {self.output_dim} "
            f"{label_target_names(self.label_mode, self.output_dim)}"
        )
        print(f"  Horizon viz mode: {self.horizon_mode}")
        if self.horizon_mode == 'weighted':
            print(f"  Weight type: {self.weight_type}, Decay: {self.weight_decay}")
        print(f"  Max accum deviation (legacy): {max_accum_deviation}")
        print(f"  Normalize: {normalize}")
        print(f"  Total samples: {len(self)}")
        print(f"  Cache: {'Loaded' if self.cache_file.exists() else 'Computed'}")
        print(f"  Split ratios - Train: {self.split_ratios[0]:.2f}, "
              f"Val: {self.split_ratios[1]:.2f}, Test: {self.split_ratios[2]:.2f}")
        print(f"  Split sizes - Train: {len(self.train_indices)}, "
              f"Val: {len(self.val_indices)}, Test: {len(self.test_indices)}")
        print(f"\n📊 Dataset Statistics (training set, {len(self.train_indices)} samples):")
        full_names = (
            ('horizon', 'slope', 'intercept')
            if self.label_mode == LABEL_LINEAR_REGIME
            else ('chance', 'wait')
        )
        print(f"  Full label {full_names}:")
        print(f"    Mean:  {self.stats['label']['mean'].tolist()}")
        print(f"    Std:   {self.stats['label']['std'].tolist()}")
        print(
            f"  Model target "
            f"{label_target_names(self.label_mode, self.output_dim)}:"
        )
        print(f"    Mean:  {self._label_stat_mean.tolist()}")
        print(f"    Std:   {self._label_stat_std.tolist()}")
        
    def _stack_features(self, features: List[np.ndarray]) -> torch.Tensor:
        shapes = [arr.shape for arr in features]
        if len(set(shapes)) != 1:
            raise ValueError(f"All features must have the same shape. Got: {shapes}")
        stacked = np.stack(features, axis=1)
        return torch.from_numpy(stacked).float()
    
    def _get_valid_starts(self) -> List[int]:
        max_start = self.T - self.total_window
        if max_start < 0:
            raise ValueError(
                f"Time series length {self.T} is too short for "
                f"past {self.past_window} + future {self.future_window} = {self.total_window} minutes"
            )
        starts = list(range(0, max_start + 1, self.step))
        if not starts:
            raise ValueError(f"No valid starts with step {self.step}")
        return starts
    
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
        
        for idx in tqdm(range(len(self)), desc="Computing labels and norm params"):
            # It traverses all samples.

            start = self.valid_starts[idx]
            x = self.Total[start:start + self.past_window, :]
            
            if self.normalize:
                _, mean, std, shape = self._normalize_window(x, compute_moments=True)
                mean_list.append(mean)
                std_list.append(std)
                shape_list.append(shape)
            else:
                # x_normalized = x
                mean = torch.zeros(self.F)
                std = torch.ones(self.F)
                # Must match normalize_window: shape is (num_shapes, F)
                shape = torch.zeros(self.num_shapes, self.F)
                mean_list.append(mean)
                std_list.append(std)
                shape_list.append(shape)

            y_start = start + self.past_window
            y_end = y_start + self.future_window
            future_prices = self.Total[y_start:y_end, self.price_feature_idx]
            history_prices = self.Total[
                start:start + self.past_window, self.price_feature_idx
            ]

            if self.label_mode == LABEL_LINEAR_REGIME:
                # Continuity labels in z-scored space (dimensionless residuals).
                if self.normalize:
                    price_mean = mean[self.price_feature_idx]
                    price_std = std[self.price_feature_idx]
                    future_for_label = (future_prices - price_mean) / price_std
                    history_for_label = (history_prices - price_mean) / price_std
                else:
                    future_for_label = future_prices
                    history_for_label = history_prices

                best_h, best_slope, best_intercept, _best_h_past = find_linear_regime(
                    future_for_label,
                    history_for_label,
                    min_horizon=self.min_trend_horizon,
                    max_horizon=self.max_trend_future_horizon,
                    fitness_threshold=self.fitness_threshold,
                    mode=self.horizon_search,
                    max_past_horizon=self.max_trend_future_horizon,
                    use_past=self.regime_use_past,
                )
                # Full cache vector; model may slice to (m,b) when output_dim=2.
                label = torch.tensor(
                    [best_h, best_slope, best_intercept], dtype=torch.float32
                )
            elif self.label_mode == LABEL_SCATTERED_HOPE:
                # True % yield vs last raw price — never z-score for hope.
                chance, wait = find_scattered_hope(
                    future_prices,
                    history_prices,
                    min_yield_percent=self.min_yield_percent,
                    max_hope_future_horizon=self.max_hope_future_horizon,
                )
                label = torch.tensor(
                    [float(chance), float(wait)], dtype=torch.float32
                )
            else:
                raise ValueError(f"Unknown label mode: {self.label_mode!r}")

            label_list.append(label)

        # Unnormalized labels and x's properties of all data samples - train, validation, and test
        label_tensor = torch.stack(label_list)
        mean_tensor = torch.stack(mean_list)
        std_tensor = torch.stack(std_list)
        shape_tensor = torch.stack(shape_list)
        
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
        
        return label_tensor, mean_tensor, std_tensor, shape_tensor, stats
    
    def _load_or_compute_labels(self):
        if not self.force_recompute and self.cache_file.exists():
            print(f"Loading cached labels from: {self.cache_file}")
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                if (cached_data.get('num_samples') == len(self) and
                    cached_data.get('F') == self.F and
                    cached_data.get('T') == self.T and
                    cached_data.get('past_window') == self.past_window and
                    cached_data.get('future_window') == self.future_window):
                    
                    label_tensor = cached_data['label_tensor']
                    mean_tensor = cached_data['mean_tensor']
                    std_tensor = cached_data['std_tensor']
                    shape_tensor = cached_data['shape_tensor']
                    stats = cached_data['stats']
                    
                    print(f"✓ Cache loaded successfully! ({label_tensor.shape[0]} samples)")
                    return label_tensor, mean_tensor, std_tensor, shape_tensor, stats
                else:
                    print("⚠ Cache mismatch. Recomputing...")
            except Exception as e:
                print(f"⚠ Failed to load cache: {e}. Recomputing...")
        
        print("Computing labels and normalization parameters...")
        label_tensor, mean_tensor, std_tensor, shape_tensor, stats = self._compute_labels_and_norm_params()
        
        try:
            cache_data = {
                'label_tensor': label_tensor,
                'mean_tensor': mean_tensor,
                'std_tensor': std_tensor,
                'shape_tensor': shape_tensor,
                'stats': stats,
                'num_samples': len(self),
                'F': self.F,
                'T': self.T,
                'past_window': self.past_window,
                'future_window': self.future_window,
                'step': self.step,
                'max_accum_deviation': self.max_accum_deviation,
                'min_trend_horizon': self.min_trend_horizon,
                'max_trend_future_horizon': self.max_trend_future_horizon,
                'max_hope_future_horizon': self.max_hope_future_horizon,
                'fitness_threshold': self.fitness_threshold,
                'horizon_search': self.horizon_search,
                'label': self.label_mode,
                'min_yield_percent': self.min_yield_percent,
                'normalize': self.normalize,
                'price_feature_idx': self.price_feature_idx,
                'cache_key': self.cache_key,
                'timestamp': time.time(),
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✓ Cache saved to: {self.cache_file}")
            print(f"  Cache size: {self.cache_file.stat().st_size / 1024:.2f} KB")
        except Exception as e:
            print(f"⚠ Failed to save cache: {e}")
        
        return label_tensor, mean_tensor, std_tensor, shape_tensor, stats
    
    def visualize_trends(
        self,
        num_samples: int = 6,
        figsize: Tuple[int, int] = (15, 10),
        interactive: bool = True,
    ):
        """
        Visualize labels for the active mode (linear regime or scattered hope).

        Args:
            num_samples: Samples per page
            figsize: Figure size
            interactive: If True, Next/Quit buttons page through unlimited batches
        """
        if self.label_tensor.shape[0] == 0:
            print("⚠ No labels available for visualization.")
            return

        print("\n" + "=" * 60)
        if self.label_mode == LABEL_SCATTERED_HOPE:
            print("VISUALIZING SCATTERED HOPE (chance / wait)")
            if interactive:
                print("  Interactive: Next = new random page, Quit = exit")
            print("=" * 60)
            visualize_scattered_hope(
                total_tensor=self.Total,
                valid_starts=self.valid_starts,
                label_tensor=self.label_tensor,
                price_feature_idx=self.price_feature_idx,
                past_window=self.past_window,
                future_window=self.future_window,
                min_yield_percent=self.min_yield_percent,
                max_hope_future_horizon=self.max_hope_future_horizon,
                num_samples=num_samples,
                figsize=figsize,
                interactive=interactive,
            )
            return

        print("VISUALIZING LINEAR TRENDS")
        if interactive:
            print("  Interactive: Next = new random page, Quit = exit")
        print("=" * 60)

        visualize_linear_trends(
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
            min_horizon=self.min_trend_horizon,
            max_horizon=self.max_trend_future_horizon,
            max_past_horizon=self.max_trend_future_horizon,
            fitness_threshold=self.fitness_threshold,
            horizon_search=self.horizon_search,
            regime_use_past=self.regime_use_past,
            normalize_prices=self.normalize,
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
          linear_regime  → (h,m,b) or (m,b)
          scattered_hope → (chance, wait)
        """
        mean = self._label_stat_mean
        std = self._label_stat_std
        if y.shape[-1] != mean.shape[-1]:
            raise ValueError(
                f"y last dim {y.shape[-1]} != expected output_dim {mean.shape[-1]}"
            )
        return (y - mean.to(y.device)) / (std.to(y.device) + 1e-8)

    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse of normalize_y for model-target vectors (dim = output_dim)."""
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
        
        start = self._valid_starts[idx]
        x = self._Total[start:start + self._past_window, :]     # assuming self._Total: (T, F)
        
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
        
        # Full label cache:
        #   linear_regime  → (h, m, b) then slice by output_dim
        #   scattered_hope → (chance, wait)  (slice is 0:2)
        y_full = self._label_tensor[idx]
        y = y_full[self._label_slice]

        # Normalize label, mean, std, and shape using dataset statistics
        y_normalized = self.normalize_y(y)

        # Shared architecture: recent past prices p (length = model_past_len).
        recent_prices = x_normalized[-self.model_past_len:, self.price_feature_idx]
        
        # Mean: (F,)
        mean_normalized = (mean - self.stats['mean']['mean']) / (self.stats['mean']['std'] + 1e-8)
        
        # Std: (F,)
        std_normalized = (std - self.stats['std']['mean']) / (self.stats['std']['std'] + 1e-8)
        
        # Shape: (num_shapes, F)
        shape_normalized = (shape - self.stats['shape']['mean']) / (self.stats['shape']['std'] + 1e-8)
        
        return x_normalized, recent_prices, mean_normalized, std_normalized, shape_normalized, y_normalized


    
    # def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Optimized __getitem__ with cached references and view() instead of unsqueeze()."""
        
    #     start = self._valid_starts[idx]
    #     x = self._Total[:, start:start + self._past_window]
        
    #     if self._normalize:
    #         # Use view() instead of unsqueeze() - no memory allocation
    #         mean = self._mean_tensor[idx].view(-1, 1)
    #         std = self._std_tensor[idx].view(-1, 1)
    #         std[std == 0] = 1
    #         x_normalized = (x - mean) / std
    #         shape = self._shape_tensor[idx]
    #     else:
    #         x_normalized = x
    #         mean = torch.zeros(self._F)
    #         std = torch.ones(self._F)
    #         shape = torch.zeros(self.num_shapes, self._F)
        
    #     # Get label
    #     y = self._label_tensor[idx]
        
    #     # Normalize label, mean, std, and shape using dataset statistics
    #     # Label: (horizon, slope, intercept)
    #     y_normalized = self.normalize_y(y)
        
    #     # Mean: (F,)
    #     mean_normalized = (mean - self.stats['mean']['mean'].view(-1, 1)) / (self.stats['mean']['std'].view(-1, 1) + 1e-8)
        
    #     # Std: (F,)
    #     std_normalized = (std - self.stats['std']['mean'].view(-1, 1)) / (self.stats['std']['std'].view(-1, 1) + 1e-8)
        
    #     # Shape: (F, num_shapes)
    #     shape_normalized = (shape - self.stats['shape']['mean']) / (self.stats['shape']['std'] + 1e-8)
        
    #     return x_normalized, y_normalized, mean_normalized, std_normalized, shape_normalized


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

            # (B, T, F), (B, max_past_horizon), (B, F), (B, F), (B, 4, F), (B, 3,). All normalized and ready for model input
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
