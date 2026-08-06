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

# Label fitness threshold used in _compute_labels_and_norm_params.
# Keep in the dataset signature so code changes invalidate caches.
LABEL_FITNESS_THRESHOLD = 0.75


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

    # Macro base names: FEATURES dict order minus excludes (deterministic, no CSV I/O)
    from fsv_dictionary import FEATURES
    exclude = set(config.get('features_to_exclude') or [])
    macro_base_names = [name for name in FEATURES.keys() if name not in exclude]
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
    future_minutes = config.get('future_minutes', 60)
    max_horizon = config.get('max_horizon', None)
    if max_horizon is None:
        max_horizon = future_minutes

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
        'prediction_target': list(config.get('prediction_target', ('UNKNOWN', 'UNKNOWN'))),
        'price_feature_idx': layout['price_feature_idx'],
        'feature_names': list(layout['feature_names']),
        'F': layout['F'],
        'T': T,  # nominal length from config (see estimate_T_from_config)
        'num_shapes': config.get('num_shapes', 4),

        # --- windowing / sampling ---
        'past_days': past_days,
        'future_minutes': future_minutes,
        'step': config.get('step', 1),

        # --- label algorithm ---
        'min_horizon': config.get('min_horizon', 3),
        'max_horizon': max_horizon,
        'fitness_threshold': LABEL_FITNESS_THRESHOLD,
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

def find_shortest_horizon(
    future,
    history,
    min_horizon=3,
    max_horizon=30,
    fitness_threshold=1.0,
):
    """
    Find the longest approximately linear future segment.

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

    Returns:
        horizon : int
        slope : Tensor
        intercept : Tensor
    """

    device = future.device
    dtype = future.dtype
    eps = 1e-8

    max_horizon = min(max_horizon, future.shape[0])
    history_sigma = history.std().clamp(min=eps)

    best_h = min_horizon

    best_slope = torch.tensor(0.0, device=device, dtype=dtype)
    best_intercept = torch.tensor(0.0, device=device, dtype=dtype)

    # ---------------------------------------------------------
    # Search progressively longer horizons
    # ---------------------------------------------------------

    for h in range(min_horizon, max_horizon + 1):

        y = future[:h]

        x = torch.arange(h, device=device, dtype=dtype)

        x_mean = x.mean()
        y_mean = y.mean()

        dx = x - x_mean

        slope = torch.sum(dx * (y - y_mean)) / (
            torch.sum(dx * dx) + eps
        )

        intercept = y_mean - slope * x_mean
        prediction = slope * x + intercept

        cumulative_deviation = torch.sum(
            torch.abs(y - prediction)
        )

        fitness = cumulative_deviation

        if fitness <= fitness_threshold:
            best_h = h
            best_slope = slope
            best_intercept = intercept
        else:
            break

    return best_h, best_slope, best_intercept


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

def _trend_to_price_space(h_s_b, price_mean: float, price_std: float):
    """Map (horizon, slope, intercept) from price-normalized space to price space."""
    horizon = max(0, int(round(float(h_s_b[0].item() if hasattr(h_s_b[0], 'item') else h_s_b[0]))))
    slope_norm = float(h_s_b[1].item() if hasattr(h_s_b[1], 'item') else h_s_b[1])
    intercept_norm = float(h_s_b[2].item() if hasattr(h_s_b[2], 'item') else h_s_b[2])
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
):
    """
    Visualize linear trends over future prices.

    Plots for each panel:
      - actual future prices
      - label linear trend (from label_tensor)
      - inferred linear trend (from pred_tensor), if provided

    label_tensor / pred_tensor are in price-normalized (horizon, slope, intercept)
    space — the same space as dataset.label_tensor / denormalize_y outputs.

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

            h_lbl, m_lbl, b_lbl = _trend_to_price_space(label_tensor[idx], price_mean, price_std)

            y_start = start + past_window
            y_end = y_start + future_window
            # Total is (T, F)
            future_prices = total_tensor[y_start:y_end, price_feature_idx].cpu().numpy()

            time_steps = np.arange(len(future_prices))
            ax.plot(time_steps, future_prices, 'b-', linewidth=2, label='Actual prices', alpha=0.75)

            # Label trend
            if h_lbl > 0:
                t_lbl = np.arange(h_lbl)
                ax.plot(
                    t_lbl, m_lbl * t_lbl + b_lbl, 'r-', linewidth=2.5,
                    label=f'Label: H={h_lbl}, m={m_lbl:.4f}, b={b_lbl:.2f}',
                )
                ax.axvline(x=h_lbl - 1, color='red', linestyle='--', linewidth=1.3, alpha=0.65)

            # Inferred trend
            if pred_tensor is not None:
                h_prd, m_prd, b_prd = _trend_to_price_space(pred_tensor[idx], price_mean, price_std)
                if h_prd > 0:
                    t_prd = np.arange(h_prd)
                    ax.plot(
                        t_prd, m_prd * t_prd + b_prd, 'g-', linewidth=2.5,
                        label=f'Inferred: H={h_prd}, m={m_prd:.4f}, b={b_prd:.2f}',
                    )
                    ax.axvline(x=h_prd - 1, color='green', linestyle='--', linewidth=1.3, alpha=0.65)
                title_extra = f' | pred H={h_prd}'
            else:
                title_extra = ''

            ax.axvline(x=0, color='gray', linestyle=':', linewidth=1.2, alpha=0.5)
            if h_lbl > 0:
                ax.axvspan(0, max(h_lbl - 1, 0), alpha=0.10, color='orange')

            display_id = sample_ids[idx] if sample_ids is not None else idx
            ax.set_title(
                f'Sample {display_id}: label H={h_lbl}, m={m_lbl:.4f}, b={b_lbl:.2f}{title_extra}',
                fontsize=9,
            )
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='best')

        for i in range(len(indices), len(axes)):
            axes[i].clear()
            axes[i].set_visible(False)

        page_label = f'Page {page_num[0]}'
        if fixed_order and n_pages is not None:
            page_label = f'Page {page_num[0]}/{n_pages}'
        has_pred = '  |  red=label  green=inferred' if pred_tensor is not None else ''
        order_note = '  |  chosen samples only' if fixed_order else ''
        fig.suptitle(
            f'Linear Trend Visualization — Mode: {mode.upper()}  |  {page_label}'
            f'{has_pred}{order_note}'
            + ('  |  [Next] / [Quit]' if interactive else ''),
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
        future_minutes: int = 60,
        step: int = 1,
        max_accum_deviation: float = 12.0,
        min_horizon: int = 3,
        max_horizon: Optional[int] = None,
        device: str = 'cpu',
        normalize: bool = True,
        data_dir: Optional[str] = None,
        force_recompute: bool = False,
        split_ratios: Optional[Tuple[float, float, float]] = None
    ):
        self.feature_names = feature_names
        self.n_market_features = n_market_features
        self.n_macro_features = n_macro_features
        self.past_days = past_days
        self.future_minutes = future_minutes
        self.price_feature_idx = price_feature_idx
        self.step = step
        self.max_accum_deviation = max_accum_deviation
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.device = device
        self.normalize = normalize
        self.force_recompute = force_recompute

        config = get_config()
        self.num_shapes = config['num_shapes']
        
        self.horizon_mode = config.get('horizon_mode', 'anchored')
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
        self.future_window = future_minutes
        self.total_window = self.past_window + self.future_window
        
        if self.max_horizon is None:
            self.max_horizon = self.future_window
        elif self.max_horizon < self.min_horizon:
            self.max_horizon = self.min_horizon
            print(f"⚠ max_horizon adjusted to min_horizon ({self.min_horizon})")
        
        self.valid_starts = self._get_valid_starts()
        
        self.cache_key = self._generate_cache_key(config)
        self.cache_file = self.data_dir / f"labels_{self.cache_key}.pkl"
        
        # Load or compute labels and normalization parameters
        self.label_tensor, self.mean_tensor, self.std_tensor, self.shape_tensor, self.stats = self._load_or_compute_labels()
        
        self._create_splits()
        
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
        print(f"  Future window: {self.future_window} minutes")
        print(f"  Step: {step}")
        print(f"  Horizon mode: {self.horizon_mode}")
        if self.horizon_mode == 'weighted':
            print(f"  Weight type: {self.weight_type}, Decay: {self.weight_decay}")
        print(f"  Max accum deviation: {max_accum_deviation}")
        print(f"  Min horizon: {min_horizon}, Max horizon: {self.max_horizon}")
        print(f"  Normalize: {normalize}")
        print(f"  Total samples: {len(self)}")
        print(f"  Cache: {'Loaded' if self.cache_file.exists() else 'Computed'}")
        print(f"  Split ratios - Train: {self.split_ratios[0]:.2f}, "
              f"Val: {self.split_ratios[1]:.2f}, Test: {self.split_ratios[2]:.2f}")
        print(f"  Split sizes - Train: {len(self.train_indices)}, "
              f"Val: {len(self.val_indices)}, Test: {len(self.test_indices)}")
        
        # Print statistics
        print(f"\n📊 Dataset Statistics (computed from TRAINING set, {len(self.train_indices)} samples):")
        print(f"  Label (horizon, slope, intercept):")
        print(f"    Mean:  {self.stats['label']['mean'].tolist()}")
        print(f"    Std:   {self.stats['label']['std'].tolist()}")
        print(f"  Mean tensor (per feature):")
        print(f"    Mean:  {self.stats['mean']['mean'].tolist()}")
        print(f"    Std:   {self.stats['mean']['std'].tolist()}")
        print(f"  Std tensor (per feature):")
        print(f"    Mean:  {self.stats['std']['mean'].tolist()}")
        print(f"    Std:   {self.stats['std']['std'].tolist()}")
        print(f"  Shape tensor (per feature):")
        print(f"    Mean:  {self.stats['shape']['mean'].tolist()}")
        print(f"    Std:   {self.stats['shape']['std'].tolist()}")
        
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
            
            if self.normalize:
                price_mean = mean[self.price_feature_idx]
                price_std = std[self.price_feature_idx]
                future_prices_normalized = (future_prices - price_mean) / price_std
            else:
                future_prices_normalized = future_prices
            
            best_h, best_slope, best_intercept = find_shortest_horizon(
                future_prices_normalized,
                self.Total[start:start + self.past_window, self.price_feature_idx],
                min_horizon=self.min_horizon,
                max_horizon=self.max_horizon,
                fitness_threshold=LABEL_FITNESS_THRESHOLD,
            )

            label = torch.tensor([best_h, best_slope, best_intercept], dtype=torch.float32)

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
                'min_horizon': self.min_horizon,
                'max_horizon': self.max_horizon,
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
        y_normalized = (y - self.stats['label']['mean']) / (self.stats['label']['std'] + 1e-8)
        return y_normalized
    
    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        device = y.device
        label_mean = self.stats['label']['mean'].to(device)
        label_std = self.stats['label']['std'].to(device)
        y_denormalized = y * label_std + label_mean
        return y_denormalized
    
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
        
        # Get label
        y = self._label_tensor[idx]
        
        # Normalize label, mean, std, and shape using dataset statistics
        # Label: (horizon, slope, intercept)
        y_normalized = self.normalize_y(y)
        
        # Mean: (F,)
        mean_normalized = (mean - self.stats['mean']['mean']) / (self.stats['mean']['std'] + 1e-8)
        
        # Std: (F,)
        std_normalized = (std - self.stats['std']['mean']) / (self.stats['std']['std'] + 1e-8)
        
        # Shape: (num_shapes, F)
        shape_normalized = (shape - self.stats['shape']['mean']) / (self.stats['shape']['std'] + 1e-8)
        
        return x_normalized, y_normalized, mean_normalized, std_normalized, shape_normalized


    
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
            
            x_list, y_list, mean_list, std_list, shape_list = [], [], [], [], []
            
            for idx in batch_indices:
                try:
                    x, y, mean, std, shape = self.dataset.__getitem__(idx)
                    x_list.append(x)
                    y_list.append(y)
                    mean_list.append(mean)
                    std_list.append(std)
                    shape_list.append(shape)
                except IndexError as e:
                    print(f"⚠ IndexError for idx={idx}, dataset length={len(self.dataset)}, indices length={len(self.current_epoch_indices)}")
                    raise
            
            x_batch = torch.stack(x_list)
            y_batch = torch.stack(y_list)
            mean_batch = torch.stack(mean_list)
            std_batch = torch.stack(std_list)
            shape_batch = torch.stack(shape_list)
            
            yield x_batch, y_batch, mean_batch, std_batch, shape_batch      # (B, T, F), (3,), (B, F), (B, F), (B, 4, F). All normalized and ready for model input
    
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
