# label_cache.py
# LabelSpec.build_cache_row — target construction for TradingDataset caches.
# Keeps label if-ladders out of data.py (Phase 3).

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch

from labels import (
    LABEL_LATENT,
    LABEL_NONE,
    LABEL_RISE_FLAG,
    LABEL_TREND,
    LABEL_YIELD,
    Label_Latent,
    Label_None,
    Label_RiseFlag,
    find_trend,
    find_yield,
    get_label_class,
)


def _scale_prices_by_window_moments(
    future_prices,
    history_prices,
    mean: torch.Tensor,
    std: torch.Tensor,
    price_feature_idx: int,
    normalize_window: bool,
):
    if not normalize_window:
        return future_prices, history_prices
    price_mean = mean[price_feature_idx]
    price_std = std[price_feature_idx]
    if float(price_std) == 0.0:
        price_std = 1.0
    return (
        (future_prices - price_mean) / price_std,
        (history_prices - price_mean) / price_std,
    )


def build_cache_row(
    label_mode: str,
    *,
    future_prices,
    history_prices,
    current_price,
    mean: torch.Tensor,
    std: torch.Tensor,
    price_feature_idx: int,
    normalize_window: bool,
    min_trend_horizon: int,
    future_window: int,
    fitness_threshold: float,
    horizon_search: str,
    regime_use_past: bool,
    min_yield_percent: float,
) -> Tuple[torch.Tensor, float]:
    """
    Build one full label-cache vector and optional h_past aux.

    Returns (y_full, h_past).
    """
    mode = get_label_class(label_mode).name

    if mode == LABEL_TREND:
        future_for_label, history_for_label = _scale_prices_by_window_moments(
            future_prices,
            history_prices,
            mean,
            std,
            price_feature_idx,
            normalize_window,
        )
        best_h, best_slope, best_intercept, best_h_past = find_trend(
            future_for_label,
            history_for_label,
            min_horizon=min_trend_horizon,
            max_horizon=future_window,
            fitness_threshold=fitness_threshold,
            mode=horizon_search,
            max_past_horizon=future_window,
            use_past=regime_use_past,
        )
        label = torch.tensor(
            [best_h, best_slope, best_intercept], dtype=torch.float32
        )
        return label, float(best_h_past)

    if mode == LABEL_YIELD:
        result = find_yield(
            future_prices,
            current_price,
            min_yield_percent=min_yield_percent,
            max_hope_future_horizon=future_window,
        )
        if len(result) == 1:
            chance, near = float(result[0]), 0.0
        else:
            chance, near = float(result[0]), float(result[1])
        return torch.tensor([chance, near], dtype=torch.float32), 0.0

    if mode == LABEL_RISE_FLAG:
        result = Label_RiseFlag.find(
            future_prices,
            current_price,
            min_yield_percent=min_yield_percent,
            max_hope_future_horizon=future_window,
        )
        flag = float(result[0] if isinstance(result, (tuple, list)) else result)
        return torch.tensor([flag], dtype=torch.float32), 0.0

    if mode == LABEL_LATENT:
        future_for_label, history_for_label = _scale_prices_by_window_moments(
            future_prices,
            history_prices,
            mean,
            std,
            price_feature_idx,
            normalize_window,
        )
        result = Label_Latent.find(future_for_label, history_for_label)
        if isinstance(result, (tuple, list)):
            y_obs = float(result[0])
        else:
            y_obs = float(result.item() if hasattr(result, 'item') else result)
        return torch.tensor([y_obs], dtype=torch.float32), 0.0

    if mode == LABEL_NONE:
        result = Label_None.find(future_prices, current_price)
        z = float(result[0] if isinstance(result, (tuple, list)) else result)
        return torch.tensor([z], dtype=torch.float32), 0.0

    raise ValueError(f"Unknown label mode for build_cache_row: {label_mode!r}")


def pack_sample_for_label(
    label_mode: str,
    *,
    x: torch.Tensor,
    p: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    shape: torch.Tensor,
    y: torch.Tensor,
    **extra: Any,
) -> Dict[str, Any]:
    """
    LabelSpec.pack_sample for current transformer labels.

    All accommodated true labels share the same model-input key set; y is the
    label-sliced / normalized target.
    """
    from sample_batch import pack_model_sample

    # Touch label class so unknown modes fail fast.
    get_label_class(label_mode)
    return pack_model_sample(
        x=x, p=p, mean=mean, std=std, shape=shape, y=y,
        label=get_label_class(label_mode).name,
        **extra,
    )
