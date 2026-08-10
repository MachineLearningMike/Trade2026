# labels.py
# Label modes, target construction, training loss, and visualization.
#
# Shared by data.py (label cache / dataset), model.py (objective), and INFER.
#
# Each label family is a class (find + loss). Bind Label_*.loss for training.
# Visualization helpers (formerly label_viz.py) live at the bottom.

import random
import time
from typing import Any, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Default label fitness threshold (overridden by config['fitness_threshold']).
LABEL_FITNESS_THRESHOLD = 0.75

# Label family ids (config key ``label``).
LABEL_TREND = 'trend'
LABEL_YIELD = 'yield'
VALID_LABEL_MODES = (LABEL_TREND, LABEL_YIELD)

LABEL_FULL_DIM_TREND = 3
LABEL_FULL_DIM_YIELD = 2
LABEL_FULL_DIM_YIELD_CHANCE = 1

# Horizon search rules for linear trend (free OLS, no f[0] anchor).
HORIZON_SEARCH_LONGEST_LINEAR = 'longest_linear'
HORIZON_SEARCH_SHORTEST_TO_THRESHOLD = 'shortest_to_threshold'
VALID_HORIZON_SEARCH_MODES = (
    HORIZON_SEARCH_LONGEST_LINEAR,
    HORIZON_SEARCH_SHORTEST_TO_THRESHOLD,
)


# ═══════════════════════════════════════════════════════════
# SHARED OLS HELPERS (linear trend)
# ═══════════════════════════════════════════════════════════

def _ols_fit_segment(
    y: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
):
    """Free OLS of y on time t. Returns slope, intercept, residual L1."""
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
    Build (y, t) for joint past+future window.
    t relative to first future bar: past negative, future 0..h_fut-1.
    """
    device = future.device
    dtype = future.dtype
    parts_y = []
    parts_t = []
    if h_past > 0:
        parts_y.append(history[-h_past:].to(device=device, dtype=dtype))
        parts_t.append(torch.arange(-h_past, 0, device=device, dtype=dtype))
    if h_fut > 0:
        parts_y.append(future[:h_fut].to(device=device, dtype=dtype))
        parts_t.append(torch.arange(0, h_fut, device=device, dtype=dtype))
    return torch.cat(parts_y, dim=0), torch.cat(parts_t, dim=0)


# ═══════════════════════════════════════════════════════════
# LABEL FAMILY: TREND
# ═══════════════════════════════════════════════════════════

class Label_Trend:
    """
    Trend labels: full cache (horizon, slope, intercept).

    Config id: 'trend'
    Price space: typically z-scored (caller responsibility).
    """

    name = LABEL_TREND
    full_dim = LABEL_FULL_DIM_TREND

    @classmethod
    def resolve_future_window(cls, config: dict) -> int:
        return int(config.get('max_trend_future_horizon', 50))

    @classmethod
    def resolve_model_past_len(cls, config: dict) -> int:
        return cls.resolve_future_window(config)

    @classmethod
    def target_slice(cls, output_dim: int, **_kwargs) -> slice:
        if output_dim == 3:
            return slice(0, 3)
        if output_dim == 2:
            return slice(1, 3)
        raise ValueError(
            f"{cls.name}: output_dim must be 2 or 3, got {output_dim}"
        )

    @classmethod
    def target_names(cls, output_dim: int, **_kwargs) -> Tuple[str, ...]:
        if output_dim == 3:
            return ('horizon', 'slope', 'intercept')
        if output_dim == 2:
            return ('slope', 'intercept')
        raise ValueError(
            f"{cls.name}: output_dim must be 2 or 3, got {output_dim}"
        )

    @classmethod
    def find(
        cls,
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
        Free OLS linear-regime search (past + future compete).

        Returns:
            horizon (h_fut), slope, intercept, h_past
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

        h_past, h_fut = 0, min_horizon
        slope, intercept, fitness = _eval(h_past, h_fut)
        best_h_fut = h_fut
        best_h_past = h_past
        best_slope, best_intercept = slope, intercept
        best_total = h_past + h_fut

        if mode == HORIZON_SEARCH_LONGEST_LINEAR:
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

                valid = []
                for hp, hf in candidates:
                    s, b, fit = _eval(hp, hf)
                    if fit <= fitness_threshold:
                        valid.append((fit, -(hp + hf), hp, hf, s, b))

                if not valid:
                    if not admissible:
                        best_h_fut, best_h_past = h_fut, h_past
                        best_slope, best_intercept = slope, intercept
                    break

                valid.sort()
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
            best_h_past = h_past
            while fitness < fitness_threshold:
                candidates = []
                if h_fut < max_horizon:
                    candidates.append((h_past, h_fut + 1))
                if h_past < max_past_horizon:
                    candidates.append((h_past + 1, h_fut))
                if not candidates:
                    break

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

    @classmethod
    def make_auxiliary(cls, config: Optional[dict] = None, **overrides) -> dict:
        """
        Class-private loss knobs for trend.

        Keys (only used inside Label_Trend.loss):
          slope_weight, intercept_weight
        """
        cfg = config or {}
        aux = {
            'slope_weight': float(cfg.get('slope_loss_weight', 0.1)),
            'intercept_weight': float(cfg.get('intercept_loss_weight', 10.0)),
        }
        aux.update(overrides)
        return aux

    @classmethod
    def loss(
        cls,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_truth: torch.Tensor,
        auxiliary: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Uniform signature: loss(x, y_pred, y_truth, auxiliary).

        MSE on (h,m,b) or (m,b). ``x`` unused for trend (kept for API uniformity).
        ``auxiliary`` may contain slope_weight / intercept_weight.
        """
        aux = auxiliary or {}
        slope_weight = float(aux.get('slope_weight', 0.1))
        intercept_weight = float(aux.get('intercept_weight', 10.0))

        out_dim = y_pred.shape[-1]
        if y_truth.shape[-1] != out_dim:
            raise ValueError(
                f"{cls.name} loss: target dim {y_truth.shape[-1]} != pred dim {out_dim}"
            )
        if out_dim == 3:
            horizon_loss = nn.functional.mse_loss(y_pred[:, 0], y_truth[:, 0])
            slope_loss = nn.functional.mse_loss(y_pred[:, 1], y_truth[:, 1])
            intercept_loss = nn.functional.mse_loss(y_pred[:, 2], y_truth[:, 2])
            return (
                horizon_loss
                + slope_weight * slope_loss
                + intercept_weight * intercept_loss
            )
        if out_dim == 2:
            slope_loss = nn.functional.mse_loss(y_pred[:, 0], y_truth[:, 0])
            intercept_loss = nn.functional.mse_loss(y_pred[:, 1], y_truth[:, 1])
            return slope_weight * slope_loss + intercept_weight * intercept_loss
        raise ValueError(
            f"{cls.name} loss: unsupported pred dim {out_dim}; expected 2 or 3"
        )


# ═══════════════════════════════════════════════════════════
# LABEL FAMILY: YIELD
# ═══════════════════════════════════════════════════════════

class Label_Yield:
    """
    Yield labels on raw prices: (chance, wait) or (chance,).

    Config id: 'yield'
    Price space: raw (true % yield); caller must not z-score inputs.
    """

    name = LABEL_YIELD

    @classmethod
    def full_dim(cls, chance_only: bool = False) -> int:
        return (
            LABEL_FULL_DIM_YIELD_CHANCE
            if chance_only
            else LABEL_FULL_DIM_YIELD
        )

    @classmethod
    def resolve_future_window(cls, config: dict) -> int:
        return int(config.get('max_hope_future_horizon', 60))

    @classmethod
    def resolve_model_past_len(cls, config: dict) -> int:
        return cls.resolve_future_window(config)

    @classmethod
    def resolve_output_dim(cls, chance_only: bool = False) -> int:
        return cls.full_dim(chance_only)

    @classmethod
    def target_slice(cls, output_dim: int, chance_only: bool = False, **_kwargs) -> slice:
        expected = cls.resolve_output_dim(chance_only)
        if output_dim != expected:
            raise ValueError(
                f"{cls.name} with chance_only={chance_only} requires "
                f"output_dim={expected}, got {output_dim}"
            )
        return slice(0, expected)

    @classmethod
    def target_names(
        cls, output_dim: int, chance_only: bool = False, **_kwargs
    ) -> Tuple[str, ...]:
        if chance_only or output_dim == 1:
            return ('chance',)
        return ('chance', 'wait')

    @classmethod
    def find(
        cls,
        future,
        history,
        min_yield_percent: float,
        max_hope_future_horizon: Optional[int] = None,
        chance_only: bool = False,
    ):
        """
        Thresholded yield events vs the last raw price history[-1].

        Inputs MUST be raw prices (not window z-scores). Percent yield at bar t:
            yv = 100 * (future[t] - current) / current
        For every t in [0, n) with yv >= min_yield_percent:
            chance += (yv - min_yield_percent)   # excess % above threshold
            wait   += t / yv                     # unless chance_only

        So chance == 0 when there are no hits, and grows with how far each hit
        exceeds the threshold (not with z-score artifacts).

        Returns:
            chance_only=False → (chance, wait)
            chance_only=True  → (chance,)
        """
        n = int(future.shape[0])
        if max_hope_future_horizon is not None:
            n = min(n, max(0, int(max_hope_future_horizon)))

        chance = 0.0
        wait = 0.0
        if n == 0 or history.numel() == 0:
            return (float(chance),) if chance_only else (float(chance), float(wait))

        current = history[-1]
        eps = 1e-8
        denom = current if abs(float(current)) > eps else (
            current + eps if float(current) >= 0 else current - eps
        )

        for t in range(n):
            yv = float(((future[t] - current) / denom) * 100)
            if yv <= min_yield_percent:
                continue
            chance += (yv - min_yield_percent)
            if not chance_only:
                wait += t / yv

        if chance_only:
            return (float(chance),)
        return (float(chance), float(wait))

    @classmethod
    def make_auxiliary(cls, config: Optional[dict] = None, **overrides) -> dict:
        """
        Class-private loss knobs for yield.

        Keys (only used inside Label_Yield.loss):
          chance_weight, wait_weight, chance_only
        """
        cfg = config or {}
        aux = {
            'chance_weight': float(cfg.get('chance_loss_weight', 1.0)),
            'wait_weight': float(cfg.get('wait_loss_weight', 1.0)),
            'chance_only': bool(cfg.get('chance_only', False)),
        }
        aux.update(overrides)
        return aux

    @classmethod
    def loss(
        cls,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_truth: torch.Tensor,
        auxiliary: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Uniform signature: loss(x, y_pred, y_truth, auxiliary).

        MSE on (chance[, wait]). ``x`` unused for yield (kept for API uniformity).
        ``auxiliary`` may contain chance_weight / wait_weight / chance_only.
        """
        aux = auxiliary or {}
        chance_weight = float(aux.get('chance_weight', 1.0))
        wait_weight = float(aux.get('wait_weight', 1.0))
        chance_only = bool(aux.get('chance_only', False))

        out_dim = y_pred.shape[-1]
        if y_truth.shape[-1] != out_dim:
            raise ValueError(
                f"{cls.name} loss: target dim {y_truth.shape[-1]} != pred dim {out_dim}"
            )
        if chance_only or out_dim == 1:
            if out_dim != 1:
                raise ValueError(
                    f"chance_only requires pred dim 1, got {out_dim}"
                )
            return chance_weight * nn.functional.mse_loss(
                y_pred[:, 0], y_truth[:, 0]
            )
        if out_dim == 2:
            chance_loss = nn.functional.mse_loss(y_pred[:, 0], y_truth[:, 0])
            wait_loss = nn.functional.mse_loss(y_pred[:, 1], y_truth[:, 1])
            return chance_weight * chance_loss + wait_weight * wait_loss
        raise ValueError(
            f"{cls.name} loss: require dim 1 (chance_only) or 2, got {out_dim}"
        )


# ═══════════════════════════════════════════════════════════
# REGISTRY + DISPATCH
# ═══════════════════════════════════════════════════════════

LABEL_CLASSES: dict[str, Type] = {
    LABEL_TREND: Label_Trend,
    LABEL_YIELD: Label_Yield,
    # legacy aliases (pre-rename configs / caches)
    'linear_regime': Label_Trend,
    'linear_trend': Label_Trend,
    'scattered_hope': Label_Yield,
}


def get_label_class(label_mode: str) -> Type:
    if label_mode not in LABEL_CLASSES:
        raise ValueError(
            f"Unknown label mode {label_mode!r}; expected one of {tuple(LABEL_CLASSES)}"
        )
    return LABEL_CLASSES[label_mode]


def resolve_hope_output_dim(chance_only: bool) -> int:
    """Yield head / label width: 1 if chance_only else 2."""
    return Label_Yield.resolve_output_dim(chance_only)


def label_target_slice(
    label_mode: str,
    output_dim: int,
    chance_only: bool = False,
) -> slice:
    return get_label_class(label_mode).target_slice(
        output_dim, chance_only=chance_only
    )


def label_target_names(
    label_mode: str,
    output_dim: int,
    chance_only: bool = False,
) -> Tuple[str, ...]:
    return get_label_class(label_mode).target_names(
        output_dim, chance_only=chance_only
    )


def resolve_future_window(config: dict) -> int:
    mode = config.get('label', LABEL_TREND)
    return get_label_class(mode).resolve_future_window(config)


def resolve_model_past_len(config: dict) -> int:
    mode = config.get('label', LABEL_TREND)
    return get_label_class(mode).resolve_model_past_len(config)


# Backward-compatible free functions (delegate to classes)

def find_trend(*args, **kwargs):
    return Label_Trend.find(*args, **kwargs)


def find_yield(*args, **kwargs):
    return Label_Yield.find(*args, **kwargs)


# Legacy names
find_linear_regime = find_trend
find_scattered_hope = find_yield


def bind_label_loss(
    label_mode: str,
    config: Optional[dict] = None,
    **overrides,
):
    """
    Return the concrete label-class loss, ready for ResumableTrainer.

    Resulting callable has signature::

        loss_fn(x, y_pred, y_truth) -> Tensor

    which is ``Label_*.loss`` with class-private ``auxiliary`` already bound
    (from config / overrides via Label_*.make_auxiliary). No generic dispatcher
    at train time — the active class is fixed here.
    """
    label_cls = get_label_class(label_mode)
    auxiliary = label_cls.make_auxiliary(config, **overrides)

    def loss_fn(
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_truth: torch.Tensor,
    ) -> torch.Tensor:
        return label_cls.loss(x, y_pred, y_truth, auxiliary)

    loss_fn.label_cls = label_cls  # type: ignore[attr-defined]
    loss_fn.auxiliary = auxiliary  # type: ignore[attr-defined]
    return loss_fn


# ═══════════════════════════════════════════════════════════
# VISUALIZATION (formerly label_viz.py)
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


def visualize_trend(
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
    h_past_tensor: Optional[torch.Tensor] = None,
    # Kept for call-site compatibility; not used (labels come from cache).
    min_horizon: int = 3,
    max_horizon: Optional[int] = None,
    max_past_horizon: Optional[int] = None,
    fitness_threshold: float = LABEL_FITNESS_THRESHOLD,
    horizon_search: str = HORIZON_SEARCH_LONGEST_LINEAR,
    regime_use_past: bool = True,
    normalize_prices: bool = True,
):
    """
    Visualize joint past+future linear regimes from the dataset cache.

    Plots for each panel:
      - actual prices (recent past + full future)
      - label regime line on t ∈ [-h_past, h_fut)  (from cache; t=0 = first future)
      - inferred trend (from pred_tensor), if provided — future span only

    label_tensor is the cached natural-unit label (not recomputed):
      len 3 → (horizon, slope, intercept) in window price-z space for m,b
      len 2 → (slope, intercept); horizon from default_pred_horizon
    h_past_tensor: optional cached past extent (same length as label_tensor).
      If None, past extent is drawn as 0 (future-only segment).

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
    if h_past_tensor is not None and h_past_tensor.is_cuda:
        h_past_tensor = h_past_tensor.cpu()

    num_available = int(label_tensor.shape[0])
    if num_available == 0:
        print("⚠ No samples available for trend visualization.")
        return

    if pred_tensor is not None and pred_tensor.shape[0] != num_available:
        raise ValueError(
            f"pred_tensor length ({pred_tensor.shape[0]}) must match "
            f"label_tensor length ({num_available})"
        )
    if h_past_tensor is not None and int(h_past_tensor.shape[0]) != num_available:
        raise ValueError(
            f"h_past_tensor length ({h_past_tensor.shape[0]}) must match "
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

            # Labels from cache only (never re-run find_trend here).
            h_lbl, m_lbl, b_lbl = _trend_to_price_space(
                label_tensor[idx], price_mean, price_std,
                default_horizon=int(max_h),
            )
            if h_past_tensor is not None:
                h_past_lbl = int(max(0, round(float(h_past_tensor[idx].item()))))
            else:
                h_past_lbl = 0

            y_start = start + past_window
            y_end = y_start + future_window

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
        line1 = f'Trend Visualization — Mode: {mode.upper()}  |  {page_label}'
        line2_parts = ['labels from dataset cache']
        if h_past_tensor is not None:
            line2_parts.append('h_past from cache')
        elif regime_use_past:
            line2_parts.append('h_past=0 (no cache)')
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


def visualize_yield(
    total_tensor: torch.Tensor,
    valid_starts: List[int],
    label_tensor: torch.Tensor,
    price_feature_idx: int,
    past_window: int,
    future_window: int,
    min_yield_percent: float = 0.0,
    max_hope_future_horizon: Optional[int] = None,
    chance_only: bool = False,
    num_samples: int = 6,
    figsize: Tuple[int, int] = (15, 10),
    interactive: bool = True,
    pred_tensor: Optional[torch.Tensor] = None,
    fixed_order: bool = False,
    sample_ids: Optional[List[int]] = None,
    past_context: int = 15,
):
    """
    Visualize yield labels in raw price space.

    Each panel:
      - recent past + future actual prices
      - horizontal line at reference price history[-1]
      - orange markers on future bars where yield % >= min_yield_percent
        (decorative only — illustrates the threshold rule on the chart)
      - title chance/wait from ``label_tensor`` (dataset cache; not recomputed)
      - small chance×wait inset: truth (and pred if given) as scatter points
        (skipped when ``chance_only`` / 1-D labels)
    """
    if total_tensor.is_cuda:
        total_tensor = total_tensor.cpu()
    if label_tensor.is_cuda:
        label_tensor = label_tensor.cpu()
    if pred_tensor is not None and pred_tensor.is_cuda:
        pred_tensor = pred_tensor.cpu()

    num_available = int(label_tensor.shape[0])
    if num_available == 0:
        print("⚠ No samples available for yield visualization.")
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
    _inset_axes: List[Any] = []  # chance×wait insets; removed on each redraw
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

    def _as_components(vec) -> Tuple[float, ...]:
        v = vec.detach().cpu().flatten() if hasattr(vec, 'detach') else np.asarray(vec).ravel()
        if chance_only or len(v) == 1:
            return (float(v[0]),)
        return (float(v[0]), float(v[1]))

    def _clear_insets():
        while _inset_axes:
            ia = _inset_axes.pop()
            try:
                ia.remove()
            except Exception:
                pass

    def _draw_page():
        page_num[0] += 1
        if fixed_order and n_pages is not None and page_num[0] > n_pages:
            page_num[0] = 1
        indices = _page_indices(page_num[0])

        _clear_insets()
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

            # Hit markers (chart decoration only; chance comes from label_tensor)
            n_hope = min(max_h, int(future.shape[0]))
            event_t, event_p = [], []
            for t in range(n_hope):
                yv = float(((future[t] - ref) / denom) * 100)
                if yv <= min_yield_percent:
                    continue
                event_t.append(t)
                event_p.append(float(future[t].item()))
            if event_t:
                ax.scatter(
                    event_t, event_p, c='orange', s=28, zorder=5,
                    label=f'hits={len(event_t)} (≥{min_yield_percent:g}%)',
                )
                for tt, pp in zip(event_t, event_p):
                    ax.plot([-1, tt], [ref, pp], color='orange', alpha=0.25, linewidth=1.0)

            if n_hope > 0:
                ax.axvspan(0, max(n_hope - 1, 0), alpha=0.08, color='#FFE0B2')

            # Labels: always from stored dataset cache (not recomputed here)
            comps_lbl = _as_components(label_tensor[idx])
            comps_prd = (
                _as_components(pred_tensor[idx]) if pred_tensor is not None else None
            )
            if chance_only or len(comps_lbl) == 1:
                title_l1 = f'chance={comps_lbl[0]:.2f}'
            else:
                title_l1 = f'chance={comps_lbl[0]:.2f}, wait={comps_lbl[1]:.2f}'
            title_l2 = None
            if comps_prd is not None:
                if chance_only or len(comps_prd) == 1:
                    title_l2 = f'pred chance={comps_prd[0]:.2f}'
                    pred_txt = f'pred c={comps_prd[0]:.1f}'
                else:
                    title_l2 = (
                        f'pred chance={comps_prd[0]:.2f}, wait={comps_prd[1]:.2f}'
                    )
                    pred_txt = f'pred c={comps_prd[0]:.1f} w={comps_prd[1]:.1f}'
                ax.text(
                    0.02, 0.98,
                    pred_txt,
                    transform=ax.transAxes, va='top', ha='left', fontsize=8,
                    color='green',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                )

            # Small chance × wait coordinate inset (truth + pred scatter)
            if not (chance_only or len(comps_lbl) < 2):
                ch_t, wt_t = float(comps_lbl[0]), float(comps_lbl[1])
                ax_cw = ax.inset_axes([0.58, 0.10, 0.40, 0.38])
                _inset_axes.append(ax_cw)
                ax_cw.scatter(
                    [ch_t], [wt_t],
                    c='#1f77b4', s=42, zorder=4, marker='o',
                    edgecolors='white', linewidths=0.6, label='truth',
                )
                xs_cw = [ch_t]
                ys_cw = [wt_t]
                if comps_prd is not None and len(comps_prd) >= 2:
                    ch_p, wt_p = float(comps_prd[0]), float(comps_prd[1])
                    ax_cw.scatter(
                        [ch_p], [wt_p],
                        c='#2ca02c', s=48, zorder=5, marker='x',
                        linewidths=1.6, label='pred',
                    )
                    ax_cw.plot(
                        [ch_t, ch_p], [wt_t, wt_p],
                        color='gray', alpha=0.45, linewidth=0.9, zorder=3,
                    )
                    xs_cw.append(ch_p)
                    ys_cw.append(wt_p)
                # Axis limits with padding; keep origin in view when values are small
                x_lo, x_hi = min(xs_cw), max(xs_cw)
                y_lo, y_hi = min(ys_cw), max(ys_cw)
                x_pad = max(0.05 * (x_hi - x_lo), abs(x_hi) * 0.08, 0.15)
                y_pad = max(0.05 * (y_hi - y_lo), abs(y_hi) * 0.08, 0.15)
                ax_cw.set_xlim(x_lo - x_pad, x_hi + x_pad)
                ax_cw.set_ylim(y_lo - y_pad, y_hi + y_pad)
                ax_cw.axhline(0.0, color='0.7', linewidth=0.6, zorder=1)
                ax_cw.axvline(0.0, color='0.7', linewidth=0.6, zorder=1)
                ax_cw.set_xlabel('chance', fontsize=6, labelpad=1)
                ax_cw.set_ylabel('wait', fontsize=6, labelpad=1)
                ax_cw.tick_params(axis='both', labelsize=5, length=2, pad=1)
                ax_cw.grid(True, alpha=0.35, linewidth=0.5)
                ax_cw.set_title('chance × wait', fontsize=6, pad=2)
                ax_cw.legend(fontsize=5, loc='best', framealpha=0.75, borderpad=0.2)
                ax_cw.set_facecolor('#FAFAFA')
                for spine in ax_cw.spines.values():
                    spine.set_linewidth(0.7)

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
        mode_note = 'chance_only' if chance_only else 'chance+wait'
        line1 = (
            f'Yield ({mode_note}) — min_yield={min_yield_percent:g}%  |  {page_label}'
        )
        line2_parts = [
            f'horizon={max_h}',
            'orange=hits ≥ threshold',
            'chance from dataset cache',
        ]
        if not chance_only:
            line2_parts.append('inset: truth/pred on chance×wait')
        if pred_tensor is not None:
            line2_parts.append('green text=pred (denorm)')
        if fixed_order:
            line2_parts.append('chosen samples only')
        if interactive:
            line2_parts.append('[Next] / [Quit]')
        line2 = '  |  '.join(line2_parts)
        fig.suptitle(f'{line1}\n{line2}', fontsize=11, fontweight='bold')
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
            print("  Quit — closing yield browser.")
            plt.close(fig)

        btn_next.on_clicked(_on_next)
        btn_quit.on_clicked(_on_quit)

        def _on_key(event):
            if event.key in ('n', 'right', ' '):
                _draw_page()
            elif event.key in ('q', 'escape'):
                print("  Quit — closing yield browser.")
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', _on_key)
        if fixed_order:
            print(
                f"Yield browser ({num_available} chosen samples): "
                f"[Next] or N/Space = next page · [Quit] or Q = exit"
            )
        else:
            print("Yield browser: [Next] or N/Space = next page · [Quit] or Q = exit")

    _draw_page()
    plt.show(block=True)


# ═══════════════════════════════════════════════════════════



# Legacy names
visualize_linear_trends = visualize_trend
visualize_scattered_hope = visualize_yield
