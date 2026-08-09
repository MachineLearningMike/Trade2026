# labels.py
# Label modes, target construction, and training loss.
#
# Shared by data.py (label cache / dataset) and model.py (objective).
# Keep free of matplotlib and dataset I/O.

import torch
import torch.nn as nn
from typing import Optional, Tuple

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

    Parallel role to find_linear_regime (same window role, different math):
      - linear_regime → continuity / OLS on z-scored prices → (h, m, b)
      - scattered_hope → thresholded yield events on raw prices → (chance, wait)

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
        wait += t / yv

    return float(chance), float(wait)




# ═══════════════════════════════════════════════════════════
# LOSS (shared with model training / validation)
# ═══════════════════════════════════════════════════════════

def custom_loss(
    P,
    X,
    Y,
    slope_weight=0.1,
    intercept_weight=10,
    recon_weight=0.01,
    label_mode: str = 'linear_regime',
    chance_weight: float = 1.0,
    wait_weight: float = 1.0,
):
    """
    Multi-output prediction loss with optional reconstruction term.

    label_mode='linear_regime':
      output_dim=3 → (horizon, slope, intercept)
      output_dim=2 → (slope, intercept)
    label_mode='scattered_hope':
      output_dim=2 → (chance, wait)
    """
    predictions, patch_map, recon = P
    out_dim = predictions.shape[-1]
    if Y.shape[-1] != out_dim:
        raise ValueError(
            f"Target dim Y.shape[-1]={Y.shape[-1]} != predictions dim {out_dim}"
        )

    if label_mode in (LABEL_SCATTERED_HOPE, 'scattered_hope'):
        if out_dim != 2:
            raise ValueError(
                f"scattered_hope requires pred dim 2, got {out_dim}"
            )
        pred_chance, pred_wait = predictions[:, 0], predictions[:, 1]
        true_chance, true_wait = Y[:, 0], Y[:, 1]
        chance_loss = nn.functional.mse_loss(pred_chance, true_chance)
        wait_loss = nn.functional.mse_loss(pred_wait, true_wait)
        pred_loss = chance_weight * chance_loss + wait_weight * wait_loss
    elif label_mode in (LABEL_LINEAR_REGIME, 'linear_regime'):
        if out_dim == 3:
            pred_horizon, pred_slope, pred_intercept = (
                predictions[:, 0], predictions[:, 1], predictions[:, 2]
            )
            true_horizon, true_slope, true_intercept = Y[:, 0], Y[:, 1], Y[:, 2]
            horizon_loss = nn.functional.mse_loss(pred_horizon, true_horizon)
            slope_loss = nn.functional.mse_loss(pred_slope, true_slope)
            intercept_loss = nn.functional.mse_loss(pred_intercept, true_intercept)
            pred_loss = (
                horizon_loss
                + slope_weight * slope_loss
                + intercept_weight * intercept_loss
            )
        elif out_dim == 2:
            pred_slope, pred_intercept = predictions[:, 0], predictions[:, 1]
            true_slope, true_intercept = Y[:, 0], Y[:, 1]
            slope_loss = nn.functional.mse_loss(pred_slope, true_slope)
            intercept_loss = nn.functional.mse_loss(pred_intercept, true_intercept)
            pred_loss = slope_weight * slope_loss + intercept_weight * intercept_loss
        else:
            raise ValueError(f"Unsupported prediction dim {out_dim}; expected 2 or 3")
    else:
        raise ValueError(
            f"Unsupported label mode {label_mode!r}; "
            f"expected one of {VALID_LABEL_MODES}"
        )

    if recon is not None:
        recon_loss = nn.functional.mse_loss(recon, X)
        return pred_loss + recon_weight * recon_loss

    return pred_loss

