# labels.py
# Label modes, target construction, and training loss.
#
# Shared by data.py (label cache / dataset) and model.py (objective).
# Keep free of matplotlib and dataset I/O.
#
# Each label family is a class (find + loss). custom_loss dispatches to them.

import torch
import torch.nn as nn
from typing import Optional, Tuple, Type

# Default label fitness threshold (overridden by config['fitness_threshold']).
LABEL_FITNESS_THRESHOLD = 0.75

# Label family ids (config key ``label``).
LABEL_LINEAR_REGIME = 'linear_regime'
LABEL_SCATTERED_HOPE = 'scattered_hope'
VALID_LABEL_MODES = (LABEL_LINEAR_REGIME, LABEL_SCATTERED_HOPE)

LABEL_FULL_DIM_LINEAR = 3
LABEL_FULL_DIM_SCATTERED = 2
LABEL_FULL_DIM_SCATTERED_CHANCE = 1

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
# LABEL FAMILY: LINEAR TREND
# ═══════════════════════════════════════════════════════════

class Label_Linear_Trend:
    """
    Linear-regime labels: full cache (horizon, slope, intercept).

    Config id: 'linear_regime'
    Price space: typically z-scored (caller responsibility).
    """

    name = LABEL_LINEAR_REGIME
    full_dim = LABEL_FULL_DIM_LINEAR

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
    def loss(
        cls,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        slope_weight: float = 0.1,
        intercept_weight: float = 10.0,
        **_kwargs,
    ) -> torch.Tensor:
        """MSE on (h,m,b) or (m,b) depending on last dim."""
        out_dim = predictions.shape[-1]
        if targets.shape[-1] != out_dim:
            raise ValueError(
                f"{cls.name} loss: target dim {targets.shape[-1]} != pred dim {out_dim}"
            )
        if out_dim == 3:
            horizon_loss = nn.functional.mse_loss(predictions[:, 0], targets[:, 0])
            slope_loss = nn.functional.mse_loss(predictions[:, 1], targets[:, 1])
            intercept_loss = nn.functional.mse_loss(predictions[:, 2], targets[:, 2])
            return (
                horizon_loss
                + slope_weight * slope_loss
                + intercept_weight * intercept_loss
            )
        if out_dim == 2:
            slope_loss = nn.functional.mse_loss(predictions[:, 0], targets[:, 0])
            intercept_loss = nn.functional.mse_loss(predictions[:, 1], targets[:, 1])
            return slope_weight * slope_loss + intercept_weight * intercept_loss
        raise ValueError(
            f"{cls.name} loss: unsupported pred dim {out_dim}; expected 2 or 3"
        )


# ═══════════════════════════════════════════════════════════
# LABEL FAMILY: SCATTERED HOPE
# ═══════════════════════════════════════════════════════════

class Label_Scattered_Hope:
    """
    Scattered-hope labels on raw prices: (chance, wait) or (chance,).

    Config id: 'scattered_hope'
    Price space: raw (true % yield); caller must not z-score inputs.
    """

    name = LABEL_SCATTERED_HOPE

    @classmethod
    def full_dim(cls, chance_only: bool = False) -> int:
        return (
            LABEL_FULL_DIM_SCATTERED_CHANCE
            if chance_only
            else LABEL_FULL_DIM_SCATTERED
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
        Thresholded yield events vs history[-1].

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
            if yv < min_yield_percent:
                continue
            chance += (yv - min_yield_percent)
            if not chance_only:
                wait += t / yv

        if chance_only:
            return (float(chance),)
        return (float(chance), float(wait))

    @classmethod
    def loss(
        cls,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        chance_weight: float = 1.0,
        wait_weight: float = 1.0,
        chance_only: bool = False,
        **_kwargs,
    ) -> torch.Tensor:
        """MSE on (chance[, wait])."""
        out_dim = predictions.shape[-1]
        if targets.shape[-1] != out_dim:
            raise ValueError(
                f"{cls.name} loss: target dim {targets.shape[-1]} != pred dim {out_dim}"
            )
        if chance_only or out_dim == 1:
            if out_dim != 1:
                raise ValueError(
                    f"chance_only requires pred dim 1, got {out_dim}"
                )
            return chance_weight * nn.functional.mse_loss(
                predictions[:, 0], targets[:, 0]
            )
        if out_dim == 2:
            chance_loss = nn.functional.mse_loss(predictions[:, 0], targets[:, 0])
            wait_loss = nn.functional.mse_loss(predictions[:, 1], targets[:, 1])
            return chance_weight * chance_loss + wait_weight * wait_loss
        raise ValueError(
            f"{cls.name} loss: require dim 1 (chance_only) or 2, got {out_dim}"
        )


# ═══════════════════════════════════════════════════════════
# REGISTRY + DISPATCH
# ═══════════════════════════════════════════════════════════

LABEL_CLASSES: dict[str, Type] = {
    LABEL_LINEAR_REGIME: Label_Linear_Trend,
    LABEL_SCATTERED_HOPE: Label_Scattered_Hope,
    # aliases used in older configs / prose
    'linear_trend': Label_Linear_Trend,
}


def get_label_class(label_mode: str) -> Type:
    if label_mode not in LABEL_CLASSES:
        raise ValueError(
            f"Unknown label mode {label_mode!r}; expected one of {tuple(LABEL_CLASSES)}"
        )
    return LABEL_CLASSES[label_mode]


def resolve_hope_output_dim(chance_only: bool) -> int:
    """Scattered-hope head / label width: 1 if chance_only else 2."""
    return Label_Scattered_Hope.resolve_output_dim(chance_only)


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
    mode = config.get('label', LABEL_LINEAR_REGIME)
    return get_label_class(mode).resolve_future_window(config)


def resolve_model_past_len(config: dict) -> int:
    mode = config.get('label', LABEL_LINEAR_REGIME)
    return get_label_class(mode).resolve_model_past_len(config)


# Backward-compatible free functions (delegate to classes)

def find_linear_regime(*args, **kwargs):
    return Label_Linear_Trend.find(*args, **kwargs)


def find_scattered_hope(*args, **kwargs):
    return Label_Scattered_Hope.find(*args, **kwargs)


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
    chance_only: bool = False,
):
    """
    Multi-output prediction loss; delegates to the active label class.
    Optional reconstruction term uses X vs recon.
    """
    predictions, patch_map, recon = P
    out_dim = predictions.shape[-1]
    if Y.shape[-1] != out_dim:
        raise ValueError(
            f"Target dim Y.shape[-1]={Y.shape[-1]} != predictions dim {out_dim}"
        )

    label_cls = get_label_class(label_mode)
    pred_loss = label_cls.loss(
        predictions,
        Y,
        slope_weight=slope_weight,
        intercept_weight=intercept_weight,
        chance_weight=chance_weight,
        wait_weight=wait_weight,
        chance_only=chance_only,
    )

    if recon is not None:
        recon_loss = nn.functional.mse_loss(recon, X)
        return pred_loss + recon_weight * recon_loss
    return pred_loss
