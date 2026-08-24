# labels.py
# Label modes, target construction, training loss, and visualization.
#
# Shared by data.py (label cache / dataset), model.py (objective), TRAIN, and INFER.
#
# Each label family is a class (find + loss). Bind Label_*.loss for training.
# Families: Label_Trend, Label_Yield, Label_Latent (density NLL; shared model).
# Visualization helpers (formerly label_viz.py) live at the bottom.

import random
import time
from typing import Any, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import StudentT
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy import special

# Default label fitness threshold (overridden by config['fitness_threshold']).
LABEL_FITNESS_THRESHOLD = 0.75

# Label family ids (config key ``label``).
LABEL_TREND = 'trend'
LABEL_YIELD = 'yield'
LABEL_LATENT = 'latent'
VALID_LABEL_MODES = (LABEL_TREND, LABEL_YIELD, LABEL_LATENT)

LABEL_FULL_DIM_TREND = 3
LABEL_FULL_DIM_YIELD = 2
LABEL_FULL_DIM_YIELD_SINGLE = 1  # chance-only or near-only head
# Latent: scalar observation in the label cache; model head width from density mode.
LABEL_FULL_DIM_LATENT = 1

# Latent parametric density (config key ``latent_distribution``).
# See Basic/Predict_normal_disttribution.py and Predict_student_t_disttribution.py.
LATENT_DIST_NORMAL = 'normal'
LATENT_DIST_STUDENT = 'student'
VALID_LATENT_DISTRIBUTIONS = (LATENT_DIST_NORMAL, LATENT_DIST_STUDENT)
# Model head widths (y_pred last dim) for each density.
LABEL_OUTPUT_DIM_NORMAL = 2   # (mu, var_raw)
LABEL_OUTPUT_DIM_STUDENT = 3  # (location, scale_raw, df_raw)

# Yield model target selection (config key ``yield_target``).
# Cache always stores full (chance, near); this selects the model slice.
YIELD_TARGET_BOTH = 'both'
YIELD_TARGET_CHANCE = 'chance'
YIELD_TARGET_NEAR = 'near'
VALID_YIELD_TARGETS = (
    YIELD_TARGET_BOTH,
    YIELD_TARGET_CHANCE,
    YIELD_TARGET_NEAR,
)

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
        return int(config.get('future_window', 50))

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
# YIELD TARGET RESOLUTION
# ═══════════════════════════════════════════════════════════

def resolve_yield_target(config: Optional[dict] = None, **overrides) -> str:
    """
    Resolve yield model target: 'both' | 'chance' | 'near'.

    Reads overrides['yield_target'] or config['yield_target']; default 'both'.
    Legacy aliases: wait → near, chance_wait → both.
    """
    cfg = dict(config or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    raw = cfg.get('yield_target', YIELD_TARGET_BOTH)
    if raw is None or str(raw).strip() == '':
        return YIELD_TARGET_BOTH
    yt = str(raw).strip().lower()
    if yt in ('both', 'chance_near', 'chance_wait', 'all'):
        yt = YIELD_TARGET_BOTH
    elif yt in ('c',):
        yt = YIELD_TARGET_CHANCE
    elif yt in ('n', 'w', 'wait'):  # 'wait' legacy → near
        yt = YIELD_TARGET_NEAR
    if yt not in VALID_YIELD_TARGETS:
        raise ValueError(
            f"yield_target must be one of {VALID_YIELD_TARGETS}, got {raw!r}"
        )
    return yt


# ═══════════════════════════════════════════════════════════
# LABEL FAMILY: YIELD
# ═══════════════════════════════════════════════════════════

class Label_Yield:
    """
    Yield labels on raw prices.

    Config id: 'yield'
    Price space: raw (true % yield); caller must not z-score inputs.

    Full cache vector is always (chance, near). Model target is selected by
    ``yield_target``: 'both' | 'chance' | 'near'.
    """

    name = LABEL_YIELD

    @classmethod
    def full_cache_dim(cls) -> int:
        """Stored label vector width (always chance + near)."""
        return LABEL_FULL_DIM_YIELD

    @classmethod
    def full_dim(cls, yield_target: str = YIELD_TARGET_BOTH, **_kwargs) -> int:
        """Model head width for the given yield_target."""
        return cls.resolve_output_dim(yield_target=yield_target)

    @classmethod
    def resolve_future_window(cls, config: dict) -> int:
        return int(config.get('future_window', 50))

    @classmethod
    def resolve_model_past_len(cls, config: dict) -> int:
        return cls.resolve_future_window(config)

    @classmethod
    def resolve_output_dim(
        cls,
        yield_target: str = YIELD_TARGET_BOTH,
        **_kwargs,
    ) -> int:
        yt = resolve_yield_target({'yield_target': yield_target})
        if yt == YIELD_TARGET_BOTH:
            return LABEL_FULL_DIM_YIELD
        return LABEL_FULL_DIM_YIELD_SINGLE

    @classmethod
    def target_slice(
        cls,
        output_dim: int,
        yield_target: str = YIELD_TARGET_BOTH,
        **_kwargs,
    ) -> slice:
        """
        Slice into full cache vector [chance, near].

          both   → [0:2]
          chance → [0:1]
          near   → [1:2]
        """
        yt = resolve_yield_target({'yield_target': yield_target})
        expected = cls.resolve_output_dim(yield_target=yt)
        if output_dim != expected:
            raise ValueError(
                f"{cls.name} with yield_target={yt!r} requires "
                f"output_dim={expected}, got {output_dim}"
            )
        if yt == YIELD_TARGET_BOTH:
            return slice(0, 2)
        if yt == YIELD_TARGET_CHANCE:
            return slice(0, 1)
        if yt == YIELD_TARGET_NEAR:
            return slice(1, 2)
        raise ValueError(f"Unknown yield_target {yt!r}")

    @classmethod
    def target_names(
        cls,
        output_dim: int,
        yield_target: str = YIELD_TARGET_BOTH,
        **_kwargs,
    ) -> Tuple[str, ...]:
        yt = resolve_yield_target({'yield_target': yield_target})
        if yt == YIELD_TARGET_BOTH:
            return ('chance', 'near')
        if yt == YIELD_TARGET_CHANCE:
            return ('chance',)
        if yt == YIELD_TARGET_NEAR:
            return ('near',)
        raise ValueError(f"Unknown yield_target {yt!r}")

    @staticmethod
    def _as_1d_float64(x) -> np.ndarray:
        """Host 1-D float64 view; accepts torch / numpy / list / scalar."""
        if x is None:
            return np.zeros(0, dtype=np.float64)
        if torch.is_tensor(x):
            # Single-series label: no GPU benefit; one host transfer, then numpy.
            # Keep full precision (do not cast through float32).
            arr = x.detach().reshape(-1).to(dtype=torch.float64).cpu().numpy()
            return np.asarray(arr, dtype=np.float64)
        return np.asarray(x, dtype=np.float64).reshape(-1)

    @classmethod
    def find(
        cls,
        future,
        current,
        min_yield_percent: float,
        max_hope_future_horizon: Optional[int] = None,
        yield_target: Optional[str] = None,
    ):
        """
        Thresholded yield events vs the known bot-time price ``current`` (= p[t]).

        TelePred / ensemble_plan signature:
            Y(t) = label( p[t+1 : t+1+delta2], p[t] )
                 = find(future, current, ...)

        ``future`` is the raw price path of length delta2; ``current`` is the
        scalar anchor p[t] (not a history window).

        Single-series only (not batched). Inputs may be torch / numpy / list /
        scalar; they are converted once to host float64, then computed in numpy
        (no per-bar tensor casts or GPU syncs).

        Always computes both chance and near. Return shape:
          - default / cache (yield_target None or 'both'): (chance, near)
          - yield_target='chance'|'near': single-component tuple

        Prefer always caching (chance, near) and slicing via target_slice.
        """
        fut = cls._as_1d_float64(future)
        # Accept scalar or length-1 array/tensor for p[t].
        cur = cls._as_1d_float64(current)
        if cur.size == 0:
            current_px = float("nan")
        else:
            # Scalar / length-1 → that value; longer series → last bar (legacy).
            current_px = float(cur[-1])

        n = int(fut.shape[0])
        if max_hope_future_horizon is not None:
            n = min(n, max(0, int(max_hope_future_horizon)))

        chance, near = 0.0, 0.0

        if n > 0 and np.isfinite(current_px):
            eps = 1e-8
            if abs(current_px) > eps:
                denom = current_px
            else:
                denom = current_px + eps if current_px >= 0.0 else current_px - eps

            # Note softplus is intuitively active over (-4, +infinity). It's almost linear over 4.0.
            # softplus(-4, 0, 4) = (0.0181, 0.6931, 4.0181)
            # Scale surplus yield so mild negatives land in the softplus ramp.
            _yield = (fut[:n] - current_px) / denom * 100.0
            # (-gentle, 0) yield maps to (-4, 0), inside the active softplus region.
            gentle = 0.3    # How gently to ignore negative yield.
            scaled_yield = _yield * 4.0 / gentle
            cared_yield = special.softplus(scaled_yield)
            # Far-future bars are down-weighted: index from the end via
            # arange(n-1 … 0), then logistic centered at warmingup/2.
            # expit([60, 30, 0, -1, -2, -3, -4]) ≈ [1, 1, 0.5, 0.27, 0.12, 0.05, 0.02]
            warmingup = 8.0 # A far future price should get fully visible slowly for 'warmingup' time steps.
            # Must be an ndarray: reversed(range(n)) is an iterator and cannot
            # do arithmetic with a float.
            rev_t = np.arange(n - 1, -1, -1, dtype=np.float64)
            viz_t = special.expit(rev_t - warmingup / 2.0)     # sigmoid
            viz_cared_yield = cared_yield * viz_t
            chance = float(viz_cared_yield.sum())
            if chance > eps:
                near = float(n - (viz_cared_yield * rev_t).sum() / chance)
            else:
                near = 0.0

        full = (float(chance), float(near))
        if yield_target is None:
            return full

        yt = resolve_yield_target({'yield_target': yield_target})
        if yt == YIELD_TARGET_BOTH:
            return full
        if yt == YIELD_TARGET_CHANCE:
            return (full[0],)
        if yt == YIELD_TARGET_NEAR:
            return (full[1],)
        return full

    @classmethod
    def make_auxiliary(cls, config: Optional[dict] = None, **overrides) -> dict:
        """
        Class-private loss knobs for yield.

        Keys: chance_weight, near_weight, yield_target
        """
        cfg = config or {}
        yt = resolve_yield_target(cfg, **overrides)
        # near_loss_weight preferred; wait_loss_weight kept as legacy alias.
        near_w = cfg.get('near_loss_weight', cfg.get('wait_loss_weight', 1.0))
        aux = {
            'chance_weight': float(cfg.get('chance_loss_weight', 1.0)),
            'near_weight': float(near_w),
            'yield_target': yt,
        }
        for k, v in overrides.items():
            if k == 'yield_target':
                continue
            if k == 'wait_weight' and 'near_weight' not in overrides:
                aux['near_weight'] = v
                continue
            if k == 'wait_loss_weight' and 'near_loss_weight' not in overrides:
                aux['near_weight'] = float(v)
                continue
            aux[k] = v
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

        MSE on model targets for the active yield_target.
        ``x`` unused for yield (kept for API uniformity).
        """
        aux = auxiliary or {}
        chance_weight = float(aux.get('chance_weight', 1.0))
        near_weight = float(
            aux.get('near_weight', aux.get('wait_weight', 1.0))
        )
        yt = resolve_yield_target(aux)

        out_dim = y_pred.shape[-1]
        if y_truth.shape[-1] != out_dim:
            raise ValueError(
                f"{cls.name} loss: target dim {y_truth.shape[-1]} != pred dim {out_dim}"
            )

        if yt == YIELD_TARGET_BOTH:
            if out_dim != 2:
                raise ValueError(
                    f"yield_target='both' requires pred dim 2, got {out_dim}"
                )
            chance_loss = nn.functional.mse_loss(y_pred[:, 0], y_truth[:, 0])
            near_loss = nn.functional.mse_loss(y_pred[:, 1], y_truth[:, 1])
            return chance_weight * chance_loss + near_weight * near_loss

        if yt in (YIELD_TARGET_CHANCE, YIELD_TARGET_NEAR):
            if out_dim != 1:
                raise ValueError(
                    f"yield_target={yt!r} requires pred dim 1, got {out_dim}"
                )
            w = chance_weight if yt == YIELD_TARGET_CHANCE else near_weight
            return w * nn.functional.mse_loss(y_pred[:, 0], y_truth[:, 0])

        raise ValueError(
            f"{cls.name} loss: unknown yield_target {yt!r}"
        )


# ═══════════════════════════════════════════════════════════
# LATENT DISTRIBUTION RESOLUTION
# ═══════════════════════════════════════════════════════════

def resolve_latent_distribution(
    config: Optional[dict] = None,
    **overrides,
) -> str:
    """
    Resolve latent density mode: 'normal' | 'student'.

    Reads overrides / config key ``latent_distribution``. Accepts aliases:
      normal  ← normal, gaussian, normal_distribution, n
      student ← student, student_t, student_distribution, t
    """
    cfg = dict(config or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    raw = cfg.get('latent_distribution', LATENT_DIST_NORMAL)
    if raw is None or str(raw).strip() == '':
        return LATENT_DIST_NORMAL
    s = str(raw).strip().lower().replace('-', '_').replace(' ', '_')
    if s in (
        LATENT_DIST_NORMAL, 'gaussian', 'normal_distribution',
        'normal_distirbution', 'n',
    ):
        return LATENT_DIST_NORMAL
    if s in (
        LATENT_DIST_STUDENT, 'student_t', 'student_t_distribution',
        'student_distribution', 'student_distirbution', 't',
    ):
        return LATENT_DIST_STUDENT
    raise ValueError(
        f"latent_distribution must be one of {VALID_LATENT_DISTRIBUTIONS} "
        f"(aliases: normal_distribution, student_distribution), got {raw!r}"
    )


# Head scale channels are unconstrained; mapped here.
# Floor variance: label-normalized y has std ~ 1. If sigma -> 1e-4,
# (y-mu)^2/sigma^2 explodes on typical shuffled residuals and NLL rises
# for the rest of the epoch even though the data order is random.
_VAR_MIN = 1e-3    # sigma >= ~0.032 in observation units
_SCALE_MIN = 1e-2
_DF_SOFTPLUS_MAX = 20.0  # df in (2, 2+cap]


def _gaussian_variance(var_raw: torch.Tensor) -> torch.Tensor:
    """True variance: softplus(var_raw) + _VAR_MIN (train and inference)."""
    return F.softplus(var_raw) + _VAR_MIN


def _student_scale(scale_raw: torch.Tensor) -> torch.Tensor:
    """True scale: softplus(scale_raw) + _SCALE_MIN (train and inference)."""
    return F.softplus(scale_raw) + _SCALE_MIN


def _gaussian_nll_loss(
    mu: torch.Tensor,
    var_raw: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    NLL for diagonal Gaussian (Basic/Predict_normal_disttribution.py).

    Head channel 1 is unconstrained ``var_raw``.
    True variance is ``softplus(var_raw) + _VAR_MIN`` (same at train and test).
    """
    finite = torch.isfinite(target) & torch.isfinite(mu) & torch.isfinite(var_raw)
    if not finite.any():
        return mu.sum() * 0.0

    var = _gaussian_variance(var_raw)
    log_var = torch.log(var)
    residual2 = (target - mu).pow(2)
    # 0.5 * (log σ² + (y-μ)² / σ²)  — omit 2π constant
    loss = 0.5 * (log_var + residual2 / var)
    loss = loss.masked_fill(~finite, 0.0)
    return loss.sum() / finite.sum().clamp(min=1).to(loss.dtype)


def _student_t_nll_loss(
    location: torch.Tensor,
    scale_raw: torch.Tensor,
    df_raw: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    NLL for Student-t (Basic/Predict_student_t_disttribution.py).

    scale = softplus(scale_raw) + _SCALE_MIN
    df    = 2 + softplus(df_raw)  (capped)
    """
    finite = (
        torch.isfinite(target)
        & torch.isfinite(location)
        & torch.isfinite(scale_raw)
        & torch.isfinite(df_raw)
    )
    if not finite.any():
        return location.sum() * 0.0

    scale = _student_scale(scale_raw)
    df = 2.0 + F.softplus(df_raw).clamp(max=_DF_SOFTPLUS_MAX)
    dist = StudentT(df=df, loc=location, scale=scale)
    nll = -dist.log_prob(target)
    nll = torch.nan_to_num(nll, nan=0.0, posinf=1e4, neginf=1e4)
    nll = nll.masked_fill(~finite, 0.0)
    return nll.sum() / finite.sum().clamp(min=1).to(nll.dtype)


def decode_latent_params(
    y_pred: torch.Tensor,
    latent_distribution: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Decode unconstrained density-head channels to (location, scale, df).

    y_pred last dim:
      normal  → 2 = (mu, var_raw);     scale = sqrt(softplus(var_raw)+eps)
      student → 3 = (loc, scale_raw, df_raw)

    Returns location/scale with the same leading shape as y_pred (df is None
    for the Gaussian). Does not denormalize into observation space.
    """
    dist = resolve_latent_distribution(
        {'latent_distribution': latent_distribution}
        if latent_distribution is not None
        else None,
        latent_distribution=latent_distribution,
    )
    if y_pred.dim() == 0:
        raise ValueError("y_pred must have a last dimension of density params")
    if dist == LATENT_DIST_NORMAL:
        if y_pred.shape[-1] != LABEL_OUTPUT_DIM_NORMAL:
            raise ValueError(
                f"normal head expects last dim {LABEL_OUTPUT_DIM_NORMAL}, "
                f"got {y_pred.shape[-1]}"
            )
        location = y_pred[..., 0]
        scale = _gaussian_variance(y_pred[..., 1]).sqrt()
        return location, scale, None
    if y_pred.shape[-1] != LABEL_OUTPUT_DIM_STUDENT:
        raise ValueError(
            f"student head expects last dim {LABEL_OUTPUT_DIM_STUDENT}, "
            f"got {y_pred.shape[-1]}"
        )
    location = y_pred[..., 0]
    scale = _student_scale(y_pred[..., 1])
    df = 2.0 + F.softplus(y_pred[..., 2]).clamp(max=_DF_SOFTPLUS_MAX)
    return location, scale, df


def latent_params_to_obs_space(
    location: torch.Tensor,
    scale: torch.Tensor,
    label_mean: torch.Tensor,
    label_std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map location/scale from label-normalized space to cache observation space."""
    mean = torch.as_tensor(label_mean, device=location.device, dtype=location.dtype)
    std = torch.as_tensor(label_std, device=location.device, dtype=location.dtype)
    mean = mean.reshape(-1)[:1]
    std = std.reshape(-1)[:1]
    std = torch.where(std.abs() < 1e-8, torch.ones_like(std), std)
    # Broadcast (1,) over leading dims of location/scale.
    while mean.dim() < location.dim():
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return location * std + mean, scale * std.abs()


# ═══════════════════════════════════════════════════════════
# LABEL FAMILY: LATENT  (parametric density over a scalar observation)
# ═══════════════════════════════════════════════════════════

class Label_Latent:
    """
    Latent labels: infer a density for the *next* bar (first future price).

    The realized observation is ``future[0]`` — one sample from a latent
    price law conditioned on the historical window. The model outputs the
    parameters of that law from market history (it does not emit a point
    forecast; the NLL trains the density against the realized next bar).

    Config id: 'latent'
    Shares ``model.MarketTransformerWithHeads`` with trend/yield.

    Mode (config ``latent_distribution``):
      'normal'  → y_pred (B, 2) = (mu, var_raw);   var = softplus(var_raw)+eps
      'student' → y_pred (B, 3) = (loc, scale_raw, df_raw)
                  scale = softplus(scale_raw)+eps; df = 2+softplus(df_raw)

    Label cache stores the realized next-bar price (dim 1). Model ``output_dim``
    is forced from the density mode (2 or 3). ``y_pred`` and ``y_truth``
    therefore have different last dims by design — loss is NLL, not MSE.
    """

    name = LABEL_LATENT

    @classmethod
    def resolve_distribution(cls, config: Optional[dict] = None, **overrides) -> str:
        return resolve_latent_distribution(config, **overrides)

    @classmethod
    def resolve_output_dim(
        cls,
        config: Optional[dict] = None,
        latent_distribution: Optional[str] = None,
        **overrides,
    ) -> int:
        """Model head width from density mode (not the label-cache width)."""
        dist = resolve_latent_distribution(
            config,
            latent_distribution=latent_distribution,
            **overrides,
        )
        if dist == LATENT_DIST_NORMAL:
            return LABEL_OUTPUT_DIM_NORMAL
        if dist == LATENT_DIST_STUDENT:
            return LABEL_OUTPUT_DIM_STUDENT
        raise ValueError(f"Unknown latent_distribution {dist!r}")

    @classmethod
    def resolve_latent_dim(cls, config: Optional[dict] = None, **overrides) -> int:
        """Alias of resolve_output_dim (model head width)."""
        return cls.resolve_output_dim(config, **overrides)

    @classmethod
    def full_cache_dim(cls) -> int:
        """Stored label vector width (scalar observation)."""
        return LABEL_FULL_DIM_LATENT

    @classmethod
    def full_dim(cls, config: Optional[dict] = None, **overrides) -> int:
        """Model head width for the active density."""
        return cls.resolve_output_dim(config, **overrides)

    @classmethod
    def resolve_future_window(cls, config: dict) -> int:
        # Need at least 1 future bar so find() can return future[0].
        return max(1, int(config.get('future_window', 50)))

    @classmethod
    def resolve_model_past_len(cls, config: dict) -> int:
        return cls.resolve_future_window(config)

    @classmethod
    def target_slice(cls, output_dim: int, **_kwargs) -> slice:
        """
        Slice into the label cache (always the scalar observation).

        ``output_dim`` is the *model* head width and is ignored for the slice;
        cache is always dim 1.
        """
        del output_dim
        return slice(0, LABEL_FULL_DIM_LATENT)

    @classmethod
    def target_names(
        cls,
        output_dim: int,
        latent_distribution: Optional[str] = None,
        **_kwargs,
    ) -> Tuple[str, ...]:
        """
        Names for the model head when ``output_dim`` is 2/3, else cache name.
        """
        if output_dim == LABEL_FULL_DIM_LATENT:
            return ('y',)
        dist = resolve_latent_distribution(
            {'latent_distribution': latent_distribution}
            if latent_distribution is not None
            else None,
            latent_distribution=latent_distribution,
        )
        if dist == LATENT_DIST_NORMAL and output_dim == LABEL_OUTPUT_DIM_NORMAL:
            return ('mu', 'var_raw')
        if dist == LATENT_DIST_STUDENT and output_dim == LABEL_OUTPUT_DIM_STUDENT:
            return ('location', 'scale_raw', 'df_raw')
        # Fallback: generic param names
        return tuple(f'param_{i}' for i in range(output_dim))

    @classmethod
    def find(
        cls,
        future,
        history,
        **_kwargs,
    ):
        """
        Realized next-bar price: first bar of ``future`` (``future[0]``).

        That price is the observation the density is fitted to — one draw
        from the latent next-bar law, conditioned on ``history``. ``history``
        is unused for the scalar itself (kept so data.py can call find the
        same way as trend/yield).

        data.py may pass window-z-scored prices; this still returns
        ``future[0]`` in that space.
        """
        del history
        if future is None:
            return (0.0,)
        # Accept torch / numpy / list; always index the first future bar.
        if torch.is_tensor(future):
            fut = future.reshape(-1)
            if fut.numel() == 0:
                return (0.0,)
            return (float(fut[0].item()),)
        fut = np.asarray(future).reshape(-1)
        if fut.size == 0:
            return (0.0,)
        return (float(fut[0]),)

    @classmethod
    def make_auxiliary(cls, config: Optional[dict] = None, **overrides) -> dict:
        """
        Loss knobs for latent density NLL.

        Keys: latent_weight, latent_distribution, output_dim
        """
        cfg = config or {}
        dist = resolve_latent_distribution(cfg, **overrides)
        aux = {
            'latent_weight': float(cfg.get('latent_loss_weight', 1.0)),
            'latent_distribution': dist,
            'output_dim': cls.resolve_output_dim(cfg, latent_distribution=dist),
        }
        for k, v in overrides.items():
            if v is not None and k != 'latent_distribution':
                aux[k] = v
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
        Negative log-likelihood under the configured density.

        Signature (same call shape as trend/yield; different semantics)::

            loss(x: Tensor, y_pred: Tensor, y_truth: Tensor, auxiliary=None)

        y_pred:
          normal  → (B, 2) = (mu, var_raw)
          student → (B, 3) = (location, scale_raw, df_raw)
        y_truth:
          (B,) or (B, 1) realized next-bar price future[0] (cache dim 1)
        """
        aux = auxiliary or {}
        weight = float(aux.get('latent_weight', 1.0))
        dist = resolve_latent_distribution(aux)

        if not torch.is_tensor(y_pred):
            raise TypeError(
                f"{cls.name}.loss expects y_pred to be a Tensor, "
                f"got {type(y_pred).__name__}"
            )
        if not torch.is_tensor(y_truth):
            raise TypeError(
                f"{cls.name}.loss expects y_truth to be a Tensor, "
                f"got {type(y_truth).__name__}"
            )

        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
        # truth → (B, 1)
        if y_truth.dim() == 0:
            y_truth = y_truth.view(1, 1)
        elif y_truth.dim() == 1:
            y_truth = y_truth.unsqueeze(-1)
        else:
            y_truth = y_truth.reshape(y_truth.shape[0], -1)[:, :1]

        expected = (
            LABEL_OUTPUT_DIM_NORMAL
            if dist == LATENT_DIST_NORMAL
            else LABEL_OUTPUT_DIM_STUDENT
        )
        if y_pred.shape[-1] != expected:
            raise ValueError(
                f"{cls.name} loss ({dist}): y_pred last dim must be {expected}, "
                f"got {y_pred.shape[-1]}"
            )

        # Finite-ize only; scale channels use softplus inside the NLL helpers.
        y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=50.0, neginf=-50.0)
        y_truth = torch.nan_to_num(y_truth, nan=0.0, posinf=0.0, neginf=0.0)

        del x  # reserved for conditioning terms later

        if dist == LATENT_DIST_NORMAL:
            mu = y_pred[:, 0:1]
            var_raw = y_pred[:, 1:2]
            nll = _gaussian_nll_loss(mu, var_raw, y_truth)
        else:
            location = y_pred[:, 0:1]
            scale_raw = y_pred[:, 1:2]
            df_raw = y_pred[:, 2:3]
            nll = _student_t_nll_loss(location, scale_raw, df_raw, y_truth)
        if not torch.isfinite(nll):
            # Last-resort: keep the graph alive with a zero that still depends on y_pred.
            nll = y_pred.sum() * 0.0
        return weight * nll


# ═══════════════════════════════════════════════════════════
# REGISTRY + DISPATCH
# ═══════════════════════════════════════════════════════════

LABEL_CLASSES: dict[str, Type] = {
    LABEL_TREND: Label_Trend,
    LABEL_YIELD: Label_Yield,
    LABEL_LATENT: Label_Latent,
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


def resolve_hope_output_dim(yield_target: Optional[str] = None) -> int:
    """Yield head / label width from yield_target."""
    return Label_Yield.resolve_output_dim(
        yield_target=yield_target or YIELD_TARGET_BOTH,
    )


def label_target_slice(
    label_mode: str,
    output_dim: int,
    yield_target: Optional[str] = None,
    latent_distribution: Optional[str] = None,
) -> slice:
    return get_label_class(label_mode).target_slice(
        output_dim,
        yield_target=yield_target or YIELD_TARGET_BOTH,
        latent_distribution=latent_distribution,
    )


def label_target_names(
    label_mode: str,
    output_dim: int,
    yield_target: Optional[str] = None,
    latent_distribution: Optional[str] = None,
) -> Tuple[str, ...]:
    return get_label_class(label_mode).target_names(
        output_dim,
        yield_target=yield_target or YIELD_TARGET_BOTH,
        latent_distribution=latent_distribution,
    )


def resolve_future_window(config: dict) -> int:
    mode = config.get('label', LABEL_TREND)
    return get_label_class(mode).resolve_future_window(config)


def resolve_model_past_len(config: dict) -> int:
    mode = config.get('label', LABEL_TREND)
    return get_label_class(mode).resolve_model_past_len(config)


def sync_output_dim(config: dict) -> int:
    """
    Force ``config['output_dim']`` from the active label family where required.

    - yield   → from ``yield_target`` (always)
    - latent  → from ``latent_distribution`` (always)
    - trend   → leave ``config['output_dim']`` unchanged (2 or 3)

    Returns the resulting ``output_dim``. Used so CLI modes and training
    share one checkpoint-key / head-width path.
    """
    cls = get_label_class(config.get('label', LABEL_TREND))
    if cls is Label_Yield:
        config['output_dim'] = Label_Yield.resolve_output_dim(
            yield_target=resolve_yield_target(config),
        )
    elif cls is Label_Latent:
        config['output_dim'] = Label_Latent.resolve_output_dim(config)
    return int(config.get('output_dim', LABEL_FULL_DIM_TREND))


# Backward-compatible free functions (delegate to classes)

def find_trend(*args, **kwargs):
    return Label_Trend.find(*args, **kwargs)


def find_yield(future, current, *args, **kwargs):
    """Y(t) = label(p[t+1:t+1+delta2], p[t]) → find(future, current, ...)."""
    return Label_Yield.find(future, current, *args, **kwargs)


def find_latent(*args, **kwargs):
    return Label_Latent.find(*args, **kwargs)


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

    which is ``Label_*.loss`` with class-private ``auxiliary`` already bound.
    If the model returns ``(predictions, patch_map, recon)``, only the head
    tensor is forwarded to the class loss. The trainer does not branch on
    label mode.
    """
    label_cls = get_label_class(label_mode)
    auxiliary = label_cls.make_auxiliary(config, **overrides)

    def loss_fn(
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_truth: torch.Tensor,
    ) -> torch.Tensor:
        # Trainer may pass full model tuple; class loss wants the head tensor.
        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]
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
    yield_target: Optional[str] = None,
    num_samples: int = 6,
    figsize: Tuple[int, int] = (15, 10),
    interactive: bool = True,
    pred_tensor: Optional[torch.Tensor] = None,
    fixed_order: bool = False,
    sample_ids: Optional[List[int]] = None,
    past_context: int = 15,
    label_mean: Optional[torch.Tensor] = None,
    label_std: Optional[torch.Tensor] = None,
):
    """
    Visualize yield labels in raw price space.

    Each panel:
      - recent past + future actual prices
      - horizontal line at reference price history[-1]
      - orange markers on future bars where yield % >= min_yield_percent
      - title chance/near from ``label_tensor`` (natural/denorm units)
      - chance×near inset in normalized space when yield_target='both'
    """
    yt = resolve_yield_target({'yield_target': yield_target})
    show_cw_inset = yt == YIELD_TARGET_BOTH

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

    # Label z-score stats for the chance×near inset (model target space)
    def _as_stat_vec(v, dim: int):
        if v is None:
            return None
        if hasattr(v, 'detach'):
            arr = v.detach().cpu().float().flatten().numpy()
        else:
            arr = np.asarray(v, dtype=np.float64).ravel()
        if arr.size < dim:
            raise ValueError(
                f"label stat length {arr.size} < required dim {dim}"
            )
        return arr[:dim].astype(np.float64)

    out_dim_stats = 2 if show_cw_inset else 1
    _y_mean = _as_stat_vec(label_mean, out_dim_stats)
    _y_std = _as_stat_vec(label_std, out_dim_stats)
    if show_cw_inset and (_y_mean is None or _y_std is None):
        lt = label_tensor.detach().cpu().float()
        if lt.ndim == 1:
            lt = lt.unsqueeze(-1)
        d = min(out_dim_stats, int(lt.shape[-1]))
        _y_mean = lt[:, :d].mean(dim=0).numpy().astype(np.float64)
        _y_std = lt[:, :d].std(dim=0).numpy().astype(np.float64)
        _y_std = np.where(_y_std < 1e-8, 1.0, _y_std)
        print(
            "  (yield inset: label_mean/std not provided — "
            "using batch label stats for normalization)"
        )
    elif _y_std is not None:
        _y_std = np.where(_y_std < 1e-8, 1.0, _y_std)

    def _normalize_cw(comps: Tuple[float, ...]) -> Tuple[float, ...]:
        """Map denorm (chance, near) → z-score using train label stats."""
        return (
            (float(comps[0]) - _y_mean[0]) / (_y_std[0] + 1e-8),
            (float(comps[1]) - _y_mean[1]) / (_y_std[1] + 1e-8),
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
    _inset_axes: List[Any] = []  # chance×near insets; removed on each redraw
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
        if yt != YIELD_TARGET_BOTH or len(v) == 1:
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

            # Labels: from provided tensor (cache or model-sliced denorm)
            comps_lbl = _as_components(label_tensor[idx])
            comps_prd = (
                _as_components(pred_tensor[idx]) if pred_tensor is not None else None
            )
            if yt == YIELD_TARGET_CHANCE or (
                yt == YIELD_TARGET_BOTH and len(comps_lbl) == 1
            ):
                title_l1 = f'chance={comps_lbl[0]:.2f}'
            elif yt == YIELD_TARGET_NEAR:
                title_l1 = f'near={comps_lbl[0]:.2f}'
            else:
                title_l1 = f'chance={comps_lbl[0]:.2f}, near={comps_lbl[1]:.2f}'
            title_l2 = None
            if comps_prd is not None:
                if yt == YIELD_TARGET_CHANCE or (
                    yt == YIELD_TARGET_BOTH and len(comps_prd) == 1
                ):
                    title_l2 = f'pred chance={comps_prd[0]:.2f}'
                    pred_txt = f'pred c={comps_prd[0]:.1f}'
                elif yt == YIELD_TARGET_NEAR:
                    title_l2 = f'pred near={comps_prd[0]:.2f}'
                    pred_txt = f'pred n={comps_prd[0]:.1f}'
                else:
                    title_l2 = (
                        f'pred chance={comps_prd[0]:.2f}, near={comps_prd[1]:.2f}'
                    )
                    pred_txt = f'pred c={comps_prd[0]:.1f} n={comps_prd[1]:.1f}'
                ax.text(
                    0.02, 0.98,
                    pred_txt,
                    transform=ax.transAxes, va='top', ha='left', fontsize=8,
                    color='green',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                )

            # Small chance × near inset in *normalized* (z-score) space
            if show_cw_inset and len(comps_lbl) >= 2 and _y_mean is not None:
                ch_t_n, nr_t_n = _normalize_cw(comps_lbl)
                ax_cw = ax.inset_axes([0.55, 0.08, 0.43, 0.42])
                _inset_axes.append(ax_cw)
                ax_cw.scatter(
                    [ch_t_n], [nr_t_n],
                    c='#1f77b4', s=42, zorder=4, marker='o',
                    edgecolors='white', linewidths=0.6, label='truth',
                )
                xs_cw = [ch_t_n]
                ys_cw = [nr_t_n]
                if comps_prd is not None and len(comps_prd) >= 2:
                    ch_p_n, nr_p_n = _normalize_cw(comps_prd)
                    ax_cw.scatter(
                        [ch_p_n], [nr_p_n],
                        c='#2ca02c', s=48, zorder=5, marker='x',
                        linewidths=1.6, label='pred',
                    )
                    ax_cw.plot(
                        [ch_t_n, ch_p_n], [nr_t_n, nr_p_n],
                        color='gray', alpha=0.45, linewidth=0.9, zorder=3,
                    )
                    xs_cw.append(ch_p_n)
                    ys_cw.append(nr_p_n)

                    # Distances in normalized space (pred − truth)
                    d_chance = float(ch_p_n - ch_t_n)
                    d_near = float(nr_p_n - nr_t_n)
                    d_l2 = float(np.hypot(d_chance, d_near))
                    # Text in the middle of the coordinate plane
                    ax_cw.text(
                        0.5, 0.5,
                        (
                            f'‖Δ‖={d_l2:.3f}\n'
                            f'Δc={d_chance:+.3f}\n'
                            f'Δn={d_near:+.3f}'
                        ),
                        transform=ax_cw.transAxes,
                        ha='center', va='center', fontsize=6,
                        color='#222222', zorder=6,
                        bbox=dict(
                            boxstyle='round,pad=0.25',
                            facecolor='white',
                            edgecolor='0.75',
                            alpha=0.88,
                        ),
                    )

                # Axis limits with padding; keep origin in view
                x_lo, x_hi = min(xs_cw + [0.0]), max(xs_cw + [0.0])
                y_lo, y_hi = min(ys_cw + [0.0]), max(ys_cw + [0.0])
                x_pad = max(0.08 * (x_hi - x_lo), 0.35)
                y_pad = max(0.08 * (y_hi - y_lo), 0.35)
                ax_cw.set_xlim(x_lo - x_pad, x_hi + x_pad)
                ax_cw.set_ylim(y_lo - y_pad, y_hi + y_pad)
                ax_cw.axhline(0.0, color='0.7', linewidth=0.6, zorder=1)
                ax_cw.axvline(0.0, color='0.7', linewidth=0.6, zorder=1)
                ax_cw.set_xlabel('chance (norm)', fontsize=6, labelpad=1)
                ax_cw.set_ylabel('near (norm)', fontsize=6, labelpad=1)
                ax_cw.tick_params(axis='both', labelsize=5, length=2, pad=1)
                ax_cw.grid(True, alpha=0.35, linewidth=0.5)
                ax_cw.set_title('chance × near (z)', fontsize=6, pad=2)
                ax_cw.legend(fontsize=5, loc='upper right', framealpha=0.75, borderpad=0.2)
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
        mode_note = f'yield_target={yt}'
        line1 = (
            f'Yield ({mode_note}) — min_yield={min_yield_percent:g}%  |  {page_label}'
        )
        line2_parts = [
            f'horizon={max_h}',
            'orange=hits ≥ threshold',
            'chance from dataset cache',
        ]
        if show_cw_inset:
            line2_parts.append('inset: truth/pred in norm chance×near')
            if pred_tensor is not None:
                line2_parts.append('center text: ||d||, dc, dn (norm)')
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


def visualize_latent(
    series: List[dict],
    latent_distribution: Optional[str] = None,
    series_per_page: int = 1,
    figsize: Tuple[int, int] = (15, 9),
    interactive: bool = True,
    **_ignored,
):
    """
    Visualize the latent next-bar density as μ(t), σ(t) over an interval.

    Each series uses two stacked axes (same time):
      top    — raw price P(t), μ_raw(t), μ±σ band
      bottom — observation-space y(t), μ(t), σ(t)
               (this is the space the density is actually fitted in)

    t=0 is the seed sample's observation bar (``future[0]`` / next bar after
    the historical window). μ(t) is the density for P(t) = that next bar.
    """
    del _ignored
    dist = resolve_latent_distribution(
        {'latent_distribution': latent_distribution}
        if latent_distribution is not None
        else None,
        latent_distribution=latent_distribution,
    )

    if not series:
        print("⚠ No latent series available for visualization.")
        return

    n_series = len(series)
    series_per_page = max(1, int(series_per_page))
    n_pages = max(1, (n_series + series_per_page - 1) // series_per_page)
    n_rows = series_per_page * 2

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    if interactive:
        fig.subplots_adjust(
            left=0.07, right=0.98, top=0.88, bottom=0.14, hspace=0.45
        )
    else:
        fig.subplots_adjust(
            left=0.07, right=0.98, top=0.88, bottom=0.08, hspace=0.45
        )

    page_num = [0]
    _widgets: List[Any] = []

    def _draw_page():
        page_num[0] += 1
        if page_num[0] > n_pages:
            page_num[0] = 1
        page = page_num[0]
        start_i = (page - 1) * series_per_page
        page_items = list(range(start_i, min(start_i + series_per_page, n_series)))

        for ax in axes:
            ax.clear()
            ax.set_visible(False)

        for slot, si in enumerate(page_items):
            rec = series[si]
            t_price = np.asarray(rec['t_price'], dtype=np.float64)
            price = np.asarray(rec['price'], dtype=np.float64)
            t_pred = np.asarray(rec['t_pred'], dtype=np.float64)
            sample_id = rec.get('sample_id', si)
            recon_mae = float(rec.get('recon_mae', float('nan')))
            mu_mae = float(rec.get('mu_mae_raw', float('nan')))

            ax_raw = axes[2 * slot]
            ax_obs = axes[2 * slot + 1]
            ax_raw.set_visible(True)
            ax_obs.set_visible(True)

            ax_raw.plot(
                t_price, price, color='#1565C0', linewidth=1.6,
                label='P(t) realized', zorder=3,
            )
            if t_pred.size > 0:
                mu_raw = np.asarray(rec['mu_raw'], dtype=np.float64)
                sig_raw = np.asarray(rec['sigma_raw'], dtype=np.float64)
                p_at = np.asarray(rec.get('price_at_pred', []), dtype=np.float64)
                lo_raw = mu_raw - sig_raw
                hi_raw = mu_raw + sig_raw
                ax_raw.plot(
                    t_pred, mu_raw, color='#2E7D32', linewidth=2.4,
                    label='μ(t)', zorder=5,
                )
                # Band is μ±σ (not P±σ). Edges make the center obvious
                # when μ sits close to the realized price.
                ax_raw.plot(
                    t_pred, hi_raw, color='#2E7D32', linewidth=1.0,
                    linestyle='--', label='μ+σ', zorder=4,
                )
                ax_raw.plot(
                    t_pred, lo_raw, color='#2E7D32', linewidth=1.0,
                    linestyle='--', label='μ−σ', zorder=4,
                )
                ax_raw.fill_between(
                    t_pred, lo_raw, hi_raw,
                    color='#2E7D32', alpha=0.16, linewidth=0,
                    label='μ ± σ', zorder=2,
                )
                if p_at.size == t_pred.size:
                    ax_raw.scatter(
                        t_pred, p_at, c='#1565C0', s=18, zorder=6,
                        label='P at μ times',
                    )
                ax_raw.scatter(
                    t_pred, mu_raw, c='#2E7D32', s=18, marker='D', zorder=7,
                    label='μ samples',
                )
            ax_raw.axvline(x=0, color='gray', linestyle=':', linewidth=1.1, alpha=0.6)
            align_mae = float(rec.get('align_mae', float('nan')))
            ax_raw.set_title(
                f'Sample {sample_id}: raw price   '
                f'|μ−P|={mu_mae:.4g}   P-align={align_mae:.4g}   '
                f'label→P MAE={recon_mae:.4g}',
                fontsize=10,
            )
            ax_raw.set_ylabel('Price (raw)')
            ax_raw.grid(True, alpha=0.3)
            ax_raw.legend(fontsize=7, loc='best')

            if t_pred.size > 0:
                y_obs = np.asarray(rec['y_obs'], dtype=np.float64)
                mu_obs = np.asarray(rec['mu_obs'], dtype=np.float64)
                sig_obs = np.asarray(rec['sigma_obs'], dtype=np.float64)
                ax_obs.plot(
                    t_pred, y_obs, color='#1565C0', linewidth=1.6,
                    label='y(t) realized', zorder=3,
                )
                lo_obs = mu_obs - sig_obs
                hi_obs = mu_obs + sig_obs
                ax_obs.plot(
                    t_pred, mu_obs, color='#2E7D32', linewidth=2.4,
                    label='μ(t)', zorder=5,
                )
                ax_obs.plot(
                    t_pred, hi_obs, color='#2E7D32', linewidth=1.0,
                    linestyle='--', label='μ+σ', zorder=4,
                )
                ax_obs.plot(
                    t_pred, lo_obs, color='#2E7D32', linewidth=1.0,
                    linestyle='--', label='μ−σ', zorder=4,
                )
                ax_obs.fill_between(
                    t_pred, lo_obs, hi_obs,
                    color='#2E7D32', alpha=0.16, linewidth=0,
                    label='μ ± σ', zorder=2,
                )
                ax_obs.plot(
                    t_pred, sig_obs, color='#EF6C00', linewidth=1.4,
                    linestyle='--', label='σ(t)', zorder=4,
                )
            ax_obs.axhline(0.0, color='0.7', linewidth=0.7)
            ax_obs.axvline(x=0, color='gray', linestyle=':', linewidth=1.1, alpha=0.6)
            ax_obs.set_title(
                f'Sample {sample_id}: observation space (density units)',
                fontsize=10,
            )
            ax_obs.set_xlabel('Time (min; t=0 = seed observation bar, future[0])')
            ax_obs.set_ylabel('y, μ, σ')
            ax_obs.grid(True, alpha=0.3)
            ax_obs.legend(fontsize=7, loc='best')

        page_label = f'Page {page}/{n_pages}'
        fig.suptitle(
            f'Latent next-bar density ({dist})  |  {page_label}\n'
            f'top: raw P(t) and inverted μ,σ   ·   '
            f'bottom: real μ(t), σ(t) next to realized y(t)=future[0]'
            + ('   ·   [Next] / [Quit]' if interactive else ''),
            fontsize=11, fontweight='bold',
        )
        fig.canvas.draw_idle()
        shown = [series[i].get('sample_id', i) for i in page_items]
        print(f"  {page_label}: seed samples {shown}")

    if interactive:
        ax_next = fig.add_axes([0.55, 0.02, 0.18, 0.055])
        ax_quit = fig.add_axes([0.76, 0.02, 0.18, 0.055])
        btn_next = Button(ax_next, 'Next')
        btn_quit = Button(ax_quit, 'Quit')
        _widgets.extend([btn_next, btn_quit, ax_next, ax_quit])

        def _on_next(_event):
            _draw_page()

        def _on_quit(_event):
            print("  Quit — closing latent browser.")
            plt.close(fig)

        btn_next.on_clicked(_on_next)
        btn_quit.on_clicked(_on_quit)

        def _on_key(event):
            if event.key in ('n', 'right', ' '):
                _draw_page()
            elif event.key in ('q', 'escape'):
                print("  Quit — closing latent browser.")
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', _on_key)
        print(
            f"Latent browser ({n_series} interval(s)): "
            f"[Next] or N/Space = next page · [Quit] or Q = exit"
        )

    _draw_page()
    plt.show(block=True)


# ═══════════════════════════════════════════════════════════



# Legacy names
visualize_linear_trends = visualize_trend
visualize_scattered_hope = visualize_yield
