# Task_Trend.py — task implementation (canonical; not a re-export)

from __future__ import annotations

import random
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from task_base import Task


__all__ = ['Task_Trend']


# ═══════════════════════════════════════════════════════════
# LABEL FAMILY: TREND
# ═══════════════════════════════════════════════════════════

class Task_Trend(Task):
    """
    Trend labels: full cache (horizon, slope, intercept).

    Config id: 'trend'
    Price space: typically z-scored (caller responsibility).
    
    """

    name = 'trend'
    full_dim = 3

    @classmethod
    def resolve_future_window(cls, config: dict) -> int:
        fw = int(config.get('future_window', 50))
        min_h = int(config.get('min_trend_horizon', 3))
        return max(fw, min_h)

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
        mode=Task.HORIZON_SEARCH_LONGEST_LINEAR,
        max_past_horizon=None,
        use_past=False,
    ):
        """
        OLS linear-regime search on the *future* window only.

        ``use_past`` / past-regime search removed (always future-only).
        Returns: horizon (h_fut), slope, intercept
        """
        if mode not in Task.VALID_HORIZON_SEARCH_MODES:
            raise ValueError(
                f"Unknown horizon search mode {mode!r}; "
                f"expected one of {Task.VALID_HORIZON_SEARCH_MODES}"
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

        # Past-regime search disabled permanently.
        max_past_horizon = 0
        history_sigma = history.std().clamp(min=eps)

        def _eval(h_past: int, h_fut: int):
            y, t = cls.regime_segment(history, future, h_past, h_fut)
            slope, intercept, residual_l1 = cls.ols_fit_segment(y, t=t, eps=eps)
            fitness = residual_l1 / history_sigma
            return slope, intercept, fitness

        h_past, h_fut = 0, min_horizon
        slope, intercept, fitness = _eval(h_past, h_fut)
        best_h_fut = h_fut
        best_slope, best_intercept = slope, intercept
        best_total = h_fut

        if mode == Task.HORIZON_SEARCH_LONGEST_LINEAR:
            admissible = fitness <= fitness_threshold
            if admissible:
                best_h_fut = h_fut
                best_slope, best_intercept = slope, intercept
                best_total = h_fut

            while True:
                candidates = []
                if h_fut < max_horizon:
                    candidates.append((0, h_fut + 1))
                if not candidates:
                    break

                valid = []
                for hp, hf in candidates:
                    s, b, fit = _eval(hp, hf)
                    if fit <= fitness_threshold:
                        valid.append((fit, -hf, hp, hf, s, b))

                if not valid:
                    if not admissible:
                        best_h_fut = h_fut
                        best_slope, best_intercept = slope, intercept
                    break

                valid.sort()
                fit, _neg, hp, hf, s, b = valid[0]
                h_past, h_fut = hp, hf
                slope, intercept, fitness = s, b, fit
                admissible = True
                if h_fut >= best_total:
                    best_total = h_fut
                    best_h_fut = h_fut
                    best_slope, best_intercept = slope, intercept
        else:
            while fitness < fitness_threshold:
                candidates = []
                if h_fut < max_horizon:
                    candidates.append((0, h_fut + 1))
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
                best_h_fut = h_fut
                best_slope, best_intercept = slope, intercept
                if fitness >= fitness_threshold:
                    break

        return best_h_fut, best_slope, best_intercept

    @classmethod
    def make_auxiliary(cls, config: Optional[dict] = None, **overrides) -> dict:
        """
        Class-private loss knobs for trend.

        Keys (only used inside Task_Trend.loss):
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

    @classmethod
    def metric_slots(cls) -> Tuple[str, ...]:
        """Declared eval metric names for this label (LabelSpec slot)."""
        return (
            'horizon_mse', 'slope_mse', 'intercept_mse',
        )

    @classmethod
    def compute_metrics(
        cls,
        y_pred: torch.Tensor,
        y_truth: torch.Tensor,
        *,
        auxiliary: Optional[dict] = None,
        dataset=None,
        config: Optional[dict] = None,
        **_kwargs,
    ) -> dict:
        """Per-channel MSE metrics (slots present depend on output_dim)."""
        out_dim = int(y_pred.shape[-1])
        metrics = {}
        if out_dim == 3:
            metrics['horizon_mse'] = float(
                nn.functional.mse_loss(y_pred[:, 0], y_truth[:, 0]).item()
            )
            metrics['slope_mse'] = float(
                nn.functional.mse_loss(y_pred[:, 1], y_truth[:, 1]).item()
            )
            metrics['intercept_mse'] = float(
                nn.functional.mse_loss(y_pred[:, 2], y_truth[:, 2]).item()
            )
        elif out_dim == 2:
            metrics['slope_mse'] = float(
                nn.functional.mse_loss(y_pred[:, 0], y_truth[:, 0]).item()
            )
            metrics['intercept_mse'] = float(
                nn.functional.mse_loss(y_pred[:, 1], y_truth[:, 1]).item()
            )
        else:
            raise ValueError(
                f"{cls.name} metrics: unsupported pred dim {out_dim}"
            )
        return metrics

    @classmethod
    def pack_metrics(cls, y_pred, y_truth, **kwargs) -> dict:
        """Opaque metrics dict — models never see this."""
        return cls.compute_metrics(y_pred, y_truth, **kwargs)

    @classmethod
    def unpack_metrics(cls, metrics: dict) -> dict:
        slots = cls.metric_slots()
        return {k: metrics[k] for k in slots if k in metrics}

    @classmethod
    def normalize_policy(cls) -> str:
        return 'zscore'

    @classmethod
    def normalize(cls, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return cls.zscore_normalize(y, mean, std)

    @classmethod
    def denormalize(cls, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return cls.zscore_denormalize(y, mean, std)

    @classmethod
    def bind_dataset_outputs(cls, config: dict) -> dict:
        output_dim = int(config.get('output_dim', 3))
        if output_dim not in (2, 3):
            raise ValueError(f"output_dim must be 2 or 3 for trend, got {output_dim}")
        return {
            'yield_target': 'both',
            'output_dim': output_dim,
            'latent_dim': output_dim,
            'label_slice': cls.target_slice(output_dim),
            'normalize_policy': cls.normalize_policy(),
        }

    @classmethod
    def sync_output_dim(cls, config: dict) -> int:
        return int(config.get('output_dim', 3))

    @classmethod
    def build_cache_row(cls, ctx: 'WindowContext', config: dict) -> torch.Tensor:
        future_for_label, history_for_label = cls.scale_prices_by_window_moments(
            ctx.future_prices, ctx.history_prices, ctx.mean, ctx.std,
            ctx.price_feature_idx, ctx.normalize_window,
        )
        future_window = int(config.get('future_window', 50))
        best_h, best_slope, best_intercept = cls.find(
            future_for_label,
            history_for_label,
            min_horizon=int(config.get('min_trend_horizon', 3)),
            max_horizon=future_window,
            fitness_threshold=float(config.get('fitness_threshold', Task.FITNESS_THRESHOLD)),
            mode=config.get('horizon_search', Task.HORIZON_SEARCH_LONGEST_LINEAR),
            use_past=False,
        )
        return torch.tensor(
            [best_h, best_slope, best_intercept], dtype=torch.float32
        )

    @classmethod
    def signature_fields(cls, config: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            'min_trend_horizon': config.get('min_trend_horizon', 3),
            'fitness_threshold': config.get(
                'fitness_threshold', Task.FITNESS_THRESHOLD
            ),
            'horizon_search': config.get(
                'horizon_search', Task.HORIZON_SEARCH_LONGEST_LINEAR
            ),
        }

    @classmethod
    def dataset_banner_lines(cls, *, bound, config, dataset) -> List[str]:
        return [
            f"  Horizon search: {config.get('horizon_search')}",
            f"  Fitness threshold: {config.get('fitness_threshold')}",
            f"  Min trend horizon: {config.get('min_trend_horizon')}, "
            f"Max trend future: {getattr(dataset, 'future_window', None)}",
        ]

    @classmethod
    def format_metric_lines(cls, metrics: Mapping[str, Any]) -> List[str]:
        lines = []
        if 'horizon_mse' in metrics:
            lines.append(f"  Horizon MSE:     {metrics['horizon_mse']:.4f}")
        if 'slope_mse' in metrics:
            lines.append(f"  Slope MSE:       {metrics['slope_mse']:.4f}")
        if 'intercept_mse' in metrics:
            lines.append(f"  Intercept MSE:   {metrics['intercept_mse']:.4f}")
        return lines or [f"  {k}: {v}" for k, v in metrics.items()]

    @classmethod
    def metrics_report_note(cls, config: Optional[dict] = None) -> str:
        return "Metrics from Task_Trend metric_slots."

    @classmethod
    def decision_threshold_from_config(cls, config: Optional[dict] = None) -> Optional[float]:
        return None

    @classmethod
    def summary_metric_headers(cls) -> Tuple[str, ...]:
        return ('horizon_mse', 'slope_mse', 'intercept_mse')

    @classmethod
    def format_summary_metric_cells(cls, metrics: Optional[Mapping[str, Any]]) -> Tuple[str, ...]:
        m = metrics or {}
        def _fmt(key):
            if key not in m:
                return '—'
            try:
                return f"{float(m[key]):.4f}"
            except Exception:
                return str(m[key])
        return tuple(_fmt(k) for k in cls.summary_metric_headers())

    @classmethod
    def ensemble_compare_ylabel(cls) -> str:
        return "Label loss(en_k, y)  [trend MSE space]"

    @classmethod
    def ensemble_loss_increase_warning(
        cls, *, k: int, loss: float, prev_k: int, prev_loss: float
    ) -> str:
        return (
            f"loss(en_{k}, y)={loss:.6f} > loss(en_{prev_k}, y)={prev_loss:.6f}. "
            f"Residual boosting expects non-increasing ensemble loss when stages "
            f"fit y-space residuals. Usually means old norms/weights — retrain "
            f"with --error-ensemble --clean."
        )


    @classmethod
    def print_prediction_samples(
        cls, preds, test_labels, *, dataset=None, yield_target=None, latent_dist=None, n: int = 5,
    ) -> None:
        n = min(int(n), len(preds))
        out_dim = preds.shape[-1]
        if out_dim == 3:
            print(
                f"  {'Index':<6} {'True H':<8} {'Pred H':<8} "
                f"{'True Slope':<12} {'Pred Slope':<12} "
                f"{'True Intercept':<14} {'Pred Intercept':<14}"
            )
            print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*14} {'-'*14}")
            for i in range(n):
                print(
                    f"  {i:<6} {test_labels[i, 0].item():<8.0f} {preds[i, 0].item():<8.0f} "
                    f"{test_labels[i, 1].item():<12.4f} {preds[i, 1].item():<12.4f} "
                    f"{test_labels[i, 2].item():<14.2f} {preds[i, 2].item():<14.2f}"
                )
        else:
            print(
                f"  {'Index':<6} {'True Slope':<12} {'Pred Slope':<12} "
                f"{'True Intercept':<14} {'Pred Intercept':<14}"
            )
            print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*14} {'-'*14}")
            for i in range(n):
                print(
                    f"  {i:<6} {test_labels[i, 0].item():<12.4f} {preds[i, 0].item():<12.4f} "
                    f"{test_labels[i, 1].item():<14.2f} {preds[i, 1].item():<14.2f}"
                )

    @staticmethod
    def _as_float(v) -> float:
        return float(v.item() if hasattr(v, 'item') else v)

    @staticmethod
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
        as_float = Task_Trend._as_float
        if n == 3:
            horizon = max(0, int(round(as_float(vec[0]))))
            slope_norm = as_float(vec[1])
            intercept_norm = as_float(vec[2])
        elif n == 2:
            horizon = max(0, int(default_horizon))
            slope_norm = as_float(vec[0])
            intercept_norm = as_float(vec[1])
        else:
            raise ValueError(f"Trend vector must have length 2 or 3, got {n}")
        if price_std == 0:
            price_std = 1.0
        slope = slope_norm * price_std
        intercept = intercept_norm * price_std + price_mean
        return horizon, slope, intercept

    @classmethod
    def visualize(cls, 
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
        normalize_prices: bool = True,
    ):
        """
        Visualize future-only linear regimes from the dataset cache.

        Plots for each panel:
          - actual prices (recent past context + full future)
          - label regime line on t ∈ [0, h_fut)  (from cache; t=0 = first future)
          - inferred trend (from pred_tensor), if provided — future span only

        label_tensor is the cached natural-unit label (not recomputed):
          len 3 → (horizon, slope, intercept) in window price-z space for m,b
          len 2 → (slope, intercept); horizon from default_pred_horizon

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

            max_h = int(future_window)

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

                # Labels from cache only (never re-run find here).
                h_lbl, m_lbl, b_lbl = cls._trend_to_price_space(
                    label_tensor[idx], price_mean, price_std,
                    default_horizon=int(max_h),
                )

                y_start = start + past_window
                y_end = y_start + future_window

                # Actual prices: fixed past context for display (regime is future-only).
                past_context = 15
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

                # Label regime: future-only t = 0 .. h_fut-1 (intercept at t=0)
                if h_lbl > 0:
                    t_lbl = np.arange(0, max(h_lbl, 1))
                    if len(t_lbl) > 0:
                        ax.plot(
                            t_lbl,
                            m_lbl * t_lbl + b_lbl,
                            'r-',
                            linewidth=2.5,
                            label=(
                                f'Label regime: H={h_lbl}, '
                                f'm={m_lbl:.4f}, b={b_lbl:.2f}'
                            ),
                        )
                        ax.axvline(
                            x=h_lbl - 1, color='red', linestyle='--',
                            linewidth=1.3, alpha=0.65,
                        )
                        ax.axvspan(
                            0, max(h_lbl - 1, 0),
                            alpha=0.12, color='#FFCC80',
                        )

                # Inferred trend (model): future side only; t=0..H-1
                if pred_tensor is not None:
                    pred_h_default = (
                        default_pred_horizon
                        if default_pred_horizon is not None
                        else max_h
                    )
                    h_prd, m_prd, b_prd = cls._trend_to_price_space(
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
                    f'Sample {display_id}: H={h_lbl}, '
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
            line2_parts = ['labels from dataset cache', 'future-only regime']
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


# --- visualization helpers ---
