# label_viz.py
# Visualization for label modes (linear regime, scattered hope).
#
# Depends on labels.py for finders / constants; not imported by the pure loss path.

import random
import time
from typing import List, Optional, Tuple, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from labels import (
    LABEL_FITNESS_THRESHOLD,
    HORIZON_SEARCH_LONGEST_LINEAR,
    find_linear_regime,
)

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
    Visualize scattered-hope labels in raw price space.

    Each panel:
      - recent past + future actual prices
      - horizontal line at reference price history[-1]
      - markers on future bars where yield % >= min_yield_percent
      - label chance[, wait]; optional pred
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

    def _as_components(vec) -> Tuple[float, ...]:
        v = vec.detach().cpu().flatten() if hasattr(vec, 'detach') else np.asarray(vec).ravel()
        if chance_only or len(v) == 1:
            return (float(v[0]),)
        return (float(v[0]), float(v[1]))

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
                for tt, pp in zip(event_t, event_p):
                    ax.plot([0, tt], [ref, pp], color='orange', alpha=0.25, linewidth=1.0)

            if n_hope > 0:
                ax.axvspan(0, max(n_hope - 1, 0), alpha=0.08, color='#FFE0B2')

            comps_lbl = _as_components(label_tensor[idx])
            if chance_only or len(comps_lbl) == 1:
                title_l1 = f'chance={comps_lbl[0]:.2f}'
            else:
                title_l1 = f'chance={comps_lbl[0]:.2f}, wait={comps_lbl[1]:.2f}'
            title_l2 = None
            if pred_tensor is not None:
                comps_prd = _as_components(pred_tensor[idx])
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
            f'Scattered Hope ({mode_note}) — min_yield={min_yield_percent:g}%  |  {page_label}'
        )
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

