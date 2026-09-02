# Task_Yield.py — yield task (config id: yield)

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import special
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from task_base import Task


class Task_Yield(Task):
    """
    Yield labels on raw prices.

    Config id: 'yield'
    Price space: raw (true % yield); caller must not z-score inputs.

    Full cache vector is always (chance, near). Model target is selected by
    ``yield_target``: 'both' | 'chance' | 'near'.
    
    """

    name = 'yield'


    @classmethod
    def report_header_fields(cls, dataset, config=None):
        cfg = dict(config or {})
        yt = getattr(dataset, 'yield_target', None)
        if yt is None:
            yt = cls.resolve_yield_target(cfg)
        return {'Yield target': yt}

    @classmethod
    def visualize_inference(cls, *, dataset, preds, test_labels, used_indices, config, **kwargs):
        import torch
        yt = cls.resolve_yield_target(
            config, yield_target=getattr(dataset, 'yield_target', None),
        )
        n_viz = len(used_indices)
        valid_starts = [dataset.valid_starts[i] for i in used_indices]
        full_labels = torch.stack([dataset.label_tensor[i] for i in used_indices])
        pred_cpu = preds.detach().cpu()
        print(
            f"\nVisualizing yield labels (yield_target={yt}) on test samples...",
            flush=True,
        )
        viz_labels = test_labels.detach().cpu()
        return cls.visualize(
            total_tensor=dataset.Total,
            valid_starts=valid_starts,
            label_tensor=viz_labels if yt != 'both' else full_labels,
            price_feature_idx=dataset.price_feature_idx,
            past_window=dataset.past_window,
            future_window=dataset.future_window,
            min_yield_percent=float(config.get('min_yield_percent', 0.0)),
            max_hope_future_horizon=config.get(
                'future_window', getattr(dataset, 'future_window', None),
            ),
            yield_target=yt,
            num_samples=min(6, n_viz),
            pred_tensor=pred_cpu,
            fixed_order=True,
            sample_ids=list(used_indices),
            label_mean=getattr(dataset, '_label_stat_mean', None),
            label_std=getattr(dataset, '_label_stat_std', None),
        )


    TARGET_BOTH = 'both'
    TARGET_CHANCE = 'chance'
    TARGET_NEAR = 'near'
    VALID_TARGETS = (TARGET_BOTH, TARGET_CHANCE, TARGET_NEAR)

    @classmethod
    def resolve_yield_target(cls, config: Optional[dict] = None, **overrides) -> str:
        """Resolve yield model target: 'both' | 'chance' | 'near'."""
        cfg = dict(config or {})
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        raw = cfg.get('yield_target', cls.TARGET_BOTH)
        if raw is None or str(raw).strip() == '':
            return cls.TARGET_BOTH
        yt = str(raw).strip().lower()
        if yt in ('both', 'chance_near', 'chance_wait', 'all'):
            yt = 'both'
        elif yt in ('c',):
            yt = 'chance'
        elif yt in ('n', 'w', 'wait'):
            yt = 'near'
        if yt not in cls.VALID_TARGETS:
            raise ValueError(
                f"yield_target must be one of {cls.VALID_TARGETS}, got {raw!r}"
            )
        return yt


    @classmethod
    def full_cache_dim(cls) -> int:
        """Stored label vector width (always chance + near)."""
        return 2

    @classmethod
    def full_dim(cls, yield_target: str = 'both', **_kwargs) -> int:
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
        yield_target: str = 'both',
        **_kwargs,
    ) -> int:
        yt = cls.resolve_yield_target({'yield_target': yield_target})
        if yt == 'both':
            return 2
        return 1

    @classmethod
    def target_slice(
        cls,
        output_dim: int,
        yield_target: str = 'both',
        **_kwargs,
    ) -> slice:
        """
        Slice into full cache vector [chance, near].

          both   → [0:2]
          chance → [0:1]
          near   → [1:2]
        """
        yt = cls.resolve_yield_target({'yield_target': yield_target})
        expected = cls.resolve_output_dim(yield_target=yt)
        if output_dim != expected:
            raise ValueError(
                f"{cls.name} with yield_target={yt!r} requires "
                f"output_dim={expected}, got {output_dim}"
            )
        if yt == 'both':
            return slice(0, 2)
        if yt == 'chance':
            return slice(0, 1)
        if yt == 'near':
            return slice(1, 2)
        raise ValueError(f"Unknown yield_target {yt!r}")

    @classmethod
    def target_names(
        cls,
        output_dim: int,
        yield_target: str = 'both',
        **_kwargs,
    ) -> Tuple[str, ...]:
        yt = cls.resolve_yield_target({'yield_target': yield_target})
        if yt == 'both':
            return ('chance', 'near')
        if yt == 'chance':
            return ('chance',)
        if yt == 'near':
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

        yt = cls.resolve_yield_target({'yield_target': yield_target})
        if yt == 'both':
            return full
        if yt == 'chance':
            return (full[0],)
        if yt == 'near':
            return (full[1],)
        return full

    @classmethod
    def make_auxiliary(cls, config: Optional[dict] = None, **overrides) -> dict:
        """
        Class-private loss knobs for yield.

        Keys: chance_weight, near_weight, yield_target
        """
        cfg = config or {}
        yt = cls.resolve_yield_target(cfg, **overrides)
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
        yt = cls.resolve_yield_target(aux)

        out_dim = y_pred.shape[-1]
        if y_truth.shape[-1] != out_dim:
            raise ValueError(
                f"{cls.name} loss: target dim {y_truth.shape[-1]} != pred dim {out_dim}"
            )

        if yt == 'both':
            if out_dim != 2:
                raise ValueError(
                    f"yield_target='both' requires pred dim 2, got {out_dim}"
                )
            chance_loss = nn.functional.mse_loss(y_pred[:, 0], y_truth[:, 0])
            near_loss = nn.functional.mse_loss(y_pred[:, 1], y_truth[:, 1])
            return chance_weight * chance_loss + near_weight * near_loss

        if yt in (cls.TARGET_CHANCE, cls.TARGET_NEAR):
            if out_dim != 1:
                raise ValueError(
                    f"yield_target={yt!r} requires pred dim 1, got {out_dim}"
                )
            w = chance_weight if yt == 'chance' else near_weight
            return w * nn.functional.mse_loss(y_pred[:, 0], y_truth[:, 0])

        raise ValueError(
            f"{cls.name} loss: unknown yield_target {yt!r}"
        )

    @classmethod
    def metric_slots(cls) -> Tuple[str, ...]:
        return ('chance_mse', 'near_mse')

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
        cfg = dict(config or {})
        if auxiliary:
            cfg.update({k: v for k, v in auxiliary.items() if v is not None})
        yt = cls.resolve_yield_target(cfg)
        out_dim = int(y_pred.shape[-1])
        metrics = {}
        if yt == 'chance':
            metrics['chance_mse'] = float(
                nn.functional.mse_loss(y_pred[:, 0], y_truth[:, 0]).item()
            )
        elif yt == 'near':
            metrics['near_mse'] = float(
                nn.functional.mse_loss(y_pred[:, 0], y_truth[:, 0]).item()
            )
        elif yt == 'both' and out_dim == 2:
            metrics['chance_mse'] = float(
                nn.functional.mse_loss(y_pred[:, 0], y_truth[:, 0]).item()
            )
            metrics['near_mse'] = float(
                nn.functional.mse_loss(y_pred[:, 1], y_truth[:, 1]).item()
            )
        else:
            raise ValueError(
                f"{cls.name} metrics: yield_target={yt!r} dim={out_dim}"
            )
        return metrics

    @classmethod
    def pack_metrics(cls, y_pred, y_truth, **kwargs) -> dict:
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
        yt = cls.resolve_yield_target(config)
        forced_dim = cls.resolve_output_dim(yield_target=yt)
        cfg_dim = int(config.get('output_dim', forced_dim))
        if cfg_dim != forced_dim:
            print(
                f"⚠ label={'yield'} yield_target={yt!r}: "
                f"forcing output_dim {cfg_dim} → {forced_dim}"
            )
        return {
            'yield_target': yt,
            'output_dim': forced_dim,
            'latent_dim': forced_dim,
            'label_slice': cls.target_slice(forced_dim, yield_target=yt),
            'normalize_policy': cls.normalize_policy(),
        }

    @classmethod
    def sync_output_dim(cls, config: dict) -> int:
        config['output_dim'] = cls.resolve_output_dim(
            yield_target=cls.resolve_yield_target(config),
        )
        return int(config['output_dim'])

    @classmethod
    def build_cache_row(cls, ctx: 'WindowContext', config: dict) -> torch.Tensor:
        future_window = int(config.get('future_window', 50))
        result = cls.find(
            ctx.future_prices,
            ctx.current_price,
            min_yield_percent=float(config.get('min_yield_percent', 0.0)),
            max_hope_future_horizon=future_window,
        )
        if len(result) == 1:
            chance, near = float(result[0]), 0.0
        else:
            chance, near = float(result[0]), float(result[1])
        return torch.tensor([chance, near], dtype=torch.float32)

    @classmethod
    def signature_fields(cls, config: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            'min_yield_percent': config.get('min_yield_percent', 0.0),
            'yield_target': config.get('yield_target', 'both'),
        }

    @classmethod
    def dataset_banner_lines(cls, *, bound, config, dataset) -> List[str]:
        return [
            f"  min_yield_percent: {config.get('min_yield_percent')}",
            f"  yield_target: {bound.get('yield_target', config.get('yield_target'))}",
        ]

    @classmethod
    def format_metric_lines(cls, metrics: Mapping[str, Any]) -> List[str]:
        lines = []
        if 'chance_mse' in metrics:
            lines.append(f"  Chance MSE:      {metrics['chance_mse']:.4f}")
        if 'near_mse' in metrics:
            lines.append(f"  Near MSE:        {metrics['near_mse']:.4f}")
        return lines or [f"  {k}: {v}" for k, v in metrics.items()]

    @classmethod
    def metrics_report_note(cls, config: Optional[dict] = None) -> str:
        return "Metrics from Task_Yield metric_slots (y-normalized MSE)."

    @classmethod
    def decision_threshold_from_config(cls, config: Optional[dict] = None) -> Optional[float]:
        return None

    @classmethod
    def summary_metric_headers(cls) -> Tuple[str, ...]:
        return ('chance_mse', 'near_mse')

    @classmethod
    def format_summary_metric_cells(cls, metrics: Optional[Mapping[str, Any]]) -> Tuple[str, ...]:
        m = metrics or {}
        out = []
        for key in cls.summary_metric_headers():
            if key not in m:
                out.append('—')
            else:
                try:
                    out.append(f"{float(m[key]):.4f}")
                except Exception:
                    out.append(str(m[key]))
        return tuple(out)

    @classmethod
    def ensemble_compare_ylabel(cls) -> str:
        return "Label loss(en_k, y)  [yield MSE, y-normalized]"

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
        if yield_target == 'chance':
            print(f"  {'Index':<6} {'True Chance':<12} {'Pred Chance':<12}")
            print(f"  {'-'*6} {'-'*12} {'-'*12}")
            for i in range(n):
                print(
                    f"  {i:<6} {test_labels[i, 0].item():<12.4f} {preds[i, 0].item():<12.4f}"
                )
            return
        if yield_target == 'near':
            print(f"  {'Index':<6} {'True Near':<12} {'Pred Near':<12}")
            print(f"  {'-'*6} {'-'*12} {'-'*12}")
            for i in range(n):
                print(
                    f"  {i:<6} {test_labels[i, 0].item():<12.4f} {preds[i, 0].item():<12.4f}"
                )
            return
        print(
            f"  {'Index':<6} {'True Chance':<12} {'Pred Chance':<12} "
            f"{'True Near':<12} {'Pred Near':<12}"
        )
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for i in range(n):
            print(
                f"  {i:<6} {test_labels[i, 0].item():<12.4f} {preds[i, 0].item():<12.4f} "
                f"{test_labels[i, 1].item():<12.4f} {preds[i, 1].item():<12.4f}"
            )

    @classmethod
    def visualize(cls, 
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
        yt = cls.resolve_yield_target({'yield_target': yield_target})
        show_cw_inset = yt == 'both'

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
            if yt != 'both' or len(v) == 1:
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
                if yt == 'chance' or (
                    yt == 'both' and len(comps_lbl) == 1
                ):
                    title_l1 = f'chance={comps_lbl[0]:.2f}'
                elif yt == 'near':
                    title_l1 = f'near={comps_lbl[0]:.2f}'
                else:
                    title_l1 = f'chance={comps_lbl[0]:.2f}, near={comps_lbl[1]:.2f}'
                title_l2 = None
                if comps_prd is not None:
                    if yt == 'chance' or (
                        yt == 'both' and len(comps_prd) == 1
                    ):
                        title_l2 = f'pred chance={comps_prd[0]:.2f}'
                        pred_txt = f'pred c={comps_prd[0]:.1f}'
                    elif yt == 'near':
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
