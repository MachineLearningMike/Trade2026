# Task_Latent.py — latent density task (config id: latent)

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import StudentT
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from task_base import Task


class Task_Latent(Task):
    """
    Latent labels: infer a density for the *next* bar (first future price).
    

    The realized observation is ``future[0]`` — one sample from a latent
    price law conditioned on the historical window. The model outputs the
    parameters of that law from market history (it does not emit a point
    forecast; the NLL trains the density against the realized next bar).

    Config id: 'latent'
    Shares ``model_regression.MarketRegressionWithHeads`` with trend/yield.

    Mode (config ``latent_distribution``):
      'normal'  → y_pred (B, 2) = (mu, var_raw);   var = softplus(var_raw)+eps
      'student' → y_pred (B, 3) = (loc, scale_raw, df_raw)
                  scale = softplus(scale_raw)+eps; df = 2+softplus(df_raw)

    Label cache stores the realized next-bar price (dim 1). Model ``output_dim``
    is forced from the density mode (2 or 3). ``y_pred`` and ``y_truth``
    therefore have different last dims by design — loss is NLL, not MSE.
    """

    name = 'latent'

    DIST_NORMAL = 'normal'
    DIST_STUDENT = 'student'
    VALID_DISTRIBUTIONS = (DIST_NORMAL, DIST_STUDENT)
    OUTPUT_DIM_NORMAL = 2
    OUTPUT_DIM_STUDENT = 3
    _VAR_MIN = 1e-3
    _SCALE_MIN = 1e-2
    _DF_SOFTPLUS_MAX = 20.0

    @classmethod
    def resolve_latent_distribution(cls, config: Optional[dict] = None, **overrides) -> str:
        """Resolve latent density mode: 'normal' | 'student'."""
        cfg = dict(config or {})
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        raw = cfg.get('latent_distribution', cls.DIST_NORMAL)
        if raw is None or str(raw).strip() == '':
            return cls.DIST_NORMAL
        s = str(raw).strip().lower().replace('-', '_').replace(' ', '_')
        if s in (
            cls.DIST_NORMAL, 'gaussian', 'normal_distribution',
            'normal_distirbution', 'n',
        ):
            return cls.DIST_NORMAL
        if s in (
            cls.DIST_STUDENT, 'student_t', 'student_t_distribution',
            'student_distribution', 'student_distirbution', 't',
        ):
            return cls.DIST_STUDENT
        raise ValueError(
            f"latent_distribution must be one of {cls.VALID_DISTRIBUTIONS} "
            f"(aliases: normal_distribution, student_distribution), got {raw!r}"
        )

    @staticmethod
    def _gaussian_variance(var_raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(var_raw) + Task_Latent._VAR_MIN

    @staticmethod
    def _student_scale(scale_raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(scale_raw) + Task_Latent._SCALE_MIN

    @classmethod
    def _gaussian_nll_loss(cls, mu, var_raw, target) -> torch.Tensor:
        finite = torch.isfinite(target) & torch.isfinite(mu) & torch.isfinite(var_raw)
        if not finite.any():
            return mu.sum() * 0.0
        var = cls._gaussian_variance(var_raw)
        log_var = torch.log(var)
        residual2 = (target - mu).pow(2)
        loss = 0.5 * (log_var + residual2 / var)
        loss = loss.masked_fill(~finite, 0.0)
        return loss.sum() / finite.sum().clamp(min=1).to(loss.dtype)

    @classmethod
    def _student_t_nll_loss(cls, location, scale_raw, df_raw, target) -> torch.Tensor:
        finite = (
            torch.isfinite(target)
            & torch.isfinite(location)
            & torch.isfinite(scale_raw)
            & torch.isfinite(df_raw)
        )
        if not finite.any():
            return location.sum() * 0.0
        scale = cls._student_scale(scale_raw)
        df = 2.0 + F.softplus(df_raw).clamp(max=cls._DF_SOFTPLUS_MAX)
        dist = StudentT(df=df, loc=location, scale=scale)
        nll = -dist.log_prob(target)
        nll = torch.nan_to_num(nll, nan=0.0, posinf=1e4, neginf=1e4)
        nll = nll.masked_fill(~finite, 0.0)
        return nll.sum() / finite.sum().clamp(min=1).to(nll.dtype)

    @classmethod
    def decode_latent_params(
        cls,
        y_pred: torch.Tensor,
        latent_distribution: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        dist = cls.resolve_latent_distribution(
            {'latent_distribution': latent_distribution}
            if latent_distribution is not None
            else None,
            latent_distribution=latent_distribution,
        )
        if y_pred.dim() == 0:
            raise ValueError("y_pred must have a last dimension of density params")
        if dist == cls.DIST_NORMAL:
            if y_pred.shape[-1] != cls.OUTPUT_DIM_NORMAL:
                raise ValueError(
                    f"normal head expects last dim {cls.OUTPUT_DIM_NORMAL}, "
                    f"got {y_pred.shape[-1]}"
                )
            location = y_pred[..., 0]
            scale = cls._gaussian_variance(y_pred[..., 1]).sqrt()
            return location, scale, None
        if y_pred.shape[-1] != cls.OUTPUT_DIM_STUDENT:
            raise ValueError(
                f"student head expects last dim {cls.OUTPUT_DIM_STUDENT}, "
                f"got {y_pred.shape[-1]}"
            )
        location = y_pred[..., 0]
        scale = cls._student_scale(y_pred[..., 1])
        df = 2.0 + F.softplus(y_pred[..., 2]).clamp(max=cls._DF_SOFTPLUS_MAX)
        return location, scale, df

    @classmethod
    def latent_params_to_obs_space(
        cls,
        location: torch.Tensor,
        scale: torch.Tensor,
        label_mean: torch.Tensor,
        label_std: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = torch.as_tensor(label_mean, device=location.device, dtype=location.dtype)
        std = torch.as_tensor(label_std, device=location.device, dtype=location.dtype)
        mean = mean.reshape(-1)[:1]
        std = std.reshape(-1)[:1]
        std = torch.where(std.abs() < 1e-8, torch.ones_like(std), std)
        while mean.dim() < location.dim():
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return location * std + mean, scale * std.abs()


    @classmethod
    def resolve_distribution(cls, config: Optional[dict] = None, **overrides) -> str:
        return cls.resolve_latent_distribution(config, **overrides)

    @classmethod
    def resolve_output_dim(
        cls,
        config: Optional[dict] = None,
        latent_distribution: Optional[str] = None,
        **overrides,
    ) -> int:
        """Model head width from density mode (not the label-cache width)."""
        dist = cls.resolve_latent_distribution(
            config,
            latent_distribution=latent_distribution,
            **overrides,
        )
        if dist == cls.DIST_NORMAL:
            return cls.OUTPUT_DIM_NORMAL
        if dist == cls.DIST_STUDENT:
            return cls.OUTPUT_DIM_STUDENT
        raise ValueError(f"Unknown latent_distribution {dist!r}")

    @classmethod
    def resolve_latent_dim(cls, config: Optional[dict] = None, **overrides) -> int:
        """Alias of resolve_output_dim (model head width)."""
        return cls.resolve_output_dim(config, **overrides)

    @classmethod
    def full_cache_dim(cls) -> int:
        """Stored label vector width (scalar observation)."""
        return 1

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
        return slice(0, 1)

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
        if output_dim == 1:
            return ('y',)
        dist = cls.resolve_latent_distribution(
            {'latent_distribution': latent_distribution}
            if latent_distribution is not None
            else None,
            latent_distribution=latent_distribution,
        )
        if dist == cls.DIST_NORMAL and output_dim == cls.OUTPUT_DIM_NORMAL:
            return ('mu', 'var_raw')
        if dist == cls.DIST_STUDENT and output_dim == cls.OUTPUT_DIM_STUDENT:
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
        dist = cls.resolve_latent_distribution(cfg, **overrides)
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
        dist = cls.resolve_latent_distribution(aux)

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
            cls.OUTPUT_DIM_NORMAL
            if dist == cls.DIST_NORMAL
            else cls.OUTPUT_DIM_STUDENT
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

        if dist == cls.DIST_NORMAL:
            mu = y_pred[:, 0:1]
            var_raw = y_pred[:, 1:2]
            nll = cls._gaussian_nll_loss(mu, var_raw, y_truth)
        else:
            location = y_pred[:, 0:1]
            scale_raw = y_pred[:, 1:2]
            df_raw = y_pred[:, 2:3]
            nll = cls._student_t_nll_loss(location, scale_raw, df_raw, y_truth)
        if not torch.isfinite(nll):
            # Last-resort: keep the graph alive with a zero that still depends on y_pred.
            nll = y_pred.sum() * 0.0
        return weight * nll

    @classmethod
    def metric_slots(cls) -> Tuple[str, ...]:
        return (
            'nll', 'location_mae', 'location_rmse',
            'mean_pred_std', 'mean_pred_df',
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
        cfg = dict(config or {})
        aux = cls.make_auxiliary(cfg, **(auxiliary or {}))
        dist = cls.resolve_latent_distribution(aux)
        metrics = {
            'nll': float(cls.loss(y_pred, y_pred, y_truth, auxiliary=aux).item()),
        }
        loc_n, scale_n, df_t = cls.decode_latent_params(
            y_pred, latent_distribution=dist
        )
        if dataset is None:
            metrics['mean_pred_std'] = float(
                scale_n.reshape(-1).float().mean().item()
            )
            return metrics
        loc_obs, scale_obs = cls.latent_params_to_obs_space(
            loc_n, scale_n,
            dataset._label_stat_mean, dataset._label_stat_std,
        )
        y_obs = dataset.denormalize_y(y_truth.detach().cpu()).reshape(-1)
        loc_obs = loc_obs.reshape(-1).to(y_obs.device)
        scale_obs = scale_obs.reshape(-1).to(y_obs.device)
        metrics['location_mae'] = float((loc_obs - y_obs).abs().mean().item())
        metrics['location_rmse'] = float(
            (loc_obs - y_obs).pow(2).mean().sqrt().item()
        )
        metrics['mean_pred_std'] = float(scale_obs.mean().item())
        if df_t is not None:
            metrics['mean_pred_df'] = float(
                df_t.reshape(-1).float().mean().item()
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
        dist = cls.resolve_latent_distribution(config)
        forced_dim = cls.resolve_output_dim(config, latent_distribution=dist)
        cfg_dim = int(config.get('output_dim', forced_dim))
        if cfg_dim != forced_dim:
            print(
                f"⚠ label={'latent'} latent_distribution={dist!r}: "
                f"forcing output_dim {cfg_dim} → {forced_dim}"
            )
        return {
            'yield_target': 'both',
            'latent_distribution': dist,
            'output_dim': forced_dim,
            'latent_dim': forced_dim,
            'label_slice': cls.target_slice(forced_dim),
            'normalize_policy': cls.normalize_policy(),
        }

    @classmethod
    def sync_output_dim(cls, config: dict) -> int:
        config['output_dim'] = cls.resolve_output_dim(config)
        return int(config['output_dim'])

    @classmethod
    def build_cache_row(cls, ctx: 'WindowContext', config: dict) -> torch.Tensor:
        future_for_label, history_for_label = cls.scale_prices_by_window_moments(
            ctx.future_prices, ctx.history_prices, ctx.mean, ctx.std,
            ctx.price_feature_idx, ctx.normalize_window,
        )
        result = cls.find(future_for_label, history_for_label)
        if isinstance(result, (tuple, list)):
            y_obs = float(result[0])
        else:
            y_obs = float(result.item() if hasattr(result, 'item') else result)
        return torch.tensor([y_obs], dtype=torch.float32)

    @classmethod
    def signature_fields(cls, config: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            'latent_distribution': config.get('latent_distribution', 'normal'),
        }

    @classmethod
    def dataset_banner_lines(cls, *, bound, config, dataset) -> List[str]:
        return [
            f"  latent_distribution: "
            f"{bound.get('latent_distribution', config.get('latent_distribution'))}",
            f"  model output_dim (density params): {bound.get('output_dim')}",
            f"  label cache dim (observation): {1}",
        ]

    @classmethod
    def format_metric_lines(cls, metrics: Mapping[str, Any]) -> List[str]:
        lines = [f"  NLL:             {metrics.get('nll', float('nan')):.4f}"]
        if 'location_mae' in metrics:
            lines.append(f"  Location MAE:    {metrics['location_mae']:.4f}")
            lines.append(f"  Location RMSE:   {metrics['location_rmse']:.4f}")
        if 'mean_pred_std' in metrics:
            lines.append(f"  Mean pred std:   {metrics['mean_pred_std']:.4f}")
        if 'mean_pred_df' in metrics:
            lines.append(f"  Mean pred df:    {metrics['mean_pred_df']:.2f}")
        return lines

    @classmethod
    def metrics_report_note(cls, config: Optional[dict] = None) -> str:
        return "Metrics from Task_Latent metric_slots (NLL / location errors)."

    @classmethod
    def decision_threshold_from_config(cls, config: Optional[dict] = None) -> Optional[float]:
        return None

    @classmethod
    def summary_metric_headers(cls) -> Tuple[str, ...]:
        return ('nll', 'location_mae', 'location_rmse')

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
        return "Label loss(en_k, y)  [latent NLL]"

    @classmethod
    def ensemble_loss_increase_warning(
        cls, *, k: int, loss: float, prev_k: int, prev_loss: float
    ) -> str:
        return (
            f"loss(en_{k}, y)={loss:.6f} > loss(en_{prev_k}, y)={prev_loss:.6f}. "
            f"Check residual/norm chain for latent ensembles."
        )


    @classmethod
    def print_prediction_samples(
        cls, preds, test_labels, *, dataset=None, yield_target=None, latent_dist=None, n: int = 5,
    ) -> None:
        n = min(int(n), len(preds))
        loc_n, scale_n, df_t = cls.decode_latent_params(
            preds, latent_distribution=latent_dist
        )
        loc_obs, scale_obs = cls.latent_params_to_obs_space(
            loc_n, scale_n,
            dataset._label_stat_mean, dataset._label_stat_std,
        )
        y_obs = dataset.denormalize_y(test_labels.detach().cpu()).reshape(-1)
        loc_obs = loc_obs.reshape(-1).to(y_obs.device)
        scale_obs = scale_obs.reshape(-1).to(y_obs.device)
        if df_t is None:
            print(f"  {'Index':<6} {'y future[0]':<12} {'μ (obs)':<12} {'σ (obs)':<12}")
            print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12}")
            for i in range(n):
                print(
                    f"  {i:<6} {y_obs[i].item():<+12.4f} "
                    f"{loc_obs[i].item():<+12.4f} {scale_obs[i].item():<12.4f}"
                )
        else:
            df_flat = df_t.reshape(-1)
            print(
                f"  {'Index':<6} {'y future[0]':<12} {'Pred loc':<12} "
                f"{'Pred scale':<12} {'Pred df':<10}"
            )
            print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
            for i in range(n):
                print(
                    f"  {i:<6} {y_obs[i].item():<+12.4f} "
                    f"{loc_obs[i].item():<+12.4f} {scale_obs[i].item():<12.4f} "
                    f"{df_flat[i].item():<10.2f}"
                )

    @classmethod
    def visualize(cls, 
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
        dist = cls.resolve_latent_distribution(
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
