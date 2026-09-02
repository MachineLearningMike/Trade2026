# task_base.py
# Base class for all tasks.
#
# data.py / trainer.py may know task ids and call Task APIs; they must not
# embed task-internal logic or task-private config key names.

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import torch


@dataclass
class WindowContext:
    """
    Internal market-window view used by Task.build_cache_row.

    Built by Task.build_raw_y from (t, total_x, total_p, config) — data.py
    does not construct this.
    """

    future_prices: Any
    history_prices: Any
    current_price: Any
    mean: torch.Tensor
    std: torch.Tensor
    price_feature_idx: int
    normalize_window: bool  # FeaturePrep two-layer flag (shared)


class Task:
    """
    Base for Task_Trend / Task_Yield / Task_Latent / Task_Rise / Task_None.

    Public API used by data.py:
      normalize_x(window), build_raw_y(t, total_x, total_p, config, *, mean, std),
      normalize / denormalize (y), bind_dataset_outputs, signature_fields,
      dataset_banner_lines, …
    """

    name: str = ''
    # Shared knobs (subclasses inherit; no selective import needed).
    FITNESS_THRESHOLD = 0.75
    HORIZON_SEARCH_LONGEST_LINEAR = 'longest_linear'
    HORIZON_SEARCH_SHORTEST_TO_THRESHOLD = 'shortest_to_threshold'
    VALID_HORIZON_SEARCH_MODES = (
        'longest_linear',
        'shortest_to_threshold',
    )

    # Shared FeaturePrep / OLS helpers — inherited as Task.*/cls.*
    @staticmethod
    def zscore_normalize(
        y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return _zscore_normalize(y, mean, std)

    @staticmethod
    def zscore_denormalize(
        y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return _zscore_denormalize(y, mean, std)

    @staticmethod
    def scale_prices_by_window_moments(
        future_prices,
        history_prices,
        mean: torch.Tensor,
        std: torch.Tensor,
        price_feature_idx: int,
        normalize_window: bool,
    ):
        return _scale_prices_by_window_moments(
            future_prices,
            history_prices,
            mean,
            std,
            price_feature_idx,
            normalize_window,
        )

    @staticmethod
    def ols_fit_segment(
        y: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ):
        return _ols_fit_segment(y, t=t, eps=eps)

    @staticmethod
    def regime_segment(
        history: torch.Tensor, future: torch.Tensor, h_past: int, h_fut: int
    ):
        return _regime_segment(history, future, h_past, h_fut)

    @classmethod
    def normalize_x(
        cls,
        window: torch.Tensor,
        compute_moments: bool = True,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Per-window feature normalization (+ optional momentum moments).

        Returns:
            normalized_x, mean, std, momentum
            (mean/std/momentum are None when compute_moments is False)
        """
        mean = window.mean(dim=0)
        std = window.std(dim=0)
        std = std.clone()
        std[std == 0] = 1
        normalized_x = (window - mean) / std

        if not compute_moments:
            return normalized_x, None, None, None

        centered = window - mean
        m3 = (centered ** 3).mean(dim=0)
        m4 = (centered ** 4).mean(dim=0)
        m5 = (centered ** 5).mean(dim=0)
        m6 = (centered ** 6).mean(dim=0)
        std_sq = std.clamp(min=1e-6)

        def signed_root(x, power):
            return torch.sign(x) * (torch.abs(x) ** (1 / power))

        ms3 = signed_root(m3, 3) / std_sq
        ms4 = signed_root(m4, 4) / std_sq
        ms5 = signed_root(m5, 5) / std_sq
        ms6 = signed_root(m6, 6) / std_sq
        momentum = torch.stack([ms3, ms4, ms5, ms6], dim=0)
        return normalized_x, mean, std, momentum

    @classmethod
    def normalize_policy(cls) -> str:
        return 'zscore'

    @classmethod
    def normalize(
        cls, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError(f"{cls.__name__}.normalize")

    @classmethod
    def denormalize(
        cls, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError(f"{cls.__name__}.denormalize")

    @classmethod
    def make_window_context(
        cls,
        t: int,
        total_x: torch.Tensor,
        total_p: torch.Tensor,
        config: Mapping[str, Any],
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> WindowContext:
        """
        Slice TelePred price windows at bot time ``t``.

        ``total_x`` is the full feature matrix (T, F); ``total_p`` is the
        1-D price series (T,). Shared window sizes come from config
        (past_window / future_window / telepred_delay).
        """
        del total_x  # reserved for tasks that need full feature context
        from telepred import (
            resolve_delay,
            resolve_past_window,
            windows_for_sample,
        )

        delay = int(config.get('telepred_delay', resolve_delay(dict(config))))
        past = int(config.get('past_window', resolve_past_window(dict(config))))
        if 'future_window' in config and config.get('future_window') is not None:
            future = int(config['future_window'])
        else:
            future = int(cls.resolve_future_window(dict(config)))
        win = windows_for_sample(int(t), delay, past, future)
        p = total_p.reshape(-1)
        return WindowContext(
            future_prices=p[win['future']],
            history_prices=p[win['label_history']],
            current_price=p[int(win['bot_time'])],
            mean=mean,
            std=std,
            price_feature_idx=int(config.get('price_feature_idx', 0)),
            normalize_window=(
                str(config.get('normalization_layers', 'two-layer'))
                .strip().lower().replace('_', '-')
                in ('two-layer', 'twolayer', '2-layer', '2layer')
            ),
        )

    @classmethod
    def resolve_future_window(cls, config: dict) -> int:
        return int(config.get('future_window', 50))

    @classmethod
    def build_raw_y(
        cls,
        t: int,
        total_x: torch.Tensor,
        total_p: torch.Tensor,
        config: dict,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Raw cache y at bot time ``t``.

        data.py supplies FeaturePrep moments from normalize_x; Task owns
        TelePred price slicing and task-private label logic.
        """
        ctx = cls.make_window_context(
            t, total_x, total_p, config, mean=mean, std=std
        )
        return cls.build_cache_row(ctx, config)

    @classmethod
    def build_cache_row(cls, ctx: WindowContext, config: dict) -> torch.Tensor:
        """Return full cache y from a sliced window. Prefer build_raw_y from data."""
        raise NotImplementedError(f"{cls.__name__}.build_cache_row")

    @classmethod
    def bind_dataset_outputs(cls, config: dict) -> dict:
        raise NotImplementedError(f"{cls.__name__}.bind_dataset_outputs")

    @classmethod
    def signature_fields(cls, config: Mapping[str, Any]) -> Dict[str, Any]:
        """Task-private fragments for dataset_signature (data.py must not name these)."""
        return {}

    @classmethod
    def cache_intra_fields(cls, config: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Task-private fragments stored in the label-cache pickle under ``intra.*``.

        Default: same as signature_fields. Override when cache meta should
        differ from signature hashing.
        """
        return dict(cls.signature_fields(config))

    @classmethod
    def dataset_banner_lines(
        cls, *, bound: Mapping[str, Any], config: Mapping[str, Any], dataset: Any
    ) -> List[str]:
        """Optional init log lines; default empty."""
        return []


    @classmethod
    def print_prediction_samples(
        cls,
        preds,
        test_labels,
        *,
        dataset=None,
        yield_target=None,
        latent_dist=None,
        n: int = 5,
    ) -> None:
        """Optional human sample table for report_test_metrics."""
        return None

    @classmethod
    def report_header_fields(cls, dataset, config: Optional[dict] = None) -> Dict[str, Any]:
        """Extra key/values for TRAIN/INFER report headers (task-owned)."""
        return {}

    # If False, INFER keeps preds/labels in loss space (e.g. latent NLL).
    denormalize_inference_outputs: bool = True

    @classmethod
    def postprocess_predictions(cls, preds, dataset):
        """
        Map model outputs toward display / metric space after inference.

        Default: denormalize via dataset when available. Override for tasks
        whose head is already in the space metrics expect (e.g. latent).
        """
        if not cls.denormalize_inference_outputs or dataset is None:
            return preds
        denorm = getattr(dataset, 'denormalize_y', None)
        if denorm is None:
            return preds
        # preds may be a batch tensor
        if hasattr(preds, 'shape') and len(preds.shape) >= 1 and preds.shape[0] > 1:
            import torch
            return torch.stack(
                [denorm(preds[i]) for i in range(preds.shape[0])]
            )
        return denorm(preds)

    @classmethod
    def decode_prediction_params(cls, preds, **kwargs):
        """Optional density-head decode; default unsupported."""
        raise NotImplementedError(f"{cls.__name__}.decode_prediction_params")

    @classmethod
    def visualize_inference(cls, *args, **kwargs):
        """INFER viz entry; default forwards to visualize if present."""
        viz = getattr(cls, 'visualize', None)
        if viz is None:
            print(f"⚠ {cls.name}: no visualize() for inference plots")
            return None
        return viz(*args, **kwargs)

    @classmethod
    def task_id(cls) -> str:
        return str(getattr(cls, 'name', cls.__name__))

# ---------------------------------------------------------------------------
# Shared label / task constants
# ---------------------------------------------------------------------------

# Default label fitness threshold (overridden by config['fitness_threshold']).
LABEL_FITNESS_THRESHOLD = 0.75

# Label family ids (config key ``label``).
LABEL_TREND = 'trend'
LABEL_YIELD = 'yield'
LABEL_LATENT = 'latent'
LABEL_RISE_FLAG = 'rise_flag'
LABEL_NONE = 'none'  # unsupervised / (vae, none) — no supervised target
VALID_LABEL_MODES = (
    LABEL_TREND,
    LABEL_YIELD,
    LABEL_LATENT,
    LABEL_RISE_FLAG,
    LABEL_NONE,
)

# Product-facing aliases / (model, task) matrix (formerly registry.py + compat.py)
TRUE_LABEL_IDS = VALID_LABEL_MODES
LABEL_ALIASES = {}

MODEL_REGRESSION = 'regression'
MODEL_VAE = 'vae'
MODEL_ENSEMBLE = 'ensemble'
ATOMIC_MODEL_IDS = (MODEL_REGRESSION, MODEL_VAE)
MODEL_IDS = ATOMIC_MODEL_IDS + (MODEL_ENSEMBLE,)
MODEL_ALIASES = {}

COMPAT = {
    (MODEL_REGRESSION, LABEL_TREND): 'supervised',
    (MODEL_REGRESSION, LABEL_YIELD): 'supervised',
    (MODEL_REGRESSION, LABEL_LATENT): 'supervised',
    (MODEL_REGRESSION, LABEL_RISE_FLAG): 'supervised',
    (MODEL_VAE, LABEL_NONE): 'unsupervised',
}

ENSEMBLE_KINDS = {
    'telepred': {
        'members': 'repeat_atomic',
        'label_relation': 'identical',
        'fusion': 'delay_bank',
        'requires_supervised': True,
    },
    'error': {
        'members': 'repeat_atomic',
        'label_relation': 'identical',
        'fusion': 'sum_residuals_y_space',
        'atomic_model': MODEL_REGRESSION,
        'atomic_labels': frozenset({LABEL_TREND, LABEL_YIELD, LABEL_RISE_FLAG}),
    },
    'stack': {
        'members': 'list_of_atomic',
        'label_relation': 'collection',
        'fusion': 'FusionSpec',
    },
}


LABEL_FULL_DIM_TREND = 3
LABEL_FULL_DIM_YIELD = 2
LABEL_FULL_DIM_YIELD_SINGLE = 1  # chance-only or near-only head
# Latent: scalar observation in the label cache; model head width from density mode.
LABEL_FULL_DIM_LATENT = 1
LABEL_FULL_DIM_RISE_FLAG = 1  # binary flag {0, 1}
LABEL_FULL_DIM_NONE = 1  # placeholder head width when no supervised y

# Horizon search rules for linear trend (free OLS, no f[0] anchor).
HORIZON_SEARCH_LONGEST_LINEAR = 'longest_linear'
HORIZON_SEARCH_SHORTEST_TO_THRESHOLD = 'shortest_to_threshold'
VALID_HORIZON_SEARCH_MODES = (
    HORIZON_SEARCH_LONGEST_LINEAR,
    HORIZON_SEARCH_SHORTEST_TO_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Shared OLS / FeaturePrep helpers (used by multiple Task families)
# ---------------------------------------------------------------------------

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


def _scale_prices_by_window_moments(
    future_prices,
    history_prices,
    mean: torch.Tensor,
    std: torch.Tensor,
    price_feature_idx: int,
    normalize_window: bool,
):
    """Optional FeaturePrep price scaling before Task.find."""
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


def _zscore_normalize(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.to(device=y.device, dtype=y.dtype)
    std = std.to(device=y.device, dtype=y.dtype)
    if y.shape[-1] != mean.shape[-1]:
        raise ValueError(
            f"y last dim {tuple(y.shape)} incompatible with mean {tuple(mean.shape)}"
        )
    return (y - mean) / (std + 1e-8)


def _zscore_denormalize(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.to(device=y.device, dtype=y.dtype)
    std = std.to(device=y.device, dtype=y.dtype)
    if y.shape[-1] != mean.shape[-1]:
        raise ValueError(
            f"y last dim {tuple(y.shape)} incompatible with mean {tuple(mean.shape)}"
        )
    return y * std + mean


# ---------------------------------------------------------------------------
# Registry (lazy imports avoid Task_* ↔ telepred cycles)
# ---------------------------------------------------------------------------

def _load_task_classes() -> Dict[str, Type]:
    from Task_Trend import Task_Trend
    from Task_Yield import Task_Yield
    from Task_Latent import Task_Latent
    from Task_Rise import Task_Rise
    from Task_None import Task_None
    return {
        LABEL_TREND: Task_Trend,
        LABEL_YIELD: Task_Yield,
        LABEL_LATENT: Task_Latent,
        LABEL_RISE_FLAG: Task_Rise,
        LABEL_NONE: Task_None,
    }


_TASK_CLASSES: Optional[Dict[str, Type]] = None


def get_task_classes() -> Dict[str, Type]:
    global _TASK_CLASSES
    if _TASK_CLASSES is None:
        _TASK_CLASSES = _load_task_classes()
    return _TASK_CLASSES


# Lazy mapping proxy so ``from task_base import TASK_CLASSES`` still works.
class _TaskClassesProxy(dict):
    def _ensure(self):
        if not dict.__len__(self):
            self.update(_load_task_classes())

    def __getitem__(self, key):
        self._ensure()
        return dict.__getitem__(self, key)

    def __contains__(self, key):
        self._ensure()
        return dict.__contains__(self, key)

    def __iter__(self):
        self._ensure()
        return dict.__iter__(self)

    def keys(self):
        self._ensure()
        return dict.keys(self)

    def values(self):
        self._ensure()
        return dict.values(self)

    def items(self):
        self._ensure()
        return dict.items(self)

    def __len__(self):
        self._ensure()
        return dict.__len__(self)

    def __repr__(self):
        self._ensure()
        return dict.__repr__(self)


TASK_CLASSES: Dict[str, Type] = _TaskClassesProxy()  # type: ignore[assignment]


def get_task_class(task_mode: str) -> Type:
    classes = get_task_classes()
    if task_mode not in classes:
        raise ValueError(
            f"Unknown task mode {task_mode!r}; expected one of {tuple(classes)}"
        )
    return classes[task_mode]


def resolve_future_window(config: dict) -> int:
    mode = config.get('label', LABEL_TREND)
    return get_task_class(mode).resolve_future_window(config)


def resolve_model_past_len(config: dict) -> int:
    mode = config.get('label', LABEL_TREND)
    return get_task_class(mode).resolve_model_past_len(config)


def sync_output_dim(config: dict) -> int:
    """Force ``config['output_dim']`` via Task.sync_output_dim."""
    cls = get_task_class(config.get('label', LABEL_TREND))
    return int(cls.sync_output_dim(config))

def plot_label_member_ensemble_metrics(
    label_mode: str,
    member_metrics_by_stage: Sequence[Mapping[str, Any]],
    ensemble_metrics_by_stage: Sequence[Mapping[str, Any]],
    *,
    save_path: Union[str, Path],
    title: Optional[str] = None,
    decision_threshold: Optional[float] = None,
    split_name: str = 'val',
):
    cls = get_task_class(label_mode)
    fn = getattr(cls, 'plot_member_ensemble_metrics', None)
    if fn is None:
        return None
    return fn(
        member_metrics_by_stage,
        ensemble_metrics_by_stage,
        save_path=save_path,
        title=title,
        decision_threshold=decision_threshold,
        split_name=split_name,
    )


def bind_task_loss(
    task,
    config: Optional[dict] = None,
    **overrides,
):
    """
    Return ``task.loss`` with auxiliary bound for ResumableTrainer.

    ``task`` is a Task class (from get_task_class / caller-owned).
    Signature: loss_fn(x, y_pred, y_truth) -> Tensor
    """
    auxiliary = task.make_auxiliary(config, **overrides)

    def loss_fn(
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_truth: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]
        return task.loss(x, y_pred, y_truth, auxiliary)

    loss_fn.task = task  # type: ignore[attr-defined]
    loss_fn.auxiliary = auxiliary  # type: ignore[attr-defined]
    return loss_fn


# ---------------------------------------------------------------------------
# Metrics orchestration (formerly label_metrics.py)
# ---------------------------------------------------------------------------

def compute_label_metrics(
    label_mode: Optional[str] = None,
    preds: torch.Tensor = None,
    y_truth: torch.Tensor = None,
    *,
    task=None,
    config: Optional[dict] = None,
    dataset=None,
    yield_target: Optional[str] = None,
    latent_distribution: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute the active task's metric slots (no printing).

    Prefer ``task=``. ``label_mode`` remains for older call sites.
    Do not borrow ``dataset.task``.
    """
    if preds is None or y_truth is None:
        raise TypeError("compute_label_metrics requires preds and y_truth")
    cfg = dict(config or {})
    if task is None:
        if label_mode is None:
            raise TypeError("compute_label_metrics needs task= or label_mode=")
        task = get_task_class(label_mode)
    mode = getattr(task, 'name', None) or str(label_mode)
    if yield_target is not None:
        cfg['yield_target'] = yield_target
    elif dataset is not None and getattr(dataset, 'yield_target', None):
        cfg['yield_target'] = dataset.yield_target
    if latent_distribution is not None:
        cfg['latent_distribution'] = latent_distribution
    elif dataset is not None and getattr(dataset, 'latent_distribution', None):
        cfg['latent_distribution'] = dataset.latent_distribution

    aux = task.make_auxiliary(cfg)
    packed = task.pack_metrics(
        preds,
        y_truth,
        auxiliary=aux,
        dataset=dataset,
        config=cfg,
    )
    metrics = task.unpack_metrics(packed)
    for k, v in packed.items():
        metrics.setdefault(k, v)
    return metrics


def report_test_metrics(
    label_mode: Optional[str] = None,
    preds: torch.Tensor = None,
    test_labels: torch.Tensor = None,
    *,
    task=None,
    config: Optional[dict] = None,
    dataset=None,
    yield_target: Optional[str] = None,
    latent_distribution: Optional[str] = None,
    print_samples: bool = True,
    n_samples: int = 5,
) -> Dict[str, Any]:
    """
    Compute + print test metrics via the active task's metric slots.

    Prefer ``task=``. Positional ``label_mode, preds, test_labels`` still
    works for older smokes. Do not borrow ``dataset.task``.
    """
    if preds is None or test_labels is None:
        raise TypeError("report_test_metrics requires preds and test_labels")

    cfg = dict(config or {})
    if task is None:
        if label_mode is None:
            raise TypeError("report_test_metrics needs task= or label_mode=")
        task = get_task_class(label_mode)
    mode = getattr(task, 'name', None) or str(label_mode)

    if yield_target is not None:
        cfg['yield_target'] = yield_target
    elif dataset is not None and getattr(dataset, 'yield_target', None):
        cfg['yield_target'] = dataset.yield_target
    if latent_distribution is not None:
        cfg['latent_distribution'] = latent_distribution
    elif dataset is not None and getattr(dataset, 'latent_distribution', None):
        cfg['latent_distribution'] = dataset.latent_distribution

    yt = cfg.get('yield_target')
    ld = cfg.get('latent_distribution')
    resolve_yt = getattr(task, 'resolve_yield_target', None)
    resolve_ld = getattr(task, 'resolve_latent_distribution', None)
    if callable(resolve_yt):
        yt = resolve_yt(cfg)
    if callable(resolve_ld):
        ld = resolve_ld(cfg)

    print("\n📊 Test Set Results:")
    print(f"  Metric slots:   {list(task.metric_slots())}")

    metrics = compute_label_metrics(
        task=task,
        preds=preds,
        y_truth=test_labels,
        config=cfg,
        dataset=dataset,
        yield_target=yt,
        latent_distribution=ld,
    )
    _print_metrics(mode, metrics)

    if print_samples and mode != LABEL_NONE:
        print(f"\n📈 Sample Predictions (first {n_samples}):")
        task.print_prediction_samples(
            preds,
            test_labels,
            dataset=dataset,
            yield_target=yt,
            latent_dist=ld,
            n=n_samples,
        )
    return metrics


def format_metric_lines(mode: str, metrics: Mapping[str, Any]) -> List[str]:
    """Human-readable lines — owned by Task.format_metric_lines."""
    return list(get_task_class(mode).format_metric_lines(metrics))


def _print_metrics(mode: str, metrics: dict) -> None:
    for line in format_metric_lines(mode, metrics):
        print(line)


def build_report_text(
    *,
    report_name: str,
    label_mode: str,
    model_type: str,
    checkpoint_key: str,
    checkpoint_path: str,
    metric_slots: Sequence[str],
    val_metrics: Optional[Mapping[str, Any]] = None,
    test_metrics: Optional[Mapping[str, Any]] = None,
    val_n: Optional[int] = None,
    test_n: Optional[int] = None,
    extra_header: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Plain-text report body for ``report.text``.

    Metrics come from the hosting label's ``metric_slots`` / pack_metrics.
    """
    mode = get_task_class(label_mode).name
    lines: List[str] = [
        f"Report:           {report_name}",
        f"Label:            {mode}",
        f"Model:            {model_type}",
        f"Checkpoint key:   {checkpoint_key}",
        f"Checkpoint path:  {checkpoint_path}",
        f"Metric slots:    {list(metric_slots)}",
    ]
    if extra_header:
        for k, v in extra_header.items():
            lines.append(f"{k}: {v}")
    lines.append("")

    def _section(title: str, metrics: Optional[Mapping[str, Any]], n: Optional[int]) -> None:
        lines.append(f"=== {title} ===")
        if n is not None:
            lines.append(f"  n_samples:       {n}")
        if metrics is None:
            lines.append("  (not computed)")
        else:
            lines.extend(format_metric_lines(mode, metrics))
        lines.append("")

    _section("Validation", val_metrics, val_n)
    _section("Test", test_metrics, test_n)
    return "\n".join(lines).rstrip() + "\n"


def write_report_text_file(
    report_dir: Union[str, Path],
    text: str,
    *,
    filename: str = "report.text",
) -> Path:
    """Write ``report.text`` under ``report_dir``; return the path."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / filename
    path.write_text(text, encoding="utf-8")
    return path


def metrics_to_jsonable(metrics: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """JSON-friendly copy of a metrics dict."""
    if metrics is None:
        return None
    out: Dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().tolist()
        elif hasattr(v, "item"):
            try:
                out[k] = v.item()
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def evaluate_loaded_model_metrics(
    model,
    dataset,
    config: dict,
    *,
    task=None,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run val + test inference on an already-loaded model; return label metrics.

    Uses the same INFER streaming path as single-run ``report`` / ``infer``.
    For ``rise_flag``, preds are logits (identity denorm) and truth is ``{0,1}``.
    ``task`` is caller-owned (do not borrow ``dataset.task``).
    """
    from INFER import (
        _select_val_indices,
        run_inference_on_indices,
        run_inference_on_test_set,
    )

    cfg = dict(config or {})
    if task is None:
        task = get_task_class(cfg.get('label', 'trend'))
    label_mode = task.name
    yt = getattr(dataset, 'yield_target', None)
    ld = getattr(dataset, 'latent_distribution', None)
    resolve_yt = getattr(task, 'resolve_yield_target', None)
    resolve_ld = getattr(task, 'resolve_latent_distribution', None)
    if callable(resolve_yt):
        yt = resolve_yt(cfg, yield_target=yt)
    if callable(resolve_ld):
        ld = resolve_ld(cfg, latent_distribution=ld)
    if device is None:
        device = str(cfg.get('device') or 'cpu')
    if batch_size is None:
        batch_size = int(
            cfg.get(
                'validate_batch_size',
                cfg.get('test_batch_size', cfg.get('batch_size', 8)),
            )
        )

    result: Dict[str, Any] = {
        'label': label_mode,
        'metric_slots': list(task.metric_slots()),
        'val_metrics': None,
        'test_metrics': None,
        'val_n': None,
        'test_n': None,
    }

    val_indices = _select_val_indices(dataset, max_samples=None)
    if val_indices:
        print("   Running validation inference for metrics...", flush=True)
        val_preds, val_labels, _ = run_inference_on_indices(
            model, dataset, val_indices,
            device=device, batch_size=batch_size, split_name='validation samples',
            task=task,
        )
        result['val_n'] = int(len(val_preds))
        result['val_metrics'] = compute_label_metrics(
            task=task, preds=val_preds, y_truth=val_labels,
            config=cfg, dataset=dataset,
            yield_target=yt, latent_distribution=ld,
        )
        print(f"   ✓ Val metrics on {result['val_n']} samples", flush=True)

    print("   Running test inference for metrics...", flush=True)
    try:
        test_preds, test_labels, _ = run_inference_on_test_set(
            model, dataset,
            device=device, batch_size=batch_size,
            max_samples=None, shuffle=False,
            task=task,
        )
        result['test_n'] = int(len(test_preds))
        result['test_metrics'] = compute_label_metrics(
            task=task, preds=test_preds, y_truth=test_labels,
            config=cfg, dataset=dataset,
            yield_target=yt, latent_distribution=ld,
        )
        print(f"   ✓ Test metrics on {result['test_n']} samples", flush=True)
    except Exception as exc:
        print(f"   ⚠ Test metrics skipped: {exc}", flush=True)

    return result


def evaluate_checkpoint_key_metrics(
    config: dict,
    dataset,
    checkpoint_key: str,
    *,
    task=None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Load ``best_val`` (or newest) weights for ``checkpoint_key`` and compute metrics."""
    from INFER import find_checkpoint_by_key, load_model

    cfg = dict(config or {})
    if task is None:
        task = get_task_class(cfg.get('label', 'trend'))
    ckpt_root = checkpoint_dir or cfg.get('checkpoint_dir', 'trading_checkpoints')
    found = find_checkpoint_by_key(str(ckpt_root), checkpoint_key)
    if found is None:
        raise FileNotFoundError(
            f"No checkpoint weights for key {checkpoint_key!r} under {ckpt_root}"
        )
    checkpoint_path, key = found
    if device is None:
        device = str(cfg.get('device') or 'cpu')
    model = load_model(checkpoint_path, cfg, dataset, device)
    out = evaluate_loaded_model_metrics(
        model, dataset, cfg, task=task, device=device, batch_size=batch_size,
    )
    out['checkpoint_key'] = key
    out['checkpoint_path'] = str(checkpoint_path)
    return out


def write_metrics_report_text(
    report_dir: Union[str, Path],
    *,
    report_name: str,
    label_mode: str,
    model_type: str,
    checkpoint_key: str,
    checkpoint_path: str,
    val_metrics: Optional[Mapping[str, Any]] = None,
    test_metrics: Optional[Mapping[str, Any]] = None,
    val_n: Optional[int] = None,
    test_n: Optional[int] = None,
    extra_header: Optional[Mapping[str, Any]] = None,
    filename: str = "report.text",
) -> Path:
    """Build + write ``report.text`` for one entity (member / en_k / single run)."""
    slots = list(get_task_class(label_mode).metric_slots())
    text = build_report_text(
        report_name=report_name,
        label_mode=label_mode,
        model_type=model_type,
        checkpoint_key=checkpoint_key,
        checkpoint_path=checkpoint_path,
        metric_slots=slots,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_n=val_n,
        test_n=test_n,
        extra_header=extra_header,
    )
    return write_report_text_file(report_dir, text, filename=filename)


def plot_member_vs_ensemble_metrics(
    label_mode: str,
    member_metrics_by_stage: List[Mapping[str, Any]],
    ensemble_metrics_by_stage: List[Mapping[str, Any]],
    *,
    save_path: Union[str, Path],
    decision_threshold: Optional[float] = None,
    split_name: str = "val",
    title: Optional[str] = None,
):
    """
    Task-owned plot: 2xn series on one axes (member solid, en dashed).

    Returns the saved path, or None if the task has no plot helper.
    """
    return plot_label_member_ensemble_metrics(
        label_mode,
        member_metrics_by_stage,
        ensemble_metrics_by_stage,
        save_path=save_path,
        title=title,
        decision_threshold=decision_threshold,
        split_name=split_name,
    )


# ---------------------------------------------------------------------------
# Product compatibility (formerly registry.py / compat.py)
# ---------------------------------------------------------------------------


def normalize_label_id(label: Optional[str]) -> str:
    raw = str(label or LABEL_TREND).lower().strip()
    if raw not in TRUE_LABEL_IDS:
        raise ValueError(
            f"Unknown label id {label!r}; expected one of {TRUE_LABEL_IDS}"
        )
    return raw


def normalize_model_id(model: Optional[str]) -> str:
    raw = str(model or MODEL_REGRESSION).lower().strip()
    return MODEL_ALIASES.get(raw, raw)


def normalize_ensemble_kind(kind: Optional[str]) -> Optional[str]:
    if kind is None:
        return None
    k = str(kind).lower().strip()
    return k or None


def is_true_label(label: Optional[str]) -> bool:
    return normalize_label_id(label) in TRUE_LABEL_IDS


def is_atomic_model(model: Optional[str]) -> bool:
    return normalize_model_id(model) in ATOMIC_MODEL_IDS


def label_cli_choices() -> Tuple[str, ...]:
    return TRUE_LABEL_IDS


def model_cli_choices() -> Tuple[str, ...]:
    return ATOMIC_MODEL_IDS


def compat_objective(model: Optional[str], label: Optional[str]) -> Optional[str]:
    m = normalize_model_id(model)
    lab = normalize_label_id(label)
    return COMPAT.get((m, lab))


def describe_matrix() -> str:
    lines = ['Allowed atomic (model, label) pairs:']
    for (m, lab), kind in sorted(COMPAT.items()):
        lines.append(f'  ({m}, {lab}) -> {kind}')
    lines.append('Ensemble kinds:')
    for name, meta in sorted(ENSEMBLE_KINDS.items()):
        lines.append(f'  {name}: {meta}')
    return chr(10).join(lines)


class CompatibilityError(ValueError):
    """Unsupported (model, label) or ensemble kind."""


def assert_compatible(
    model: Optional[str],
    label: Optional[str],
    ensemble_kind: Optional[str] = None,
) -> str:
    """
    Validate an atomic or ensemble training/infer slot.

    Returns the COMPAT objective kind for atomic pairs (e.g. 'supervised'),
    or the ensemble kind name when ``ensemble_kind`` is set.

    Raises CompatibilityError with the allowed matrix on failure.
    """
    m = normalize_model_id(model)
    lab = normalize_label_id(label)
    kind = normalize_ensemble_kind(ensemble_kind)

    if m == MODEL_ENSEMBLE and kind is None:
        raise CompatibilityError(
            f"model_type={m!r} requires an ensemble_kind "
            f"(one of {tuple(ENSEMBLE_KINDS)}).\n"
            f"{describe_matrix()}"
        )

    if kind is not None:
        return _assert_ensemble(m, lab, kind)

    if m not in (MODEL_REGRESSION, MODEL_VAE):
        raise CompatibilityError(
            f"Unknown or non-atomic model_type={m!r}. "
            f"Atomic models: {(MODEL_REGRESSION, MODEL_VAE)}; "
            f"use ensemble_kind for composed models.\n"
            f"{describe_matrix()}"
        )

    objective = COMPAT.get((m, lab))
    if objective is None:
        # Helpful hint for the old silent VAE+supervised lie.
        hint = ''
        if m == MODEL_VAE and lab != 'none':
            hint = (
                f"\nHint: VAE train objective ignores supervised labels; "
                f"use --label none (pair (vae, none)). "
                f"Got label={lab!r}."
            )
        raise CompatibilityError(
            f"Unsupported pair (model_type={m!r}, label={lab!r})."
            f"{hint}\n"
            f"{describe_matrix()}"
        )
    return objective


def _assert_ensemble(model: str, label: str, kind: str) -> str:
    meta = ENSEMBLE_KINDS.get(kind)
    if meta is None:
        raise CompatibilityError(
            f"Unknown ensemble_kind={kind!r}; "
            f"expected one of {tuple(ENSEMBLE_KINDS)}.\n"
            f"{describe_matrix()}"
        )

    if kind == 'telepred':
        # Member is a single atomic pair, repeated over delays.
        # model should be atomic (regression/vae), not 'ensemble'.
        m = model if model != MODEL_ENSEMBLE else MODEL_REGRESSION
        # Prefer explicit model from config; if caller passed ensemble, still
        # validate label against supervised COMPAT via regression default — but
        # TRAIN passes real model_type.
        objective = COMPAT.get((normalize_model_id(m), label))
        if objective is None:
            raise CompatibilityError(
                f"ensemble_kind='telepred' requires a COMPAT atomic pair; "
                f"got (model_type={m!r}, label={label!r}).\n"
                f"{describe_matrix()}"
            )
        if meta.get('requires_supervised') and objective != 'supervised':
            raise CompatibilityError(
                f"ensemble_kind='telepred' requires a supervised pair; "
                f"got ({m!r}, {label!r}) → {objective!r}.\n"
                f"{describe_matrix()}"
            )
        return kind

    if kind == 'error':
        m = normalize_model_id(model)
        if m == MODEL_ENSEMBLE:
            m = MODEL_REGRESSION
        allowed_model = meta.get('atomic_model', MODEL_REGRESSION)
        allowed_labels = meta.get('atomic_labels', frozenset())
        if m != allowed_model:
            raise CompatibilityError(
                f"ensemble_kind='error' requires model_type={allowed_model!r}; "
                f"got {m!r}.\n"
                f"{describe_matrix()}"
            )
        if label not in allowed_labels:
            raise CompatibilityError(
                f"ensemble_kind='error' requires label in "
                f"{sorted(allowed_labels)}; got {label!r}.\n"
                f"{describe_matrix()}"
            )
        # Also must be a COMPAT cell
        if COMPAT.get((m, label)) is None:
            raise CompatibilityError(
                f"ensemble_kind='error' pair ({m!r}, {label!r}) not in COMPAT.\n"
                f"{describe_matrix()}"
            )
        return kind

    if kind == 'stack':
        # Members validated by caller when FusionSpec is implemented.
        return kind

    raise CompatibilityError(
        f"ensemble_kind={kind!r} has no assert rules yet.\n"
        f"{describe_matrix()}"
    )


def assert_config_compatible(
    config: dict,
    *,
    ensemble_kind: Optional[str] = None,
) -> str:
    """assert_compatible using config['model_type'] / config['label']."""
    return assert_compatible(
        config.get('model_type', MODEL_REGRESSION),
        config.get('label', 'trend'),
        ensemble_kind=ensemble_kind,
    )


def resolve_atomic_slot(
    config: dict,
    *,
    coerce_vae_to_none: bool = True,
    warn: bool = True,
) -> str:
    """
    Normalize config to a COMPAT atomic slot and return the objective kind.

    Phase 4: if model_type=vae and label is not none, coerce label→none
    (default) so the product cell is honest: (vae, none) unsupervised.
    Mutates ``config`` in place when coercing.
    """
    m = normalize_model_id(config.get('model_type', MODEL_REGRESSION))
    lab = normalize_label_id(config.get('label', 'trend'))
    config['model_type'] = m

    if coerce_vae_to_none and m == MODEL_VAE and lab != LABEL_NONE:
        if warn:
            print(
                f"⚠ model_type=vae requires label=none (COMPAT unsupervised); "
                f"coercing label {lab!r} → {LABEL_NONE!r}"
            )
        lab = LABEL_NONE
        config['label'] = LABEL_NONE

    config['label'] = lab
    return assert_compatible(m, lab, ensemble_kind=None)

