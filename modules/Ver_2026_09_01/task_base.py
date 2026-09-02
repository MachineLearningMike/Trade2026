# task_base.py
# Polymorphic Task interface + shared helpers.
#
# Task is an interface only — never instantiate / call Task.* for behavior.
# Concrete Task_* classes implement the methods. Shared math/windowing lives
# as module-level helpers below (not as filled-in Task methods).

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch


class Task:
    """
    Polymorphic task interface.

    Implement on concrete subclasses. Do not call Task.method for behavior.
    """

    name: str = ''
    MODEL_INPUT_KEYS: Tuple[str, ...] = ('x', 'p', 'mean', 'std', 'momentum')
    SAMPLE_TENSOR_KEYS: Tuple[str, ...] = MODEL_INPUT_KEYS + ('y',)
    denormalize_inference_outputs: bool = True

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
    def resolve_future_window(cls, config: dict) -> int:
        raise NotImplementedError(f"{cls.__name__}.resolve_future_window")

    @classmethod
    def resolve_model_past_len(cls, config: dict) -> int:
        raise NotImplementedError(f"{cls.__name__}.resolve_model_past_len")

    @classmethod
    def sync_output_dim(cls, config: dict) -> int:
        raise NotImplementedError(f"{cls.__name__}.sync_output_dim")

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
        raise NotImplementedError(f"{cls.__name__}.build_raw_y")

    @classmethod
    def bind_dataset_outputs(cls, config: dict) -> dict:
        raise NotImplementedError(f"{cls.__name__}.bind_dataset_outputs")

    @classmethod
    def signature_fields(cls, config: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(f"{cls.__name__}.signature_fields")

    @classmethod
    def cache_intra_fields(cls, config: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(f"{cls.__name__}.cache_intra_fields")

    @classmethod
    def pack_datapoint(cls, dataset: Any, idx: int) -> dict:
        raise NotImplementedError(f"{cls.__name__}.pack_datapoint")

    @classmethod
    def unpack_datapoint(
        cls, batch: Mapping[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError(f"{cls.__name__}.unpack_datapoint")

    @classmethod
    def loss(
        cls,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_truth: torch.Tensor,
        config: Optional[dict] = None,
    ) -> torch.Tensor:
        """Task-owned loss. Reads its own knobs from ``config``."""
        raise NotImplementedError(f"{cls.__name__}.loss")

    @classmethod
    def metric_slots(cls) -> Tuple[str, ...]:
        raise NotImplementedError(f"{cls.__name__}.metric_slots")

    @classmethod
    def pack_metrics(cls, y_pred, y_truth, **kwargs) -> dict:
        raise NotImplementedError(f"{cls.__name__}.pack_metrics")

    @classmethod
    def unpack_metrics(cls, metrics: dict) -> dict:
        raise NotImplementedError(f"{cls.__name__}.unpack_metrics")

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
        raise NotImplementedError(f"{cls.__name__}.print_prediction_samples")

    @classmethod
    def report_header_fields(
        cls, dataset, config: Optional[dict] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError(f"{cls.__name__}.report_header_fields")

    @classmethod
    def postprocess_predictions(cls, preds, dataset):
        raise NotImplementedError(f"{cls.__name__}.postprocess_predictions")

    @classmethod
    def decode_prediction_params(cls, preds, **kwargs):
        raise NotImplementedError(f"{cls.__name__}.decode_prediction_params")

    @classmethod
    def visualize_inference(cls, *args, **kwargs):
        raise NotImplementedError(f"{cls.__name__}.visualize_inference")


# ---------------------------------------------------------------------------
# Shared helpers (module-level — not Task method bodies)
# ---------------------------------------------------------------------------

def normalize_x(
    window: torch.Tensor,
    compute_moments: bool = True,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Per-window feature normalization (+ optional momentum moments)."""
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

    momentum = torch.stack(
        [
            signed_root(m3, 3) / std_sq,
            signed_root(m4, 4) / std_sq,
            signed_root(m5, 5) / std_sq,
            signed_root(m6, 6) / std_sq,
        ],
        dim=0,
    )
    return normalized_x, mean, std, momentum


def scale_prices_by_window_moments(
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


def ols_fit_segment(
    y: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
):
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


def regime_segment(
    history: torch.Tensor, future: torch.Tensor, h_past: int, h_fut: int
):
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


def pack_datapoint(task, dataset: Any, idx: int) -> dict:
    """Default datapoint pack (x, p, mean, std, momentum, y, label)."""
    from telepred import windows_for_sample

    t = dataset.valid_t[idx]
    win = windows_for_sample(
        t, dataset.telepred_delay, dataset.past_window, dataset.future_window
    )
    x_raw = dataset.Total[win['input'], :]
    mode = getattr(dataset, 'normalization_layers', None) or 'two-layer'
    if mode == 'two-layer':
        mean = dataset.mean_tensor[idx]
        std = dataset.std_tensor[idx].clone()
        std[std == 0] = 1
        x_normalized = (x_raw - mean) / std
        momentum = dataset.momentum_tensor[idx]
        stats = dataset.stats
        mean_normalized = (
            (mean - stats['mean']['mean']) / (stats['mean']['std'] + 1e-8)
        )
        std_normalized = (
            (std - stats['std']['mean']) / (stats['std']['std'] + 1e-8)
        )
        momentum_normalized = (
            (momentum - stats['momentum']['mean'])
            / (stats['momentum']['std'] + 1e-8)
        )
    else:
        x_normalized = x_raw
        mean_normalized = torch.zeros(dataset.F)
        std_normalized = torch.ones(dataset.F)
        momentum_normalized = torch.zeros(dataset.num_momentum, dataset.F)

    y = dataset.label_tensor[idx][dataset.label_slice]
    y_normalized = task.normalize(
        y, dataset.label_stat_mean, dataset.label_stat_std
    )
    recent_prices = x_normalized[
        -dataset.model_past_len:, dataset.price_feature_idx
    ]
    return {
        'x': x_normalized,
        'p': recent_prices,
        'mean': mean_normalized,
        'std': std_normalized,
        'momentum': momentum_normalized,
        'y': y_normalized,
        'label': task.name,
    }


def unpack_datapoint(
    task, batch: Mapping[str, Any]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Default unpack → (model_kwargs, y)."""
    assert isinstance(batch, Mapping)
    keys = task.MODEL_INPUT_KEYS
    missing = [k for k in keys if k not in batch or batch[k] is None]
    assert not missing, f"{task.__name__}: model input missing {missing}"
    assert 'y' in batch, f"{task.__name__}: datapoint missing 'y'"
    return {k: batch[k] for k in keys}, batch['y']


def postprocess_predictions(task, preds, dataset):
    """Default: denormalize with dataset label stats."""
    if not getattr(task, 'denormalize_inference_outputs', True) or dataset is None:
        return preds
    return task.denormalize(
        preds, dataset.label_stat_mean, dataset.label_stat_std
    )


