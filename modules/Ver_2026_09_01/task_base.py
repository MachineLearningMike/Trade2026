# task_base.py
# Polymorphic Task interface + shared helpers.
#
# Task is an interface only — never instantiate / call Task.* for behavior.
# Concrete Task_* classes implement the methods. Shared math/windowing lives
# as module-level helpers below (not as filled-in Task methods).

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch


# Neural stack knobs owned by Task.resolve_model_spec (TRAIN/CLI may override
# via config before apply_model_spec writes them back).
MODEL_SPEC_KEYS: Tuple[str, ...] = (
    'model_type',
    'patch_size',
    'num_layers',
    'num_heads',
    'n_latent_macros',
    'ff_dim_multiplier',
    'dropout',
    'causal',
    'head_hidden_dim_multiplier',
    'num_momentum',
    'use_reconstruction',
    'emit_reconstruction',
    'vae_latent_dim',
    'vae_kl_weight',
    'reconstruction_loss_weight',
)


class Task:
    """
    Polymorphic task interface.

    Implement on concrete subclasses. Do not call Task.method for behavior.
    """

    name: str = ''
    MODEL_INPUT_KEYS: Tuple[str, ...] = ('x', 'mean', 'std', 'momentum')
    SAMPLE_TENSOR_KEYS: Tuple[str, ...] = MODEL_INPUT_KEYS + ('y',)
    denormalize_inference_outputs: bool = True
    # If True, raw labels are normalized in-place into the label cache
    # (train mean/std); build_datapoint then just slices that y.
    normalize_y: bool = True
    # Optional map applied to model head output before loss / metrics.
    # None → identity. Example: torch.sigmoid for Rise.
    post_neural_function = None

    @classmethod
    def model_spec_defaults(cls) -> Dict[str, Any]:
        """Task-owned default neural model_type + hyperparameters."""
        return {
            'model_type': 'regression',
            'patch_size': 10,
            'num_layers': 1,
            'num_heads': 5,
            'n_latent_macros': 30,
            'ff_dim_multiplier': 4,
            'dropout': 0.1,
            'causal': False,
            'head_hidden_dim_multiplier': 4,
            'num_momentum': 4,
            'use_reconstruction': False,
            'emit_reconstruction': False,
            'vae_latent_dim': 64,
            'vae_kl_weight': 0.1,
            'reconstruction_loss_weight': 1.0,
        }

    @classmethod
    def refine_model_spec(
        cls, spec: Dict[str, Any], config: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Subclass hook to force policy (e.g. Task_Error arch, VAE↔none)."""
        del config
        return spec

    @classmethod
    def resolve_model_spec(cls, config: Optional[dict] = None) -> Dict[str, Any]:
        """
        Neural model_type + hyperparameters for this Task.

        Starts from ``model_spec_defaults``, then applies non-``None`` values
        from ``config`` (CLI / sweep overrides), then ``refine_model_spec``.
        """
        cfg = dict(config or {})
        spec = dict(cls.model_spec_defaults())
        for key in MODEL_SPEC_KEYS:
            if key in cfg and cfg[key] is not None:
                spec[key] = cfg[key]
        return cls.refine_model_spec(spec, cfg)

    @classmethod
    def apply_model_spec(cls, config: dict) -> dict:
        """Write ``resolve_model_spec`` into ``config`` (in place) and return it."""
        config.update(cls.resolve_model_spec(config))
        return config


def merge_error_stage_model_spec(config: dict, stage: int) -> dict:
    """
    Copy ``config`` and overlay ``error_ensemble_model_specs[stage]``.

    Specs may be a list (index = stage) or a dict keyed by int/str stage.
    Only ``MODEL_SPEC_KEYS`` entries are copied. Missing stage → no overlay.
    """
    cfg = dict(config)
    specs = cfg.get('error_ensemble_model_specs')
    stage_spec = None
    if isinstance(specs, (list, tuple)):
        if 0 <= int(stage) < len(specs):
            stage_spec = specs[int(stage)]
    elif isinstance(specs, Mapping):
        stage_spec = specs.get(int(stage), specs.get(str(int(stage))))
    if stage_spec:
        for key in MODEL_SPEC_KEYS:
            if key in stage_spec and stage_spec[key] is not None:
                cfg[key] = stage_spec[key]
    cfg['error_ensemble_stage'] = int(stage)
    return cfg

    @classmethod
    def denormalize(
        cls, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError(f"{cls.__name__}.denormalize")

    @classmethod
    def resolve_future_window(cls, config: dict) -> int:
        raise NotImplementedError(f"{cls.__name__}.resolve_future_window")

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
    def build_datapoint(cls, dataset: Any, idx: int) -> dict:
        raise NotImplementedError(f"{cls.__name__}.build_datapoint")

    @classmethod
    def get_error(
        cls,
        checkpoint_key: str,
        dataset: Any,
        y: torch.Tensor,
        *,
        config: dict,
        device: Optional[Any] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(f"{cls.__name__}.get_error")

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


def build_datapoint(task, dataset: Any, idx: int) -> dict:
    """Build one sample dict (x, mean, std, momentum, y, task)."""
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
        # one-layer: single train-set global affine (no per-window z-score).
        x_mean = dataset.x_stat_mean
        x_std = dataset.x_stat_std.clone()
        x_std[x_std == 0] = 1
        x_normalized = (x_raw - x_mean) / (x_std + 1e-8)
        mean_normalized = torch.zeros(dataset.F)
        std_normalized = torch.ones(dataset.F)
        momentum_normalized = torch.zeros(dataset.num_momentum, dataset.F)

    # y is already in the right space (raw or train-normalized) from the cache.
    y = dataset.label_tensor[idx][dataset.label_channels]
    return {
        'x': x_normalized,
        'mean': mean_normalized,
        'std': std_normalized,
        'momentum': momentum_normalized,
        'y': y,
        'task': task.name,
    }


def get_error(
    task,
    checkpoint_key: str,
    dataset: Any,
    y: torch.Tensor,
    *,
    config: dict,
    device: Optional[Any] = None,
) -> torch.Tensor:
    """
    Default residual ``y - pred`` for a stored checkpoint.

    Steps through ``dataset`` (e.g. ``train_dataset``) with
    ``validate_batch_size``; applies ``task.post_neural_function`` when set.
    Returns ``(N,)`` when ``output_dim == 1``, else ``(N, output_dim)``.
    """
    from data import ThreadedDataLoader
    from INFER import find_checkpoint_by_key, load_model

    cfg = dict(config)
    task.apply_model_spec(cfg)
    if device is None:
        device = cfg.get('device') or 'cpu'
    ckpt_dir = cfg.get('checkpoint_dir', 'trading_checkpoints')
    found = find_checkpoint_by_key(str(ckpt_dir), key=str(checkpoint_key))
    if found is None:
        raise FileNotFoundError(
            f"{getattr(task, '__name__', type(task).__name__)}.get_error: "
            f"no checkpoint for key {checkpoint_key!r} under {ckpt_dir}"
        )
    checkpoint_path, _ = found
    # Subset → underlying TradingDataset for architecture dims.
    full_ds = getattr(dataset, 'dataset', dataset)
    model = load_model(checkpoint_path, cfg, full_ds, device=device)
    model.eval()

    n = len(dataset)
    if int(y.shape[0]) != n:
        raise ValueError(
            f"{getattr(task, '__name__', type(task).__name__)}.get_error: "
            f"dataset length {n} != y length {int(y.shape[0])}"
        )
    step = int(cfg.get('validate_batch_size', 32) or 32)
    touch = task.post_neural_function
    chunks: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, n, step):
            end = min(start + step, n)
            datapoints = [dataset[i] for i in range(start, end)]
            batch = ThreadedDataLoader.collate_datapoints(datapoints)
            x = batch['x'].to(device, non_blocking=True)
            mean = batch['mean'].to(device, non_blocking=True)
            std = batch['std'].to(device, non_blocking=True)
            momentum = batch['momentum'].to(device, non_blocking=True)
            y_b = y[start:end].to(device, non_blocking=True)
            out = model(x, mean, std, momentum)
            pred = out[0] if isinstance(out, (tuple, list)) else out
            if touch is not None:
                pred = touch(pred)
            err = y_b - pred
            if err.dim() > 1 and err.shape[-1] == 1:
                err = err.squeeze(-1)
            chunks.append(err.detach().cpu())

    return torch.cat(chunks, dim=0)


def postprocess_predictions(task, preds, dataset):
    """Default: denormalize with dataset label stats."""
    if not getattr(task, 'denormalize_inference_outputs', True) or dataset is None:
        return preds
    return task.denormalize(
        preds, dataset.label_stat_mean, dataset.label_stat_std
    )


