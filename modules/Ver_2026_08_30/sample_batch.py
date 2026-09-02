# sample_batch.py
# Opaque sample dict ABI + model input unpack (Phase 3).
#
# Sample keys (atomic supervised / current transformer):
#   x, p, mean, std, shape, y
# Batches are dicts with the same keys, tensors stacked on dim 0.
# Legacy 6-tuples are still accepted by coerce_sample / coerce_batch.

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

Sample = Dict[str, Any]
Batch = Dict[str, Any]

MODEL_INPUT_KEYS: Tuple[str, ...] = ('x', 'p', 'mean', 'std', 'shape')
SAMPLE_TENSOR_KEYS: Tuple[str, ...] = MODEL_INPUT_KEYS + ('y',)


def resolve_normalization_mode(config: Optional[dict] = None) -> str:
    """
    Return 'two-layer' | 'one-layer'.

    Prefers config['normalization']; else maps normalize_window True→two-layer,
    False→one-layer (legacy).
    """
    cfg = config or {}
    raw = cfg.get('normalization', None)
    if raw is not None and str(raw).strip() != '':
        mode = str(raw).strip().lower().replace('_', '-')
        if mode in ('two-layer', 'twolayer', '2-layer', '2layer'):
            return 'two-layer'
        if mode in ('one-layer', 'onelayer', '1-layer', '1layer'):
            return 'one-layer'
        raise ValueError(
            f"normalization must be 'one-layer' or 'two-layer', got {raw!r}"
        )
    return 'two-layer' if bool(cfg.get('normalize_window', True)) else 'one-layer'


def coerce_sample(sample: Union[Sample, Sequence]) -> Sample:
    """Normalize a dataset __getitem__ result to a sample dict."""
    if isinstance(sample, Mapping):
        return dict(sample)
    if isinstance(sample, (tuple, list)):
        if len(sample) == 6:
            x, p, mean, std, shape, y = sample
            return {
                'x': x, 'p': p, 'mean': mean, 'std': std, 'shape': shape, 'y': y,
            }
        if len(sample) == 5:
            x, y, mean, std, shape = sample
            return {
                'x': x, 'p': None, 'mean': mean, 'std': std, 'shape': shape, 'y': y,
            }
    raise TypeError(
        f"sample must be dict or 5/6-tuple, got {type(sample).__name__}"
    )


def collate_samples(samples: List[Union[Sample, Sequence]]) -> Batch:
    """Stack a list of samples into a batched dict."""
    dicts = [coerce_sample(s) for s in samples]
    if not dicts:
        return {}
    out: Batch = {}
    for key in SAMPLE_TENSOR_KEYS:
        vals = [d[key] for d in dicts]
        if vals[0] is None:
            out[key] = None
            continue
        out[key] = torch.stack(vals, dim=0)
    # Pass through non-tensor meta from first sample if present
    for key, val in dicts[0].items():
        if key not in out:
            out[key] = val
    return out


def batch_to_device(batch: Union[Batch, Sequence], device, non_blocking: bool = True) -> Batch:
    """Move tensor fields of a batch (dict or legacy tuple) to device."""
    b = coerce_sample(batch) if not isinstance(batch, Mapping) else dict(batch)
    # If it was a stacked batch tuple from old loader: (x,p,mean,std,shape,y)
    if not isinstance(batch, Mapping) and isinstance(batch, (tuple, list)) and len(batch) == 6:
        x, p, mean, std, shape, y = batch
        b = {'x': x, 'p': p, 'mean': mean, 'std': std, 'shape': shape, 'y': y}
    out: Batch = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


def unpack_model_input(sample_or_batch: Union[Sample, Batch, Sequence]) -> Dict[str, torch.Tensor]:
    """
    ModelSpec.unpack_input — kwargs for forward(x, p, mean, std, shape).

    Works on single samples or batched dicts / legacy tuples.
    """
    if isinstance(sample_or_batch, Mapping):
        b = sample_or_batch
    elif isinstance(sample_or_batch, (tuple, list)) and len(sample_or_batch) >= 5:
        # batch tuple (x,p,mean,std,shape,y) or sample
        if len(sample_or_batch) == 6:
            x, p, mean, std, shape, _y = sample_or_batch
        else:
            x, p, mean, std, shape = sample_or_batch[:5]
        b = {'x': x, 'p': p, 'mean': mean, 'std': std, 'shape': shape}
    else:
        b = coerce_sample(sample_or_batch)
    missing = [k for k in MODEL_INPUT_KEYS if k not in b or b[k] is None]
    if missing:
        raise KeyError(f"model input missing keys {missing}; have {list(b)}")
    return {k: b[k] for k in MODEL_INPUT_KEYS}


def unpack_target(sample_or_batch: Union[Sample, Batch, Sequence]) -> torch.Tensor:
    """LabelSpec.unpack_target — supervised y tensor."""
    if isinstance(sample_or_batch, Mapping):
        if 'y' not in sample_or_batch:
            raise KeyError("sample/batch missing 'y'")
        return sample_or_batch['y']
    if isinstance(sample_or_batch, (tuple, list)) and len(sample_or_batch) == 6:
        return sample_or_batch[5]
    return coerce_sample(sample_or_batch)['y']


def as_train_tuple(
    batch: Union[Batch, Sequence],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Legacy (x, p, mean, std, shape, y) view of a batch dict or tuple.

    Prefer unpack_model_input / unpack_target in new code; this exists for
    ErrorEnsemble and other call sites still on the 6-tuple ABI.
    """
    if isinstance(batch, Mapping):
        kwargs = unpack_model_input(batch)
        y = unpack_target(batch)
        return (
            kwargs['x'], kwargs['p'], kwargs['mean'], kwargs['std'],
            kwargs['shape'], y,
        )
    if isinstance(batch, (tuple, list)) and len(batch) >= 6:
        return tuple(batch[:6])  # type: ignore[return-value]
    b = coerce_sample(batch)
    kwargs = unpack_model_input(b)
    y = unpack_target(b)
    return (
        kwargs['x'], kwargs['p'], kwargs['mean'], kwargs['std'],
        kwargs['shape'], y,
    )


def pack_model_sample(
    *,
    x: torch.Tensor,
    p: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    shape: torch.Tensor,
    y: torch.Tensor,
    **extra: Any,
) -> Sample:
    """Label-facing pack of the current transformer sample schema."""
    sample: Sample = {
        'x': x,
        'p': p,
        'mean': mean,
        'std': std,
        'shape': shape,
        'y': y,
    }
    sample.update(extra)
    return sample
