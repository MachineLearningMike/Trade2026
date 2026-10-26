# Task_Error.py — residual error stage (Y₀ contribution after denorm).

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from task_base import Task


__all__ = ['Task_Error']


class Task_Error(Task):
    """
    Predict a residual already expressed in the parent Task's normalized y
    space (Y₀). ``normalize_y`` whitens that residual for training; denorm
    with this Task's label stats maps head output back to Y₀ for the flat
    ensemble sum.
    """

    name = 'error'
    normalize_y = True

    @classmethod
    def model_spec_defaults(cls) -> Dict[str, Any]:
        # Same regressor stack as ordinary supervised Tasks by default.
        return super().model_spec_defaults()

    @classmethod
    def resolve_future_window(cls, config: dict) -> int:
        return int(config.get('future_window', 50))

    @classmethod
    def resolve_output_dim(cls, config: Optional[dict] = None, **_kwargs) -> int:
        cfg = config or {}
        return int(cfg.get('output_dim', 1))

    @classmethod
    def sync_output_dim(cls, config: dict) -> int:
        config['output_dim'] = cls.resolve_output_dim(config)
        return int(config['output_dim'])

    @classmethod
    def target_slice(cls, output_dim: int, **_kwargs):
        return slice(0, int(output_dim))

    @classmethod
    def target_names(cls, output_dim: int, **_kwargs) -> Tuple[str, ...]:
        d = int(output_dim)
        if d == 1:
            return ('residual',)
        return tuple(f'residual_{i}' for i in range(d))

    @classmethod
    def bind_dataset_outputs(cls, config: dict) -> dict:
        d = cls.resolve_output_dim(config)
        return {
            'yield_target': 'both',
            'output_dim': d,
            'latent_dim': d,
            'label_channels': cls.target_slice(d),
        }

    @classmethod
    def signature_fields(cls, config) -> Dict[str, Any]:
        return {
            'task': cls.name,
            'output_dim': int(config.get('output_dim', 1)),
            'error_ensemble_stage': config.get('error_ensemble_stage'),
        }

    @classmethod
    def cache_intra_fields(cls, config):
        return dict(cls.signature_fields(config))

    @classmethod
    def build_raw_y(cls, t, total_x, total_p, config, *, mean, std):
        raise RuntimeError(
            f"{cls.__name__}.build_raw_y: residuals are injected via "
            f"ErrorStageDataset, not built from prices"
        )

    @classmethod
    def denormalize(
        cls, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        mean = mean.to(device=y.device, dtype=y.dtype)
        std = std.to(device=y.device, dtype=y.dtype)
        return y * std + mean

    @classmethod
    def loss(
        cls,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_truth: torch.Tensor,
        config: Optional[dict] = None,
    ) -> torch.Tensor:
        del x, config
        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]
        return nn.functional.mse_loss(y_pred, y_truth)

    @classmethod
    def metric_slots(cls) -> Tuple[str, ...]:
        return ('mse',)

    @classmethod
    def pack_metrics(cls, y_pred, y_truth, **kwargs) -> dict:
        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]
        return {
            'mse': float(nn.functional.mse_loss(y_pred, y_truth).item()),
        }

    @classmethod
    def unpack_metrics(cls, metrics: dict) -> dict:
        return {k: metrics[k] for k in cls.metric_slots() if k in metrics}

    @classmethod
    def build_datapoint(cls, dataset, idx):
        from task_base import build_datapoint as _build
        return _build(cls, dataset, idx)

    @classmethod
    def get_error(cls, checkpoint_key, dataset, y, *, config, device=None):
        from task_base import get_error as _get_error
        return _get_error(
            cls, checkpoint_key, dataset, y, config=config, device=device,
        )

    @classmethod
    def report_header_fields(cls, dataset, config=None):
        return {
            'task': cls.name,
            'output_dim': int(getattr(dataset, 'output_dim', 1)),
            'error_ensemble_stage': (config or {}).get('error_ensemble_stage'),
        }

    @classmethod
    def print_prediction_samples(cls, preds, test_labels, **kwargs):
        return None

    @classmethod
    def postprocess_predictions(cls, preds, dataset):
        from task_base import postprocess_predictions as _post
        return _post(cls, preds, dataset)

    @classmethod
    def visualize_inference(cls, *args, **kwargs):
        return None
