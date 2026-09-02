# Task_None.py — task implementation (canonical; not a re-export)

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from task_base import Task


__all__ = ['Task_None']


class Task_None(Task):
    """
    No supervised target — for ``(vae, none)`` and similar unsupervised slots.

    Cache holds a placeholder scalar of zeros. Loss is a zero that stays on
    the prediction graph (VAE wrap_model_loss ignores y anyway).
    """

    name = 'none'

    @classmethod
    def full_cache_dim(cls) -> int:
        return 1

    @classmethod
    def full_dim(cls, **_kwargs) -> int:
        return 1

    @classmethod
    def resolve_future_window(cls, config: dict) -> int:
        return int(config.get('future_window', 50))

    @classmethod
    def resolve_model_past_len(cls, config: dict) -> int:
        return cls.resolve_future_window(config)

    @classmethod
    def resolve_output_dim(cls, config: Optional[dict] = None, **_kwargs) -> int:
        # Placeholder head width is always 1 (cache row is a scalar zero).
        return 1

    @classmethod
    def target_slice(cls, output_dim: int, **_kwargs) -> slice:
        return slice(0, max(1, int(output_dim)))

    @classmethod
    def target_names(cls, output_dim: int, **_kwargs) -> Tuple[str, ...]:
        return ('none',)

    @classmethod
    def find(cls, future, current, **_kwargs):
        return (0.0,)

    @classmethod
    def loss(
        cls,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_truth: torch.Tensor,
        config: Optional[dict] = None,
    ) -> torch.Tensor:
        del x, y_truth, config
        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]
        return y_pred.sum() * 0.0

    @classmethod
    def metric_slots(cls) -> Tuple[str, ...]:
        return ()

    @classmethod
    def compute_metrics(
        cls,
        y_pred: torch.Tensor,
        y_truth: torch.Tensor,
        *,
        dataset=None,
        config: Optional[dict] = None,
        **_kwargs,
    ) -> dict:
        del y_pred, y_truth, dataset, config
        return {'note': 'label=none (no supervised metrics)'}

    @classmethod
    def pack_metrics(cls, y_pred, y_truth, **kwargs) -> dict:
        return cls.compute_metrics(y_pred, y_truth, **kwargs)

    @classmethod
    def unpack_metrics(cls, metrics: dict) -> dict:
        return {}

    @classmethod
    def normalize(cls, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return y

    @classmethod
    def denormalize(cls, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return y

    @classmethod
    def bind_dataset_outputs(cls, config: dict) -> dict:
        forced_dim = cls.resolve_output_dim(config)
        return {
            'yield_target': 'both',
            'output_dim': forced_dim,
            'latent_dim': forced_dim,
            'label_slice': cls.target_slice(forced_dim),
        }

    @classmethod
    def sync_output_dim(cls, config: dict) -> int:
        config['output_dim'] = cls.resolve_output_dim(config)
        return int(config['output_dim'])

    @classmethod
    def build_raw_y(
        cls, t, total_x, total_p, config, *, mean, std,
    ) -> torch.Tensor:
        from telepred import windows_for_sample

        del total_x, mean, std
        delay = int(config.get('telepred_delay', 0))
        past = int(config['past_window'])
        future = int(config.get('future_window', cls.resolve_future_window(config)))
        win = windows_for_sample(int(t), delay, past, future)
        p = total_p.reshape(-1)
        result = cls.find(p[win['future']], p[int(win['bot_time'])])
        z = float(result[0] if isinstance(result, (tuple, list)) else result)
        return torch.tensor([z], dtype=torch.float32)


    @classmethod
    def cache_intra_fields(cls, config):
        return dict(cls.signature_fields(config))

    @classmethod
    def pack_datapoint(cls, dataset, idx):
        from task_base import pack_datapoint as _pack
        return _pack(cls, dataset, idx)

    @classmethod
    def unpack_datapoint(cls, batch):
        from task_base import unpack_datapoint as _unpack
        return _unpack(cls, batch)

    @classmethod
    def print_prediction_samples(cls, preds, test_labels, *, dataset=None,
                                 yield_target=None, latent_dist=None, n=5):
        return None

    @classmethod
    def report_header_fields(cls, dataset, config=None):
        return {}

    @classmethod
    def postprocess_predictions(cls, preds, dataset):
        from task_base import postprocess_predictions as _post
        return _post(cls, preds, dataset)

    @classmethod
    def visualize_inference(cls, *args, **kwargs):
        viz = getattr(cls, 'visualize', None)
        if viz is None:
            print(f"⚠ {cls.name}: no visualize() for inference plots")
            return None
        return viz(*args, **kwargs)

    @classmethod
    def signature_fields(cls, config: Mapping[str, Any]) -> Dict[str, Any]:
        return {}


    @classmethod
    def format_metric_lines(cls, metrics: Mapping[str, Any]) -> List[str]:
        note = metrics.get('note')
        if note:
            return [f"  ({note})"]
        return ["  (label=none — unsupervised; skipping supervised metrics)"]

    @classmethod
    def metrics_report_note(cls, config: Optional[dict] = None) -> str:
        return "label=none: no supervised metrics."

    @classmethod
    def decision_threshold_from_config(cls, config: Optional[dict] = None) -> Optional[float]:
        return None

    @classmethod
    def summary_metric_headers(cls) -> Tuple[str, ...]:
        return ('note',)

    @classmethod
    def format_summary_metric_cells(cls, metrics: Optional[Mapping[str, Any]]) -> Tuple[str, ...]:
        m = metrics or {}
        return (str(m.get('note', '—')),)

    @classmethod
    def ensemble_compare_ylabel(cls) -> str:
        return "Label loss(en_k, y)"

    @classmethod
    def ensemble_loss_increase_warning(
        cls, *, k: int, loss: float, prev_k: int, prev_loss: float
    ) -> str:
        return (
            f"loss(en_{k}, y)={loss:.6f} > loss(en_{prev_k}, y)={prev_loss:.6f}."
        )

