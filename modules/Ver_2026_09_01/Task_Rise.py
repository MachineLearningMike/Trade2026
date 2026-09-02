# Task_Rise.py — task implementation (canonical; not a re-export)

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from task_base import Task


__all__ = ['Task_Rise']


# ═══════════════════════════════════════════════════════════
# LABEL FAMILY: RISE FLAG (binary; refine find() as needed)
# ═══════════════════════════════════════════════════════════

class Task_Rise(Task):
    """
    Binary classifier: rise flag on raw prices (scaffold for refinement).

    Config id: ``rise_flag``.
    Cache / model target: single scalar in {0.0, 1.0} (class label).
    Model head: one logit (output_dim=1). At inference, P(rise)=sigmoid(logit).
    Price space: raw (same TelePred contract as Yield):
        Y(t) = find( p[t+1 : t+1+delta2], p[t] )

    Initial ``find`` is a crude chance-like threshold stub — refine freely.

    Loss (binary classification):
        BCE-with-logits ≡ BCE(sigmoid(y_pred), y_truth)
        with ``y_pred`` = logits, ``y_truth`` ∈ {0,1}.
        (Fused form is used for numerical stability; same objective as
        explicit sigmoid + BCE.)

    Do not z-score these targets in the dataset (identity label stats).
    """

    name = 'rise_flag'


    @classmethod
    def visualize_inference(cls, *, dataset, preds, test_labels, used_indices, config, **kwargs):
        print(
            "\nrise_flag: skipping window viz (binary logits; metrics printed above).",
            flush=True,
        )
        return None


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
        return 1

    @classmethod
    def target_slice(cls, output_dim: int, **_kwargs) -> slice:
        if output_dim != 1:
            raise ValueError(
                f"{cls.name}: output_dim must be {1}, "
                f"got {output_dim}"
            )
        return slice(0, 1)

    @classmethod
    def target_names(cls, output_dim: int, **_kwargs) -> Tuple[str, ...]:
        if output_dim != 1:
            raise ValueError(
                f"{cls.name}: output_dim must be {1}, "
                f"got {output_dim}"
            )
        return ('rise_flag',)

    @staticmethod
    def _as_1d_float64(x) -> np.ndarray:
        if x is None:
            return np.zeros(0, dtype=np.float64)
        if torch.is_tensor(x):
            arr = x.detach().reshape(-1).to(dtype=torch.float64).cpu().numpy()
            return np.asarray(arr, dtype=np.float64)
        return np.asarray(x, dtype=np.float64).reshape(-1)

    @classmethod
    def find(
        cls,
        future,
        current,
        min_yield_percent: float = 0.26,
        max_hope_future_horizon: Optional[int] = None,
        **_kwargs,
    ):
        """
        Binary rise flag stub (REFINE THIS).

        Current rule (chance-like, hard threshold — replace as needed):
          1.0 if any bar in ``future`` has percent rise vs ``current``
              >= ``min_yield_percent``, else 0.0.

        TelePred signature (same as Yield):
            Y(t) = find( p[t+1 : t+1+delta2], p[t] )
        """
        fut = cls._as_1d_float64(future)
        cur = cls._as_1d_float64(current)
        if cur.size == 0:
            current_px = float('nan')
        else:
            current_px = float(cur[-1])

        n = int(fut.shape[0])
        if max_hope_future_horizon is not None:
            n = min(n, max(0, int(max_hope_future_horizon)))

        flag = 0.0
        if n > 0 and np.isfinite(current_px):
            eps = 1e-8
            denom = current_px if abs(current_px) > eps else (
                current_px + eps if current_px >= 0.0 else current_px - eps
            )
            pct = (fut[:n] - current_px) / denom * 100.0
            thr = float(min_yield_percent)
            # STUB: hard OR over the future window. Refine freely.
            if np.any(np.isfinite(pct) & (pct >= thr)):
                flag = 1.0

        return (float(flag),)

    @classmethod
    def loss(
        cls,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_truth: torch.Tensor,
        config: Optional[dict] = None,
    ) -> torch.Tensor:
        """BCE-with-logits on a single rise logit. Knobs from ``config``."""
        del x
        cfg = config or {}
        w = float(cfg.get('rise_flag_loss_weight', 1.0))
        if y_pred.shape[-1] != 1 or y_truth.shape[-1] != 1:
            raise ValueError(
                f"{cls.name} loss expects pred/truth last dim 1, "
                f"got pred={tuple(y_pred.shape)} truth={tuple(y_truth.shape)}"
            )
        logits = y_pred[:, 0]
        target = y_truth[:, 0].clamp(0.0, 1.0)
        return w * F.binary_cross_entropy_with_logits(logits, target)

    @classmethod
    def metric_slots(cls) -> Tuple[str, ...]:
        """
        Binary-classifier metric slots.

        Confusion counts use decision P(rise)=sigmoid(logit) >= 0.5.
        """
        return (
            'bce_with_logits',
            'accuracy',
            'true_positive', 'true_negative',
            'false_positive', 'false_negative',
            'tpr', 'tnr', 'fpr', 'fnr', 'precision',
            'mean_prob',
            'positive_rate_true', 'positive_rate_pred',
            'decision_threshold',
        )

    @classmethod
    def compute_metrics(
        cls,
        y_pred: torch.Tensor,
        y_truth: torch.Tensor,
        *,
        dataset=None,
        config: Optional[dict] = None,
        decision_threshold: float = 0.5,
        **_kwargs,
    ) -> dict:
        """
        Binary classification metrics including TP/TN/FP/FN.

        Positive class = rise (y=1). Decision: sigmoid(logit) >= threshold
        (default 0.5).
        """
        del dataset
        if y_pred.shape[-1] != 1 or y_truth.shape[-1] != 1:
            raise ValueError(
                f"{cls.name} metrics expect pred/truth last dim 1, "
                f"got pred={tuple(y_pred.shape)} truth={tuple(y_truth.shape)}"
            )
        cfg = config or {}
        thr = float(cfg.get('rise_flag_decision_threshold', decision_threshold))

        logits = y_pred[:, 0]
        target = y_truth[:, 0].clamp(0.0, 1.0)
        prob = torch.sigmoid(logits)
        pred_pos = prob >= thr
        true_pos = target >= 0.5

        tp = int((pred_pos & true_pos).sum().item())
        tn = int((~pred_pos & ~true_pos).sum().item())
        fp = int((pred_pos & ~true_pos).sum().item())
        fn = int((~pred_pos & true_pos).sum().item())
        eps = 1e-12
        tpr = tp / (tp + fn + eps)  # recall / sensitivity
        tnr = tn / (tn + fp + eps)  # specificity
        fpr = fp / (fp + tn + eps)
        fnr = fn / (fn + tp + eps)
        precision = tp / (tp + fp + eps)

        metrics = {
            'bce_with_logits': float(
                F.binary_cross_entropy_with_logits(logits, target).item()
            ),
            'accuracy': float((pred_pos == true_pos).float().mean().item()),
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'tpr': float(tpr),
            'tnr': float(tnr),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'precision': float(precision),
            'mean_prob': float(prob.mean().item()),
            'positive_rate_true': float(true_pos.float().mean().item()),
            'positive_rate_pred': float(pred_pos.float().mean().item()),
            'decision_threshold': thr,
        }
        return metrics

    @classmethod
    def pack_metrics(
        cls,
        y_pred: torch.Tensor,
        y_truth: torch.Tensor,
        *,
        dataset=None,
        config: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        """
        Pack eval metrics into an opaque dict (LabelSpec ownership).

        Models never see this — only INFER/report call sites.
        """
        return cls.compute_metrics(
            y_pred, y_truth,
            dataset=dataset, config=config, **kwargs,
        )

    @classmethod
    def unpack_metrics(cls, metrics: dict) -> dict:
        """Select declared metric slots from a packed metrics dict."""
        slots = cls.metric_slots()
        return {k: metrics[k] for k in slots if k in metrics}

    @classmethod
    def normalize(cls, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Binary {0,1} stays raw for BCE-with-logits."""
        return y

    @classmethod
    def denormalize(cls, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return y

    @classmethod
    def bind_dataset_outputs(cls, config: dict) -> dict:
        forced_dim = cls.resolve_output_dim(config)
        cfg_dim = int(config.get('output_dim', forced_dim))
        if cfg_dim != forced_dim:
            print(
                f"⚠ label={'rise_flag'}: forcing output_dim "
                f"{cfg_dim} → {forced_dim}"
            )
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
        result = cls.find(
            p[win['future']],
            p[int(win['bot_time'])],
            min_yield_percent=float(config.get('min_yield_percent', 0.0)),
            max_hope_future_horizon=future,
        )
        flag = float(result[0] if isinstance(result, (tuple, list)) else result)
        return torch.tensor([flag], dtype=torch.float32)


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
        return {
            'min_yield_percent': config.get('min_yield_percent', 0.0),
        }


    @classmethod
    def plot_metric_slots(cls) -> Tuple[str, ...]:
        """
        Subset of metric_slots drawn on the member-vs-ensemble report plot.

        Rates only (not raw TP/TN/FP/FN counts). Decision threshold from
        config ``rise_flag_decision_threshold`` affects all of these.
        """
        return (
            'accuracy',
            'tpr',
            'tnr',
            'fpr',
            'fnr',
            'precision',
        )

    @classmethod
    @classmethod
    def _plot_member_ensemble_metrics_impl(
            cls,
        member_metrics_by_stage: Sequence[Mapping[str, Any]],
        ensemble_metrics_by_stage: Sequence[Mapping[str, Any]],
        *,
        save_path: Union[str, Path],
        title: Optional[str] = None,
        decision_threshold: Optional[float] = None,
        split_name: str = 'val',
    ) -> Optional[Path]:
        """Shared drawer for Label_*.plot_member_ensemble_metrics."""
        slots = tuple(getattr(cls, 'plot_metric_slots', lambda: ())())
        if not slots:
            return None

        n_m = len(member_metrics_by_stage)
        n_e = len(ensemble_metrics_by_stage)
        n = min(n_m, n_e)
        if n <= 0:
            print("⚠ plot_member_ensemble_metrics: no aligned stages; skip")
            return None
        if n_m != n_e:
            print(
                f"⚠ plot_member_ensemble_metrics: member stages={n_m} "
                f"ensemble stages={n_e}; using common prefix n={n}"
            )

        xs = list(range(n))
        # One color per metric; solid=member, dashed=ensemble.
        try:
            cmap = matplotlib.colormaps['tab10']
        except Exception:
            cmap = plt.cm.get_cmap('tab10')
        fig, ax = plt.subplots(figsize=(11, 6))
        for j, slot in enumerate(slots):
            color = cmap(j % 10)
            mem_ys = []
            en_ys = []
            for i in range(n):
                mv = member_metrics_by_stage[i].get(slot) if member_metrics_by_stage[i] else None
                ev = ensemble_metrics_by_stage[i].get(slot) if ensemble_metrics_by_stage[i] else None
                mem_ys.append(float(mv) if mv is not None else float('nan'))
                en_ys.append(float(ev) if ev is not None else float('nan'))
            ax.plot(
                xs, mem_ys, 'o-', color=color, linewidth=2, markersize=7,
                label=f'member_{slot}',
            )
            ax.plot(
                xs, en_ys, 's--', color=color, linewidth=2, markersize=7,
                label=f'en_{slot}',
            )

        ax.set_xlabel('Stage index i  (member model_i  /  ensemble en_i)')
        ax.set_ylabel('Metric value')
        thr_txt = (
            f', decision_threshold={float(decision_threshold):g}'
            if decision_threshold is not None else ''
        )
        ax.set_title(
            title
            or (
                f'{getattr(cls, "name", cls.__name__)} member vs ensemble metrics '
                f'({split_name}{thr_txt})'
            )
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([f'{i}\nmodel_{i}/en_{i}' for i in xs])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, ncol=2)
        fig.tight_layout()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Label metric plot saved to: {save_path}")
        return save_path

    def plot_member_ensemble_metrics(
        cls,
        member_metrics_by_stage: Sequence[Mapping[str, Any]],
        ensemble_metrics_by_stage: Sequence[Mapping[str, Any]],
        *,
        save_path: Union[str, Path],
        title: Optional[str] = None,
        decision_threshold: Optional[float] = None,
        split_name: str = 'val',
    ) -> Optional[Path]:
        """
        One axes, 2×n series: member_* (solid) and en_* (dashed) per plot slot.

        X = stage index i; member_i and en_i share tick i.
        """
        return cls._plot_member_ensemble_metrics_impl(
            member_metrics_by_stage,
            ensemble_metrics_by_stage,
            save_path=save_path,
            title=title,
            decision_threshold=decision_threshold,
            split_name=split_name,
        )

    @classmethod
    def format_metric_lines(cls, metrics: Mapping[str, Any]) -> List[str]:
        return [
            f"  BCE (logits):    {metrics.get('bce_with_logits', float('nan')):.4f}",
            (
                f"  Accuracy @thr:   {metrics.get('accuracy', float('nan')):.4f}"
                f"  (thr={metrics.get('decision_threshold', 0.5)})"
            ),
            (
                f"  Confusion:       TP={metrics.get('true_positive', 0)}  "
                f"TN={metrics.get('true_negative', 0)}  "
                f"FP={metrics.get('false_positive', 0)}  "
                f"FN={metrics.get('false_negative', 0)}"
            ),
            (
                f"  Rates:           TPR={metrics.get('tpr', float('nan')):.4f}  "
                f"TNR={metrics.get('tnr', float('nan')):.4f}  "
                f"FPR={metrics.get('fpr', float('nan')):.4f}  "
                f"FNR={metrics.get('fnr', float('nan')):.4f}"
            ),
            f"  Precision:       {metrics.get('precision', float('nan')):.4f}",
            f"  Mean P(rise):    {metrics.get('mean_prob', float('nan')):.4f}",
            f"  True pos rate:   {metrics.get('positive_rate_true', float('nan')):.4f}",
            f"  Pred pos rate:   {metrics.get('positive_rate_pred', float('nan')):.4f}",
        ]

    @classmethod
    def metrics_report_note(cls, config: Optional[dict] = None) -> str:
        thr = cls.decision_threshold_from_config(config)
        thr_s = f"{float(thr):g}" if thr is not None else "0.5"
        return (
            "rise_flag: member/en_k tensors scored as logits "
            f"(P=sigmoid(z), thr={thr_s} from rise_flag_decision_threshold). "
            "Additive en_k follows Yield residual combine; classifier residual "
            "semantics are experimental - see _error_ensemble_logic.txt."
        )

    @classmethod
    def decision_threshold_from_config(cls, config: Optional[dict] = None) -> Optional[float]:
        cfg = dict(config or {})
        return float(cfg.get('rise_flag_decision_threshold', 0.5))

    @classmethod
    def summary_metric_headers(cls) -> Tuple[str, ...]:
        return ('TP', 'TN', 'FP', 'FN', 'accuracy')

    @classmethod
    def format_summary_metric_cells(cls, metrics: Optional[Mapping[str, Any]]) -> Tuple[str, ...]:
        m = metrics or {}
        if 'true_positive' not in m:
            return tuple('—' for _ in cls.summary_metric_headers())
        return (
            str(int(m.get('true_positive', 0))),
            str(int(m.get('true_negative', 0))),
            str(int(m.get('false_positive', 0))),
            str(int(m.get('false_negative', 0))),
            f"{float(m.get('accuracy', float('nan'))):.4f}",
        )

    @classmethod
    def ensemble_compare_ylabel(cls) -> str:
        return "Label loss(en_k, y)  [rise_flag: BCE-with-logits]"

    @classmethod
    def ensemble_loss_increase_warning(
        cls, *, k: int, loss: float, prev_k: int, prev_loss: float
    ) -> str:
        return (
            f"loss(en_{k}, y)={loss:.6f} > loss(en_{prev_k}, y)={prev_loss:.6f}. "
            f"For rise_flag, additive en_k mixes BCE-logits with a Yield-style "
            f"residual chain (_error_ensemble_logic.txt is MSE/y-space oriented) "
            f"— rising ensemble BCE is expected until a classifier residual "
            f"design exists."
        )

    @classmethod
    def ensemble_logic_markdown_notes(cls) -> List[str]:
        """Extra summary.md bullets for this label (may be empty)."""
        return [
            "> **rise_flag note:** `_error_ensemble_logic.txt` is Yield/MSE-oriented.",
            "> Heads emit logits; targets are `{0,1}`. Growing `loss(en_k)` while",
            "> member val loss falls is expected under the current residual recipe.",
            "",
        ]

    @classmethod
    def print_prediction_samples(
        cls, preds, test_labels, *, dataset=None, yield_target=None, latent_dist=None, n: int = 5,
    ) -> None:
        import torch
        n = min(int(n), len(preds))
        print(
            f"  {'Index':<6} {'True':<8} {'Logit':<12} {'P(rise)':<12} {'Pred@0.5':<10}"
        )
        print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*12} {'-'*10}")
        for i in range(n):
            logit = preds[i, 0].item()
            prob = float(torch.sigmoid(preds[i, 0]).item())
            pred_bin = 1 if prob >= 0.5 else 0
            print(
                f"  {i:<6} {test_labels[i, 0].item():<8.0f} "
                f"{logit:<+12.4f} {prob:<12.4f} {pred_bin:<10d}"
            )

