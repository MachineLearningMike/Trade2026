# error_ensemble.py
# Residual / error-boosting ensemble: model_i fits the leftover error of model_0..i-1.
#
# Each stage trains in its own normalized target space (saved mean/std):
#   stage 0: target = normalize_0(y_raw)
#   stage i: e_raw = y_raw - sum_{j<i} denormalize_j(y_j)
#            target = normalize_i(e_raw)
#
# Nested ensembles are formed in **raw / denormalized** label space:
#   en_k = sum_{i=0..k} denormalize_i(model_i(x))
#
# Training is sequential: freeze earlier stages, fit the next on the residual.

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from labels import bind_label_loss, get_label_class


ModelLike = Union[nn.Module, Callable[[], nn.Module]]
LossFn = Callable[[torch.Tensor, Any, torch.Tensor], torch.Tensor]

# Sidecar next to each stage checkpoint key (mean/std of that stage's target).
STAGE_NORM_FILENAME = 'error_stage_norm.json'


def head_prediction(output: Any) -> torch.Tensor:
    """Extract the prediction head tensor from a model forward output."""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def apply_normalize(
    y: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """(y - mean) / (std + eps) with broadcast over batch."""
    mean = mean.to(device=y.device, dtype=y.dtype)
    std = std.to(device=y.device, dtype=y.dtype)
    return (y - mean) / (std + 1e-8)


def apply_denormalize(
    y: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """y * std + mean with broadcast over batch."""
    mean = mean.to(device=y.device, dtype=y.dtype)
    std = std.to(device=y.device, dtype=y.dtype)
    return y * std + mean


def label_norm_from_dataset(dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dataset label affine used by ``TradingDataset.normalize_y`` (stage 0)."""
    mean = torch.as_tensor(dataset._label_stat_mean, dtype=torch.float32).reshape(-1)
    std = torch.as_tensor(dataset._label_stat_std, dtype=torch.float32).reshape(-1)
    std = torch.where(std.abs() < 1e-8, torch.ones_like(std), std)
    return mean.detach().cpu(), std.detach().cpu()


def save_stage_norm(
    checkpoint_dir: Path,
    *,
    stage: int,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    label_mean: torch.Tensor,
    label_std: torch.Tensor,
) -> Path:
    """Persist per-stage target affine next to the stage checkpoint key."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / STAGE_NORM_FILENAME
    payload = {
        'stage': int(stage),
        'target_mean': torch.as_tensor(target_mean).detach().cpu().tolist(),
        'target_std': torch.as_tensor(target_std).detach().cpu().tolist(),
        'label_mean': torch.as_tensor(label_mean).detach().cpu().tolist(),
        'label_std': torch.as_tensor(label_std).detach().cpu().tolist(),
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return path


def load_stage_norm(
    checkpoint_dir: Path,
    *,
    fallback_mean: Optional[torch.Tensor] = None,
    fallback_std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load ``(target_mean, target_std)`` for a stage.

    Falls back to label stats when the sidecar is missing (legacy runs).
    """
    path = Path(checkpoint_dir) / STAGE_NORM_FILENAME
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        mean = torch.tensor(data['target_mean'], dtype=torch.float32)
        std = torch.tensor(data['target_std'], dtype=torch.float32)
        return mean, std
    if fallback_mean is None or fallback_std is None:
        raise FileNotFoundError(
            f"Missing {STAGE_NORM_FILENAME} in {checkpoint_dir} "
            f"and no fallback label mean/std provided"
        )
    print(
        f"⚠ {path.name} missing in {checkpoint_dir.name}; "
        f"falling back to dataset label mean/std"
    )
    return (
        torch.as_tensor(fallback_mean, dtype=torch.float32).reshape(-1),
        torch.as_tensor(fallback_std, dtype=torch.float32).reshape(-1),
    )


def compute_residuals(
    y: torch.Tensor,
    stage_preds_raw: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Residuals in the same space as ``y`` / ``stage_preds_raw``:

      e_0 = y - y_0;  e_i = e_{i-1} - y_i.
    """
    if not stage_preds_raw:
        return []
    residuals: List[torch.Tensor] = []
    err = y
    for pred in stage_preds_raw:
        err = err - pred
        residuals.append(err)
    return residuals


def sum_denormalized_stages(
    stage_preds_norm: Sequence[torch.Tensor],
    stage_norms: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """
    en_k in raw label space: sum_i denormalize_i(pred_i).

    ``stage_norms[i] = (mean_i, std_i)`` saved for that stage's target space.
    """
    if len(stage_preds_norm) != len(stage_norms):
        raise ValueError(
            f"pred/norm length mismatch: {len(stage_preds_norm)} vs {len(stage_norms)}"
        )
    total = None
    for pred, (mean, std) in zip(stage_preds_norm, stage_norms):
        raw = apply_denormalize(pred, mean, std)
        total = raw if total is None else total + raw
    if total is None:
        raise ValueError("sum_denormalized_stages requires at least one stage")
    return total


def clone_model(model: nn.Module) -> nn.Module:
    """Deep-copy a module (new parameters; same architecture)."""
    return copy.deepcopy(model)


def expand_models(model: ModelLike, n: int) -> List[nn.Module]:
    """Build ``n`` independent model instances from a module or zero-arg factory."""
    if n < 1:
        raise ValueError(f"error ensemble needs n >= 1, got {n}")
    models: List[nn.Module] = []
    if isinstance(model, nn.Module):
        models.append(model)
        for _ in range(n - 1):
            models.append(clone_model(model))
        return models
    if callable(model):
        for _ in range(n):
            m = model()
            if not isinstance(m, nn.Module):
                raise TypeError(
                    f"model factory must return nn.Module, got {type(m).__name__}"
                )
            models.append(m)
        return models
    raise TypeError(
        f"model must be nn.Module or Callable[[], nn.Module], "
        f"got {type(model).__name__}"
    )


class ResidualStageWrapper(nn.Module):
    """
    Train one error stage while earlier models are frozen.

    ``forward`` returns the *active* stage output (same tuple layout as the
    base market model). Previous stages are denormalized with their saved
    mean/std and summed in raw space into ``last_prev_sum_raw``.
    """

    def __init__(
        self,
        active: nn.Module,
        previous: Optional[Sequence[nn.Module]] = None,
        *,
        prev_norms: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        active_norm: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        label_norm: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        super().__init__()
        self.active = active
        prev_list = list(previous or [])
        self.prev = nn.ModuleList(prev_list)
        for p in self.prev.parameters():
            p.requires_grad = False

        if prev_norms is None:
            prev_norms = []
        if len(prev_norms) != len(prev_list):
            raise ValueError(
                f"prev_norms length {len(prev_norms)} != previous models "
                f"{len(prev_list)}"
            )
        # Register affines as buffers so .to(device) moves them.
        for i, (mean, std) in enumerate(prev_norms):
            self.register_buffer(f'prev_mean_{i}', mean.detach().float().reshape(-1))
            self.register_buffer(f'prev_std_{i}', std.detach().float().reshape(-1))
        self._n_prev_norms = len(prev_norms)

        if active_norm is None:
            raise ValueError("active_norm=(mean, std) is required")
        a_mean, a_std = active_norm
        self.register_buffer('active_mean', a_mean.detach().float().reshape(-1))
        self.register_buffer('active_std', a_std.detach().float().reshape(-1))

        if label_norm is None:
            # Stage-0 / dataloader y affine defaults to active_norm.
            label_norm = active_norm
        l_mean, l_std = label_norm
        self.register_buffer('label_mean', l_mean.detach().float().reshape(-1))
        self.register_buffer('label_std', l_std.detach().float().reshape(-1))

        self.last_prev_sum_raw: Optional[torch.Tensor] = None
        # Mirror common attrs used by diagnostics / INFER helpers.
        for attr in (
            'output_dim',
            'n_max_past_horizon',
            'patch_size',
            'embed_dim',
            'max_seq_len',
            'look_price',
            'use_reconstruction',
        ):
            if hasattr(active, attr):
                setattr(self, attr, getattr(active, attr))

    @property
    def stage_index(self) -> int:
        return len(self.prev)

    def _prev_norm_pair(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            getattr(self, f'prev_mean_{i}'),
            getattr(self, f'prev_std_{i}'),
        )

    def forward(self, x, p, mean, std, shape):
        if len(self.prev) == 0:
            self.last_prev_sum_raw = None
        else:
            with torch.no_grad():
                total_raw = None
                for i, m in enumerate(self.prev):
                    pred_norm = head_prediction(m(x, p, mean, std, shape))
                    p_mean, p_std = self._prev_norm_pair(i)
                    pred_raw = apply_denormalize(pred_norm, p_mean, p_std)
                    total_raw = (
                        pred_raw if total_raw is None else total_raw + pred_raw
                    )
                self.last_prev_sum_raw = total_raw
        return self.active(x, p, mean, std, shape)

    # Persist / resume only the active stage weights (prev stages have their
    # own checkpoint keys). Keep Module.parameters() intact so .to(device)
    # still moves frozen prev stages.
    def state_dict(self, *args, **kwargs):
        return self.active.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, **kwargs):
        return self.active.load_state_dict(state_dict, **kwargs)


def bind_residual_stage_loss(
    base_loss_fn: LossFn,
    stage_model: ResidualStageWrapper,
) -> LossFn:
    """
    Stage loss against a **re-normalized** residual target:

      y_raw = denormalize_label(y_dataloader)
      e_raw = y_raw - sum_{j<i} denormalize_j(y_j)   (0 if i==0)
      target = normalize_i(e_raw)   # active stage mean/std
    """

    def loss_fn(x, y_pred, y_truth):
        # y_truth from the dataloader is label-normalized (stage-0 affine).
        y_raw = apply_denormalize(
            y_truth, stage_model.label_mean, stage_model.label_std
        )
        prev_raw = stage_model.last_prev_sum_raw
        if prev_raw is None:
            e_raw = y_raw
        else:
            e_raw = y_raw - prev_raw
        target = apply_normalize(
            e_raw, stage_model.active_mean, stage_model.active_std
        )
        return base_loss_fn(x, y_pred, target)

    loss_fn.label_cls = getattr(base_loss_fn, 'label_cls', None)  # type: ignore[attr-defined]
    loss_fn.auxiliary = getattr(base_loss_fn, 'auxiliary', None)  # type: ignore[attr-defined]
    loss_fn.base_loss_fn = base_loss_fn  # type: ignore[attr-defined]
    loss_fn.stage_model = stage_model  # type: ignore[attr-defined]
    return loss_fn


class ErrorEnsemble(nn.Module):
    """
    Ensemble of residual error models ``{model_0, ..., model_{n-1}}``.

    Nested ensembles are formed in raw label space:

      en_k = sum_{i=0..k} denormalize_i(model_i(x))

    using each stage's saved target mean/std.
    """

    def __init__(
        self,
        model: Optional[ModelLike] = None,
        label: Any = None,
        loss: Optional[LossFn] = None,
        *,
        n: Optional[int] = None,
        models: Optional[Sequence[nn.Module]] = None,
        stage_norms: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        label_norm: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if models is not None:
            stage_models = list(models)
            if not stage_models:
                raise ValueError("ErrorEnsemble: models must be non-empty")
        elif model is not None:
            n_models = int(n if n is not None else 1)
            stage_models = expand_models(model, n_models)
        else:
            raise ValueError("ErrorEnsemble requires `model` or `models`")

        self.models = nn.ModuleList(stage_models)
        if stage_norms is None:
            raise ValueError(
                "ErrorEnsemble requires stage_norms=[(mean_i, std_i), ...] "
                "to denormalize each stage before summing"
            )
        if len(stage_norms) != len(stage_models):
            raise ValueError(
                f"stage_norms length {len(stage_norms)} != models "
                f"{len(stage_models)}"
            )
        for i, (mean, std) in enumerate(stage_norms):
            self.register_buffer(
                f'stage_mean_{i}', mean.detach().float().reshape(-1)
            )
            self.register_buffer(
                f'stage_std_{i}', std.detach().float().reshape(-1)
            )
        if label_norm is None:
            label_norm = stage_norms[0]
        self.register_buffer(
            'label_mean', label_norm[0].detach().float().reshape(-1)
        )
        self.register_buffer(
            'label_std', label_norm[1].detach().float().reshape(-1)
        )

        self.label = label
        self.label_cls = None
        if label is not None:
            if isinstance(label, str):
                self.label_cls = get_label_class(label)
            elif isinstance(label, type):
                self.label_cls = label
            else:
                self.label_cls = label

        if loss is not None:
            self.loss_fn = loss
        elif self.label_cls is not None:
            mode = getattr(self.label_cls, 'name', None) or str(label)
            self.loss_fn = bind_label_loss(mode, config)
        else:
            self.loss_fn = None

        self.last_stage_preds_norm: List[torch.Tensor] = []
        self.last_stage_preds_raw: List[torch.Tensor] = []
        # Back-compat alias used by older helpers.
        self.last_stage_preds: List[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.models)

    @property
    def n(self) -> int:
        return len(self.models)

    def stage_norm_pair(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            getattr(self, f'stage_mean_{i}'),
            getattr(self, f'stage_std_{i}'),
        )

    def stage_predictions(
        self,
        x,
        p,
        mean,
        std,
        shape,
    ) -> List[torch.Tensor]:
        """Normalized head outputs (each stage's own target space)."""
        preds: List[torch.Tensor] = []
        for m in self.models:
            preds.append(head_prediction(m(x, p, mean, std, shape)))
        self.last_stage_preds_norm = preds
        self.last_stage_preds = preds
        self.last_stage_preds_raw = [
            apply_denormalize(pred, *self.stage_norm_pair(i))
            for i, pred in enumerate(preds)
        ]
        return preds

    def forward(self, x, p, mean, std, shape):
        """
        Return ``(en_m, patch_map, recon)`` in **raw** label space:

          en_m = sum_i denormalize_i(model_i(x))

        ``patch_map`` / ``recon`` come from the last stage (API compatibility).
        """
        last_full = None
        preds_norm: List[torch.Tensor] = []
        preds_raw: List[torch.Tensor] = []
        total_raw = None
        for i, m in enumerate(self.models):
            out = m(x, p, mean, std, shape)
            last_full = out
            pred_norm = head_prediction(out)
            pred_raw = apply_denormalize(pred_norm, *self.stage_norm_pair(i))
            preds_norm.append(pred_norm)
            preds_raw.append(pred_raw)
            total_raw = pred_raw if total_raw is None else total_raw + pred_raw
        self.last_stage_preds_norm = preds_norm
        self.last_stage_preds_raw = preds_raw
        self.last_stage_preds = preds_norm
        if isinstance(last_full, (tuple, list)):
            patch_map = last_full[1] if len(last_full) > 1 else None
            recon = last_full[2] if len(last_full) > 2 else None
            return total_raw, patch_map, recon
        return total_raw, None, None

    def residuals(self, y_raw: torch.Tensor) -> List[torch.Tensor]:
        """Raw-space residuals from the most recent forward."""
        return compute_residuals(y_raw, self.last_stage_preds_raw)

    def ensemble_prediction(self, k: Optional[int] = None) -> torch.Tensor:
        """
        ``en_k`` from cached raw stage preds. ``k=None`` → full ensemble.
        """
        if not self.last_stage_preds_raw:
            raise RuntimeError("No stage predictions; call forward() first")
        if k is None:
            k = len(self.last_stage_preds_raw) - 1
        if k < 0 or k >= len(self.last_stage_preds_raw):
            raise IndexError(f"en_k index {k} out of range")
        total = self.last_stage_preds_raw[0]
        for pred in self.last_stage_preds_raw[1 : k + 1]:
            total = total + pred
        return total

    def ensemble_loss(self, x: torch.Tensor, y_raw: torch.Tensor) -> torch.Tensor:
        """Loss of full ``en_m`` against raw ``y``."""
        if self.loss_fn is None:
            raise RuntimeError("ErrorEnsemble has no loss_fn; pass loss=... at init")
        return self.loss_fn(x, self.ensemble_prediction(), y_raw)


def stage_checkpoint_key(config: dict, dataset_signature: str, stage: int) -> str:
    """Checkpoint key for error-ensemble stage ``stage`` (0-based)."""
    from trainer import generate_checkpoint_key

    cfg = dict(config)
    cfg['error_ensemble_stage'] = int(stage)
    return generate_checkpoint_key(cfg, dataset_signature)


def error_ensemble_stage_keys(
    config: dict,
    *,
    n: Optional[int] = None,
    dataset_signature: Optional[str] = None,
) -> List[str]:
    """Checkpoint keys for ``model_0 .. model_{n-1}`` under the current config."""
    from data import generate_dataset_signature
    from labels import sync_output_dim

    cfg = dict(config)
    sync_output_dim(cfg)
    n_models = int(n if n is not None else cfg.get('error_ensemble_n', 3) or 3)
    if n_models < 1:
        raise ValueError(f"error_ensemble_n must be >= 1, got {n_models}")
    sig = (
        dataset_signature
        if dataset_signature is not None
        else generate_dataset_signature(cfg)
    )
    return [stage_checkpoint_key(cfg, sig, stage) for stage in range(n_models)]


def clean_error_ensemble_checkpoints(
    config: dict,
    *,
    n: Optional[int] = None,
) -> List[str]:
    """
    Delete all ErrorEnsemble stage checkpoint folders for this config.

    Returns the list of removed checkpoint keys.
    """
    import shutil
    from pathlib import Path

    from config import project_root_dir

    cfg = dict(config)
    n_models = int(n if n is not None else cfg.get('error_ensemble_n', 3) or 3)
    keys = error_ensemble_stage_keys(cfg, n=n_models)
    ckpt_root = Path(cfg.get('checkpoint_dir', 'trading_checkpoints'))
    if not ckpt_root.is_absolute():
        ckpt_root = project_root_dir() / ckpt_root

    removed: List[str] = []
    print("\n" + "=" * 60)
    print("ERROR ENSEMBLE CLEAN")
    print("=" * 60)
    print(f"  stages: {n_models}")
    print(f"  checkpoint_dir: {ckpt_root}")
    for stage, key in enumerate(keys):
        key_dir = ckpt_root / key
        if key_dir.exists():
            shutil.rmtree(key_dir)
            removed.append(key)
            print(f"  ✓ removed model_{stage} key={key}")
            print(f"      {key_dir}")
        else:
            print(f"  · no folder for model_{stage} key={key}")
    print("=" * 60 + "\n")
    return removed


def _load_stage_weights(
    model: nn.Module,
    config: dict,
    dataset_signature: str,
    stage: int,
    device: torch.device,
) -> nn.Module:
    """Load best/latest weights for a finished stage into ``model``."""
    from INFER import find_checkpoint_by_key

    key = stage_checkpoint_key(config, dataset_signature, stage)
    found = find_checkpoint_by_key(config['checkpoint_dir'], key=key)
    if found is None:
        raise FileNotFoundError(
            f"ErrorEnsemble stage {stage}: no checkpoint for key {key}"
        )
    path, _ = found
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def evaluate_ensemble_over_stages(
    models: Sequence[nn.Module],
    loss_fn: LossFn,
    loader,
    *,
    stage_norms: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    label_norm: Tuple[torch.Tensor, torch.Tensor],
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Compare nested ensembles in **raw / denormalized** label space.

    Given ``model_0 .. model_m``:

      en_k = sum_{i=0..k} denormalize_i(model_i(x))

    ``loader`` yields ``(x, p, mean, std, shape, y_norm)`` where ``y_norm``
    uses the dataset label affine ``label_norm``. Metrics compare ``en_k`` to
    ``y_raw = denormalize_label(y_norm)``.
    """
    stage_models = list(models)
    if not stage_models:
        return []
    if len(stage_norms) != len(stage_models):
        raise ValueError(
            f"stage_norms length {len(stage_norms)} != models {len(stage_models)}"
        )

    if device is None:
        device = next(stage_models[0].parameters()).device

    for m in stage_models:
        m.eval()
        m.to(device)

    label_mean, label_std = label_norm
    n = len(stage_models)  # models 0..n-1 → ensembles en_0..en_{n-1}
    sum_loss = [0.0] * n
    sum_mse = [0.0] * n
    sum_l1 = [0.0] * n
    n_examples = 0
    n_batches = 0

    def _iter_batches():
        # ThreadedDataLoader: start_epoch + get_next_batch (no __iter__).
        if hasattr(loader, 'start_epoch') and hasattr(loader, 'get_next_batch'):
            loader.start_epoch(0)
            while True:
                batch, _wait, _load = loader.get_next_batch()
                if batch is None:
                    break
                yield batch
            return
        # Plain iterable of batches.
        for batch in loader:
            yield batch

    for batch in _iter_batches():
        if max_batches is not None and n_batches >= int(max_batches):
            break
        if not isinstance(batch, (tuple, list)) or len(batch) < 6:
            raise ValueError(
                "evaluate_ensemble_over_stages expects batches "
                "(x, p, mean, std, shape, y)"
            )
        x, p, mean, std, shape, y_norm = batch[:6]
        x = x.to(device)
        p = p.to(device)
        mean = mean.to(device)
        std = std.to(device)
        shape = shape.to(device)
        y_norm = y_norm.to(device)
        y_raw = apply_denormalize(y_norm, label_mean, label_std)

        # Denormalize each stage, then form en_k = sum_{i<=k} raw_i
        preds_raw: List[torch.Tensor] = []
        for i, m in enumerate(stage_models):
            pred_norm = head_prediction(m(x, p, mean, std, shape))
            s_mean, s_std = stage_norms[i]
            preds_raw.append(apply_denormalize(pred_norm, s_mean, s_std))

        total_raw = None
        bsz = int(y_raw.shape[0])
        for k, pred_raw in enumerate(preds_raw):
            total_raw = pred_raw if total_raw is None else total_raw + pred_raw
            loss_k = loss_fn(x, total_raw, y_raw)
            resid = y_raw - total_raw
            sum_loss[k] += float(loss_k.item()) * bsz
            sum_mse[k] += float(resid.pow(2).mean().item()) * bsz
            sum_l1[k] += float(resid.abs().mean().item()) * bsz

        n_examples += bsz
        n_batches += 1

    if n_examples <= 0:
        return []

    out: List[Dict[str, Any]] = []
    for k in range(n):
        members = [f'model_{i}' for i in range(k + 1)]
        out.append(
            {
                'k': k,
                'ensemble': f'en_{k}',
                'definition': ' + '.join(
                    f'denorm_{i}({m})' for i, m in enumerate(members)
                ),
                'members': members,
                'n_models': k + 1,
                'n_stages': k + 1,
                'last_model': f'model_{k}',
                'space': 'denormalized',
                'loss': sum_loss[k] / n_examples,
                'residual_mse': sum_mse[k] / n_examples,
                'residual_l1': sum_l1[k] / n_examples,
                'n_examples': n_examples,
                'n_batches': n_batches,
            }
        )
    return out


@torch.no_grad()
def compute_stage_target_stats(
    *,
    stage: int,
    dataset,
    prev_models: Sequence[nn.Module],
    prev_norms: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    label_norm: Tuple[torch.Tensor, torch.Tensor],
    config: dict,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mean/std of the raw residual target for ``stage`` over the train split.

    Stage 0 → dataset label mean/std.
    Stage i>0 → stats of ``y_raw - sum_{j<i} denorm_j(model_j(x))``.
    """
    if stage == 0:
        return label_norm_from_dataset(dataset)

    if len(prev_models) != stage or len(prev_norms) != stage:
        raise ValueError(
            f"stage {stage} needs {stage} prev models/norms; "
            f"got {len(prev_models)}/{len(prev_norms)}"
        )

    from data import ThreadedDataLoader

    for m in prev_models:
        m.eval()
        m.to(device)

    label_mean, label_std = label_norm
    bs = int(config.get('batch_size', 32))
    loader = ThreadedDataLoader(
        dataset.train_dataset, batch_size=bs, shuffle=False
    )
    loader.start_epoch(0)

    # Welford-friendly accumulators in float64.
    out_dim = int(label_mean.numel())
    total_sum = torch.zeros(out_dim, dtype=torch.float64)
    total_sq = torch.zeros(out_dim, dtype=torch.float64)
    n_examples = 0
    n_batches = 0

    while True:
        if max_batches is not None and n_batches >= int(max_batches):
            break
        batch, _w, _l = loader.get_next_batch()
        if batch is None:
            break
        x, p, mean, std, shape, y_norm = batch[:6]
        x = x.to(device)
        p = p.to(device)
        mean = mean.to(device)
        std = std.to(device)
        shape = shape.to(device)
        y_norm = y_norm.to(device)
        y_raw = apply_denormalize(y_norm, label_mean, label_std)

        prev_raw = None
        for i, m in enumerate(prev_models):
            pred_norm = head_prediction(m(x, p, mean, std, shape))
            p_mean, p_std = prev_norms[i]
            pred_raw = apply_denormalize(pred_norm, p_mean, p_std)
            prev_raw = pred_raw if prev_raw is None else prev_raw + pred_raw
        e_raw = y_raw - prev_raw  # type: ignore[operator]

        e_cpu = e_raw.detach().to(dtype=torch.float64, device='cpu')
        # Sum over batch
        total_sum += e_cpu.sum(dim=0)
        total_sq += e_cpu.pow(2).sum(dim=0)
        n_examples += int(e_cpu.shape[0])
        n_batches += 1

    if n_examples <= 0:
        raise RuntimeError(f"compute_stage_target_stats stage={stage}: no samples")

    mean = (total_sum / n_examples).to(dtype=torch.float32)
    var = (total_sq / n_examples) - mean.double().pow(2)
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var).to(dtype=torch.float32)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    print(
        f"  stage {stage} residual target stats "
        f"(n={n_examples}): mean={mean.tolist()} std={std.tolist()}"
    )
    return mean, std


def load_error_ensemble_stages(
    config: dict,
    dataset,
    *,
    n: Optional[int] = None,
    stage_keys: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    require_finished: bool = False,
) -> Tuple[
    List[nn.Module],
    List[str],
    List[int],
    List[Tuple[torch.Tensor, torch.Tensor]],
    Tuple[torch.Tensor, torch.Tensor],
]:
    """
    Load consecutive ErrorEnsemble stage weights starting from model_0.

    Stops at the first missing checkpoint (cumulative prefix only).
    Returns
    ``(models, keys_loaded, stage_indices, stage_norms, label_norm)``.
    """
    from config import project_root_dir
    from labels import resolve_model_past_len, sync_output_dim
    from model_vae import build_market_model, resolve_max_seq_len
    from telepred import resolve_past_window
    from INFER import find_checkpoint_by_key

    cfg = dict(config)
    sync_output_dim(cfg)
    n_models = int(n if n is not None else cfg.get('error_ensemble_n', 3) or 3)
    if stage_keys is None:
        keys = error_ensemble_stage_keys(cfg, n=n_models)
    else:
        keys = list(stage_keys)

    if device is None:
        device = torch.device(cfg['device'] if cfg.get('device') else 'cpu')
    else:
        device = torch.device(device)

    past_window = int(resolve_past_window(cfg))
    F = int(getattr(dataset, 'F', dataset.n_market_features + dataset.n_macro_features))
    n_features_after_latent = dataset.n_market_features + cfg['n_latent_macros']
    cfg['embed_dim'] = n_features_after_latent * cfg['patch_size']
    cfg['max_seq_len'] = resolve_max_seq_len(cfg, past_window)
    cfg['output_dim'] = int(getattr(dataset, 'output_dim', cfg.get('output_dim', 2)))
    model_past_len = int(
        getattr(dataset, 'model_past_len', resolve_model_past_len(cfg))
    )

    ckpt_root = Path(cfg.get('checkpoint_dir', 'trading_checkpoints'))
    if not ckpt_root.is_absolute():
        ckpt_root = project_root_dir() / ckpt_root

    label_norm = label_norm_from_dataset(dataset)
    models: List[nn.Module] = []
    keys_loaded: List[str] = []
    stages_loaded: List[int] = []
    stage_norms: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for stage, key in enumerate(keys[:n_models]):
        key_dir = ckpt_root / key
        if not key_dir.exists():
            break
        if require_finished:
            state_file = key_dir / 'training_state.json'
            finished = False
            if state_file.exists():
                try:
                    with open(state_file, 'r', encoding='utf-8') as f:
                        finished = bool(json.load(f).get('training_finished'))
                except Exception:
                    finished = False
            if not finished:
                break

        found = find_checkpoint_by_key(str(ckpt_root), key=key)
        if found is None:
            break
        path, _ = found
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        max_seq_len = ckpt.get('max_seq_len', cfg['max_seq_len'])
        model = build_market_model(
            cfg,
            in_features=F,
            n_market_features=dataset.n_market_features,
            n_macro_features=dataset.n_macro_features,
            n_max_past_horizon=model_past_len,
            max_seq_len=int(max_seq_len),
        )
        state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        s_mean, s_std = load_stage_norm(
            key_dir,
            fallback_mean=label_norm[0],
            fallback_std=label_norm[1],
        )
        models.append(model)
        keys_loaded.append(key)
        stages_loaded.append(stage)
        stage_norms.append((s_mean, s_std))

    return models, keys_loaded, stages_loaded, stage_norms, label_norm


def run_error_ensemble_train(
    config: dict,
    dataset,
    total_features,
    *,
    n: Optional[int] = None,
    num_epochs: Optional[float] = None,
    resume: Optional[bool] = None,
    extend: bool = False,
    extend_epochs: float = 0.0,
    probe_weights: bool = True,
    continue_on_error: bool = False,
) -> Dict[str, Any]:
    """
    Sequentially train ``model_0 .. model_{n-1}`` on residuals.

    Stage ``i`` uses checkpoint identity ``error_ensemble_stage=i``.
    Earlier stages are loaded frozen to form the residual target.
    """
    from model_vae import build_market_model, resolve_max_seq_len, wrap_model_loss
    from train_run import train_one_run
    from labels import resolve_model_past_len
    from telepred import resolve_past_window

    cfg_base = dict(config)
    n_models = int(n if n is not None else cfg_base.get('error_ensemble_n', 3))
    if n_models < 1:
        raise ValueError(f"error_ensemble_n must be >= 1, got {n_models}")

    label_mode = str(
        getattr(dataset, 'label_mode', cfg_base.get('label', 'yield'))
    ).lower()
    model_type = str(cfg_base.get('model_type', 'regression')).lower()
    if model_type == 'vae':
        raise ValueError(
            "ErrorEnsemble requires model_type='regression' "
            "(VAE has no label residual to fit)."
        )
    if label_mode == 'latent':
        raise ValueError(
            "ErrorEnsemble does not support label='latent' "
            "(density params are not residual-additive point targets)."
        )

    cfg_base['error_ensemble_n'] = n_models
    device = torch.device(
        cfg_base['device'] if cfg_base.get('device') else 'cpu'
    )

    print("\n" + "=" * 60)
    print("ERROR ENSEMBLE TRAINING (residual boosting)")
    print("=" * 60)
    print(f"  stages (n):     {n_models}  → model_0 .. model_{n_models - 1}")
    print(f"  label:          {label_mode}")
    print(f"  model_type:     {model_type}")
    print(f"  telepred_delay: {int(cfg_base.get('telepred_delay', 0) or 0)}")
    print(f"  device:         {device}")
    print("=" * 60 + "\n")

    results: List[Dict[str, Any]] = []
    for stage in range(n_models):
        print("\n" + "-" * 60)
        print(f"ERROR ENSEMBLE STAGE {stage + 1}/{n_models}: model_{stage}")
        print("-" * 60)

        stage_cfg = dict(cfg_base)
        stage_cfg['error_ensemble_stage'] = int(stage)

        try:
            # Fresh active model for this stage.
            past_window = int(resolve_past_window(stage_cfg))
            F = (
                total_features.shape[0]
                if hasattr(total_features, 'shape')
                else len(total_features)
            )
            n_features_after_latent = (
                dataset.n_market_features + stage_cfg['n_latent_macros']
            )
            embed_dim = n_features_after_latent * stage_cfg['patch_size']
            if embed_dim % stage_cfg['num_heads'] != 0:
                raise ValueError(
                    f"embed_dim ({embed_dim}) must be divisible by num_heads "
                    f"({stage_cfg['num_heads']})."
                )
            stage_cfg['embed_dim'] = embed_dim
            stage_cfg['max_seq_len'] = resolve_max_seq_len(stage_cfg, past_window)
            stage_cfg['output_dim'] = dataset.output_dim
            model_past_len = int(
                getattr(dataset, 'model_past_len', resolve_model_past_len(stage_cfg))
            )

            active = build_market_model(
                stage_cfg,
                in_features=F,
                n_market_features=dataset.n_market_features,
                n_macro_features=dataset.n_macro_features,
                n_max_past_horizon=model_past_len,
                max_seq_len=stage_cfg['max_seq_len'],
            )

            from config import project_root_dir

            label_norm = label_norm_from_dataset(dataset)
            ckpt_root = Path(stage_cfg.get('checkpoint_dir', 'trading_checkpoints'))
            if not ckpt_root.is_absolute():
                ckpt_root = project_root_dir() / ckpt_root
            stage_key = stage_checkpoint_key(
                stage_cfg, dataset.cache_key, stage
            )
            stage_dir = ckpt_root / stage_key
            stage_dir.mkdir(parents=True, exist_ok=True)

            prev_models: List[nn.Module] = []
            prev_norms: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for prev_stage in range(stage):
                prev = build_market_model(
                    stage_cfg,
                    in_features=F,
                    n_market_features=dataset.n_market_features,
                    n_macro_features=dataset.n_macro_features,
                    n_max_past_horizon=model_past_len,
                    max_seq_len=stage_cfg['max_seq_len'],
                )
                _load_stage_weights(
                    prev,
                    stage_cfg,
                    dataset.cache_key,
                    prev_stage,
                    device,
                )
                prev_models.append(prev)
                prev_key = stage_checkpoint_key(
                    stage_cfg, dataset.cache_key, prev_stage
                )
                prev_norms.append(
                    load_stage_norm(
                        ckpt_root / prev_key,
                        fallback_mean=label_norm[0],
                        fallback_std=label_norm[1],
                    )
                )

            # Per-stage residual target affine (saved for denorm-at-sum).
            # Reuse sidecar if present so resume/skip keeps training-time stats.
            norm_path = stage_dir / STAGE_NORM_FILENAME
            if norm_path.exists():
                active_norm = load_stage_norm(stage_dir)
                print(
                    f"  loaded stage norm from {norm_path.name}: "
                    f"mean={active_norm[0].tolist()} std={active_norm[1].tolist()}"
                )
            else:
                active_norm = compute_stage_target_stats(
                    stage=stage,
                    dataset=dataset,
                    prev_models=prev_models,
                    prev_norms=prev_norms,
                    label_norm=label_norm,
                    config=stage_cfg,
                    device=device,
                )
                save_stage_norm(
                    stage_dir,
                    stage=stage,
                    target_mean=active_norm[0],
                    target_std=active_norm[1],
                    label_mean=label_norm[0],
                    label_std=label_norm[1],
                )

            wrapper = ResidualStageWrapper(
                active,
                prev_models,
                prev_norms=prev_norms,
                active_norm=active_norm,
                label_norm=label_norm,
            )
            wrapper = wrapper.to(device)

            base_loss_fn = bind_label_loss(
                dataset.label_mode,
                stage_cfg,
                yield_target=getattr(
                    dataset, 'yield_target', stage_cfg.get('yield_target', 'both')
                ),
                latent_distribution=getattr(
                    dataset,
                    'latent_distribution',
                    stage_cfg.get('latent_distribution', 'normal'),
                ),
            )
            # wrap_model_loss is a no-op branch for regression; keep for parity.
            base_loss_fn = wrap_model_loss(wrapper.active, base_loss_fn, stage_cfg)
            loss_fn = bind_residual_stage_loss(base_loss_fn, wrapper)

            outcome = train_one_run(
                stage_cfg,
                dataset,
                total_features,
                num_epochs=num_epochs,
                resume=resume,
                extend=extend,
                extend_epochs=extend_epochs,
                probe_weights=probe_weights,
                show_banner=True,
                suppress_history_plot=True,
                model=wrapper,
                loss_fn=loss_fn,
            )
            outcome = dict(outcome)
            outcome['error_ensemble_stage'] = stage
            outcome['status'] = 'ok'
            results.append(outcome)

            if outcome.get('skipped_finished'):
                print(f"  ✓ stage {stage} already finished — skipped")
            else:
                print(
                    f"  ✓ stage {stage} done  "
                    f"key={outcome.get('checkpoint_key')}  "
                    f"best_val={outcome.get('best_val_loss')}"
                )
        except Exception as exc:
            err = {
                'error_ensemble_stage': stage,
                'status': 'error',
                'error': f"{type(exc).__name__}: {exc}",
            }
            results.append(err)
            print(f"  ✗ stage {stage} failed: {err['error']}")
            if not continue_on_error:
                raise

    # Build a combined ErrorEnsemble from trained stage checkpoints (best effort).
    ensemble = None
    try:
        loaded, _keys, _stages, stage_norms, label_norm = load_error_ensemble_stages(
            cfg_base,
            dataset,
            n=n_models,
            device=device,
            require_finished=False,
        )
        if loaded:
            base_loss = bind_label_loss(label_mode, cfg_base)
            ensemble = ErrorEnsemble(
                models=loaded,
                label=label_mode,
                loss=base_loss,
                stage_norms=stage_norms,
                label_norm=label_norm,
                config=cfg_base,
            )
    except Exception as exc:
        print(f"⚠ Could not assemble final ErrorEnsemble: {exc}")

    n_ok = sum(1 for r in results if r.get('status') == 'ok')
    print("\n" + "=" * 60)
    print(f"ERROR ENSEMBLE DONE: {n_ok}/{n_models} stages ok")
    print("=" * 60 + "\n")

    return {
        'n': n_models,
        'results': results,
        'ensemble': ensemble,
        'n_ok': n_ok,
    }
