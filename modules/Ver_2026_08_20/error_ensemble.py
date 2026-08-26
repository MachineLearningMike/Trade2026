# error_ensemble.py
# Residual / error-boosting ensemble.
#
# SOURCE OF TRUTH: error_ensemble_logic.txt — keep this module aligned with it.
#
# All additive structure is in **y-normalized** Yield space:
#   e_0 = y - y_0
#   model_{i+1} fits norm_i(e_i);  e_i_hat = denorm_i(y_{i+1})
#   e_{i+1} = e_i - e_i_hat
#   en_k = y_0 + denorm_0(y_1) + ... + denorm_{k-1}(y_k)
#   score en_k against y (not y_raw)
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
# v2: residual stats / summing in y-normalized space (error_ensemble_logic.txt).
STAGE_NORM_VERSION = 2


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
        'version': int(STAGE_NORM_VERSION),
        'stage': int(stage),
        'target_mean': torch.as_tensor(target_mean).detach().cpu().tolist(),
        'target_std': torch.as_tensor(target_std).detach().cpu().tolist(),
        'label_mean': torch.as_tensor(label_mean).detach().cpu().tolist(),
        'label_std': torch.as_tensor(label_std).detach().cpu().tolist(),
        'space': 'y_normalized',
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return path


def load_stage_norm(
    checkpoint_dir: Path,
    *,
    fallback_mean: Optional[torch.Tensor] = None,
    fallback_std: Optional[torch.Tensor] = None,
    require_current_version: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load ``(target_mean, target_std)`` for a stage.

    If the sidecar is missing or from an older residual-space scheme, falls
    back to ``fallback_mean/std`` unless ``require_current_version`` is set.
    """
    path = Path(checkpoint_dir) / STAGE_NORM_FILENAME
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ver = int(data.get('version', 1) or 1)
        if ver >= STAGE_NORM_VERSION:
            mean = torch.tensor(data['target_mean'], dtype=torch.float32)
            std = torch.tensor(data['target_std'], dtype=torch.float32)
            return mean, std
        if require_current_version:
            raise FileNotFoundError(
                f"{path} has version={ver}; need >={STAGE_NORM_VERSION} "
                f"(y-space residual norms). Re-run with --clean."
            )
        print(
            f"⚠ {path.name} in {checkpoint_dir.name} is version={ver} "
            f"(need {STAGE_NORM_VERSION} y-space norms); using fallback"
        )
    elif require_current_version:
        raise FileNotFoundError(
            f"Missing {STAGE_NORM_FILENAME} in {checkpoint_dir}"
        )
    if fallback_mean is None or fallback_std is None:
        raise FileNotFoundError(
            f"Missing usable {STAGE_NORM_FILENAME} in {checkpoint_dir} "
            f"and no fallback mean/std provided"
        )
    if not path.exists():
        print(
            f"⚠ {path.name} missing in {checkpoint_dir.name}; "
            f"falling back to provided mean/std"
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


def contribution_in_y_space(
    stage: int,
    pred: torch.Tensor,
    stage_norm: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Map ``model_stage`` output into y-normalized space for additive ensembling.

    Per error_ensemble_logic.txt:
      - model_0: ``y_0`` already in y-space → return ``pred``
      - model_i (i>=1): ``pred`` is in norm_{i-1}-space; ``stage_norm`` is
        ``(mean_{i-1}, std_{i-1})`` stored in model_i's sidecar
        → return ``denorm_{i-1}(pred)``
    """
    if int(stage) == 0:
        return pred
    mean, std = stage_norm
    return apply_denormalize(pred, mean, std)


def sum_stages_in_y_space(
    stage_preds: Sequence[torch.Tensor],
    stage_norms: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """
    ``en = y_0 + denorm(y_1) + denorm(y_2) + ...`` in y-normalized space.
    """
    if len(stage_preds) != len(stage_norms):
        raise ValueError(
            f"pred/norm length mismatch: {len(stage_preds)} vs {len(stage_norms)}"
        )
    total = None
    for i, (pred, norm) in enumerate(zip(stage_preds, stage_norms)):
        contrib = contribution_in_y_space(i, pred, norm)
        total = contrib if total is None else total + contrib
    if total is None:
        raise ValueError("sum_stages_in_y_space requires at least one stage")
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

    ``forward`` returns the active stage output. Previous stages are mapped
    into y-normalized space and summed into ``last_prev_sum_y``:

      stage 0 contrib = y_0
      stage j>0 contrib = denorm_{j-1}(y_j)  # sidecar of model_j holds (mean_{j-1}, std_{j-1})
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
            label_norm = active_norm
        l_mean, l_std = label_norm
        self.register_buffer('label_mean', l_mean.detach().float().reshape(-1))
        self.register_buffer('label_std', l_std.detach().float().reshape(-1))

        self.last_prev_sum_y: Optional[torch.Tensor] = None
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
            self.last_prev_sum_y = None
        else:
            with torch.no_grad():
                total_y = None
                for i, m in enumerate(self.prev):
                    pred = head_prediction(m(x, p, mean, std, shape))
                    contrib = contribution_in_y_space(
                        i, pred, self._prev_norm_pair(i)
                    )
                    total_y = contrib if total_y is None else total_y + contrib
                self.last_prev_sum_y = total_y
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
    Stage loss in y-normalized space (error_ensemble_logic.txt):

      e = y - sum_{j<i} contrib_y(model_j)     # e_{-1} path: e = y
      target = normalize_i(e)                  # active stage mean/std
    """

    def loss_fn(x, y_pred, y_truth):
        # y_truth is already y-normalized from the dataloader.
        prev_y = stage_model.last_prev_sum_y
        if prev_y is None:
            e = y_truth
        else:
            e = y_truth - prev_y
        target = apply_normalize(
            e, stage_model.active_mean, stage_model.active_std
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

    Flat ensembles in y-normalized space (error_ensemble_logic.txt):

      en_k = y_0 + denorm_0(y_1) + ... + denorm_{k-1}(y_k)
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
        self.last_stage_preds_y: List[torch.Tensor] = []
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
        """Head outputs in each stage's own target space."""
        preds: List[torch.Tensor] = []
        for m in self.models:
            preds.append(head_prediction(m(x, p, mean, std, shape)))
        self.last_stage_preds_norm = preds
        self.last_stage_preds = preds
        self.last_stage_preds_y = [
            contribution_in_y_space(i, pred, self.stage_norm_pair(i))
            for i, pred in enumerate(preds)
        ]
        return preds

    def forward(self, x, p, mean, std, shape):
        """
        Return ``(en_m, patch_map, recon)`` in **y-normalized** space:

          en_m = y_0 + denorm(y_1) + ... + denorm(y_m)
        """
        last_full = None
        preds_norm: List[torch.Tensor] = []
        preds_y: List[torch.Tensor] = []
        total_y = None
        for i, m in enumerate(self.models):
            out = m(x, p, mean, std, shape)
            last_full = out
            pred_norm = head_prediction(out)
            pred_y = contribution_in_y_space(i, pred_norm, self.stage_norm_pair(i))
            preds_norm.append(pred_norm)
            preds_y.append(pred_y)
            total_y = pred_y if total_y is None else total_y + pred_y
        self.last_stage_preds_norm = preds_norm
        self.last_stage_preds_y = preds_y
        self.last_stage_preds = preds_norm
        if isinstance(last_full, (tuple, list)):
            patch_map = last_full[1] if len(last_full) > 1 else None
            recon = last_full[2] if len(last_full) > 2 else None
            return total_y, patch_map, recon
        return total_y, None, None

    def residuals(self, y: torch.Tensor) -> List[torch.Tensor]:
        """Y-space residuals from the most recent forward."""
        return compute_residuals(y, self.last_stage_preds_y)

    def ensemble_prediction(self, k: Optional[int] = None) -> torch.Tensor:
        """``en_k`` from cached y-space stage contribs. ``k=None`` → full."""
        if not self.last_stage_preds_y:
            raise RuntimeError("No stage predictions; call forward() first")
        if k is None:
            k = len(self.last_stage_preds_y) - 1
        if k < 0 or k >= len(self.last_stage_preds_y):
            raise IndexError(f"en_k index {k} out of range")
        total = self.last_stage_preds_y[0]
        for pred in self.last_stage_preds_y[1 : k + 1]:
            total = total + pred
        return total

    def ensemble_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Loss of full ``en_m`` against y-normalized ``y``."""
        if self.loss_fn is None:
            raise RuntimeError("ErrorEnsemble has no loss_fn; pass loss=... at init")
        return self.loss_fn(x, self.ensemble_prediction(), y)


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


def _iter_loader_batches(loader, max_batches: Optional[int] = None):
    """Yield batches from ThreadedDataLoader or a plain iterable."""
    n_batches = 0
    if hasattr(loader, 'start_epoch') and hasattr(loader, 'get_next_batch'):
        loader.start_epoch(0)
        while True:
            if max_batches is not None and n_batches >= int(max_batches):
                break
            batch, _wait, _load = loader.get_next_batch()
            if batch is None:
                break
            yield batch
            n_batches += 1
        return
    for batch in loader:
        if max_batches is not None and n_batches >= int(max_batches):
            break
        yield batch
        n_batches += 1


def _release_model(model: Optional[nn.Module]) -> None:
    """Drop a stage module and free CUDA cache when possible."""
    if model is None:
        return
    try:
        model.to('cpu')
    except Exception:
        pass
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_one_error_stage(
    config: dict,
    dataset,
    *,
    stage: int,
    key: Optional[str] = None,
    device: Optional[torch.device] = None,
    require_current_norm_version: bool = True,
) -> Tuple[nn.Module, str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Load a **single** ErrorEnsemble stage (weights + target affine).

    Returns ``(model, checkpoint_key, (target_mean, target_std))``.
    Caller must release the model when finished (see ``_release_model``).

    By default requires STAGE_NORM_VERSION y-space sidecars. Old (v1) norms
    mix raw/label affines into y-space sums and make loss(en_i, y) rise with i.
    """
    from config import project_root_dir
    from labels import resolve_model_past_len, sync_output_dim
    from model_vae import build_market_model, resolve_max_seq_len
    from telepred import resolve_past_window
    from INFER import find_checkpoint_by_key

    cfg = dict(config)
    sync_output_dim(cfg)
    if device is None:
        device = torch.device(cfg['device'] if cfg.get('device') else 'cpu')
    else:
        device = torch.device(device)

    if key is None:
        key = stage_checkpoint_key(cfg, dataset.cache_key, int(stage))

    ckpt_root = Path(cfg.get('checkpoint_dir', 'trading_checkpoints'))
    if not ckpt_root.is_absolute():
        ckpt_root = project_root_dir() / ckpt_root
    key_dir = ckpt_root / key
    if not key_dir.exists():
        raise FileNotFoundError(f"stage {stage}: missing checkpoint dir {key_dir}")

    found = find_checkpoint_by_key(str(ckpt_root), key=key)
    if found is None:
        raise FileNotFoundError(f"stage {stage}: no weights under key {key}")
    path, _ = found
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    past_window = int(resolve_past_window(cfg))
    F = int(getattr(dataset, 'F', dataset.n_market_features + dataset.n_macro_features))
    n_features_after_latent = dataset.n_market_features + cfg['n_latent_macros']
    cfg['embed_dim'] = n_features_after_latent * cfg['patch_size']
    cfg['max_seq_len'] = int(ckpt.get('max_seq_len') or resolve_max_seq_len(cfg, past_window))
    cfg['output_dim'] = int(getattr(dataset, 'output_dim', cfg.get('output_dim', 2)))
    model_past_len = int(
        getattr(dataset, 'model_past_len', resolve_model_past_len(cfg))
    )

    model = build_market_model(
        cfg,
        in_features=F,
        n_market_features=dataset.n_market_features,
        n_macro_features=dataset.n_macro_features,
        n_max_past_horizon=model_past_len,
        max_seq_len=cfg['max_seq_len'],
    )
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    out_dim = int(getattr(dataset, 'output_dim', 2))
    identity = (
        torch.zeros(out_dim, dtype=torch.float32),
        torch.ones(out_dim, dtype=torch.float32),
    )
    # Stage 0 may use identity; stage>0 must have current y-space residual norms.
    if require_current_norm_version:
        stage_norm = load_stage_norm(
            key_dir,
            fallback_mean=identity[0] if int(stage) == 0 else None,
            fallback_std=identity[1] if int(stage) == 0 else None,
            require_current_version=(int(stage) > 0),
        )
        if int(stage) == 0:
            # Even if an old sidecar stored label mean/std, stage-0 contrib is y_0.
            stage_norm = identity
    else:
        label_norm = label_norm_from_dataset(dataset)
        stage_norm = load_stage_norm(
            key_dir,
            fallback_mean=label_norm[0],
            fallback_std=label_norm[1],
            require_current_version=False,
        )
    return model, key, stage_norm


@torch.no_grad()
def evaluate_ensemble_over_stages(
    loss_fn: LossFn,
    loader,
    *,
    config: dict,
    dataset,
    stage_keys: Sequence[str],
    label_norm: Tuple[torch.Tensor, torch.Tensor],
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    require_finished: bool = False,
) -> List[Dict[str, Any]]:
    """
    Wholesale ensemble eval over dataset ``D`` (the loader), compared in
    **y-normalized** label space (not raw):

        en_0 = model_0(D)                         # already in y-space
        en_k = en_{k-1} + denorm_{k-1}(model_k(D))  # k >= 1
        score loss(en_k, y)                       # both in y-space

    For each stage ``i``:
      1. load **only** ``model_i``
      2. full pass over ``D`` → contribution in y-space on CPU
      3. ``running += contribution``
      4. free ``model_i`` before loading ``model_{i+1}``

    Never keeps several models resident, and never does
    ``model_0(batch)+model_1(batch)+...`` inside one batch loop.
    """
    from config import project_root_dir

    if device is None:
        device = torch.device(config.get('device') or 'cpu')
    else:
        device = torch.device(device)

    # label_norm is used by load_one_error_stage fallbacks; metrics use y itself.
    _ = label_norm

    ckpt_root = Path(config.get('checkpoint_dir', 'trading_checkpoints'))
    if not ckpt_root.is_absolute():
        ckpt_root = project_root_dir() / ckpt_root

    # Which stages are available (dirs / finished flag) — no weight load yet.
    available: List[Tuple[int, str]] = []
    for stage, key in enumerate(stage_keys):
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
        available.append((stage, key))

    if not available:
        return []

    # Pass over D once with no model: collect y (normalized) on CPU.
    y_chunks: List[torch.Tensor] = []
    n_batches = 0
    for batch in _iter_loader_batches(loader, max_batches=max_batches):
        if not isinstance(batch, (tuple, list)) or len(batch) < 6:
            raise ValueError(
                "evaluate_ensemble_over_stages expects batches "
                "(x, p, mean, std, shape, y)"
            )
        y_norm = batch[5].detach().cpu()
        y_chunks.append(y_norm)
        n_batches += 1
    if not y_chunks:
        return []
    y_all = torch.cat(y_chunks, dim=0)
    n_examples = int(y_all.shape[0])

    running: Optional[torch.Tensor] = None  # CPU, y-normalized space
    out: List[Dict[str, Any]] = []

    for stage, key in available:
        # Wholesale: model_i(D) as a whole pass, then add into en (y-space).
        print(f"  wholesale model_{stage}(D)  key={key}")
        model = None
        try:
            model, _key, stage_norm = load_one_error_stage(
                config, dataset, stage=stage, key=key, device=device
            )
            s_mean, s_std = stage_norm
            contrib_chunks: List[torch.Tensor] = []
            for batch in _iter_loader_batches(loader, max_batches=max_batches):
                x, p, mean, std, shape, _y = batch[:6]
                x = x.to(device)
                p = p.to(device)
                mean = mean.to(device)
                std = std.to(device)
                shape = shape.to(device)
                pred = head_prediction(model(x, p, mean, std, shape))
                contrib_b = contribution_in_y_space(stage, pred, (s_mean, s_std))
                contrib_chunks.append(contrib_b.detach().cpu())
            # Full-dataset contribution model_i(D), then free model_i.
            contrib = torch.cat(contrib_chunks, dim=0)
            if contrib.shape != y_all.shape:
                raise RuntimeError(
                    f"model_{stage}(D) shape {tuple(contrib.shape)} "
                    f"!= y(D) shape {tuple(y_all.shape)} "
                    f"(loader order must be stable; use shuffle=False)"
                )
            running = contrib if running is None else running + contrib
        finally:
            _release_model(model)

        assert running is not None
        # Metrics for en_k vs y in normalized label space.
        resid = y_all - running
        loss = _ensemble_loss_cpu(loss_fn, running, y_all)
        k = len(out)
        members = [f'model_{available[j][0]}' for j in range(k + 1)]
        row = {
            'k': available[k][0],
            'ensemble': f'en_{available[k][0]}',
            'definition': ' + '.join(
                (
                    f'model_{available[j][0]}(D)'
                    if available[j][0] == 0
                    else f'denorm_{available[j][0]-1}(model_{available[j][0]}(D))'
                )
                for j in range(k + 1)
            ),
            'members': members,
            'n_models': k + 1,
            'n_stages': k + 1,
            'last_model': f'model_{available[k][0]}',
            'space': 'y_normalized',
            'load_mode': 'wholesale_one_model_over_D',
            'loss': loss,
            'residual_mse': float(resid.pow(2).mean().item()),
            'residual_l1': float(resid.abs().mean().item()),
            'n_examples': n_examples,
            'n_batches': n_batches,
        }
        if out and loss > float(out[-1]['loss']) + 1e-8:
            prev = float(out[-1]['loss'])
            print(
                f"⚠ loss(en_{row['k']}, y)={loss:.6f} > loss(en_{out[-1]['k']}, y)"
                f"={prev:.6f}. Residual boosting expects non-increasing ensembl"
                f"e loss when stages fit y-space residuals. Usually means old "
                f"raw-space checkpoints/norms — retrain with "
                f"--error-ensemble --clean."
            )
            row['loss_increased_vs_prev'] = True
        out.append(row)

    return out


def _ensemble_loss_cpu(
    loss_fn: LossFn,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Run label loss on CPU tensors; ``x`` is unused by yield/trend MSE."""
    dummy_x = pred  # shape-compatible placeholder
    val = loss_fn(dummy_x, pred, target)
    return float(val.item())


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
    Mean/std of the **y-space** residual target for ``stage``.

    Stage 0 → identity (0, 1): ``y`` is already label-normalized; no extra affine.
    Stage i>0 → stats of ``e = y - sum_{j<i} contrib_y(model_j)``.
    """
    out_dim = int(torch.as_tensor(label_norm[0]).numel())
    if stage == 0:
        # Identity affine so loss target stays y (already normalized).
        return (
            torch.zeros(out_dim, dtype=torch.float32),
            torch.ones(out_dim, dtype=torch.float32),
        )

    if len(prev_models) != stage or len(prev_norms) != stage:
        raise ValueError(
            f"stage {stage} needs {stage} prev models/norms; "
            f"got {len(prev_models)}/{len(prev_norms)}"
        )

    from data import ThreadedDataLoader

    for m in prev_models:
        m.eval()
        m.to(device)

    bs = int(config.get('batch_size', 32))
    loader = ThreadedDataLoader(
        dataset.train_dataset, batch_size=bs, shuffle=False
    )
    loader.start_epoch(0)

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

        prev_y = None
        for i, m in enumerate(prev_models):
            pred = head_prediction(m(x, p, mean, std, shape))
            contrib = contribution_in_y_space(i, pred, prev_norms[i])
            prev_y = contrib if prev_y is None else prev_y + contrib
        e = y_norm - prev_y  # type: ignore[operator]

        e_cpu = e.detach().to(dtype=torch.float64, device='cpu')
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
        f"  stage {stage} y-space residual stats "
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

        out_dim = int(cfg['output_dim'])
        identity_norm = (
            torch.zeros(out_dim, dtype=torch.float32),
            torch.ones(out_dim, dtype=torch.float32),
        )
        if stage == 0:
            s_mean, s_std = identity_norm
        else:
            s_mean, s_std = load_stage_norm(
                key_dir,
                require_current_version=True,
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
            out_dim = int(stage_cfg['output_dim'])
            identity_norm = (
                torch.zeros(out_dim, dtype=torch.float32),
                torch.ones(out_dim, dtype=torch.float32),
            )
            ckpt_root = Path(stage_cfg.get('checkpoint_dir', 'trading_checkpoints'))
            if not ckpt_root.is_absolute():
                ckpt_root = project_root_dir() / ckpt_root
            stage_key = stage_checkpoint_key(
                stage_cfg, dataset.cache_key, stage
            )
            stage_dir = ckpt_root / stage_key
            stage_dir.mkdir(parents=True, exist_ok=True)

            # Prev models + their sidecars per logic txt:
            #   model_0 → identity; model_j (j>=1) → (mean_{j-1}, std_{j-1}) of e_{j-1}
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
                if prev_stage == 0:
                    prev_norms.append(identity_norm)
                else:
                    prev_key = stage_checkpoint_key(
                        stage_cfg, dataset.cache_key, prev_stage
                    )
                    prev_norms.append(
                        load_stage_norm(
                            ckpt_root / prev_key,
                            require_current_version=True,
                        )
                    )

            # Active target affine (logic txt):
            #   model_0 → identity (fits y as-is)
            #   model_i (i>=1) → (mean_{i-1}, std_{i-1}) = stats of e_{i-1}
            # Reuse current-version sidecar; else recompute + save under model_i/.
            if stage == 0:
                active_norm = identity_norm
                save_stage_norm(
                    stage_dir,
                    stage=stage,
                    target_mean=active_norm[0],
                    target_std=active_norm[1],
                    label_mean=label_norm[0],
                    label_std=label_norm[1],
                )
            else:
                reuse_norm = False
                norm_path = stage_dir / STAGE_NORM_FILENAME
                if norm_path.exists():
                    try:
                        with open(norm_path, 'r', encoding='utf-8') as f:
                            _meta = json.load(f)
                        reuse_norm = (
                            int(_meta.get('version', 1) or 1) >= STAGE_NORM_VERSION
                        )
                    except Exception:
                        reuse_norm = False
                if reuse_norm:
                    active_norm = load_stage_norm(
                        stage_dir, require_current_version=True
                    )
                    print(
                        f"  loaded stage norm from {norm_path.name}: "
                        f"mean={active_norm[0].tolist()} "
                        f"std={active_norm[1].tolist()}"
                    )
                else:
                    if norm_path.exists():
                        print(
                            f"  recomputing stage norm "
                            f"(need v{STAGE_NORM_VERSION} y-space sidecar)"
                        )
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
