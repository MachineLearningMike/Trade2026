# error_ensemble.py
# Residual / error-boosting ensemble: model_i fits the leftover error of model_0..i-1.
#
#   model_0: y_0 ≈ y,            e_0 = y - y_0
#   model_1: y_1 ≈ e_0,          e_1 = e_0 - y_1
#   model_i: y_i ≈ e_{i-1},      e_i = e_{i-1} - y_i
#
# Ensemble prediction: ŷ = Σ_i y_i   (so e_n = y - ŷ).
#
# Training is sequential: freeze earlier stages, fit the next on the residual.

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from labels import bind_label_loss, get_label_class


ModelLike = Union[nn.Module, Callable[[], nn.Module]]
LossFn = Callable[[torch.Tensor, Any, torch.Tensor], torch.Tensor]


def head_prediction(output: Any) -> torch.Tensor:
    """Extract the prediction head tensor from a model forward output."""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def compute_residuals(
    y: torch.Tensor,
    stage_preds: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """
    e_0 = y - y_0;  e_i = e_{i-1} - y_i.

    Returns ``[e_0, e_1, ..., e_{n-1}]`` for ``n = len(stage_preds)``.
    """
    if not stage_preds:
        return []
    residuals: List[torch.Tensor] = []
    err = y
    for pred in stage_preds:
        err = err - pred
        residuals.append(err)
    return residuals


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
    base market model). ``last_prev_sum`` is Σ y_j for j < stage (or None for
    stage 0), used by the residual loss to form target e_{stage-1}.
    """

    def __init__(
        self,
        active: nn.Module,
        previous: Optional[Sequence[nn.Module]] = None,
    ):
        super().__init__()
        self.active = active
        prev_list = list(previous or [])
        self.prev = nn.ModuleList(prev_list)
        for p in self.prev.parameters():
            p.requires_grad = False
        self.last_prev_sum: Optional[torch.Tensor] = None
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

    def forward(self, x, p, mean, std, shape):
        if len(self.prev) == 0:
            self.last_prev_sum = None
        else:
            with torch.no_grad():
                total = None
                for m in self.prev:
                    pred = head_prediction(m(x, p, mean, std, shape))
                    total = pred if total is None else total + pred
                self.last_prev_sum = total
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
    ``loss(x, y_i, e_{i-1})`` with ``e_{-1} := y`` and
    ``e_{i-1} = y - Σ_{j<i} y_j`` (prev sum from the wrapper forward).
    """

    def loss_fn(x, y_pred, y_truth):
        prev = stage_model.last_prev_sum
        if prev is None:
            target = y_truth
        else:
            target = y_truth - prev
        return base_loss_fn(x, y_pred, target)

    loss_fn.label_cls = getattr(base_loss_fn, 'label_cls', None)  # type: ignore[attr-defined]
    loss_fn.auxiliary = getattr(base_loss_fn, 'auxiliary', None)  # type: ignore[attr-defined]
    loss_fn.base_loss_fn = base_loss_fn  # type: ignore[attr-defined]
    loss_fn.stage_model = stage_model  # type: ignore[attr-defined]
    return loss_fn


class ErrorEnsemble(nn.Module):
    """
    Ensemble of residual error models ``{model_0, ..., model_{n-1}}``.

    Args:
        model: ``nn.Module`` prototype (deep-copied) or zero-arg factory, OR a
            sequence of already-built stage models when ``models`` is omitted
            and ``n`` is None.
        label: label mode string (``'yield'|'trend'|...``) or label class.
        loss: loss callable ``(x, y_pred, y_truth) -> Tensor``. If omitted and
            ``label`` is set, binds the label-family loss from config.
        n: number of stages when expanding from a single prototype/factory.
        models: explicit list of stage modules (overrides ``model``/``n``).
        config: optional config for default label-loss binding.
    """

    def __init__(
        self,
        model: Optional[ModelLike] = None,
        label: Any = None,
        loss: Optional[LossFn] = None,
        *,
        n: Optional[int] = None,
        models: Optional[Sequence[nn.Module]] = None,
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

        self.last_stage_preds: List[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.models)

    @property
    def n(self) -> int:
        return len(self.models)

    def stage_predictions(
        self,
        x,
        p,
        mean,
        std,
        shape,
    ) -> List[torch.Tensor]:
        preds: List[torch.Tensor] = []
        for m in self.models:
            preds.append(head_prediction(m(x, p, mean, std, shape)))
        self.last_stage_preds = preds
        return preds

    def forward(self, x, p, mean, std, shape):
        """
        Return ``(ŷ, patch_map, recon)`` where ŷ = Σ y_i.

        ``patch_map`` / ``recon`` come from the last stage (API compatibility
        with MarketTransformerWithHeads / trainer / INFER).
        """
        total = None
        last_full = None
        stage_preds: List[torch.Tensor] = []
        for m in self.models:
            out = m(x, p, mean, std, shape)
            last_full = out
            pred = head_prediction(out)
            stage_preds.append(pred)
            total = pred if total is None else total + pred
        self.last_stage_preds = stage_preds
        if isinstance(last_full, (tuple, list)):
            patch_map = last_full[1] if len(last_full) > 1 else None
            recon = last_full[2] if len(last_full) > 2 else None
            return total, patch_map, recon
        return total, None, None

    def residuals(self, y: torch.Tensor) -> List[torch.Tensor]:
        """Residuals from the most recent ``forward`` / ``stage_predictions``."""
        return compute_residuals(y, self.last_stage_preds)

    def stage_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        stage: int,
    ) -> torch.Tensor:
        """
        Loss for stage ``i`` against target ``e_{i-1}`` (``e_{-1}=y``).

        Uses predictions cached in ``last_stage_preds`` (call forward first).
        """
        if self.loss_fn is None:
            raise RuntimeError("ErrorEnsemble has no loss_fn; pass loss=... at init")
        if not self.last_stage_preds:
            raise RuntimeError("No stage predictions; call forward() first")
        if stage < 0 or stage >= len(self.last_stage_preds):
            raise IndexError(
                f"stage {stage} out of range for n={len(self.last_stage_preds)}"
            )
        if stage == 0:
            target = y
        else:
            prev_sum = self.last_stage_preds[0]
            for pred in self.last_stage_preds[1:stage]:
                prev_sum = prev_sum + pred
            target = y - prev_sum
        return self.loss_fn(x, self.last_stage_preds[stage], target)

    def ensemble_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Loss of the summed prediction ŷ against y."""
        if self.loss_fn is None:
            raise RuntimeError("ErrorEnsemble has no loss_fn; pass loss=... at init")
        if not self.last_stage_preds:
            raise RuntimeError("No stage predictions; call forward() first")
        total = self.last_stage_preds[0]
        for pred in self.last_stage_preds[1:]:
            total = total + pred
        return self.loss_fn(x, total, y)


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
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Compare the nested ensembles built from trained error-models.

    Given ``model_0 .. model_m`` (m+1 models), form m+1 ensembles:

      en_0 = model_0
      en_1 = model_0 + model_1
      en_k = sum_{i=0..k} model_i

    For each ``en_k``, report mean ``loss_fn(x, en_k(x), y)``, residual MSE,
    and residual L1 over ``loader`` batches ``(x, p, mean, std, shape, y)``.
    """
    stage_models = list(models)
    if not stage_models:
        return []

    if device is None:
        device = next(stage_models[0].parameters()).device

    for m in stage_models:
        m.eval()
        m.to(device)

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
        x, p, mean, std, shape, y = batch[:6]
        x = x.to(device)
        p = p.to(device)
        mean = mean.to(device)
        std = std.to(device)
        shape = shape.to(device)
        y = y.to(device)

        stage_preds: List[torch.Tensor] = []
        for m in stage_models:
            stage_preds.append(head_prediction(m(x, p, mean, std, shape)))

        # en_k = sum_{i=0..k} model_i
        total = None
        bsz = int(y.shape[0])
        for k, pred in enumerate(stage_preds):
            total = pred if total is None else total + pred
            loss_k = loss_fn(x, total, y)
            resid = y - total
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
                'definition': ' + '.join(members),
                'members': members,
                'n_models': k + 1,
                # Aliases kept for older report readers.
                'n_stages': k + 1,
                'last_model': f'model_{k}',
                'loss': sum_loss[k] / n_examples,
                'residual_mse': sum_mse[k] / n_examples,
                'residual_l1': sum_l1[k] / n_examples,
                'n_examples': n_examples,
                'n_batches': n_batches,
            }
        )
    return out


def load_error_ensemble_stages(
    config: dict,
    dataset,
    *,
    n: Optional[int] = None,
    stage_keys: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    require_finished: bool = False,
) -> Tuple[List[nn.Module], List[str], List[int]]:
    """
    Load consecutive ErrorEnsemble stage weights starting from model_0.

    Stops at the first missing checkpoint (cumulative prefix only).
    Returns ``(models, keys_loaded, stage_indices)``.
    """
    import json
    from pathlib import Path

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

    models: List[nn.Module] = []
    keys_loaded: List[str] = []
    stages_loaded: List[int] = []

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
        models.append(model)
        keys_loaded.append(key)
        stages_loaded.append(stage)

    return models, keys_loaded, stages_loaded


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

            prev_models: List[nn.Module] = []
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

            wrapper = ResidualStageWrapper(active, prev_models)
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
        past_window = int(resolve_past_window(cfg_base))
        F = (
            total_features.shape[0]
            if hasattr(total_features, 'shape')
            else len(total_features)
        )
        n_features_after_latent = (
            dataset.n_market_features + cfg_base['n_latent_macros']
        )
        cfg_base['embed_dim'] = n_features_after_latent * cfg_base['patch_size']
        cfg_base['max_seq_len'] = resolve_max_seq_len(cfg_base, past_window)
        cfg_base['output_dim'] = dataset.output_dim
        model_past_len = int(
            getattr(dataset, 'model_past_len', resolve_model_past_len(cfg_base))
        )

        loaded: List[nn.Module] = []
        for stage in range(n_models):
            m = build_market_model(
                cfg_base,
                in_features=F,
                n_market_features=dataset.n_market_features,
                n_macro_features=dataset.n_macro_features,
                n_max_past_horizon=model_past_len,
                max_seq_len=cfg_base['max_seq_len'],
            )
            _load_stage_weights(m, cfg_base, dataset.cache_key, stage, device)
            loaded.append(m)
        base_loss = bind_label_loss(label_mode, cfg_base)
        ensemble = ErrorEnsemble(
            models=loaded,
            label=label_mode,
            loss=base_loss,
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
