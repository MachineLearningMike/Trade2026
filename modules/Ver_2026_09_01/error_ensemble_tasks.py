# error_ensemble_tasks.py
# Flat residual ensemble in Y₀ using Task + Task_Error + train_one_run.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Subset

from Task_Error import Task_Error
from data import ThreadedDataLoader
from task_base import merge_error_stage_model_spec
from trainer import train_one_run, generate_checkpoint_key


def _device(config: dict):
    return torch.device(config['device'] if config.get('device') else 'cpu')


def _print_stage_model_spec(stage: int, cfg: dict) -> None:
    print(
        f"  model_spec stage={stage}: type={cfg.get('model_type')} "
        f"patch={cfg.get('patch_size')} layers={cfg.get('num_layers')} "
        f"heads={cfg.get('num_heads')} dropout={cfg.get('dropout')}"
    )


def _as_2d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        return t.unsqueeze(-1)
    return t


def collect_y0(dataset) -> torch.Tensor:
    """Stack datapoint ``y`` for every index (already in parent Y₀ / Eₖ)."""
    n = len(dataset)
    rows = [dataset[i]['y'] for i in range(n)]
    return _as_2d(torch.stack(rows, dim=0))


@torch.no_grad()
def predict_head(
    task,
    model: torch.nn.Module,
    dataset,
    *,
    config: dict,
    indices: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Fast batched head predictions (after ``post_neural_function``).

    One model kept in memory; steps with ``validate_batch_size``.
    Returns CPU tensor ``(N, D)`` aligned with ``indices`` (or all samples).
    """
    device = _device(config)
    model = model.to(device)
    model.eval()
    if indices is None:
        indices = list(range(len(dataset)))
    else:
        indices = list(indices)
    step = int(config.get('validate_batch_size', 32) or 32)
    touch = task.post_neural_function
    chunks: List[torch.Tensor] = []
    for start in range(0, len(indices), step):
        end = min(start + step, len(indices))
        batch_idx = indices[start:end]
        datapoints = [dataset[int(i)] for i in batch_idx]
        batch = ThreadedDataLoader.collate_datapoints(datapoints)
        x = batch['x'].to(device, non_blocking=True)
        mean = batch['mean'].to(device, non_blocking=True)
        std = batch['std'].to(device, non_blocking=True)
        momentum = batch['momentum'].to(device, non_blocking=True)
        out = model(x, mean, std, momentum)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        if touch is not None:
            pred = touch(pred)
        chunks.append(_as_2d(pred.detach().cpu()))
    return torch.cat(chunks, dim=0)


def denorm_to_y0(
    pred_ek: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Map Task_Error head space Eₖ → Y₀ with that stage's label stats."""
    mean = mean.reshape(1, -1).to(pred_ek)
    std = std.reshape(1, -1).to(pred_ek)
    return pred_ek * std + mean


class ErrorStageDataset:
    """
    Reuses FeaturePrep tensors from ``base``; swaps task + residual labels.

    ``residual_y0`` is the stage's *raw* target in Y₀ (before this Task's
    global normalize_y whitening).
    """

    def __init__(
        self,
        base,
        task,
        residual_y0: torch.Tensor,
        *,
        stage: int,
        config: dict,
    ):
        self.base = base
        self.task = task
        self.config = dict(config)
        self.config['error_ensemble_stage'] = int(stage)
        self.config['label'] = task.name
        residual_y0 = _as_2d(residual_y0.detach().cpu().float())
        if residual_y0.shape[0] != len(base):
            raise ValueError(
                f"residual length {residual_y0.shape[0]} != dataset {len(base)}"
            )
        self.output_dim = int(residual_y0.shape[-1])
        self.config['output_dim'] = self.output_dim
        task.sync_output_dim(self.config)

        self.label_channels = slice(0, self.output_dim)
        self.train_indices = list(base.train_indices)
        self.val_indices = list(base.val_indices)
        self.test_indices = list(base.test_indices)

        tr = residual_y0[self.train_indices]
        self.label_stat_mean = tr.mean(dim=0)
        self.label_stat_std = tr.std(dim=0).clamp_min(1e-8)
        self.label_tensor = residual_y0.clone()
        if task.normalize_y:
            self.label_tensor.sub_(self.label_stat_mean).div_(
                self.label_stat_std + 1e-8
            )

        # FeaturePrep / identity mirrors (read by build_datapoint / train_one_run).
        self.Total = base.Total
        self.mean_tensor = base.mean_tensor
        self.std_tensor = base.std_tensor
        self.momentum_tensor = base.momentum_tensor
        self.stats = base.stats
        self.F = base.F
        self.T = base.T
        self.n_market_features = base.n_market_features
        self.n_macro_features = base.n_macro_features
        self.past_window = base.past_window
        self.future_window = base.future_window
        self.telepred_delay = base.telepred_delay
        self.price_feature_idx = base.price_feature_idx
        self.num_momentum = base.num_momentum
        self.normalization_layers = base.normalization_layers
        self.normalize_window = base.normalize_window
        self.valid_t = base.valid_t
        self.valid_starts = base.valid_starts
        self.total_features = base.total_features
        if hasattr(base, 'x_stat_mean'):
            self.x_stat_mean = base.x_stat_mean
            self.x_stat_std = base.x_stat_std
        # Distinct cache key identity for this residual stage.
        self.cache_key = f"{base.cache_key}_err{int(stage)}"
        self.train_dataset = Subset(self, self.train_indices)
        self.val_dataset = Subset(self, self.val_indices)
        self.test_dataset = Subset(self, self.test_indices)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        return self.task.build_datapoint(self, idx)


@torch.no_grad()
def evaluate_ensemble_y0(
    members: Sequence[Tuple[Any, torch.nn.Module, Optional[torch.Tensor], Optional[torch.Tensor]]],
    dataset,
    *,
    config: dict,
    indices: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """
    Fast ensemble eval in Y₀: one batched pass per member, sum contributions.

    ``members``: list of
      (task, model, denorm_mean_or_None, denorm_std_or_None)
    Base Task uses ``None, None`` (prediction already in Y₀).
    Task_Error uses its ``label_stat_mean/std`` to map Eₖ → Y₀.
    """
    if indices is None:
        indices = list(dataset.val_indices)
    else:
        indices = list(indices)
    y = torch.stack([dataset[int(i)]['y'] for i in indices], dim=0)
    y = _as_2d(y)
    # Parent Y₀: if this dataset is ErrorStageDataset, y is in Eₖ — caller
    # should pass the *base* dataset for eval against original y.
    running = None
    for task, model, mu, std in members:
        pred = predict_head(
            task, model, dataset, config=config, indices=indices,
        )
        if mu is not None and std is not None:
            contrib = denorm_to_y0(pred, mu, std)
        else:
            contrib = pred
        running = contrib if running is None else running + contrib
    assert running is not None
    resid = y - running
    mse = float(resid.pow(2).mean().item())
    mae = float(resid.abs().mean().item())
    return {
        'n': float(len(indices)),
        'mse': mse,
        'mae': mae,
        'rmse': float(mse ** 0.5),
    }


def run_error_task_ensemble(
    config: dict,
    dataset,
    total_features,
    *,
    task,
    n: int = 3,
    start_stage: int = 0,
    num_epochs: Optional[float] = None,
    resume: bool = True,
    extend: bool = False,
    extend_epochs: float = 0.0,
    probe_weights: bool = True,
) -> Dict[str, Any]:
    """
    Train base ``task`` then ``n-1`` successive ``Task_Error`` stages.

    Flat additive ensemble in the base Task's Y₀. Evaluation uses one forward
    per member over the val split (models kept / reloaded once each).
    """
    from INFER import find_checkpoint_by_key, load_model

    n = int(n)
    if n < 1:
        raise ValueError(f"error_ensemble_n must be >= 1, got {n}")
    cfg0 = dict(config)
    cfg0['error_ensemble_n'] = n
    device = _device(cfg0)

    members: List[
        Tuple[Any, torch.nn.Module, Optional[torch.Tensor], Optional[torch.Tensor]]
    ] = []
    results: List[Dict[str, Any]] = []

    # ---- Stage 0: original Task ----
    print("\n" + "=" * 60)
    print(f"ErrorEnsemble Task stage 0 / {n}  task={task.name}")
    print("=" * 60)
    cfg_s0 = merge_error_stage_model_spec(cfg0, 0)
    _print_stage_model_spec(0, cfg_s0)
    if start_stage <= 0:
        out0 = train_one_run(
            cfg_s0,
            dataset,
            total_features,
            task=task,
            num_epochs=num_epochs,
            resume=resume,
            extend=extend,
            extend_epochs=extend_epochs,
            probe_weights=probe_weights,
            show_banner=True,
            suppress_history_plot=True,
        )
        results.append(dict(out0, error_ensemble_stage=0, status='ok'))
    else:
        print("  (start_stage > 0: skipping base train; loading checkpoint)")
        results.append({'error_ensemble_stage': 0, 'status': 'skipped_train'})

    task.apply_model_spec(cfg_s0)
    key0 = generate_checkpoint_key(cfg_s0, dataset.cache_key)
    found0 = find_checkpoint_by_key(
        str(cfg_s0.get('checkpoint_dir', 'trading_checkpoints')), key=key0,
    )
    if found0 is None:
        raise FileNotFoundError(f"base stage checkpoint missing key={key0}")
    model0 = load_model(found0[0], cfg_s0, dataset, device=device)
    members.append((task, model0, None, None))

    # Y₀ targets + running residual on the full index set (for next-stage y).
    # Base datapoint y is already Y₀ (or Rise probs space after touch on pred).
    y0_all = collect_y0(dataset)
    pred0_all = predict_head(task, model0, dataset, config=cfg_s0)
    if pred0_all.shape != y0_all.shape:
        # Rise: y (N,1), pred (N,1) after sigmoid; allow broadcast fix
        pred0_all = _as_2d(pred0_all)
        y0_all = _as_2d(y0_all)
    residual = y0_all - pred0_all  # error_0 in Y₀

    # ---- Stages 1 .. n-1: Task_Error ----
    for stage in range(1, n):
        print("\n" + "=" * 60)
        print(f"ErrorEnsemble Task stage {stage} / {n}  task={Task_Error.name}")
        print("=" * 60)
        cfg_stage = merge_error_stage_model_spec(cfg0, stage)
        _print_stage_model_spec(stage, cfg_stage)
        err_ds = ErrorStageDataset(
            dataset,
            Task_Error,
            residual,
            stage=stage,
            config=cfg_stage,
        )
        cfg_s = dict(err_ds.config)
        cfg_s['error_ensemble_n'] = n
        # Re-apply stage neural knobs after ErrorStageDataset copies config.
        cfg_s = merge_error_stage_model_spec(cfg_s, stage)
        if stage >= start_stage:
            out_s = train_one_run(
                cfg_s,
                err_ds,
                total_features,
                task=Task_Error,
                num_epochs=num_epochs,
                resume=resume,
                extend=extend,
                extend_epochs=extend_epochs,
                probe_weights=probe_weights,
                show_banner=True,
                suppress_history_plot=True,
            )
            results.append(
                dict(out_s, error_ensemble_stage=stage, status='ok')
            )
        else:
            print(f"  (start_stage={start_stage}: skipping train for stage {stage})")
            results.append(
                {'error_ensemble_stage': stage, 'status': 'skipped_train'}
            )

        Task_Error.apply_model_spec(cfg_s)
        key_s = generate_checkpoint_key(cfg_s, err_ds.cache_key)
        found_s = find_checkpoint_by_key(
            str(cfg_s.get('checkpoint_dir', 'trading_checkpoints')), key=key_s,
        )
        if found_s is None:
            raise FileNotFoundError(
                f"error stage {stage} checkpoint missing key={key_s}"
            )
        model_s = load_model(found_s[0], cfg_s, err_ds, device=device)
        mu = err_ds.label_stat_mean
        std = err_ds.label_stat_std
        members.append((Task_Error, model_s, mu, std))

        # Update residual in Y₀: error_k = error_{k-1} - denorm(ê_k)
        pred_ek = predict_head(
            Task_Error, model_s, err_ds, config=cfg_s,
        )
        contrib = denorm_to_y0(pred_ek, mu, std)
        residual = residual - contrib

    # ---- Fast val ensemble eval in Y₀ (base dataset y) ----
    print("\n" + "=" * 60)
    print("ErrorEnsemble fast val eval (Y₀, one forward / member)")
    print("=" * 60)
    # Evaluate cumulative prefixes for a quick learning curve.
    prefix_metrics = []
    for k in range(1, len(members) + 1):
        # Predictions must use the dataset each model was trained with for
        # FeaturePrep; base features are shared, but ErrorStageDataset swaps y.
        # For speed: run each member on the *base* dataset FeaturePrep (same x).
        metrics = _eval_prefix_on_base(
            members[:k], dataset, config=cfg0,
        )
        metrics['k'] = k
        prefix_metrics.append(metrics)
        print(
            f"  ens_k={k}: mse={metrics['mse']:.6f}  mae={metrics['mae']:.6f}  "
            f"n={int(metrics['n'])}"
        )

    return {
        'n': n,
        'results': results,
        'val_prefix_metrics': prefix_metrics,
        'val_metrics': prefix_metrics[-1] if prefix_metrics else {},
    }


@torch.no_grad()
def _eval_prefix_on_base(
    members: Sequence[Tuple[Any, torch.nn.Module, Optional[torch.Tensor], Optional[torch.Tensor]]],
    base_dataset,
    *,
    config: dict,
) -> Dict[str, float]:
    """
    Sum member contributions in Y₀ on ``base_dataset`` val indices.

    All members share FeaturePrep from ``base_dataset`` (x/mean/std/momentum);
    only the head + optional denorm differ. One batched pass per member.
    """
    indices = list(base_dataset.val_indices)
    y = _as_2d(torch.stack([base_dataset[i]['y'] for i in indices], dim=0))
    running = None
    for task, model, mu, std in members:
        pred = predict_head(
            task, model, base_dataset, config=config, indices=indices,
        )
        if mu is not None and std is not None:
            contrib = denorm_to_y0(pred, mu, std)
        else:
            contrib = pred
        running = contrib if running is None else running + contrib
    resid = y - running
    mse = float(resid.pow(2).mean().item())
    return {
        'n': float(len(indices)),
        'mse': mse,
        'mae': float(resid.abs().mean().item()),
        'rmse': float(mse ** 0.5),
    }
