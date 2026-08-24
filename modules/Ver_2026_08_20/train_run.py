# train_run.py
# Shared single-element training used by TRAIN, --ensemble, and sweeps.

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.optim as optim

from data import ThreadedDataLoader
from labels import bind_label_loss, label_target_names, resolve_model_past_len
from model import report_model_size, check_all_weights_updated
from model_vae import build_market_model, resolve_max_seq_len, wrap_model_loss
from trainer import ResumableTrainer, generate_checkpoint_key


def train_one_run(
    config: dict,
    dataset,
    total_features,
    *,
    num_epochs: Optional[float] = None,
    resume: Optional[bool] = None,
    extend: bool = False,
    extend_epochs: float = 0.0,
    probe_weights: bool = True,
    show_banner: bool = True,
    suppress_history_plot: bool = False,
) -> Dict[str, Any]:
    """
    Build model + trainer and run ResumableTrainer.train for one config.

    ``config['telepred_delay']`` must already match ``dataset`` construction.
    Returns metrics and checkpoint_key for logging / ensemble orchestration.
    """
    cfg = config
    from telepred import resolve_past_window
    past_window = int(resolve_past_window(cfg))
    F = (
        total_features.shape[0]
        if hasattr(total_features, 'shape')
        else len(total_features)
    )
    n_features_after_latent = (
        dataset.n_market_features + cfg['n_latent_macros']
    )
    embed_dim = n_features_after_latent * cfg['patch_size']
    if embed_dim % cfg['num_heads'] != 0:
        raise ValueError(
            f"embed_dim ({embed_dim}) must be divisible by num_heads "
            f"({cfg['num_heads']})."
        )
    cfg['embed_dim'] = embed_dim
    cfg['max_seq_len'] = resolve_max_seq_len(cfg, past_window)
    cfg['output_dim'] = dataset.output_dim

    _model_type = str(cfg.get('model_type', 'regression')).lower()
    delay = int(cfg.get('telepred_delay', 0) or 0)

    if show_banner:
        print(f"\nData shapes:")
        print(f"  Input x: (F={F}, past_window={past_window})")
        print(f"  TelePred delay d: {delay}")
        print(f"  Label mode: {getattr(dataset, 'label_mode', cfg.get('label'))}")
        print(f"  Model type: {_model_type}")
        _od = cfg['output_dim']
        _yt = getattr(dataset, 'yield_target', cfg.get('yield_target', 'both'))
        _ld = getattr(
            dataset, 'latent_distribution', cfg.get('latent_distribution', 'normal')
        )
        _onames = label_target_names(
            getattr(dataset, 'label_mode', cfg.get('label', 'trend')),
            _od,
            yield_target=_yt,
            latent_distribution=_ld,
        )
        print(f"  Output: {_onames} - {_od} values (output_dim)")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Train/Val/Test: {len(dataset.train_indices)}/"
              f"{len(dataset.val_indices)}/{len(dataset.test_indices)}")

    _model_past_len = int(
        getattr(dataset, 'model_past_len', resolve_model_past_len(cfg))
    )
    model = build_market_model(
        cfg,
        in_features=F,
        n_market_features=dataset.n_market_features,
        n_macro_features=dataset.n_macro_features,
        n_max_past_horizon=_model_past_len,
        max_seq_len=cfg['max_seq_len'],
    )
    _model_name = type(model).__name__
    if show_banner:
        report_model_size(model, name=_model_name)

    _probe_device = (
        torch.device(cfg['device']) if cfg.get('device') else torch.device('cpu')
    )
    model = model.to(_probe_device)

    base_loss_fn = bind_label_loss(
        dataset.label_mode,
        cfg,
        yield_target=getattr(dataset, 'yield_target', cfg.get('yield_target', 'both')),
        latent_distribution=getattr(
            dataset, 'latent_distribution', cfg.get('latent_distribution', 'normal')
        ),
    )
    loss_fn = wrap_model_loss(model, base_loss_fn, cfg)

    if probe_weights:
        _probe_T = (past_window // cfg['patch_size']) * cfg['patch_size']
        check_all_weights_updated(
            model,
            sample_batch=(
                torch.randn(2, _probe_T, F, device=_probe_device),
                torch.randn(2, _model_past_len, device=_probe_device),
                torch.randn(2, F, device=_probe_device),
                torch.randn(2, F, device=_probe_device).abs() + 1e-3,
                torch.randn(
                    2, cfg.get('num_shapes', 4), F, device=_probe_device
                ),
            ),
            loss_fn=loss_fn,
            name=_model_name,
        )

    train_loader = ThreadedDataLoader(
        dataset.train_dataset, batch_size=cfg['batch_size'], shuffle=True
    )
    val_loader = ThreadedDataLoader(
        dataset.val_dataset, batch_size=cfg['validate_batch_size'], shuffle=False
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    ckpt_key = generate_checkpoint_key(cfg, dataset.cache_key)
    trainer = ResumableTrainer(
        dataset_signature=dataset.cache_key,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        dataloader=train_loader,
        save_dir=cfg['checkpoint_dir'],
        device=cfg['device'],
        save_every_n_minutes=cfg['save_every_n_minutes'],
        val_loader=val_loader,
        enable_validation=cfg.get('validate', True),
        validate_every_n_epochs=cfg.get('validate_every_n_epochs', 1),
        # Pass explicit key from cfg (not get_config()) so sweeps/ensemble
        # delays do not all collapse into the default-config folder.
        checkpoint_key=ckpt_key,
        embed_dim=cfg['embed_dim'],
        max_seq_len=cfg['max_seq_len'],
        recent_checkpoints=cfg.get(
            'recent_checkpoints', cfg.get('max_checkpoints', 3)
        ),
    )
    if show_banner:
        print("\n" + "=" * 60)
        print("ELEMENT TRAINING")
        print("=" * 60)
        print(f"  telepred_delay: {delay}")
        print(f"  checkpoint_key: {ckpt_key}")
        print(f"  epochs: {num_epochs if num_epochs is not None else cfg['epochs']}")
        print(f"  device: {trainer.device}")
        print("=" * 60 + "\n")

    _orig_show = None
    if suppress_history_plot:
        _orig_show = ResumableTrainer._show_history_if_available
        ResumableTrainer._show_history_if_available = lambda self: False

    try:
        epochs = (
            float(cfg['epochs']) if num_epochs is None else float(num_epochs)
        )
        do_resume = cfg.get('resume', True) if resume is None else bool(resume)
        train_outcome = trainer.train(
            num_epochs=epochs,
            resume=do_resume,
            extend=bool(extend),
            extend_epochs=float(extend_epochs) if extend else 0.0,
        )
    finally:
        if _orig_show is not None:
            ResumableTrainer._show_history_if_available = _orig_show

    tr = list(trainer.history.train_losses)
    va = list(trainer.history.val_losses)

    def _finite_min(values):
        nums = [
            float(v)
            for v in values
            if v is not None and math.isfinite(float(v))
        ]
        return min(nums) if nums else None

    best_v = trainer.state_manager.best_val_loss
    if best_v is None:
        best_v = _finite_min(va)

    return {
        'checkpoint_key': ckpt_key,
        'telepred_delay': delay,
        'dataset_signature': dataset.cache_key,
        'epochs_ran': len(tr),
        'final_train_loss': tr[-1] if tr else None,
        'best_train_loss': _finite_min(tr),
        'final_val_loss': va[-1] if va and va[-1] is not None else None,
        'best_val_loss': best_v,
        'train_outcome': train_outcome or 'trained',
        'skipped_finished': train_outcome == 'skipped_finished',
        'trainer': trainer,
        'model': model,
    }
