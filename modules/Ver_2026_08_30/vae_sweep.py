# vae_sweep.py
# Sequential hyperparameter scout: patch_size × num_layers, CSV logging.
# Works for config model_type in {'regression', 'vae'}.

from __future__ import annotations

import csv
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.optim as optim

from labels import bind_label_loss
from model_vae import build_market_model, resolve_max_seq_len, wrap_model_loss
from trainer import ResumableTrainer, generate_checkpoint_key

DEFAULT_PATCH_SIZES = (5, 10, 20)
DEFAULT_NUM_LAYERS = (2, 4, 6)

CSV_FIELDS = [
    'trial_id',
    'timestamp',
    'status',
    'model_type',
    'patch_size',
    'num_layers',
    'delay',
    'num_heads',
    'ff_dim_multiplier',
    'vae_latent_dim',
    'n_latent_macros',
    'embed_dim',
    'max_seq_len',
    'reconstruction_loss_weight',
    'vae_kl_weight',
    'epochs_planned',
    'epochs_ran',
    'best_train_loss',
    'final_train_loss',
    'best_val_loss',
    'final_val_loss',
    'checkpoint_key',
    'error',
    'notes',
]


def project_root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def sweep_csv_path(model_type: str = 'vae') -> Path:
    """CSV under ``_save/<model_type>_sweep/results.csv``."""
    mt = str(model_type or 'regression').lower().strip() or 'regression'
    return project_root_dir() / '_save' / f'{mt}_sweep' / 'results.csv'


def _append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in CSV_FIELDS})


def _finite_min(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return min(nums) if nums else None


def _trial_grid(
    patch_sizes: Iterable[int],
    num_layers_list: Iterable[int],
) -> List[Tuple[int, int]]:
    return [(int(p), int(L)) for p in patch_sizes for L in num_layers_list]


def run_hparam_sweep(
    config: dict,
    dataset,
    total_features,
    *,
    model_type: Optional[str] = None,
    patch_sizes: Sequence[int] = DEFAULT_PATCH_SIZES,
    num_layers_list: Sequence[int] = DEFAULT_NUM_LAYERS,
    epochs_per_trial: int = 1,
    extend_epochs: float = 1.0,
    csv_path: Optional[Path] = None,
) -> Path:
    """
    Train / extend one trial per (patch_size, num_layers) for ``model_type``.

    Dataset is shared across trials. Each architecture keeps its own checkpoint
    key (hash includes model_type). With ``extend_epochs>0`` and existing
    finished trials, training continues and a **new** CSV row is appended.
    """
    from data import ThreadedDataLoader
    from labels import resolve_model_past_len

    cfg = dict(config)
    mt = str(
        model_type
        if model_type is not None
        else cfg.get('model_type', 'regression')
    ).lower().strip()
    if mt not in ('regression', 'vae'):
        raise ValueError(
            f"model_type must be 'regression' or 'vae', got {mt!r}"
        )
    cfg['model_type'] = mt
    cfg['resume'] = True

    if csv_path is None:
        csv_path = sweep_csv_path(mt)

    from telepred import resolve_past_window
    past_window = int(resolve_past_window(cfg))
    F = (
        total_features.shape[0]
        if hasattr(total_features, 'shape')
        else len(total_features)
    )
    n_market = int(dataset.n_market_features)
    n_macro = int(dataset.n_macro_features)
    n_latent = int(cfg['n_latent_macros'])
    n_heads = int(cfg['num_heads'])
    _model_past_len = int(
        getattr(dataset, 'model_past_len', resolve_model_past_len(cfg))
    )
    cfg['output_dim'] = int(dataset.output_dim)
    extend_epochs = float(extend_epochs)

    grid = _trial_grid(patch_sizes, num_layers_list)
    print("\n" + "=" * 60)
    print(f"{mt.upper()} HYPERPARAMETER SWEEP")
    print("=" * 60)
    print(f"  model_type: {mt}")
    print(f"  Grid: patch_size={list(patch_sizes)} × num_layers={list(num_layers_list)}")
    print(f"  Trials: {len(grid)} (invalid embed_dim%%num_heads skipped)")
    print(f"  Base epochs / trial: {epochs_per_trial}")
    print(f"  Extend epochs:       {extend_epochs:g} (resume existing keys)")
    print(f"  CSV: {csv_path}")
    print(f"  Device: {cfg.get('device')}")
    if mt == 'vae':
        print(
            f"  Loss: {cfg.get('reconstruction_loss_weight', 1.0):g}·MSE + "
            f"{cfg.get('vae_kl_weight', 0.1):g}·KL (no yield label)"
        )
    else:
        print(
            f"  Loss: supervised label={cfg.get('label', 'trend')} "
            f"(yield_target={cfg.get('yield_target', 'both')})"
        )
    print("=" * 60 + "\n")

    _orig_show = ResumableTrainer._show_history_if_available
    ResumableTrainer._show_history_if_available = lambda self: False

    trial_id = 0
    try:
        for patch_size, num_layers in grid:
            trial_id += 1
            ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            row: Dict[str, Any] = {
                'trial_id': trial_id,
                'timestamp': ts,
                'status': 'ok',
                'model_type': mt,
                'patch_size': patch_size,
                'num_layers': num_layers,
                'delay': int(cfg.get('telepred_delay', 0) or 0),
                'num_heads': n_heads,
                'ff_dim_multiplier': cfg.get('ff_dim_multiplier'),
                'vae_latent_dim': (
                    cfg.get('vae_latent_dim', 64) if mt == 'vae' else ''
                ),
                'n_latent_macros': n_latent,
                'reconstruction_loss_weight': (
                    cfg.get('reconstruction_loss_weight', 1.0) if mt == 'vae' else ''
                ),
                'vae_kl_weight': (
                    cfg.get('vae_kl_weight', 0.1) if mt == 'vae' else ''
                ),
                'epochs_planned': epochs_per_trial,
                'epochs_ran': 0,
                'best_train_loss': '',
                'final_train_loss': '',
                'best_val_loss': '',
                'final_val_loss': '',
                'checkpoint_key': '',
                'error': '',
                'notes': '',
            }

            embed_dim = (n_market + n_latent) * int(patch_size)
            row['embed_dim'] = embed_dim
            if embed_dim % n_heads != 0:
                row['status'] = 'skipped_invalid'
                row['error'] = (
                    f'embed_dim={embed_dim} not divisible by num_heads={n_heads}'
                )
                print(
                    f"⏭ Trial {trial_id}/{len(grid)}: "
                    f"patch_size={patch_size} num_layers={num_layers} — SKIP "
                    f"({row['error']})"
                )
                _append_csv_row(csv_path, row)
                continue

            trial_cfg = dict(cfg)
            trial_cfg['patch_size'] = int(patch_size)
            trial_cfg['num_layers'] = int(num_layers)
            trial_cfg['embed_dim'] = embed_dim
            trial_cfg['max_seq_len'] = resolve_max_seq_len(trial_cfg, past_window)
            row['max_seq_len'] = trial_cfg['max_seq_len']

            ckpt_key = generate_checkpoint_key(trial_cfg, dataset.cache_key)
            row['checkpoint_key'] = ckpt_key

            print(
                f"\n▶ Trial {trial_id}/{len(grid)} [{mt}]: "
                f"patch_size={patch_size} num_layers={num_layers} "
                f"embed_dim={embed_dim} max_seq_len={trial_cfg['max_seq_len']}"
            )
            print(f"  checkpoint_key={ckpt_key}")

            try:
                model = build_market_model(
                    trial_cfg,
                    in_features=F,
                    n_market_features=n_market,
                    n_macro_features=n_macro,
                    n_max_past_horizon=_model_past_len,
                    max_seq_len=trial_cfg['max_seq_len'],
                )
                device = (
                    torch.device(trial_cfg['device'])
                    if trial_cfg.get('device')
                    else torch.device('cpu')
                )
                model = model.to(device)

                base_loss = bind_label_loss(
                    dataset.label_mode,
                    trial_cfg,
                    yield_target=getattr(
                        dataset, 'yield_target', trial_cfg.get('yield_target', 'both')
                    ),
                    latent_distribution=getattr(
                        dataset,
                        'latent_distribution',
                        trial_cfg.get('latent_distribution', 'normal'),
                    ),
                )
                loss_fn = wrap_model_loss(model, base_loss, trial_cfg)

                train_loader = ThreadedDataLoader(
                    dataset.train_dataset,
                    batch_size=trial_cfg['batch_size'],
                    shuffle=True,
                )
                val_loader = ThreadedDataLoader(
                    dataset.val_dataset,
                    batch_size=trial_cfg['validate_batch_size'],
                    shuffle=False,
                )
                optimizer = optim.Adam(
                    model.parameters(), lr=trial_cfg['learning_rate']
                )
                trainer = ResumableTrainer(
                    dataset_signature=dataset.cache_key,
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    dataloader=train_loader,
                    save_dir=trial_cfg['checkpoint_dir'],
                    device=trial_cfg['device'],
                    save_every_n_minutes=trial_cfg['save_every_n_minutes'],
                    val_loader=val_loader,
                    enable_validation=True,
                    validate_every_n_epochs=1,
                    # Must pass trial key: otherwise trainer re-hashes get_config()
                    # (default patch/layers) and every sweep trial shares one folder.
                    checkpoint_key=ckpt_key,
                    embed_dim=trial_cfg['embed_dim'],
                    max_seq_len=trial_cfg['max_seq_len'],
                    recent_checkpoints=trial_cfg.get(
                        'recent_checkpoints', trial_cfg.get('max_checkpoints', 3)
                    ),
                )

                key_dir = Path(trial_cfg['checkpoint_dir']) / ckpt_key
                has_prior = key_dir.exists() and any(key_dir.glob('*.pt'))
                do_extend = has_prior and abs(extend_epochs) > 1e-15
                if do_extend:
                    print(
                        f"  Resuming prior checkpoint; extending by "
                        f"{extend_epochs:g} epoch(s)"
                    )
                trainer.train(
                    num_epochs=float(epochs_per_trial),
                    resume=has_prior,
                    extend=do_extend,
                    extend_epochs=extend_epochs if do_extend else 0.0,
                )

                tr = list(trainer.history.train_losses)
                va = list(trainer.history.val_losses)
                row['epochs_ran'] = len(tr)
                row['epochs_planned'] = (
                    trainer.state_manager.get_total_epochs()
                    or epochs_per_trial
                )
                row['final_train_loss'] = tr[-1] if tr else ''
                row['best_train_loss'] = _finite_min(tr) if tr else ''
                row['final_val_loss'] = (
                    va[-1] if va and va[-1] is not None else ''
                )
                best_v = trainer.state_manager.best_val_loss
                if best_v is None:
                    best_v = _finite_min(va)
                row['best_val_loss'] = best_v if best_v is not None else ''
                row['status'] = 'ok'
                row['notes'] = (
                    f"F={F} past_window={past_window} "
                    f"train_n={len(dataset.train_indices)} "
                    f"val_n={len(dataset.val_indices)} "
                    f"extend={extend_epochs:g}"
                )
                print(
                    f"  ✓ epochs_ran={row['epochs_ran']}  "
                    f"train_final={row['final_train_loss']}  "
                    f"val_best={row['best_val_loss']}"
                )
            except Exception as e:
                row['status'] = 'failed'
                row['error'] = f'{type(e).__name__}: {e}'
                row['notes'] = traceback.format_exc(limit=4)
                print(f"  ✗ FAILED: {row['error']}")

            _append_csv_row(csv_path, row)
            print(f"  → appended CSV row → {csv_path}")

    finally:
        ResumableTrainer._show_history_if_available = _orig_show

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print(f"  model_type: {mt}")
    print(f"  Results: {csv_path}")
    _print_csv_leaderboard(csv_path)
    print("=" * 60)
    return csv_path


def run_vae_hparam_sweep(
    config: dict,
    dataset,
    total_features,
    *,
    patch_sizes: Sequence[int] = DEFAULT_PATCH_SIZES,
    num_layers_list: Sequence[int] = DEFAULT_NUM_LAYERS,
    epochs_per_trial: int = 1,
    extend_epochs: float = 1.0,
    csv_path: Optional[Path] = None,
    run_name: str = 'vae_sweep',
) -> Path:
    """Backward-compatible alias: force model_type=vae. """
    if csv_path is None and run_name == 'vae_sweep':
        csv_path = None  # run_hparam_sweep picks vae_sweep path
    elif csv_path is None:
        csv_path = project_root_dir() / '_save' / run_name / 'results.csv'
    return run_hparam_sweep(
        config,
        dataset,
        total_features,
        model_type='vae',
        patch_sizes=patch_sizes,
        num_layers_list=num_layers_list,
        epochs_per_trial=epochs_per_trial,
        extend_epochs=extend_epochs,
        csv_path=csv_path,
    )


def _print_csv_leaderboard(csv_path: Path, top_k: int = 5) -> None:
    if not csv_path.exists():
        return
    rows: List[Dict[str, str]] = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get('status') != 'ok':
                continue
            try:
                float(r.get('best_val_loss') or 'nan')
            except ValueError:
                continue
            rows.append(r)
    rows.sort(key=lambda r: float(r['best_val_loss']))
    print("  Leaderboard (best_val_loss, ok trials only):")
    if not rows:
        print("    (no successful trials yet)")
        return
    for i, r in enumerate(rows[:top_k], 1):
        print(
            f"    {i}. [{r.get('model_type', '?')}] "
            f"patch={r['patch_size']} layers={r['num_layers']}  "
            f"val={r['best_val_loss']}  train={r['final_train_loss']}  "
            f"key={r['checkpoint_key']}"
        )
