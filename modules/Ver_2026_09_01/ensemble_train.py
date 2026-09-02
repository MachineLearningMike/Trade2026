# ensemble_train.py
# Orchestrate TelePred delay-bank training and ensemble hyperparameter sweeps.

from __future__ import annotations

import csv
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from telepred import parse_delay_list
from train_run import train_one_run

DEFAULT_PATCH_SIZES = (5, 10, 20)
DEFAULT_NUM_LAYERS = (2, 4, 6)

CSV_FIELDS = [
    'trial_id',
    'timestamp',
    'status',
    'model_type',
    'label',
    'patch_size',
    'num_layers',
    'delay',
    'num_heads',
    'embed_dim',
    'max_seq_len',
    'epochs_planned',
    'epochs_ran',
    'best_train_loss',
    'final_train_loss',
    'best_val_loss',
    'final_val_loss',
    'checkpoint_key',
    'dataset_signature',
    'error',
    'notes',
]


def project_root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def ensemble_csv_path(
    label: str = 'yield',
    model_type: str = 'regression',
) -> Path:
    """CSV for ``--ensemble`` (fixed hyperparams, bank over delays)."""
    lab = str(label or 'yield').lower().strip() or 'yield'
    mt = str(model_type or 'regression').lower().strip() or 'regression'
    return (
        project_root_dir()
        / '_save'
        / f'ensemble_{lab}_{mt}'
        / 'results.csv'
    )


def ensemble_sweep_csv_path(
    label: str = 'yield',
    model_type: str = 'regression',
) -> Path:
    """CSV for ``--ensemble-sweep`` (hparam grid × delay bank)."""
    lab = str(label or 'yield').lower().strip() or 'yield'
    mt = str(model_type or 'regression').lower().strip() or 'regression'
    return (
        project_root_dir()
        / '_save'
        / f'ensemble_{lab}_{mt}_sweep'
        / 'results.csv'
    )


def _append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in CSV_FIELDS})


def _finite_min(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [
        float(v) for v in values if v is not None and math.isfinite(float(v))
    ]
    return min(nums) if nums else None


def run_ensemble_train(
    config: dict,
    *,
    delays: Optional[Sequence[int]] = None,
    get_dataset=None,
    num_epochs: Optional[float] = None,
    extend: bool = False,
    extend_epochs: float = 0.0,
    continue_on_error: bool = True,
    probe_weights: bool = True,
    suppress_history_plot: bool = True,
    write_csv: bool = True,
    csv_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Train one TelePred(d) element per delay (separate dataset + checkpoint).

    ``get_dataset`` defaults to ``dataset_build.get_dataset``.
    When ``write_csv`` is True (default for ``--ensemble``), appends one row
    per delay to ``_save/ensemble_<label>_<model_type>/results.csv``.
    ``--ensemble-sweep`` passes ``write_csv=False`` and writes its own CSV.
    """
    if get_dataset is None:
        from dataset_build import get_dataset as get_dataset

    if delays is None:
        delays = parse_delay_list(
            config.get('telepred_delays', [config.get('telepred_delay', 0)])
        )
    else:
        delays = parse_delay_list(list(delays))

    label = str(config.get('label', 'yield')).lower()
    mt = str(config.get('model_type', 'regression')).lower()
    from task_base import assert_compatible
    from ensemble_spec import EnsembleModelSpec, describe_ensemble
    assert_compatible(mt, label, ensemble_kind='telepred')
    telepred_spec = EnsembleModelSpec.telepred(mt, label, delays)
    if write_csv and csv_path is None:
        csv_path = ensemble_csv_path(label=label, model_type=mt)

    print("\n" + "=" * 60)
    print("ENSEMBLE TRAINING (TelePred delay bank)")
    print("=" * 60)
    print(describe_ensemble(telepred_spec))
    print(f"  continue_on_error: {continue_on_error}")
    if write_csv and csv_path is not None:
        print(f"  CSV: {csv_path}")
    print("=" * 60 + "\n")

    results: List[Dict[str, Any]] = []
    for i, d in enumerate(delays):
        print("\n" + "-" * 60)
        print(f"ENSEMBLE ELEMENT {i + 1}/{len(delays)}: telepred_delay={d}")
        print("-" * 60)
        elem_cfg = dict(config)
        elem_cfg['telepred_delay'] = int(d)
        ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        try:
            from task_base import get_task_class
            task = get_task_class(elem_cfg.get('label', 'trend'))
            dataset, total_features = get_dataset(elem_cfg, task=task)
            # Extend only when this delay already has weights (same policy as vae_sweep).
            from trainer import generate_checkpoint_key
            ckpt_key = generate_checkpoint_key(elem_cfg, dataset.cache_key)
            key_dir = Path(elem_cfg['checkpoint_dir']) / ckpt_key
            has_prior = (
                key_dir.exists()
                and (key_dir / 'training_state.json').exists()
                and any(key_dir.glob('*.pt'))
            )
            do_extend = bool(extend) and has_prior and abs(float(extend_epochs)) > 1e-15
            if has_prior:
                print(f"  ↻ Resume candidates in {key_dir.name}/ "
                      f"({len(list(key_dir.glob('*.pt')))} .pt files)")
            else:
                print(f"  ○ No prior state+weights for key {ckpt_key} — fresh run")
            out = train_one_run(
                elem_cfg,
                dataset,
                total_features,
                task=task,
                num_epochs=num_epochs,
                # Always try resume when artifacts exist; trainer no-ops if
                # already finished and plan is already satisfied.
                resume=True if has_prior else elem_cfg.get('resume', True),
                extend=do_extend,
                extend_epochs=float(extend_epochs) if do_extend else 0.0,
                probe_weights=probe_weights,
                show_banner=True,
                suppress_history_plot=suppress_history_plot,
            )
            out['telepred_delay'] = int(d)
            out['delay'] = int(d)
            skipped = bool(out.get('skipped_finished'))
            out['status'] = 'skipped_finished' if skipped else 'ok'
            out['error'] = ''
            results.append(out)
            if skipped:
                print(
                    f"↷ delay={d} already finished — skip CSV "
                    f"(key={out['checkpoint_key']})"
                )
            else:
                print(
                    f"✓ delay={d} key={out['checkpoint_key']} "
                    f"val_best={out.get('best_val_loss')}"
                )
            # Do not re-log members that were already finished (no new training).
            if write_csv and csv_path is not None and not skipped:
                _append_csv_row(
                    csv_path,
                    {
                        'trial_id': i + 1,
                        'timestamp': ts,
                        'status': 'ok',
                        'model_type': mt,
                        'label': label,
                        'patch_size': elem_cfg.get('patch_size', ''),
                        'num_layers': elem_cfg.get('num_layers', ''),
                        'delay': int(d),
                        'num_heads': elem_cfg.get('num_heads', ''),
                        'embed_dim': elem_cfg.get('embed_dim', ''),
                        'max_seq_len': elem_cfg.get('max_seq_len', ''),
                        'epochs_planned': num_epochs
                        if num_epochs is not None
                        else elem_cfg.get('epochs', ''),
                        'epochs_ran': out.get('epochs_ran', ''),
                        'best_train_loss': out.get('best_train_loss', ''),
                        'final_train_loss': out.get('final_train_loss', ''),
                        'best_val_loss': out.get('best_val_loss', ''),
                        'final_val_loss': out.get('final_val_loss', ''),
                        'checkpoint_key': out.get('checkpoint_key', ''),
                        'dataset_signature': out.get('dataset_signature', ''),
                        'error': '',
                        'notes': f"ensemble delay bank; extend={do_extend}",
                    },
                )
        except Exception as e:
            err = {
                'status': 'failed',
                'telepred_delay': int(d),
                'delay': int(d),
                'checkpoint_key': '',
                'dataset_signature': '',
                'epochs_ran': 0,
                'best_val_loss': None,
                'final_val_loss': None,
                'best_train_loss': None,
                'final_train_loss': None,
                'error': f'{type(e).__name__}: {e}',
                'notes': traceback.format_exc(limit=6),
            }
            results.append(err)
            print(f"✗ delay={d} FAILED: {err['error']}")
            if write_csv and csv_path is not None:
                _append_csv_row(
                    csv_path,
                    {
                        'trial_id': i + 1,
                        'timestamp': ts,
                        'status': 'failed',
                        'model_type': mt,
                        'label': label,
                        'patch_size': config.get('patch_size', ''),
                        'num_layers': config.get('num_layers', ''),
                        'delay': int(d),
                        'num_heads': config.get('num_heads', ''),
                        'epochs_planned': num_epochs
                        if num_epochs is not None
                        else config.get('epochs', ''),
                        'error': err['error'],
                        'notes': err.get('notes', ''),
                    },
                )
            if not continue_on_error:
                raise

    n_ok = sum(1 for r in results if r.get('status') == 'ok')
    print("\n" + "=" * 60)
    print(f"ENSEMBLE TRAINING DONE: {n_ok}/{len(results)} elements ok")
    for r in results:
        print(
            f"  d={r.get('telepred_delay', r.get('delay'))}: {r.get('status')} "
            f"key={r.get('checkpoint_key') or '—'} "
            f"val_best={r.get('best_val_loss')}"
        )
    if write_csv and csv_path is not None:
        print(f"  Results CSV: {csv_path}")
    print("=" * 60)
    return results


def run_ensemble_sweep(
    config: dict,
    *,
    delays: Optional[Sequence[int]] = None,
    patch_sizes: Sequence[int] = DEFAULT_PATCH_SIZES,
    num_layers_list: Sequence[int] = DEFAULT_NUM_LAYERS,
    epochs_per_trial: int = 1,
    extend_epochs: float = 1.0,
    csv_path: Optional[Path] = None,
    get_dataset=None,
    continue_on_error: bool = True,
) -> Path:
    """
    Outer hyperparameter grid; each trial trains the full TelePred delay bank.
    """
    if get_dataset is None:
        from dataset_build import get_dataset as get_dataset

    cfg = dict(config)
    cfg['resume'] = True
    mt = str(cfg.get('model_type', 'regression')).lower()
    label = str(cfg.get('label', 'yield')).lower()
    if delays is None:
        delays = parse_delay_list(
            cfg.get('telepred_delays', [cfg.get('telepred_delay', 0)])
        )
    else:
        delays = parse_delay_list(list(delays))

    if csv_path is None:
        csv_path = ensemble_sweep_csv_path(label=label, model_type=mt)

    grid = [
        (int(p), int(L)) for p in patch_sizes for L in num_layers_list
    ]
    n_heads = int(cfg.get('num_heads', 4))
    n_latent = int(cfg.get('n_latent_macros', 30))

    print("\n" + "=" * 60)
    print("ENSEMBLE HYPERPARAMETER SWEEP")
    print("=" * 60)
    print(f"  Outer grid: patch_size={list(patch_sizes)} × "
          f"num_layers={list(num_layers_list)}")
    print(f"  Inner bank delays: {list(delays)}")
    print(f"  Trials: {len(grid)} (× {len(delays)} elements)")
    print(f"  epochs_per_trial: {epochs_per_trial}")
    print(f"  extend_epochs: {extend_epochs:g}")
    print(f"  CSV: {csv_path}")
    print("=" * 60 + "\n")

    # Probe F / n_market once with delay=0 dataset (layout independent of d).
    from task_base import get_task_class
    probe_cfg = dict(cfg)
    probe_cfg['telepred_delay'] = 0
    dataset0, total_features = get_dataset(
        probe_cfg, task=get_task_class(probe_cfg.get('label', 'trend'))
    )
    n_market = int(dataset0.n_market_features)
    # Drop large probe; each delay rebuilds its own dataset inside ensemble.
    del dataset0

    trial_id = 0
    for patch_size, num_layers in grid:
        trial_id += 1
        embed_dim = (n_market + n_latent) * int(patch_size)
        ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        print(
            f"\n▶ Ensemble trial {trial_id}/{len(grid)}: "
            f"patch_size={patch_size} num_layers={num_layers}"
        )
        if embed_dim % n_heads != 0:
            for d in delays:
                _append_csv_row(
                    csv_path,
                    {
                        'trial_id': trial_id,
                        'timestamp': ts,
                        'status': 'skipped_invalid',
                        'model_type': mt,
                        'label': label,
                        'patch_size': patch_size,
                        'num_layers': num_layers,
                        'delay': d,
                        'num_heads': n_heads,
                        'embed_dim': embed_dim,
                        'error': (
                            f'embed_dim={embed_dim} not divisible by '
                            f'num_heads={n_heads}'
                        ),
                    },
                )
            print(f"  ⏭ SKIP invalid embed_dim={embed_dim}")
            continue

        trial_cfg = dict(cfg)
        trial_cfg['patch_size'] = int(patch_size)
        trial_cfg['num_layers'] = int(num_layers)
        trial_cfg['embed_dim'] = embed_dim
        trial_cfg['telepred_delays'] = list(delays)

        # Per-element resume/extend decided inside run_ensemble_train.
        results = run_ensemble_train(
            trial_cfg,
            delays=delays,
            get_dataset=get_dataset,
            num_epochs=float(epochs_per_trial),
            extend=abs(float(extend_epochs)) > 1e-15,
            extend_epochs=float(extend_epochs),
            continue_on_error=continue_on_error,
            probe_weights=(trial_id == 1),
            suppress_history_plot=True,
            write_csv=False,  # this sweep writes its own CSV below
        )
        for r in results:
            _append_csv_row(
                csv_path,
                {
                    'trial_id': trial_id,
                    'timestamp': ts,
                    'status': r.get('status', 'ok'),
                    'model_type': mt,
                    'label': label,
                    'patch_size': patch_size,
                    'num_layers': num_layers,
                    'delay': r.get('telepred_delay', r.get('delay', '')),
                    'num_heads': n_heads,
                    'embed_dim': embed_dim,
                    'max_seq_len': trial_cfg.get('max_seq_len', ''),
                    'epochs_planned': epochs_per_trial,
                    'epochs_ran': r.get('epochs_ran', ''),
                    'best_train_loss': r.get('best_train_loss', ''),
                    'final_train_loss': r.get('final_train_loss', ''),
                    'best_val_loss': r.get('best_val_loss', ''),
                    'final_val_loss': r.get('final_val_loss', ''),
                    'checkpoint_key': r.get('checkpoint_key', ''),
                    'dataset_signature': r.get('dataset_signature', ''),
                    'error': r.get('error', ''),
                    'notes': r.get('notes', ''),
                },
            )
        print(f"  → appended {len(results)} CSV row(s) → {csv_path}")

    print("\n" + "=" * 60)
    print("ENSEMBLE SWEEP COMPLETE")
    print(f"  Results: {csv_path}")
    print("=" * 60)
    return csv_path
