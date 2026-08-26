# TRAIN.py
# Main training script - Clean version with all logic in trainer

import sys
from pathlib import Path

import argparse
import json
import shutil

from config import update_config
from dataset_build import get_dataset, project_root_dir
from labels import sync_output_dim
from telepred import parse_delay_list
from trainer import resolve_checkpoint_identity
from train_run import train_one_run

# Compatibility re-export for callers that still do ``from TRAIN import get_dataset``.
__all__ = ['main', 'get_dataset', 'project_root_dir']


def _resolve_report_dir(report_name: str, *, flag: str = '--report'):
    """
    Validate a folder *name* and return ``<project_root>/_save/<name>``.

    Returns ``(report_dir, error_message)``. On success ``error_message`` is None.
    """
    name = str(report_name).strip()
    if not name:
        return None, (
            f"⚠ {flag} requires a non-empty directory name, "
            f"e.g. {flag} 'my run'."
        )
    if name in ('.', '..') or '/' in name or '\\' in name:
        return None, (
            f"⚠ {flag} takes a directory name only (no path separators). "
            f"Example: {flag} 'dir name'  →  <project_root>/_save/dir name/"
        )
    project_root = project_root_dir()
    report_dir = (project_root / '_save' / name).resolve()
    save_root = (project_root / '_save').resolve()
    if save_root not in report_dir.parents and report_dir != save_root:
        return None, f"⚠ Refusing to write outside {save_root}: {report_dir}"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir, None


def _export_validation_epoch_plots(source_dir: Path, report_dir: Path) -> dict:
    """
    Write full + p0_30 validation curves for every epoch under ``source_dir``.

    Same artifacts as ``--report`` / ``--ensemble-report`` member folders.
    """
    from trainer import ValidationEpochLoss, list_validation_loss_epochs

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    epochs = list_validation_loss_epochs(source_dir)
    if not epochs:
        return {
            'epochs': [],
            'n_full': 0,
            'n_slice': 0,
            'n_missing': 0,
            'max_epoch': 0,
        }

    max_epoch = max(epochs)
    epoch_width = max(3, len(str(max_epoch)))
    n_full = 0
    n_slice = 0
    n_missing = 0
    for e in range(1, max_epoch + 1):
        ve = ValidationEpochLoss.load(source_dir, e)
        if ve is None or not ve.batch_losses:
            print(f"⚠ Skipping epoch {e}: no validation loss curve")
            n_missing += 1
            continue

        e_tag = f'{e:0{epoch_width}d}'
        full_path = report_dir / f'validation_e{e_tag}.png'
        fig = ve.plot(
            save_path=str(full_path),
            percent_lo=0.0,
            percent_hi=100.0,
            show=False,
        )
        if fig is not None:
            n_full += 1

        slice_path = report_dir / f'validation_e{e_tag}_p0_30.png'
        fig30 = ve.plot(
            save_path=str(slice_path),
            percent_lo=0.0,
            percent_hi=30.0,
            show=False,
        )
        if fig30 is not None:
            n_slice += 1

    return {
        'epochs': epochs,
        'n_full': n_full,
        'n_slice': n_slice,
        'n_missing': n_missing,
        'max_epoch': max_epoch,
        'epoch_width': epoch_width,
    }


def main(config):
    """Main training function."""
    
    # Ensure price_feature_idx exists
    if 'price_feature_idx' not in config:
        config['price_feature_idx'] = 0
    
    parser = argparse.ArgumentParser(description='Resilient Trading Training (regression / VAE)')
    parser.add_argument('--epochs', type=int, default=config['epochs'], help='Number of epochs')
    parser.add_argument(
        '--save-every', type=float, default=config['save_every_n_minutes'],
        help='Save checkpoint every N wall-clock minutes',
    )
    parser.add_argument(
        '--clean', action='store_true',
        help=(
            'Delete checkpoints for the current run identity, then train fresh. '
            'Single-model: one checkpoint key. '
            'With --error-ensemble: all stage keys model_0..model_{n-1}. '
            'Other key folders under checkpoint_dir are preserved.'
        ),
    )
    parser.add_argument(
        '--extend', type=float, default=None,
        help=(
            'Adjust planned total epochs by N (float OK). '
            'Positive adds beyond max(plan, progress); negative shortens the plan '
            '(not below current progress). Examples: --extend 10, --extend 1.5, --extend -5'
        ),
    )
    parser.add_argument('--trend', action='store_true', default=False,
                       help='Generate and visualize trends without training')
    parser.add_argument(
        '--history', nargs='?', const='current', default=None,
        help=(
            'Visualize training history without loading market data. '
            'Use --history (current config key), --history all, or '
            '--history <checkpoint_key> (e.g. 541f3423a2cec4a6).'
        ),
    )
    parser.add_argument(
        '--loss', nargs='?', const='current', default=None,
        help=(
            'Display per-minibatch loss of the latest training epoch '
            '(no market data load). '
            'Use --loss (current config key), --loss all, or '
            '--loss <checkpoint_key>.'
        ),
    )
    parser.add_argument(
        '--validation', nargs='*', default=None, metavar='E',
        help=(
            'Plot per-batch validation loss for epoch E (no market data load), '
            'including moving average and running loss. Curves come from '
            'validation_loss_epoch_<E>.json written during training. '
            'Optional percent slice of the saved batch series: '
            '--validation → latest epoch, [0%%, 100%%]; '
            '--validation E → epoch E, [0%%, 100%%]; '
            '--validation E V2 → epoch E, [0%%, V2%%]; '
            '--validation E V1 V2 → epoch E, [V1%%, V2%%].'
        ),
    )
    parser.add_argument(
        '--report', type=str, default=None, metavar='DIR_NAME',
        help=(
            "Batch-export validation loss plots for every saved epoch under the "
            "current checkpoint key. Canonical form: --report 'dir name'. "
            "Images are written to <project_root>/_save/<dir_name>/ as "
            "validation_e001.png (= --validation e) and "
            "validation_e001_p0_30.png (= --validation e 30), "
            "with zero-padded epoch ids for alphabetical order. "
            "No interactive plots."
        ),
    )
    parser.add_argument(
        '--ensemble-report', type=str, default=None, metavar='DIR_NAME',
        help=(
            "Like --report, but for every TelePred delay trained by --ensemble. "
            "Canonical form: --ensemble-report 'folder name'. "
            "Writes <project_root>/_save/<folder name>/d<delay>/validation_eXXX.png "
            "(and _p0_30.png) for each delay in config/--telepred-delays. "
            "No interactive plots."
        ),
    )
    parser.add_argument(
        '--checkpoint-key', type=str, default=None, metavar='KEY',
        help=(
            'Force a checkpoint folder under checkpoint_dir (e.g. a prior '
            'yield transformer run). Use with --validate / --test / '
            '--validate-yield. Architecture must still match the weights '
            '(set --model-type regression for yield models).'
        ),
    )
    parser.add_argument(
        '--vae-compare', type=float, nargs='*', default=None, metavar='V',
        help=(
            'Compare per-sample VAE loss vs yield loss on sorted val samples '
            '(scatter + dual series + Pearson/Spearman). Requires --vae-key '
            'and --yield-key. Optional percent slice like --validate: '
            '--vae-compare | --vae-compare V | --vae-compare V1 V2.'
        ),
    )
    parser.add_argument(
        '--vae-key', type=str, default=None, metavar='KEY',
        help='VAE checkpoint folder name for --vae-compare.',
    )
    parser.add_argument(
        '--yield-key', type=str, default=None, metavar='KEY',
        help='Yield (regression) checkpoint folder name for --vae-compare.',
    )
    parser.add_argument(
        '--sweep', action='store_true', default=False,
        help=(
            'Hyperparameter scout over patch_size × num_layers '
            '(default 5/10/20 × 2/4/6) for the designated model_type '
            '(config or --model-type). Appends to '
            '<project_root>/_save/<model_type>_sweep/results.csv. '
            'Resumes existing trial keys; use --sweep-extend to add epochs.'
        ),
    )
    parser.add_argument(
        '--vae-sweep', action='store_true', default=False,
        help='Alias for --sweep --model-type vae.',
    )
    parser.add_argument(
        '--sweep-epochs', type=int, default=1, metavar='N',
        help='Base epoch plan per trial for --sweep (default 1).',
    )
    parser.add_argument(
        '--sweep-extend', type=float, default=1.0, metavar='N',
        help=(
            'If a trial checkpoint already exists, add this many epochs '
            '(default 1) and append a new row to results.csv. '
            'Fresh trials still use --sweep-epochs only.'
        ),
    )
    parser.add_argument(
        '--telepred-delay', type=int,
        default=int(config.get('telepred_delay', 0) or 0),
        metavar='D',
        help='TelePred delay d (bars): input ends at bot_time-d (default 0)',
    )
    parser.add_argument(
        '--telepred-delays', type=str, default=None, metavar='LIST',
        help=(
            'Comma-separated delays for --ensemble / --ensemble-sweep '
            '(default: config telepred_delays)'
        ),
    )
    parser.add_argument(
        '--ensemble', action='store_true', default=False,
        help=(
            'Train the full TelePred delay bank (one checkpoint per d). '
            'Appends rows to '
            '<project_root>/_save/ensemble_<label>_<model_type>/results.csv'
        ),
    )
    parser.add_argument(
        '--ensemble-sweep', action='store_true', default=False,
        help=(
            'Hyperparameter scout: outer patch_size×num_layers; '
            'each trial trains the full TelePred delay bank'
        ),
    )
    parser.add_argument(
        '--error-ensemble', action='store_true', default=False,
        help=(
            'Train ErrorEnsemble residual stages sequentially: '
            'model_0 fits y; model_i fits e_{i-1}=e_{i-2}-y_{i-1} '
            '(e_{-1}:=y). One checkpoint key per stage. '
            'Uses current telepred_delay / label / model_type=regression. '
            'Pair with --clean to wipe all stage keys and start over.'
        ),
    )
    parser.add_argument(
        '--error-ensemble-n', type=int, default=None, metavar='N',
        help=(
            'Number of ErrorEnsemble stages/models (model_0..model_{N-1}). '
            f"Default: config error_ensemble_n "
            f"({int(config.get('error_ensemble_n', 3))})."
        ),
    )
    parser.add_argument(
        '--validate', type=float, nargs='*', default=None, metavar='V',
        help=(
            'Run validation with the best checkpoint on a percent slice of '
            'sorted validation samples, then plot batch/MA/running-loss '
            '(same series as --validation; no files saved). '
            '--validate → [0%%, 100%%]; '
            '--validate V → [0%%, V%%]; '
            '--validate V1 V2 → [V1%%, V2%%]. '
            'Examples: --validate   |   --validate 30   |   --validate 20 40'
        ),
    )
    parser.add_argument(
        '--last-validate', type=float, nargs='*', default=None, metavar='V',
        help=(
            'Same as --validate, but loads the latest available checkpoint '
            '(training_state latest_checkpoint / newest .pt) instead of best_val. '
            '--last-validate → [0%%, 100%%]; '
            '--last-validate V → [0%%, V%%]; '
            '--last-validate V1 V2 → [V1%%, V2%%].'
        ),
    )
    parser.add_argument('--features', action='store_true', default=False,
                       help='Visualize total_features without training')
    parser.add_argument(
        '--validate-yield', nargs='?', const=-1, default=None, type=int,
        metavar='N',
        help=(
            'Load the latest checkpoint and plot chance / near over '
            'validation samples (yield label mode). '
            '--validate-yield uses all val samples; --validate-yield N uses '
            'N samples spaced across the chronological validation period. '
            'Shows series + pred-vs-true plots, then interactive yield windows. '
            'Batch size is config validate_batch_size.'
        ),
    )
    parser.add_argument(
        '--test', nargs='?', const=-1, default=None, type=int, metavar='N',
        help=(
            'Run test-set evaluation without training (same as INFER.py). '
            '--test uses all test samples; --test N uses N samples spaced '
            'across the chronological test period (not a fixed-seed head). '
            'Batch size is config test_batch_size; latent viz count is '
            'config test_viz_samples.'
        ),
    )
    parser.add_argument(
        '--past-window', type=int,
        default=int(config.get('past_window', 1440)),
        help='TelePred delta1: past input length in bars',
    )
    parser.add_argument(
        '--future-window', type=int,
        default=int(config.get('future_window', 50)),
        help='TelePred delta2: label future length in bars (yield/trend/latent)',
    )
    parser.add_argument('--batch-size', type=int, default=config['batch_size'], help='Batch size')
    parser.add_argument('--total-days', type=int, default=config['total_synthetic_days'], help='Total days of data')
    parser.add_argument('--step', type=int, default=config['step'], help='Step size for start indices')
    parser.add_argument('--max-accum-deviation', type=float, default=config['max_accum_deviation'],
                       help='Threshold for cumulative absolute deviation')
    parser.add_argument(
        '--min-trend-horizon', type=int,
        default=config.get('min_trend_horizon', 3),
        help='Minimum linear-regime horizon',
    )
    parser.add_argument(
        '--min-yield-percent', type=float,
        default=config.get('min_yield_percent', 0.0),
        help='Minimum yield percent for yield',
    )
    parser.add_argument(
        '--model-type', type=str,
        default=config.get('model_type', 'regression'),
        choices=['regression', 'vae'],
        help="Model backend: 'regression' or 'vae' (x recon; mean/std FiLM cond)",
    )
    parser.add_argument(
        '--label', type=str, default=config.get('label', 'trend'),
        choices=['trend', 'yield', 'latent'],
        help='Label family (shared head contract for regression and VAE)',
    )
    parser.add_argument(
        '--latent-distribution', type=str,
        default=config.get('latent_distribution', 'normal'),
        choices=[
            'normal', 'student',
            'normal_distribution', 'student_distribution', 'student_t',
        ],
        help=(
            'Density for --label latent: normal → output_dim=2 (mu, var_raw); '
            'student → output_dim=3 (loc, scale_raw, df_raw)'
        ),
    )
    parser.add_argument('--no-normalize-window', action='store_true', help='Disable per-window z-score normalization')
    parser.add_argument('--lr', type=float, default=config['learning_rate'], help='Learning rate')
    parser.add_argument('--slope-weight', type=float, default=config['slope_loss_weight'],
                       help='Weight for slope loss')
    parser.add_argument('--intercept-weight', type=float, default=config['intercept_loss_weight'],
                       help='Weight for intercept loss')
    parser.add_argument('--recon-weight', type=float, default=config['reconstruction_loss_weight'],
                       help='Weight for reconstruction loss')
    parser.add_argument('--patch-size', type=int, default=config['patch_size'],
                       help='Patch size for temporal patching')
    parser.add_argument('--num-layers', type=int, default=config['num_layers'],
                       help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=config['num_heads'],
                       help='Number of attention heads')
    parser.add_argument('--ff-dim-multiplier', type=int, default=config['ff_dim_multiplier'],
                       help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=config['dropout'], help='Dropout rate')
    parser.add_argument('--head-hidden-dim-multiplier', type=int, default=config['head_hidden_dim_multiplier'],
                       help='Hidden dimension multiplier for prediction heads')
    parser.add_argument('--no-causal', action='store_true', default=not config['causal'],
                       help='Disable causal attention')
    parser.add_argument('--no-reconstruction', action='store_true', default=not config['use_reconstruction'],
                       help='Disable reconstruction auxiliary loss')
    parser.add_argument(
        '--no-look-price',
        action='store_true',
        default=not config.get('look_price', True),
        help='Do not concat recent past prices p into the prediction head',
    )
    parser.add_argument('--use-binance', action='store_true', default=config['use_binance_data'],
                       help='Use real Binance data instead of synthetic')
    parser.add_argument('--symbol', type=str, default=config.get('binance_primary_symbol', 'BTCUSDT'),
                       help='Primary trading pair symbol')
    
    parser.add_argument('--train-ratio', type=float, default=config.get('train_ratio', 0.70),
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=config.get('val_ratio', 0.20),
                       help='Validation set ratio')
    
    parser.add_argument('--no-validate', action='store_true', default=not config.get('validate', True),
                       help='Disable validation')
    parser.add_argument('--validate-every', type=int, default=config.get('validate_every_n_epochs', 1),
                       help='Validate every N epochs')
    parser.add_argument('--val-batch-size', type=int, default=config.get('validate_batch_size', 64),
                       help='Batch size for validation')
    
    args = parser.parse_args()
    
    # Update config with command-line arguments
    config['epochs'] = args.epochs
    config['save_every_n_minutes'] = args.save_every
    config['clean'] = args.clean
    config['past_window'] = int(args.past_window)
    config['future_window'] = int(args.future_window)
    config['batch_size'] = args.batch_size
    config['total_synthetic_days'] = args.total_days
    config['step'] = args.step
    config['max_accum_deviation'] = args.max_accum_deviation
    config['min_trend_horizon'] = args.min_trend_horizon
    config['min_yield_percent'] = args.min_yield_percent
    config['model_type'] = args.model_type
    config['label'] = args.label
    config['latent_distribution'] = args.latent_distribution
    if getattr(args, 'checkpoint_key', None):
        config['checkpoint_key'] = str(args.checkpoint_key).strip()
    # Force head width for yield/latent so checkpoint keys match training.
    sync_output_dim(config)
    config['learning_rate'] = args.lr
    config['slope_loss_weight'] = args.slope_weight
    config['intercept_loss_weight'] = args.intercept_weight
    config['reconstruction_loss_weight'] = args.recon_weight
    config['patch_size'] = args.patch_size
    config['num_layers'] = args.num_layers
    config['num_heads'] = args.num_heads
    config['ff_dim_multiplier'] = args.ff_dim_multiplier
    config['dropout'] = args.dropout
    config['head_hidden_dim_multiplier'] = args.head_hidden_dim_multiplier
    config['causal'] = not args.no_causal
    config['use_reconstruction'] = not args.no_reconstruction
    config['look_price'] = not args.no_look_price
    config['use_binance_data'] = args.use_binance
    config['binance_primary_symbol'] = args.symbol
    config['train_ratio'] = args.train_ratio
    config['val_ratio'] = args.val_ratio
    config['validate'] = not args.no_validate
    config['validate_every_n_epochs'] = args.validate_every
    config['validate_batch_size'] = args.val_batch_size
    config['telepred_delay'] = int(args.telepred_delay)
    if args.telepred_delays is not None:
        config['telepred_delays'] = parse_delay_list(args.telepred_delays)
    else:
        config['telepred_delays'] = parse_delay_list(
            config.get('telepred_delays', [config['telepred_delay']])
        )
    if getattr(args, 'error_ensemble_n', None) is not None:
        config['error_ensemble_n'] = max(1, int(args.error_ensemble_n))
    else:
        config['error_ensemble_n'] = max(
            1, int(config.get('error_ensemble_n', 3) or 3)
        )

    if args.no_normalize_window:
        config['normalize_window'] = False

    # Push CLI overrides into the module config so any get_config() fallback
    # (and signature helpers) see the same values as this run.
    update_config(config)

    # ============================================================
    # STEP 1: Handle --trend flag (visualize trends only, no training)
    # ============================================================
    if args.trend:
        print("\n" + "=" * 60)
        print("📈 TREND VISUALIZATION MODE")
        print("=" * 60)
        print("Loading dataset, then paging random trend batches.")
        print("  Window buttons: [Next]  [Quit]")
        print("  Keyboard:       N / Space = next,  Q / Esc = quit")
        print("=" * 60 + "\n")

        dataset, total_features = get_dataset(config)
        dataset.visualize_trends(num_samples=6, interactive=True)

        print("\n✅ Trend visualization completed!")
        return

    # ============================================================
    # STEP 2: Handle --history flag (visualize history only, no training)
    # ============================================================
    if args.history is not None:
        print("\n" + "=" * 60)
        print("📊 TRAINING HISTORY VISUALIZATION MODE")
        print("=" * 60)

        checkpoint_dir = Path(config['checkpoint_dir'])

        if not checkpoint_dir.exists():
            print(f"⚠ Checkpoint directory not found: {checkpoint_dir}")
            print("No training history available.")
            return

        from trainer import TrainingHistory

        def _history_epochs(history_data) -> int:
            if not isinstance(history_data, dict):
                return 0
            epochs = history_data.get('epochs') or []
            return len(epochs)

        def _list_history_runs():
            """Return [(subdir, state_info, mtime)] for dirs with usable history."""
            runs = []
            for subdir in checkpoint_dir.iterdir():
                if not subdir.is_dir():
                    continue
                state_file = subdir / 'training_state.json'
                if not state_file.exists():
                    continue
                try:
                    with open(state_file, 'r') as f:
                        state_info = json.load(f)
                except Exception:
                    continue
                if _history_epochs(state_info.get('history')) <= 0:
                    continue
                runs.append((subdir, state_info, state_file.stat().st_mtime))
            runs.sort(key=lambda item: item[2], reverse=True)
            return runs

        def _plot_history(subdir: Path, state_info: dict, label: str = "") -> bool:
            history_data = state_info.get('history')
            if _history_epochs(history_data) <= 0:
                print(f"⚠ No epoch history in {subdir.name}")
                return False
            try:
                temp_history = TrainingHistory()
                temp_history.from_dict(history_data)

                print(f"📁 Checkpoint: {subdir.name}{label}")
                print(f"   Total epochs: {len(temp_history.epochs)}")
                if temp_history.train_losses:
                    print(f"   Final training loss: {temp_history.train_losses[-1]:.6f}")
                val_losses = [v for v in temp_history.val_losses if v is not None]
                if val_losses:
                    print(f"   Final validation loss: {val_losses[-1]:.6f}")

                plot_path = subdir / 'training_history_view.png'
                temp_history.plot(save_path=str(plot_path))
                print(f"   History plot saved to: {plot_path}\n")
                return True
            except Exception as e:
                print(f"⚠ Failed to load history from {subdir.name}: {e}")
                return False

        # Case 1: --history all
        if args.history == 'all':
            print("Showing ALL training histories...\n")
            history_found = False
            for subdir, state_info, _ in _list_history_runs():
                if _plot_history(subdir, state_info):
                    history_found = True
            # Also surface dirs that exist but have empty history
            if not history_found:
                print("⚠ No training history found in any checkpoint directory.")
                print("   Train a model first to generate history.")
                available = [p.name for p in checkpoint_dir.iterdir() if p.is_dir()]
                if available:
                    print(f"   Available checkpoint keys: {', '.join(available)}")
            else:
                print("✅ All training histories visualized!")
            return

        # Case 2: --history <explicit_key>
        if args.history not in ('current', None):
            explicit_key = args.history
            current_dir = checkpoint_dir / explicit_key
            print(f"🔑 Requested checkpoint key: {explicit_key}")
            if not current_dir.exists():
                print(f"⚠ Checkpoint directory not found: {current_dir}")
                available = [p.name for p in checkpoint_dir.iterdir() if p.is_dir()]
                if available:
                    print(f"   Available keys: {', '.join(available)}")
                return
            state_file = current_dir / 'training_state.json'
            if not state_file.exists():
                print(f"⚠ No training_state.json in {current_dir}")
                return
            with open(state_file, 'r') as f:
                state_info = json.load(f)
            if _plot_history(current_dir, state_info):
                print("✅ Training history visualization completed!")
            return

        # Case 3: --history (current config) — config-only key, no get_dataset()
        dataset_signature, current_key, current_dir = resolve_checkpoint_identity(config)
        print(f"🔑 Dataset signature: {dataset_signature}")
        print(f"🔑 Checkpoint key:    {current_key}")

        if current_dir.exists():
            state_file = current_dir / 'training_state.json'
            if not state_file.exists():
                print(f"⚠ No training_state.json in {current_dir}")
                return
            with open(state_file, 'r') as f:
                state_info = json.load(f)
            if _plot_history(current_dir, state_info, label=" (exact config match)"):
                print("✅ Training history visualization completed!")
            return

        # Exact key missing — common after signature/key scheme changes.
        print(f"⚠ No checkpoint folder for current config key: {current_key}")
        print("   (Keys change when signature/key generation is updated; old runs keep old keys.)")

        runs = _list_history_runs()
        available = [p.name for p in checkpoint_dir.iterdir() if p.is_dir()]
        if available:
            print(f"   Available checkpoint keys: {', '.join(available)}")
        if not runs:
            print("   No runs with non-empty epoch history found.")
            print("   Train under the current key, or use: --history <key> / --history all")
            return

        # Fall back to most recently updated run that has epoch history
        subdir, state_info, _ = runs[0]
        print(f"   Falling back to most recent run with history: {subdir.name}")
        print(f"   Tip: pin a run with  --history {subdir.name}")
        if _plot_history(subdir, state_info, label=" (fallback: latest with history)"):
            print("✅ Training history visualization completed!")
        return

    # ============================================================
    # STEP 2b: Handle --loss flag (latest-epoch minibatch curve)
    # ============================================================
    if args.loss is not None:
        print("\n" + "=" * 60)
        print("📉 LATEST EPOCH LOSS VISUALIZATION MODE")
        print("=" * 60)

        checkpoint_dir = Path(config['checkpoint_dir'])
        if not checkpoint_dir.exists():
            print(f"⚠ Checkpoint directory not found: {checkpoint_dir}")
            print("No latest-epoch loss history available.")
            return

        from trainer import LatestEpochLoss, LATEST_EPOCH_LOSS_FILENAME

        def _has_latest_loss(subdir: Path) -> bool:
            if (subdir / LATEST_EPOCH_LOSS_FILENAME).exists():
                return True
            state_file = subdir / 'training_state.json'
            if not state_file.exists():
                return False
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                data = state.get('latest_epoch_loss') or {}
                return bool(data.get('batch_losses'))
            except Exception:
                return False

        def _list_loss_runs():
            runs = []
            for subdir in checkpoint_dir.iterdir():
                if not subdir.is_dir():
                    continue
                if not _has_latest_loss(subdir):
                    continue
                # Prefer mtime of latest_epoch_loss.json, else state file
                loss_file = subdir / LATEST_EPOCH_LOSS_FILENAME
                state_file = subdir / 'training_state.json'
                mtime_src = loss_file if loss_file.exists() else state_file
                runs.append((subdir, mtime_src.stat().st_mtime))
            runs.sort(key=lambda item: item[1], reverse=True)
            return runs

        def _plot_latest_loss(subdir: Path, label: str = "") -> bool:
            le = LatestEpochLoss.load(subdir)
            if le is None or not le.batch_losses:
                print(f"⚠ No latest-epoch loss data in {subdir.name}")
                return False
            try:
                print(f"📁 Checkpoint: {subdir.name}{label}")
                le.print_summary()
                plot_path = subdir / 'latest_epoch_loss_view.png'
                le.plot(save_path=str(plot_path))
                print(f"   Loss plot saved to: {plot_path}\n")
                return True
            except Exception as e:
                print(f"⚠ Failed to plot latest-epoch loss from {subdir.name}: {e}")
                return False

        # Case 1: --loss all
        if args.loss == 'all':
            print("Showing ALL latest-epoch loss curves...\n")
            found = False
            for subdir, _ in _list_loss_runs():
                if _plot_latest_loss(subdir):
                    found = True
            if not found:
                print("⚠ No latest-epoch loss history found.")
                print("   Train at least one epoch to generate it.")
                available = [p.name for p in checkpoint_dir.iterdir() if p.is_dir()]
                if available:
                    print(f"   Available checkpoint keys: {', '.join(available)}")
            else:
                print("✅ All latest-epoch loss curves visualized!")
            return

        # Case 2: --loss <explicit_key>
        if args.loss not in ('current', None):
            explicit_key = args.loss
            current_dir = checkpoint_dir / explicit_key
            print(f"🔑 Requested checkpoint key: {explicit_key}")
            if not current_dir.exists():
                print(f"⚠ Checkpoint directory not found: {current_dir}")
                available = [p.name for p in checkpoint_dir.iterdir() if p.is_dir()]
                if available:
                    print(f"   Available keys: {', '.join(available)}")
                return
            if _plot_latest_loss(current_dir):
                print("✅ Latest-epoch loss visualization completed!")
            return

        # Case 3: --loss (current config key)
        dataset_signature, current_key, current_dir = resolve_checkpoint_identity(config)
        print(f"🔑 Dataset signature: {dataset_signature}")
        print(f"🔑 Checkpoint key:    {current_key}")

        if current_dir.exists() and _plot_latest_loss(current_dir, label=" (exact config match)"):
            print("✅ Latest-epoch loss visualization completed!")
            return

        print(f"⚠ No latest-epoch loss for current config key: {current_key}")
        runs = _list_loss_runs()
        available = [p.name for p in checkpoint_dir.iterdir() if p.is_dir()]
        if available:
            print(f"   Available checkpoint keys: {', '.join(available)}")
        if not runs:
            print("   No runs with latest-epoch loss found.")
            print("   Train under the current key, or use: --loss <key> / --loss all")
            return

        subdir, _ = runs[0]
        print(f"   Falling back to most recent run with loss data: {subdir.name}")
        print(f"   Tip: pin a run with  --loss {subdir.name}")
        if _plot_latest_loss(subdir, label=" (fallback: latest with loss data)"):
            print("✅ Latest-epoch loss visualization completed!")
        return

    # ============================================================
    # STEP 2c: Handle --validation [E [V1] [V2]] (per-batch val curve)
    # ============================================================
    if args.validation is not None:
        print("\n" + "=" * 60)
        print("📉 VALIDATION LOSS VISUALIZATION MODE")
        print("=" * 60)

        checkpoint_dir = Path(config['checkpoint_dir'])
        if not checkpoint_dir.exists():
            print(f"⚠ Checkpoint directory not found: {checkpoint_dir}")
            print("No validation loss history available.")
            return

        from trainer import (
            ValidationEpochLoss,
            list_validation_loss_epochs,
            validation_loss_path,
        )

        # --validation → latest, [0,100]
        # --validation E → epoch E, [0,100]
        # --validation E V2 → epoch E, [0, V2]
        # --validation E V1 V2 → epoch E, [V1, V2]
        val_args = list(args.validation)
        if len(val_args) > 3:
            print(
                "⚠ --validation accepts at most three values: "
                "E  |  E V2  |  E V1 V2  (or bare --validation for latest)."
            )
            return
        requested_epoch = None
        percent_lo, percent_hi = 0.0, 100.0
        if len(val_args) >= 1:
            try:
                requested_epoch = int(float(val_args[0]))
            except (TypeError, ValueError):
                print(f"⚠ --validation epoch must be an integer (got {val_args[0]!r}).")
                return
            if requested_epoch < 1:
                print("⚠ Epoch must be >= 1 (1-based), or omit E for the latest saved run.")
                return
        if len(val_args) == 2:
            try:
                percent_lo, percent_hi = 0.0, float(val_args[1])
            except (TypeError, ValueError):
                print(f"⚠ --validation V2 must be a number (got {val_args[1]!r}).")
                return
        elif len(val_args) == 3:
            try:
                percent_lo, percent_hi = float(val_args[1]), float(val_args[2])
            except (TypeError, ValueError):
                print(
                    f"⚠ --validation V1 V2 must be numbers "
                    f"(got {val_args[1]!r}, {val_args[2]!r})."
                )
                return
        if not (0.0 <= percent_lo < percent_hi <= 100.0):
            print(
                f"⚠ --validation requires 0 <= V1 < V2 <= 100 "
                f"(got V1={percent_lo:g}, V2={percent_hi:g})"
            )
            return

        def _list_val_runs():
            runs = []
            for subdir in checkpoint_dir.iterdir():
                if not subdir.is_dir():
                    continue
                epochs = list_validation_loss_epochs(subdir)
                if not epochs:
                    continue
                newest = validation_loss_path(subdir, epochs[-1])
                runs.append((subdir, epochs, newest.stat().st_mtime))
            runs.sort(key=lambda item: item[2], reverse=True)
            return runs

        def _plot_validation(subdir: Path, epoch: int, label: str = "") -> bool:
            ve = ValidationEpochLoss.load(subdir, epoch)
            if ve is None or not ve.batch_losses:
                print(f"⚠ No validation loss data for epoch {epoch} in {subdir.name}")
                return False
            try:
                print(f"📁 Checkpoint: {subdir.name}{label}")
                ve.print_summary(percent_lo=percent_lo, percent_hi=percent_hi)
                plot_path = subdir / f'validation_loss_epoch_{epoch}_view.png'
                ve.plot(
                    save_path=str(plot_path),
                    percent_lo=percent_lo,
                    percent_hi=percent_hi,
                )
                print(f"   Validation loss plot saved to: {plot_path}\n")
                return True
            except Exception as e:
                print(f"⚠ Failed to plot validation loss from {subdir.name}: {e}")
                return False

        def _resolve_epoch(subdir: Path, epochs: list):
            if not epochs:
                return None
            if requested_epoch is None:
                return epochs[-1]
            if requested_epoch in epochs:
                return requested_epoch
            return None

        # Prefer current config key (same signature/key scheme as --loss / --history).
        dataset_signature, current_key, current_dir = resolve_checkpoint_identity(config)
        print(f"🔑 Dataset signature: {dataset_signature}")
        print(f"🔑 Checkpoint key:    {current_key}")
        if requested_epoch is None:
            print("🎯 Epoch:             latest saved")
        else:
            print(f"🎯 Epoch:             {requested_epoch}")
        print(f"🎯 Coverage:          [{percent_lo:g}%, {percent_hi:g}%] of val batches")

        if current_dir.exists():
            epochs = list_validation_loss_epochs(current_dir)
            epoch = _resolve_epoch(current_dir, epochs)
            if epoch is not None and _plot_validation(
                current_dir, epoch, label=" (exact config match)"
            ):
                print("✅ Validation loss visualization completed!")
                return
            if requested_epoch is not None and epochs:
                print(
                    f"⚠ Epoch {requested_epoch} not found under current key. "
                    f"Available epochs: {', '.join(str(e) for e in epochs)}"
                )
            elif not epochs:
                print(f"⚠ No validation_loss_epoch_*.json under current key: {current_key}")

        runs = _list_val_runs()
        available = [p.name for p in checkpoint_dir.iterdir() if p.is_dir()]
        if available:
            print(f"   Available checkpoint keys: {', '.join(available)}")
        if not runs:
            print("   No runs with validation loss history found.")
            print("   Train with validation enabled to generate validation_loss_epoch_<E>.json")
            return

        # Fall back: first run (newest) that has the requested epoch, else newest latest.
        for subdir, epochs, _ in runs:
            epoch = _resolve_epoch(subdir, epochs)
            if epoch is None:
                continue
            if subdir.name == current_key:
                continue
            print(f"   Falling back to: {subdir.name} (epoch {epoch})")
            print(f"   Tip: train under the current key, or inspect available epochs there.")
            if _plot_validation(subdir, epoch, label=" (fallback)"):
                print("✅ Validation loss visualization completed!")
            return

        if requested_epoch is not None:
            print(f"⚠ No checkpoint key has validation loss for epoch {requested_epoch}.")
        else:
            print("⚠ No validation loss history could be plotted.")
        return

    # ============================================================
    # STEP 2d: Handle --report 'dir name' → <project_root>/_save/<dir_name>/
    # ============================================================
    if args.report is not None:
        print("\n" + "=" * 60)
        print("📑 VALIDATION REPORT EXPORT MODE")
        print("=" * 60)

        report_dir, err = _resolve_report_dir(args.report, flag='--report')
        if err:
            print(err)
            return

        checkpoint_dir = Path(config['checkpoint_dir'])
        if not checkpoint_dir.exists():
            print(f"⚠ Checkpoint directory not found: {checkpoint_dir}")
            return

        from trainer import list_validation_loss_epochs, validation_loss_path

        dataset_signature, current_key, current_dir = resolve_checkpoint_identity(config)
        print(f"🔑 Dataset signature: {dataset_signature}")
        print(f"🔑 Checkpoint key:    {current_key}")
        print(f"📁 Project root:      {project_root_dir()}")
        print(f"📁 Report dir:        {report_dir}")

        def _list_report_runs():
            runs = []
            for subdir in checkpoint_dir.iterdir():
                if not subdir.is_dir():
                    continue
                ep = list_validation_loss_epochs(subdir)
                if not ep:
                    continue
                newest = validation_loss_path(subdir, ep[-1])
                runs.append((subdir, ep, newest.stat().st_mtime))
            runs.sort(key=lambda item: item[2], reverse=True)
            return runs

        # Prefer exact config match; else newest run that has validation curves
        # (same fallback policy as --validation).
        source_dir = None
        source_label = ''
        if current_dir.exists():
            epochs = list_validation_loss_epochs(current_dir)
            if epochs:
                source_dir = current_dir
                source_label = 'exact config match'
        if source_dir is None:
            if current_dir.exists():
                print(
                    f"⚠ No validation_loss_epoch_*.json under current key: "
                    f"{current_key}"
                )
            else:
                print(f"⚠ No checkpoint folder for current key: {current_dir}")
            runs = _list_report_runs()
            available = [p.name for p in checkpoint_dir.iterdir() if p.is_dir()]
            if available:
                print(f"   Available checkpoint keys: {', '.join(available)}")
            if not runs:
                print("   No runs with validation loss history found.")
                print(
                    "   Train with validation enabled to generate "
                    "validation_loss_epoch_<E>.json"
                )
                return
            source_dir, _epochs, _ = runs[0]
            source_label = 'fallback: newest with val curves'
            print(f"   Falling back to: {source_dir.name}")
            print("   Tip: train under the current key, or match config to that run.")

        print(f"📁 Source checkpoints: {source_dir} ({source_label})")
        stats = _export_validation_epoch_plots(source_dir, report_dir)
        if stats['max_epoch'] > 0:
            e_example = f"e{1:0{stats['epoch_width']}d}"
            print(
                f"🎯 Epochs:            1..{stats['max_epoch']} "
                f"(saved: {', '.join(str(e) for e in stats['epochs'])}; "
                f"files like {e_example}.png)"
            )
        print(
            f"\n✅ Report export completed: "
            f"{stats['n_full']} full + {stats['n_slice']} p0_30 plots → {report_dir}"
        )
        if stats['n_missing']:
            print(f"   Skipped {stats['n_missing']} missing epoch curve(s).")
        return

    # ============================================================
    # STEP 2e: --ensemble-report 'folder name' (all --ensemble delays)
    # ============================================================
    if getattr(args, 'ensemble_report', None) is not None:
        print("\n" + "=" * 60)
        print("📑 ENSEMBLE VALIDATION REPORT EXPORT MODE")
        print("=" * 60)

        report_root, err = _resolve_report_dir(
            args.ensemble_report, flag='--ensemble-report'
        )
        if err:
            print(err)
            return

        delays = parse_delay_list(
            config.get('telepred_delays', [config.get('telepred_delay', 0)])
        )
        checkpoint_dir = Path(config['checkpoint_dir'])
        print(f"📁 Project root: {project_root_dir()}")
        print(f"📁 Report root:  {report_root}")
        print(f"🎯 Delays:       {list(delays)}")
        if not checkpoint_dir.exists():
            print(f"⚠ Checkpoint directory not found: {checkpoint_dir}")
            return

        from trainer import list_validation_loss_epochs

        n_members_ok = 0
        n_members_skip = 0
        total_full = 0
        total_slice = 0
        for d in delays:
            elem_cfg = dict(config)
            elem_cfg['telepred_delay'] = int(d)
            sig, key, key_dir = resolve_checkpoint_identity(elem_cfg)
            member_dir = report_root / f'd{int(d)}'
            print("\n" + "-" * 60)
            print(f"ENSEMBLE REPORT d={d}")
            print("-" * 60)
            print(f"🔑 Dataset signature: {sig}")
            print(f"🔑 Checkpoint key:    {key}")
            print(f"📁 Member report dir: {member_dir}")

            if not key_dir.exists():
                print(f"⚠ No checkpoint folder: {key_dir}")
                n_members_skip += 1
                continue
            epochs = list_validation_loss_epochs(key_dir)
            if not epochs:
                print(
                    f"⚠ No validation_loss_epoch_*.json under {key_dir.name}/"
                )
                n_members_skip += 1
                continue

            stats = _export_validation_epoch_plots(key_dir, member_dir)
            print(
                f"✓ d={d}: {stats['n_full']} full + {stats['n_slice']} p0_30 "
                f"(epochs saved: {', '.join(str(e) for e in stats['epochs'])})"
            )
            if stats['n_missing']:
                print(f"   Skipped {stats['n_missing']} missing epoch curve(s).")
            n_members_ok += 1
            total_full += stats['n_full']
            total_slice += stats['n_slice']

        print("\n" + "=" * 60)
        print(
            f"✅ Ensemble report completed: {n_members_ok}/{len(delays)} "
            f"members with curves; {total_full} full + {total_slice} p0_30 plots"
        )
        print(f"   → {report_root}")
        if n_members_skip:
            print(f"   Skipped {n_members_skip} member(s) without val history.")
        print("=" * 60)
        return

    # ============================================================
    # STEP 3: Handle --clean flag
    # ============================================================
    # Only remove this config's key folder(s). Sibling checkpoint keys stay intact.
    checkpoint_dir = Path(config['checkpoint_dir'])
    if config['clean']:
        if getattr(args, 'error_ensemble', False):
            # Stage keys include error_ensemble_stage; the bare (non-stage) key
            # from resolve_checkpoint_identity does NOT match those folders.
            from error_ensemble import clean_error_ensemble_checkpoints
            clean_error_ensemble_checkpoints(
                config, n=int(config.get('error_ensemble_n', 3))
            )
            # Force a fresh start even if a stub dir is recreated later.
            config['resume'] = False
        else:
            # Match ResumableTrainer key (signature + synced output_dim).
            dataset_signature, current_key, key_dir = resolve_checkpoint_identity(config)
            if key_dir.exists():
                shutil.rmtree(key_dir)
                print(f"✓ Cleaned checkpoints for key: {current_key}")
                print(f"  Removed: {key_dir}")
                print(f"  Other keys under {checkpoint_dir}/ were left untouched.\n")
            else:
                print(f"✓ No checkpoints to clean for key: {current_key}")
                print(f"  Looked in: {key_dir}\n")
            config['resume'] = False

    # ============================================================
    # STEP : Handle --features flag (visualize total_features only, no training)
    # ============================================================
    if args.features:
        print("\n" + "=" * 60)
        print("📈 FEATURES VISUALIZATION MODE")
        print("=" * 60)
        print("Visualizing features...")
        print("=" * 60 + "\n")
        
        # Get data
        dataset, total_features = get_dataset(config)
        
        for fid, feature in enumerate(total_features):
            try:
                dataset.visualize_feature(fid, feature, dataset.feature_names[fid])
            except Exception as e:
                print(f"⚠ Failed to visualize feature {fid}: {e}")
                print(dataset.feature_names[fid])

        print("\n✅ Features visualization completed!")
        return

    # ============================================================
    # STEP 3a0: Handle --vae-compare [--vae-key] [--yield-key]
    # ============================================================
    if args.vae_compare is not None:
        vals = [float(x) for x in args.vae_compare]
        if len(vals) == 0:
            percent_lo, percent_hi = 0.0, 100.0
        elif len(vals) == 1:
            percent_lo, percent_hi = 0.0, vals[0]
        elif len(vals) == 2:
            percent_lo, percent_hi = vals[0], vals[1]
        else:
            print("⚠ --vae-compare accepts zero, one, or two values:  |  V  |  V1 V2")
            return
        if not args.vae_key or not args.yield_key:
            print("⚠ --vae-compare requires both --vae-key and --yield-key.")
            return
        from INFER import run_vae_compare
        run_vae_compare(
            config,
            vae_key=args.vae_key,
            yield_key=args.yield_key,
            percent_lo=percent_lo,
            percent_hi=percent_hi,
        )
        print("\n✅ VAE↔yield compare completed!")
        return

    # ============================================================
    # STEP 3a: Handle --validate / --last-validate [V [V2]]
    # ============================================================
    if args.validate is not None and getattr(args, 'last_validate', None) is not None:
        print("⚠ Use only one of --validate or --last-validate, not both.")
        return

    _validate_args = None
    _validate_mode = None
    if args.validate is not None:
        _validate_args = args.validate
        _validate_mode = 'best'
    elif getattr(args, 'last_validate', None) is not None:
        _validate_args = args.last_validate
        _validate_mode = 'latest'

    if _validate_args is not None:
        vals = [float(x) for x in _validate_args]
        flag = '--validate' if _validate_mode == 'best' else '--last-validate'
        # bare flag → [0, 100]; V → [0, V]; V1 V2 → [V1, V2]
        if len(vals) == 0:
            percent_lo, percent_hi = 0.0, 100.0
        elif len(vals) == 1:
            percent_lo, percent_hi = 0.0, vals[0]
        elif len(vals) == 2:
            percent_lo, percent_hi = vals[0], vals[1]
        else:
            print(f"⚠ {flag} accepts zero, one, or two values:  |  V  |  V1 V2")
            return
        if not (0.0 <= percent_lo < percent_hi <= 100.0):
            print(
                f"⚠ {flag} requires 0 <= V1 < V2 <= 100 "
                f"(got V1={percent_lo:g}, V2={percent_hi:g})"
            )
            return

        val_bs = config.get(
            'validate_batch_size',
            config.get('test_batch_size', config.get('batch_size', 8)),
        )

        print("\n" + "=" * 60)
        if _validate_mode == 'best':
            print("✅ BEST-CHECKPOINT VALIDATION MODE")
            print("=" * 60)
            print("Loads dataset + best_val checkpoint, evaluates sorted val samples.")
        else:
            print("✅ LATEST-CHECKPOINT VALIDATION MODE")
            print("=" * 60)
            print("Loads dataset + latest checkpoint, evaluates sorted val samples.")
        print(
            f"  Coverage: [{percent_lo:g}%, {percent_hi:g}%] of sorted validation samples"
        )
        print(f"  Batch size (config validate_batch_size): {val_bs}")
        print("  Plot: batch loss / moving avg / running loss (no save)")
        print("=" * 60 + "\n")

        # Lazy import keeps TRAIN → INFER one-way (dataset_build owns get_dataset).
        from INFER import run_validate

        run_validate(
            config,
            percent_lo=percent_lo,
            percent_hi=percent_hi,
            checkpoint_mode=_validate_mode,
            checkpoint_key=config.get('checkpoint_key'),
        )
        return

    # ============================================================
    # STEP 3a2: Handle --validate-yield / --validate-yield N
    # ============================================================
    if args.validate_yield is not None:
        max_samples = None if args.validate_yield < 0 else args.validate_yield
        val_bs = config.get(
            'validate_batch_size',
            config.get('test_batch_size', config.get('batch_size', 8)),
        )

        print("\n" + "=" * 60)
        print("✅ VALIDATE-YIELD MODE")
        print("=" * 60)
        print("Loads dataset + latest checkpoint, evaluates validation samples.")
        print(f"  Samples: {'all' if max_samples is None else max_samples}")
        print(f"  Batch size (config validate_batch_size): {val_bs}")
        print("  Plots: chance / near over val samples + interactive windows")
        print("=" * 60 + "\n")

        from INFER import run_validate_yield

        run_validate_yield(config, max_samples=max_samples, visualize=True)
        print("\n✅ Validate-yield completed!")
        return

    # ============================================================
    # STEP 3b: Handle --test / --test N (test-set eval like INFER.py)
    # ============================================================
    if args.test is not None:
        # const=-1 means bare --test → all samples; --test N → N samples
        max_samples = None if args.test < 0 else args.test
        test_bs = config.get(
            'test_batch_size',
            config.get('validate_batch_size', config.get('batch_size', 8)),
        )

        print("\n" + "=" * 60)
        print("🧪 TEST / INFERENCE MODE")
        print("=" * 60)
        print("Loads dataset + checkpoint, streams shuffled test samples.")
        print(f"  Samples: {'all' if max_samples is None else max_samples}")
        print(f"  Batch size (config test_batch_size): {test_bs}")
        print("=" * 60 + "\n")

        # Lazy import keeps TRAIN → INFER one-way (dataset_build owns get_dataset).
        from INFER import run_test

        run_test(config, max_samples=max_samples, visualize=True)
        print("\n✅ Test evaluation completed!")
        return


    # ============================================================
    # STEP 4: Ensemble sweep / ensemble train / single-element train
    # ============================================================

    if getattr(args, 'ensemble_sweep', False):
        from ensemble_train import run_ensemble_sweep
        csv_path = run_ensemble_sweep(
            config,
            delays=config.get('telepred_delays'),
            epochs_per_trial=max(1, int(getattr(args, 'sweep_epochs', 1))),
            extend_epochs=float(getattr(args, 'sweep_extend', 1.0)),
            get_dataset=get_dataset,
        )
        print(f"\n✅ Ensemble sweep completed → {csv_path}")
        return

    if getattr(args, 'ensemble', False):
        from ensemble_train import run_ensemble_train, ensemble_csv_path
        run_ensemble_train(
            config,
            delays=config.get('telepred_delays'),
            get_dataset=get_dataset,
            num_epochs=float(config['epochs']),
            extend=(args.extend is not None and args.extend != 0),
            extend_epochs=args.extend if args.extend is not None else 0.0,
            write_csv=True,
        )
        csv_path = ensemble_csv_path(
            label=str(config.get('label', 'yield')),
            model_type=str(config.get('model_type', 'regression')),
        )
        print(f"\n✅ Ensemble training completed → {csv_path}")
        return

    # Single-element path (and legacy --sweep on one delay's dataset)
    dataset, total_features = get_dataset(config)

    if getattr(args, 'error_ensemble', False):
        from error_ensemble import run_error_ensemble_train
        # --clean already wiped stage folders and set resume=False.
        outcome = run_error_ensemble_train(
            config,
            dataset,
            total_features,
            n=int(config.get('error_ensemble_n', 3)),
            num_epochs=float(config['epochs']),
            resume=False if config.get('clean') else config.get('resume', True),
            extend=(args.extend is not None and args.extend != 0),
            extend_epochs=args.extend if args.extend is not None else 0.0,
            probe_weights=True,
        )
        print(
            f"\n✅ ErrorEnsemble training completed: "
            f"{outcome['n_ok']}/{outcome['n']} stages ok"
        )
        return

    if getattr(args, 'sweep', False) or getattr(args, 'vae_sweep', False):
        from vae_sweep import run_hparam_sweep
        if getattr(args, 'vae_sweep', False):
            config['model_type'] = 'vae'
        mt = str(config.get('model_type', 'regression')).lower()
        csv_path = run_hparam_sweep(
            config,
            dataset,
            total_features,
            model_type=mt,
            epochs_per_trial=max(1, int(getattr(args, 'sweep_epochs', 1))),
            extend_epochs=float(getattr(args, 'sweep_extend', 1.0)),
        )
        print(f"\n✅ {mt} sweep completed → {csv_path}")
        return

    train_one_run(
        config,
        dataset,
        total_features,
        num_epochs=float(config['epochs']),
        resume=config.get('resume', True),
        extend=(args.extend is not None and args.extend != 0),
        extend_epochs=args.extend if args.extend is not None else 0.0,
        probe_weights=True,
        show_banner=True,
    )




if __name__ == "__main__":

    from config import get_config

    main(get_config())