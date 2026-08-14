# TRAIN.py
# Main training script - Clean version with all logic in trainer

import sys
import os
from pathlib import Path

import argparse
import json
import math
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data import (
    TradingDataset,
    ThreadedDataLoader,
    get_binance_data,
    generate_dataset_signature,
)
from labels import (
    bind_label_loss,
    label_target_names,
    resolve_future_window,
    resolve_model_past_len,
)
from model import (
    MarketTransformerWithHeads,
    report_model_size,
    check_all_weights_updated,
)
from trainer import ResumableTrainer, generate_checkpoint_key


def generate_synthetic_data(T: int, F: int, price_feature_idx: int = 0):
    """Generate synthetic trading data for testing."""
    features = []
    for i in range(F):
        if i == price_feature_idx:
            price = np.cumsum(np.random.randn(T) * 0.5 + 0.1)
            features.append(price.astype(np.float32))
        elif i == 1:
            volume = np.abs(np.random.randn(T) * 100 + 1000)
            features.append(volume.astype(np.float32))
        else:
            features.append(np.random.randn(T).astype(np.float32))
    return features

def get_dataset(config):
    """Build TradingDataset from Binance (or synthetic) market data + FSV macros."""
    market_features = None
    target_feature_id = None
    market_feature_names = None

    if config.get('use_binance_data', False):
        start_s = config.get('binance_start_date')
        end_s = config.get('binance_end_date')
        if start_s and end_s and str(start_s) >= str(end_s):
            # String compare works for 'YYYY-MM-DD HH:MM:SS' layout.
            print(
                f"⚠ binance_start_date ({start_s}) must be strictly before "
                f"binance_end_date ({end_s}). Zero-length range yields T=0."
            )
            sys.exit(1)
        market_features, T, nForget, target_feature_id, market_feature_names = get_binance_data(config)
        if market_features is None or target_feature_id is None:
            print("⚠ Failed to fetch Binance data.")
            sys.exit(1)
        # get_binance_data returns a list of 1-d arrays → stack to (F_m, T)
        if isinstance(market_features, list):
            market_arr = np.stack(market_features, axis=0).astype(np.float32)
        else:
            market_arr = np.asarray(market_features, dtype=np.float32)
        if market_arr.ndim != 2 or market_arr.shape[1] == 0:
            print(
                f"⚠ Market series is empty (shape={getattr(market_arr, 'shape', None)}). "
                f"Check binance_start_date / binance_end_date and data availability."
            )
            sys.exit(1)
    else:
        T = 60 * 24 * config.get('total_synthetic_days', 30)
        F = config.get('num_features', 5)
        target_feature_id = config.get('price_feature_idx', 0)
        market_list = generate_synthetic_data(T, F, target_feature_id)
        market_arr = np.stack(market_list, axis=0).astype(np.float32)
        market_feature_names = [f'feature_{i}' for i in range(F)]
        print(f"Generated synthetic market data: F={F}, T={T}")

    # Add Financial State Vectors (FSV) to features
    from fsv_03_query_value_age import get_value_age_array
    macro_features, macro_feature_names = get_value_age_array(config)
    macro_features = np.asarray(macro_features, dtype=np.float32)
    if macro_features.ndim == 1:
        # Legacy empty / corrupt cache: promote to (0, T) or fail clearly
        if macro_features.size == 0:
            macro_features = np.zeros((0, market_arr.shape[1]), dtype=np.float32)
        else:
            raise ValueError(
                f"macro_features must be 2-D (n_channels, T); got 1-D shape "
                f"{macro_features.shape}. Recompute FSV cache (delete "
                f"{config.get('data_dir', 'dataset_cache')}/fsv_value_decay_*.npy)."
            )
    if macro_features.ndim != 2:
        raise ValueError(
            f"macro_features must be 2-D (n_channels, T); got shape "
            f"{macro_features.shape}"
        )

    macro_feature_names_full = []
    for name in (macro_feature_names or []):
        macro_feature_names_full.append(name + "_value")
        macro_feature_names_full.append(name + "_decay")

    if macro_features.shape[0] == 0:
        print(
            "⚠ No macro FSV channels loaded (empty feature list or FSV failure). "
            "Continuing with market features only."
        )
    elif macro_features.shape[0] != len(macro_feature_names_full):
        print(
            f"⚠ macro channel count {macro_features.shape[0]} != "
            f"name count {len(macro_feature_names_full)}; "
            f"trimming names/channels to the shorter length."
        )
        n_ch = min(macro_features.shape[0], len(macro_feature_names_full))
        macro_features = macro_features[:n_ch]
        macro_feature_names_full = macro_feature_names_full[:n_ch]

    # Align time lengths (market vs macro caches can differ slightly)
    if macro_features.shape[0] > 0:
        T = min(int(market_arr.shape[1]), int(macro_features.shape[1]))
        if market_arr.shape[1] != macro_features.shape[1]:
            print(
                f"⚠ Aligning time lengths: market T={market_arr.shape[1]}, "
                f"macro T={macro_features.shape[1]} → using T={T}"
            )
        market_arr = market_arr[:, :T]
        macro_features = macro_features[:, :T]
        total_features = np.concatenate([market_arr, macro_features], axis=0)  # (F, T)
    else:
        T = int(market_arr.shape[1])
        total_features = market_arr
    n_market_features = market_arr.shape[0]
    n_macro_features = macro_features.shape[0]

    past_bars = 60 * 24 * int(config.get('past_days', 7))
    future_bars = int(resolve_future_window(config))
    need_T = past_bars + future_bars
    if T < need_T:
        print(
            f"⚠ Time length T={T} is shorter than past+future "
            f"({past_bars}+{future_bars}={need_T}). "
            f"Widen binance_start_date..binance_end_date or reduce past_days / horizons."
        )
        sys.exit(1)

    # TradingDataset stacks list/rows on axis=1 → (T, F)
    features_for_dataset = [total_features[i] for i in range(total_features.shape[0])]

    train_ratio = config.get('train_ratio', 0.70)
    val_ratio = config.get('val_ratio', 0.20)
    test_ratio = 1.0 - train_ratio - val_ratio
    split_ratios = (train_ratio, val_ratio, test_ratio)

    dataset = TradingDataset(
        features=features_for_dataset,
        n_market_features=n_market_features,
        n_macro_features=n_macro_features,
        feature_names=market_feature_names + macro_feature_names_full,
        price_feature_idx=target_feature_id,
        past_days=config['past_days'],
        future_minutes=resolve_future_window(config),
        step=config['step'],
        max_accum_deviation=config['max_accum_deviation'],
        min_trend_horizon=config.get('min_trend_horizon', 3),
        max_trend_future_horizon=config.get('max_trend_future_horizon'),
        device=config['device'],
        normalize=config['normalize'],
        data_dir=config.get('data_dir', 'dataset_cache'),
        force_recompute=config.get('force_recompute', False),
        split_ratios=split_ratios
    )
    return dataset, total_features


def main(config):
    """Main training function."""
    
    # Ensure price_feature_idx exists
    if 'price_feature_idx' not in config:
        config['price_feature_idx'] = 0
    
    parser = argparse.ArgumentParser(description='Resilient Trading Training with Transformer')
    parser.add_argument('--epochs', type=int, default=config['epochs'], help='Number of epochs')
    parser.add_argument('--save-every', type=int, default=config['save_every_n_minibatches'],
                       help='Save every N minibatches')
    parser.add_argument('--clean', action='store_true', help='Delete old checkpoints')
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
    parser.add_argument('--features', action='store_true', default=False,
                       help='Visualize total_features without training')
    parser.add_argument(
        '--test', nargs='?', const=-1, default=None, type=int, metavar='N',
        help=(
            'Run test-set evaluation without training (same as INFER.py). '
            '--test uses all shuffled test samples; --test N uses N shuffled '
            'test samples. Batch size is config test_batch_size.'
        ),
    )
    parser.add_argument('--past-days', type=int, default=config['past_days'], help='Past days for input')
    parser.add_argument(
        '--max-trend-future-horizon', type=int,
        default=config.get('max_trend_future_horizon', 50),
        help='Max future trend horizon (also future_window for trend)',
    )
    parser.add_argument(
        '--max-hope-future-horizon', type=int,
        default=config.get('max_hope_future_horizon', 60),
        help='Max hope future horizon (also future_window for yield)',
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
        '--label', type=str, default=config.get('label', 'trend'),
        choices=['trend', 'yield', 'latent'],
        help='Label family (latent uses shared MarketTransformerWithHeads)',
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
    parser.add_argument('--no-normalize', action='store_true', help='Disable per-window normalization')
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
    config['save_every_n_minibatches'] = args.save_every
    config['clean'] = args.clean
    config['past_days'] = args.past_days
    config['max_trend_future_horizon'] = args.max_trend_future_horizon
    config['max_hope_future_horizon'] = args.max_hope_future_horizon
    config['batch_size'] = args.batch_size
    config['total_synthetic_days'] = args.total_days
    config['step'] = args.step
    config['max_accum_deviation'] = args.max_accum_deviation
    config['min_trend_horizon'] = args.min_trend_horizon
    config['min_yield_percent'] = args.min_yield_percent
    config['label'] = args.label
    config['latent_distribution'] = args.latent_distribution
    if args.label == 'latent':
        # Force shared model head width from density mode.
        from labels import Label_Latent
        config['output_dim'] = Label_Latent.resolve_output_dim(config)
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
    
    if args.no_normalize:
        config['normalize'] = False
    
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
        dataset_signature = generate_dataset_signature(config)
        current_key = generate_checkpoint_key(config, dataset_signature)
        current_dir = checkpoint_dir / current_key
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
        dataset_signature = generate_dataset_signature(config)
        current_key = generate_checkpoint_key(config, dataset_signature)
        current_dir = checkpoint_dir / current_key
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
    # STEP 3: Handle --clean flag
    # ============================================================
    checkpoint_dir = Path(config['checkpoint_dir'])
    if config['clean'] and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print(f"✓ Cleaned previous checkpoints from: {config['checkpoint_dir']}\n")

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

        # Lazy import: INFER imports get_dataset from this module.
        from INFER import run_test

        run_test(config, max_samples=max_samples, visualize=True)
        print("\n✅ Test evaluation completed!")
        return

    # ============================================================
    # STEP 4: Get data (Binance or Synthetic)
    # ============================================================

    dataset, total_features = get_dataset(config)
    
    past_window = 60 * 24 * config['past_days']
    # total_features is (F, T); model uses latent macro compression for embed_dim
    F = total_features.shape[0] if hasattr(total_features, 'shape') else len(total_features)
    n_features_after_latent = dataset.n_market_features + config['n_latent_macro_features']
    embed_dim = n_features_after_latent * config['patch_size']
    if embed_dim % config['num_heads'] != 0:
        raise ValueError(
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({config['num_heads']}). "
            f"Adjust patch_size / n_latent_macro_features / num_heads."
        )
    config['embed_dim'] = embed_dim
    config['max_seq_len'] = past_window // config['patch_size']
    
    # Keep config in sync if dataset forced output_dim (e.g. yield → 2)
    config['output_dim'] = dataset.output_dim
    print(f"\nData shapes:")
    print(f"  Input x: (F={F}, past_window={past_window})")
    print(f"  Label mode: {getattr(dataset, 'label_mode', config.get('label', 'trend'))}")
    _od = config['output_dim']
    _yt = getattr(dataset, 'yield_target', config.get('yield_target', 'both'))
    _ld = getattr(
        dataset, 'latent_distribution', config.get('latent_distribution', 'normal')
    )
    _onames = label_target_names(
        getattr(dataset, 'label_mode', config.get('label', 'trend')),
        _od,
        yield_target=_yt,
        latent_distribution=_ld,
    )
    print(f"  Output: {_onames} - {_od} values (output_dim)")
    if getattr(dataset, 'label_mode', None) == 'yield' or config.get('label') == 'yield':
        print(f"  yield_target: {_yt}")
    if getattr(dataset, 'label_mode', None) == 'latent' or config.get('label') == 'latent':
        print(f"  latent_distribution: {_ld}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Train samples: {len(dataset.train_indices)}")
    print(f"  Val samples: {len(dataset.val_indices)}")
    print(f"  Test samples: {len(dataset.test_indices)}")
    print(f"  n_market_features: {dataset.n_market_features}")
    print(f"  n_macro_features: {dataset.n_macro_features}")
    print(f"\nTransformer config:")
    print(f"  Patch size: {config['patch_size']}")
    print(f"  Embed dim: {config['embed_dim']} "
          f"(= (n_market + n_latent_macro) * patch_size)")
    print(f"  Max seq len: {config['max_seq_len']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  Num heads: {config['num_heads']}")
    print(f"  FF dim multiplier: {config['ff_dim_multiplier']}")
    print(f"  Causal: {config['causal']}")
    print(f"  Use reconstruction: {config['use_reconstruction']}")
    print(f"  look_price: {config.get('look_price', True)}")
    
    _model_past_len = int(
        getattr(dataset, 'model_past_len', resolve_model_past_len(config))
    )
    model = MarketTransformerWithHeads(
        in_features=F,
        n_market_features=dataset.n_market_features,
        n_macro_features=dataset.n_macro_features,
        n_latent_macro_features=config['n_latent_macro_features'],
        n_max_past_horizon=_model_past_len,
        patch_size=config['patch_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim_multiplier=config['ff_dim_multiplier'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        causal=config['causal'],
        head_hidden_dim_multiplier=config['head_hidden_dim_multiplier'],
        output_dim=config['output_dim'],
        use_reconstruction=config['use_reconstruction'],
        num_shapes=config.get('num_shapes', 4),
        look_price=config.get('look_price', True),
    )

    # Size report + one-step gradient coverage check at start
    report_model_size(model, name="MarketTransformerWithHeads")
    _probe_T = (past_window // config['patch_size']) * config['patch_size']
    _probe_B = 2
    _probe_device = torch.device(config['device']) if config.get('device') else torch.device('cpu')
    model = model.to(_probe_device)
    _probe_P = _model_past_len
    # Concrete label-class loss (trend / yield / latent) — not a generic custom_loss.
    # All three share MarketTransformerWithHeads.
    loss_fn = bind_label_loss(
        dataset.label_mode,
        config,
        yield_target=getattr(dataset, 'yield_target', config.get('yield_target', 'both')),
        latent_distribution=getattr(
            dataset, 'latent_distribution', config.get('latent_distribution', 'normal')
        ),
    )

    check_all_weights_updated(
        model,
        sample_batch=(
            torch.randn(_probe_B, _probe_T, F, device=_probe_device),
            torch.randn(_probe_B, _probe_P, device=_probe_device),  # recent past p
            torch.randn(_probe_B, F, device=_probe_device),
            torch.randn(_probe_B, F, device=_probe_device).abs() + 1e-3,
            torch.randn(_probe_B, config.get('num_shapes', 4), F, device=_probe_device),
        ),
        loss_fn=loss_fn,
        name="MarketTransformerWithHeads",
    )

    # Create dataloaders
    train_loader = ThreadedDataLoader(dataset.train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = ThreadedDataLoader(dataset.val_dataset, batch_size=config['validate_batch_size'], shuffle=False)
    test_loader = ThreadedDataLoader(dataset.test_dataset, batch_size=config['validate_batch_size'], shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Create trainer with validation
    trainer = ResumableTrainer(
        dataset_signature=dataset.cache_key,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        dataloader=train_loader,
        save_dir=config['checkpoint_dir'],
        device=config['device'],
        save_every_n_minibatches=config['save_every_n_minibatches'],
        val_loader=val_loader,
        enable_validation=config['validate'],
        validate_every_n_epochs=config['validate_every_n_epochs'],
        embed_dim=config['embed_dim'],
        max_seq_len=config['max_seq_len'],
        recent_checkpoints=config.get(
            'recent_checkpoints', config.get('max_checkpoints', 3)
        ),
    )
    
    print("\n" + "=" * 60)
    print("TRADING MODEL TRAINING WITH TRANSFORMER")
    print("=" * 60)
    print(f"Dataset: {len(dataset)} samples, batch size: {config['batch_size']}")
    print(f"  Train: {len(dataset.train_indices)} samples")
    print(f"  Val: {len(dataset.val_indices)} samples")
    print(f"  Test: {len(dataset.test_indices)} samples")
    print(f"Past days: {config['past_days']} ({past_window} minutes)")
    print(f"Label mode: {config.get('label', 'trend')}")
    print(f"Future window: {dataset.future_window} minutes")
    print(f"Step: {config['step']} minute(s)")
    print(f"Max accum deviation: {config['max_accum_deviation']}")
    if config.get('label', 'trend') == 'yield':
        print(
            f"min_yield_percent: {config.get('min_yield_percent')}, "
            f"max_hope_future_horizon: {config.get('max_hope_future_horizon')}, "
            f"yield_target: {config.get('yield_target', getattr(dataset, 'yield_target', 'both'))}"
        )
    elif config.get('label') == 'latent':
        _ld = getattr(
            dataset, 'latent_distribution', config.get('latent_distribution', 'normal')
        )
        print(
            f"latent_distribution: {_ld}, "
            f"output_dim (density params): {config.get('output_dim')}, "
            f"shared model: MarketTransformerWithHeads"
        )
    else:
        print(
            f"min_trend_horizon: {config.get('min_trend_horizon')}, "
            f"max_trend_future_horizon: {config.get('max_trend_future_horizon')}"
        )
    print(f"Normalize: {config['normalize']}")
    print(f"Epochs: {config['epochs']}")
    if args.extend is not None and args.extend != 0:
        sign = '+' if args.extend > 0 else ''
        print(f"  (Will adjust plan by {sign}{args.extend} epochs)")
    print(f"Minibatches per epoch: {math.ceil(len(dataset.train_indices) / config['batch_size'])}")
    print(f"Model: Transformer with {config['num_layers']} layers, {config['num_heads']} heads")
    print(f"Device: {trainer.device}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Slope loss weight: {config['slope_loss_weight']}")
    print(f"Intercept loss weight: {config['intercept_loss_weight']}")
    print(f"Reconstruction loss weight: {config['reconstruction_loss_weight']}")
    print(f"Save every: {config['save_every_n_minibatches']} minibatches")
    print(f"Checkpoints: {config['checkpoint_dir']}/")
    print(f"Validation: {'Enabled' if config['validate'] else 'Disabled'}")
    if config['validate']:
        print(f"  Validate every {config['validate_every_n_epochs']} epochs")
        print(f"  Val batch size: {config['validate_batch_size']}")
    print("=" * 60)
    print()
    
    # ============================================================
    # STEP 5: START TRAINING - ALL logic is in trainer
    # ============================================================
    trainer.train(
        num_epochs=config['epochs'],
        resume=config['resume'],
        extend=(args.extend is not None and args.extend != 0),
        extend_epochs=args.extend if args.extend is not None else 0.0,
    )


if __name__ == "__main__":

    from config import get_config

    main(get_config())