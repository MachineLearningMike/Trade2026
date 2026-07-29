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

from config import get_config
from data import TradingDataset, ThreadedDataLoader, get_binance_data
from model import MarketTransformerWithHeads, custom_loss
from trainer import ResumableTrainer


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


def main():
    """Main training function."""
    config = get_config()
    
    # Ensure price_feature_idx exists
    if 'price_feature_idx' not in config:
        config['price_feature_idx'] = 0
    
    parser = argparse.ArgumentParser(description='Resilient Trading Training with Transformer')
    parser.add_argument('--epochs', type=int, default=config['epochs'], help='Number of epochs')
    parser.add_argument('--save-every', type=int, default=config['save_every_n_minibatches'],
                       help='Save every N minibatches')
    parser.add_argument('--clean', action='store_true', help='Delete old checkpoints')
    parser.add_argument('--extend', type=int, default=None,
                       help='Extend training by N additional epochs (can be used multiple times)')
    parser.add_argument('--trend', action='store_true', default=False,
                       help='Generate and visualize trends without training')
    parser.add_argument('--past-days', type=int, default=config['past_days'], help='Past days for input')
    parser.add_argument('--future-minutes', type=int, default=config['future_minutes'],
                       help='Future minutes for prediction')
    parser.add_argument('--batch-size', type=int, default=config['batch_size'], help='Batch size')
    parser.add_argument('--num-features', type=int, default=config['num_features'], help='Number of features')
    parser.add_argument('--total-days', type=int, default=config['total_synthetic_days'], help='Total days of data')
    parser.add_argument('--step', type=int, default=config['step'], help='Step size for start indices')
    parser.add_argument('--max-accum-deviation', type=float, default=config['max_accum_deviation'],
                       help='Threshold for cumulative absolute deviation')
    parser.add_argument('--min-horizon', type=int, default=config['min_horizon'],
                       help='Minimum horizon to consider')
    parser.add_argument('--max-horizon', type=int, default=config['max_horizon'],
                       help='Maximum horizon to consider')
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
    parser.add_argument('--ff-dim', type=int, default=config['ff_dim'],
                       help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=config['dropout'], help='Dropout rate')
    parser.add_argument('--head-hidden-dim', type=int, default=config['head_hidden_dim'],
                       help='Hidden dimension for prediction heads')
    parser.add_argument('--no-causal', action='store_true', default=not config['causal'],
                       help='Disable causal attention')
    parser.add_argument('--no-reconstruction', action='store_true', default=not config['use_reconstruction'],
                       help='Disable reconstruction auxiliary loss')
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
    config['future_minutes'] = args.future_minutes
    config['batch_size'] = args.batch_size
    config['num_features'] = args.num_features
    config['total_synthetic_days'] = args.total_days
    config['step'] = args.step
    config['max_accum_deviation'] = args.max_accum_deviation
    config['min_horizon'] = args.min_horizon
    config['max_horizon'] = args.max_horizon
    config['learning_rate'] = args.lr
    config['slope_loss_weight'] = args.slope_weight
    config['intercept_loss_weight'] = args.intercept_weight
    config['reconstruction_loss_weight'] = args.recon_weight
    config['patch_size'] = args.patch_size
    config['num_layers'] = args.num_layers
    config['num_heads'] = args.num_heads
    config['ff_dim'] = args.ff_dim
    config['dropout'] = args.dropout
    config['head_hidden_dim'] = args.head_hidden_dim
    config['causal'] = not args.no_causal
    config['use_reconstruction'] = not args.no_reconstruction
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
        print("Generating/loading dataset and visualizing trends...")
        print("=" * 60 + "\n")
        
        # Get data
        features = None
        target_feature_id = None
        
        if config['use_binance_data']:
            features, T, nForget, target_feature_id = get_binance_data(config)
            if features is None or target_feature_id is None:
                print("⚠ Failed to fetch Binance data. Falling back to synthetic data.")
        
        if features is None:
            T = 60 * 24 * config['total_synthetic_days']
            F = config['num_features']
            print(f"Generating synthetic trading data: {T} minutes, {F} features")
            price_idx = config.get('price_feature_idx', 0)
            features = generate_synthetic_data(T, F, price_idx)
            target_feature_id = price_idx
        
        # Create dataset without automatic visualization
        dataset = TradingDataset(
            features=features,
            price_feature_idx=target_feature_id,
            past_days=config['past_days'],
            future_minutes=config['future_minutes'],
            step=config['step'],
            max_accum_deviation=config['max_accum_deviation'],
            min_horizon=config['min_horizon'],
            max_horizon=config['max_horizon'],
            device=config['device'],
            normalize=config['normalize'],
            cache_dir=config.get('cache_dir', 'dataset_cache'),
            force_recompute=config.get('force_recompute', False),
            split_ratios=(0.7, 0.2, 0.1)
        )
        
        # Explicitly visualize trends
        dataset.visualize_trends(num_samples=6)
        
        print("\n✅ Trend visualization completed!")
        return
    
    # ============================================================
    # STEP 2: Handle --clean flag
    # ============================================================
    checkpoint_dir = Path(config['checkpoint_dir'])
    if config['clean'] and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print(f"✓ Cleaned previous checkpoints from: {config['checkpoint_dir']}\n")
    
    # ============================================================
    # STEP 3: Get data (Binance or Synthetic)
    # ============================================================
    features = None
    target_feature_id = None
    
    if config['use_binance_data']:
        features, T, nForget, target_feature_id = get_binance_data(config)
        if features is None or target_feature_id is None:
            print("⚠ Failed to fetch Binance data. Falling back to synthetic data.")
    
    if features is None:
        T = 60 * 24 * config['total_synthetic_days']
        F = config['num_features']
        print(f"Generating synthetic trading data: {T} minutes, {F} features")
        price_idx = config.get('price_feature_idx', 0)
        features = generate_synthetic_data(T, F, price_idx)
        target_feature_id = price_idx
    
    config['num_features'] = len(features)
    print(f"Using device: {config['device']}")
    
    # ============================================================
    # STEP 4: Create dataset and train
    # ============================================================
    train_ratio = config.get('train_ratio', 0.70)
    val_ratio = config.get('val_ratio', 0.20)
    test_ratio = 1.0 - train_ratio - val_ratio
    split_ratios = (train_ratio, val_ratio, test_ratio)
    
    dataset = TradingDataset(
        features=features,
        price_feature_idx=target_feature_id,
        past_days=config['past_days'],
        future_minutes=config['future_minutes'],
        step=config['step'],
        max_accum_deviation=config['max_accum_deviation'],
        min_horizon=config['min_horizon'],
        max_horizon=config['max_horizon'],
        device=config['device'],
        normalize=config['normalize'],
        cache_dir=config.get('cache_dir', 'dataset_cache'),
        force_recompute=config.get('force_recompute', False),
        split_ratios=split_ratios
    )
    
    past_window = 60 * 24 * config['past_days']
    F = len(features)
    
    base_embed_dim = F * config['patch_size']
    while base_embed_dim % config['num_heads'] != 0:
        base_embed_dim += 1
    config['embed_dim'] = base_embed_dim
    config['max_seq_len'] = past_window // config['patch_size']
    
    print(f"\nData shapes:")
    print(f"  Input x: (F={F}, past_window={past_window})")
    print(f"  Output: (horizon, slope, intercept) - 3 values")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Train samples: {len(dataset.train_indices)}")
    print(f"  Val samples: {len(dataset.val_indices)}")
    print(f"  Test samples: {len(dataset.test_indices)}")
    print(f"\nTransformer config:")
    print(f"  Patch size: {config['patch_size']}")
    print(f"  Embed dim: {config['embed_dim']}")
    print(f"  Max seq len: {config['max_seq_len']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  Num heads: {config['num_heads']}")
    print(f"  FF dim: {config['ff_dim']}")
    print(f"  Causal: {config['causal']}")
    print(f"  Use reconstruction: {config['use_reconstruction']}")
    
    model = MarketTransformerWithHeads(
        in_features=F,
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        causal=config['causal'],
        head_hidden_dim=config['head_hidden_dim'],
        output_dim=config['output_dim'],
        use_reconstruction=config['use_reconstruction']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Create dataloaders
    train_loader = ThreadedDataLoader(dataset.train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = ThreadedDataLoader(dataset.val_dataset, batch_size=config['validate_batch_size'], shuffle=False)
    test_loader = ThreadedDataLoader(dataset.test_dataset, batch_size=config['validate_batch_size'], shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Create trainer with validation
    trainer = ResumableTrainer(
        model=model,
        loss_fn=lambda P, X, Y: custom_loss(
            P, X, Y,
            slope_weight=config['slope_loss_weight'],
            intercept_weight=config['intercept_loss_weight'],
            recon_weight=config['reconstruction_loss_weight']
        ),
        optimizer=optimizer,
        dataloader=train_loader,
        save_dir=config['checkpoint_dir'],
        device=config['device'],
        save_every_n_minibatches=config['save_every_n_minibatches'],
        val_loader=val_loader,
        enable_validation=config['validate'],
        validate_every_n_epochs=config['validate_every_n_epochs'],
        embed_dim=config['embed_dim'],
        max_seq_len=config['max_seq_len']
    )
    
    print("\n" + "=" * 60)
    print("TRADING MODEL TRAINING WITH TRANSFORMER")
    print("=" * 60)
    print(f"Dataset: {len(dataset)} samples, batch size: {config['batch_size']}")
    print(f"  Train: {len(dataset.train_indices)} samples")
    print(f"  Val: {len(dataset.val_indices)} samples")
    print(f"  Test: {len(dataset.test_indices)} samples")
    print(f"Past days: {config['past_days']} ({past_window} minutes)")
    print(f"Future minutes: {config['future_minutes']}")
    print(f"Step: {config['step']} minute(s)")
    print(f"Max accum deviation: {config['max_accum_deviation']}")
    print(f"Min horizon: {config['min_horizon']}, Max horizon: {config['max_horizon']}")
    print(f"Normalize: {config['normalize']}")
    print(f"Epochs: {config['epochs']}")
    if args.extend is not None and args.extend > 0:
        print(f"  (Will extend by {args.extend} epochs)")
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
        extend=(args.extend is not None and args.extend > 0),
        extend_epochs=args.extend if args.extend is not None else 0
    )


if __name__ == "__main__":
    main()