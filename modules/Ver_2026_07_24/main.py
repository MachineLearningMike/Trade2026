# main.py
# Main training script

import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import argparse
import shutil

from config import get_config, update_config
from data import TradingDataset, ThreadedDataLoader
from model import MarketTransformerWithHeads, custom_loss
from trainer import ResumableTrainer


def generate_synthetic_data(T: int, F: int, price_feature_idx: int = 0):
    """Generate synthetic trading data for testing."""
    features = []
    for i in range(F):
        if i == price_feature_idx:  # Price feature - random walk with drift
            price = np.cumsum(np.random.randn(T) * 0.5 + 0.1)
            features.append(price.astype(np.float32))
        elif i == 1:  # Volume - positive
            volume = np.abs(np.random.randn(T) * 100 + 1000)
            features.append(volume.astype(np.float32))
        else:  # Other features
            features.append(np.random.randn(T).astype(np.float32))
    return features


def main():
    """Main training function."""
    
    # Get config
    config = get_config()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Resilient Trading Training with Transformer')
    parser.add_argument('--epochs', type=int, default=config['epochs'], help='Number of epochs')
    parser.add_argument('--save-every', type=int, default=config['save_every_n_minibatches'], 
                       help='Save every N minibatches')
    parser.add_argument('--clean', action='store_true', default=config['clean'], 
                       help='Delete old checkpoints')
    parser.add_argument('--past-days', type=int, default=config['past_days'], 
                       help='Past days for input')
    parser.add_argument('--future-minutes', type=int, default=config['future_minutes'], 
                       help='Future minutes for prediction')
    parser.add_argument('--batch-size', type=int, default=config['batch_size'], 
                       help='Batch size')
    parser.add_argument('--num-features', type=int, default=config['num_features'], 
                       help='Number of features')
    parser.add_argument('--total-days', type=int, default=config['total_days'], 
                       help='Total days of data')
    parser.add_argument('--step', type=int, default=config['step'], 
                       help='Step size for start indices')
    parser.add_argument('--max-accum-deviation', type=float, default=config['max_accum_deviation'], 
                       help='Threshold for cumulative absolute deviation')
    parser.add_argument('--min-horizon', type=int, default=config['min_horizon'], 
                       help='Minimum horizon to consider')
    parser.add_argument('--max-horizon', type=int, default=config['max_horizon'], 
                       help='Maximum horizon to consider')
    parser.add_argument('--no-normalize', action='store_true', 
                       help='Disable per-window normalization')
    parser.add_argument('--lr', type=float, default=config['learning_rate'], 
                       help='Learning rate')
    parser.add_argument('--slope-weight', type=float, default=config['slope_loss_weight'], 
                       help='Weight for slope loss')
    parser.add_argument('--recon-weight', type=float, default=config['reconstruction_loss_weight'], 
                       help='Weight for reconstruction loss')
    
    # Transformer-specific arguments
    parser.add_argument('--patch-size', type=int, default=config['patch_size'], 
                       help='Patch size for temporal patching')
    parser.add_argument('--num-layers', type=int, default=config['num_layers'], 
                       help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=config['num_heads'], 
                       help='Number of attention heads')
    parser.add_argument('--ff-dim', type=int, default=config['ff_dim'], 
                       help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=config['dropout'], 
                       help='Dropout rate')
    parser.add_argument('--head-hidden-dim', type=int, default=config['head_hidden_dim'], 
                       help='Hidden dimension for prediction heads')
    parser.add_argument('--no-causal', action='store_true', 
                       help='Disable causal attention')
    parser.add_argument('--no-reconstruction', action='store_true', 
                       help='Disable reconstruction auxiliary loss')
    
    args = parser.parse_args()
    
    # Update config with command-line arguments
    config['epochs'] = args.epochs
    config['save_every_n_minibatches'] = args.save_every
    config['clean'] = args.clean
    config['past_days'] = args.past_days
    config['future_minutes'] = args.future_minutes
    config['batch_size'] = args.batch_size
    config['num_features'] = args.num_features
    config['total_days'] = args.total_days
    config['step'] = args.step
    config['max_accum_deviation'] = args.max_accum_deviation
    config['min_horizon'] = args.min_horizon
    config['max_horizon'] = args.max_horizon
    config['learning_rate'] = args.lr
    config['slope_loss_weight'] = args.slope_weight
    config['reconstruction_loss_weight'] = args.recon_weight
    
    # Update transformer config
    config['patch_size'] = args.patch_size
    config['num_layers'] = args.num_layers
    config['num_heads'] = args.num_heads
    config['ff_dim'] = args.ff_dim
    config['dropout'] = args.dropout
    config['head_hidden_dim'] = args.head_hidden_dim
    config['causal'] = not args.no_causal
    config['use_reconstruction'] = not args.no_reconstruction
    
    if args.no_normalize:
        config['normalize'] = False
    
    # Clean start if requested
    if config['clean'] and Path(config['checkpoint_dir']).exists():
        shutil.rmtree(config['checkpoint_dir'])
        print("Cleaned previous checkpoints.\n")
    
    # Create synthetic trading data
    T = 60 * 24 * config['total_days']  # total_days of minute data
    F = config['num_features']
    
    print(f"Generating synthetic trading data: {T} minutes, {F} features")
    print(f"Using device: {config['device']}")
    
    features = generate_synthetic_data(T, F, config['price_feature_idx'])
    
    # Create dataset with per-window normalization
    dataset = TradingDataset(
        features=features,
        price_feature_idx=config['price_feature_idx'],
        past_days=config['past_days'],
        future_minutes=config['future_minutes'],
        step=config['step'],
        max_accum_deviation=config['max_accum_deviation'],
        min_horizon=config['min_horizon'],
        max_horizon=config['max_horizon'],
        device=config['device'],
        normalize=config['normalize']
    )
    
    # Calculate dimensions
    past_window = 60 * 24 * config['past_days']
    future_window = config['future_minutes']
    F = len(features)
    
    # Compute transformer dimensions - ensure embed_dim is divisible by num_heads
    base_embed_dim = F * config['patch_size']
    # Make sure embed_dim is divisible by num_heads
    while base_embed_dim % config['num_heads'] != 0:
        base_embed_dim += 1
    config['embed_dim'] = base_embed_dim
    config['max_seq_len'] = past_window // config['patch_size']
    
    print(f"\nData shapes:")
    print(f"  Input x: (F={F}, past_window={past_window})")
    print(f"  Output: (horizon, slope) - 2 values")
    print(f"  Total samples: {len(dataset)}")
    print(f"\nTransformer config:")
    print(f"  Patch size: {config['patch_size']}")
    print(f"  Embed dim: {config['embed_dim']} (adjusted for divisibility by num_heads)")
    print(f"  Max seq len: {config['max_seq_len']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  Num heads: {config['num_heads']}")
    print(f"  FF dim: {config['ff_dim']}")
    print(f"  Causal: {config['causal']}")
    print(f"  Use reconstruction: {config['use_reconstruction']}")
    
    # Create model
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
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Create dataloader
    dataloader = ThreadedDataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Create trainer
    trainer = ResumableTrainer(
        model=model,
        loss_fn=lambda P, X, Y: custom_loss(
            P, X, Y, 
            slope_weight=config['slope_loss_weight'],
            recon_weight=config['reconstruction_loss_weight']
        ),
        optimizer=optimizer,
        dataloader=dataloader,
        save_dir=config['checkpoint_dir'],
        device=config['device'],
        save_every_n_minibatches=config['save_every_n_minibatches']
    )
    
    # Show status
    print("\n" + "=" * 60)
    print("TRADING MODEL TRAINING WITH TRANSFORMER")
    print("=" * 60)
    print(f"Dataset: {len(dataset)} samples, batch size: {config['batch_size']}")
    print(f"Past days: {config['past_days']} ({past_window} minutes)")
    print(f"Future minutes: {config['future_minutes']}")
    print(f"Step: {config['step']} minute(s)")
    print(f"Max accum deviation: {config['max_accum_deviation']}")
    print(f"Min horizon: {config['min_horizon']}, Max horizon: {config['max_horizon']}")
    print(f"Normalize: {config['normalize']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Minibatches per epoch: {math.ceil(len(dataset) / config['batch_size'])}")
    print(f"Model: Transformer with {config['num_layers']} layers, {config['num_heads']} heads")
    print(f"Device: {trainer.device}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Slope loss weight: {config['slope_loss_weight']}")
    print(f"Reconstruction loss weight: {config['reconstruction_loss_weight']}")
    print(f"Save every: {config['save_every_n_minibatches']} minibatches")
    print(f"Checkpoints: {config['checkpoint_dir']}/")
    print("=" * 60)
    
    # Check for existing checkpoint
    state_file = Path(config['checkpoint_dir']) / 'training_state.json'
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            print(f"Found previous checkpoint at minibatch {state['total_minibatches_processed']}")
            print(f"Epoch {state['epoch']+1}, batch {state['batch_idx']}")
        except:
            pass
    else:
        print("No previous checkpoint - fresh start")
    
    print()
    
    # Train
    trainer.train(num_epochs=config['epochs'], resume=config['resume'])


if __name__ == "__main__":
    main()