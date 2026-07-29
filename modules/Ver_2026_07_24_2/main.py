# main.py
# Main training script

import sys
import os
from pathlib import Path

# Add parent directory to Python path for Legacy imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
grandparent_dir = parent_dir.parent

sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(grandparent_dir))
sys.path.insert(0, str(current_dir))

import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import argparse
import shutil
from datetime import datetime

from config import get_config, update_config
from data import TradingDataset, ThreadedDataLoader
from model import MarketTransformerWithHeads, custom_loss
from trainer import ResumableTrainer

# Try to import Binance
try:
    from Legacy.Binance_new import Binance
    print("Successfully imported Binance from Legacy.Binance_new")
except ImportError as e:
    print(f"Failed to import from Legacy.Binance_new: {e}")
    try:
        import Legacy.Binance_new as BinanceModule
        Binance = BinanceModule.Binance
        print("Successfully imported Binance via alternative method")
    except ImportError as e2:
        print(f"Failed to import Legacy: {e2}")
        print("Will use synthetic data only.")
        Binance = None


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


def get_binance_data(config):
    """
    Get data from Binance using the configured parameters.
    Returns features list and metadata.
    """

    if Binance is None:
        print("⚠ Binance not available.")
        return None, None, None
    
    try:
        # Instantiate Binance
        bn = Binance.instantiate(connect=False)
        
        # Get parameters from config
        datatype = config['binance_datatype']
        interval = config['binance_interval']
        start_date = config['binance_start_date']
        end_date = config['binance_end_date']
        symbol_dict = config['binance_symbols']
        
        # Parse dates
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        print(f"Fetching Binance data:")
        print(f"  Datatype: {datatype}")
        print(f"  Interval: {interval}")
        print(f"  Start: {start_date}")
        print(f"  End: {end_date}")
        print(f"  Symbols: {list(symbol_dict.keys())}")
        
        # Get combined history
        features, nForget = bn.get_combined_history(datatype, interval, start, end, symbol_dict)
        
        # Validate features
        T = features[0].shape[0] if len(features) > 0 else 0
        for feature in features:
            assert feature.shape[0] == T, "All features must have the same number of time points."
        
        print(f"\nBinance data loaded:")
        print(f"  Number of features: {len(features)}")
        print(f"  Time points: {T}")
        print(f"  Forgotten points: {nForget}")
        
        return features, T, nForget
        
    except Exception as e:
        print(f"⚠ Error fetching Binance data: {e}")
        return None, None, None


def main():
    """Main training function."""
    
    config = get_config()
    
    parser = argparse.ArgumentParser(description='Resilient Trading Training with Transformer')
    parser.add_argument('--epochs', type=int, default=config['epochs'], help='Number of epochs')
    parser.add_argument('--save-every', type=int, default=config['save_every_n_minibatches'], 
                       help='Save every N minibatches')
    parser.add_argument('--clean', action='store_true', help='Delete old checkpoints')
    parser.add_argument('--past-days', type=int, default=config['past_days'], help='Past days for input')
    parser.add_argument('--future-minutes', type=int, default=config['future_minutes'], help='Future minutes for prediction')
    parser.add_argument('--batch-size', type=int, default=config['batch_size'], help='Batch size')
    parser.add_argument('--num-features', type=int, default=config['num_features'], help='Number of features')
    parser.add_argument('--total-days', type=int, default=config['total_synthetic_days'], help='Total days of data')
    parser.add_argument('--step', type=int, default=config['step'], help='Step size for start indices')
    parser.add_argument('--max-accum-deviation', type=float, default=config['max_accum_deviation'], help='Threshold for cumulative absolute deviation')
    parser.add_argument('--min-horizon', type=int, default=config['min_horizon'], help='Minimum horizon to consider')
    parser.add_argument('--max-horizon', type=int, default=config['max_horizon'], help='Maximum horizon to consider')
    parser.add_argument('--no-normalize', action='store_true', help='Disable per-window normalization')
    parser.add_argument('--lr', type=float, default=config['learning_rate'], help='Learning rate')
    parser.add_argument('--slope-weight', type=float, default=config['slope_loss_weight'], help='Weight for slope loss')
    parser.add_argument('--recon-weight', type=float, default=config['reconstruction_loss_weight'], help='Weight for reconstruction loss')
    parser.add_argument('--patch-size', type=int, default=config['patch_size'], help='Patch size for temporal patching')
    parser.add_argument('--num-layers', type=int, default=config['num_layers'], help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=config['num_heads'], help='Number of attention heads')
    parser.add_argument('--ff-dim', type=int, default=config['ff_dim'], help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=config['dropout'], help='Dropout rate')
    parser.add_argument('--head-hidden-dim', type=int, default=config['head_hidden_dim'], help='Hidden dimension for prediction heads')
    parser.add_argument('--no-causal', action='store_true', help='Disable causal attention')
    parser.add_argument('--no-reconstruction', action='store_true', help='Disable reconstruction auxiliary loss')
    
    # Binance data arguments
    parser.add_argument('--use-binance', action='store_true', default=config['use_binance_data'], 
                       help='Use real Binance data instead of synthetic')
    parser.add_argument('--symbol', type=str, default=config['binance_primary_symbol'], 
                       help='Primary trading pair symbol')
    
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
    
    if args.no_normalize:
        config['normalize'] = False
    
    # ============================================================
    # STEP 1: Handle --clean flag
    # ============================================================
    checkpoint_dir = Path(config['checkpoint_dir'])
    if config['clean']:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            print(f"✓ Cleaned previous checkpoints from: {config['checkpoint_dir']}\n")
        else:
            print(f"✓ No checkpoints to clean at: {config['checkpoint_dir']}\n")
    
    # ============================================================
    # STEP 2: Check if training is already finished (skip if --clean)
    # ============================================================
    if config['resume'] and not config['clean'] and checkpoint_dir.exists():
        # Create a lightweight trainer just to check training status
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(1, 1)
            def forward(self, x):
                return self.fc(x)
        
        class DummyDataset:
            def __len__(self): return 1
            def __getitem__(self, idx): return torch.zeros(1, 1), torch.zeros(1)
        
        class DummyLoader:
            def __init__(self):
                self.dataset = DummyDataset()
                self.batch_size = 1
                self.current_epoch_indices = [0]
                self.current_batch_start = 0
            def start_epoch(self, epoch, resume_indices=None, resume_batch_start=0):
                pass
            def get_total_batches_in_epoch(self):
                return 1
            def get_next_batch(self):
                return None, 0, 0
            def get_state(self):
                return {}
            def restore_state(self, state):
                pass
        
        dummy_model = DummyModel()
        dummy_optimizer = optim.Adam(dummy_model.parameters(), lr=0.001)
        dummy_dataloader = DummyLoader()
        dummy_loss_fn = lambda P, X, Y: torch.tensor(0.0)
        
        dummy_trainer = ResumableTrainer(
            model=dummy_model,
            loss_fn=dummy_loss_fn,
            optimizer=dummy_optimizer,
            dataloader=dummy_dataloader,
            save_dir=config['checkpoint_dir'],
            device='cpu',
            save_every_n_minibatches=config['save_every_n_minibatches']
        )
        
        # Check if training is already finished
        if dummy_trainer.state_manager.is_training_finished():
            print("\n" + "="*60)
            print("✓ TRAINING ALREADY FINISHED")
            print("="*60)
            print(f"All {config['epochs']} epochs completed in previous run.")
            print("Nothing to train.")
            print("Use --clean to start fresh, or increase --epochs.")
            print("="*60)
            return
    
    # ============================================================
    # STEP 3: Get data (Binance or Synthetic)
    # ============================================================
    
    features = None
    
    if config['use_binance_data'] and Binance is not None:
        features, T, nForget = get_binance_data(config)
        if features is None:
            print("⚠ Failed to fetch Binance data. Falling back to synthetic data.")
        print(f"\nUsing {len(features) if features else 0} features for training.-----------------------------------------------------------------")
    
    # If Binance data not available, use synthetic
    if features is None:
        T = 60 * 24 * config['total_synthetic_days']
        F = config['num_features']
        print(f"Generating synthetic trading data: {T} minutes, {F} features")
        features = generate_synthetic_data(T, F, config['price_feature_idx'])
    
    # Update num_features to match actual data
    config['num_features'] = len(features)
    
    print(f"Using device: {config['device']}")
    
    # ============================================================
    # STEP 4: Create dataset and train
    # ============================================================
    
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
        normalize=config['normalize'],
        force_recompute=config.get('force_recompute', False)
    )
    
    # Calculate dimensions
    past_window = 60 * 24 * config['past_days']
    future_window = config['future_minutes']
    F = len(features)
    
    # Compute transformer dimensions
    base_embed_dim = F * config['patch_size']
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