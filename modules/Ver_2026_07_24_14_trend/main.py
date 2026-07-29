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
from data import TradingDataset, ThreadedDataLoader
from model import MarketTransformerWithHeads, custom_loss
from trainer import ResumableTrainer

# Try to import Binance
try:
    from Legacy.Binance_new import Binance
    print("Successfully imported Binance from Legacy.Binance_new")
except ImportError:
    try:
        import Legacy.Binance_new as BinanceModule
        Binance = BinanceModule.Binance
        print("Successfully imported Binance via alternative method")
    except ImportError:
        print("Failed to import Legacy. Will use synthetic data only.")
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
    """Get data from Binance using the configured parameters."""
    if Binance is None:
        print("⚠ Binance not available.")
        return None, None, None, None
    
    try:
        bn = Binance.instantiate(connect=False)
        
        datatype = config['binance_datatype']
        interval = config['binance_interval']
        start_date = config['binance_start_date']
        end_date = config['binance_end_date']
        symbol_dict = config['binance_symbols']
        target = config['prediction_target']
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        print(f"Fetching Binance data:")
        print(f"  Datatype: {datatype}")
        print(f"  Interval: {interval}")
        print(f"  Start: {start_date}")
        print(f"  End: {end_date}")
        print(f"  Symbols: {list(symbol_dict.keys())}")

        features, nForget, target_feature_id = bn.get_combined_history_2(
            datatype, interval, start, end, symbol_dict, target
        )
        
        T = features[0].shape[0] if len(features) > 0 else 0
        for feature in features:
            assert feature.shape[0] == T, "All features must have the same number of time points."
        
        print(f"\nBinance data loaded:")
        print(f"  Number of features: {len(features)}")
        print(f"  Time points: {T}")
        print(f"  Forgotten points: {nForget}")
        print(f"  Target feature Id: {target_feature_id}")

        return features, T, nForget, target_feature_id
        
    except Exception as e:
        print(f"⚠ Error fetching Binance data: {e}")
        return None, None, None, None


def main():
    """Main training function."""
    config = get_config()
    
    parser = argparse.ArgumentParser(description='Resilient Trading Training with Transformer')
    parser.add_argument('--epochs', type=int, default=config['epochs'], help='Number of epochs')
    parser.add_argument('--save-every', type=int, default=config['save_every_n_minibatches'],
                       help='Save every N minibatches')
    parser.add_argument('--clean', action='store_true', help='Delete old checkpoints')
    parser.add_argument('--extend', type=int, default=None,
                       help='Extend training by N additional epochs (can be used multiple times)')
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
    
    # Split arguments
    parser.add_argument('--train-ratio', type=float, default=config.get('train_ratio', 0.70),
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=config.get('val_ratio', 0.20),
                       help='Validation set ratio')
    
    # Validation arguments
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
    # STEP 1: Handle --clean flag
    # ============================================================
    checkpoint_dir = Path(config['checkpoint_dir'])
    if config['clean'] and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print(f"✓ Cleaned previous checkpoints from: {config['checkpoint_dir']}\n")
    
    # ============================================================
    # STEP 2: Handle --extend flag (can be used multiple times)
    # ============================================================
    extend_epochs = args.extend if args.extend is not None else 0
    
    if extend_epochs > 0:
        # Read the checkpoint to get actual completed epochs
        completed_epochs = 0
        state_file = checkpoint_dir / 'training_state.json'
        
        if not config['clean'] and state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                completed_epochs = state.get('epoch', 0) + 1  # 0-indexed to 1-indexed
                print(f"✓ Found checkpoint with {completed_epochs} completed epochs.")
            except Exception as e:
                print(f"⚠ Could not read checkpoint: {e}")
                completed_epochs = config['epochs']
        else:
            # If no checkpoint or --clean, use the epochs value
            completed_epochs = config['epochs']
        
        # Set new total epochs: completed + extend
        new_total = completed_epochs + extend_epochs
        config['epochs'] = new_total
        
        print(f"\n{'='*60}")
        print(f"⚠ EXTENDING TRAINING BY {extend_epochs} EPOCHS")
        print(f"{'='*60}")
        print(f"  Completed epochs: {completed_epochs}")
        print(f"  Adding: {extend_epochs} epochs")
        print(f"  New total epochs: {new_total}")
        print(f"{'='*60}\n")
    
    # ============================================================
    # STEP 3: Check if training is already finished (skip if --clean)
    # ============================================================
    training_finished = False
    completed_epochs = 0
    
    if config['resume'] and not config['clean'] and checkpoint_dir.exists():
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(1, 1)
            def forward(self, x, mean=None, std=None, shape=None):
                return self.fc(x)
        
        class DummyDataset:
            def __len__(self): return 1
            def __getitem__(self, idx): 
                return torch.zeros(1, 1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1, 4)
        
        class DummyLoader:
            def __init__(self):
                self.dataset = DummyDataset()
                self.batch_size = 1
                self.current_epoch_indices = [0]
                self.current_batch_start = 0
            def start_epoch(self, epoch, resume_indices=None, resume_batch_start=0): pass
            def get_total_batches_in_epoch(self): return 1
            def get_next_batch(self): return None, 0, 0
            def get_state(self): return {}
            def restore_state(self, state): pass
        
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
        
        if dummy_trainer.state_manager.is_training_finished():
            training_finished = True
            
            # Read the actual completed epochs from checkpoint
            try:
                state_file = checkpoint_dir / 'training_state.json'
                if state_file.exists():
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                    completed_epochs = state.get('epoch', 0) + 1  # 0-indexed to 1-indexed
                else:
                    completed_epochs = config['epochs']
            except:
                completed_epochs = config['epochs']
            
            if extend_epochs > 0:
                print(f"⚠ Training was finished ({completed_epochs} epochs), but --extend was provided.")
                print("  Continuing with extension...\n")
                dummy_trainer.state_manager.clear_finished_flag()
            else:
                print("\n" + "=" * 60)
                print("✓ TRAINING ALREADY FINISHED")
                print("=" * 60)
                print(f"All {completed_epochs} epochs completed in previous run.")
                print("Nothing to train.")
                print("Use --clean to start fresh, or --extend N to add N more epochs.")
                print("=" * 60)
                return
    
    # ============================================================
    # STEP 4: Get data (Binance or Synthetic)
    # ============================================================
    features = None
    target_feature_id = None
    
    if config['use_binance_data'] and Binance is not None:
        features, T, nForget, target_feature_id = get_binance_data(config)
        if features is None or target_feature_id is None:
            print("⚠ Failed to fetch Binance data. Falling back to synthetic data.")
    
    if features is None:
        T = 60 * 24 * config['total_synthetic_days']
        F = config['num_features']
        print(f"Generating synthetic trading data: {T} minutes, {F} features")
        features = generate_synthetic_data(T, F, config['price_feature_idx'])
        target_feature_id = config['price_feature_idx']
    
    config['num_features'] = len(features)
    print(f"Using device: {config['device']}")
    
    # ============================================================
    # STEP 5: Create dataset and train
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
        validate_every_n_epochs=config['validate_every_n_epochs']
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
    if extend_epochs > 0:
        print(f"  (Extended by {extend_epochs} epochs)")
    if training_finished and extend_epochs > 0:
        print(f"  (Resuming finished training with extension)")
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
    
    state_file = Path(config['checkpoint_dir']) / 'training_state.json'
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            print(f"Found previous checkpoint at minibatch {state['total_minibatches_processed']}")
            print(f"Epoch {state['epoch']+1}, batch {state['batch_idx']}")
            if state.get('val_loss') is not None:
                print(f"Val Loss: {state['val_loss']:.4f}")
        except:
            pass
    else:
        print("No previous checkpoint - fresh start")
    
    print()
    
    # ============================================================
    # START TRAINING with extend flag
    # ============================================================
    trainer.train(
        num_epochs=config['epochs'], 
        resume=config['resume'],
        extend=(extend_epochs > 0)
    )


if __name__ == "__main__":
    main()