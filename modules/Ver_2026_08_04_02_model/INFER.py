# INFER.py
# Inference script for trained model

import torch
import torch.nn as nn
import numpy as np
import json
import time
import random
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from config import get_config
from model import MarketTransformerWithHeads
from data import TradingDataset, normalize_window, get_binance_data, generate_synthetic_data
from trainer import generate_checkpoint_key


# ═══════════════════════════════════════════════════════════
# VISUALIZATION FUNCTION (copied from data.py)
# ═══════════════════════════════════════════════════════════

def visualize_linear_trends(
    total_tensor: torch.Tensor,
    valid_starts: List[int],
    label_tensor: torch.Tensor,
    mean_tensor: torch.Tensor,
    std_tensor: torch.Tensor,
    price_feature_idx: int,
    past_window: int,
    future_window: int,
    mode: str = 'anchored',
    num_samples: int = 6,
    figsize: Tuple[int, int] = (15, 10)
):
    """Visualize linear trends for randomly selected samples."""
    if total_tensor.is_cuda:
        total_tensor = total_tensor.cpu()
    
    num_available = label_tensor.shape[0]
    num_samples = min(num_samples, num_available)
    if num_samples % 2 != 0:
        num_samples = 6

    # Set seed based on current second from epoch
    seed = int(time.time())  # Current second
    random.seed(seed)
    
    indices = random.sample(range(num_available), num_samples)
    n_cols = 3
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
        
        ax = axes[i]
        start = valid_starts[idx]
        label = label_tensor[idx]
        
        # Fix: Ensure mean and std are 1D by squeezing and flattening
        mean = mean_tensor[idx]
        std = std_tensor[idx]
        
        # If mean has more than 1 dimension, squeeze it
        if mean.dim() > 1:
            mean = mean.squeeze()
        if std.dim() > 1:
            std = std.squeeze()
        
        # If still not 1D, flatten
        if mean.dim() != 1:
            mean = mean.flatten()
        if std.dim() != 1:
            std = std.flatten()
        
        horizon = int(label[0].item())
        slope_norm = label[1].item()
        intercept_norm = label[2].item()
        
        price_mean = mean[price_feature_idx].item()
        price_std = std[price_feature_idx].item()
        
        slope_original = slope_norm * price_std
        intercept_original = intercept_norm * price_std + price_mean
        
        y_start = start + past_window
        y_end = y_start + future_window
        future_prices = total_tensor[price_feature_idx, y_start:y_end].cpu().numpy()
        
        time_steps = np.arange(future_window)
        ax.plot(time_steps, future_prices, 'b-', linewidth=2, label='Actual Prices', alpha=0.7)
        
        if horizon > 0:
            trend_t = np.arange(horizon)
            trend_prices = slope_original * trend_t + intercept_original
            ax.plot(trend_t, trend_prices, 'r-', linewidth=2.5,
                    label=f'Trend: y={slope_original:.4f}t+{intercept_original:.2f}')
            ax.axvline(x=horizon - 1, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axvline(x=0, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
            ax.axvspan(0, horizon - 1, alpha=0.15, color='orange')
        
        ax.set_title(f'Sample {idx+1}: H={horizon}, m={slope_original:.4f}, b={intercept_original:.2f}\n', fontsize=10)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    
    for i in range(len(indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Linear Trend Visualization - Mode: {mode.upper()}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def find_checkpoint_by_key(
    checkpoint_dir: str = 'trading_checkpoints',
    key: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """
    Find checkpoint by key. If key is None, use the latest.
    
    Returns:
        Tuple of (checkpoint_path, checkpoint_key) or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"⚠ Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # If key is provided, look for that specific key
    if key is not None:
        key_dir = checkpoint_dir / key
        if not key_dir.exists():
            print(f"⚠ Checkpoint key directory not found: {key_dir}")
            return None
        
        # Look for training_state.json first
        state_file = key_dir / 'training_state.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                latest = state.get('latest_checkpoint')
                if latest and Path(latest).exists():
                    return latest, key
            except:
                pass
        
        # Search for .pt files
        checkpoints = list(key_dir.glob('*.pt'))
        if not checkpoints:
            print(f"⚠ No checkpoint files found in {key_dir}")
            return None
        
        # Return the latest by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"✓ Found checkpoint: {latest} (Key: {key})")
        return str(latest), key
    
    # If no key provided, find the latest checkpoint
    all_checkpoints = []
    
    for subdir in checkpoint_dir.iterdir():
        if subdir.is_dir():
            for f in subdir.glob('*.pt'):
                all_checkpoints.append((f, subdir.name))
    
    if not all_checkpoints:
        print(f"⚠ No checkpoint files found in {checkpoint_dir}")
        return None
    
    latest, key = max(all_checkpoints, key=lambda p: p[0].stat().st_mtime)
    print(f"✓ Found latest checkpoint: {latest} (Key: {key})")
    return str(latest), key


def load_test_data(
    config: dict,
    features: List[np.ndarray],
    target_feature_id: int,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], int, int]:
    """
    Load test data from the dataset.
    """
    # Create dataset with test split - SAME as main.py
    dataset = TradingDataset(
        features=features,
        price_feature_idx=target_feature_id,
        past_days=config['past_days'],
        future_minutes=config['future_minutes'],
        step=config['step'],
        max_accum_deviation=config['max_accum_deviation'],
        min_horizon=config['min_horizon'],
        max_horizon=config['max_horizon'],
        device='cpu',
        normalize=config['normalize'],
        data_dir=config.get('data_dir', 'dataset_cache'),
        force_recompute=config.get('force_recompute', False),
        split_ratios=(0.7, 0.2, 0.1)
    )
    
    # Collect test samples
    test_windows = []
    test_labels = []
    test_means = []
    test_stds = []
    test_shapes = []
    test_valid_starts = []
    
    for idx in dataset.test_indices:
        x, y, mean, std, shape = dataset[idx]
        test_windows.append(x.cpu())
        test_labels.append(y.cpu())
        test_means.append(mean.cpu())      # mean is (F,)
        test_stds.append(std.cpu())        # std is (F,)
        test_shapes.append(shape.cpu())    # shape is (F, num_shapes)
        test_valid_starts.append(dataset.valid_starts[idx])
    
    # Stack into tensors
    test_windows = torch.stack(test_windows)  # (N, F, T)
    test_labels = torch.stack(test_labels)    # (N, 3)
    test_means = torch.stack(test_means)      # (N, F)
    test_stds = torch.stack(test_stds)        # (N, F)
    test_shapes = torch.stack(test_shapes)    # (N, F, num_shapes)
    
    print(f"✓ Loaded {len(test_windows)} test samples")
    print(f"  test_means shape: {test_means.shape}")
    print(f"  test_stds shape: {test_stds.shape}")
    
    return (test_windows, test_labels, test_means, test_stds, test_shapes,
            dataset.Total, test_valid_starts, dataset.past_window, dataset.future_window)


def load_model(
    checkpoint_path: str,
    config: dict,
    device: str = 'cuda',
):
    """
    Load a trained model and its statistics from a checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load embed_dim and max_seq_len from checkpoint
    embed_dim = checkpoint.get('embed_dim')
    max_seq_len = checkpoint.get('max_seq_len')
    
    # If not in checkpoint, compute them (backward compatibility)
    if embed_dim is None or max_seq_len is None:
        print("⚠ embed_dim or max_seq_len not in checkpoint. Computing...")
        past_window = 60 * 24 * config['past_days']
        F = config['num_features']
        base_embed_dim = F * config['patch_size']
        while base_embed_dim % config['num_heads'] != 0:
            base_embed_dim += 1
        embed_dim = base_embed_dim
        max_seq_len = past_window // config['patch_size']
    
    # Recreate model
    model = MarketTransformerWithHeads(
        in_features=config['num_features'],
        patch_size=config['patch_size'],
        embed_dim=embed_dim,
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim_multiplier=config['ff_dim_multiplier'],
        dropout=config['dropout'],
        max_seq_len=max_seq_len,
        causal=config['causal'],
        head_hidden_dim=config['head_hidden_dim'],
        output_dim=config['output_dim'],
        use_reconstruction=config['use_reconstruction']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def run_inference_on_test_set(
    model,
    dataset,
    test_windows: torch.Tensor,
    test_labels: torch.Tensor,
    test_means: torch.Tensor,
    test_stds: torch.Tensor,
    test_shapes: torch.Tensor,
    device: str = 'cuda',
):
    """
    Run inference on the entire test set.
    """
    model.eval()
    
    # Move everything to device
    test_windows = test_windows.to(device)
    test_labels = test_labels.to(device)
    test_means = test_means.to(device)
    test_stds = test_stds.to(device)
    test_shapes = test_shapes.to(device)
    
    # Transpose for the model: (N, F, T) -> (N, T, F)
    x = test_windows.transpose(1, 2)
    
    with torch.no_grad():
        pred_norm, _, _ = model(x, test_means, test_stds, test_shapes)
    
    # Denormalize predictions using dataset's denormalize method
    preds = []
    for i in range(pred_norm.shape[0]):
        # pred_norm[i] is (3,)
        pred_denorm = dataset.denormalize_y(pred_norm[i])
        
        # Now pred_denorm should be (3,)
        horizon = int(round(pred_denorm[0].item()))  # Round to nearest integer
        slope = pred_denorm[1].item()
        intercept = pred_denorm[2].item()
        
        pred = torch.tensor([horizon, slope, intercept], device=device)
        preds.append(pred)
    
    preds = torch.stack(preds)
    
    return preds, test_labels


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Load config
    config = get_config()
    device = config['device']
    
    # Generate the checkpoint key
    checkpoint_key = generate_checkpoint_key(config)
    print(f"🔑 Generated Checkpoint Key: {checkpoint_key}")
    
    # Find checkpoint by key
    result = find_checkpoint_by_key('trading_checkpoints', checkpoint_key)
    if result is None:
        print(f"⚠ No checkpoint found for key: {checkpoint_key}")
        print("  Looking for latest checkpoint instead...")
        result = find_checkpoint_by_key('trading_checkpoints', None)
        if result is None:
            print("No checkpoint found. Exiting.")
            exit(1)
    
    checkpoint_path, checkpoint_key = result
    
    print(f"\nLoading model from: {checkpoint_path}")
    print(f"Checkpoint Key: {checkpoint_key}")
    
    model = load_model(checkpoint_path, config, device)
    print("✓ Model loaded successfully")
    
    # Get data (Binance or Synthetic) - EXACTLY like main.py
    features = None
    target_feature_id = None
    
    if config.get('use_binance_data', False):
        features, T, nForget, target_feature_id = get_binance_data(config)  # ← 4 values
        print(f"DEBUG: Got Binance data, target_feature_id = {target_feature_id}")
    
    if features is None:
        T = 60 * 24 * config['total_synthetic_days']
        F = config['num_features']
        price_feature_idx = config.get('price_feature_idx', 2)  # Close is at index 2
        print(f"Generating synthetic data: {T} minutes, {F} features")
        print(f"  Price feature index: {price_feature_idx}")
        features = generate_synthetic_data(T, F, price_feature_idx)
        target_feature_id = price_feature_idx
        print(f"DEBUG: Generated synthetic data, target_feature_id = {target_feature_id}")
    
    print(f"DEBUG: Final target_feature_id = {target_feature_id}")
    
    # Load the dataset
    print("\nLoading dataset...")
    dataset = TradingDataset(
        features=features,
        price_feature_idx=target_feature_id,
        past_days=config['past_days'],
        future_minutes=config['future_minutes'],
        step=config['step'],
        max_accum_deviation=config['max_accum_deviation'],
        min_horizon=config['min_horizon'],
        max_horizon=config['max_horizon'],
        device='cpu',
        normalize=config['normalize'],
        data_dir=config.get('data_dir', 'dataset_cache'),
        force_recompute=config.get('force_recompute', False),
        split_ratios=(0.7, 0.2, 0.1)  # Train, Val, Test
    )
    
    # Collect test samples from dataset
    print("\nCollecting test samples...")
    test_windows = []
    test_labels = []
    test_means = []
    test_stds = []
    test_shapes = []
    valid_starts = []
    
    for idx in dataset.test_indices:
        x, y, mean, std, shape = dataset[idx]
        test_windows.append(x.cpu())
        test_labels.append(y.cpu())
        test_means.append(mean.cpu())
        test_stds.append(std.cpu())
        test_shapes.append(shape.cpu())
        valid_starts.append(dataset.valid_starts[idx])
    
    # Stack into tensors
    test_windows = torch.stack(test_windows)
    test_labels = torch.stack(test_labels)
    test_means = torch.stack(test_means)
    test_stds = torch.stack(test_stds)
    test_shapes = torch.stack(test_shapes)
    
    total_tensor = dataset.Total
    past_window = dataset.past_window
    future_window = dataset.future_window
    
    print(f"✓ Loaded {len(test_windows)} test samples")
    
    # Run inference on test set
    print(f"\nRunning inference on {len(test_windows)} test samples...")
    preds, test_labels_device = run_inference_on_test_set(
        model,
        dataset,  # ← Pass the dataset for denormalization
        test_windows, test_labels, test_means, test_stds, test_shapes,
        device
    )
    
    # Compute metrics
    print("\n📊 Test Set Results:")
    
    # # Horizon accuracy (rounded to nearest integer)
    # horizon_pred = preds[:, 0].round().int()
    # horizon_true = test_labels_device[:, 0].round().int()
    # horizon_accuracy = (horizon_pred == horizon_true).float().mean().item()
    # print(f"  Horizon Accuracy: {horizon_accuracy:.4f} ({horizon_accuracy*100:.2f}%)")
    
    # Slope and Intercept MSE
    horizon_mse = torch.nn.functional.mse_loss(preds[:, 0], test_labels_device[:, 0]).item()
    slope_mse = torch.nn.functional.mse_loss(preds[:, 1], test_labels_device[:, 1]).item()
    intercept_mse = torch.nn.functional.mse_loss(preds[:, 2], test_labels_device[:, 2]).item()
    print(f"  Horizon MSE:     {horizon_mse:.4f}")
    print(f"  Slope MSE:     {slope_mse:.4f}")
    print(f"  Intercept MSE: {intercept_mse:.4f}")
    
    # Show first few predictions
    print(f"\n📈 Sample Predictions (first 5):")
    print(f"  {'Index':<6} {'True H':<8} {'Pred H':<8} {'True Slope':<12} {'Pred Slope':<12} {'True Intercept':<14} {'Pred Intercept':<14}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*14} {'-'*14}")
    for i in range(min(5, len(preds))):
        print(f"  {i:<6} {test_labels_device[i, 0].item():<8.0f} {preds[i, 0].item():<8.0f} {test_labels_device[i, 1].item():<12.4f} {preds[i, 1].item():<12.4f} {test_labels_device[i, 2].item():<14.2f} {preds[i, 2].item():<14.2f}")
    
    # Visualize true trends
    print("\n📈 Visualizing true linear trends...")
    visualize_linear_trends(
        total_tensor=total_tensor,
        valid_starts=valid_starts,
        label_tensor=test_labels_device.cpu(),
        mean_tensor=test_means.cpu(),
        std_tensor=test_stds.cpu(),
        price_feature_idx=target_feature_id,
        past_window=past_window,
        future_window=future_window,
        mode=config.get('horizon_mode', 'anchored'),
        num_samples=6
    )
    
    print(f"\n✓ Inference complete!")