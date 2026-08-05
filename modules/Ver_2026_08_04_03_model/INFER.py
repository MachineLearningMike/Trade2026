# INFER.py
# Inference script for trained model

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, List

from config import get_config
from model import MarketTransformerWithHeads
from data import visualize_linear_trends
from trainer import generate_checkpoint_key
from TRAIN import get_dataset


# ═══════════════════════════════════════════════════════════
# CHECKPOINT LOOKUP
# ═══════════════════════════════════════════════════════════

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

    if key is not None:
        key_dir = checkpoint_dir / key
        if not key_dir.exists():
            print(f"⚠ Checkpoint key directory not found: {key_dir}")
            return None

        state_file = key_dir / 'training_state.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                latest = state.get('latest_checkpoint')
                if latest and Path(latest).exists():
                    return latest, key
            except Exception:
                pass

        checkpoints = list(key_dir.glob('*.pt'))
        if not checkpoints:
            print(f"⚠ No checkpoint files found in {key_dir}")
            return None

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"✓ Found checkpoint: {latest} (Key: {key})")
        return str(latest), key

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


def load_model(
    checkpoint_path: str,
    config: dict,
    dataset,
    device: str = 'cuda',
):
    """
    Load a trained model from a checkpoint.

    Architecture dims are taken from `dataset` + `config` so they match training.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    past_window = 60 * 24 * config['past_days']
    max_seq_len = checkpoint.get('max_seq_len')
    if max_seq_len is None:
        max_seq_len = past_window // config['patch_size']

    F = dataset.F
    model = MarketTransformerWithHeads(
        in_features=F,
        n_market_features=dataset.n_market_features,
        n_macro_features=dataset.n_macro_features,
        n_latent_macro_features=config['n_latent_macro_features'],
        patch_size=config['patch_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim_multiplier=config['ff_dim_multiplier'],
        dropout=config['dropout'],
        max_seq_len=max_seq_len,
        causal=config['causal'],
        head_hidden_dim_multiplier=config['head_hidden_dim_multiplier'],
        output_dim=config['output_dim'],
        use_reconstruction=config['use_reconstruction'],
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

    test_windows: (N, T, F)  — same layout as dataset __getitem__
    """
    model.eval()

    test_windows = test_windows.to(device)
    test_labels = test_labels.to(device)
    test_means = test_means.to(device)
    test_stds = test_stds.to(device)
    test_shapes = test_shapes.to(device)

    # Dataset already returns (T, F); stacked → (N, T, F). No transpose needed.
    x = test_windows

    with torch.no_grad():
        pred_norm, _, _ = model(x, test_means, test_stds, test_shapes)

    preds = []
    for i in range(pred_norm.shape[0]):
        pred_denorm = dataset.denormalize_y(pred_norm[i].cpu())
        horizon = int(round(pred_denorm[0].item()))
        slope = pred_denorm[1].item()
        intercept = pred_denorm[2].item()
        preds.append(torch.tensor([horizon, slope, intercept], device=device, dtype=torch.float32))

    preds = torch.stack(preds)
    return preds, test_labels


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    config = get_config()
    device = config['device']

    print("\nLoading dataset (same path as TRAIN)...")
    dataset, total_features = get_dataset(config)

    checkpoint_key = generate_checkpoint_key(config, dataset.cache_key)
    print(f"🔑 Generated Checkpoint Key: {checkpoint_key}")

    result = find_checkpoint_by_key(config.get('checkpoint_dir', 'trading_checkpoints'), checkpoint_key)
    if result is None:
        print(f"⚠ No checkpoint found for key: {checkpoint_key}")
        print("  Looking for latest checkpoint instead...")
        result = find_checkpoint_by_key(config.get('checkpoint_dir', 'trading_checkpoints'), None)
        if result is None:
            print("No checkpoint found. Exiting.")
            raise SystemExit(1)

    checkpoint_path, checkpoint_key = result

    print(f"\nLoading model from: {checkpoint_path}")
    print(f"Checkpoint Key: {checkpoint_key}")

    model = load_model(checkpoint_path, config, dataset, device)
    print("✓ Model loaded successfully")

    print("\nCollecting test samples...")
    test_windows = []
    test_labels = []
    test_means = []
    test_stds = []
    test_shapes = []
    valid_starts = []

    for idx in dataset.test_indices:
        x, y, mean, std, shape = dataset[idx]
        test_windows.append(x)
        test_labels.append(y)
        test_means.append(mean)
        test_stds.append(std)
        test_shapes.append(shape)
        valid_starts.append(dataset.valid_starts[idx])

    test_windows = torch.stack(test_windows)   # (N, T, F)
    test_labels = torch.stack(test_labels)     # (N, 3)
    test_means = torch.stack(test_means)       # (N, F)
    test_stds = torch.stack(test_stds)         # (N, F)
    test_shapes = torch.stack(test_shapes)     # (N, num_shapes, F)

    print(f"✓ Loaded {len(test_windows)} test samples")
    print(f"  windows: {tuple(test_windows.shape)}  means: {tuple(test_means.shape)}  "
          f"shapes: {tuple(test_shapes.shape)}")

    print(f"\nRunning inference on {len(test_windows)} test samples...")
    preds, test_labels_device = run_inference_on_test_set(
        model,
        dataset,
        test_windows, test_labels, test_means, test_stds, test_shapes,
        device,
    )

    print("\n📊 Test Set Results:")
    horizon_mse = torch.nn.functional.mse_loss(preds[:, 0], test_labels_device[:, 0]).item()
    slope_mse = torch.nn.functional.mse_loss(preds[:, 1], test_labels_device[:, 1]).item()
    intercept_mse = torch.nn.functional.mse_loss(preds[:, 2], test_labels_device[:, 2]).item()
    print(f"  Horizon MSE:     {horizon_mse:.4f}")
    print(f"  Slope MSE:     {slope_mse:.4f}")
    print(f"  Intercept MSE: {intercept_mse:.4f}")

    print(f"\n📈 Sample Predictions (first 5):")
    print(f"  {'Index':<6} {'True H':<8} {'Pred H':<8} {'True Slope':<12} {'Pred Slope':<12} "
          f"{'True Intercept':<14} {'Pred Intercept':<14}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*14} {'-'*14}")
    for i in range(min(5, len(preds))):
        print(
            f"  {i:<6} {test_labels_device[i, 0].item():<8.0f} {preds[i, 0].item():<8.0f} "
            f"{test_labels_device[i, 1].item():<12.4f} {preds[i, 1].item():<12.4f} "
            f"{test_labels_device[i, 2].item():<14.2f} {preds[i, 2].item():<14.2f}"
        )

    # For visualization of *true* labels, use raw (denormalized) label tensor +
    # per-sample mean/std from the dataset cache (not the train-stats-normalized ones).
    print("\n📈 Visualizing true linear trends...")
    raw_means = torch.stack([dataset.mean_tensor[i] for i in dataset.test_indices])
    raw_stds = torch.stack([dataset.std_tensor[i] for i in dataset.test_indices])
    raw_labels = torch.stack([dataset.label_tensor[i] for i in dataset.test_indices])

    visualize_linear_trends(
        total_tensor=dataset.Total,
        valid_starts=valid_starts,
        label_tensor=raw_labels,
        mean_tensor=raw_means,
        std_tensor=raw_stds,
        price_feature_idx=dataset.price_feature_idx,
        past_window=dataset.past_window,
        future_window=dataset.future_window,
        mode=config.get('horizon_mode', 'anchored'),
        num_samples=6,
    )

    print(f"\n✓ Inference complete!")
