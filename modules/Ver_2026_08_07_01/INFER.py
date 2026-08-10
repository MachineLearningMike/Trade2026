# INFER.py
# Inference script for trained model
#
# Important: never materialize the full test set of windows in RAM.
# With ~430k samples, past_window=10080, F≈56, the test split alone is ~90+ GB
# if stacked. Always stream samples in small batches.

import json
import random
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn

from config import get_config
from model import MarketTransformerWithHeads
from labels import resolve_model_past_len
from labels import visualize_trend, visualize_yield
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
    n_max_past = int(
        getattr(dataset, 'model_past_len', None)
        or resolve_model_past_len(config)
    )
    model = MarketTransformerWithHeads(
        in_features=F,
        n_market_features=dataset.n_market_features,
        n_macro_features=dataset.n_macro_features,
        n_latent_macro_features=config['n_latent_macro_features'],
        n_max_past_horizon=n_max_past,
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
        num_shapes=config.get('num_shapes', 4),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def _collate_batch(dataset, indices: List[int]):
    """Load a small batch of samples from the dataset (CPU tensors)."""
    xs, ps, ys, means, stds, shapes = [], [], [], [], [], []
    for idx in indices:
        sample = dataset[idx]
        if len(sample) == 6:
            x, p, mean, std, shape, y = sample
        else:
            x, y, mean, std, shape = sample
            n_p = int(
                getattr(dataset, 'model_past_len', None)
                or getattr(dataset, 'max_past_horizon', None)
                or 0
            )
            p = x[-n_p:, getattr(dataset, 'price_feature_idx', 0)] if n_p else x.new_zeros(0)
        xs.append(x)
        ps.append(p)
        ys.append(y)
        means.append(mean)
        stds.append(std)
        shapes.append(shape)
    return (
        torch.stack(xs),      # (B, T, F)
        torch.stack(ps),      # (B, max_past_horizon)
        torch.stack(ys),      # (B, output_dim)
        torch.stack(means),   # (B, F)
        torch.stack(stds),    # (B, F)
        torch.stack(shapes),  # (B, num_shapes, F)
    )


def run_inference_on_test_set(
    model,
    dataset,
    device: str = 'cuda',
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Stream the test split in batches. Never stacks the full test windows.

    model.forward expects:
        x:     (B, T, F)
        p:     (B, max_past_horizon) recent past prices
        mean:  (B, F)
        std:   (B, F)
        shape: (B, num_shapes, F)

    Returns:
        preds:  (N, output_dim) denormalized predictions on `device`
        labels: (N, output_dim) denormalized model targets on `device`
        used_indices: list of dataset indices that were evaluated
    """
    model.eval()

    test_indices = list(dataset.test_indices)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(test_indices)

    if max_samples is not None and max_samples < len(test_indices):
        test_indices = test_indices[:max_samples]
        print(f"  (using {max_samples} shuffled test samples)", flush=True)
    else:
        print(f"  (using all {len(test_indices)} shuffled test samples)", flush=True)

    n = len(test_indices)
    if n == 0:
        raise RuntimeError("No test samples available.")

    # Rough memory note so hangs are not mysterious next time
    bytes_per_window = dataset.past_window * dataset.F * 4
    full_gb = n * bytes_per_window / (1024 ** 3)
    print(
        f"  Test samples: {n}  |  window≈{bytes_per_window / 1e6:.2f} MB each  "
        f"|  full stack would be ~{full_gb:.1f} GB → streaming batch_size={batch_size}",
        flush=True,
    )

    pred_chunks = []
    label_chunks = []
    n_batches = (n + batch_size - 1) // batch_size

    with torch.no_grad():
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n)
            batch_idx = test_indices[start:end]

            x, p, y, mean, std, shape = _collate_batch(dataset, batch_idx)
            # Shapes into model: x (B,T,F), p (B,P), mean/std (B,F), shape (B,S,F)
            x = x.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            mean = mean.to(device, non_blocking=True)
            std = std.to(device, non_blocking=True)
            shape = shape.to(device, non_blocking=True)

            pred_norm, _, _ = model(x, p, mean, std, shape)  # (B, output_dim) normalized

            # Denormalize on CPU (dataset stats live on CPU)
            pred_denorm = torch.stack(
                [dataset.denormalize_y(pred_norm[i].detach().cpu()) for i in range(pred_norm.shape[0])]
            )
            # y from __getitem__ is model-target-normalized → denormalize for metrics
            y_denorm = torch.stack(
                [dataset.denormalize_y(y[i]) for i in range(y.shape[0])]
            )

            pred_chunks.append(pred_denorm)
            label_chunks.append(y_denorm)

            if b == 0 or (b + 1) % max(1, n_batches // 20) == 0 or (b + 1) == n_batches:
                print(f"  Inference batch {b + 1}/{n_batches}  ({end}/{n} samples)", flush=True)

            # Free GPU tensors early
            del x, p, mean, std, shape, pred_norm

    preds = torch.cat(pred_chunks, dim=0).to(device)
    labels = torch.cat(label_chunks, dim=0).to(device)
    return preds, labels, test_indices


# ═══════════════════════════════════════════════════════════
# FULL TEST PIPELINE (shared by INFER.py and TRAIN.py --test)
# ═══════════════════════════════════════════════════════════

def run_test(
    config: dict,
    max_samples: Optional[int] = None,
    visualize: bool = True,
):
    """
    Load the dataset + checkpoint matching `config`, run batched test-set
    inference, print metrics, and optionally visualize a few true trends.

    Args:
        config: Full training/inference config (same as TRAIN).
        max_samples: If set, evaluate this many shuffled test samples;
                     otherwise evaluate the full shuffled test split.
        visualize: If True, plot a few labeled future trends after metrics.

    Batch size always comes from config['test_batch_size']
    (falls back to validate_batch_size, then batch_size).

    Returns:
        dict with preds, labels, used_indices, metrics, checkpoint_path, checkpoint_key
    """
    device = config['device']
    batch_size = config.get(
        'test_batch_size',
        config.get('validate_batch_size', config.get('batch_size', 8)),
    )

    print("\nLoading dataset (same path as TRAIN)...", flush=True)
    dataset, total_features = get_dataset(config)

    checkpoint_key = generate_checkpoint_key(config, dataset.cache_key)
    print(f"🔑 Generated Checkpoint Key: {checkpoint_key}", flush=True)

    result = find_checkpoint_by_key(
        config.get('checkpoint_dir', 'trading_checkpoints'), checkpoint_key
    )
    if result is None:
        print(f"⚠ No checkpoint found for key: {checkpoint_key}")
        print("  Looking for latest checkpoint instead...")
        result = find_checkpoint_by_key(
            config.get('checkpoint_dir', 'trading_checkpoints'), None
        )
        if result is None:
            print("No checkpoint found. Exiting.")
            raise SystemExit(1)

    checkpoint_path, checkpoint_key = result

    print(f"\nLoading model from: {checkpoint_path}", flush=True)
    print(f"Checkpoint Key: {checkpoint_key}", flush=True)
    print(f"Test batch size: {batch_size}", flush=True)
    if max_samples is not None:
        print(f"Test sample cap: {max_samples}", flush=True)
    else:
        print("Test sample cap: all", flush=True)

    model = load_model(checkpoint_path, config, dataset, device)
    print("✓ Model loaded successfully", flush=True)

    print("\nRunning batched inference on test samples...", flush=True)
    preds, test_labels, used_indices = run_inference_on_test_set(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        max_samples=max_samples,
        shuffle=True,
    )
    print(f"✓ Finished inference on {len(preds)} test samples", flush=True)

    out_dim = preds.shape[-1]
    label_mode = config.get('label', getattr(dataset, 'label_mode', 'trend'))
    chance_only = bool(
        config.get('chance_only', getattr(dataset, 'chance_only', False))
    )
    print("\n📊 Test Set Results:")
    metrics = {}
    if label_mode == 'yield' and (chance_only or out_dim == 1):
        metrics['chance_mse'] = torch.nn.functional.mse_loss(
            preds[:, 0], test_labels[:, 0]
        ).item()
        print(f"  Chance MSE:      {metrics['chance_mse']:.4f}")
    elif label_mode == 'yield' and out_dim == 2:
        metrics['chance_mse'] = torch.nn.functional.mse_loss(
            preds[:, 0], test_labels[:, 0]
        ).item()
        metrics['wait_mse'] = torch.nn.functional.mse_loss(
            preds[:, 1], test_labels[:, 1]
        ).item()
        print(f"  Chance MSE:      {metrics['chance_mse']:.4f}")
        print(f"  Wait MSE:        {metrics['wait_mse']:.4f}")
    elif out_dim == 3:
        metrics['horizon_mse'] = torch.nn.functional.mse_loss(
            preds[:, 0], test_labels[:, 0]
        ).item()
        metrics['slope_mse'] = torch.nn.functional.mse_loss(
            preds[:, 1], test_labels[:, 1]
        ).item()
        metrics['intercept_mse'] = torch.nn.functional.mse_loss(
            preds[:, 2], test_labels[:, 2]
        ).item()
        print(f"  Horizon MSE:     {metrics['horizon_mse']:.4f}")
        print(f"  Slope MSE:       {metrics['slope_mse']:.4f}")
        print(f"  Intercept MSE:   {metrics['intercept_mse']:.4f}")
    elif out_dim == 2:
        metrics['slope_mse'] = torch.nn.functional.mse_loss(
            preds[:, 0], test_labels[:, 0]
        ).item()
        metrics['intercept_mse'] = torch.nn.functional.mse_loss(
            preds[:, 1], test_labels[:, 1]
        ).item()
        print(f"  Slope MSE:       {metrics['slope_mse']:.4f}")
        print(f"  Intercept MSE:   {metrics['intercept_mse']:.4f}")
    else:
        raise ValueError(f"Unexpected pred dim {out_dim}")

    print(f"\n📈 Sample Predictions (first 5):")
    if label_mode == 'yield' and (chance_only or out_dim == 1):
        print(f"  {'Index':<6} {'True Chance':<12} {'Pred Chance':<12}")
        print(f"  {'-'*6} {'-'*12} {'-'*12}")
        for i in range(min(5, len(preds))):
            print(
                f"  {i:<6} {test_labels[i, 0].item():<12.4f} {preds[i, 0].item():<12.4f}"
            )
    elif label_mode == 'yield' and out_dim == 2:
        print(
            f"  {'Index':<6} {'True Chance':<12} {'Pred Chance':<12} "
            f"{'True Wait':<12} {'Pred Wait':<12}"
        )
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for i in range(min(5, len(preds))):
            print(
                f"  {i:<6} {test_labels[i, 0].item():<12.4f} {preds[i, 0].item():<12.4f} "
                f"{test_labels[i, 1].item():<12.4f} {preds[i, 1].item():<12.4f}"
            )
    elif out_dim == 3:
        print(
            f"  {'Index':<6} {'True H':<8} {'Pred H':<8} {'True Slope':<12} {'Pred Slope':<12} "
            f"{'True Intercept':<14} {'Pred Intercept':<14}"
        )
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*14} {'-'*14}")
        for i in range(min(5, len(preds))):
            print(
                f"  {i:<6} {test_labels[i, 0].item():<8.0f} {preds[i, 0].item():<8.0f} "
                f"{test_labels[i, 1].item():<12.4f} {preds[i, 1].item():<12.4f} "
                f"{test_labels[i, 2].item():<14.2f} {preds[i, 2].item():<14.2f}"
            )
    else:
        print(
            f"  {'Index':<6} {'True Slope':<12} {'Pred Slope':<12} "
            f"{'True Intercept':<14} {'Pred Intercept':<14}"
        )
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*14} {'-'*14}")
        for i in range(min(5, len(preds))):
            print(
                f"  {i:<6} {test_labels[i, 0].item():<12.4f} {preds[i, 0].item():<12.4f} "
                f"{test_labels[i, 1].item():<14.2f} {preds[i, 1].item():<14.2f}"
            )

    if visualize:
        n_viz = len(used_indices)
        valid_starts = [dataset.valid_starts[i] for i in used_indices]
        full_labels = torch.stack(
            [dataset.label_tensor[i] for i in used_indices]
        )
        pred_cpu = preds.detach().cpu()

        if label_mode == 'yield':
            print(
                "\n📈 Visualizing yield labels (chance/wait) on test samples...",
                flush=True,
            )
            visualize_yield(
                total_tensor=dataset.Total,
                valid_starts=valid_starts,
                label_tensor=full_labels,
                price_feature_idx=dataset.price_feature_idx,
                past_window=dataset.past_window,
                future_window=dataset.future_window,
                min_yield_percent=config.get(
                    'min_yield_percent',
                    getattr(dataset, 'min_yield_percent', 0.0),
                ),
                max_hope_future_horizon=config.get(
                    'max_hope_future_horizon',
                    getattr(dataset, 'max_hope_future_horizon', None),
                ),
                chance_only=chance_only,
                num_samples=min(6, n_viz),
                pred_tensor=pred_cpu,
                fixed_order=True,
                sample_ids=list(used_indices),
                # Train-set label stats so inset is model-normalized (z) space
                label_mean=getattr(dataset, '_label_stat_mean', None),
                label_std=getattr(dataset, '_label_stat_std', None),
            )
        else:
            print(
                "\n📈 Visualizing label vs inferred trends on test samples...",
                flush=True,
            )
            raw_means = torch.stack([dataset.mean_tensor[i] for i in used_indices])
            raw_stds = torch.stack([dataset.std_tensor[i] for i in used_indices])
            pred_horizon = int(
                config.get('max_trend_future_horizon')
                if config.get('max_trend_future_horizon') is not None
                else getattr(
                    dataset, 'max_trend_future_horizon',
                    getattr(dataset, 'max_horizon', dataset.future_window),
                )
            )
            h_past_full = getattr(dataset, 'h_past_tensor', None)
            h_past_viz = None
            if h_past_full is not None:
                h_past_viz = h_past_full[list(used_indices)]
            visualize_trend(
                total_tensor=dataset.Total,
                valid_starts=valid_starts,
                label_tensor=full_labels,
                mean_tensor=raw_means,
                std_tensor=raw_stds,
                price_feature_idx=dataset.price_feature_idx,
                past_window=dataset.past_window,
                future_window=dataset.future_window,
                mode=config.get('horizon_mode', 'anchored'),
                num_samples=min(6, n_viz),
                pred_tensor=pred_cpu,
                fixed_order=True,
                sample_ids=list(used_indices),
                default_pred_horizon=pred_horizon,
                h_past_tensor=h_past_viz,
                min_horizon=config.get(
                    'min_trend_horizon',
                    getattr(dataset, 'min_trend_horizon', getattr(dataset, 'min_horizon', 3)),
                ),
                max_horizon=config.get(
                    'max_trend_future_horizon',
                    getattr(dataset, 'max_trend_future_horizon', None),
                ),
                max_past_horizon=config.get(
                    'max_trend_future_horizon',
                    getattr(dataset, 'max_trend_future_horizon', None),
                ),
                fitness_threshold=config.get(
                    'fitness_threshold', getattr(dataset, 'fitness_threshold', 0.75)
                ),
                horizon_search=config.get(
                    'horizon_search', getattr(dataset, 'horizon_search', 'longest_linear')
                ),
                regime_use_past=config.get(
                    'regime_use_past', getattr(dataset, 'regime_use_past', True)
                ),
                normalize_prices=config.get('normalize', getattr(dataset, 'normalize', True)),
            )

    print(f"\n✓ Inference complete!", flush=True)

    return {
        'preds': preds,
        'labels': test_labels,
        'used_indices': used_indices,
        'metrics': metrics,
        'output_dim': out_dim,
        'checkpoint_path': checkpoint_path,
        'checkpoint_key': checkpoint_key,
    }


if __name__ == "__main__":
    run_test(get_config())
