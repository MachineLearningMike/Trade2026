# config.py
# Central configuration for all hyper-parameters

config = {
    # ============================================================
    # DATA CONFIGURATION
    # ============================================================
    
    # Number of days of historical data used as input (X)
    # Determines past_window = 60 * 24 * past_days (minutes)
    # Larger values give more context but increase memory/compute
    'past_days': 7,
    
    # Number of minutes of future data used for labeling (Y)
    # This is the maximum horizon that can be predicted
    # Must be >= max_horizon (labeling config)
    'future_minutes': 60,
    
    # Step size for sliding window sampling (minutes)
    # Step = 1 means every minute, Step = 60 means every hour
    # Affects dataset size: total_samples = (T - total_window) // step
    # Trade-off: smaller step = more samples, larger step = less correlation
    'step': 60,
    
    # Number of features (F) in the input data
    # Determines embed_dim = F * patch_size (transformer config)
    # Must match the actual feature dimension of your data
    'num_features': 5,
    
    # Total days of synthetic data to generate (for testing)
    # Determines total time steps T = 60 * 24 * total_days
    'total_synthetic_days': 30,
    
    # Index of the price feature in the features list (0-based)
    # Used for:
    #   1. Extracting future prices for horizon labeling
    #   2. Normalizing y using the same stats as x's price feature
    # Must be a valid index in range(0, num_features)
    'price_feature_idx': 0,
    
    
    # ============================================================
    # BINANCE DATA CONFIGURATION
    # ============================================================
    
    # Whether to use real Binance data instead of synthetic
    'use_binance_data': True,
    
    # Binance data type: 'klines' or 'aggTrades'
    'binance_datatype': 'klines',
    
    # Kline interval: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'
    'binance_interval': '1m',
    
    # Start and end dates for data collection (format: YYYY-MM-DD)
    'binance_start_date': '2026-01-01',     # 2024
    'binance_end_date': '2026-06-30',
    
    # Symbol configuration: symbol -> list of features to extract
    # Available features: "Open", "High", "Low", "Close", "Volume"
    'binance_symbols': {
        "ETHUSDT": ["High", "Low", "Close", "Volume"],
        "BTCUSDT": ["Close"]
    },
    
    # Primary symbol to use as index 0 in features
    'binance_primary_symbol': "ETHUSDT",
    
    
    # ============================================================
    # HORIZON LABELING CONFIGURATION
    # ============================================================
    
    # Threshold for cumulative absolute deviation
    # The horizon is the first point where cumulative deviation >= this value
    # Smaller threshold = shorter horizons, larger = longer horizons
    # Related to min_horizon, max_horizon (must be within [min, max])
    'max_accum_deviation': 12.0,
    
    # Minimum horizon to consider (in minutes)
    # Skips very short horizons that might be noisy
    # Must be >= 2 (need at least 2 points for a trend line)
    # Must be < max_horizon
    'min_horizon': 3,
    
    # Maximum horizon to consider (in minutes)
    # Caps the search to avoid looking too far into the future
    # Must be <= future_minutes (data config)
    # If None, uses future_minutes as max
    'max_horizon': 35,
    
    
    # ============================================================
    # NORMALIZATION CONFIGURATION
    # ============================================================
    
    # Whether to normalize each window by its own mean/std
    # True: Per-window normalization (recommended for non-stationary data)
    # False: No normalization (use if data is already stationary)
    # When True: each (x, y) pair is normalized using x's stats
    # Slope labels are computed in normalized space
    'normalize': True,
    
    
    # ============================================================
    # CACHE CONFIGURATION
    # ============================================================
    
    # Directory to cache computed labels and normalization parameters
    # Caching avoids recomputing labels on every run (saves time)
    # Cache is invalidated when parameters change
    'cache_dir': 'dataset_cache',
    
    # Force recomputation of labels even if cache exists
    # Set to True to ignore cache and recompute everything
    'force_recompute': False,
    
    
    # ============================================================
    # TRANSFORMER CONFIGURATION
    # ============================================================
    
    # Patch size for temporal patching (P)
    # Divides the sequence into patches of length P
    # Number of patches N = past_window // patch_size
    # Must divide past_window evenly (no remainder)
    # Related to embed_dim: embed_dim = num_features * patch_size
    # Larger patch = fewer patches, more compression, less temporal resolution
    'patch_size': 30,
    
    # Embedding dimension (D) - AUTO-COMPUTED
    # D = num_features * patch_size (then adjusted for num_heads divisibility)
    # This is the dimension of the transformer's internal representation
    # Must be divisible by num_heads (transformer constraint)
    # Larger D = more expressive power, more memory/compute
    'embed_dim': None,  # Will be computed automatically
    
    # Number of transformer encoder layers
    # More layers = deeper network, more capacity, longer training
    # Trade-off: 4-8 for small data, 12-24 for large data
    'num_layers': 4,
    
    # Number of attention heads
    # Must divide embed_dim (critical constraint!)
    # More heads = more parallel attention patterns
    # Common values: 4, 8, 16 (must be <= embed_dim)
    'num_heads': 4,
    
    # Feed-forward dimension in transformer
    # Typically 2x-4x of embed_dim
    # Controls the capacity of the feed-forward network
    'ff_dim': 256,
    
    # Dropout rate for regularization
    # Higher = more regularization, less overfitting
    # Typical range: 0.1 - 0.3
    'dropout': 0.1,
    
    # Whether to use causal (masked) attention
    # True: Each position only attends to previous positions (good for forecasting)
    # False: Full bidirectional attention (good for reconstruction/autoencoding)
    'causal': False,
    
    # Maximum sequence length (N = past_window // patch_size)
    # AUTO-COMPUTED from past_days and patch_size
    # Used for positional encoding buffer size
    'max_seq_len': None,  # Will be computed automatically
    
    
    # ============================================================
    # HEAD CONFIGURATION
    # ============================================================
    
    # Hidden dimension for prediction heads
    # Intermediate size between embed_dim and output_dim
    # Larger = more capacity for prediction, more parameters
    'head_hidden_dim': 64,
    
    # Output dimension (must be 2 for horizon, slope)
    # Position 0: horizon (time to reach threshold)
    # Position 1: slope (trend strength in normalized space)
    'output_dim': 2,
    
    
    # ============================================================
    # TRAINING CONFIGURATION
    # ============================================================
    
    # Number of training epochs
    # One epoch = one pass through the entire dataset
    'epochs': 3,
    
    # Batch size for training
    # Number of samples processed together
    # Larger = more stable gradients, more memory usage
    'batch_size': 16,
    
    # Learning rate for Adam optimizer
    # Controls step size during gradient descent
    # Too high = unstable, too low = slow convergence
    'learning_rate': 0.001,
    
    # Save checkpoint every N minibatches
    # More frequent = safer but slower (disk I/O)
    # Related to max_checkpoints (disk space management)
    'save_every_n_minibatches': 50,
    
    # Directory to save checkpoints
    'checkpoint_dir': 'trading_checkpoints',
    
    # Maximum number of checkpoints to keep
    # Older checkpoints are automatically deleted
    # Related to save_every_n_minibatches (frequency affects count)
    'max_checkpoints': 3,
    
    
    # ============================================================
    # DEVICE CONFIGURATION
    # ============================================================
    
    # Device for computation: 'cuda', 'mps', or 'cpu'
    # AUTO-DETECTED: uses GPU if available, otherwise CPU
    # Related: model.to(device), tensor.to(device)
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
    
    
    # ============================================================
    # RESUME CONFIGURATION
    # ============================================================
    
    # Whether to resume from checkpoint if available
    # True: Loads latest checkpoint and continues training
    # False: Starts fresh (ignores existing checkpoints)
    'resume': True,
    
    # Whether to delete old checkpoints before starting
    # True: Removes all existing checkpoints (fresh start)
    # False: Keeps existing checkpoints (may resume or overwrite)
    # NOTE: This does NOT delete the dataset cache
    'clean': False,
    
    
    # ============================================================
    # LOSS CONFIGURATION
    # ============================================================
    
    # Weight for slope loss relative to horizon loss
    # Total loss = horizon_loss + slope_weight * slope_loss
    # If slope is less important: use smaller value (e.g., 0.1)
    # If slope is equally important: use 1.0
    'slope_loss_weight': 0.1,
    
    # Weight for reconstruction loss (auxiliary task)
    # Total loss = pred_loss + recon_weight * recon_loss
    # Reconstruction helps learn better representations
    # Set to 0 to disable reconstruction loss
    'reconstruction_loss_weight': 0.01,
    
    # Whether to use reconstruction loss (auxiliary)
    # True: Adds reconstruction loss to help learn better representations
    # False: Only uses prediction loss (horizon and slope)
    'use_reconstruction': True,
}


def get_config():
    """Return a copy of the config dictionary."""
    return config.copy()


def update_config(new_config):
    """Update config with new values."""
    config.update(new_config)


def print_config():
    """Print all config values in a formatted way."""
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key:30} = {value}")
    print("=" * 60)


def get_config_summary():
    """Return a summary of key configuration parameters."""
    summary = {
        'data': {
            'past_days': config['past_days'],
            'future_minutes': config['future_minutes'],
            'step': config['step'],
            'num_features': config['num_features'],
            'total_synthetic_days': config['total_synthetic_days'],
        },
        'binance': {
            'use_binance_data': config['use_binance_data'],
            'interval': config['binance_interval'],
            'start_date': config['binance_start_date'],
            'end_date': config['binance_end_date'],
            'primary_symbol': config['binance_primary_symbol'],
        },
        'horizon_labeling': {
            'max_accum_deviation': config['max_accum_deviation'],
            'min_horizon': config['min_horizon'],
            'max_horizon': config['max_horizon'],
        },
        'model': {
            'patch_size': config['patch_size'],
            'embed_dim': config['embed_dim'],
            'num_layers': config['num_layers'],
            'num_heads': config['num_heads'],
            'ff_dim': config['ff_dim'],
            'dropout': config['dropout'],
            'causal': config['causal'],
        },
        'training': {
            'epochs': config['epochs'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'save_every': config['save_every_n_minibatches'],
        },
        'device': config['device'],
    }
    return summary


if __name__ == "__main__":
    print_config()