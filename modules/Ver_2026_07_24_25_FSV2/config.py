# config.py
# Central configuration for all hyper-parameters

config = {
    # ============================================================
    # DATA CONFIGURATION
    # ============================================================
    'past_days': 7,     # 1, 2, 
    'future_minutes': 60,
    'step': 3,
    'num_features': 5,
    'num_shapes': 4,
    'total_synthetic_days': 30,
    
    # ============================================================
    # DATA SPLIT CONFIGURATION
    # ============================================================
    'train_ratio': 0.70,
    'val_ratio': 0.20,
    'shuffle_split': True,
    
    # ============================================================
    # VALIDATION CONFIGURATION
    # ============================================================
    'validate': True,
    'validate_every_n_epochs': 1,
    'validate_batch_size': 64,
    
    # ============================================================
    # HORIZON LABELING CONFIGURATION
    # ============================================================
    'max_accum_deviation': 0.35,
    'min_horizon': 3,
    'max_horizon': 50,
    
    # ============================================================
    # HORIZON LABELING MODE
    # ============================================================
    # Choose one of: 'floating', 'anchored', 'weighted'
    #   'floating'  - Unconstrained OLS fit (no anchor)
    #   'anchored'  - Constrained to pass through f[0] (first future price)
    #   'weighted'  - Weighted OLS with far future getting higher weights
    'horizon_mode': 'floating',
    
    # ============================================================
    # WEIGHTED LABELING CONFIGURATION (only used when mode='weighted')
    # ============================================================
    'weight_type': 'exponential',  # 'exponential', 'linear', 'sqrt', 'sigmoid', 'step', 'none'
    'weight_decay': 0.9,           # Decay factor for exponential weighting (0-1)
    
    # ============================================================
    # BINANCE DATA CONFIGURATION
    # ============================================================
    'use_binance_data': True,
    'binance_datatype': 'klines',
    'binance_interval': '1m',
    'binance_start_date': '2024-01-01',
    'binance_end_date': '2026-06-30',
    'binance_symbols': {
        "BTCUSDT": ["High", "Low", "Close", "Volume"],
        "ETHUSDT": ["Close"]
    },
    'prediction_target': ("BTCUSDT", "Close"),
    'binance_primary_symbol': "BTCUSDT",
    
    # ============================================================
    # NORMALIZATION CONFIGURATION
    # ============================================================
    'normalize': True,
    
    # ============================================================
    # CACHE CONFIGURATION
    # ============================================================
    'cache_dir': 'dataset_cache',
    'force_recompute': False,
    
    # ============================================================
    # TRANSFORMER CONFIGURATION
    # ============================================================
    'patch_size': 60,
    'embed_dim': None,  # AUTO-COMPUTED
    'num_layers': 16,
    'num_heads': 8,
    'ff_dim': 256,
    'dropout': 0.1,
    'causal': False,
    'max_seq_len': None,  # AUTO-COMPUTED
    
    # ============================================================
    # HEAD CONFIGURATION
    # ============================================================
    'head_hidden_dim': 64,
    'output_dim': 3,  # horizon, slope, intercept
    
    # ============================================================
    # TRAINING CONFIGURATION
    # ============================================================
    'epochs': 200,
    'batch_size': 32,
    'learning_rate': 0.005,
    'save_every_n_minibatches': 500,
    'checkpoint_dir': 'trading_checkpoints',
    'max_checkpoints': 3,
    
    # ============================================================
    # DEVICE CONFIGURATION
    # ============================================================
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
    
    # ============================================================
    # RESUME CONFIGURATION
    # ============================================================
    'resume': True,
    'clean': False,
    
    # ============================================================
    # LOSS WEIGHT CONFIGURATION
    # ============================================================
    'slope_loss_weight': 30.0,
    'intercept_loss_weight': 1.0,
    'reconstruction_loss_weight': 0.01,
    'use_reconstruction': False,
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


if __name__ == "__main__":
    print_config()