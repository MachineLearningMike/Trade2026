# config.py
# Central configuration for all hyper-parameters

config = {
    # ============================================================
    # DATA CONFIGURATION
    # ============================================================
    
    # Number of days of historical data used as input (X)
    # Determines past_window = 60 * 24 * past_days (minutes)
    'past_days': 7,
    
    # Number of minutes of future data used for labeling (Y)
    # This is the maximum horizon that can be predicted
    'future_minutes': 60,
    
    # Step size for sliding window sampling (minutes)
    'step': 60,
    
    # Number of features (F) in the input data
    'num_features': 5,

    # Number of shape parameters of the distribution of a market feature
    # Generalize it to transition params, later
    'num_shapes': 4,
    
    # Total days of synthetic data to generate (for testing)
    'total_synthetic_days': 30,
    
    
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
    'binance_primary_symbol': "BTCUSDT",
    'binance_primary_feature': "Close",    
    
    # ============================================================
    # HORIZON LABELING CONFIGURATION
    # ============================================================
    
    'max_accum_deviation': 12.0,
    'min_horizon': 3,
    'max_horizon': 35,
    
    
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
    
    'patch_size': 15,
    'embed_dim': None,  # AUTO-COMPUTED
    'num_layers': 8,
    'num_heads': 4,
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
    
    'epochs': 10,
    'batch_size': 32,
    'learning_rate': 0.001,
    'save_every_n_minibatches': 100,
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
    # LOSS CONFIGURATION
    # ============================================================
    
    'slope_loss_weight': 1.0,
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