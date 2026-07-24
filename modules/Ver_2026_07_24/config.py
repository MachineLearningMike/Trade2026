# config.py
# Central configuration for all hyper-parameters

config = {
    # ============================================================
    # DATA CONFIGURATION
    # ============================================================
    'past_days': 7,
    'future_minutes': 60,
    'step': 60,
    'num_features': 5,
    'total_days': 30,
    'price_feature_idx': 0,
    
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
    # TRANSFORMER CONFIGURATION
    # ============================================================
    'patch_size': 30,
    'embed_dim': None,  # AUTO-COMPUTED
    'num_layers': 4,
    'num_heads': 4,
    'ff_dim': 256,
    'dropout': 0.1,
    'causal': True,
    'max_seq_len': None,  # AUTO-COMPUTED
    
    # ============================================================
    # HEAD CONFIGURATION
    # ============================================================
    'head_hidden_dim': 64,
    'output_dim': 2,
    
    # ============================================================
    # TRAINING CONFIGURATION
    # ============================================================
    'epochs': 3,
    'batch_size': 16,
    'learning_rate': 0.001,
    'save_every_n_minibatches': 50,
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
    'slope_loss_weight': 0.1,
    'reconstruction_loss_weight': 0.01,
    
    # ============================================================
    # RECONSTRUCTION CONFIGURATION
    # ============================================================
    'use_reconstruction': True,  # <-- ADD THIS KEY
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