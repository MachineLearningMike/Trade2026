# config.py
# Central configuration for all hyper-parameters

config = {
    # ============================================================
    # DATA STORAGE CONFIGURATION
    # ============================================================
    'data_dir': 'dataset_cache',

    # ============================================================
    # DATA CONFIGURATION
    # ============================================================
    'past_days': 2,     # 1, 2,
    # future_window length is NOT set here — see label-mode horizons below:
    #   linear_regime  → max_trend_future_horizon
    #   scattered_hope → max_hope_future_horizon
    'step': 5,
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
    'validate_batch_size': 16,

    # ============================================================
    # TEST / INFERENCE CONFIGURATION
    # ============================================================
    'test_batch_size': 16,
    
    # ============================================================
    # LABEL FAMILY
    # ============================================================
    # Choose how supervised targets are built from the future price window:
    #   'linear_regime'   → (horizon, slope, intercept) via find_linear_regime
    #                        model may use output_dim 2 (m,b) or 3 (h,m,b)
    #   'scattered_hope'  → (chance, wait) via find_scattered_hope
    #                        requires output_dim=2
    'label': 'scattered_hope',

    # ------------------------------------------------------------
    # SCATTERED HOPE (distrubution of chance and wait)
    # ------------------------------------------------------------
    'min_yield_percent': 0.3,
    # Also sets the sample future_window when label=scattered_hope
    'max_hope_future_horizon': 60,

    # ------------------------------------------------------------
    # TREND / LINEAR-REGIME HORIZONS (linear_regime)
    # ------------------------------------------------------------
    'max_accum_deviation': 0.35,   # legacy alias; prefer fitness_threshold
    'min_trend_horizon': 3,
    # Max future bars for the linear regime search; also:
    #   - sample future_window when label=linear_regime
    #   - max past extent for joint past+future regime + model p vector
    'max_trend_future_horizon': 50,
    'regime_use_past': True,
    # Dimensionless fitness: sum(|residual|) / history_sigma
    'fitness_threshold': 0.75,

    # ============================================================
    # HORIZON SEARCH RULE (linear_regime)
    # ============================================================
    # Free (unanchored) OLS on joint [recent past ‖ nearest future].
    #   'longest_linear'         - grow while fitness <= threshold; keep last good
    #   'shortest_to_threshold'  - first window where fitness >= threshold
    'horizon_search': 'longest_linear',

    # ============================================================
    # HORIZON FIT / VISUALIZATION MODE (display + legacy weighted path)
    # ============================================================
    # Labeling always uses free OLS (not anchored through f[0]).
    # Choose one of: 'floating', 'anchored', 'weighted'
    #   'floating'  - Unconstrained OLS (matches label math)
    #   'anchored'  - Viz only: line constrained through f[0]
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
    'binance_start_date': '2024-01-01 00:00:00',
    'binance_end_date': '2026-07-01 00:00:00',
    'binance_symbols': {
        "BTCUSDT": ["Close", "Volume"],
        "ETHUSDT": ["Close", "Volume"],
    },
    'prediction_target': ("BTCUSDT", "Close"),
    'binance_primary_symbol': "BTCUSDT",

    # ============================================================
    # MACRO / FSV CATALOG
    # ============================================================
    # Series definitions live in fsv_dictionary.FEATURES only.
    # features_to_include: None → all FEATURES minus exclude; list → whitelist.
    # features_to_exclude: applied after include.
    'features_to_include': None,
    'features_to_exclude': ["ted_spread", "commercial_paper_3m", "fhfa_price"],

    # ============================================================
    # NORMALIZATION CONFIGURATION
    # ============================================================
    'normalize': True,
    
    'force_recompute': False,

    'n_latent_macro_features': 30,  # Number of latent macro features to use in the model
    
    # ============================================================
    # TRANSFORMER CONFIGURATION
    # ============================================================
    'patch_size': 15,
    'embed_dim': None,  # AUTO-COMPUTED
    'num_layers': 8,
    'num_heads': 6,
    'ff_dim_multiplier': 4,
    'dropout': 0.1,
    'causal': False,
    'max_seq_len': None,  # AUTO-COMPUTED
    
    # ============================================================
    # HEAD CONFIGURATION
    # ============================================================
    'head_hidden_dim_multiplier': 4,
    # Prediction head width. Must match label family:
    #   linear_regime:  3 = (h,m,b),  2 = (m,b)
    #   scattered_hope: 2 = (chance, wait)  [forced if mismatch]
    'output_dim': 2,
    
    # ============================================================
    # TRAINING CONFIGURATION
    # ============================================================
    'epochs': 200,
    'batch_size': 8,
    'learning_rate': 0.00002,
    'save_every_n_minibatches': 5000,
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
    # linear_regime (dim 3 uses horizon unweighted; dim 2 omits horizon):
    'slope_loss_weight': 1.0,
    'intercept_loss_weight': 10.0,
    # scattered_hope:
    'chance_loss_weight': 1.0,
    'wait_loss_weight': 1.0,
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