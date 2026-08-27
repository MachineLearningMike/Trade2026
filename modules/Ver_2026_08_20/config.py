# config.py
# Central configuration for all hyper-parameters
#
# Window lengths (1m bars): past_window = TelePred delta1, future_window = delta2.
# Shared by yield / trend / latent. Legacy names past_hours,
# max_hope_future_horizon, max_trend_future_horizon, max_latent_future_horizon
# were removed (clean signature break).
# Deferred: max_accum_deviation (prefer fitness_threshold);
#           scattered_hope / linear_regime (legacy label ids in labels.py).

import os
from pathlib import Path


def project_root_dir() -> Path:
    """Repo root ``Trade2026`` (this file lives in ``modules/Ver_…``)."""
    return Path(__file__).resolve().parents[2]


def resolve_data_dir(data_dir=None) -> str:
    """
    Absolute dataset cache directory.

    Priority:
      1. env MACRO_DATA_DIR (or legacy FSV_DATA_DIR)
      2. absolute ``data_dir`` as given
      3. relative ``data_dir`` → ``<project_root>/<data_dir>``
         (default ``dataset_cache`` on the project root)
    """
    env = os.environ.get("MACRO_DATA_DIR") or os.environ.get("FSV_DATA_DIR")
    if env:
        return str(Path(env).expanduser().resolve())
    raw = "dataset_cache" if data_dir is None else str(data_dir)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = project_root_dir() / path
    return str(path.resolve())


config = {

    # ============================================================
    # DATA STORAGE CONFIGURATION
    # ============================================================
    # Relative paths resolve to <project_root>/… (not the process cwd).
    'data_dir': 'dataset_cache',

    # ============================================================
    # DATA / TELEPRED WINDOWS (bars on binance_interval, usually 1m)
    # ============================================================
    # delta1: model input length X(t,d)
    'past_window': 60 * 48,   # 24h of 1m bars (formerly past_hours=24)
    # delta2: label future length p[t+1 : t+1+future_window]
    # Shared by yield, trend, and latent (TelePred delta2).
    'future_window': 40,
    'step': 3,

    # ============================================================
    # TELEPRED / ENSEMBLE (see ensemble_plan.txt, telepred.py)
    # ============================================================
    # Delay d (bars): model input ends at bot_time - d; label Y(t) unchanged.
    # telepred_delay: single-element TRAIN / INFER.
    # telepred_delays: bank for --ensemble / --ensemble-sweep.
    'telepred_delay': 0,
    'telepred_delays': [0, 2, 5, 10],
    # Bump via telepred.DEFINITION_ID when slice math changes.
    'telepred_definition_id': 'telepred_v1',

    # ============================================================
    # ERROR ENSEMBLE (residual boosting; see error_ensemble.py)
    # ============================================================
    # model_0 fits y; model_i fits residual e_{i-1} = e_{i-2} - y_{i-1}
    # (e_{-1} := y). Final ŷ = Σ y_i. Train with TRAIN.py --error-ensemble.
    # Number of stages / models: model_0 .. model_{n-1}.
    'error_ensemble_n': 3,
    # Set at runtime per stage (0-based); included in checkpoint key.
    # 'error_ensemble_stage': None,
    'num_features': 5,
    'num_shapes': 4,
    'total_synthetic_days': 30,
    
    # ============================================================
    # DATA SPLIT CONFIGURATION
    # ============================================================
    'train_ratio': 0.80,
    'val_ratio': 0.10,
    # False: train / val / test are consecutive in time (earliest → latest).
    # Training still shuffles the train split each epoch (DataLoader).
    'shuffle_split': False,
    
    # ============================================================
    # VALIDATION CONFIGURATION
    # ============================================================
    'validate': True,
    'validate_every_n_epochs': 1,
    'validate_batch_size': 32,

    # ============================================================
    # TEST / INFERENCE CONFIGURATION
    # ============================================================
    'test_batch_size': 32,
    # How many distinct test windows to precompute for latent μ(t)/σ(t)
    # visualization after --test. Seeds are spaced across the test period
    # (not the first few after a fixed shuffle).
    'test_viz_samples': 16,
    
    # ============================================================
    # LABEL FAMILY
    # ============================================================
    # Choose how supervised targets are built from the future price window:
    #   'trend'   → (horizon, slope, intercept) via find_trend
    #                        model may use output_dim 2 (m,b) or 3 (h,m,b)
    #   'yield'   → (chance, near) via find_yield; model target selected by
    #                        yield_target (cache always stores both)
    #   'latent'  → density over a scalar observation (shares model head)
    'label': 'yield',

    # ------------------------------------------------------------
    # YIELD (distribution of chance and near)
    # ------------------------------------------------------------
    'min_yield_percent': 0.26,   #
    # Model target from full cache (chance, near):
    #   'both'   → output_dim=2  (chance, near)
    #   'chance' → output_dim=1  (chance,)
    #   'near'   → output_dim=1  (near,)
    'yield_target': 'near',

    # ------------------------------------------------------------
    # LATENT (label=latent) — same model as trend/yield
    # ------------------------------------------------------------
    # Parametric density for the *next* bar (future[0] after the history
    # window). The realized next price is one sample from that latent law;
    # the head outputs density params, not a point forecast (NLL loss).
    #   'normal'  → output_dim=2  (mu, var_raw)              Gaussian NLL
    #   'student' → output_dim=3  (loc, scale_raw, df_raw)   Student-t NLL
    # Scale channels use softplus internally (stable + learnable).
    # Aliases: normal_distribution, student_distribution, student_t, ...
    'latent_distribution': 'normal',
    # Latent uses shared future_window (need >= 1 so future[0] exists).
    'latent_loss_weight': 1.0,

    # ------------------------------------------------------------
    # TREND (label=trend) — uses shared future_window as max horizon /
    # joint past+future search cap / model p length
    # ------------------------------------------------------------
    'max_accum_deviation': 0.351,   # legacy alias; prefer fitness_threshold
    'min_trend_horizon': 3,
    'regime_use_past': True,
    # Dimensionless fitness: sum(|residual|) / history_sigma
    'fitness_threshold': 0.75,

    # ============================================================
    # HORIZON SEARCH RULE (trend)
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
    # Half-open-ish span in minutes: need end > start. Need enough bars for
    # past_window + future_window (plus telepred_delay if d > 0).
    # Times are UTC wall clock (naive strings). Binance klines are UTC; FSV
    # converts US-local release schedules to UTC before aligning on this grid.
    'binance_start_date': '2023-11-01 00:00:00',    # 2023
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
    # Series definitions live in macro_dictionary.FEATURES only.
    # macros_to_include: None → all FEATURES minus exclude; list → whitelist.
    # macros_to_exclude: applied after include.
    'macros_to_include': None,
    'macros_to_exclude': ["ted_spread", "commercial_paper_3m", "fhfa_price"],   # They have NaN's.
    # Used by macro_01_downloader (env FRED_API_KEY / TWELVEDATA_API_KEY override).
    'fred_api_key': 'ef542ded8a399000da02e42d5eb0321f',
    'twelvedata_api_key': '95b1a91f6be84db89dcdbbdfaf9e041c',

    # ============================================================
    # NORMALIZATION CONFIGURATION
    # ============================================================
    # Per-sample window z-score of x (and related moments) in TradingDataset.
    'normalize_window': True,

    'n_latent_macros': 30,  # Number of latent macro features to use in the model

    # ============================================================
    # MODEL SELECTION
    # ============================================================
    # 'regression' → model.MarketTransformerWithHeads
    # 'vae'         → model_vae.MarketVAEWithHeads
    #                 Reconstructs x; mean/std FiLM-condition encoder+decoder;
    #                 shape accepted but unused. Train loss is recon+KL only
    #                 (no yield/chance/near label loss).
    'model_type': 'regression',
    'vae_latent_dim': 64,
    'vae_kl_weight': 0.1,
    
    # ============================================================
    # TRANSFORMER / VAE ENCODER CONFIGURATION
    # ============================================================
    'patch_size': 15,   # 20, 10
    'embed_dim': None,  # AUTO-COMPUTED
    'num_layers': 6,    # 6, 2
    'num_heads': 5,     # 8, 4
    'ff_dim_multiplier': 4,
    'dropout': 0.1,
    'causal': False,
    'max_seq_len': None,  # AUTO-COMPUTED
    
    # ============================================================
    # HEAD CONFIGURATION
    # ============================================================
    'head_hidden_dim_multiplier': 4,
    # If True, concat recent past prices p onto the flattened decoder
    # tokens before the prediction head (adds n_max_past_horizon to
    # head input_dim). If False, the head sees decoder tokens only.
    'look_price': False,
    # Prediction head width. Must match label family:
    #   trend:   3 = (h,m,b),  2 = (m,b)
    #   yield:   2 = both;  1 = chance or near  (forced from yield_target)
    #   latent:  forced from latent_distribution (2=normal, 3=student)
    'output_dim': 2,
    
    # ============================================================
    # TRAINING CONFIGURATION
    # ============================================================
    'epochs': 1,
    'batch_size': 20,
    'learning_rate': 0.000_01, # 0.000_01 works long. 0.000_05, # 0.000_000_1 gives the similar speed as 0.000_000_1 x 100
    'save_every_n_minutes': 5,
    'checkpoint_dir': 'trading_checkpoints',
    # How many newest rolling weight files to keep (recency, not quality).
    # Best-val snapshot is stored separately as best_val.pt and is not rotated.
    'recent_checkpoints': 3,
    
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
    # trend (dim 3 uses horizon unweighted; dim 2 omits horizon):
    'slope_loss_weight': 1.0,
    'intercept_loss_weight': 10.0,
    # yield:
    'chance_loss_weight': 1.0,
    'near_loss_weight': 1.0,
    'reconstruction_loss_weight': 1.0,
    'use_reconstruction': False,
}


# Pin data_dir to an absolute project-root path at import time so callers that
# read ``config`` or ``get_config()`` never depend on process cwd.
config['data_dir'] = resolve_data_dir(config.get('data_dir', 'dataset_cache'))


def get_config():
    """Return a copy of the config dictionary (data_dir already absolute)."""
    cfg = config.copy()
    # Re-resolve in case env MACRO_DATA_DIR changed after import / update_config.
    cfg['data_dir'] = resolve_data_dir(cfg.get('data_dir', 'dataset_cache'))
    return cfg


def update_config(new_config):
    """Update config with new values."""
    config.update(new_config)
    if 'data_dir' in new_config:
        config['data_dir'] = resolve_data_dir(config.get('data_dir', 'dataset_cache'))


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