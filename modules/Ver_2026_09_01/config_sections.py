# config_sections.py
# Phase 6 — sectioned view of the flat config dict (runtime stays flat).
#
# Sections: paths | data | label | model | train | ensemble | device
# apply_overrides(cfg, mapping) mutates a *copy-friendly* local cfg.
# Prefer: cfg = get_config(); apply_overrides(cfg, ...); never update_config
# for one-off runs.

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

# Keys belonging to each logical section (documentation + filtering).
CONFIG_SECTIONS: Dict[str, Sequence[str]] = {
    'paths': (
        'data_dir', 'checkpoint_dir',
    ),
    'data': (
        'past_window', 'future_window', 'step',
        'telepred_delay', 'telepred_delays', 'telepred_definition_id',
        'num_momentum',
        'train_ratio', 'val_ratio', 'shuffle_split',
        'binance_primary_symbol', 'binance_symbols',
        'binance_datatype', 'binance_interval',
        'binance_start_date', 'binance_end_date',
        'normalization_layers',
        'price_feature_idx', 'prediction_target',
        'macros_to_exclude', 'macros_to_include',
        'n_forget',
    ),
    'label': (
        'label', 'yield_target', 'latent_distribution',
        'min_yield_percent', 'min_trend_horizon',
        'max_accum_deviation', 'fitness_threshold',
        'horizon_search', 'horizon_mode',
        'output_dim',
        'slope_loss_weight', 'intercept_loss_weight',
        'chance_loss_weight', 'near_loss_weight',
        'rise_flag_loss_weight', 'rise_flag_decision_threshold',
        'latent_loss_weight',
    ),
    'model': (
        'model_type', 'vae_latent_dim', 'vae_kl_weight',
        'patch_size', 'embed_dim', 'num_layers', 'num_heads',
        'ff_dim_multiplier', 'dropout', 'causal', 'max_seq_len',
        'head_hidden_dim_multiplier',
        'n_latent_macros',
        'emit_reconstruction', 'use_reconstruction',
        'reconstruction_loss_weight',
    ),
    'train': (
        'epochs', 'batch_size', 'learning_rate',
        'save_every_n_minutes', 'recent_checkpoints',
        'resume', 'clean', 'checkpoint_key',
        'validate', 'validate_every_n_epochs', 'validate_batch_size',
        'test_batch_size', 'test_viz_samples',
    ),
    'ensemble': (
        'error_ensemble_n', 'error_ensemble_stage',
        'telepred_delays', 'telepred_delay',
    ),
    'device': (
        'device',
    ),
}

# Label-private knobs (ignore when another label is active — soft filter).
LABEL_PRIVATE_KEYS: Dict[str, Sequence[str]] = {
    'trend': (
        'min_trend_horizon', 'fitness_threshold', 'horizon_search',
        'slope_loss_weight', 'intercept_loss_weight',
        'max_accum_deviation', 'horizon_mode',
    ),
    'yield': (
        'yield_target', 'min_yield_percent',
        'chance_loss_weight', 'near_loss_weight',
    ),
    'latent': (
        'latent_distribution', 'latent_loss_weight',
    ),
    'rise_flag': (
        'min_yield_percent', 'rise_flag_loss_weight',
        'rise_flag_decision_threshold',
    ),
    'none': (),
}

MODEL_PRIVATE_KEYS: Dict[str, Sequence[str]] = {
    'regression': (
        'emit_reconstruction', 'use_reconstruction',
    ),
    'vae': (
        'vae_latent_dim', 'vae_kl_weight', 'reconstruction_loss_weight',
    ),
}


def clone_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Shallow copy suitable for per-trial / per-delay overrides."""
    return dict(cfg)


def apply_overrides(
    cfg: MutableMapping[str, Any],
    overrides: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> MutableMapping[str, Any]:
    """
    Apply overrides onto a local config mapping (mutates ``cfg``).

    Does not touch the process-global ``config`` module dict.
    """
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
    for k, v in kwargs.items():
        if v is not None:
            cfg[k] = v
    return cfg


def section_view(cfg: Mapping[str, Any], section: str) -> Dict[str, Any]:
    """Return {key: value} for keys declared in CONFIG_SECTIONS[section]."""
    if section not in CONFIG_SECTIONS:
        raise KeyError(
            f"Unknown section {section!r}; expected one of {tuple(CONFIG_SECTIONS)}"
        )
    keys = CONFIG_SECTIONS[section]
    return {k: cfg[k] for k in keys if k in cfg}


def all_sections(cfg: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {name: section_view(cfg, name) for name in CONFIG_SECTIONS}


def active_label_keys(label: str) -> List[str]:
    """Shared label keys + private keys for the active label family."""
    shared = ['label', 'output_dim', 'future_window']
    priv = list(LABEL_PRIVATE_KEYS.get(str(label).lower(), ()))
    return shared + priv


def active_model_keys(model_type: str) -> List[str]:
    shared = [
        'model_type', 'patch_size', 'num_layers', 'num_heads',
        'ff_dim_multiplier', 'dropout', 'causal',
        'n_latent_macros', 'head_hidden_dim_multiplier',
    ]
    priv = list(MODEL_PRIVATE_KEYS.get(str(model_type).lower(), ()))
    return shared + priv


def filter_active_knobs(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Subset of cfg relevant to the active (model, label) product cell.

    Other label/model private knobs are omitted (still present in full cfg).
    """
    label = str(cfg.get('label', 'trend')).lower()
    model = str(cfg.get('model_type', 'regression')).lower()
    keys = set(active_label_keys(label)) | set(active_model_keys(model))
    # Always keep data/train/ensemble/paths essentials
    for sec in ('paths', 'data', 'train', 'ensemble', 'device'):
        keys.update(CONFIG_SECTIONS[sec])
    return {k: cfg[k] for k in sorted(keys) if k in cfg}


def apply_cli_namespace(cfg: MutableMapping[str, Any], args: Any) -> MutableMapping[str, Any]:
    """
    Map TRAIN argparse namespace fields onto a local cfg (Phase 6).

    Centralizes the old TRAIN.py hand-written assignments.
    """
    mapping = {
        'epochs': getattr(args, 'epochs', None),
        'save_every_n_minutes': getattr(args, 'save_every', None),
        'clean': getattr(args, 'clean', None),
        'past_window': getattr(args, 'past_window', None),
        'future_window': getattr(args, 'future_window', None),
        'batch_size': getattr(args, 'batch_size', None),
        'step': getattr(args, 'step', None),
        'max_accum_deviation': getattr(args, 'max_accum_deviation', None),
        'min_trend_horizon': getattr(args, 'min_trend_horizon', None),
        'min_yield_percent': getattr(args, 'min_yield_percent', None),
        'model_type': getattr(args, 'model_type', None),
        'label': getattr(args, 'label', None),
        'yield_target': getattr(args, 'yield_target', None),
        'latent_distribution': getattr(args, 'latent_distribution', None),
        'learning_rate': getattr(args, 'lr', None),
        'slope_loss_weight': getattr(args, 'slope_weight', None),
        'intercept_loss_weight': getattr(args, 'intercept_weight', None),
        'chance_loss_weight': getattr(args, 'chance_weight', None),
        'near_loss_weight': getattr(args, 'near_weight', None),
        'rise_flag_loss_weight': getattr(args, 'rise_flag_weight', None),
        'reconstruction_loss_weight': getattr(args, 'recon_weight', None),
        'vae_kl_weight': getattr(args, 'vae_kl_weight', None),
        'patch_size': getattr(args, 'patch_size', None),
        'num_layers': getattr(args, 'num_layers', None),
        'num_heads': getattr(args, 'num_heads', None),
        'ff_dim_multiplier': getattr(args, 'ff_dim_multiplier', None),
        'dropout': getattr(args, 'dropout', None),
        'head_hidden_dim_multiplier': getattr(
            args, 'head_hidden_dim_multiplier', None
        ),
        'train_ratio': getattr(args, 'train_ratio', None),
        'val_ratio': getattr(args, 'val_ratio', None),
        'validate_every_n_epochs': getattr(args, 'validate_every', None),
        'validate_batch_size': getattr(args, 'val_batch_size', None),
        'telepred_delay': getattr(args, 'telepred_delay', None),
        'binance_primary_symbol': getattr(args, 'symbol', None),
    }
    # Invert-flags: only apply when explicitly set (not None).
    # Avoid stamping CLI defaults back onto cfg every run — that can churn
    # dataset_signature / label-cache keys even when the user only changed
    # epochs (which are intentionally NOT part of the label cache key).
    if getattr(args, 'no_causal', None) is not None:
        mapping['causal'] = not bool(args.no_causal)
    if getattr(args, 'no_reconstruction', None) is not None:
        emit = not bool(args.no_reconstruction)
        mapping['use_reconstruction'] = emit
        mapping['emit_reconstruction'] = emit
    if getattr(args, 'no_validate', None) is not None:
        mapping['validate'] = not bool(args.no_validate)
    if getattr(args, 'normalization_layers', None):
        mapping['normalization_layers'] = str(args.normalization_layers).strip().lower()
    if getattr(args, 'checkpoint_key', None):
        mapping['checkpoint_key'] = str(args.checkpoint_key).strip()

    apply_overrides(cfg, mapping)
    return cfg
