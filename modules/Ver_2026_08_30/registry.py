# registry.py
# Explicit (model, label) inventory for Ver_2026_08_30 — see _refactor_plan.txt §0.
#
# True labels live in labels.LABEL_CLASSES. This module is the product-facing
# registry: ids, COMPAT pairs, ensemble kinds, and CLI choice lists.

from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Set, Tuple, Type

# ---------------------------------------------------------------------------
# Label ids (true supervised / unsupervised targets)
# ---------------------------------------------------------------------------

LABEL_TREND = 'trend'
LABEL_YIELD = 'yield'
LABEL_LATENT = 'latent'
LABEL_RISE_FLAG = 'rise_flag'
LABEL_NONE = 'none'

TRUE_LABEL_IDS: Tuple[str, ...] = (
    LABEL_TREND,
    LABEL_YIELD,
    LABEL_LATENT,
    LABEL_RISE_FLAG,
    LABEL_NONE,
)

# Aliases accepted on CLI / config → canonical id
LABEL_ALIASES: Dict[str, str] = {
    'linear_regime': LABEL_TREND,
    'linear_trend': LABEL_TREND,
    'scattered_hope': LABEL_YIELD,
    'riseflag': LABEL_RISE_FLAG,
    'rise': LABEL_RISE_FLAG,
}

# ---------------------------------------------------------------------------
# Model ids
# ---------------------------------------------------------------------------

MODEL_REGRESSION = 'regression'
MODEL_VAE = 'vae'
MODEL_ENSEMBLE = 'ensemble'

ATOMIC_MODEL_IDS: Tuple[str, ...] = (
    MODEL_REGRESSION,
    MODEL_VAE,
)

MODEL_IDS: Tuple[str, ...] = ATOMIC_MODEL_IDS + (MODEL_ENSEMBLE,)

MODEL_ALIASES: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# Atomic COMPAT: (model_id, label_id) → objective kind
# ---------------------------------------------------------------------------

COMPAT: Dict[Tuple[str, str], str] = {
    (MODEL_REGRESSION, LABEL_TREND): 'supervised',
    (MODEL_REGRESSION, LABEL_YIELD): 'supervised',
    (MODEL_REGRESSION, LABEL_LATENT): 'supervised',
    (MODEL_REGRESSION, LABEL_RISE_FLAG): 'supervised',
    (MODEL_VAE, LABEL_NONE): 'unsupervised',
}

# ---------------------------------------------------------------------------
# Ensemble kinds (reuse atomic COMPAT members)
# ---------------------------------------------------------------------------

ENSEMBLE_KINDS: Dict[str, dict] = {
    'telepred': {
        'members': 'repeat_atomic',
        'label_relation': 'identical',
        'fusion': 'delay_bank',
        # any supervised atomic pair
        'requires_supervised': True,
    },
    'error': {
        'members': 'repeat_atomic',
        'label_relation': 'identical',
        'fusion': 'sum_residuals_y_space',
        'atomic_model': MODEL_REGRESSION,
        'atomic_labels': frozenset(
            {LABEL_TREND, LABEL_YIELD, LABEL_RISE_FLAG}
        ),
    },
    'stack': {
        'members': 'list_of_atomic',
        'label_relation': 'collection',
        'fusion': 'FusionSpec',
    },
}


def normalize_label_id(label: Optional[str]) -> str:
    raw = str(label or LABEL_TREND).lower().strip()
    return LABEL_ALIASES.get(raw, raw)


def normalize_model_id(model: Optional[str]) -> str:
    raw = str(model or MODEL_REGRESSION).lower().strip()
    return MODEL_ALIASES.get(raw, raw)


def normalize_ensemble_kind(kind: Optional[str]) -> Optional[str]:
    if kind is None:
        return None
    k = str(kind).lower().strip()
    return k or None


def is_true_label(label: Optional[str]) -> bool:
    return normalize_label_id(label) in TRUE_LABEL_IDS


def is_atomic_model(model: Optional[str]) -> bool:
    return normalize_model_id(model) in ATOMIC_MODEL_IDS


def label_cli_choices() -> Tuple[str, ...]:
    """Canonical --label choices (no aliases)."""
    return TRUE_LABEL_IDS


def model_cli_choices() -> Tuple[str, ...]:
    """Canonical --model-type choices for atomic trains."""
    return ATOMIC_MODEL_IDS


def get_label_class(label: Optional[str]) -> Type:
    """Resolve Label_* class via labels.py (includes aliases)."""
    from labels import get_label_class as _get

    return _get(normalize_label_id(label))


def compat_objective(model: Optional[str], label: Optional[str]) -> Optional[str]:
    """Return objective kind string if pair is in COMPAT, else None."""
    m = normalize_model_id(model)
    lab = normalize_label_id(label)
    return COMPAT.get((m, lab))


def describe_matrix() -> str:
    """Human-readable allowed atomic pairs + ensemble kinds."""
    lines = [
        'Allowed atomic (model, label) pairs:',
    ]
    for (m, lab), kind in sorted(COMPAT.items()):
        lines.append(f'  ({m}, {lab}) → {kind}')
    lines.append('Ensemble kinds:')
    for name, meta in sorted(ENSEMBLE_KINDS.items()):
        lines.append(f'  {name}: {meta}')
    return '\n'.join(lines)
