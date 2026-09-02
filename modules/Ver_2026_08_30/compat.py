# compat.py
# Fail-fast gates for (model, label) and ensemble kinds — refactor_plan §0 / Phase 2.

from __future__ import annotations

from typing import Optional

from registry import (
    COMPAT,
    ENSEMBLE_KINDS,
    MODEL_ENSEMBLE,
    MODEL_REGRESSION,
    MODEL_VAE,
    describe_matrix,
    normalize_ensemble_kind,
    normalize_label_id,
    normalize_model_id,
)


class CompatibilityError(ValueError):
    """Unsupported (model, label) or ensemble kind."""


def assert_compatible(
    model: Optional[str],
    label: Optional[str],
    ensemble_kind: Optional[str] = None,
) -> str:
    """
    Validate an atomic or ensemble training/infer slot.

    Returns the COMPAT objective kind for atomic pairs (e.g. 'supervised'),
    or the ensemble kind name when ``ensemble_kind`` is set.

    Raises CompatibilityError with the allowed matrix on failure.
    """
    m = normalize_model_id(model)
    lab = normalize_label_id(label)
    kind = normalize_ensemble_kind(ensemble_kind)

    if m == MODEL_ENSEMBLE and kind is None:
        raise CompatibilityError(
            f"model_type={m!r} requires an ensemble_kind "
            f"(one of {tuple(ENSEMBLE_KINDS)}).\n"
            f"{describe_matrix()}"
        )

    if kind is not None:
        return _assert_ensemble(m, lab, kind)

    if m not in (MODEL_REGRESSION, MODEL_VAE):
        raise CompatibilityError(
            f"Unknown or non-atomic model_type={m!r}. "
            f"Atomic models: {(MODEL_REGRESSION, MODEL_VAE)}; "
            f"use ensemble_kind for composed models.\n"
            f"{describe_matrix()}"
        )

    objective = COMPAT.get((m, lab))
    if objective is None:
        # Helpful hint for the old silent VAE+supervised lie.
        hint = ''
        if m == MODEL_VAE and lab != 'none':
            hint = (
                f"\nHint: VAE train objective ignores supervised labels; "
                f"use --label none (pair (vae, none)). "
                f"Got label={lab!r}."
            )
        raise CompatibilityError(
            f"Unsupported pair (model_type={m!r}, label={lab!r})."
            f"{hint}\n"
            f"{describe_matrix()}"
        )
    return objective


def _assert_ensemble(model: str, label: str, kind: str) -> str:
    meta = ENSEMBLE_KINDS.get(kind)
    if meta is None:
        raise CompatibilityError(
            f"Unknown ensemble_kind={kind!r}; "
            f"expected one of {tuple(ENSEMBLE_KINDS)}.\n"
            f"{describe_matrix()}"
        )

    if kind == 'telepred':
        # Member is a single atomic pair, repeated over delays.
        # model should be atomic (regression/vae), not 'ensemble'.
        m = model if model != MODEL_ENSEMBLE else MODEL_REGRESSION
        # Prefer explicit model from config; if caller passed ensemble, still
        # validate label against supervised COMPAT via regression default — but
        # TRAIN passes real model_type.
        objective = COMPAT.get((normalize_model_id(m), label))
        if objective is None:
            raise CompatibilityError(
                f"ensemble_kind='telepred' requires a COMPAT atomic pair; "
                f"got (model_type={m!r}, label={label!r}).\n"
                f"{describe_matrix()}"
            )
        if meta.get('requires_supervised') and objective != 'supervised':
            raise CompatibilityError(
                f"ensemble_kind='telepred' requires a supervised pair; "
                f"got ({m!r}, {label!r}) → {objective!r}.\n"
                f"{describe_matrix()}"
            )
        return kind

    if kind == 'error':
        m = normalize_model_id(model)
        if m == MODEL_ENSEMBLE:
            m = MODEL_REGRESSION
        allowed_model = meta.get('atomic_model', MODEL_REGRESSION)
        allowed_labels = meta.get('atomic_labels', frozenset())
        if m != allowed_model:
            raise CompatibilityError(
                f"ensemble_kind='error' requires model_type={allowed_model!r}; "
                f"got {m!r}.\n"
                f"{describe_matrix()}"
            )
        if label not in allowed_labels:
            raise CompatibilityError(
                f"ensemble_kind='error' requires label in "
                f"{sorted(allowed_labels)}; got {label!r}.\n"
                f"{describe_matrix()}"
            )
        # Also must be a COMPAT cell
        if COMPAT.get((m, label)) is None:
            raise CompatibilityError(
                f"ensemble_kind='error' pair ({m!r}, {label!r}) not in COMPAT.\n"
                f"{describe_matrix()}"
            )
        return kind

    if kind == 'stack':
        # Members validated by caller when FusionSpec is implemented.
        return kind

    raise CompatibilityError(
        f"ensemble_kind={kind!r} has no assert rules yet.\n"
        f"{describe_matrix()}"
    )


def assert_config_compatible(
    config: dict,
    *,
    ensemble_kind: Optional[str] = None,
) -> str:
    """assert_compatible using config['model_type'] / config['label']."""
    return assert_compatible(
        config.get('model_type', MODEL_REGRESSION),
        config.get('label', 'trend'),
        ensemble_kind=ensemble_kind,
    )


def resolve_atomic_slot(
    config: dict,
    *,
    coerce_vae_to_none: bool = True,
    warn: bool = True,
) -> str:
    """
    Normalize config to a COMPAT atomic slot and return the objective kind.

    Phase 4: if model_type=vae and label is not none, coerce label→none
    (default) so the product cell is honest: (vae, none) unsupervised.
    Mutates ``config`` in place when coercing.
    """
    from registry import LABEL_NONE

    m = normalize_model_id(config.get('model_type', MODEL_REGRESSION))
    lab = normalize_label_id(config.get('label', 'trend'))
    config['model_type'] = m

    if coerce_vae_to_none and m == MODEL_VAE and lab != LABEL_NONE:
        if warn:
            print(
                f"⚠ model_type=vae requires label=none (COMPAT unsupervised); "
                f"coercing label {lab!r} → {LABEL_NONE!r}"
            )
        lab = LABEL_NONE
        config['label'] = LABEL_NONE

    config['label'] = lab
    return assert_compatible(m, lab, ensemble_kind=None)
