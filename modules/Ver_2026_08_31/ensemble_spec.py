# ensemble_spec.py
# EnsembleModelSpec + LabelCollection (Phase 4 / refactor_plan §2.7).
#
# An ensemble does NOT have a true supervised label. ensemble_label is a
# LabelCollection (bag of member labels). Member trains stay atomic
# train_one_run; this module describes the bag for infer / reporting.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from task_base import (
    ENSEMBLE_KINDS,
    normalize_ensemble_kind,
    normalize_label_id,
    normalize_model_id,
)


@dataclass
class AtomicMember:
    """One COMPAT atomic (model, label) slot, optionally with delay/stage meta."""

    model_id: str
    label_id: str
    member_id: str = ''
    telepred_delay: Optional[int] = None
    error_stage: Optional[int] = None
    checkpoint_key: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.model_id = normalize_model_id(self.model_id)
        self.label_id = normalize_label_id(self.label_id)
        if not self.member_id:
            parts = [self.model_id, self.label_id]
            if self.telepred_delay is not None:
                parts.append(f'd{int(self.telepred_delay)}')
            if self.error_stage is not None:
                parts.append(f's{int(self.error_stage)}')
            self.member_id = '_'.join(parts)


@dataclass
class LabelCollection:
    """
    ensemble_label — bag of member LabelSpecs (NOT a true find/loss label).

    is_true_label is always False.
    """

    member_labels: List[str]
    kind: Optional[str] = None

    def __post_init__(self):
        self.member_labels = [normalize_label_id(x) for x in self.member_labels]

    @property
    def is_true_label(self) -> bool:
        return False

    def label_signature(self) -> Tuple[str, ...]:
        return tuple(self.member_labels)

    def pack_sample(self, ctx, rows_by_member: Dict[str, dict], stats_by_member: Dict[str, Any]):
        """
        Namespaced sample dict for ensemble infer.

        Expects each member's LabelSpec.pack_sample already applied into
        ``rows_by_member[member_id]``.
        """
        return {
            'features': ctx,
            'members': dict(rows_by_member),
            'fusion_meta': {'kind': self.kind, 'labels': list(self.member_labels)},
            'label_stats': dict(stats_by_member),
        }


@dataclass
class FusionSpec:
    """How to combine member predictions (kind-specific)."""

    kind: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        k = normalize_ensemble_kind(self.kind)
        if k is None or k not in ENSEMBLE_KINDS:
            raise ValueError(
                f"FusionSpec.kind must be one of {tuple(ENSEMBLE_KINDS)}, got {self.kind!r}"
            )
        self.kind = k


@dataclass
class EnsembleModelSpec:
    """
    Model that is a bag of atomic (model, label) members + fusion.

    Not an nn.Module itself — orchestration for telepred / error / stack.
    """

    kind: str
    members: List[AtomicMember]
    fusion: FusionSpec
    label_collection: LabelCollection = field(init=False)

    def __post_init__(self):
        k = normalize_ensemble_kind(self.kind)
        if k is None or k not in ENSEMBLE_KINDS:
            raise ValueError(
                f"EnsembleModelSpec.kind must be one of {tuple(ENSEMBLE_KINDS)}, "
                f"got {self.kind!r}"
            )
        self.kind = k
        if self.fusion.kind != self.kind:
            raise ValueError(
                f"FusionSpec.kind {self.fusion.kind!r} != ensemble kind {self.kind!r}"
            )
        labels = [m.label_id for m in self.members]
        self.label_collection = LabelCollection(member_labels=labels, kind=self.kind)

    @classmethod
    def telepred(
        cls,
        model_id: str,
        label_id: str,
        delays: Sequence[int],
        checkpoint_keys: Optional[Sequence[Optional[str]]] = None,
    ) -> 'EnsembleModelSpec':
        """Repeat one atomic pair over TelePred delays."""
        keys = list(checkpoint_keys or [None] * len(delays))
        if len(keys) != len(delays):
            raise ValueError('checkpoint_keys length must match delays')
        members = [
            AtomicMember(
                model_id=model_id,
                label_id=label_id,
                telepred_delay=int(d),
                checkpoint_key=keys[i],
                member_id=f'{normalize_model_id(model_id)}_{normalize_label_id(label_id)}_d{int(d)}',
            )
            for i, d in enumerate(delays)
        ]
        return cls(
            kind='telepred',
            members=members,
            fusion=FusionSpec(kind='telepred', params={'delays': list(map(int, delays))}),
        )

    @classmethod
    def error(
        cls,
        label_id: str,
        n_stages: int,
        model_id: str = 'regression',
        checkpoint_keys: Optional[Sequence[Optional[str]]] = None,
    ) -> 'EnsembleModelSpec':
        """Residual ErrorEnsemble stages model_0 .. model_{n-1}."""
        n = int(n_stages)
        if n < 1:
            raise ValueError(f'n_stages must be >= 1, got {n}')
        keys = list(checkpoint_keys or [None] * n)
        if len(keys) != n:
            raise ValueError('checkpoint_keys length must match n_stages')
        members = [
            AtomicMember(
                model_id=model_id,
                label_id=label_id,
                error_stage=s,
                checkpoint_key=keys[s],
                member_id=f'{normalize_model_id(model_id)}_{normalize_label_id(label_id)}_s{s}',
            )
            for s in range(n)
        ]
        return cls(
            kind='error',
            members=members,
            fusion=FusionSpec(
                kind='error',
                params={'n_stages': n, 'space': 'y_normalized'},
            ),
        )


def describe_ensemble(spec: EnsembleModelSpec) -> str:
    lines = [
        f'EnsembleModelSpec kind={spec.kind}',
        f'  fusion={spec.fusion.params}',
        f'  ensemble_label (collection)={list(spec.label_collection.member_labels)}',
        f'  members ({len(spec.members)}):',
    ]
    for m in spec.members:
        lines.append(
            f'    - {m.member_id}: ({m.model_id}, {m.label_id}) '
            f'delay={m.telepred_delay} stage={m.error_stage} key={m.checkpoint_key}'
        )
    return '\n'.join(lines)
