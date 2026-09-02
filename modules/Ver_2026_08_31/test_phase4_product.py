#!/usr/bin/env python
"""Phase 4 smoke: COMPAT product coercion, wrap_loss, ensemble specs, head dims."""
from __future__ import annotations

import traceback

import torch
import torch.nn as nn


def main() -> int:
    from task_base import CompatibilityError, assert_compatible, resolve_atomic_slot
    from ensemble_spec import EnsembleModelSpec, LabelCollection, describe_ensemble
    from model_regression import PredictionHead
    from model_v_autoencoder import wrap_loss

    # output_dim relaxed
    h = PredictionHead(8, 16, output_dim=5)
    assert h.output_dim == 5
    try:
        PredictionHead(8, 16, output_dim=0)
        raise AssertionError('expected fail on output_dim=0')
    except ValueError:
        pass

    # vae ⇒ none coerce
    cfg = {'model_type': 'vae', 'label': 'yield'}
    obj = resolve_atomic_slot(cfg, coerce_vae_to_none=True, warn=False)
    assert obj == 'unsupervised'
    assert cfg['label'] == 'none'

    assert assert_compatible('regression', 'rise_flag') == 'supervised'
    try:
        assert_compatible('vae', 'yield')
        raise AssertionError('vae+yield must fail without coerce')
    except CompatibilityError:
        pass

    # wrap_loss supervised vs unsupervised
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.last_x = torch.zeros(2, 3, 4)
            self.last_x_hat = torch.zeros(2, 3, 4)
            self.last_kl = torch.tensor(0.1)

    def lab_loss(x, pred, y):
        return (pred - y).pow(2).mean()

    m = _M()
    sup = wrap_loss(
        m, lab_loss, {'model_type': 'regression', 'label': 'trend'},
        objective='supervised',
    )
    pred = torch.zeros(2, 1)
    y = torch.ones(2, 1)
    assert float(sup(torch.zeros(2, 3, 4), pred, y)) >= 0.0

    unsup = wrap_loss(
        m, lab_loss, {'model_type': 'vae', 'label': 'none'},
        objective='unsupervised',
    )
    loss = unsup(torch.zeros(2, 3, 4), pred, y)
    assert float(loss) > 0.0

    # Ensemble specs
    tele = EnsembleModelSpec.telepred('regression', 'yield', [0, 2, 5])
    assert tele.kind == 'telepred'
    assert len(tele.members) == 3
    assert isinstance(tele.label_collection, LabelCollection)
    assert tele.label_collection.is_true_label is False
    assert tele.label_collection.member_labels == ['yield', 'yield', 'yield']

    err = EnsembleModelSpec.error('rise_flag', 3)
    assert err.kind == 'error'
    assert err.members[1].error_stage == 1
    print(describe_ensemble(tele))
    print(describe_ensemble(err))

    print('phase4 product smoke ok')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
