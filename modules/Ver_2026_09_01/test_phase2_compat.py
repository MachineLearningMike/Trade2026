#!/usr/bin/env python
"""Phase 2 smoke: registry / compat gates (no full train required)."""
from __future__ import annotations

import sys
import traceback


def main() -> int:
    from task_base import CompatibilityError, assert_compatible, assert_config_compatible
    from task_base import COMPAT, TRUE_LABEL_IDS, describe_matrix, label_cli_choices

    # Inventory present
    assert 'none' in TRUE_LABEL_IDS
    assert 'none' in label_cli_choices()
    assert ('regression', 'rise_flag') in COMPAT
    assert ('vae', 'none') in COMPAT
    assert ('vae', 'yield') not in COMPAT

    # Allowed
    assert assert_compatible('regression', 'rise_flag') == 'supervised'
    assert assert_compatible('vae', 'none') == 'unsupervised'
    assert assert_compatible('regression', 'yield', ensemble_kind='telepred') == 'telepred'
    assert assert_compatible('regression', 'yield', ensemble_kind='error') == 'error'
    assert assert_compatible('regression', 'rise_flag', ensemble_kind='error') == 'error'

    # Rejected
    for args in (
        ('vae', 'yield'),
        ('vae', 'rise_flag'),
        ('regression', 'none'),
        ('regression', 'latent', 'error'),
        ('vae', 'none', 'telepred'),
    ):
        try:
            assert_compatible(*args)
        except CompatibilityError as e:
            assert 'Allowed atomic' in str(e) or 'ensemble_kind' in str(e)
        else:
            raise AssertionError(f'expected CompatibilityError for {args}')

    cfg = {'model_type': 'regression', 'label': 'trend'}
    assert assert_config_compatible(cfg) == 'supervised'

    # labels.Task_None wired
    from Task_None import Task_None
    from task_base import get_task_class, sync_output_dim
    assert get_task_class('none') is Task_None
    c = {'label': 'none', 'output_dim': 2}
    assert sync_output_dim(c) == 1  # Task_None forces placeholder dim 1

    print('phase2 compat smoke ok')
    print(describe_matrix())
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
