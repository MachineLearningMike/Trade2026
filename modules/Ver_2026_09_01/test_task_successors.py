#!/usr/bin/env python
"""Smoke: Task base + Task_* subclasses (no registry in task_base)."""
from __future__ import annotations

import traceback


def main() -> int:
    from task_base import Task
    from Task_Trend import Task_Trend
    from Task_Yield import Task_Yield
    from Task_Latent import Task_Latent
    from Task_Rise import Task_Rise
    from Task_None import Task_None
    from INFER import _resolve_task

    for cls in (Task_Trend, Task_Yield, Task_Latent, Task_Rise, Task_None):
        assert issubclass(cls, Task), cls

    assert _resolve_task({'label': 'trend'}) is Task_Trend
    assert _resolve_task({'label': 'yield'}) is Task_Yield
    assert _resolve_task({'label': 'latent'}) is Task_Latent
    assert _resolve_task({'label': 'rise_flag'}) is Task_Rise
    assert _resolve_task({'label': 'none'}) is Task_None

    assert Task_Rise.normalize_y is False
    assert Task_None.normalize_y is True
    assert Task_Trend.normalize_y is True
    assert Task_Yield.normalize_y is True
    assert Task_Latent.normalize_y is True

    assert Task_Trend.resolve_model_spec({}).get('model_type') == 'regression'
    assert Task_None.resolve_model_spec({}).get('model_type') == 'vae'
    # Config / CLI override wins when set.
    assert (
        Task_None.resolve_model_spec({'model_type': 'regression'}).get('model_type')
        == 'regression'
    )

    print('task hierarchy smoke ok')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
