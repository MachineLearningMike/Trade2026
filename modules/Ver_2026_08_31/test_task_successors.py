#!/usr/bin/env python
"""Smoke: Task base + Task_* classes (no Label_* compat)."""
from __future__ import annotations

import traceback


def main() -> int:
    from task_base import Task
    from Task_Trend import Task_Trend
    from Task_Yield import Task_Yield
    from Task_Latent import Task_Latent
    from Task_Rise import Task_Rise
    from Task_None import Task_None
    from task_base import get_task_class, TASK_CLASSES
    from Task_Trend import Task_Trend as TT
    from Task_Yield import Task_Yield as TY
    from Task_Latent import Task_Latent as TL
    from Task_Rise import Task_Rise as TR

    for cls in (Task_Trend, Task_Yield, Task_Latent, Task_Rise, Task_None):
        assert issubclass(cls, Task), cls

    assert TT is Task_Trend and TY is Task_Yield
    assert TL is Task_Latent and TR is Task_Rise

    assert get_task_class('trend') is Task_Trend
    assert get_task_class('yield') is Task_Yield
    assert get_task_class('latent') is Task_Latent
    assert get_task_class('rise_flag') is Task_Rise
    assert get_task_class('none') is Task_None
    assert set(TASK_CLASSES) == {'trend', 'yield', 'latent', 'rise_flag', 'none'}

    assert Task_Rise.normalize_policy() == 'identity'
    assert Task_Trend.normalize_policy() == 'zscore'

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
