#!/usr/bin/env python
"""Phase 2 smoke: task resolve still works (compat matrix removed from task_base)."""
from __future__ import annotations

import traceback


def main() -> int:
    from Task_None import Task_None
    from INFER import _resolve_task

    assert _resolve_task({'label': 'none'}) is Task_None
    c = {'label': 'none', 'output_dim': 2}
    assert _resolve_task(c).sync_output_dim(c) == 1

    print('phase2 task resolve smoke ok')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
