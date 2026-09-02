#!/usr/bin/env python
"""Phase 5 smoke: subcommand split + task.pack_metrics."""
from __future__ import annotations

import traceback

import torch


def main() -> int:
    from cli_commands import (
        COMMANDS,
        apply_command_to_namespace,
        split_command_argv,
    )

    assert 'train' in COMMANDS and 'error-ensemble' in COMMANDS

    cmd, rest = split_command_argv(['infer', '--label', 'rise_flag'])
    assert cmd == 'infer' and rest == ['--label', 'rise_flag']
    cmd2, rest2 = split_command_argv(['--label', 'yield'])
    assert cmd2 is None and rest2 == ['--label', 'yield']

    class NS:
        test = None
        validate = None
        ensemble = False
        sweep = False
        error_ensemble = False

    ns = NS()
    apply_command_to_namespace('infer', ns)
    assert ns.test == -1
    ns2 = NS()
    apply_command_to_namespace('ensemble', ns2)
    assert ns2.ensemble is True
    ns3 = NS()
    apply_command_to_namespace('error-ensemble', ns3)
    assert ns3.error_ensemble is True

    preds = torch.tensor([[2.0], [-1.0], [0.1], [3.0]])
    labels = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
    from Task_Rise import Task_Rise
    from Task_Trend import Task_Trend
    packed = Task_Rise.pack_metrics(preds, labels, config={})
    m = Task_Rise.unpack_metrics(packed)
    for k, v in packed.items():
        m.setdefault(k, v)
    assert 'bce_with_logits' in m and 'accuracy' in m
    assert 0.0 <= m['accuracy'] <= 1.0

    preds_t = torch.randn(4, 2)
    labels_t = torch.randn(4, 2)
    packed_t = Task_Trend.pack_metrics(
        preds_t, labels_t, config={'output_dim': 2},
    )
    m2 = Task_Trend.unpack_metrics(packed_t)
    for k, v in packed_t.items():
        m2.setdefault(k, v)
    assert 'slope_mse' in m2 and 'intercept_mse' in m2

    print('phase5 cli+metrics smoke ok')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
