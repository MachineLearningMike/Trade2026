#!/usr/bin/env python
"""Phase 5 smoke: subcommand split + label_metrics.report_test_metrics."""
from __future__ import annotations

import traceback

import torch


def main() -> int:
    from cli_commands import (
        COMMANDS,
        apply_command_to_namespace,
        split_command_argv,
    )
    from label_metrics import report_test_metrics

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

    # rise_flag metrics via LabelSpec path
    preds = torch.tensor([[2.0], [-1.0], [0.1], [3.0]])
    labels = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
    m = report_test_metrics(
        'rise_flag', preds, labels, print_samples=False,
    )
    assert 'bce_with_logits' in m and 'accuracy' in m
    assert 0.0 <= m['accuracy'] <= 1.0

    # trend dim 2
    preds_t = torch.randn(4, 2)
    labels_t = torch.randn(4, 2)
    m2 = report_test_metrics(
        'trend', preds_t, labels_t, config={'output_dim': 2}, print_samples=False,
    )
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
