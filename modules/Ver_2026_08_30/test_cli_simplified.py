#!/usr/bin/env python
"""Smoke: simplified TRAIN CLI parses to the right mode flags."""
from __future__ import annotations

import traceback


def main() -> int:
    from config import get_config
    from cli_app import try_parse_simplified, wants_legacy_parser, apply_set_pairs

    cfg = get_config()

    assert wants_legacy_parser(['--history']) is True
    assert wants_legacy_parser(['train', '--label', 'yield']) is False
    assert wants_legacy_parser(['--label', 'yield', '--epochs', '2']) is False

    a = try_parse_simplified(cfg, ['train', '--model', 'regression', '--label', 'rise_flag', '--epochs', '3'])
    assert a is not None
    assert a.label == 'rise_flag'
    assert a.model_type == 'regression'
    assert a.epochs == 3
    assert a.test is None
    assert a.ensemble is False

    b = try_parse_simplified(cfg, ['infer', '--label', 'yield', '--test', '16', '--no-viz'])
    assert b.test == 16
    assert b.no_viz is True

    c = try_parse_simplified(cfg, ['validate', '--latest', '--slice', '0:30'])
    assert c.last_validate == [0.0, 30.0]
    assert c.validate is None

    d = try_parse_simplified(cfg, ['ensemble', '--delays', '0,2,5'])
    assert d.ensemble is True
    assert d.telepred_delays == '0,2,5'

    e = try_parse_simplified(cfg, ['error-ensemble', '--stages', '4', '--extend-stages', '2'])
    assert e.error_ensemble is True
    assert e.error_ensemble_extend == 2
    assert any(str(x).startswith('error_ensemble_n=4') for x in (e.set_pairs or []))

    f = try_parse_simplified(cfg, ['plot', '--history'])
    assert f.history == 'current'

    g = try_parse_simplified(cfg, ['report', 'my_run'])
    assert g.report == 'my_run'
    assert g.report_git_commit is False

    h = try_parse_simplified(cfg, ['report', 'ens', '--ensemble'])
    assert h.ensemble_report == 'ens'
    assert h.report_git_commit is False

    bare = try_parse_simplified(cfg, ['report'])
    assert bare.report is not None
    assert bare.report_git_commit is True
    # YYYYMMDD_HHMMSS
    assert len(str(bare.report)) == 15 and str(bare.report)[8] == '_'

    bare_nc = try_parse_simplified(cfg, ['report', '--no-commit'])
    assert bare_nc.report_git_commit is False
    assert len(str(bare_nc.report)) == 15

    # Footgun: ``train report`` → treat as ``report``
    stacked = try_parse_simplified(cfg, ['train', 'report', '--no-commit'])
    assert stacked.cli_command == 'report'
    assert stacked.report is not None
    assert stacked.report_git_commit is False

    i = try_parse_simplified(cfg, ['tools', 'viz-trend'])
    assert i.trend is True

    j = try_parse_simplified(
        cfg, ['train', '--label', 'yield', '--set', 'patch_size=15', '--set', 'num_layers=2']
    )
    local = dict(cfg)
    apply_set_pairs(local, j.set_pairs)
    assert local['patch_size'] == 15
    assert local['num_layers'] == 2

    # flags-only → defaults to train
    k = try_parse_simplified(cfg, ['--label', 'latent', '--epochs', '1'])
    assert k is not None
    assert k.label == 'latent'
    assert k.cli_command == 'train'

    print('cli simplified smoke ok')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
