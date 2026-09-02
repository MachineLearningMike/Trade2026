# cli_app.py
# Simplified TRAIN CLI (_cli_simplify_plan.txt).
#
# Remember ~8 verbs, not ~60 flags:
#   train | infer | validate | ensemble | error-ensemble | plot | report | search | tools
#
# Shared product flags: --model --label --yield-target --latent-dist
#   --epochs --extend --clean --delay --delays --checkpoint --set KEY=VAL
# Escape hatch: --set patch_size=15
# Legacy mega-flag invocations still work (deprecation warning).

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple

from cli_commands import COMMANDS

# Old-style exclusive flags → force legacy mega-parser (with warning).
_LEGACY_HINT_FLAGS = (
    '--history', '--loss', '--validation', '--trend', '--features',
    '--vae-compare', '--vae-key', '--yield-key', '--validate-yield',
    '--ensemble-report', '--last-validate',
    '--vae-sweep', '--vae-search', '--ensemble-sweep',
    '--search', '--sweep',
    '--model-type', '--checkpoint-key', '--telepred-delay', '--telepred-delays',
    '--error-ensemble', '--error-ensemble-extend',
)

_CHEATSHEET = """
Daily cheat-sheet
  python TRAIN.py train --model regression --label rise_flag --epochs 12
  python TRAIN.py infer --label rise_flag --test 32
  python TRAIN.py validate --latest --slice 0:30
  python TRAIN.py ensemble --label yield --delays 0,2,5,10
  python TRAIN.py error-ensemble --label yield --stages 3 --clean
  python TRAIN.py plot --history
  python TRAIN.py plot --loss
  python TRAIN.py report              # → _save/YYYYMMDD_HHMMSS/ + git commit
  python TRAIN.py report my_run
  python TRAIN.py train --label yield --set patch_size=15 --set num_layers=6

Rare:  python TRAIN.py tools compare --vae-key K1 --yield-key K2
       python TRAIN.py tools viz-trend | viz-features
       python TRAIN.py search --model vae
"""


def _blank_args() -> SimpleNamespace:
    """Namespace with every attribute TRAIN.main / apply_cli_namespace may read."""
    return SimpleNamespace(
        # run control
        epochs=None, save_every=None, clean=False, extend=None,
        # modes (mutually selected by command)
        trend=False, history=None, loss=None, validation=None,
        report=None, ensemble_report=None,
        report_git_commit=False,
        checkpoint_key=None,
        vae_compare=None, vae_key=None, yield_key=None,
        sweep=False, vae_sweep=False, sweep_epochs=1, sweep_extend=1.0,
        telepred_delay=None, telepred_delays=None,
        ensemble=False, ensemble_sweep=False,
        validate=None, last_validate=None,
        features=False, validate_yield=None, test=None,
        # product / data
        past_window=None, future_window=None, batch_size=None,
        step=None, max_accum_deviation=None,
        min_trend_horizon=None, min_yield_percent=None,
        model_type=None, label=None, yield_target=None,
        latent_distribution=None,
        error_ensemble=False, error_ensemble_extend=None,
        normalization_layers=None,
        lr=None, slope_weight=None, intercept_weight=None,
        chance_weight=None, near_weight=None, rise_flag_weight=None,
        recon_weight=None, vae_kl_weight=None,
        patch_size=None, num_layers=None, num_heads=None,
        ff_dim_multiplier=None, dropout=None,
        head_hidden_dim_multiplier=None,
        # None = leave config.py unchanged (do not re-stamp every run)
        no_causal=None, no_reconstruction=None, no_look_price=None,
        symbol=None,
        train_ratio=None, val_ratio=None,
        no_validate=None, validate_every=None, val_batch_size=None,
        # simplified extras
        set_pairs=None, no_viz=False, cli_command=None,
        _simplified=True,
    )


def _add_shared(p: argparse.ArgumentParser, config: dict) -> None:
    p.add_argument(
        '--model', '--model-type', dest='model_type',
        default=None, choices=['regression', 'vae'],
        help="Model: regression | vae (default from config)",
    )
    p.add_argument(
        '--label', default=None, choices=['trend', 'yield', 'latent', 'rise_flag', 'none'],
        help='Label: trend|yield|latent|rise_flag|none',
    )
    p.add_argument(
        '--yield-target', default=None, choices=['both', 'chance', 'near'],
        help='Yield head when --label yield',
    )
    p.add_argument(
        '--latent-dist', '--latent-distribution', dest='latent_distribution',
        default=None,
        choices=['normal', 'student', 'normal_distribution',
                 'student_distribution', 'student_t'],
        help='Latent density when --label latent',
    )
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--extend', type=float, default=None)
    p.add_argument('--clean', action='store_true', default=False)
    p.add_argument(
        '--delay', '--telepred-delay', dest='telepred_delay',
        type=int, default=None, metavar='D',
        help='Single TelePred delay (bars)',
    )
    p.add_argument(
        '--delays', '--telepred-delays', dest='telepred_delays',
        type=str, default=None, metavar='LIST',
        help='Comma-separated TelePred delays',
    )
    p.add_argument(
        '--checkpoint', '--checkpoint-key', dest='checkpoint_key',
        type=str, default=None, metavar='KEY',
    )
    p.add_argument(
        '--past', '--past-window', dest='past_window', type=int, default=None,
    )
    p.add_argument(
        '--future', '--future-window', dest='future_window', type=int, default=None,
    )
    p.add_argument(
        '--batch', '--batch-size', dest='batch_size', type=int, default=None,
    )
    p.add_argument(
        '--set', dest='set_pairs', action='append', default=None, metavar='KEY=VAL',
        help='Override config key (repeatable), e.g. --set patch_size=15',
    )


def _parse_slice(s: Optional[str]) -> Optional[List[float]]:
    if s is None or str(s).strip() == '':
        return []
    text = str(s).strip().replace(',', ':')
    if ':' in text:
        a, b = text.split(':', 1)
        return [float(a), float(b)]
    return [float(text)]


def build_simplified_parser(config: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='TRAIN.py',
        description='Trading TRAIN/INFER — simplified command CLI',
        epilog=_CHEATSHEET,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='cli_command', required=True)

    # train
    p_train = sub.add_parser('train', help='Single-element training',
                             formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_shared(p_train, config)
    p_train.add_argument('--save-every', type=float, default=None)
    p_train.add_argument('--no-validate', action='store_true', default=False)

    # infer
    p_infer = sub.add_parser('infer', aliases=['test'], help='Test-set evaluation',
                             formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_shared(p_infer, config)
    p_infer.add_argument(
        '--test', nargs='?', const=-1, default=-1, type=int, metavar='N',
        help='Sample cap (default: all)',
    )
    p_infer.add_argument('--no-viz', action='store_true', default=False)

    # validate
    p_val = sub.add_parser('validate', help='Checkpoint validation curves',
                           formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_shared(p_val, config)
    p_val.set_defaults(validate_mode='best')
    g = p_val.add_mutually_exclusive_group()
    g.add_argument('--best', action='store_const', const='best', dest='validate_mode',
                   help='Use best_val checkpoint (default)')
    g.add_argument('--latest', action='store_const', const='latest', dest='validate_mode',
                   help='Use latest checkpoint')
    p_val.add_argument(
        '--slice', dest='val_slice', type=str, default=None, metavar='V|V1:V2',
        help='Percent slice, e.g. 30 or 0:30',
    )

    # ensemble
    p_ens = sub.add_parser('ensemble', help='TelePred delay bank',
                           formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_shared(p_ens, config)
    p_ens.add_argument('--save-every', type=float, default=None)

    # error-ensemble
    p_err = sub.add_parser(
        'error-ensemble', help='Residual ErrorEnsemble stages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_shared(p_err, config)
    p_err.add_argument('--stages', type=int, default=None, metavar='N',
                       help='Total stages (error_ensemble_n)')
    p_err.add_argument(
        '--extend-stages', type=int, default=None, metavar='N',
        help='Add N stages after finished ones',
    )
    p_err.add_argument('--save-every', type=float, default=None)

    # plot
    p_plot = sub.add_parser('plot', help='History / loss / validation curves',
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_shared(p_plot, config)
    p_plot.add_argument('--history', nargs='?', const='current', default=None)
    p_plot.add_argument('--loss', nargs='?', const='current', default=None)
    p_plot.add_argument('--validation', nargs='*', default=None, metavar='E')

    # report
    p_rep = sub.add_parser(
        'report',
        help=(
            'Batch-export validation + history plots under _save/<name>/. '
            'Omit name → YYYYMMDD_HHMMSS and git-commit all changes with that message.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_shared(p_rep, config)
    p_rep.add_argument(
        'out_dir', nargs='?', default=None,
        help=(
            'Folder name under _save/. If omitted, uses local time YYYYMMDD_HHMMSS '
            'and git-commits all current changes with that message.'
        ),
    )
    p_rep.add_argument('--out', dest='out_dir_flag', default=None)
    p_rep.add_argument('--ensemble', action='store_true', default=False,
                       help='Export for all TelePred delays')
    p_rep.add_argument(
        '--no-commit', action='store_true', default=False,
        help='With bare report (no DIR), skip the auto git commit',
    )

    # search (hparam scout; formerly "sweep")
    p_sw = sub.add_parser(
        'search', aliases=['sweep'],
        help='Hyperparameter search (formerly sweep)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_shared(p_sw, config)
    p_sw.add_argument('--epochs-per-trial', type=int, default=1)
    p_sw.add_argument('--search-extend', '--sweep-extend', dest='sweep_extend',
                      type=float, default=1.0)
    p_sw.add_argument('--with-delays', action='store_true', default=False,
                      help='Outer hparam grid × TelePred delay bank')

    # tools
    p_tools = sub.add_parser('tools', help='Rare utilities',
                             formatter_class=argparse.RawDescriptionHelpFormatter)
    tsub = p_tools.add_subparsers(dest='tools_cmd', required=True)
    p_cmp = tsub.add_parser('compare', help='VAE vs yield loss compare')
    _add_shared(p_cmp, config)
    p_cmp.add_argument('--vae-key', required=True)
    p_cmp.add_argument('--yield-key', required=True)
    p_cmp.add_argument('--slice', dest='val_slice', type=str, default=None)
    p_tr = tsub.add_parser('viz-trend', help='Visualize trend labels')
    _add_shared(p_tr, config)
    p_ft = tsub.add_parser('viz-features', help='Visualize features')
    _add_shared(p_ft, config)
    p_yw = tsub.add_parser('yield-windows', help='Validate-yield window plots')
    _add_shared(p_yw, config)
    p_yw.add_argument('--n', type=int, default=None, metavar='N')

    return parser


def _to_train_namespace(ns: argparse.Namespace, config: dict) -> SimpleNamespace:
    """Map simplified parse result onto TRAIN.main flat-flag namespace."""
    args = _blank_args()
    # copy shared
    for name in (
        'model_type', 'label', 'yield_target', 'latent_distribution',
        'epochs', 'extend', 'clean', 'telepred_delay', 'telepred_delays',
        'checkpoint_key', 'past_window', 'future_window', 'batch_size',
        'set_pairs', 'save_every', 'no_validate',
    ):
        if hasattr(ns, name) and getattr(ns, name) is not None:
            setattr(args, name, getattr(ns, name))
        elif hasattr(ns, name) and name in ('clean', 'no_validate'):
            setattr(args, name, bool(getattr(ns, name)))

    if getattr(ns, 'clean', False):
        args.clean = True
    # Only fill CLI defaults that TRAIN.main requires as non-None numbers.
    # Do NOT stamp the rest of config onto args — that made apply_cli_namespace
    # rewrite signature fields every run and could force label-cache misses
    # even when the user only changed epochs.
    if getattr(ns, 'epochs', None) is None:
        args.epochs = config.get('epochs')
    if getattr(ns, 'save_every', None) is None:
        args.save_every = config.get('save_every_n_minutes')
    if args.model_type is None:
        args.model_type = config.get('model_type', 'regression')
    if args.label is None:
        args.label = config.get('label', 'trend')
    # yield_target / latent_distribution / telepred_delay: leave None unless
    # user passed them; config.py values stay as loaded by get_config().
    if args.telepred_delay is None and getattr(ns, 'telepred_delay', None) is not None:
        args.telepred_delay = int(ns.telepred_delay)

    cmd = getattr(ns, 'cli_command', None)
    args.cli_command = cmd

    if cmd in ('infer', 'test'):
        args.test = getattr(ns, 'test', -1)
        args.no_viz = bool(getattr(ns, 'no_viz', False))
    elif cmd == 'validate':
        sl = _parse_slice(getattr(ns, 'val_slice', None))
        if getattr(ns, 'validate_mode', 'best') == 'latest':
            args.last_validate = sl if sl is not None else []
            args.validate = None
        else:
            args.validate = sl if sl is not None else []
            args.last_validate = None
    elif cmd == 'ensemble':
        args.ensemble = True
    elif cmd == 'error-ensemble':
        args.error_ensemble = True
        if getattr(ns, 'extend_stages', None) is not None:
            args.error_ensemble_extend = int(ns.extend_stages)
        if getattr(ns, 'stages', None) is not None:
            # stash via set_pairs for apply
            pairs = list(args.set_pairs or [])
            pairs.append(f"error_ensemble_n={int(ns.stages)}")
            args.set_pairs = pairs
    elif cmd == 'plot':
        args.history = getattr(ns, 'history', None)
        args.loss = getattr(ns, 'loss', None)
        args.validation = getattr(ns, 'validation', None)
        if args.history is None and args.loss is None and args.validation is None:
            raise SystemExit(
                "plot requires one of --history / --loss / --validation"
            )
    elif cmd == 'report':
        out = getattr(ns, 'out_dir_flag', None) or getattr(ns, 'out_dir', None)
        auto_name = not out
        if auto_name:
            from datetime import datetime
            out = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Bare report: export + commit with message == folder name
            # (same convention as error_ensemble_report.py).
            args.report_git_commit = not bool(getattr(ns, 'no_commit', False))
        else:
            args.report_git_commit = False
        if getattr(ns, 'ensemble', False):
            args.ensemble_report = out
        else:
            args.report = out
    elif cmd in ('search', 'sweep'):
        if getattr(ns, 'with_delays', False):
            args.ensemble_sweep = True
        else:
            args.sweep = True
        args.sweep_epochs = int(getattr(ns, 'epochs_per_trial', 1) or 1)
        args.sweep_extend = float(getattr(ns, 'sweep_extend', 1.0) or 1.0)
        if str(args.model_type).lower() == 'vae':
            args.vae_sweep = True
    elif cmd == 'tools':
        tcmd = getattr(ns, 'tools_cmd', None)
        if tcmd == 'compare':
            args.vae_compare = _parse_slice(getattr(ns, 'val_slice', None))
            if args.vae_compare is None:
                args.vae_compare = []
            args.vae_key = ns.vae_key
            args.yield_key = ns.yield_key
        elif tcmd == 'viz-trend':
            args.trend = True
        elif tcmd == 'viz-features':
            args.features = True
        elif tcmd == 'yield-windows':
            n = getattr(ns, 'n', None)
            args.validate_yield = -1 if n is None else int(n)
        else:
            raise SystemExit(f"unknown tools command {tcmd!r}")
    elif cmd == 'train':
        if getattr(ns, 'no_validate', False):
            args.no_validate = True
    else:
        raise SystemExit(f"unknown command {cmd!r}")

    # Architecture / splits / loss weights / invert-flags stay None unless
    # the user passed them — config.py remains the source of truth.
    return args


def wants_legacy_parser(argv: Optional[Sequence[str]] = None) -> bool:
    args = list(sys.argv[1:] if argv is None else argv)
    if os.environ.get('TRAIN_CLI_LEGACY', '').strip() in ('1', 'true', 'yes'):
        return True
    if not args:
        return False
    # Explicit new command
    if args[0] in COMMANDS or args[0] in (
        'plot', 'tools', 'error-ensemble', 'infer', 'test', 'validate',
        'ensemble', 'sweep', 'search', 'report', 'train',
    ):
        return False
    if args[0] in ('-h', '--help'):
        return False
    # Starts with flag → if any legacy-only hint flag, use legacy
    if args[0].startswith('-'):
        joined = set(args)
        if any(f in joined for f in _LEGACY_HINT_FLAGS):
            return True
        # Plain shared-style flags without subcommand → treat as new `train`
        return False
    return True  # unknown positional → legacy attempt / error later


_VERB_SET = frozenset(COMMANDS) | frozenset({
    'plot', 'tools', 'error-ensemble', 'infer', 'test', 'validate',
    'ensemble', 'sweep', 'search', 'report', 'train',
})


def _unstick_stacked_verbs(raw: List[str]) -> List[str]:
    """
    Fix footguns like ``TRAIN.py train report``.

    Verbs are mutually exclusive; a second verb after the first is almost
    always a mistake. Drop the first and keep the rest (with a one-line hint).
    """
    if len(raw) < 2:
        return raw
    first = str(raw[0]).strip().lower()
    second = str(raw[1]).strip().lower()
    if first in _VERB_SET and second in _VERB_SET:
        kept = list(raw[1:])
        print(
            f"⚠ Two commands given ({first} {second}). "
            f"Using: {' '.join(kept)}"
        )
        print(f"   Tip: run  python TRAIN.py {second} …")
        return kept
    return raw


def try_parse_simplified(
    config: dict,
    argv: Optional[Sequence[str]] = None,
) -> Optional[SimpleNamespace]:
    """
    Parse simplified CLI. Returns Namespace for TRAIN.main, or None → legacy.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    if wants_legacy_parser(raw):
        return None

    # Default verb = train when flags-only invocation
    if raw and raw[0].startswith('-') and raw[0] not in ('-h', '--help'):
        raw = ['train'] + raw
    if not raw or raw[0] in ('-h', '--help'):
        # Show simplified help
        build_simplified_parser(config).print_help()
        raise SystemExit(0)

    raw = _unstick_stacked_verbs(raw)

    parser = build_simplified_parser(config)
    ns = parser.parse_args(raw)
    return _to_train_namespace(ns, config)


def apply_set_pairs(cfg: dict, set_pairs: Optional[List[str]]) -> None:
    """Apply repeated --set KEY=VAL into cfg."""
    if not set_pairs:
        return
    from config_sections import apply_overrides
    overrides = {}
    for item in set_pairs:
        if '=' not in str(item):
            raise SystemExit(f"--set expects KEY=VAL, got {item!r}")
        k, v = str(item).split('=', 1)
        k = k.strip()
        v = v.strip()
        # light typing
        if v.lower() in ('true', 'false'):
            typed: Any = v.lower() == 'true'
        else:
            try:
                if '.' in v:
                    typed = float(v)
                else:
                    typed = int(v)
            except ValueError:
                typed = v
        overrides[k] = typed
    apply_overrides(cfg, overrides)


def warn_legacy_once() -> None:
    print(
        "⚠ Legacy flat-flag CLI detected. Prefer:\n"
        "    python TRAIN.py train|infer|validate|ensemble|error-ensemble|plot|report|search|tools …\n"
        "  Cheat-sheet: python TRAIN.py --help\n"
        "  Force legacy: TRAIN_CLI_LEGACY=1"
    )
