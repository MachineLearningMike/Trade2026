# cli_commands.py
# Phase 5 subcommand front-end for TRAIN.py (flat flags remain supported).
#
#   python TRAIN.py train ...
#   python TRAIN.py infer [--test N] ...
#   python TRAIN.py validate ...
#   python TRAIN.py ensemble ...
#   python TRAIN.py search ...
#   python TRAIN.py error-ensemble ...
#   python TRAIN.py report ['dir name'] ...
#
# Unknown first token → legacy flat-flag argparse in TRAIN.main.

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

COMMANDS = (
    'train',
    'infer',
    'test',
    'validate',
    'ensemble',
    'search',
    'sweep',  # deprecated alias → search
    'error-ensemble',
    'report',
    'plot',
    'tools',
)


def split_command_argv(argv: Optional[Sequence[str]] = None) -> Tuple[Optional[str], List[str]]:
    """
    If argv[0] is a known command, return (command, rest).
    Otherwise return (None, full argv) for legacy flat parsing.
    """
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return None, []
    cmd = str(args[0]).strip().lower()
    if cmd in COMMANDS:
        return cmd, args[1:]
    return None, args


def apply_command_to_namespace(cmd: str, ns) -> None:
    """
    Map a subcommand onto the flat-flag namespace used by TRAIN.main.

    Mutates ``ns`` in place (argparse.Namespace).
    """
    c = str(cmd).strip().lower()
    if c == 'train':
        return
    if c in ('infer', 'test'):
        # Bare infer/test → all samples (TRAIN uses const=-1 for bare --test)
        if getattr(ns, 'test', None) is None:
            ns.test = -1
        return
    if c == 'validate':
        if getattr(ns, 'validate', None) is None:
            ns.validate = []
        return
    if c == 'ensemble':
        ns.ensemble = True
        return
    if c in ('search', 'sweep'):
        ns.sweep = True
        return
    if c == 'error-ensemble':
        ns.error_ensemble = True
        return
    if c == 'report':
        # report name may already be in ns.report from subparser
        return
    raise ValueError(f'unknown command {cmd!r}')


def command_help_epilog() -> str:
    return (
        'Subcommands (optional; flat flags still work):\n'
        '  train              single-element training (default)\n'
        '  infer | test       test-set eval (same as --test)\n'
        '  validate           validation curves (same as --validate)\n'
        '  ensemble           TelePred delay bank (--ensemble)\n'
        '  search             hparam search (alias: sweep)\n'
        '  error-ensemble     residual stages (--error-ensemble)\n'
        '  report [DIR]       batch-export val+history plots under _save/;\n'
        '                     omit DIR → YYYYMMDD_HHMMSS + git commit\n'
    )
