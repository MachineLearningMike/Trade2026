# error_ensemble_report.py
# Build a training report for a finished (or partial) ErrorEnsemble run.
# Usage:
#   python error_ensemble_report.py --report-name TIME_STRING
# Writes <project_root>/_save/<TIME_STRING>/ with:
#   - summary.json / summary.md
#   - report.text + model_i/report.text + en_k/report.text
#       (label metric slots via Label_* hooks; report stays label-agnostic)
#   - training_history.png (all stages overlay) + per-stage history
#   - model_i/validation_eXXX.png (+ _p0_30.png)  [same as TRAIN --report]
#   - ensemble_performance_vs_stage.png  (compare en_0..en_m on train/val/test)
#   - ensemble_residual_vs_stage.png
# where en_k = model_0 + ... + model_k  (flat y-space sum; see logic txt)

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import get_config, project_root_dir
from task_base import bind_label_loss, sync_output_dim
from data import generate_dataset_signature
from trainer import (
    LatestEpochLoss,
    TrainingHistory,
    ValidationEpochLoss,
    generate_checkpoint_key,
    list_validation_loss_epochs,
)


def _run_git(args: List[str], *, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def commit_code_with_report_name(report_name: str) -> Optional[str]:
    """
    ``git commit`` **all** current working-tree changes with message identical
    to ``report_name`` (the ``_save/<report_name>/`` folder).

    Uses ``git add -A`` (honors ``.gitignore``). Returns the short commit hash,
    or None if there was nothing to commit. Does not push.
    """
    root = project_root_dir()
    name = str(report_name).strip()
    if not name:
        raise ValueError("report_name must be non-empty for git commit message")
    if any(c in name for c in ("\n", "\r", '"')):
        raise ValueError(f"refusing unsafe commit message: {name!r}")

    # Stage every change in the repo (tracked mods/deletes + untracked),
    # excluding paths ignored by .gitignore.
    add = _run_git(["add", "-A"], cwd=root)
    if add.returncode != 0:
        raise RuntimeError(
            f"git add -A failed: {(add.stderr or add.stdout).strip()}"
        )

    staged = _run_git(["diff", "--cached", "--name-only"], cwd=root)
    staged_files = [
        line.strip()
        for line in (staged.stdout or "").splitlines()
        if line.strip()
    ]
    if not staged_files:
        print(f"✓ Git: nothing to commit for report-name {name!r}")
        return None

    # Commit text == report-name.
    commit = _run_git(["commit", "-m", name], cwd=root)
    if commit.returncode != 0:
        raise RuntimeError(
            f"git commit failed: {(commit.stderr or commit.stdout).strip()}"
        )

    head = _run_git(["rev-parse", "--short", "HEAD"], cwd=root)
    sha = (head.stdout or "").strip() or None
    print(f"✓ Git commit {sha}: {name}  ({len(staged_files)} path(s))")
    for p in staged_files:
        print(f"    {p}")
    return sha


def _export_validation_epoch_plots(source_dir: Path, report_dir: Path) -> dict:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    epochs = list_validation_loss_epochs(source_dir)
    if not epochs:
        return {
            "epochs": [],
            "n_full": 0,
            "n_slice": 0,
            "n_missing": 0,
            "max_epoch": 0,
            "epoch_width": 3,
        }

    max_epoch = max(epochs)
    epoch_width = max(3, len(str(max_epoch)))
    n_full = 0
    n_slice = 0
    n_missing = 0
    for e in range(1, max_epoch + 1):
        ve = ValidationEpochLoss.load(source_dir, e)
        if ve is None or not ve.batch_losses:
            n_missing += 1
            continue
        e_tag = f"{e:0{epoch_width}d}"
        full_path = report_dir / f"validation_e{e_tag}.png"
        fig = ve.plot(
            save_path=str(full_path),
            percent_lo=0.0,
            percent_hi=100.0,
            show=False,
        )
        if fig is not None:
            n_full += 1
            plt.close(fig)
        slice_path = report_dir / f"validation_e{e_tag}_p0_30.png"
        fig30 = ve.plot(
            save_path=str(slice_path),
            percent_lo=0.0,
            percent_hi=30.0,
            show=False,
        )
        if fig30 is not None:
            n_slice += 1
            plt.close(fig30)
    return {
        "epochs": epochs,
        "n_full": n_full,
        "n_slice": n_slice,
        "n_missing": n_missing,
        "max_epoch": max_epoch,
        "epoch_width": epoch_width,
    }


def _load_history(ckpt_dir: Path) -> Optional[TrainingHistory]:
    state_file = ckpt_dir / "training_state.json"
    if not state_file.exists():
        return None
    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
    hist = state.get("history") or {}
    if not hist.get("epochs"):
        return None
    th = TrainingHistory()
    th.from_dict(hist)
    return th


def _plot_ensemble_vs_stage(
    train_metrics: Optional[List[dict]],
    val_metrics: Optional[List[dict]],
    report_dir: Path,
    *,
    label_mode: str = "trend",
) -> Dict[str, Optional[str]]:
    """
    Write plots that compare en_0, en_1, ..., en_m.

    en_k = model_0 + ... + model_k
    """
    from task_base import label_ensemble_compare_ylabel

    artifacts: Dict[str, Optional[str]] = {
        "ensemble_performance_vs_stage": None,
        "ensemble_residual_vs_stage": None,
    }
    if not train_metrics and not val_metrics:
        return artifacts

    def _ks(rows):
        return [int(r.get("k", int(r["n_stages"]) - 1)) for r in rows]

    def _ys(rows, key):
        return [float(r[key]) for r in rows]

    def _tick_labels(ks):
        return [f"en_{k}" for k in ks]

    ks_all = []
    if train_metrics:
        ks_all.extend(_ks(train_metrics))
    if val_metrics:
        ks_all.extend(_ks(val_metrics))
    ks_sorted = sorted(set(ks_all))

    # Loss: compare en_0 .. en_m
    fig, ax = plt.subplots(figsize=(10, 6))
    if train_metrics:
        ax.plot(
            _ks(train_metrics),
            _ys(train_metrics, "loss"),
            "o-",
            linewidth=2,
            markersize=8,
            label="Train loss(en_k, y)",
        )
    if val_metrics:
        ax.plot(
            _ks(val_metrics),
            _ys(val_metrics, "loss"),
            "s--",
            linewidth=2,
            markersize=8,
            label="Val loss(en_k, y)",
        )
    ax.set_xlabel("Ensemble en_k = model_0(D)+...+model_k(D)")
    ax.set_ylabel(label_ensemble_compare_ylabel(label_mode))
    ax.set_title(
        "Compare ensembles en_0 .. en_m\n"
        "(additive combine; not the same as per-stage training loss)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    if ks_sorted:
        ax.set_xticks(ks_sorted)
        ax.set_xticklabels(_tick_labels(ks_sorted))
    fig.tight_layout()
    loss_path = report_dir / "ensemble_performance_vs_stage.png"
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    artifacts["ensemble_performance_vs_stage"] = loss_path.name

    # Residual after each en_k
    fig, ax = plt.subplots(figsize=(10, 6))
    if train_metrics:
        ax.plot(
            _ks(train_metrics),
            _ys(train_metrics, "residual_mse"),
            "o-",
            linewidth=2,
            label="Train residual MSE",
        )
        ax.plot(
            _ks(train_metrics),
            _ys(train_metrics, "residual_l1"),
            "o:",
            linewidth=1.5,
            alpha=0.85,
            label="Train residual L1",
        )
    if val_metrics:
        ax.plot(
            _ks(val_metrics),
            _ys(val_metrics, "residual_mse"),
            "s--",
            linewidth=2,
            label="Val residual MSE",
        )
        ax.plot(
            _ks(val_metrics),
            _ys(val_metrics, "residual_l1"),
            "s:",
            linewidth=1.5,
            alpha=0.85,
            label="Val residual L1",
        )
    ax.set_xlabel("Ensemble en_k = model_0(D)+...+model_k(D)")
    ax.set_ylabel("MSE / L1 of (y − en_k)  [not BCE; not member train loss]")
    ax.set_title("Residual after each ensemble en_k")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if ks_sorted:
        ax.set_xticks(ks_sorted)
        ax.set_xticklabels(_tick_labels(ks_sorted))
    fig.tight_layout()
    resid_path = report_dir / "ensemble_residual_vs_stage.png"
    fig.savefig(resid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    artifacts["ensemble_residual_vs_stage"] = resid_path.name
    return artifacts


def _evaluate_loaded_ensemble(
    cfg: dict,
    dataset,
    stage_keys: List[str],
    *,
    max_batches: Optional[int] = None,
    require_finished: bool = False,
    attach_label_metrics: bool = True,
) -> Dict:
    """
    Score en_k = model_0(D)+...+model_k(D) on train + val + test.

    Loads **one stage at a time** (never keeps several models resident).
    When ``attach_label_metrics`` is True, each en_k row includes the
    hosting label's metric slots (label-defined).
    """
    import torch
    from data import ThreadedDataLoader
    from error_ensemble import (
        evaluate_ensemble_over_stages,
        label_norm_from_dataset,
    )

    device = torch.device(cfg["device"] if cfg.get("device") else "cpu")
    label_norm = label_norm_from_dataset(dataset)

    loss_fn = bind_label_loss(
        getattr(dataset, "label_mode", cfg.get("label", "yield")),
        cfg,
        yield_target=getattr(dataset, "yield_target", cfg.get("yield_target", "both")),
        latent_distribution=getattr(
            dataset,
            "latent_distribution",
            cfg.get("latent_distribution", "normal"),
        ),
    )

    bs_train = int(cfg.get("batch_size", 32))
    bs_val = int(cfg.get("validate_batch_size", bs_train))
    bs_test = int(cfg.get("test_batch_size", bs_val))

    print(
        "Wholesale eval: en_k = model_0(D) + model_1(D) + ... "
        "(one full pass per model, one model resident at a time)"
    )

    train_loader = ThreadedDataLoader(
        dataset.train_dataset, batch_size=bs_train, shuffle=False
    )
    val_loader = ThreadedDataLoader(
        dataset.val_dataset, batch_size=bs_val, shuffle=False
    )
    test_loader = ThreadedDataLoader(
        dataset.test_dataset, batch_size=bs_test, shuffle=False
    )

    common = dict(
        config=cfg,
        dataset=dataset,
        stage_keys=stage_keys,
        label_norm=label_norm,
        device=device,
        max_batches=max_batches,
        require_finished=require_finished,
        attach_label_metrics=attach_label_metrics,
    )

    print("  train split...")
    train_metrics = evaluate_ensemble_over_stages(
        loss_fn, train_loader, **common
    )
    print("  val split...")
    val_metrics = evaluate_ensemble_over_stages(
        loss_fn, val_loader, **common
    )
    print("  test split...")
    test_metrics = evaluate_ensemble_over_stages(
        loss_fn, test_loader, **common
    )

    stages_loaded = [
        int(r["k"])
        for r in (val_metrics or test_metrics or train_metrics or [])
    ]
    keys_loaded = [
        stage_keys[s] for s in stages_loaded if 0 <= s < len(stage_keys)
    ]
    return {
        "n_loaded": len(stages_loaded),
        "keys_loaded": keys_loaded,
        "stages_loaded": stages_loaded,
        "load_mode": "sequential_one_model_at_a_time",
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }


def _write_label_metric_reports(
    cfg: dict,
    dataset,
    report_dir: Path,
    *,
    report_name: str,
    stage_keys: List[str],
    summary: Dict,
    ckpt_root: Path,
) -> None:
    """
    Write label metric ``report.text`` for each member and each en_k.

    Members: best_val weights, val+test via INFER streaming.
    Ensembles: reuse metrics attached during wholesale en_k eval (val+test).
    """
    from task_base import (
        evaluate_checkpoint_key_metrics,
        format_metric_lines,
        metrics_to_jsonable,
        write_metrics_report_text,
    )
    from task_base import (
        get_task_class,
        label_decision_threshold,
        label_format_summary_metric_cells,
        label_metric_slots,
        label_metrics_report_note,
        label_summary_metric_headers,
    )

    label_mode = get_task_class(
        getattr(dataset, "label_mode", cfg.get("label", "trend"))
    ).name
    model_type = str(cfg.get("model_type", "regression"))
    slots = list(label_metric_slots(label_mode))
    summary["metric_slots"] = slots
    summary["label_metrics_note"] = label_metrics_report_note(label_mode, cfg)
    summary_headers = label_summary_metric_headers(label_mode)
    summary["summary_metric_headers"] = list(summary_headers)

    member_rows: List[Dict] = []
    print("\n📊 Per-member label metrics (best_val, val+test)...")
    for stage, key in enumerate(stage_keys):
        stage_dir = report_dir / f"model_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        try:
            m = evaluate_checkpoint_key_metrics(
                cfg, dataset, key, checkpoint_dir=ckpt_root,
            )
        except Exception as exc:
            print(f"  ⚠ model_{stage} metrics failed: {exc}")
            member_rows.append(
                {
                    "stage": stage,
                    "model": f"model_{stage}",
                    "checkpoint_key": key,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        write_metrics_report_text(
            stage_dir,
            report_name=f"{report_name}/model_{stage}",
            label_mode=label_mode,
            model_type=model_type,
            checkpoint_key=str(m.get("checkpoint_key", key)),
            checkpoint_path=str(m.get("checkpoint_path", "")),
            val_metrics=m.get("val_metrics"),
            test_metrics=m.get("test_metrics"),
            val_n=m.get("val_n"),
            test_n=m.get("test_n"),
            extra_header={
                "Entity": f"model_{stage} (member best_val)",
                "Dataset signature": getattr(dataset, "cache_key", None),
            },
        )
        row = {
            "stage": stage,
            "model": f"model_{stage}",
            "checkpoint_key": m.get("checkpoint_key", key),
            "checkpoint_path": m.get("checkpoint_path"),
            "val_n": m.get("val_n"),
            "test_n": m.get("test_n"),
            "val_metrics": metrics_to_jsonable(m.get("val_metrics")),
            "test_metrics": metrics_to_jsonable(m.get("test_metrics")),
        }
        member_rows.append(row)
        # Attach onto existing stage summary entry when present.
        for s in summary.get("stages") or []:
            if int(s.get("stage", -1)) == stage:
                s["label_metrics"] = {
                    "val": row["val_metrics"],
                    "test": row["test_metrics"],
                    "val_n": row["val_n"],
                    "test_n": row["test_n"],
                    "report_text": f"model_{stage}/report.text",
                }
                break
        vm = m.get("val_metrics") or {}
        cells = label_format_summary_metric_cells(label_mode, vm)
        print(
            f"  model_{stage}: val "
            + " ".join(f"{h}={c}" for h, c in zip(summary_headers, cells))
        )

    summary["member_label_metrics"] = member_rows

    ens = summary.get("ensemble_over_stages") or {}
    en_rows: List[Dict] = []
    print("\n📊 Per-ensemble en_k label metrics (from wholesale val/test)...")
    val_by_k = {
        int(r.get("k", int(r["n_stages"]) - 1)): r
        for r in (ens.get("val") or [])
    }
    test_by_k = {
        int(r.get("k", int(r["n_stages"]) - 1)): r
        for r in (ens.get("test") or [])
    }
    for k in sorted(set(val_by_k) | set(test_by_k)):
        en_dir = report_dir / f"en_{k}"
        en_dir.mkdir(parents=True, exist_ok=True)
        va = val_by_k.get(k) or {}
        te = test_by_k.get(k) or {}
        va_m = va.get("label_metrics")
        te_m = te.get("label_metrics")
        write_metrics_report_text(
            en_dir,
            report_name=f"{report_name}/en_{k}",
            label_mode=label_mode,
            model_type=model_type,
            checkpoint_key=",".join(
                str(x) for x in (ens.get("keys_loaded") or stage_keys)[: k + 1]
            ),
            checkpoint_path=f"ensemble en_{k} (flat y-space sum)",
            val_metrics=va_m,
            test_metrics=te_m,
            val_n=va.get("n_examples"),
            test_n=te.get("n_examples"),
            extra_header={
                "Entity": f"en_{k}",
                "Definition": va.get("definition") or te.get("definition"),
                "Note": summary.get("label_metrics_note"),
            },
        )
        en_rows.append(
            {
                "k": k,
                "ensemble": f"en_{k}",
                "definition": va.get("definition") or te.get("definition"),
                "val_n": va.get("n_examples"),
                "test_n": te.get("n_examples"),
                "val_metrics": metrics_to_jsonable(va_m),
                "test_metrics": metrics_to_jsonable(te_m),
                "report_text": f"en_{k}/report.text",
            }
        )
        if va_m:
            cells = label_format_summary_metric_cells(label_mode, va_m)
            print(
                f"  en_{k}: val "
                + " ".join(f"{h}={c}" for h, c in zip(summary_headers, cells))
            )

    summary["ensemble_label_metrics"] = en_rows

    # Overview report.text at report root.
    overview_lines = [
        f"Report:           {report_name}",
        f"Label:            {label_mode}",
        f"Model:            {model_type}",
        f"Metric slots:    {slots}",
        f"Note:             {summary.get('label_metrics_note')}",
        "",
        "=== Members (best_val) ===",
        "",
    ]
    for row in member_rows:
        overview_lines.append(f"--- {row.get('model')} key={row.get('checkpoint_key')} ---")
        if row.get("error"):
            overview_lines.append(f"  error: {row['error']}")
            overview_lines.append("")
            continue
        overview_lines.append(f"  val n={row.get('val_n')}")
        overview_lines.extend(
            format_metric_lines(label_mode, row.get("val_metrics") or {})
        )
        overview_lines.append(f"  test n={row.get('test_n')}")
        overview_lines.extend(
            format_metric_lines(label_mode, row.get("test_metrics") or {})
        )
        overview_lines.append("")
    overview_lines.append("=== Ensembles en_k (flat combine) ===")
    overview_lines.append("")
    for row in en_rows:
        overview_lines.append(
            f"--- {row.get('ensemble')} | {row.get('definition')} ---"
        )
        overview_lines.append(f"  val n={row.get('val_n')}")
        overview_lines.extend(
            format_metric_lines(label_mode, row.get("val_metrics") or {})
        )
        overview_lines.append(f"  test n={row.get('test_n')}")
        overview_lines.extend(
            format_metric_lines(label_mode, row.get("test_metrics") or {})
        )
        overview_lines.append("")
    (report_dir / "report.text").write_text(
        "\n".join(overview_lines).rstrip() + "\n", encoding="utf-8"
    )
    print(f"📄 Overview label metrics → {report_dir / 'report.text'}")

    # Label-owned member vs ensemble metric plot (2×n series on one axes).
    from task_base import plot_member_vs_ensemble_metrics

    member_by_stage = []
    for stage in range(len(stage_keys)):
        row = next(
            (r for r in member_rows if int(r.get("stage", -1)) == stage),
            None,
        )
        member_by_stage.append((row or {}).get("val_metrics") or {})
    ensemble_by_stage = []
    for stage in range(len(stage_keys)):
        row = next(
            (r for r in en_rows if int(r.get("k", -1)) == stage),
            None,
        )
        ensemble_by_stage.append((row or {}).get("val_metrics") or {})

    plot_path = plot_member_vs_ensemble_metrics(
        label_mode,
        member_by_stage,
        ensemble_by_stage,
        save_path=report_dir / "label_metrics_member_vs_ensemble.png",
        decision_threshold=label_decision_threshold(label_mode, cfg),
        split_name="val",
    )
    if plot_path is not None:
        summary["label_metrics_plot"] = plot_path.name
        print(f"📈 Member vs ensemble metrics → {plot_path}")


def _discover_stage_keys_from_sidecars(
    ckpt_root: Path,
    n: int,
) -> Optional[List[str]]:
    """
    Find model_0.. prefix from error_stage_norm.json ``stage`` fields.

    Used when config-derived checkpoint keys do not match an existing train run
    (dataset signature / hparam drift).
    """
    from error_ensemble import STAGE_NORM_FILENAME, STAGE_NORM_VERSION

    found: Dict[int, tuple] = {}  # stage -> (mtime, key)
    if not ckpt_root.is_dir():
        return None
    for sub in ckpt_root.iterdir():
        if not sub.is_dir():
            continue
        norm_path = sub / STAGE_NORM_FILENAME
        if not norm_path.exists():
            continue
        try:
            with open(norm_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ver = int(data.get("version", 1) or 1)
            stage = int(data.get("stage", -1))
        except Exception:
            continue
        if ver < STAGE_NORM_VERSION or stage < 0 or stage >= n:
            continue
        mtime = norm_path.stat().st_mtime
        prev = found.get(stage)
        if prev is None or mtime >= prev[0]:
            found[stage] = (mtime, sub.name)
    keys: List[str] = []
    for i in range(n):
        if i not in found:
            break
        keys.append(found[i][1])
    return keys or None


def build_report(
    report_name: str,
    stage_keys: Optional[List[str]] = None,
    *,
    eval_ensemble: bool = True,
    max_eval_batches: Optional[int] = None,
    require_finished: bool = False,
    dataset=None,
    total_features=None,
    git_commit: bool = True,
) -> Path:
    from error_ensemble import (
        STAGE_RUN_CONFIG_FILENAME,
        load_error_ensemble_run_config,
        error_ensemble_stage_keys,
    )

    cfg = get_config()
    sync_output_dim(cfg)
    n = int(cfg.get("error_ensemble_n", 3) or 3)
    ckpt_root = Path(cfg["checkpoint_dir"])
    if not ckpt_root.is_absolute():
        ckpt_root = project_root_dir() / ckpt_root

    key_source = "cli"
    if stage_keys is None:
        # First try config-derived keys (cheap, no dataset load).
        prelim_sig = generate_dataset_signature(cfg)
        prelim_keys = []
        for stage in range(n):
            c = dict(cfg)
            c["error_ensemble_stage"] = stage
            prelim_keys.append(generate_checkpoint_key(c, prelim_sig))
        if all((ckpt_root / k).exists() for k in prelim_keys):
            stage_keys = prelim_keys
            key_source = "config+signature"
        else:
            discovered = _discover_stage_keys_from_sidecars(ckpt_root, n)
            if discovered:
                print(
                    "⚠ Config-derived checkpoint keys not found on disk:\n"
                    f"    {prelim_keys}\n"
                    "  Falling back to ErrorEnsemble sidecars on disk:\n"
                    f"    {discovered}"
                )
                stage_keys = discovered
                key_source = "discovered_sidecars"
            else:
                stage_keys = prelim_keys
                key_source = "config+signature_missing"
                print(
                    "⚠ No ErrorEnsemble checkpoint keys found.\n"
                    f"  Looked for: {prelim_keys}\n"
                    f"  under: {ckpt_root}"
                )

    # Restore train-time config from stage-0 snapshot when present so dataset
    # windows / label / arch match the weights.
    if stage_keys:
        run_cfg = load_error_ensemble_run_config(ckpt_root / stage_keys[0])
        if run_cfg:
            print(
                f"Restoring train config from "
                f"{stage_keys[0]}/{STAGE_RUN_CONFIG_FILENAME}"
            )
            for k, v in run_cfg.items():
                if v is not None and k != "error_ensemble_logic":
                    cfg[k] = v
            sync_output_dim(cfg)

    # Prefer dataset.cache_key (same identity train_one_run uses) when available.
    if dataset is None and (eval_ensemble or key_source.startswith("discovered")):
        from dataset_build import get_dataset

        print("Loading dataset for checkpoint-key identity + ensemble eval...")
        dataset, total_features = get_dataset(cfg)

    if dataset is not None and hasattr(dataset, "cache_key"):
        sig = str(dataset.cache_key)
    else:
        sig = generate_dataset_signature(cfg)

    # If we only had prelim config keys, recompute with dataset.cache_key.
    if key_source == "config+signature" and dataset is not None:
        stage_keys = error_ensemble_stage_keys(cfg, n=n, dataset_signature=sig)
        key_source = "dataset.cache_key"

    report_dir = project_root_dir() / "_save" / report_name
    report_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict = {
        "report_name": report_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_signature": sig,
        "error_ensemble_n": n,
        "label": cfg.get("label"),
        "model_type": cfg.get("model_type"),
        "telepred_delay": cfg.get("telepred_delay"),
        "epochs_planned": cfg.get("epochs"),
        "stage_keys": list(stage_keys),
        "stage_key_source": key_source,
        "stages": [],
        "ensemble_over_stages": None,
    }
    print(f"Report stage keys ({key_source}): {stage_keys}")

    # Combined train/val history overlay (per-stage training curves)
    fig, ax = plt.subplots(figsize=(12, 6))
    any_hist = False

    for stage, key in enumerate(stage_keys):
        stage_info = {
            "stage": stage,
            "model": f"model_{stage}",
            "checkpoint_key": key,
            "checkpoint_dir": str(ckpt_root / key),
            "training_finished": False,
            "history_epochs": 0,
            "final_train_loss": None,
            "final_val_loss": None,
            "best_val_loss": None,
            "validation_export": None,
        }
        ckpt_dir = ckpt_root / key
        stage_dir = report_dir / f"model_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        if not ckpt_dir.exists():
            stage_info["error"] = "checkpoint dir missing"
            summary["stages"].append(stage_info)
            continue

        state_file = ckpt_dir / "training_state.json"
        state = {}
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        stage_info["training_finished"] = bool(state.get("training_finished"))
        stage_info["best_val_loss"] = state.get("best_val_loss")

        th = _load_history(ckpt_dir)
        if th is not None and th.epochs:
            any_hist = True
            stage_info["history_epochs"] = len(th.epochs)
            stage_info["final_train_loss"] = (
                th.train_losses[-1] if th.train_losses else None
            )
            vals = [v for v in th.val_losses if v is not None]
            stage_info["final_val_loss"] = vals[-1] if vals else None
            hist_path = stage_dir / "training_history.png"
            th.plot(save_path=str(hist_path), show=False)
            plt.close("all")
            ax.plot(
                th.epochs,
                th.train_losses,
                "-",
                linewidth=2,
                label=f"model_{stage} train",
            )
            val_epochs = [
                th.epochs[i] for i, v in enumerate(th.val_losses) if v is not None
            ]
            val_losses = [v for v in th.val_losses if v is not None]
            if val_losses:
                ax.plot(
                    val_epochs,
                    val_losses,
                    "--",
                    linewidth=2,
                    label=f"model_{stage} val",
                )

        le = LatestEpochLoss.load(ckpt_dir)
        if le is not None and le.batch_losses:
            stage_info["latest_epoch"] = {
                "epoch": le.epoch,
                "num_batches": len(le.batch_losses),
                "avg_loss": le.avg_loss,
                "complete": bool(le.complete),
            }
            le_path = stage_dir / "latest_epoch_loss.png"
            fig_le = le.plot(save_path=str(le_path))
            if fig_le is not None:
                plt.close(fig_le)
            else:
                plt.close("all")

        stats = _export_validation_epoch_plots(ckpt_dir, stage_dir)
        stage_info["validation_export"] = stats
        summary["stages"].append(stage_info)
        print(
            f"model_{stage} key={key}: history_epochs={stage_info['history_epochs']} "
            f"val_plots={stats['n_full']}+{stats['n_slice']} "
            f"finished={stage_info['training_finished']}"
        )

    if any_hist:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Stage loss (member vs its own residual target)")
        ax.set_title(
            "Per-stage training history "
            "(each model fits its residual target — not en_k loss)"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        overlay_path = report_dir / "training_history.png"
        fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)

    # ------------------------------------------------------------------
    # Cumulative ensemble performance vs number of stages
    # ------------------------------------------------------------------
    ensemble_artifacts = {
        "ensemble_performance_vs_stage": None,
        "ensemble_residual_vs_stage": None,
    }
    if eval_ensemble:
        try:
            if dataset is None:
                from dataset_build import get_dataset

                print("Loading dataset for ensemble-over-stages evaluation...")
                dataset, total_features = get_dataset(cfg)
            ens = _evaluate_loaded_ensemble(
                cfg,
                dataset,
                stage_keys,
                max_batches=max_eval_batches,
                require_finished=require_finished,
                attach_label_metrics=True,
            )
            summary["ensemble_over_stages"] = ens
            _label_for_plot = str(
                getattr(dataset, "label_mode", cfg.get("label", "trend"))
            )
            ensemble_artifacts = _plot_ensemble_vs_stage(
                ens.get("train") or [],
                ens.get("val") or [],
                report_dir,
                label_mode=_label_for_plot,
            )
            if ens.get("train") or ens.get("val") or ens.get("test"):
                from task_base import (
                    label_format_summary_metric_cells,
                    label_summary_metric_headers,
                )
                _hdrs = label_summary_metric_headers(_label_for_plot)
                print("Compare ensembles en_k = sum_{i=0..k} model_i:")
                for split in ("train", "val", "test"):
                    rows = ens.get(split) or []
                    for r in rows:
                        name = r.get("ensemble", f"en_{r.get('k', '?')}")
                        lm = r.get("label_metrics") or {}
                        cells = label_format_summary_metric_cells(
                            _label_for_plot, lm
                        )
                        extra = "  " + " ".join(
                            f"{h}={c}" for h, c in zip(_hdrs, cells)
                        )
                        print(
                            f"  {split} {name}: "
                            f"loss={r['loss']:.6f}  "
                            f"resid_mse={r['residual_mse']:.6f}  "
                            f"resid_l1={r['residual_l1']:.6f}"
                            f"{extra}  "
                            f"[{r.get('definition', '')}]"
                        )
        except Exception as exc:
            summary["ensemble_over_stages"] = {
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"⚠ Ensemble-over-stages eval failed: {exc}")

    # ------------------------------------------------------------------
    # Label metrics: per-member best_val + per en_k (val/test) → report.text
    # ------------------------------------------------------------------
    try:
        if dataset is None:
            from dataset_build import get_dataset

            print("Loading dataset for label metrics...")
            dataset, total_features = get_dataset(cfg)
        _write_label_metric_reports(
            cfg,
            dataset,
            report_dir,
            report_name=report_name,
            stage_keys=stage_keys,
            summary=summary,
            ckpt_root=ckpt_root,
        )
    except Exception as exc:
        summary["label_metrics_error"] = f"{type(exc).__name__}: {exc}"
        print(f"⚠ Label metrics export failed: {exc}")

    # summary.json
    with open(report_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # summary.md
    n_finished = sum(1 for s in summary["stages"] if s.get("training_finished"))
    summary["stages_finished"] = n_finished
    lines = [
        "# ErrorEnsemble training report",
        "",
        f"- Report: `{report_name}`",
        f"- Created: {summary['created_at']}",
        f"- Dataset signature: `{sig}`",
        f"- Label / model: `{summary['label']}` / `{summary['model_type']}`",
        f"- TelePred delay: {summary['telepred_delay']}",
        f"- Stages: {n} (finished: {n_finished}/{n})",
        f"- Planned epochs/stage: {summary['epochs_planned']}",
        "",
        "## Compare ensembles en_0 .. en_m",
        "",
        "After training `model_0 .. model_m`, the report scores these ensembles",
        "in **y-normalized** space (same space as dataloader `y`):",
        "",
        "- Wholesale over dataset D:",
        "  `en_0 = model_0(D)`",
        "  `en_k = model_0(D) + denorm_0(model_1(D)) + ... + denorm_{k-1}(model_k(D))`",
        "",
        "For each i: load only model_i → full pass over D → add in y-space → free → next.",
        "Never loads several models at once; never sums inside one shared batch loop.",
        "",
        "Metrics compare `en_k` to `y` (normalized), not `y_raw`.",
        "Invariant (Yield/MSE labels): loss(en_i, y) should not grow with i.",
        "",
    ]
    from task_base import label_ensemble_logic_markdown_notes
    lines.extend(label_ensemble_logic_markdown_notes(str(summary.get("label", "trend"))))
    ens = summary.get("ensemble_over_stages") or {}
    if ens.get("error") and not ens.get("train") and not ens.get("val"):
        lines.append(f"- Eval error: {ens['error']}")
        lines.append("")
    elif ens.get("train") or ens.get("val") or ens.get("test"):
        lines.append(
            f"- Loaded models: {ens.get('stages_loaded')} "
            f"(keys: {ens.get('keys_loaded')})"
        )
        lines.append("")
        lines.append(
            "| ensemble | definition | train loss | val loss | test loss | "
            "val resid MSE |"
        )
        lines.append(
            "|----------|------------|------------|----------|-----------|"
            "--------------|"
        )

        def _row_key(r: dict) -> int:
            if "k" in r:
                return int(r["k"])
            return int(r["n_stages"]) - 1

        train_by_k = {_row_key(r): r for r in (ens.get("train") or [])}
        val_by_k = {_row_key(r): r for r in (ens.get("val") or [])}
        test_by_k = {_row_key(r): r for r in (ens.get("test") or [])}
        ks = sorted(set(train_by_k) | set(val_by_k) | set(test_by_k))
        for k in ks:
            tr = train_by_k.get(k) or {}
            va = val_by_k.get(k) or {}
            te = test_by_k.get(k) or {}
            name = (
                tr.get("ensemble")
                or va.get("ensemble")
                or te.get("ensemble")
                or f"en_{k}"
            )
            definition = (
                tr.get("definition")
                or va.get("definition")
                or te.get("definition")
                or " + ".join(f"model_{i}" for i in range(k + 1))
            )
            lines.append(
                f"| `{name}` | `{definition}` | "
                f"{tr.get('loss', float('nan')):.6f} | "
                f"{va.get('loss', float('nan')):.6f} | "
                f"{te.get('loss', float('nan')):.6f} | "
                f"{va.get('residual_mse', float('nan')):.6f} |"
            )
        lines.append("")
        if ensemble_artifacts.get("ensemble_performance_vs_stage"):
            lines.append(
                f"- Plot: `{ensemble_artifacts['ensemble_performance_vs_stage']}`"
            )
        if ensemble_artifacts.get("ensemble_residual_vs_stage"):
            lines.append(
                f"- Residual plot: `{ensemble_artifacts['ensemble_residual_vs_stage']}`"
            )
        lines.append("")
    else:
        lines.append("- No ensemble comparison (no loadable models, or eval skipped).")
        lines.append("")

    # Label metric tables — column set owned by Label_*.summary_metric_headers
    from task_base import (
        label_format_summary_metric_cells,
        label_summary_metric_headers,
    )
    _lab = str(summary.get("label", "trend"))
    _hdrs = list(
        summary.get("summary_metric_headers")
        or label_summary_metric_headers(_lab)
    )
    _hdr_row = " | ".join(_hdrs) if _hdrs else "metrics"
    _sep_tail = "|".join(["----"] * max(1, len(_hdrs)))

    def _metric_cells(m: Optional[dict]) -> str:
        cells = label_format_summary_metric_cells(_lab, m)
        if not cells:
            return "—"
        return " | ".join(cells)

    lines.extend(["## Label metrics (best_val members + en_k)", ""])
    if summary.get("label_metrics_note"):
        lines.append(f"- Note: {summary['label_metrics_note']}")
        lines.append("")
    if summary.get("label_metrics_error"):
        lines.append(f"- Error: {summary['label_metrics_error']}")
        lines.append("")

    lines.append("### Members")
    lines.append("")
    lines.append(f"| member | split | {_hdr_row} |")
    lines.append(f"|--------|-------|{_sep_tail}|")
    for row in summary.get("member_label_metrics") or []:
        name = row.get("model", "?")
        if row.get("error"):
            lines.append(f"| `{name}` | — | error: {row['error']} |")
            continue
        lines.append(
            f"| `{name}` | val | {_metric_cells(row.get('val_metrics'))} |"
        )
        lines.append(
            f"| `{name}` | test | {_metric_cells(row.get('test_metrics'))} |"
        )
    lines.append("")
    lines.append("### Ensembles en_k")
    lines.append("")
    lines.append(f"| ensemble | split | {_hdr_row} |")
    lines.append(f"|----------|-------|{_sep_tail}|")
    for row in summary.get("ensemble_label_metrics") or []:
        name = row.get("ensemble", "?")
        lines.append(
            f"| `{name}` | val | {_metric_cells(row.get('val_metrics'))} |"
        )
        lines.append(
            f"| `{name}` | test | {_metric_cells(row.get('test_metrics'))} |"
        )
    lines.append("")
    lines.append("- Details: `report.text`, `model_i/report.text`, `en_k/report.text`")
    if summary.get("label_metrics_plot"):
        lines.append(
            f"- Plot: `{summary['label_metrics_plot']}` "
            "(2xn series: member solid / en dashed, shared stage axis)"
        )
    lines.append("")

    lines.extend(
        [
            "## Per-stage training",
            "",
        ]
    )
    for s in summary["stages"]:
        lines.append(f"### model_{s['stage']} (`{s['checkpoint_key']}`)")
        lines.append(f"- Finished: {s.get('training_finished')}")
        lines.append(f"- History epochs: {s.get('history_epochs')}")
        lines.append(f"- Final train loss: {s.get('final_train_loss')}")
        lines.append(f"- Final val loss: {s.get('final_val_loss')}")
        lines.append(f"- Best val loss: {s.get('best_val_loss')}")
        le = s.get("latest_epoch") or {}
        if le:
            lines.append(
                f"- Latest epoch curve: epoch={le.get('epoch')} "
                f"batches={le.get('num_batches')} "
                f"avg={le.get('avg_loss')} complete={le.get('complete')}"
            )
        ve = s.get("validation_export") or {}
        lines.append(
            f"- Validation plots: {ve.get('n_full', 0)} full + "
            f"{ve.get('n_slice', 0)} p0_30 (epochs {ve.get('epochs')})"
        )
        lm = s.get("label_metrics") or {}
        if lm.get("val"):
            lines.append(
                f"- Label metrics (val): {_metric_cells(lm.get('val'))}"
            )
        if s.get("error"):
            lines.append(f"- Error: {s['error']}")
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "- `report.text` - overview label metrics (members + en_k)",
            "- `model_i/report.text` - member best_val label metrics (val+test)",
            "- `en_k/report.text` - cumulative ensemble label metrics (val+test)",
            "- `label_metrics_member_vs_ensemble.png` - label plot: 2xn series "
            "(member solid / en dashed) on shared stage axis",
            "- `ensemble_performance_vs_stage.png` - compare loss(en_k, y) for en_0..en_m",
            "- `ensemble_residual_vs_stage.png` - MSE/L1(y−en_k) for en_0..en_m",
            "- `training_history.png` - overlay of per-stage train/val curves",
            "- `model_i/training_history.png` - per-stage train vs val",
            "- `model_i/latest_epoch_loss.png` - partial/interrupted epoch curve",
            "- `model_i/validation_eXXX.png` - full val batch curves (`--report`)",
            "- `model_i/validation_eXXX_p0_30.png` - [0%, 30%] slice (`--report`)",
        ]
    )
    with open(report_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report written → {report_dir}")

    # Commit message is identical to the report folder name.
    if git_commit:
        try:
            sha = commit_code_with_report_name(report_name)
            summary["git_commit"] = sha
            summary["git_commit_message"] = report_name if sha else None
            with open(report_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            if sha:
                # Append commit line to summary.md for traceability.
                with open(report_dir / "summary.md", "a", encoding="utf-8") as f:
                    f.write(f"\n## Git\n\n- Commit: `{sha}`\n- Message: `{report_name}`\n")
        except Exception as exc:
            print(f"⚠ Git commit skipped/failed: {exc}")

    return report_dir


def default_report_name(now: Optional[datetime] = None) -> str:
    """Timestamp used when ``--report-name`` is omitted: ``YYYYMMDD_HHMMSS``."""
    when = now if now is not None else datetime.now()
    return when.strftime("%Y%m%d_%H%M%S")


def main():
    parser = argparse.ArgumentParser(
        description="Build ErrorEnsemble training + ensemble-over-stages report"
    )
    parser.add_argument(
        "--report-name",
        default=None,
        help=(
            "Folder name under <project_root>/_save/, and git commit message "
            "(unless --no-commit). If omitted, uses local time YYYYMMDD_HHMMSS."
        ),
    )
    parser.add_argument(
        "--stage-keys",
        nargs="*",
        default=None,
        help="Optional explicit checkpoint keys (model_0 .. model_n-1)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip cumulative ensemble train/val evaluation (plots/checkpoints only)",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        metavar="N",
        help="Optional cap on train/val batches for a quicker ensemble eval",
    )
    parser.add_argument(
        "--require-finished",
        action="store_true",
        help="Only include stages with training_finished=true in ensemble eval",
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help=(
            "Do not git-commit (default: git add -A and commit all current "
            "changes with message=report-name)"
        ),
    )
    args = parser.parse_args()
    report_name = args.report_name or default_report_name()
    if args.report_name is None:
        print(f"report-name omitted → using {report_name}")
    build_report(
        report_name,
        stage_keys=args.stage_keys,
        eval_ensemble=not args.skip_eval,
        max_eval_batches=args.max_eval_batches,
        require_finished=args.require_finished,
        git_commit=not args.no_commit,
    )


if __name__ == "__main__":
    main()
