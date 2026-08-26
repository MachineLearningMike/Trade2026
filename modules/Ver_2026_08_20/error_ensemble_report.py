# error_ensemble_report.py
# Build a training report for a finished (or partial) ErrorEnsemble run.
# Usage:
#   python error_ensemble_report.py --report-name TIME_STRING
# Writes <project_root>/_save/<TIME_STRING>/ with:
#   - summary.json / summary.md
#   - training_history.png (all stages overlay) + per-stage history
#   - model_i/validation_eXXX.png (+ _p0_30.png)  [same as TRAIN --report]
#   - ensemble_performance_vs_stage.png  (compare en_0..en_m on train/val)
#   - ensemble_residual_vs_stage.png
# where en_k = model_0 + ... + model_k

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
from labels import bind_label_loss, sync_output_dim
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
) -> Dict[str, Optional[str]]:
    """
    Write plots that compare en_0, en_1, ..., en_m.

    en_k = model_0 + ... + model_k
    """
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
    ax.set_ylabel("Label loss (y-normalized space)")
    ax.set_title("Compare ensembles en_0 .. en_m")
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
    ax.set_ylabel("Remaining residual y - en_k (y-normalized)")
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
) -> Dict:
    """
    Score en_k = model_0(D)+...+model_k(D) on train + val.

    Loads **one stage at a time** (never keeps several models resident).
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

    print("  train split...")
    train_metrics = evaluate_ensemble_over_stages(
        loss_fn,
        train_loader,
        config=cfg,
        dataset=dataset,
        stage_keys=stage_keys,
        label_norm=label_norm,
        device=device,
        max_batches=max_batches,
        require_finished=require_finished,
    )
    print("  val split...")
    val_metrics = evaluate_ensemble_over_stages(
        loss_fn,
        val_loader,
        config=cfg,
        dataset=dataset,
        stage_keys=stage_keys,
        label_norm=label_norm,
        device=device,
        max_batches=max_batches,
        require_finished=require_finished,
    )

    stages_loaded = [int(r["k"]) for r in (val_metrics or train_metrics or [])]
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
    }


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
    cfg = get_config()
    sync_output_dim(cfg)
    n = int(cfg.get("error_ensemble_n", 3) or 3)
    sig = generate_dataset_signature(cfg)
    ckpt_root = Path(cfg["checkpoint_dir"])
    if not ckpt_root.is_absolute():
        ckpt_root = project_root_dir() / ckpt_root

    if stage_keys is None:
        stage_keys = []
        for stage in range(n):
            c = dict(cfg)
            c["error_ensemble_stage"] = stage
            stage_keys.append(generate_checkpoint_key(c, sig))

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
        "stages": [],
        "ensemble_over_stages": None,
    }

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
            th.plot(save_path=str(hist_path))
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
        ax.set_ylabel("Loss")
        ax.set_title("Per-stage training history (each model fits its residual target)")
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
            )
            summary["ensemble_over_stages"] = ens
            ensemble_artifacts = _plot_ensemble_vs_stage(
                ens.get("train") or [],
                ens.get("val") or [],
                report_dir,
            )
            if ens.get("train") or ens.get("val"):
                print("Compare ensembles en_k = sum_{i=0..k} model_i:")
                for split in ("train", "val"):
                    rows = ens.get(split) or []
                    for r in rows:
                        name = r.get("ensemble", f"en_{r.get('k', '?')}")
                        print(
                            f"  {split} {name}: "
                            f"loss={r['loss']:.6f}  "
                            f"resid_mse={r['residual_mse']:.6f}  "
                            f"resid_l1={r['residual_l1']:.6f}  "
                            f"[{r.get('definition', '')}]"
                        )
        except Exception as exc:
            summary["ensemble_over_stages"] = {
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"⚠ Ensemble-over-stages eval failed: {exc}")

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
        "Invariant: loss(en_i, y) must not grow with i (retrain with --clean if it does).",
        "",
    ]
    ens = summary.get("ensemble_over_stages") or {}
    if ens.get("error") and not ens.get("train") and not ens.get("val"):
        lines.append(f"- Eval error: {ens['error']}")
        lines.append("")
    elif ens.get("train") or ens.get("val"):
        lines.append(
            f"- Loaded models: {ens.get('stages_loaded')} "
            f"(keys: {ens.get('keys_loaded')})"
        )
        lines.append("")
        lines.append(
            "| ensemble | definition | train loss | val loss | "
            "train resid MSE | val resid MSE |"
        )
        lines.append(
            "|----------|------------|------------|----------|"
            "-----------------|---------------|"
        )

        def _row_key(r: dict) -> int:
            if "k" in r:
                return int(r["k"])
            return int(r["n_stages"]) - 1

        train_by_k = {_row_key(r): r for r in (ens.get("train") or [])}
        val_by_k = {_row_key(r): r for r in (ens.get("val") or [])}
        ks = sorted(set(train_by_k) | set(val_by_k))
        for k in ks:
            tr = train_by_k.get(k) or {}
            va = val_by_k.get(k) or {}
            name = tr.get("ensemble") or va.get("ensemble") or f"en_{k}"
            definition = (
                tr.get("definition")
                or va.get("definition")
                or " + ".join(f"model_{i}" for i in range(k + 1))
            )
            lines.append(
                f"| `{name}` | `{definition}` | "
                f"{tr.get('loss', float('nan')):.6f} | "
                f"{va.get('loss', float('nan')):.6f} | "
                f"{tr.get('residual_mse', float('nan')):.6f} | "
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
        if s.get("error"):
            lines.append(f"- Error: {s['error']}")
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "- `ensemble_performance_vs_stage.png` - compare loss(en_k, y) for en_0..en_m",
            "- `ensemble_residual_vs_stage.png` - residual y-en_k for en_0..en_m",
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
