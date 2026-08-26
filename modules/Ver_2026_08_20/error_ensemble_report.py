# error_ensemble_report.py
# Build a training report for a finished ErrorEnsemble run.
# Usage:
#   python error_ensemble_report.py --report-name TIME_STRING
# Writes <project_root>/_save/<TIME_STRING>/ with:
#   - summary.json / summary.md
#   - training_history.png (all stages overlay) + per-stage history
#   - model_i/validation_eXXX.png (+ _p0_30.png)  [same as TRAIN --report]

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import get_config, project_root_dir
from labels import sync_output_dim
from data import generate_dataset_signature
from trainer import (
    LatestEpochLoss,
    TrainingHistory,
    ValidationEpochLoss,
    generate_checkpoint_key,
    list_validation_loss_epochs,
)


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


def build_report(report_name: str, stage_keys: Optional[List[str]] = None) -> Path:
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
    }

    # Combined train/val history overlay
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
            # TrainingHistory.plot always calls plt.show(); Agg makes it a no-op.
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

        # Partial / interrupted runs: save latest-epoch minibatch curve.
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

        # Same artifacts as TRAIN --report
        stats = _export_validation_epoch_plots(ckpt_dir, stage_dir)
        stage_info["validation_export"] = stats
        summary["stages"].append(stage_info)
        print(
            f"model_{stage} key={key}: history_epochs={stage_info['history_epochs']} "
            f"val_plots={stats['n_full']}+{stats['n_slice']} finished={stage_info['training_finished']}"
        )

    if any_hist:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("ErrorEnsemble — Training vs Validation Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        overlay_path = report_dir / "training_history.png"
        fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)

    # summary.json
    with open(report_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # summary.md
    n_finished = sum(1 for s in summary["stages"] if s.get("training_finished"))
    lines = [
        f"# ErrorEnsemble training report",
        f"",
        f"- Report: `{report_name}`",
        f"- Created: {summary['created_at']}",
        f"- Dataset signature: `{sig}`",
        f"- Label / model: `{summary['label']}` / `{summary['model_type']}`",
        f"- TelePred delay: {summary['telepred_delay']}",
        f"- Stages: {n} (finished: {n_finished}/{n})",
        f"- Planned epochs/stage: {summary['epochs_planned']}",
        f"",
        f"## Stages",
        f"",
    ]
    summary["stages_finished"] = n_finished
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
    lines.append("## Artifacts")
    lines.append("- `training_history.png` — overlay of all stages (if any epoch finished)")
    lines.append("- `model_i/training_history.png` — per-stage train vs val")
    lines.append("- `model_i/latest_epoch_loss.png` — partial/interrupted epoch minibatch curve")
    lines.append("- `model_i/validation_eXXX.png` — full val batch curves (`--report`)")
    lines.append("- `model_i/validation_eXXX_p0_30.png` — [0%, 30%] slice (`--report`)")
    with open(report_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report written → {report_dir}")
    return report_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-name",
        required=True,
        help="Folder name under <project_root>/_save/",
    )
    parser.add_argument(
        "--stage-keys",
        nargs="*",
        default=None,
        help="Optional explicit checkpoint keys (model_0 .. model_n-1)",
    )
    args = parser.parse_args()
    build_report(args.report_name, stage_keys=args.stage_keys)


if __name__ == "__main__":
    main()
