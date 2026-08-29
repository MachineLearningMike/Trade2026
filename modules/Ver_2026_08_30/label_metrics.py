# label_metrics.py
# LabelSpec test/infer metrics + sample prints.
# Prefers Label_*.compute_measures / measure_slots (every label defines slots).

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from labels import (
    LABEL_NONE,
    Label_Latent,
    compute_label_measures,
    decode_latent_params,
    get_label_class,
    label_measure_slots,
    latent_params_to_obs_space,
    resolve_latent_distribution,
    resolve_yield_target,
)


def report_test_metrics(
    label_mode: str,
    preds: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    config: Optional[dict] = None,
    dataset=None,
    yield_target: Optional[str] = None,
    latent_distribution: Optional[str] = None,
    print_samples: bool = True,
    n_samples: int = 5,
) -> Dict[str, Any]:
    """
    Compute + print test metrics via the active label's measure slots.

    Returns a metrics dict.
    """
    cfg = dict(config or {})
    cls = get_label_class(label_mode)
    mode = cls.name
    if yield_target is not None:
        cfg['yield_target'] = yield_target
    elif dataset is not None and getattr(dataset, 'yield_target', None):
        cfg['yield_target'] = dataset.yield_target
    if latent_distribution is not None:
        cfg['latent_distribution'] = latent_distribution
    elif dataset is not None and getattr(dataset, 'latent_distribution', None):
        cfg['latent_distribution'] = dataset.latent_distribution

    yt = resolve_yield_target(cfg)
    ld = resolve_latent_distribution(cfg)
    aux = cls.make_auxiliary(cfg)

    print("\n📊 Test Set Results:")
    print(f"  Measure slots:   {list(label_measure_slots(mode))}")

    # Measures are packed by the hosting label (hidden from models).
    packed = compute_label_measures(
        mode,
        preds,
        test_labels,
        auxiliary=aux,
        dataset=dataset,
        config=cfg,
    )
    from labels import unpack_label_measures
    metrics = unpack_label_measures(mode, packed)
    # Keep any extra diagnostic keys the label packed (e.g. note).
    for k, v in packed.items():
        metrics.setdefault(k, v)
    _print_measures(mode, metrics)

    if print_samples and mode != LABEL_NONE:
        print(f"\n📈 Sample Predictions (first {n_samples}):")
        _print_samples(
            mode, preds, test_labels,
            dataset=dataset, yield_target=yt, latent_dist=ld, n=n_samples,
        )
    return metrics


def _print_measures(mode: str, metrics: dict) -> None:
    if mode == 'rise_flag':
        print(f"  BCE (logits):    {metrics.get('bce_with_logits', float('nan')):.4f}")
        print(f"  Accuracy @thr:   {metrics.get('accuracy', float('nan')):.4f}"
              f"  (thr={metrics.get('decision_threshold', 0.5)})")
        print(
            f"  Confusion:       TP={metrics.get('true_positive', 0)}  "
            f"TN={metrics.get('true_negative', 0)}  "
            f"FP={metrics.get('false_positive', 0)}  "
            f"FN={metrics.get('false_negative', 0)}"
        )
        print(
            f"  Rates:           TPR={metrics.get('tpr', float('nan')):.4f}  "
            f"TNR={metrics.get('tnr', float('nan')):.4f}  "
            f"FPR={metrics.get('fpr', float('nan')):.4f}  "
            f"FNR={metrics.get('fnr', float('nan')):.4f}"
        )
        print(f"  Precision:       {metrics.get('precision', float('nan')):.4f}")
        print(f"  Mean P(rise):    {metrics.get('mean_prob', float('nan')):.4f}")
        print(f"  True pos rate:   {metrics.get('positive_rate_true', float('nan')):.4f}")
        print(f"  Pred pos rate:   {metrics.get('positive_rate_pred', float('nan')):.4f}")
        return
    if mode == 'latent':
        print(f"  NLL:             {metrics.get('nll', float('nan')):.4f}")
        if 'location_mae' in metrics:
            print(f"  Location MAE:    {metrics['location_mae']:.4f}")
            print(f"  Location RMSE:   {metrics['location_rmse']:.4f}")
        if 'mean_pred_std' in metrics:
            print(f"  Mean pred std:   {metrics['mean_pred_std']:.4f}")
        if 'mean_pred_df' in metrics:
            print(f"  Mean pred df:    {metrics['mean_pred_df']:.2f}")
        return
    if mode == 'yield':
        if 'chance_mse' in metrics:
            print(f"  Chance MSE:      {metrics['chance_mse']:.4f}")
        if 'near_mse' in metrics:
            print(f"  Near MSE:        {metrics['near_mse']:.4f}")
        return
    if mode == 'trend':
        if 'horizon_mse' in metrics:
            print(f"  Horizon MSE:     {metrics['horizon_mse']:.4f}")
        if 'slope_mse' in metrics:
            print(f"  Slope MSE:       {metrics['slope_mse']:.4f}")
        if 'intercept_mse' in metrics:
            print(f"  Intercept MSE:   {metrics['intercept_mse']:.4f}")
        return
    if mode == 'none':
        print("  (label=none — unsupervised; skipping supervised metrics)")
        return
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def _print_samples(mode, preds, test_labels, *, dataset, yield_target, latent_dist, n: int):
    n = min(int(n), len(preds))
    if mode == 'latent':
        loc_n, scale_n, df_t = decode_latent_params(preds, latent_distribution=latent_dist)
        loc_obs, scale_obs = latent_params_to_obs_space(
            loc_n, scale_n,
            dataset._label_stat_mean, dataset._label_stat_std,
        )
        y_obs = dataset.denormalize_y(test_labels.detach().cpu()).reshape(-1)
        loc_obs = loc_obs.reshape(-1).to(y_obs.device)
        scale_obs = scale_obs.reshape(-1).to(y_obs.device)
        if df_t is None:
            print(f"  {'Index':<6} {'y future[0]':<12} {'μ (obs)':<12} {'σ (obs)':<12}")
            print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12}")
            for i in range(n):
                print(
                    f"  {i:<6} {y_obs[i].item():<+12.4f} "
                    f"{loc_obs[i].item():<+12.4f} {scale_obs[i].item():<12.4f}"
                )
        else:
            df_flat = df_t.reshape(-1)
            print(
                f"  {'Index':<6} {'y future[0]':<12} {'Pred loc':<12} "
                f"{'Pred scale':<12} {'Pred df':<10}"
            )
            print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
            for i in range(n):
                print(
                    f"  {i:<6} {y_obs[i].item():<+12.4f} "
                    f"{loc_obs[i].item():<+12.4f} {scale_obs[i].item():<12.4f} "
                    f"{df_flat[i].item():<10.2f}"
                )
        return

    if mode == 'yield' and yield_target == 'chance':
        print(f"  {'Index':<6} {'True Chance':<12} {'Pred Chance':<12}")
        print(f"  {'-'*6} {'-'*12} {'-'*12}")
        for i in range(n):
            print(
                f"  {i:<6} {test_labels[i, 0].item():<12.4f} {preds[i, 0].item():<12.4f}"
            )
        return
    if mode == 'yield' and yield_target == 'near':
        print(f"  {'Index':<6} {'True Near':<12} {'Pred Near':<12}")
        print(f"  {'-'*6} {'-'*12} {'-'*12}")
        for i in range(n):
            print(
                f"  {i:<6} {test_labels[i, 0].item():<12.4f} {preds[i, 0].item():<12.4f}"
            )
        return
    if mode == 'yield':
        print(
            f"  {'Index':<6} {'True Chance':<12} {'Pred Chance':<12} "
            f"{'True Near':<12} {'Pred Near':<12}"
        )
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for i in range(n):
            print(
                f"  {i:<6} {test_labels[i, 0].item():<12.4f} {preds[i, 0].item():<12.4f} "
                f"{test_labels[i, 1].item():<12.4f} {preds[i, 1].item():<12.4f}"
            )
        return

    if mode == 'rise_flag':
        print(
            f"  {'Index':<6} {'True':<8} {'Logit':<12} {'P(rise)':<12} {'Pred@0.5':<10}"
        )
        print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*12} {'-'*10}")
        for i in range(n):
            logit = preds[i, 0].item()
            prob = float(torch.sigmoid(preds[i, 0]).item())
            pred_bin = 1 if prob >= 0.5 else 0
            print(
                f"  {i:<6} {test_labels[i, 0].item():<8.0f} "
                f"{logit:<+12.4f} {prob:<12.4f} {pred_bin:<10d}"
            )
        return

    out_dim = preds.shape[-1]
    if out_dim == 3:
        print(
            f"  {'Index':<6} {'True H':<8} {'Pred H':<8} {'True Slope':<12} {'Pred Slope':<12} "
            f"{'True Intercept':<14} {'Pred Intercept':<14}"
        )
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*14} {'-'*14}")
        for i in range(n):
            print(
                f"  {i:<6} {test_labels[i, 0].item():<8.0f} {preds[i, 0].item():<8.0f} "
                f"{test_labels[i, 1].item():<12.4f} {preds[i, 1].item():<12.4f} "
                f"{test_labels[i, 2].item():<14.2f} {preds[i, 2].item():<14.2f}"
            )
    else:
        print(
            f"  {'Index':<6} {'True Slope':<12} {'Pred Slope':<12} "
            f"{'True Intercept':<14} {'Pred Intercept':<14}"
        )
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*14} {'-'*14}")
        for i in range(n):
            print(
                f"  {i:<6} {test_labels[i, 0].item():<12.4f} {preds[i, 0].item():<12.4f} "
                f"{test_labels[i, 1].item():<14.2f} {preds[i, 1].item():<14.2f}"
            )
