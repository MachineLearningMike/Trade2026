# INFER.py
# Inference script for trained model
#
# Important: never materialize the full test set of windows in RAM.
# With ~430k samples, past_window=10080, F≈56, the test split alone is ~90+ GB
# if stacked. Always stream samples in small batches.

import json
import random
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import get_config
from model_vae import build_market_model, resolve_max_seq_len, wrap_model_loss
from labels import resolve_model_past_len
from labels import visualize_trend, visualize_yield, visualize_latent
from labels import Label_Latent, resolve_latent_distribution
from labels import decode_latent_params, latent_params_to_obs_space
from trainer import generate_checkpoint_key
from dataset_build import get_dataset
from telepred import resolve_past_window
from data import normalize_window


# ═══════════════════════════════════════════════════════════
# CHECKPOINT LOOKUP
# ═══════════════════════════════════════════════════════════

def find_checkpoint_by_key(
    checkpoint_dir: str = 'trading_checkpoints',
    key: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """
    Checkpoint lookup for --test / INFER only (not training resume).

    Prefer best_val.pt when it exists. Otherwise the newest remaining .pt
    (rolling / emergency / finished). Resume uses a different rule: newest
    save time among all files, including best_val.

    Returns:
        Tuple of (checkpoint_path, checkpoint_key) or None
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        print(f"⚠ Checkpoint directory not found: {checkpoint_dir}")
        return None

    if key is not None:
        key_dir = checkpoint_dir / key
        if not key_dir.exists():
            print(f"⚠ Checkpoint key directory not found: {key_dir}")
            return None

        best = key_dir / 'best_val.pt'
        if best.exists():
            print(f"✓ Using best-val checkpoint: {best} (Key: {key})")
            return str(best), key

        state_file = key_dir / 'training_state.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                best_path = state.get('best_checkpoint')
                if best_path and Path(best_path).exists():
                    print(f"✓ Using best-val checkpoint: {best_path} (Key: {key})")
                    return best_path, key
                latest = state.get('latest_checkpoint')
                if latest and Path(latest).exists():
                    return latest, key
            except Exception:
                pass

        checkpoints = list(key_dir.glob('*.pt'))
        if not checkpoints:
            print(f"⚠ No checkpoint files found in {key_dir}")
            return None

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"✓ Found checkpoint: {latest} (Key: {key})")
        return str(latest), key

    bests = []
    all_checkpoints = []
    for subdir in checkpoint_dir.iterdir():
        if subdir.is_dir():
            best = subdir / 'best_val.pt'
            if best.exists():
                bests.append((best, subdir.name))
            for f in subdir.glob('*.pt'):
                all_checkpoints.append((f, subdir.name))

    if bests:
        latest, key = max(bests, key=lambda p: p[0].stat().st_mtime)
        print(f"✓ Using best-val checkpoint: {latest} (Key: {key})")
        return str(latest), key

    if not all_checkpoints:
        print(f"⚠ No checkpoint files found in {checkpoint_dir}")
        return None

    latest, key = max(all_checkpoints, key=lambda p: p[0].stat().st_mtime)
    print(f"✓ Found latest checkpoint: {latest} (Key: {key})")
    return str(latest), key


def find_latest_checkpoint_by_key(
    checkpoint_dir: str = 'trading_checkpoints',
    key: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """
    Checkpoint lookup for --last-validate: newest available weights, not best_val.

    Preference order within a key directory:
      1. training_state.json → latest_checkpoint (if the file still exists)
      2. newest *.pt by mtime (rolling / emergency / finished / best_val)

    When ``key`` is None, picks the newest such file across all key folders.
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        print(f"⚠ Checkpoint directory not found: {checkpoint_dir}")
        return None

    def _latest_in_dir(key_dir: Path, key_name: str) -> Optional[Tuple[str, str]]:
        state_file = key_dir / 'training_state.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                latest = state.get('latest_checkpoint')
                if latest and Path(latest).exists():
                    print(f"✓ Using latest checkpoint: {latest} (Key: {key_name})")
                    return str(latest), key_name
            except Exception:
                pass

        checkpoints = list(key_dir.glob('*.pt'))
        if not checkpoints:
            return None
        newest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"✓ Using newest checkpoint: {newest} (Key: {key_name})")
        return str(newest), key_name

    if key is not None:
        key_dir = checkpoint_dir / key
        if not key_dir.exists():
            print(f"⚠ Checkpoint key directory not found: {key_dir}")
            return None
        result = _latest_in_dir(key_dir, key)
        if result is None:
            print(f"⚠ No checkpoint files found in {key_dir}")
        return result

    candidates = []
    for subdir in checkpoint_dir.iterdir():
        if not subdir.is_dir():
            continue
        result = _latest_in_dir(subdir, subdir.name)
        if result is not None:
            candidates.append((Path(result[0]), result[1]))

    if not candidates:
        print(f"⚠ No checkpoint files found in {checkpoint_dir}")
        return None

    newest_path, newest_key = max(candidates, key=lambda p: p[0].stat().st_mtime)
    print(f"✓ Using newest checkpoint across keys: {newest_path} (Key: {newest_key})")
    return str(newest_path), newest_key


def load_model(
    checkpoint_path: str,
    config: dict,
    dataset,
    device: str = 'cuda',
):
    """
    Load a trained model from a checkpoint.

    Architecture dims are taken from `dataset` + `config` so they match training.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    past_window = int(resolve_past_window(config))
    max_seq_len = checkpoint.get('max_seq_len')
    if max_seq_len is None:
        max_seq_len = resolve_max_seq_len(config, past_window)

    F = dataset.F
    n_max_past = int(
        getattr(dataset, 'model_past_len', None)
        or resolve_model_past_len(config)
    )
    model = build_market_model(
        config,
        in_features=F,
        n_market_features=dataset.n_market_features,
        n_macro_features=dataset.n_macro_features,
        n_max_past_horizon=n_max_past,
        max_seq_len=max_seq_len,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def _collate_batch(dataset, indices: List[int]):
    """Load a small batch of samples from the dataset (CPU tensors)."""
    xs, ps, ys, means, stds, shapes = [], [], [], [], [], []
    for idx in indices:
        sample = dataset[idx]
        if len(sample) == 6:
            x, p, mean, std, shape, y = sample
        else:
            x, y, mean, std, shape = sample
            n_p = int(
                getattr(dataset, 'model_past_len', None)
                or getattr(dataset, 'max_past_horizon', None)
                or 0
            )
            p = x[-n_p:, getattr(dataset, 'price_feature_idx', 0)] if n_p else x.new_zeros(0)
        xs.append(x)
        ps.append(p)
        ys.append(y)
        means.append(mean)
        stds.append(std)
        shapes.append(shape)
    return (
        torch.stack(xs),      # (B, T, F)
        torch.stack(ps),      # (B, max_past_horizon)
        torch.stack(ys),      # (B, target_dim)  1 for latent; output_dim otherwise
        torch.stack(means),   # (B, F)
        torch.stack(stds),    # (B, F)
        torch.stack(shapes),  # (B, num_shapes, F)
    )


def _evenly_spaced_indices(indices: List[int], n: int) -> List[int]:
    """
    Pick n indices covering the full span of `indices` (in the given order).

    Used so --test N and latent viz seeds are spread across the test period
    instead of clustering on the head of a fixed-seed shuffle.
    """
    if n <= 0 or not indices:
        return []
    if n >= len(indices):
        return list(indices)
    if n == 1:
        return [indices[len(indices) // 2]]
    # Inclusive endpoints: first and last of the ordered list.
    out: List[int] = []
    last = len(indices) - 1
    for i in range(n):
        pos = int(round(i * last / (n - 1)))
        out.append(indices[pos])
    # Deduplicate while preserving order (can happen when n is close to len).
    seen = set()
    unique: List[int] = []
    for idx in out:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    # If rounding collapsed slots, fill from unused positions.
    if len(unique) < n:
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)
            if len(unique) >= n:
                break
    return unique


def _select_test_indices(
    dataset,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> List[int]:
    """
    Choose which dataset indices to evaluate on the test split.

    When max_samples is set, take an evenly spaced subset of the *chronological*
    test split so metrics/viz cover early, mid, and late test windows — not
    just whatever lands first after seed=42 shuffle.
    Presentation order may still be shuffled for batching.
    """
    # Chronological order of the test split (valid_starts order when
    # shuffle_split=False; arbitrary but stable otherwise).
    chronological = sorted(dataset.test_indices)
    if not chronological:
        return []

    if max_samples is not None and max_samples < len(chronological):
        selected = _evenly_spaced_indices(chronological, int(max_samples))
        print(
            f"  (using {len(selected)} test samples spaced across the "
            f"{len(chronological)}-window test period)",
            flush=True,
        )
    else:
        selected = list(chronological)
        print(
            f"  (using all {len(selected)} test samples)",
            flush=True,
        )

    if shuffle and len(selected) > 1:
        rng = random.Random(seed)
        rng.shuffle(selected)
    return selected


def run_inference_on_indices(
    model,
    dataset,
    indices: List[int],
    device: str = 'cuda',
    batch_size: int = 8,
    split_name: str = 'samples',
):
    """
    Stream arbitrary dataset indices in batches. Never stacks full windows.

    model.forward expects:
        x:     (B, T, F)
        p:     (B, max_past_horizon) recent past prices
        mean:  (B, F)
        std:   (B, F)
        shape: (B, num_shapes, F)

    Returns:
        preds:  (N, output_dim) on `device`
                trend/yield → denormalized targets
                latent      → raw density-head params (not denormalized)
        labels: (N, target_dim) on `device`
                trend/yield → denormalized targets (target_dim = output_dim)
                latent      → label-normalized scalar observation (target_dim = 1)
        used_indices: list of dataset indices that were evaluated (same order)
    """
    model.eval()

    used_indices = list(indices)
    n = len(used_indices)
    if n == 0:
        raise RuntimeError(f"No {split_name} available.")

    bytes_per_window = dataset.past_window * dataset.F * 4
    full_gb = n * bytes_per_window / (1024 ** 3)
    print(
        f"  {split_name.capitalize()}: {n}  |  "
        f"window≈{bytes_per_window / 1e6:.2f} MB each  "
        f"|  full stack would be ~{full_gb:.1f} GB → streaming batch_size={batch_size}",
        flush=True,
    )

    pred_chunks = []
    label_chunks = []
    n_batches = (n + batch_size - 1) // batch_size

    with torch.no_grad():
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n)
            batch_idx = used_indices[start:end]

            x, p, y, mean, std, shape = _collate_batch(dataset, batch_idx)
            x = x.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            mean = mean.to(device, non_blocking=True)
            std = std.to(device, non_blocking=True)
            shape = shape.to(device, non_blocking=True)

            pred_norm, _, _ = model(x, p, mean, std, shape)  # (B, output_dim)

            if getattr(dataset, 'label_mode', None) == 'latent':
                # Head is density params (mu, var_raw) / (loc, scale_raw, df_raw).
                # Label stats are the scalar observation (dim 1). Do not
                # denormalize preds. Keep y in __getitem__ (normalized) space
                # so test NLL matches training.
                pred_chunks.append(pred_norm.detach().cpu())
                label_chunks.append(y)
            else:
                pred_denorm = torch.stack(
                    [
                        dataset.denormalize_y(pred_norm[i].detach().cpu())
                        for i in range(pred_norm.shape[0])
                    ]
                )
                y_denorm = torch.stack(
                    [dataset.denormalize_y(y[i]) for i in range(y.shape[0])]
                )
                pred_chunks.append(pred_denorm)
                label_chunks.append(y_denorm)

            if b == 0 or (b + 1) % max(1, n_batches // 20) == 0 or (b + 1) == n_batches:
                print(
                    f"  Inference batch {b + 1}/{n_batches}  ({end}/{n} samples)",
                    flush=True,
                )

            del x, p, mean, std, shape, pred_norm

    preds = torch.cat(pred_chunks, dim=0).to(device)
    labels = torch.cat(label_chunks, dim=0).to(device)
    return preds, labels, used_indices


def run_inference_on_test_set(
    model,
    dataset,
    device: str = 'cuda',
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Stream the test split in batches. Never stacks the full test windows.

    Returns:
        preds, labels, used_indices — same contract as ``run_inference_on_indices``.
    """
    test_indices = _select_test_indices(
        dataset,
        max_samples=max_samples,
        shuffle=shuffle,
        seed=seed,
    )
    return run_inference_on_indices(
        model,
        dataset,
        test_indices,
        device=device,
        batch_size=batch_size,
        split_name='test samples',
    )


def _sample_observation_time(dataset, idx: int) -> int:
    """
    Bar index of the latent observation (``future[0]`` = p[t+1]).

    TelePred: bot time ``t`` in ``dataset.valid_t``; observation is ``t + 1``.
    Legacy fallback: ``valid_starts[idx] + past_window``.
    """
    valid_t = getattr(dataset, 'valid_t', None)
    if valid_t is not None:
        return int(valid_t[idx]) + 1
    return int(dataset.valid_starts[idx]) + int(dataset.past_window)


def _item_at_observation_bar(dataset, t_obs: int):
    """
    Build one model sample whose latent observation is bar ``t_obs``.

    History is ``Total[t_obs - past_window : t_obs]`` (ends one bar before
    the observation). Label is window-z of ``Total[t_obs]`` = future[0],
    then dataset label-normalized — same convention as Label_Latent.find
    + data.py latent labeling.
    """
    pw = int(dataset.past_window)
    start = int(t_obs) - pw
    T = int(dataset.T)
    # Need full history and the observation bar itself.
    if start < 0 or t_obs < 0 or t_obs >= T or start + pw > T:
        return None
    x = dataset.Total[start:start + pw]
    price_idx = int(dataset.price_feature_idx)
    if dataset.normalize:
        _, mean, std, shape = normalize_window(x, compute_moments=True)
        std = std.clone()
        std[std == 0] = 1
        x_n = (x - mean) / std
        # future[0] in the same window-z space as the label cache
        obs_win = (dataset.Total[t_obs, price_idx] - mean[price_idx]) / std[price_idx]
    else:
        mean = torch.zeros(dataset.F)
        std = torch.ones(dataset.F)
        shape = torch.zeros(dataset.num_shapes, dataset.F)
        x_n = x
        obs_win = dataset.Total[t_obs, price_idx]
    n_p = int(
        getattr(dataset, 'model_past_len', None)
        or getattr(dataset, 'max_past_horizon', 0)
        or 0
    )
    p = x_n[-n_p:, price_idx] if n_p else x_n.new_zeros(0)
    y = dataset.normalize_y(obs_win.reshape(1).float())
    st = dataset.stats
    mean_n = (mean - st['mean']['mean']) / (st['mean']['std'] + 1e-8)
    std_n = (std - st['std']['mean']) / (st['std']['std'] + 1e-8)
    shape_n = (shape - st['shape']['mean']) / (st['shape']['std'] + 1e-8)
    return x_n, p, mean_n, std_n, shape_n, y, mean, std


def collect_latent_price_series(
    model,
    dataset,
    seed_indices: List[int],
    device: str = 'cuda',
    batch_size: int = 16,
    interval_minutes: int = 120,
    lookback_minutes: int = 60,
    latent_distribution: str = 'normal',
) -> List[dict]:
    """
    For each seed sample, run the model at every observation bar in
    [t0 - lookback, t0 + interval] (not only dataset.step) so μ(t)
    shares P(t)'s 1-minute grid.

    At each t, history ends at t-1 and the density targets P(t)=future[0]
    (Label_Latent.find). t0 is the seed sample's observation bar.
    """
    model.eval()
    label_std = dataset._label_stat_std
    normalize = bool(getattr(dataset, 'normalize_window', True))
    price_idx = int(dataset.price_feature_idx)
    pw = int(dataset.past_window)
    T = int(dataset.T)

    series: List[dict] = []
    label_std_s = float(torch.as_tensor(label_std).reshape(-1)[0].abs().clamp(min=1e-8))
    with torch.no_grad():
        for seed in seed_indices:
            t0 = _sample_observation_time(dataset, seed)
            t_lo = t0 - int(lookback_minutes)
            t_hi = t0 + int(interval_minutes)
            # Observation bar needs pw bars of history before it.
            times = [
                t for t in range(max(pw, t_lo), min(T - 1, t_hi) + 1)
            ]
            if not times:
                continue

            mu_obs_parts = []
            sig_obs_parts = []
            y_obs_parts = []
            t_pred = []
            price_at = []
            y_raw_parts = []
            mu_raw_parts = []
            sig_raw_parts = []
            for b0 in range(0, len(times), batch_size):
                bars = times[b0:b0 + batch_size]
                xs, ps, ys, means, stds, shapes = [], [], [], [], [], []
                raw_means, raw_stds = [], []
                good_bars = []
                for to in bars:
                    item = _item_at_observation_bar(dataset, to)
                    if item is None:
                        continue
                    x, p, mean_n, std_n, shape_n, y, mean_r, std_r = item
                    xs.append(x)
                    ps.append(p)
                    ys.append(y)
                    means.append(mean_n)
                    stds.append(std_n)
                    shapes.append(shape_n)
                    raw_means.append(mean_r)
                    raw_stds.append(std_r)
                    good_bars.append(to)
                if not xs:
                    continue
                x = torch.stack(xs).to(device, non_blocking=True)
                p = torch.stack(ps).to(device, non_blocking=True)
                y = torch.stack(ys)
                mean = torch.stack(means).to(device, non_blocking=True)
                std = torch.stack(stds).to(device, non_blocking=True)
                shape = torch.stack(shapes).to(device, non_blocking=True)
                pred, _, _ = model(x, p, mean, std, shape)
                loc_n, scale_n, _df = decode_latent_params(
                    pred.detach(), latent_distribution=latent_distribution
                )
                loc_n_cpu = loc_n.reshape(-1, 1).detach().cpu()
                scale_n_cpu = scale_n.reshape(-1).detach().cpu()
                loc_obs = dataset.denormalize_y(loc_n_cpu).reshape(-1)
                scale_obs = scale_n_cpu * label_std_s
                y_obs = dataset.denormalize_y(y.reshape(-1, 1)).reshape(-1)

                win_mean = torch.stack(raw_means)[:, price_idx].cpu().float()
                win_std = torch.stack(raw_stds)[:, price_idx].cpu().float()
                win_std = torch.where(win_std == 0, torch.ones_like(win_std), win_std)
                if normalize:
                    loc_raw = loc_obs * win_std + win_mean
                    sig_raw = scale_obs * win_std.abs()
                    y_raw = y_obs * win_std + win_mean
                else:
                    loc_raw = loc_obs
                    sig_raw = scale_obs
                    y_raw = y_obs

                mu_obs_parts.append(loc_obs)
                sig_obs_parts.append(scale_obs)
                y_obs_parts.append(y_obs)
                mu_raw_parts.append(loc_raw)
                sig_raw_parts.append(sig_raw)
                y_raw_parts.append(y_raw)
                for to in good_bars:
                    t_pred.append(to - t0)
                    # P at observation bar = future[0] target
                    price_at.append(float(dataset.Total[to, price_idx].item()))
                del x, p, y, mean, std, shape, pred

            ctx_start = max(0, t_lo)
            ctx_end = min(T, t_hi + 1)
            t_price = np.arange(ctx_start, ctx_end, dtype=np.float64) - float(t0)
            price_curve = dataset.Total[ctx_start:ctx_end, price_idx].detach().cpu().numpy()
            y_raw_np = torch.cat(y_raw_parts).numpy()
            price_at_np = np.asarray(price_at, dtype=np.float64)
            t_pred_np = np.asarray(t_pred, dtype=np.float64)
            # Dense P(t) at the same bars as μ(t): must match Total[observation].
            align_err = []
            for tp, p_at in zip(t_pred_np, price_at_np):
                j = int(round(tp - t_price[0]))
                if 0 <= j < len(price_curve):
                    align_err.append(abs(float(price_curve[j]) - float(p_at)))
            align_mae = float(np.mean(align_err)) if align_err else float('nan')
            recon_mae = float(np.mean(np.abs(y_raw_np - price_at_np)))
            mu_raw_np = torch.cat(mu_raw_parts).numpy()
            lag_mae = {}
            for lag in (-2, -1, 0, 1, 2):
                errs = []
                for tp, mu_v in zip(t_pred_np, mu_raw_np):
                    bar = int(t0 + tp + lag)
                    if 0 <= bar < T:
                        errs.append(abs(float(mu_v) - float(dataset.Total[bar, price_idx])))
                lag_mae[lag] = float(np.mean(errs)) if errs else float('nan')
            series.append({
                'sample_id': seed,
                't_price': t_price,
                'price': np.asarray(price_curve, dtype=np.float64),
                't_pred': t_pred_np,
                'price_at_pred': price_at_np,
                'y_obs': torch.cat(y_obs_parts).numpy(),
                'mu_obs': torch.cat(mu_obs_parts).numpy(),
                'sigma_obs': torch.cat(sig_obs_parts).numpy(),
                'y_raw': y_raw_np,
                'mu_raw': mu_raw_np,
                'sigma_raw': torch.cat(sig_raw_parts).numpy(),
                'recon_mae': recon_mae,
                'align_mae': align_mae,
                'mu_mae_raw': float(np.mean(np.abs(mu_raw_np - price_at_np))),
                'lag_mae': lag_mae,
            })
            print(
                f"  latent series seed={seed}: n={len(times)}  "
                f"P-align MAE={align_mae:.4g}  "
                f"label→price MAE={recon_mae:.4g}  "
                f"|μ-P| MAE={series[-1]['mu_mae_raw']:.4g}  "
                f"lag MAE "
                + " ".join(f"Δ{k}={v:.4g}" for k, v in lag_mae.items()),
                flush=True,
            )
            if align_err and align_mae > 1e-4:
                print(
                    f"  ⚠ μ(t) and P(t) time grids disagree "
                    f"(P-align MAE={align_mae:.4g})",
                    flush=True,
                )
    return series


# ═══════════════════════════════════════════════════════════
# FULL TEST PIPELINE (shared by INFER.py and TRAIN.py --test)
# ═══════════════════════════════════════════════════════════

def run_test(
    config: dict,
    max_samples: Optional[int] = None,
    visualize: bool = True,
):
    """
    Load the dataset + checkpoint matching `config`, run batched test-set
    inference, print metrics, and optionally visualize a few true trends.

    Args:
        config: Full training/inference config (same as TRAIN).
        max_samples: If set, evaluate this many test samples spaced across
                     the chronological test period; otherwise evaluate all.
        visualize: If True, plot labeled trends / latent series after metrics.
                     Latent seeds use config['test_viz_samples'] windows spaced
                     across the evaluated set (not only the first few).

    Batch size always comes from config['test_batch_size']
    (falls back to validate_batch_size, then batch_size).

    Returns:
        dict with preds, labels, used_indices, metrics, checkpoint_path, checkpoint_key
    """
    device = config['device']
    batch_size = config.get(
        'test_batch_size',
        config.get('validate_batch_size', config.get('batch_size', 8)),
    )

    print("\nLoading dataset (same path as TRAIN)...", flush=True)
    dataset, total_features = get_dataset(config)
    # Align head width with resolved yield_target / trend slice / latent density
    config['output_dim'] = dataset.output_dim
    if getattr(dataset, 'label_mode', None) == 'yield':
        config['yield_target'] = getattr(dataset, 'yield_target', 'both')
    if getattr(dataset, 'label_mode', None) == 'latent':
        config['latent_distribution'] = getattr(
            dataset, 'latent_distribution', config.get('latent_distribution', 'normal')
        )

    checkpoint_key = generate_checkpoint_key(config, dataset.cache_key)
    print(f"🔑 Generated Checkpoint Key: {checkpoint_key}", flush=True)

    result = find_checkpoint_by_key(
        config.get('checkpoint_dir', 'trading_checkpoints'), checkpoint_key
    )
    if result is None:
        print(f"⚠ No checkpoint found for key: {checkpoint_key}")
        print("  Looking for latest checkpoint instead...")
        result = find_checkpoint_by_key(
            config.get('checkpoint_dir', 'trading_checkpoints'), None
        )
        if result is None:
            print("No checkpoint found. Exiting.")
            raise SystemExit(1)

    checkpoint_path, checkpoint_key = result

    print(f"\nLoading model from: {checkpoint_path}", flush=True)
    print(f"Checkpoint Key: {checkpoint_key}", flush=True)
    print(f"Test batch size: {batch_size}", flush=True)
    if max_samples is not None:
        print(f"Test sample cap: {max_samples}", flush=True)
    else:
        print("Test sample cap: all", flush=True)

    model = load_model(checkpoint_path, config, dataset, device)
    print("✓ Model loaded successfully", flush=True)

    print("\nRunning batched inference on test samples...", flush=True)
    preds, test_labels, used_indices = run_inference_on_test_set(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        max_samples=max_samples,
        shuffle=True,
    )
    print(f"✓ Finished inference on {len(preds)} test samples", flush=True)

    out_dim = preds.shape[-1]
    label_mode = config.get('label', getattr(dataset, 'label_mode', 'trend'))
    from labels import resolve_yield_target
    yield_target = resolve_yield_target(
        config,
        yield_target=getattr(dataset, 'yield_target', None),
    )
    latent_dist = resolve_latent_distribution(
        config,
        latent_distribution=getattr(dataset, 'latent_distribution', None),
    )
    print("\n📊 Test Set Results:")
    metrics = {}
    if label_mode == 'latent':
        aux = Label_Latent.make_auxiliary(config, latent_distribution=latent_dist)
        metrics['nll'] = Label_Latent.loss(
            preds, preds, test_labels, auxiliary=aux
        ).item()
        loc_n, scale_n, df_t = decode_latent_params(preds, latent_distribution=latent_dist)
        loc_obs, scale_obs = latent_params_to_obs_space(
            loc_n, scale_n,
            dataset._label_stat_mean, dataset._label_stat_std,
        )
        y_obs = dataset.denormalize_y(test_labels.detach().cpu()).reshape(-1)
        loc_obs = loc_obs.reshape(-1).to(y_obs.device)
        scale_obs = scale_obs.reshape(-1).to(y_obs.device)
        metrics['location_mae'] = (loc_obs - y_obs).abs().mean().item()
        metrics['location_rmse'] = (loc_obs - y_obs).pow(2).mean().sqrt().item()
        metrics['mean_pred_std'] = scale_obs.mean().item()
        print(f"  Density:         {latent_dist}")
        print(f"  NLL:             {metrics['nll']:.4f}")
        print(f"  Location MAE:    {metrics['location_mae']:.4f}")
        print(f"  Location RMSE:   {metrics['location_rmse']:.4f}")
        print(f"  Mean pred std:   {metrics['mean_pred_std']:.4f}")
        if df_t is not None:
            metrics['mean_pred_df'] = df_t.reshape(-1).float().mean().item()
            print(f"  Mean pred df:    {metrics['mean_pred_df']:.2f}")
    elif label_mode == 'yield' and yield_target == 'chance':
        metrics['chance_mse'] = torch.nn.functional.mse_loss(
            preds[:, 0], test_labels[:, 0]
        ).item()
        print(f"  Chance MSE:      {metrics['chance_mse']:.4f}")
    elif label_mode == 'yield' and yield_target == 'near':
        metrics['near_mse'] = torch.nn.functional.mse_loss(
            preds[:, 0], test_labels[:, 0]
        ).item()
        print(f"  Near MSE:        {metrics['near_mse']:.4f}")
    elif label_mode == 'yield' and out_dim == 2:
        metrics['chance_mse'] = torch.nn.functional.mse_loss(
            preds[:, 0], test_labels[:, 0]
        ).item()
        metrics['near_mse'] = torch.nn.functional.mse_loss(
            preds[:, 1], test_labels[:, 1]
        ).item()
        print(f"  Chance MSE:      {metrics['chance_mse']:.4f}")
        print(f"  Near MSE:        {metrics['near_mse']:.4f}")
    elif out_dim == 3:
        metrics['horizon_mse'] = torch.nn.functional.mse_loss(
            preds[:, 0], test_labels[:, 0]
        ).item()
        metrics['slope_mse'] = torch.nn.functional.mse_loss(
            preds[:, 1], test_labels[:, 1]
        ).item()
        metrics['intercept_mse'] = torch.nn.functional.mse_loss(
            preds[:, 2], test_labels[:, 2]
        ).item()
        print(f"  Horizon MSE:     {metrics['horizon_mse']:.4f}")
        print(f"  Slope MSE:       {metrics['slope_mse']:.4f}")
        print(f"  Intercept MSE:   {metrics['intercept_mse']:.4f}")
    elif out_dim == 2:
        metrics['slope_mse'] = torch.nn.functional.mse_loss(
            preds[:, 0], test_labels[:, 0]
        ).item()
        metrics['intercept_mse'] = torch.nn.functional.mse_loss(
            preds[:, 1], test_labels[:, 1]
        ).item()
        print(f"  Slope MSE:       {metrics['slope_mse']:.4f}")
        print(f"  Intercept MSE:   {metrics['intercept_mse']:.4f}")
    else:
        raise ValueError(f"Unexpected pred dim {out_dim}")

    print(f"\n📈 Sample Predictions (first 5):")
    if label_mode == 'latent':
        loc_n, scale_n, df_t = decode_latent_params(preds, latent_distribution=latent_dist)
        loc_obs, scale_obs = latent_params_to_obs_space(
            loc_n, scale_n,
            dataset._label_stat_mean, dataset._label_stat_std,
        )
        y_obs = dataset.denormalize_y(test_labels.detach().cpu()).reshape(-1)
        loc_obs = loc_obs.reshape(-1).to(y_obs.device)
        scale_obs = scale_obs.reshape(-1).to(y_obs.device)
        if df_t is None:
            print(
                f"  {'Index':<6} {'y future[0]':<12} {'μ (obs)':<12} {'σ (obs)':<12}"
            )
            print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12}")
            for i in range(min(5, len(preds))):
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
            for i in range(min(5, len(preds))):
                print(
                    f"  {i:<6} {y_obs[i].item():<+12.4f} "
                    f"{loc_obs[i].item():<+12.4f} {scale_obs[i].item():<12.4f} "
                    f"{df_flat[i].item():<10.2f}"
                )
    elif label_mode == 'yield' and yield_target == 'chance':
        print(f"  {'Index':<6} {'True Chance':<12} {'Pred Chance':<12}")
        print(f"  {'-'*6} {'-'*12} {'-'*12}")
        for i in range(min(5, len(preds))):
            print(
                f"  {i:<6} {test_labels[i, 0].item():<12.4f} {preds[i, 0].item():<12.4f}"
            )
    elif label_mode == 'yield' and yield_target == 'near':
        print(f"  {'Index':<6} {'True Near':<12} {'Pred Near':<12}")
        print(f"  {'-'*6} {'-'*12} {'-'*12}")
        for i in range(min(5, len(preds))):
            print(
                f"  {i:<6} {test_labels[i, 0].item():<12.4f} {preds[i, 0].item():<12.4f}"
            )
    elif label_mode == 'yield' and out_dim == 2:
        print(
            f"  {'Index':<6} {'True Chance':<12} {'Pred Chance':<12} "
            f"{'True Near':<12} {'Pred Near':<12}"
        )
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for i in range(min(5, len(preds))):
            print(
                f"  {i:<6} {test_labels[i, 0].item():<12.4f} {preds[i, 0].item():<12.4f} "
                f"{test_labels[i, 1].item():<12.4f} {preds[i, 1].item():<12.4f}"
            )
    elif out_dim == 3:
        print(
            f"  {'Index':<6} {'True H':<8} {'Pred H':<8} {'True Slope':<12} {'Pred Slope':<12} "
            f"{'True Intercept':<14} {'Pred Intercept':<14}"
        )
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*14} {'-'*14}")
        for i in range(min(5, len(preds))):
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
        for i in range(min(5, len(preds))):
            print(
                f"  {i:<6} {test_labels[i, 0].item():<12.4f} {preds[i, 0].item():<12.4f} "
                f"{test_labels[i, 1].item():<14.2f} {preds[i, 1].item():<14.2f}"
            )

    if visualize:
        n_viz = len(used_indices)

        if label_mode == 'latent':
            print(
                f"\n📈 Visualizing latent next-bar density "
                f"({latent_dist}) as μ(t), σ(t) curves "
                f"(observation = future[0])...",
                flush=True,
            )
            interval = int(
                config.get('future_window')
                or getattr(dataset, 'future_window', 50)
            )
            interval = max(interval, 120)
            # Space seeds across the evaluated test period (chronological
            # valid_starts), not the first few after shuffle — otherwise
            # Next only cycles a handful of near-duplicate windows.
            n_want = int(config.get('test_viz_samples', 16))
            n_seeds = min(max(1, n_want), max(1, n_viz))
            chrono_used = sorted(
                used_indices,
                key=lambda i: int(dataset.valid_starts[i]),
            )
            seeds = _evenly_spaced_indices(chrono_used, n_seeds)
            print(
                f"  Latent viz seeds: {len(seeds)} windows spaced across "
                f"{n_viz} evaluated test samples "
                f"(config test_viz_samples={n_want})",
                flush=True,
            )
            latent_series = collect_latent_price_series(
                model,
                dataset,
                seeds,
                device=device,
                batch_size=batch_size,
                interval_minutes=interval,
                lookback_minutes=60,
                latent_distribution=latent_dist,
            )
            visualize_latent(
                latent_series,
                latent_distribution=latent_dist,
                series_per_page=1,
            )
        elif label_mode == 'yield':
            valid_starts = [dataset.valid_starts[i] for i in used_indices]
            full_labels = torch.stack(
                [dataset.label_tensor[i] for i in used_indices]
            )
            pred_cpu = preds.detach().cpu()
            print(
                f"\n📈 Visualizing yield labels "
                f"(yield_target={yield_target}) on test samples...",
                flush=True,
            )
            # Model-aligned denorm labels (sliced) match pred_tensor layout
            viz_labels = test_labels.detach().cpu()
            # For chance×near inset we need full (chance, near) in natural units
            # when yield_target=both; for single modes titles use sliced tensors.
            visualize_yield(
                total_tensor=dataset.Total,
                valid_starts=valid_starts,
                label_tensor=viz_labels if yield_target != 'both' else full_labels,
                price_feature_idx=dataset.price_feature_idx,
                past_window=dataset.past_window,
                future_window=dataset.future_window,
                min_yield_percent=config.get(
                    'min_yield_percent',
                    getattr(dataset, 'min_yield_percent', 0.0),
                ),
                max_hope_future_horizon=config.get(
                    'future_window',
                    getattr(dataset, 'future_window', None),
                ),
                yield_target=yield_target,
                num_samples=min(6, n_viz),
                pred_tensor=pred_cpu,
                fixed_order=True,
                sample_ids=list(used_indices),
                # Train-set label stats so inset is model-normalized (z) space
                label_mean=getattr(dataset, '_label_stat_mean', None),
                label_std=getattr(dataset, '_label_stat_std', None),
            )
        else:
            valid_starts = [dataset.valid_starts[i] for i in used_indices]
            full_labels = torch.stack(
                [dataset.label_tensor[i] for i in used_indices]
            )
            pred_cpu = preds.detach().cpu()
            print(
                "\n📈 Visualizing label vs inferred trends on test samples...",
                flush=True,
            )
            raw_means = torch.stack([dataset.mean_tensor[i] for i in used_indices])
            raw_stds = torch.stack([dataset.std_tensor[i] for i in used_indices])
            pred_horizon = int(
                config.get('future_window',
                    getattr(dataset, 'future_window',
                            getattr(dataset, 'max_horizon', 50)))
            )
            h_past_full = getattr(dataset, 'h_past_tensor', None)
            h_past_viz = None
            if h_past_full is not None:
                h_past_viz = h_past_full[list(used_indices)]
            visualize_trend(
                total_tensor=dataset.Total,
                valid_starts=valid_starts,
                label_tensor=full_labels,
                mean_tensor=raw_means,
                std_tensor=raw_stds,
                price_feature_idx=dataset.price_feature_idx,
                past_window=dataset.past_window,
                future_window=dataset.future_window,
                mode=config.get('horizon_mode', 'anchored'),
                num_samples=min(6, n_viz),
                pred_tensor=pred_cpu,
                fixed_order=True,
                sample_ids=list(used_indices),
                default_pred_horizon=pred_horizon,
                h_past_tensor=h_past_viz,
                min_horizon=config.get(
                    'min_trend_horizon',
                    getattr(dataset, 'min_trend_horizon', getattr(dataset, 'min_horizon', 3)),
                ),
                max_horizon=config.get(
                    'future_window',
                    getattr(dataset, 'future_window', None),
                ),
                max_past_horizon=config.get(
                    'future_window',
                    getattr(dataset, 'future_window', None),
                ),
                fitness_threshold=config.get(
                    'fitness_threshold', getattr(dataset, 'fitness_threshold', 0.75)
                ),
                horizon_search=config.get(
                    'horizon_search', getattr(dataset, 'horizon_search', 'longest_linear')
                ),
                regime_use_past=config.get(
                    'regime_use_past', getattr(dataset, 'regime_use_past', True)
                ),
                normalize_prices=config.get('normalize_window', getattr(dataset, 'normalize_window', True)),
            )

    print(f"\n✓ Inference complete!", flush=True)

    return {
        'preds': preds,
        'labels': test_labels,
        'used_indices': used_indices,
        'metrics': metrics,
        'output_dim': out_dim,
        'checkpoint_path': checkpoint_path,
        'checkpoint_key': checkpoint_key,
    }


# ═══════════════════════════════════════════════════════════
# CHECKPOINT VALIDATION (--validate / --last-validate)
# ═══════════════════════════════════════════════════════════

def run_validate(
    config: dict,
    percent_hi: float = 100.0,
    percent_lo: float = 0.0,
    checkpoint_mode: str = 'best',
    checkpoint_key: Optional[str] = None,
):
    """
    Load a checkpoint for ``config``, evaluate a percent slice of sorted
    validation samples, print mean val loss, and plot the same legend series
    as ``--validation`` (batch / moving avg / running loss). Does not write
    any files.

    Args:
        config: Full training/inference config (same as TRAIN).
        percent_hi: End of the slice in percent of the val split, in
                    (percent_lo, 100].
        percent_lo: Start of the slice in percent, in [0, percent_hi).
                    ``--validate`` / ``--last-validate`` → 0..100
                    ``... V``                            → 0..V
                    ``... V1 V2``                        → V1..V2
        checkpoint_mode:
            ``'best'``   → prefer best_val.pt (``--validate``)
            ``'latest'`` → prefer newest / latest_checkpoint (``--last-validate``)
        checkpoint_key: If set, load from this folder under checkpoint_dir
            instead of hashing the current config (e.g. a prior yield run).

    Samples are taken from ``sorted(dataset.val_indices)[start:end]``.

    Returns:
        dict with avg_loss, n_used, n_val_total, percent_lo, percent_hi,
        epoch, checkpoint_mode, checkpoint_path, checkpoint_key, used_indices
    """
    from labels import bind_label_loss
    from trainer import ValidationEpochLoss

    mode = str(checkpoint_mode or 'best').strip().lower()
    if mode not in ('best', 'latest'):
        raise ValueError(
            f"checkpoint_mode must be 'best' or 'latest', got {checkpoint_mode!r}"
        )
    flag_name = '--validate' if mode == 'best' else '--last-validate'

    percent_lo = float(percent_lo)
    percent_hi = float(percent_hi)
    if not (0.0 <= percent_lo < percent_hi <= 100.0):
        raise ValueError(
            f"{flag_name} requires 0 <= V1 < V2 <= 100 "
            f"(got V1={percent_lo}, V2={percent_hi})"
        )

    device = config['device']
    batch_size = config.get(
        'validate_batch_size',
        config.get('test_batch_size', config.get('batch_size', 8)),
    )

    print("\nLoading dataset (same path as TRAIN)...", flush=True)
    dataset, _total_features = get_dataset(config)
    config['output_dim'] = dataset.output_dim
    if getattr(dataset, 'label_mode', None) == 'yield':
        config['yield_target'] = getattr(dataset, 'yield_target', 'both')
    if getattr(dataset, 'label_mode', None) == 'latent':
        config['latent_distribution'] = getattr(
            dataset, 'latent_distribution', config.get('latent_distribution', 'normal')
        )

    forced_key = (checkpoint_key or config.get('checkpoint_key') or '').strip() or None
    hashed_key = generate_checkpoint_key(config, dataset.cache_key)
    if forced_key:
        checkpoint_key = forced_key
        print(f"🔑 Using forced Checkpoint Key: {checkpoint_key}", flush=True)
        print(f"   (config hash key would be: {hashed_key})", flush=True)
    else:
        checkpoint_key = hashed_key
        print(f"🔑 Generated Checkpoint Key: {checkpoint_key}", flush=True)

    ckpt_dir = config.get('checkpoint_dir', 'trading_checkpoints')
    if mode == 'best':
        result = find_checkpoint_by_key(ckpt_dir, checkpoint_key)
        if result is None and not forced_key:
            print(f"⚠ No checkpoint found for key: {checkpoint_key}")
            print("  Looking for latest best-val checkpoint instead...")
            result = find_checkpoint_by_key(ckpt_dir, None)
    else:
        result = find_latest_checkpoint_by_key(ckpt_dir, checkpoint_key)
        if result is None and not forced_key:
            print(f"⚠ No checkpoint found for key: {checkpoint_key}")
            print("  Looking for newest checkpoint across keys instead...")
            result = find_latest_checkpoint_by_key(ckpt_dir, None)

    if result is None:
        if forced_key:
            print(
                f"No checkpoint found for forced key {forced_key!r} under {ckpt_dir}. "
                "Exiting."
            )
        else:
            print("No checkpoint found. Exiting.")
        raise SystemExit(1)

    checkpoint_path, checkpoint_key = result
    if mode == 'best' and Path(checkpoint_path).name != 'best_val.pt':
        print(
            f"⚠ Preferred best_val.pt was not found; using: {checkpoint_path}",
            flush=True,
        )

    # 1-based epoch for the plot title (same numbering as --validation).
    plot_epoch = None
    state_path = Path(checkpoint_path).parent / 'training_state.json'
    state = None
    if state_path.exists():
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
        except Exception:
            state = None

    if mode == 'best' and state is not None and state.get('best_epoch') is not None:
        plot_epoch = int(state['best_epoch'])
    if plot_epoch is None:
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if ckpt.get('epoch') is not None:
                # Checkpoints store 0-based epoch; history / --validation use 1-based.
                plot_epoch = int(ckpt['epoch']) + 1
        except Exception:
            plot_epoch = None

    val_sorted = sorted(dataset.val_indices)
    n_total = len(val_sorted)
    if n_total == 0:
        raise RuntimeError("No validation samples available.")

    start = int(n_total * percent_lo / 100.0)
    if percent_hi >= 100.0:
        end = n_total
    else:
        end = int(n_total * percent_hi / 100.0)
    start = max(0, min(n_total, start))
    end = max(0, min(n_total, end))
    used_indices = val_sorted[start:end]
    n_take = len(used_indices)
    if n_take == 0:
        raise RuntimeError(
            f"Percent slice [{percent_lo:g}%, {percent_hi:g}%] selected 0 of "
            f"{n_total} validation samples (start={start}, end={end})."
        )

    print(f"\nLoading model from: {checkpoint_path}", flush=True)
    print(f"Checkpoint Key: {checkpoint_key}", flush=True)
    print(f"Checkpoint mode: {mode}", flush=True)
    print(
        f"Val samples: indices [{start}:{end}) → {n_take} / {n_total} "
        f"([{percent_lo:g}%, {percent_hi:g}%] of sorted val split)",
        flush=True,
    )

    model = load_model(checkpoint_path, config, dataset, device)
    print("✓ Model loaded successfully", flush=True)

    _base_loss_fn = bind_label_loss(
        getattr(dataset, 'label_mode', config.get('label', 'trend')),
        config,
        yield_target=getattr(dataset, 'yield_target', config.get('yield_target', 'both')),
        latent_distribution=getattr(
            dataset, 'latent_distribution', config.get('latent_distribution', 'normal')
        ),
    )
    loss_fn = wrap_model_loss(model, _base_loss_fn, config)

    model.eval()
    weighted_loss_sum = 0.0
    n_seen = 0
    n_batches = (n_take + batch_size - 1) // batch_size
    val_curve = ValidationEpochLoss()
    val_curve.reset(epoch=int(plot_epoch) if plot_epoch is not None else 0)
    if plot_epoch is not None:
        label = 'Best' if mode == 'best' else 'Latest'
        print(f"{label} checkpoint epoch: {plot_epoch}", flush=True)

    mode_tag = 'validate' if mode == 'best' else 'last-validate'
    with torch.no_grad():
        pbar = tqdm(
            total=n_batches,
            desc=(
                f"{mode_tag} [{percent_lo:g}%, {percent_hi:g}%] "
                f"({n_take} samples)"
            ),
            mininterval=0.5,
            miniters=1,
            smoothing=0.1,
            dynamic_ncols=True,
            bar_format=(
                '{l_bar}{bar}| {n_fmt}/{total_fmt} '
                '[{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            ),
        )
        for b in range(n_batches):
            batch_start = b * batch_size
            batch_end = min(batch_start + batch_size, n_take)
            batch_idx = used_indices[batch_start:batch_end]
            bs = batch_end - batch_start

            x, p, y, mean, std, shape = _collate_batch(dataset, batch_idx)
            x = x.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mean = mean.to(device, non_blocking=True)
            std = std.to(device, non_blocking=True)
            shape = shape.to(device, non_blocking=True)

            pred = model(x, p, mean, std, shape)
            loss = loss_fn(x, pred, y)
            batch_loss = float(loss.item())
            weighted_loss_sum += batch_loss * bs
            n_seen += bs
            val_curve.add_batch(b + 1, batch_loss)

            avg_so_far = weighted_loss_sum / max(n_seen, 1)
            pbar.update(1)
            pbar.set_postfix_str(f'loss={avg_so_far:.4f}', refresh=False)

            del x, p, y, mean, std, shape, pred, loss

        pbar.close()

    val_curve.mark_complete()
    avg_loss = weighted_loss_sum / max(n_seen, 1)
    # No files written — print mean and show the same series as --validation.
    print(f"Mean val loss: {avg_loss:.6f}", flush=True)
    val_curve.print_summary()
    epoch_label = plot_epoch if plot_epoch is not None else '?'
    val_curve.plot(
        title=(
            f'Validation Loss — epoch {epoch_label} '
            f'({mode_tag} [{percent_lo:g}%, {percent_hi:g}%], '
            f'n={len(val_curve.batch_losses)} batches)'
        ),
    )

    return {
        'avg_loss': avg_loss,
        'n_used': n_seen,
        'n_val_total': n_total,
        'percent_lo': percent_lo,
        'percent_hi': percent_hi,
        'epoch': plot_epoch,
        'checkpoint_mode': mode,
        'checkpoint_path': checkpoint_path,
        'checkpoint_key': checkpoint_key,
        'used_indices': used_indices,
    }


# ═══════════════════════════════════════════════════════════
# VAE vs YIELD COVARIANCE (--vae-compare)
# ═══════════════════════════════════════════════════════════

def run_vae_compare(
    config: dict,
    vae_key: str,
    yield_key: str,
    percent_hi: float = 100.0,
    percent_lo: float = 0.0,
):
    """
    Load a VAE checkpoint and a yield (regression) checkpoint, evaluate the
    same sorted validation samples, and plot per-sample losses to inspect
    covariance (scatter + dual series + Pearson/Spearman).
    """
    import matplotlib.pyplot as plt
    from labels import Label_Yield, resolve_yield_target

    vae_key = str(vae_key).strip()
    yield_key = str(yield_key).strip()
    if not vae_key or not yield_key:
        raise ValueError("--vae-compare requires both --vae-key and --yield-key")

    percent_lo = float(percent_lo)
    percent_hi = float(percent_hi)
    if not (0.0 <= percent_lo < percent_hi <= 100.0):
        raise ValueError(
            f"--vae-compare requires 0 <= V1 < V2 <= 100 "
            f"(got V1={percent_lo}, V2={percent_hi})"
        )

    device = config['device']
    batch_size = int(
        config.get(
            'validate_batch_size',
            config.get('test_batch_size', config.get('batch_size', 8)),
        )
    )
    recon_w = float(config.get('reconstruction_loss_weight', 1.0))
    kl_w = float(config.get('vae_kl_weight', 0.1))

    print("\n" + "=" * 60)
    print("VAE ↔ YIELD LOSS COVARIANCE")
    print("=" * 60)
    print(f"  VAE key:   {vae_key}")
    print(f"  Yield key: {yield_key}")
    print(f"  Coverage:  [{percent_lo:g}%, {percent_hi:g}%] of sorted val samples")
    print(f"  VAE loss:  {recon_w:g}·MSE(x̂,x) + {kl_w:g}·KL  (per sample)")
    print("  Yield loss: chance/near MSE (per sample)")
    print("=" * 60 + "\n")

    print("Loading dataset...", flush=True)
    dataset, _ = get_dataset(config)
    config['output_dim'] = dataset.output_dim
    if getattr(dataset, 'label_mode', None) != 'yield':
        print(
            f"⚠ label_mode={getattr(dataset, 'label_mode', None)!r}; "
            "yield branch expects label=yield."
        )

    ckpt_dir = config.get('checkpoint_dir', 'trading_checkpoints')
    vae_found = find_checkpoint_by_key(ckpt_dir, vae_key)
    yield_found = find_checkpoint_by_key(ckpt_dir, yield_key)
    if vae_found is None:
        raise SystemExit(f"No VAE checkpoint under key {vae_key!r}")
    if yield_found is None:
        raise SystemExit(f"No yield checkpoint under key {yield_key!r}")
    vae_path, _ = vae_found
    yield_path, _ = yield_found

    cfg_vae = dict(config)
    cfg_vae['model_type'] = 'vae'
    cfg_yield = dict(config)
    cfg_yield['model_type'] = 'regression'

    print(f"Loading VAE from:   {vae_path}", flush=True)
    model_vae = load_model(vae_path, cfg_vae, dataset, device)
    print(f"Loading yield from: {yield_path}", flush=True)
    model_yield = load_model(yield_path, cfg_yield, dataset, device)

    yield_aux = Label_Yield.make_auxiliary(config)
    chance_w = float(yield_aux.get('chance_weight', 1.0))
    near_w = float(yield_aux.get('near_weight', yield_aux.get('wait_weight', 1.0)))
    yt = resolve_yield_target(config)

    val_sorted = sorted(dataset.val_indices)
    n_total = len(val_sorted)
    start = int(n_total * percent_lo / 100.0)
    end = n_total if percent_hi >= 100.0 else int(n_total * percent_hi / 100.0)
    end = max(0, min(n_total, end))
    used_indices = val_sorted[start:end]
    n_take = len(used_indices)
    if n_take == 0:
        raise RuntimeError(
            f"Percent slice [{percent_lo:g}%, {percent_hi:g}%] selected 0 samples"
        )
    print(
        f"Val samples: [{start}:{end}) → {n_take} / {n_total}",
        flush=True,
    )

    # Map each sample to calendar time at the last history bar.
    # binance_start_date is UTC wall time; weekday() here is therefore UTC DOW.
    from datetime import datetime, timedelta

    try:
        series_origin = datetime.strptime(
            str(config.get('binance_start_date')), '%Y-%m-%d %H:%M:%S'
        )
    except Exception:
        series_origin = None
        print(
            "⚠ Could not parse binance_start_date; day-of-week breakdown skipped.",
            flush=True,
        )
    past_window = int(getattr(dataset, 'past_window', resolve_past_window(config)))
    dow_names = (
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    )

    vae_losses: List[float] = []
    yield_losses: List[float] = []
    dow_ids: List[int] = []
    n_batches = (n_take + batch_size - 1) // batch_size

    model_vae.eval()
    model_yield.eval()
    with torch.no_grad():
        pbar = tqdm(
            total=n_batches,
            desc=f"vae-compare [{percent_lo:g}%, {percent_hi:g}%]",
            mininterval=0.5,
            dynamic_ncols=True,
        )
        for b in range(n_batches):
            batch_start = b * batch_size
            batch_end = min(batch_start + batch_size, n_take)
            batch_idx = used_indices[batch_start:batch_end]

            x, p, y, mean, std, shape = _collate_batch(dataset, batch_idx)
            x = x.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mean = mean.to(device, non_blocking=True)
            std = std.to(device, non_blocking=True)
            shape = shape.to(device, non_blocking=True)

            # ---- VAE per-sample loss ----
            _ = model_vae(x, p, mean, std, shape)
            x_hat = model_vae.last_x_hat
            recon_per = ((x_hat - x) ** 2).reshape(x.shape[0], -1).mean(dim=1)
            kl_per = model_vae.last_kl_per
            if kl_per is None:
                kl_per = torch.zeros_like(recon_per)
            vae_per = recon_w * recon_per + kl_w * kl_per

            # ---- Yield per-sample loss ----
            y_pred_t = model_yield(x, p, mean, std, shape)
            if isinstance(y_pred_t, (tuple, list)):
                y_pred = y_pred_t[0]
            else:
                y_pred = y_pred_t
            if yt == 'both' and y_pred.shape[-1] >= 2 and y.shape[-1] >= 2:
                yield_per = (
                    chance_w * ((y_pred[:, 0] - y[:, 0]) ** 2)
                    + near_w * ((y_pred[:, 1] - y[:, 1]) ** 2)
                )
            else:
                yield_per = ((y_pred - y) ** 2).mean(dim=-1)

            vae_losses.extend(vae_per.detach().cpu().tolist())
            yield_losses.extend(yield_per.detach().cpu().tolist())
            if series_origin is not None:
                for di in batch_idx:
                    bar0 = int(dataset.valid_starts[int(di)])
                    # Last bar of the history window ≈ decision / sample time.
                    ts = series_origin + timedelta(minutes=bar0 + past_window - 1)
                    dow_ids.append(int(ts.weekday()))  # Mon=0 … Sun=6
            pbar.update(1)
            del x, p, y, mean, std, shape, y_pred_t, y_pred, vae_per, yield_per
        pbar.close()

    vae_arr = np.asarray(vae_losses, dtype=np.float64)
    yield_arr = np.asarray(yield_losses, dtype=np.float64)
    n = len(vae_arr)
    if n < 2:
        raise RuntimeError("Need at least 2 samples for correlation")

    # Pearson
    pearson = float(np.corrcoef(vae_arr, yield_arr)[0, 1])
    # Spearman via rank
    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a), dtype=np.float64)
        return ranks

    spearman = float(np.corrcoef(_rank(vae_arr), _rank(yield_arr))[0, 1])

    def _pct(a: np.ndarray, q: float) -> float:
        return float(np.percentile(a, q))

    print("\n" + "=" * 60)
    print("COVARIANCE SUMMARY")
    print("=" * 60)
    print(f"  n samples:     {n}")
    print(
        f"  VAE loss:      mean={vae_arr.mean():.6f}  std={vae_arr.std():.6f}  "
        f"median={_pct(vae_arr, 50):.6f}  p90={_pct(vae_arr, 90):.6f}"
    )
    print(
        f"  Yield loss:    mean={yield_arr.mean():.6f}  std={yield_arr.std():.6f}  "
        f"median={_pct(yield_arr, 50):.6f}  p90={_pct(yield_arr, 90):.6f}"
    )
    print(f"  Pearson r:     {pearson:.6f}")
    print(f"  Spearman ρ:    {spearman:.6f}")
    if yield_arr.mean() > 1e-8 and _pct(yield_arr, 50) < 0.1 * yield_arr.mean():
        print(
            "  note: yield median ≪ mean → heavy-tailed / many near-zero samples"
        )
    print("=" * 60)

    dow_stats = None
    if series_origin is not None and len(dow_ids) == n:
        dow_arr = np.asarray(dow_ids, dtype=np.int64)
        print("\n" + "=" * 60)
        print("DAY-OF-WEEK BREAKDOWN  (UTC; time = last history bar)")
        print("=" * 60)
        print(
            f"  {'weekday':<10} {'n':>7}  "
            f"{'VAE mean':>10} {'VAE med':>10}  "
            f"{'Yld mean':>10} {'Yld med':>10}"
        )
        dow_stats = []
        for d in range(7):
            mask = dow_arr == d
            nd = int(mask.sum())
            if nd == 0:
                dow_stats.append(
                    {
                        'weekday': dow_names[d],
                        'n': 0,
                        'vae_mean': float('nan'),
                        'vae_median': float('nan'),
                        'yield_mean': float('nan'),
                        'yield_median': float('nan'),
                    }
                )
                print(
                    f"  {dow_names[d]:<10} {nd:7d}  "
                    f"{'—':>10} {'—':>10}  {'—':>10} {'—':>10}"
                )
                continue
            vm = float(vae_arr[mask].mean())
            vmed = float(np.median(vae_arr[mask]))
            ym = float(yield_arr[mask].mean())
            ymed = float(np.median(yield_arr[mask]))
            dow_stats.append(
                {
                    'weekday': dow_names[d],
                    'n': nd,
                    'vae_mean': vm,
                    'vae_median': vmed,
                    'yield_mean': ym,
                    'yield_median': ymed,
                }
            )
            print(
                f"  {dow_names[d]:<10} {nd:7d}  "
                f"{vm:10.6f} {vmed:10.6f}  {ym:10.6f} {ymed:10.6f}"
            )
        print("=" * 60)

    idx = np.arange(n)
    fig = plt.figure(figsize=(14, 9))
    ax0 = fig.add_subplot(2, 2, 1)
    ax1 = fig.add_subplot(2, 2, 2)
    ax2 = fig.add_subplot(2, 1, 2)

    ax0.scatter(vae_arr, yield_arr, s=8, alpha=0.35, c='#1565C0', edgecolors='none')
    ax0.set_xlabel(f'VAE loss ({recon_w:g}·MSE + {kl_w:g}·KL)')
    ax0.set_ylabel('Yield loss (chance/near MSE)')
    ax0.set_title(f'Scatter  Pearson r={pearson:.3f}  Spearman ρ={spearman:.3f}')
    ax0.grid(True, alpha=0.3)

    color_yield = '#C62828'
    color_vae = '#2E7D32'
    ln_y = ax1.plot(
        idx, yield_arr, color=color_yield, linewidth=1.0, alpha=0.85,
        label='Yield loss', zorder=1,
    )
    ax1.set_xlabel('Validation sample index (sorted chronologically)')
    ax1.set_ylabel('Yield loss', color=color_yield)
    ax1.tick_params(axis='y', labelcolor=color_yield)
    ax1.grid(True, alpha=0.3)

    ax1b = ax1.twinx()
    ln_v = ax1b.plot(
        idx, vae_arr, color=color_vae, linewidth=1.0, alpha=0.85,
        label='VAE loss', zorder=2,
    )
    ax1b.set_ylabel('VAE loss', color=color_vae)
    ax1b.tick_params(axis='y', labelcolor=color_vae)

    lines = ln_y + ln_v
    ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=9, loc='upper right')
    ax1.set_title('Dual series (twin y-axes)')

    # Day-of-week bar chart (VAE + yield means)
    if dow_stats is not None:
        x_pos = np.arange(7)
        width = 0.38
        vae_means = [
            s['vae_mean'] if s['n'] > 0 else 0.0 for s in dow_stats
        ]
        yld_means = [
            s['yield_mean'] if s['n'] > 0 else 0.0 for s in dow_stats
        ]
        bars_v = ax2.bar(
            x_pos - width / 2, vae_means, width,
            color=color_vae, alpha=0.85, label='VAE mean',
        )
        ax2.set_ylabel('VAE mean loss', color=color_vae)
        ax2.tick_params(axis='y', labelcolor=color_vae)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([s['weekday'][:3] for s in dow_stats])
        ax2.set_xlabel('Day of week (last history bar)')
        ax2.grid(True, axis='y', alpha=0.3)

        ax2b = ax2.twinx()
        bars_y = ax2b.bar(
            x_pos + width / 2, yld_means, width,
            color=color_yield, alpha=0.85, label='Yield mean',
        )
        ax2b.set_ylabel('Yield mean loss', color=color_yield)
        ax2b.tick_params(axis='y', labelcolor=color_yield)

        # Annotate sample counts
        for i, s in enumerate(dow_stats):
            ax2.text(
                i, 0.0, f"n={s['n']}", ha='center', va='bottom',
                fontsize=8, color='#333333',
            )

        ax2.legend(
            [bars_v, bars_y], ['VAE mean', 'Yield mean'],
            fontsize=9, loc='upper right',
        )
        ax2.set_title('Mean loss by day of week')
    else:
        ax2.set_visible(False)

    fig.suptitle(
        f'VAE ({vae_key[:8]}…) vs Yield ({yield_key[:8]}…)  '
        f'[{percent_lo:g}%, {percent_hi:g}%]',
        fontsize=12,
        fontweight='bold',
    )
    plt.tight_layout()
    plt.show()

    return {
        'vae_key': vae_key,
        'yield_key': yield_key,
        'n': n,
        'vae_losses': vae_arr,
        'yield_losses': yield_arr,
        'pearson': pearson,
        'spearman': spearman,
        'percent_lo': percent_lo,
        'percent_hi': percent_hi,
        'dow_stats': dow_stats,
    }


# ═══════════════════════════════════════════════════════════
# YIELD VALIDATION (--validate-yield)
# ═══════════════════════════════════════════════════════════

def _select_val_indices(
    dataset,
    max_samples: Optional[int] = None,
):
    """
    Choose validation indices in chronological order.

    When max_samples is set, take an evenly spaced subset of the sorted
    validation split (same coverage idea as --test N).
    """
    chronological = sorted(dataset.val_indices)
    if not chronological:
        return []

    if max_samples is not None and max_samples < len(chronological):
        selected = _evenly_spaced_indices(chronological, int(max_samples))
        print(
            f"  (using {len(selected)} val samples spaced across the "
            f"{len(chronological)}-window validation period)",
            flush=True,
        )
    else:
        selected = list(chronological)
        print(
            f"  (using all {len(selected)} validation samples)",
            flush=True,
        )
    return selected


def _plot_chance_near_over_samples(
    preds: torch.Tensor,
    labels: torch.Tensor,
    yield_target: str,
    title: str = 'Chance / Near over validation samples',
    save_path: Optional[str] = None,
):
    """
    Plot true vs predicted chance and near against sample order.

    Layout depends on yield_target:
      both   → 2×2: chance series, near series, chance scatter, near scatter
      chance → 1×2: chance series + scatter
      near   → 1×2: near series + scatter
    """
    import matplotlib.pyplot as plt

    pred = preds.detach().cpu().float()
    lab = labels.detach().cpu().float()
    if pred.ndim == 1:
        pred = pred.unsqueeze(-1)
    if lab.ndim == 1:
        lab = lab.unsqueeze(-1)

    x = np.arange(pred.shape[0])
    yt = str(yield_target).lower()

    def _series_and_scatter(ax_s, ax_sc, true_v, pred_v, name: str, color: str):
        ax_s.plot(x, true_v, label=f'true {name}', color=color, alpha=0.85, linewidth=1.2)
        ax_s.plot(
            x, pred_v, label=f'pred {name}', color=color,
            alpha=0.55, linewidth=1.2, linestyle='--',
        )
        ax_s.set_xlabel('validation sample (chronological)')
        ax_s.set_ylabel(name)
        ax_s.set_title(f'{name} over validation samples')
        ax_s.legend(loc='best', fontsize=8)
        ax_s.grid(True, alpha=0.3)

        ax_sc.scatter(true_v, pred_v, s=12, alpha=0.55, color=color, edgecolors='none')
        lo = float(min(true_v.min(), pred_v.min()))
        hi = float(max(true_v.max(), pred_v.max()))
        pad = 0.05 * (hi - lo + 1e-8)
        ax_sc.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', linewidth=1.0, alpha=0.6)
        ax_sc.set_xlabel(f'true {name}')
        ax_sc.set_ylabel(f'pred {name}')
        ax_sc.set_title(f'{name}: pred vs true')
        ax_sc.grid(True, alpha=0.3)
        ax_sc.set_aspect('equal', adjustable='datalim')

    if yt == 'chance':
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        _series_and_scatter(
            axes[0], axes[1],
            lab[:, 0].numpy(), pred[:, 0].numpy(),
            'chance', '#1f77b4',
        )
    elif yt == 'near':
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        _series_and_scatter(
            axes[0], axes[1],
            lab[:, 0].numpy(), pred[:, 0].numpy(),
            'near', '#ff7f0e',
        )
    else:
        # yield_target='both' → (chance, near)
        if pred.shape[-1] < 2 or lab.shape[-1] < 2:
            raise ValueError(
                f"yield_target=both requires 2-D preds/labels; "
                f"got pred={tuple(pred.shape)}, labels={tuple(lab.shape)}"
            )
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        _series_and_scatter(
            axes[0, 0], axes[0, 1],
            lab[:, 0].numpy(), pred[:, 0].numpy(),
            'chance', '#1f77b4',
        )
        _series_and_scatter(
            axes[1, 0], axes[1, 1],
            lab[:, 1].numpy(), pred[:, 1].numpy(),
            'near', '#ff7f0e',
        )

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"  Chance/near plot saved to: {save_path}", flush=True)
    plt.show()
    return fig


def run_validate_yield(
    config: dict,
    max_samples: Optional[int] = None,
    visualize: bool = True,
):
    """
    Load the latest checkpoint for ``config``, run inference on validation
    samples, print chance/near metrics, plot chance and near over the val
    set, and optionally open the interactive yield visualizer.

    Args:
        config: Full training/inference config (same as TRAIN).
        max_samples: If set, evaluate this many val samples spaced across
                     the chronological validation period; otherwise all.
        visualize: If True, also page through yield price windows.

    Batch size comes from config['validate_batch_size']
    (falls back to test_batch_size, then batch_size).
    """
    from labels import resolve_yield_target

    device = config['device']
    batch_size = config.get(
        'validate_batch_size',
        config.get('test_batch_size', config.get('batch_size', 8)),
    )

    # This mode is specifically for yield labels.
    label_mode = str(config.get('label', 'trend')).lower()
    if label_mode not in ('yield', 'scattered_hope'):
        print(
            f"⚠ --validate-yield expects label=yield "
            f"(config currently has label={config.get('label')!r}). "
            f"Forcing label='yield' for this run.",
            flush=True,
        )
        config['label'] = 'yield'

    print("\nLoading dataset (same path as TRAIN)...", flush=True)
    dataset, _total_features = get_dataset(config)
    config['output_dim'] = dataset.output_dim
    if getattr(dataset, 'label_mode', None) != 'yield':
        raise RuntimeError(
            f"--validate-yield requires a yield dataset; got label_mode="
            f"{getattr(dataset, 'label_mode', None)!r}"
        )
    config['yield_target'] = getattr(dataset, 'yield_target', 'both')
    yield_target = resolve_yield_target(
        config,
        yield_target=getattr(dataset, 'yield_target', None),
    )

    checkpoint_key = generate_checkpoint_key(config, dataset.cache_key)
    print(f"🔑 Generated Checkpoint Key: {checkpoint_key}", flush=True)

    ckpt_dir = config.get('checkpoint_dir', 'trading_checkpoints')
    result = find_latest_checkpoint_by_key(ckpt_dir, checkpoint_key)
    if result is None:
        print(f"⚠ No checkpoint found for key: {checkpoint_key}")
        print("  Looking for newest checkpoint across keys instead...")
        result = find_latest_checkpoint_by_key(ckpt_dir, None)
    if result is None:
        print("No checkpoint found. Exiting.")
        raise SystemExit(1)

    checkpoint_path, checkpoint_key = result

    print(f"\nLoading model from: {checkpoint_path}", flush=True)
    print(f"Checkpoint Key: {checkpoint_key}", flush=True)
    print(f"Validate-yield batch size: {batch_size}", flush=True)
    if max_samples is not None:
        print(f"Val sample cap: {max_samples}", flush=True)
    else:
        print("Val sample cap: all", flush=True)

    model = load_model(checkpoint_path, config, dataset, device)
    print("✓ Model loaded successfully", flush=True)

    val_indices = _select_val_indices(dataset, max_samples=max_samples)
    print("\nRunning batched inference on validation samples...", flush=True)
    preds, val_labels, used_indices = run_inference_on_indices(
        model,
        dataset,
        val_indices,
        device=device,
        batch_size=batch_size,
        split_name='validation samples',
    )
    print(f"✓ Finished inference on {len(preds)} validation samples", flush=True)

    out_dim = preds.shape[-1]
    print("\n📊 Validation Yield Results:")
    metrics = {}
    if yield_target == 'chance':
        metrics['chance_mse'] = torch.nn.functional.mse_loss(
            preds[:, 0], val_labels[:, 0]
        ).item()
        print(f"  Chance MSE:      {metrics['chance_mse']:.4f}")
    elif yield_target == 'near':
        metrics['near_mse'] = torch.nn.functional.mse_loss(
            preds[:, 0], val_labels[:, 0]
        ).item()
        print(f"  Near MSE:        {metrics['near_mse']:.4f}")
    elif out_dim == 2:
        metrics['chance_mse'] = torch.nn.functional.mse_loss(
            preds[:, 0], val_labels[:, 0]
        ).item()
        metrics['near_mse'] = torch.nn.functional.mse_loss(
            preds[:, 1], val_labels[:, 1]
        ).item()
        print(f"  Chance MSE:      {metrics['chance_mse']:.4f}")
        print(f"  Near MSE:        {metrics['near_mse']:.4f}")
    else:
        raise ValueError(
            f"Unexpected yield pred dim {out_dim} for yield_target={yield_target!r}"
        )

    print(f"\n📈 Sample Predictions (first 5):")
    if yield_target == 'chance':
        print(f"  {'Index':<6} {'True Chance':<12} {'Pred Chance':<12}")
        print(f"  {'-'*6} {'-'*12} {'-'*12}")
        for i in range(min(5, len(preds))):
            print(
                f"  {i:<6} {val_labels[i, 0].item():<12.4f} "
                f"{preds[i, 0].item():<12.4f}"
            )
    elif yield_target == 'near':
        print(f"  {'Index':<6} {'True Near':<12} {'Pred Near':<12}")
        print(f"  {'-'*6} {'-'*12} {'-'*12}")
        for i in range(min(5, len(preds))):
            print(
                f"  {i:<6} {val_labels[i, 0].item():<12.4f} "
                f"{preds[i, 0].item():<12.4f}"
            )
    else:
        print(
            f"  {'Index':<6} {'True Chance':<12} {'Pred Chance':<12} "
            f"{'True Near':<12} {'Pred Near':<12}"
        )
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for i in range(min(5, len(preds))):
            print(
                f"  {i:<6} {val_labels[i, 0].item():<12.4f} "
                f"{preds[i, 0].item():<12.4f} "
                f"{val_labels[i, 1].item():<12.4f} "
                f"{preds[i, 1].item():<12.4f}"
            )

    pred_cpu = preds.detach().cpu()
    lab_cpu = val_labels.detach().cpu()
    plot_path = Path(checkpoint_path).parent / 'validate_yield_chance_near.png'
    print("\n📈 Plotting chance and near over validation samples...", flush=True)
    _plot_chance_near_over_samples(
        pred_cpu,
        lab_cpu,
        yield_target=yield_target,
        title=(
            f'Chance / Near — validation '
            f'(latest ckpt, n={len(used_indices)}, yield_target={yield_target})'
        ),
        save_path=str(plot_path),
    )

    if visualize:
        n_viz = len(used_indices)
        valid_starts = [dataset.valid_starts[i] for i in used_indices]
        full_labels = torch.stack(
            [dataset.label_tensor[i] for i in used_indices]
        )
        print(
            f"\n📈 Visualizing yield labels "
            f"(yield_target={yield_target}) on validation samples...",
            flush=True,
        )
        visualize_yield(
            total_tensor=dataset.Total,
            valid_starts=valid_starts,
            label_tensor=lab_cpu if yield_target != 'both' else full_labels,
            price_feature_idx=dataset.price_feature_idx,
            past_window=dataset.past_window,
            future_window=dataset.future_window,
            min_yield_percent=config.get(
                'min_yield_percent',
                getattr(dataset, 'min_yield_percent', 0.0),
            ),
            max_hope_future_horizon=config.get(
                'future_window',
                getattr(dataset, 'future_window', None),
            ),
            yield_target=yield_target,
            num_samples=min(6, n_viz),
            pred_tensor=pred_cpu,
            fixed_order=True,
            sample_ids=list(used_indices),
            label_mean=getattr(dataset, '_label_stat_mean', None),
            label_std=getattr(dataset, '_label_stat_std', None),
        )

    print("\n✓ Validate-yield complete!", flush=True)
    return {
        'preds': preds,
        'labels': val_labels,
        'used_indices': used_indices,
        'metrics': metrics,
        'output_dim': out_dim,
        'yield_target': yield_target,
        'checkpoint_path': checkpoint_path,
        'checkpoint_key': checkpoint_key,
        'plot_path': str(plot_path),
    }


if __name__ == "__main__":
    run_test(get_config())
