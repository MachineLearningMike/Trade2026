# data.py
# Dataset and DataLoader with per-window normalization and caching

import math
import torch
from torch.utils.data import Dataset, Subset
import threading
import queue
import random
import time
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import pickle
import hashlib
import matplotlib.pyplot as plt
from config import get_config


# ═══════════════════════════════════════════════════════════
# WEIGHTING FUNCTIONS - FAR FUTURE GETS HIGHER WEIGHTS
# ═══════════════════════════════════════════════════════════

def get_weights(horizon: int, weight_type: str = 'exponential', decay: float = 0.9) -> torch.Tensor:
    """
    Generate weights for different time slots.
    FAR FUTURE (closer to t=horizon-1) gets higher weights.
    NEAR FUTURE (closer to t=0) gets lower weights.
    
    Args:
        horizon: Number of time steps
        weight_type: 'exponential', 'linear', 'sqrt', 'sigmoid', 'step', 'none'
        decay: Decay factor for exponential weighting (0-1)
    
    Returns:
        weights: Tensor of shape (horizon,)
    """
    t = torch.arange(horizon, dtype=torch.float32)
    
    if weight_type == 'exponential':
        weights = torch.exp(decay * t)
        weights = weights / weights.sum()
    elif weight_type == 'linear':
        weights = torch.linspace(0.1, 1, horizon)
        weights = weights / weights.sum()
    elif weight_type == 'sqrt':
        weights = torch.sqrt(t + 1)
        weights = weights / weights.sum()
    elif weight_type == 'sigmoid':
        mid = horizon / 2
        weights = 1 / (1 + torch.exp(-(t - mid) / (horizon / 6)))
        weights = weights + 0.1
        weights = weights / weights.sum()
    elif weight_type == 'step':
        weights = torch.ones(horizon)
        mid = horizon // 2
        weights[:mid] = 0.2
        weights = weights / weights.sum()
    elif weight_type == 'none' or weight_type == 'uniform':
        weights = torch.ones(horizon, dtype=torch.float32)
        weights = weights / weights.sum()
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")
    
    return weights


# ═══════════════════════════════════════════════════════════
# HORIZON LABELING FUNCTION - SUPPORTS ALL MODES
# ═══════════════════════════════════════════════════════════

def find_shortest_horizon(
    future_prices: torch.Tensor,
    max_accum_deviation: float,
    min_horizon: int = 2,
    max_horizon: Optional[int] = None,
    device: str = 'cpu',
    mode: str = 'anchored',
    weight_type: str = 'exponential',
    weight_decay: float = 0.9,
    verbose: bool = False
) -> Tuple:
    """Find the SINGLE SHORTEST horizon where cumulative deviation reaches max_accum_deviation."""
    
    prices = future_prices.to(device)
    if prices.dim() > 1:
        prices = prices.squeeze()
    
    future_length = prices.shape[0]
    
    if max_horizon is None:
        max_horizon = future_length
    else:
        max_horizon = min(max_horizon, future_length)
    
    all_horizons = []
    all_cum_devs = []
    all_slopes = []
    all_intercepts = []
    all_weights = []
    
    if verbose:
        print(f"\nMode: {mode.upper()}")
        print(f"{'H':>4} | {'Slope':>12} | {'Cum Dev':>12} | {'Status':>15}")
        print("-" * 50)
    
    for h in range(min_horizon, max_horizon + 1):
        segment = prices[:h]
        t = torch.arange(h, dtype=torch.float32, device=device)
        
        if mode == 'floating':
            y_mean = segment.mean()
            t_mean = t.mean()
            t_centered = t - t_mean
            y_centered = segment - y_mean
            numerator = torch.sum(t_centered * y_centered)
            denominator = torch.sum(t_centered ** 2)
            if denominator < 1e-10:
                slope = 0.0
                intercept = y_mean.item()
            else:
                slope = (numerator / denominator).item()
                intercept = (y_mean - slope * t_mean).item()
            
        elif mode == 'anchored':
            f0 = prices[0].item()
            y_centered = segment - f0
            numerator = torch.sum(t * y_centered)
            denominator = torch.sum(t ** 2)
            if denominator < 1e-10:
                slope = 0.0
            else:
                slope = (numerator / denominator).item()
            intercept = f0
            
        elif mode == 'weighted':
            weights = get_weights(h, weight_type, weight_decay)
            all_weights.append(weights.cpu().numpy())
            w = weights.to(device)
            w_sum = w.sum()
            t_mean_w = torch.sum(w * t) / w_sum
            y_mean_w = torch.sum(w * segment) / w_sum
            t_centered = t - t_mean_w
            y_centered = segment - y_mean_w
            numerator = torch.sum(w * t_centered * y_centered)
            denominator = torch.sum(w * t_centered ** 2)
            if denominator < 1e-10:
                slope = 0.0
                intercept = y_mean_w.item()
            else:
                slope = (numerator / denominator).item()
                intercept = (y_mean_w - slope * t_mean_w).item()
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        predicted = slope * t + intercept
        deviations = torch.abs(segment - predicted)
        cumulative_deviation = torch.sum(deviations).item()
        # torch.sqrt(torch.mean((segment - predicted) ** 2)).item() 
        # torch.mean(torch.abs(segment - predicted)).item() 
        # # torch.sum(deviations).item()
        
        all_horizons.append(h)
        all_cum_devs.append(cumulative_deviation)
        all_slopes.append(slope)
        all_intercepts.append(intercept)
        
        if verbose:
            status = "✓ REACHED" if cumulative_deviation >= max_accum_deviation else "..."
            print(f"{h:>4} | {slope:>12.6f} | {cumulative_deviation:>12.4f} | {status:>15}")
        
        if cumulative_deviation >= max_accum_deviation:
            if mode == 'weighted':
                return h, slope, intercept, cumulative_deviation, all_horizons, all_cum_devs, all_weights, all_slopes, all_intercepts
            else:
                return h, slope, intercept, cumulative_deviation, all_horizons, all_cum_devs, all_slopes
    
    # If we never reach the threshold, return the max horizon
    h = max_horizon
    segment = prices[:h]
    t = torch.arange(h, dtype=torch.float32, device=device)
    
    if mode == 'floating':
        y_mean = segment.mean()
        t_mean = t.mean()
        t_centered = t - t_mean
        y_centered = segment - y_mean
        numerator = torch.sum(t_centered * y_centered)
        denominator = torch.sum(t_centered ** 2)
        if denominator < 1e-10:
            slope = 0.0
            intercept = y_mean.item()
        else:
            slope = (numerator / denominator).item()
            intercept = (y_mean - slope * t_mean).item()
            
    elif mode == 'anchored':
        f0 = prices[0].item()
        y_centered = segment - f0
        numerator = torch.sum(t * y_centered)
        denominator = torch.sum(t ** 2)
        if denominator < 1e-10:
            slope = 0.0
        else:
            slope = (numerator / denominator).item()
        intercept = f0
        
    elif mode == 'weighted':
        weights = get_weights(h, weight_type, weight_decay)
        all_weights.append(weights.cpu().numpy())
        w = weights.to(device)
        w_sum = w.sum()
        t_mean_w = torch.sum(w * t) / w_sum
        y_mean_w = torch.sum(w * segment) / w_sum
        t_centered = t - t_mean_w
        y_centered = segment - y_mean_w
        numerator = torch.sum(w * t_centered * y_centered)
        denominator = torch.sum(w * t_centered ** 2)
        if denominator < 1e-10:
            slope = 0.0
            intercept = y_mean_w.item()
        else:
            slope = (numerator / denominator).item()
            intercept = (y_mean_w - slope * t_mean_w).item()
    
    predicted = slope * t + intercept
    deviations = torch.abs(segment - predicted)
    cumulative_deviation = torch.sum(deviations).item()
    
    if mode == 'weighted':
        return h, slope, intercept, cumulative_deviation, all_horizons, all_cum_devs, all_weights, all_slopes, all_intercepts
    else:
        return h, slope, intercept, cumulative_deviation, all_horizons, all_cum_devs, all_slopes


# ═══════════════════════════════════════════════════════════
# VISUALIZATION FUNCTION
# ═══════════════════════════════════════════════════════════

def visualize_linear_trends(
    total_tensor: torch.Tensor,
    valid_starts: List[int],
    label_tensor: torch.Tensor,
    mean_tensor: torch.Tensor,
    std_tensor: torch.Tensor,
    price_feature_idx: int,
    past_window: int,
    future_window: int,
    mode: str = 'anchored',
    num_samples: int = 6,
    figsize: Tuple[int, int] = (15, 10)
):
    """Visualize linear trends for randomly selected samples."""
    if total_tensor.is_cuda:
        total_tensor = total_tensor.cpu()
    
    num_available = label_tensor.shape[0]
    num_samples = min(num_samples, num_available)
    if num_samples % 2 != 0:
        num_samples = 6

    # Set seed based on current second from epoch
    seed = int(time.time())  # Current second
    random.seed(seed)
    
    indices = random.sample(range(num_available), num_samples)
    n_cols = 3
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
        
        ax = axes[i]
        start = valid_starts[idx]
        label = label_tensor[idx]
        mean = mean_tensor[idx]
        std = std_tensor[idx]
        
        horizon = int(label[0].item())
        slope_norm = label[1].item()
        intercept_norm = label[2].item()
        cum_dev = label[3].item()
        threshold_reached = label[4].item()
        
        price_mean = mean[price_feature_idx].item()
        price_std = std[price_feature_idx].item()
        
        slope_original = slope_norm * price_std
        intercept_original = intercept_norm * price_std + price_mean
        
        y_start = start + past_window
        y_end = y_start + future_window
        future_prices = total_tensor[price_feature_idx, y_start:y_end].cpu().numpy()
        
        time_steps = np.arange(future_window)
        ax.plot(time_steps, future_prices, 'b-', linewidth=2, label='Actual Prices', alpha=0.7)
        
        if horizon > 0:
            trend_t = np.arange(horizon)
            trend_prices = slope_original * trend_t + intercept_original
            ax.plot(trend_t, trend_prices, 'r-', linewidth=2.5,
                    label=f'Trend: y={slope_original:.4f}t+{intercept_original:.2f}')
            ax.axvline(x=horizon - 1, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axvline(x=0, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
            ax.axvspan(0, horizon - 1, alpha=0.15, color='orange')
        
        status = '✓ Reached' if threshold_reached else '✗ Not Reached'
        ax.set_title(f'Sample {idx+1}: H={horizon}, m={slope_original:.4f}, b={intercept_original:.2f}\n'
                     f'Cum Dev={cum_dev:.2f}, {status}', fontsize=10)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    
    for i in range(len(indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Linear Trend Visualization - Mode: {mode.upper()}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════
# TRADING DATASET
# ═══════════════════════════════════════════════════════════

class TradingDataset(Dataset):
    """Trading dataset with per-window normalization and label caching."""
    
    def __init__(
        self, 
        features: List[np.ndarray], 
        price_feature_idx: int,
        past_days: int = 7,
        future_minutes: int = 60,
        step: int = 1,
        max_accum_deviation: float = 12.0,
        min_horizon: int = 3,
        max_horizon: Optional[int] = None,
        device: str = 'cpu',
        normalize: bool = True,
        cache_dir: Optional[str] = None,
        force_recompute: bool = False,
        visualize: bool = True,
        split_ratios: Optional[Tuple[float, float, float]] = None
    ):
        self.past_days = past_days
        self.future_minutes = future_minutes
        self.price_feature_idx = price_feature_idx
        self.step = step
        self.max_accum_deviation = max_accum_deviation
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.device = device
        self.normalize = normalize
        self.force_recompute = force_recompute
        self.visualize = visualize

        config = get_config()
        self.num_shapes = config['num_shapes']
        
        self.horizon_mode = config.get('horizon_mode', 'anchored')
        self.weight_type = config.get('weight_type', 'exponential')
        self.weight_decay = config.get('weight_decay', 0.9)
        
        if split_ratios is None:
            train_ratio = config.get('train_ratio', 0.70)
            val_ratio = config.get('val_ratio', 0.20)
            test_ratio = 1.0 - train_ratio - val_ratio
            split_ratios = (train_ratio, val_ratio, test_ratio)
        self.split_ratios = split_ratios
        self.shuffle_split = config.get('shuffle_split', True)
        
        if cache_dir is None:
            cache_dir = config.get('cache_dir', 'dataset_cache')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.Total = self._stack_features(features)
        self.F, self.T = self.Total.shape
        
        self.past_window = 60 * 24 * past_days
        self.future_window = future_minutes
        self.total_window = self.past_window + self.future_window
        
        if self.max_horizon is None:
            self.max_horizon = self.future_window
        elif self.max_horizon < self.min_horizon:
            self.max_horizon = self.min_horizon
            print(f"⚠ max_horizon adjusted to min_horizon ({self.min_horizon})")
        
        self.valid_starts = self._get_valid_starts()
        
        self.cache_key = self._generate_cache_key()
        self.cache_file = self.cache_dir / f"{self.cache_key}.pkl"
        
        self.label_tensor, self.mean_tensor, self.std_tensor, self.shape_tensor = self._load_or_compute_labels()
        
        self._create_splits()
        
        # Cache local references for faster __getitem__
        self._mean_tensor = self.mean_tensor
        self._std_tensor = self.std_tensor
        self._shape_tensor = self.shape_tensor
        self._label_tensor = self.label_tensor
        self._valid_starts = self.valid_starts
        self._Total = self.Total
        self._F = self.F
        self._past_window = self.past_window
        self._normalize = self.normalize
        
        if self.visualize and self.label_tensor.shape[0] > 0:
            print("\n" + "="*60)
            print("VISUALIZING LINEAR TRENDS")
            print("="*60)
            visualize_linear_trends(
                total_tensor=self.Total,
                valid_starts=self.valid_starts,
                label_tensor=self.label_tensor,
                mean_tensor=self.mean_tensor,
                std_tensor=self.std_tensor,
                price_feature_idx=self.price_feature_idx,
                past_window=self.past_window,
                future_window=self.future_window,
                mode=self.horizon_mode,
                num_samples=6
            )
        
        print(f"\nTradingDataset initialized:")
        print(f"  Features: {self.F}, Time steps: {self.T}")
        print(f"  Past window: {self.past_window} minutes ({past_days} days)")
        print(f"  Future window: {self.future_window} minutes")
        print(f"  Step: {step}")
        print(f"  Horizon mode: {self.horizon_mode}")
        if self.horizon_mode == 'weighted':
            print(f"  Weight type: {self.weight_type}, Decay: {self.weight_decay}")
        print(f"  Max accum deviation: {max_accum_deviation}")
        print(f"  Min horizon: {min_horizon}, Max horizon: {self.max_horizon}")
        print(f"  Normalize: {normalize}")
        print(f"  Total samples: {len(self)}")
        print(f"  Cache: {'Loaded' if self.cache_file.exists() else 'Computed'}")
        print(f"  Split ratios - Train: {self.split_ratios[0]:.2f}, "
              f"Val: {self.split_ratios[1]:.2f}, Test: {self.split_ratios[2]:.2f}")
        print(f"  Split sizes - Train: {len(self.train_indices)}, "
              f"Val: {len(self.val_indices)}, Test: {len(self.test_indices)}")
        
    def _stack_features(self, features: List[np.ndarray]) -> torch.Tensor:
        shapes = [arr.shape for arr in features]
        if len(set(shapes)) != 1:
            raise ValueError(f"All features must have the same shape. Got: {shapes}")
        stacked = np.stack(features, axis=0)
        return torch.from_numpy(stacked).float()
    
    def _get_valid_starts(self) -> List[int]:
        max_start = self.T - self.total_window
        if max_start < 0:
            raise ValueError(
                f"Time series length {self.T} is too short for "
                f"past {self.past_window} + future {self.future_window} = {self.total_window} minutes"
            )
        starts = list(range(0, max_start + 1, self.step))
        if not starts:
            raise ValueError(f"No valid starts with step {self.step}")
        return starts
    
    def _generate_cache_key(self) -> str:
        import hashlib
        config = get_config()
        symbol, feature = config.get('prediction_target', ('UNKNOWN', 'UNKNOWN'))
        symbols_str = "_".join(config.get('binance_symbols', {}).keys())
        
        params = (
            f"past_days={self.past_days}"
            f"future_minutes={self.future_minutes}"
            f"step={self.step}"
            f"max_accum_deviation={self.max_accum_deviation}"
            f"min_horizon={self.min_horizon}"
            f"max_horizon={self.max_horizon}"
            f"normalize={self.normalize}"
            f"horizon_mode={self.horizon_mode}"
            f"weight_type={self.weight_type}"
            f"weight_decay={self.weight_decay}"
            f"use_binance_data={config.get('use_binance_data', False)}"
            f"binance_datatype={config.get('binance_datatype', 'klines')}"
            f"binance_interval={config.get('binance_interval', '1m')}"
            f"binance_start_date={config.get('binance_start_date', '2024-01-01')}"
            f"binance_end_date={config.get('binance_end_date', '2026-06-30')}"
            f"binance_symbols={symbols_str}"
            f"prediction_target={symbol}.{feature}"
            f"F={self.F}"
            f"T={self.T}"
            f"past_window={self.past_window}"
            f"future_window={self.future_window}"
            f"num_samples={len(self.valid_starts)}"
            f"num_shapes={self.num_shapes}"
        )
        return hashlib.md5(params.encode()).hexdigest()[:16]
    
    def _normalize_window(
        self, 
        window: torch.Tensor, 
        compute_moments: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Normalize a window and optionally compute higher-order moments.
        
        Args:
            window: Input window of shape (F, T)
            compute_moments: If True, compute shape moments (skewness, kurtosis, etc.)
        
        Returns:
            normalized: Normalized window (always returned)
            mean: Mean of window (None if compute_moments=False)
            std: Standard deviation of window (None if compute_moments=False)
            shape: Higher-order moments (None if compute_moments=False)
        """
        mean = window.mean(dim=1, keepdim=True)
        std = window.std(dim=1, keepdim=True)
        std[std == 0] = 1
        
        normalized = (window - mean) / std
        
        if not compute_moments:
            return normalized, None, None, None
        
        mean = mean.squeeze()
        std = std.squeeze()
        
        centered = window - mean.unsqueeze(1)
        
        m3 = (centered ** 3).mean(dim=1)
        m4 = (centered ** 4).mean(dim=1)
        m5 = (centered ** 5).mean(dim=1)
        m6 = (centered ** 6).mean(dim=1)
        
        std_sq = std.clamp(min=1e-6)
        
        def signed_root(x, power):
            return torch.sign(x) * (torch.abs(x) ** (1/power))
        
        ms3 = signed_root(m3, 3) / std_sq
        ms4 = signed_root(m4, 4) / std_sq
        ms5 = signed_root(m5, 5) / std_sq
        ms6 = signed_root(m6, 6) / std_sq
        
        shape = torch.stack([ms3, ms4, ms5, ms6], dim=1)
        
        return normalized, mean, std, shape
    
    def _create_splits(self):
        num_samples = len(self.valid_starts)
        indices = list(range(num_samples))
        
        if self.shuffle_split:
            random.seed(42)
            random.shuffle(indices)
        
        train_ratio, val_ratio, test_ratio = self.split_ratios
        
        train_end = int(train_ratio * num_samples)
        val_end = train_end + int(val_ratio * num_samples)
        
        self.train_indices = indices[:train_end]
        self.val_indices = indices[train_end:val_end]
        self.test_indices = indices[val_end:]
        
        self.train_dataset = Subset(self, self.train_indices)
        self.val_dataset = Subset(self, self.val_indices)
        self.test_dataset = Subset(self, self.test_indices)
    
    def _compute_labels_and_norm_params(self):
        label_list, mean_list, std_list, shape_list = [], [], [], []
        
        for idx in tqdm(range(len(self)), desc="Computing labels and norm params"):
            start = self.valid_starts[idx]
            x = self.Total[:, start:start + self.past_window]
            
            if self.normalize:
                x_normalized, mean, std, shape = self._normalize_window(x, compute_moments=True)
                mean_list.append(mean)
                std_list.append(std)
                shape_list.append(shape)
            else:
                x_normalized = x
                mean = torch.zeros(self.F)
                std = torch.ones(self.F)
                shape = torch.zeros(self.F, self.num_shapes)
                mean_list.append(mean)
                std_list.append(std)
                shape_list.append(shape)

            y_start = start + self.past_window
            y_end = y_start + self.future_window
            future_prices = self.Total[self.price_feature_idx, y_start:y_end]
            
            if self.normalize:
                price_mean = mean[self.price_feature_idx]
                price_std = std[self.price_feature_idx]
                future_prices_normalized = (future_prices - price_mean) / price_std
            else:
                future_prices_normalized = future_prices
            
            if self.horizon_mode == 'weighted':
                h, m, b, cum, _, _, _, _, _ = find_shortest_horizon(
                    future_prices=future_prices_normalized,
                    max_accum_deviation=self.max_accum_deviation,
                    min_horizon=self.min_horizon,
                    max_horizon=self.max_horizon,
                    device=self.device,
                    mode=self.horizon_mode,
                    weight_type=self.weight_type,
                    weight_decay=self.weight_decay,
                    verbose=False
                )
            else:
                h, m, b, cum, _, _, _ = find_shortest_horizon(
                    future_prices=future_prices_normalized,
                    max_accum_deviation=self.max_accum_deviation,
                    min_horizon=self.min_horizon,
                    max_horizon=self.max_horizon,
                    device=self.device,
                    mode=self.horizon_mode,
                    verbose=False
                )

            threshold_reached = 1.0 if cum >= self.max_accum_deviation else 0.0
            label = torch.tensor([h, m, b, cum, threshold_reached], dtype=torch.float32)

            label_list.append(label)
        
        label_tensor = torch.stack(label_list)
        mean_tensor = torch.stack(mean_list)
        std_tensor = torch.stack(std_list)
        shape_tensor = torch.stack(shape_list)
        
        return label_tensor, mean_tensor, std_tensor, shape_tensor
    
    def _load_or_compute_labels(self):
        if not self.force_recompute and self.cache_file.exists():
            print(f"Loading cached labels from: {self.cache_file}")
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                if (cached_data.get('num_samples') == len(self) and
                    cached_data.get('F') == self.F and
                    cached_data.get('T') == self.T and
                    cached_data.get('past_window') == self.past_window and
                    cached_data.get('future_window') == self.future_window):
                    
                    label_tensor = cached_data['label_tensor']
                    mean_tensor = cached_data['mean_tensor']
                    std_tensor = cached_data['std_tensor']
                    shape_tensor = cached_data['shape_tensor']
                    
                    print(f"✓ Cache loaded successfully! ({label_tensor.shape[0]} samples)")
                    return label_tensor, mean_tensor, std_tensor, shape_tensor
                else:
                    print("⚠ Cache mismatch. Recomputing...")
            except Exception as e:
                print(f"⚠ Failed to load cache: {e}. Recomputing...")
        
        print("Computing labels and normalization parameters...")
        label_tensor, mean_tensor, std_tensor, shape_tensor = self._compute_labels_and_norm_params()
        
        try:
            cache_data = {
                'label_tensor': label_tensor,
                'mean_tensor': mean_tensor,
                'std_tensor': std_tensor,
                'shape_tensor': shape_tensor,
                'num_samples': len(self),
                'F': self.F,
                'T': self.T,
                'past_window': self.past_window,
                'future_window': self.future_window,
                'step': self.step,
                'max_accum_deviation': self.max_accum_deviation,
                'min_horizon': self.min_horizon,
                'max_horizon': self.max_horizon,
                'normalize': self.normalize,
                'horizon_mode': self.horizon_mode,
                'weight_type': self.weight_type,
                'weight_decay': self.weight_decay,
                'price_feature_idx': self.price_feature_idx,
                'cache_key': self.cache_key,
                'timestamp': time.time()
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✓ Cache saved to: {self.cache_file}")
            print(f"  Cache size: {self.cache_file.stat().st_size / 1024:.2f} KB")
        except Exception as e:
            print(f"⚠ Failed to save cache: {e}")
        
        return label_tensor, mean_tensor, std_tensor, shape_tensor
    
    def clear_cache(self):
        if self.cache_file.exists():
            self.cache_file.unlink()
            print(f"✓ Cache cleared: {self.cache_file}")
    
    @classmethod
    def clear_all_caches(cls, cache_dir: Optional[str] = None):
        if cache_dir is None:
            config = get_config()
            cache_dir = config.get('cache_dir', 'dataset_cache')
        cache_path = Path(cache_dir)
        if cache_path.exists():
            import shutil
            shutil.rmtree(cache_path)
            print(f"✓ All caches cleared from: {cache_path}")
        else:
            print(f"✓ No cache directory found at: {cache_path}")
    
    def __len__(self) -> int:
        return len(self.valid_starts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimized __getitem__ with cached references and view() instead of unsqueeze()."""
        
        start = self._valid_starts[idx]
        x = self._Total[:, start:start + self._past_window]
        
        if self._normalize:
            # Use view() instead of unsqueeze() - no memory allocation
            mean = self._mean_tensor[idx].view(-1, 1)
            std = self._std_tensor[idx].view(-1, 1)
            std[std == 0] = 1
            x_normalized = (x - mean) / std
            shape = self._shape_tensor[idx]
        else:
            x_normalized = x
            mean = torch.zeros(self._F)
            std = torch.ones(self._F)
            shape = torch.zeros(self._F, self.num_shapes)
        
        # No clone() - just return reference
        y = self._label_tensor[idx]
        
        return x_normalized, y, mean, std, shape


# ═══════════════════════════════════════════════════════════
# OPTIMIZED THREADED DATA LOADER
# ═══════════════════════════════════════════════════════════

# data.py - Updated ThreadedDataLoader with continuous prefetch thread

class ThreadedDataLoader:
    """DataLoader with continuous prefetch thread (doesn't restart per batch)."""
    
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.current_epoch_indices = None
        self.current_batch_start = 0
        self.epoch = 0
        
        # Larger queue to buffer batches
        self.prefetch_queue = queue.Queue(maxsize=10)
        self.prefetch_thread = None
        self.running = False
        self.iterator = None
        self.load_start_time = None
        
    def start_epoch(self, epoch, resume_indices=None, resume_batch_start=0):
        """Start a new epoch."""
        # Stop any existing thread
        self._stop_prefetch()
        
        # Clear the queue
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except:
                break
        
        self.epoch = epoch
        
        if resume_indices is not None:
            self.current_epoch_indices = resume_indices
        else:
            self.current_epoch_indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.seed(42 + epoch)
                random.shuffle(self.current_epoch_indices)
        
        self.current_batch_start = resume_batch_start
        self.running = True
        self._create_iterator()
        self._start_prefetch()  # ← Start ONCE for the entire epoch
    
    def _create_iterator(self):
        start = self.current_batch_start * self.batch_size
        remaining_indices = self.current_epoch_indices[start:]
        self.iterator = self._batch_generator(remaining_indices)
    
    def _batch_generator(self, indices):
        """Generator that yields batches."""
        dataset_get = self.dataset.__getitem__
        batch_size = self.batch_size
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            x_list, y_list, mean_list, std_list, shape_list = [], [], [], [], []
            
            for idx in batch_indices:
                x, y, mean, std, shape = dataset_get(idx)
                x_list.append(x)
                y_list.append(y)
                mean_list.append(mean)
                std_list.append(std)
                shape_list.append(shape)
            
            x_batch = torch.stack(x_list)
            y_batch = torch.stack(y_list)
            mean_batch = torch.stack(mean_list)
            std_batch = torch.stack(std_list)
            shape_batch = torch.stack(shape_list)
            
            yield x_batch, y_batch, mean_batch, std_batch, shape_batch
    
    def get_total_batches_in_epoch(self):
        """Get total number of batches in current epoch."""
        if self.current_epoch_indices is None:
            return math.ceil(len(self.dataset) / self.batch_size)
        return math.ceil(len(self.current_epoch_indices) / self.batch_size)
    
    def get_state(self):
        """Get current state for resuming."""
        return {
            'epoch_indices': self.current_epoch_indices,
            'batch_start': self.current_batch_start,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'epoch': self.epoch,
        }
    
    def restore_state(self, state):
        """Restore state from checkpoint."""
        self.current_epoch_indices = state['epoch_indices']
        self.current_batch_start = state['batch_start']
        self.batch_size = state['batch_size']
        self.shuffle = state['shuffle']
        self.epoch = state.get('epoch', 0)
    
    def _start_prefetch(self):
        """Start the prefetch thread (only if not already running)."""
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            return  # ← Don't restart if already running!
        
        self.load_start_time = time.time()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    
    def _prefetch_worker(self):
        """Continuous worker that prefetches ALL batches."""
        try:
            while self.running:
                try:
                    batch = next(self.iterator)
                    load_time = time.time() - self.load_start_time
                    self.prefetch_queue.put(('data', batch, load_time))
                    self.load_start_time = time.time()
                except StopIteration:
                    # Iterator exhausted - put STOP and exit
                    break
                except Exception as e:
                    self.prefetch_queue.put(('error', e, 0))
                    return
            
            # Only put STOP if we're still running (not stopped externally)
            if self.running:
                self.prefetch_queue.put(('stop', None, 0))
            self.running = False
            
        except Exception as e:
            self.prefetch_queue.put(('error', e, 0))
    
    def _stop_prefetch(self):
        """Stop the prefetch thread."""
        self.running = False
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=0.5)
        self.prefetch_thread = None
        
        # Clear the queue
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except:
                break
    
    def get_next_batch(self):
        """Get the next batch from the queue."""
        # Wait indefinitely for data
        status, data, load_time = self.prefetch_queue.get()
        
        if status == 'stop':
            self.running = False
            return None, load_time
        elif status == 'error':
            raise data
        
        return data, load_time
    
    def __del__(self):
        """Cleanup on deletion."""
        self._stop_prefetch()


# ═══════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("DATA MODULE DEMO")
    print("=" * 60)
    
    config = get_config()
    device = config['device']
    
    T = 60 * 24 * config['total_synthetic_days']
    F = config['num_features']
    
    print(f"Generating synthetic data: {T} minutes, {F} features")
    
    features = []
    for i in range(F):
        if i == config['price_feature_idx']:
            price = np.cumsum(np.random.randn(T) * 0.5 + 0.1)
            features.append(price.astype(np.float32))
        elif i == 1:
            volume = np.abs(np.random.randn(T) * 100 + 1000)
            features.append(volume.astype(np.float32))
        else:
            features.append(np.random.randn(T).astype(np.float32))
    
    dataset = TradingDataset(
        features=features,
        price_feature_idx=config['price_feature_idx'],
        past_days=config['past_days'],
        future_minutes=config['future_minutes'],
        step=config['step'],
        max_accum_deviation=config['max_accum_deviation'],
        min_horizon=config['min_horizon'],
        max_horizon=config['max_horizon'],
        device=device,
        normalize=config['normalize'],
        cache_dir=config.get('cache_dir', 'dataset_cache'),
        force_recompute=config.get('force_recompute', False),
        visualize=True
    )
    
    dataloader = ThreadedDataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    dataloader.start_epoch(0)
    
    print("\nTesting batch iteration...")
    batch, wait_time, load_time = dataloader.get_next_batch()
    if batch is not None:
        X, Y, MEAN, STD, SHAPE = batch
        print(f"Batch X shape: {X.shape}")
        print(f"Batch Y shape: {Y.shape}")
        print(f"Sample X[0] shape: {X[0].shape}")
        print(f"Sample Y[0]: {Y[0]}")
        print(f"  Horizon: {Y[0][0].item():.2f}")
        print(f"  Slope: {Y[0][1].item():.4f}")
        print(f"  Intercept: {Y[0][2].item():.4f}")
        print(f"  Wait time: {wait_time*1000:.2f}ms")
        print(f"  Load time: {load_time*1000:.2f}ms")
    
    print("\nData module demo complete!")