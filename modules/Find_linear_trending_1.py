"""
Dynamic Horizon Linear Trend Labeling for Neural Networks
==========================================================
This module generates labels (tangent, horizon) from future price sequences
where horizon is dynamically determined by cumulative deviation threshold.

ENHANCED VERSION: Includes differentiable labeling, advanced visualization,
validation metrics, realistic data generation, and production-ready features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List, Union, Any
from scipy import stats
import warnings
from collections import defaultdict
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

# Try to import seaborn, fall back to default style if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
    # Set style for better visualizations if seaborn is available
    sns.set_palette("husl")
    plt.style.use('seaborn-v0_8-darkgrid')
except ImportError:
    HAS_SEABORN = False
    print("Seaborn not found. Using default matplotlib styles.")
    # Use default matplotlib style
    plt.style.use('default')

warnings.filterwarnings('ignore')

# ============================================================================
# PART 0: CONFIGURATION AND DATA CLASSES
# ============================================================================

@dataclass
class LabelingConfig:
    """Configuration for dynamic horizon labeling."""
    max_accum_deviation: float = 15.0
    min_horizon: int = 3
    max_horizon: Optional[int] = None
    temperature: float = 1.0  # For differentiable labeling
    device: str = 'cpu'
    
@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    tangent_weight: float = 1.0
    horizon_weight: float = 0.5
    patience: int = 20
    min_delta: float = 1e-4
    gradient_clip: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10

# ============================================================================
# PART 1: ENHANCED LABEL GENERATION FUNCTIONS
# ============================================================================

def generate_labels_dynamic_horizon(
    y: torch.Tensor,
    max_accum_deviation: float,
    min_horizon: int = 2,
    max_horizon: Optional[int] = None,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate labels where horizon is dynamically extended until 
    cumulative deviation reaches max_accum_deviation.
    
    Args:
        y: Future price sequences, shape (batch_size, future_length)
        max_accum_deviation: Maximum cumulative absolute deviation allowed (C)
        min_horizon: Minimum horizon to consider (avoid degenerate cases)
        max_horizon: Maximum horizon to consider (optional)
        device: Device for output tensors
    
    Returns:
        tangents: Slope of the best-fit line at the optimal horizon
        horizons: Dynamic horizon where cumulative deviation ≈ max_accum_deviation
    """
    
    if y.dim() == 1:
        y = y.unsqueeze(0)
    
    batch_size, future_length = y.shape
    
    if max_horizon is None:
        max_horizon = future_length
    else:
        max_horizon = min(max_horizon, future_length)
    
    tangents = []
    horizons = []
    
    for b in range(batch_size):
        prices = y[b]
        
        # Pre-compute time indices
        t = torch.arange(max_horizon, dtype=torch.float32, device=prices.device)
        t_mean = t.mean()
        t_centered = t - t_mean
        
        best_tangent = 0.0
        best_horizon = min_horizon
        
        for h in range(min_horizon, max_horizon + 1):
            segment = prices[:h]
            t_h = t[:h]
            t_c_h = t_centered[:h]
            
            # Fit linear trend using OLS
            y_mean = segment.mean()
            numerator = torch.sum(t_c_h * (segment - y_mean))
            denominator = torch.sum(t_c_h ** 2)
            
            if denominator < 1e-10:
                m = 0.0
                b_val = y_mean
            else:
                m = numerator / denominator
                b_val = y_mean - m * t_h.mean()
            
            # Calculate cumulative absolute deviation
            predicted = m * t_h + b_val
            deviations = torch.abs(segment - predicted)
            cumulative_deviation = torch.sum(deviations)
            
            # Track this horizon if within budget
            if cumulative_deviation <= max_accum_deviation:
                best_tangent = m.item()
                best_horizon = h
            else:
                # Once we exceed the threshold, stop
                break
        
        tangents.append(best_tangent)
        horizons.append(best_horizon)
    
    return torch.tensor(tangents, device=device), torch.tensor(horizons, dtype=torch.long, device=device)


def generate_labels_dynamic_horizon_differentiable(
    y: torch.Tensor,
    max_accum_deviation: float,
    min_horizon: int = 2,
    max_horizon: Optional[int] = None,
    temperature: float = 1.0,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable version using soft selection for gradient flow.
    Enables end-to-end training with the labeling process.
    
    Args:
        y: Future price sequences, shape (batch_size, future_length)
        max_accum_deviation: Maximum cumulative absolute deviation allowed
        min_horizon: Minimum horizon to consider
        max_horizon: Maximum horizon to consider
        temperature: Temperature for softmax selection (higher = softer)
        device: Device for output tensors
    
    Returns:
        tangents: Smoothed slope, shape (batch_size,)
        horizons: Smoothed horizon, shape (batch_size,)
    """
    
    if y.dim() == 1:
        y = y.unsqueeze(0)
    
    batch_size, future_length = y.shape
    
    if max_horizon is None:
        max_horizon = future_length
    else:
        max_horizon = min(max_horizon, future_length)
    
    # Pre-compute time indices
    t = torch.arange(max_horizon, dtype=torch.float32, device=y.device)
    
    # Compute trends for all horizons at once
    all_tangents = []
    all_cum_devs = []
    all_intercepts = []
    
    for h in range(min_horizon, max_horizon + 1):
        t_h = t[:h]
        segment = y[:, :h]
        
        # Compute OLS for all batch items
        t_mean = t_h.mean()
        t_centered = t_h - t_mean
        y_mean = segment.mean(dim=1, keepdim=True)
        
        numerator = torch.sum(t_centered * (segment - y_mean), dim=1)
        denominator = torch.sum(t_centered ** 2)
        m = numerator / (denominator + 1e-10)
        
        # Compute intercept
        b_val = y_mean.squeeze(1) - m * t_mean
        
        # Compute cumulative deviation
        predicted = m.unsqueeze(1) * t_centered + y_mean
        deviations = torch.abs(segment - predicted)
        cum_dev = torch.sum(deviations, dim=1)
        
        all_tangents.append(m)
        all_intercepts.append(b_val)
        all_cum_devs.append(cum_dev)
    
    # Stack results
    all_tangents = torch.stack(all_tangents, dim=1)  # (batch, n_horizons)
    all_intercepts = torch.stack(all_intercepts, dim=1)  # (batch, n_horizons)
    all_cum_devs = torch.stack(all_cum_devs, dim=1)  # (batch, n_horizons)
    
    # Soft selection based on cumulative deviation
    threshold = torch.tensor(max_accum_deviation, device=y.device)
    
    # Compute selection weights with temperature
    # We want to select horizons where cum_dev <= threshold
    # But we also want smooth selection
    scores = -(all_cum_devs - threshold) / temperature
    weights = torch.softmax(scores, dim=1)
    
    # Weighted average for differentiable output
    tangents = torch.sum(weights * all_tangents, dim=1)
    horizons = torch.sum(
        weights * torch.arange(min_horizon, max_horizon + 1, device=y.device).float(),
        dim=1
    )
    
    return tangents, horizons


def generate_labels_with_multiple_thresholds(
    y: torch.Tensor,
    thresholds: List[float],
    min_horizon: int = 2,
    max_horizon: Optional[int] = None,
    device: str = 'cpu'
) -> Dict[float, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate labels for multiple deviation thresholds.
    Useful for creating multi-scale labels.
    
    Args:
        y: Future price sequences
        thresholds: List of max_accum_deviation values
        min_horizon: Minimum horizon
        max_horizon: Maximum horizon
        device: Device for output tensors
    
    Returns:
        Dictionary mapping threshold to (tangents, horizons)
    """
    results = {}
    for threshold in thresholds:
        tangents, horizons = generate_labels_dynamic_horizon(
            y=y,
            max_accum_deviation=threshold,
            min_horizon=min_horizon,
            max_horizon=max_horizon,
            device=device
        )
        results[threshold] = (tangents, horizons)
    return results


def calculate_label_statistics(
    tangents: torch.Tensor,
    horizons: torch.Tensor,
    future_prices: torch.Tensor,
    max_accum_deviation: float
) -> Dict[str, Any]:
    """
    Calculate statistics about generated labels for validation.
    
    Args:
        tangents: Tangent labels, shape (n_samples,)
        horizons: Horizon labels, shape (n_samples,)
        future_prices: Future price sequences, shape (n_samples, future_length)
        max_accum_deviation: Maximum cumulative deviation used
    
    Returns:
        Dictionary of statistics
    """
    stats_dict = {
        'tangent_mean': tangents.mean().item(),
        'tangent_std': tangents.std().item(),
        'tangent_min': tangents.min().item(),
        'tangent_max': tangents.max().item(),
        'horizon_mean': horizons.float().mean().item(),
        'horizon_std': horizons.float().std().item(),
        'horizon_min': horizons.min().item(),
        'horizon_max': horizons.max().item(),
        'horizon_distribution': {},
        'cumulative_deviation_mean': None,
        'cumulative_deviation_std': None,
        'samples_with_valid_horizon': 0
    }
    
    # Calculate cumulative deviations for each sample
    cum_devs = []
    valid_samples = 0
    
    for i in range(future_prices.shape[0]):
        h = horizons[i].item()
        if h > 0:
            segment = future_prices[i, :h]
            # Ensure segment is float for computations
            segment = segment.float()
            t = torch.arange(h, device=future_prices.device, dtype=torch.float32)
            m = tangents[i]
            t_mean = t.mean()
            t_centered = t - t_mean
            y_mean = segment.mean()
            predicted = m * t_centered + y_mean
            dev = torch.sum(torch.abs(segment - predicted)).item()
            cum_devs.append(dev)
            valid_samples += 1
    
    if cum_devs:
        stats_dict['cumulative_deviation_mean'] = np.mean(cum_devs)
        stats_dict['cumulative_deviation_std'] = np.std(cum_devs)
    
    stats_dict['samples_with_valid_horizon'] = valid_samples
    
    # Horizon distribution
    unique_h, counts = torch.unique(horizons, return_counts=True)
    for h, c in zip(unique_h.tolist(), counts.tolist()):
        stats_dict['horizon_distribution'][h] = c / len(horizons)
    
    return stats_dict


def find_optimal_threshold(
    future_prices: torch.Tensor,
    horizon_range: Tuple[int, int] = (5, 50),
    threshold_range: Tuple[float, float] = (5.0, 30.0),
    n_thresholds: int = 20,
    min_horizon: int = 3,
    device: str = 'cpu'
) -> Dict[float, Dict[str, float]]:
    """
    Find optimal threshold by analyzing horizon vs threshold relationship.
    
    Args:
        future_prices: Future price sequences, shape (n_samples, future_length)
        horizon_range: (min_horizon, max_horizon)
        threshold_range: (min_threshold, max_threshold)
        n_thresholds: Number of thresholds to test
        min_horizon: Minimum horizon
        device: Device for computation
    
    Returns:
        Dictionary mapping threshold to statistics
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
    results = {}
    
    for threshold in tqdm(thresholds, desc="Finding optimal threshold"):
        tangents, horizons = generate_labels_dynamic_horizon(
            y=future_prices,
            max_accum_deviation=threshold,
            min_horizon=min_horizon,
            max_horizon=horizon_range[1],
            device=device
        )
        
        # Calculate statistics
        mean_horizon = horizons.float().mean().item()
        std_horizon = horizons.float().std().item()
        mean_tangent = tangents.mean().item()
        std_tangent = tangents.std().item()
        
        # Calculate stability metrics
        results[threshold] = {
            'mean_horizon': mean_horizon,
            'std_horizon': std_horizon,
            'mean_tangent': mean_tangent,
            'std_tangent': std_tangent,
            'horizon_cv': std_horizon / (mean_horizon + 1e-6),  # Coefficient of variation
            'tangent_cv': std_tangent / (abs(mean_tangent) + 1e-6)
        }
    
    return results


# ============================================================================
# PART 2: ENHANCED DATA GENERATION
# ============================================================================

def generate_synthetic_data(
    history_length: int = 100,
    future_length: int = 50,
    trend_type: str = 'upward',
    noise_scale: float = 2.0,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic price data for demonstration.
    
    Args:
        history_length: Length of historical data
        future_length: Length of future data
        trend_type: 'upward', 'downward', 'sideways', 'volatile', 'mixed', 'reversal'
        noise_scale: Scale of random noise
        seed: Random seed
    
    Returns:
        historical_prices, future_prices
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate historical data (random walk)
    historical = np.cumsum(np.random.randn(history_length) * 0.5) + 100
    
    if trend_type == 'upward':
        future = np.cumsum(np.random.randn(future_length) * 0.5 + 0.3) + historical[-1]
    elif trend_type == 'downward':
        future = np.cumsum(np.random.randn(future_length) * 0.5 - 0.3) + historical[-1]
    elif trend_type == 'sideways':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
    elif trend_type == 'volatile':
        future = np.cumsum(np.random.randn(future_length) * 1.5) + historical[-1]
    elif trend_type == 'mixed':
        trend = np.sin(np.linspace(0, 2*np.pi, future_length)) * 5
        future = np.cumsum(np.random.randn(future_length) * 0.3) + trend + historical[-1]
    else:  # 'reversal'
        future = np.cumsum(np.random.randn(future_length) * 0.4 - 0.2) + historical[-1]
    
    return torch.tensor(historical, dtype=torch.float32), torch.tensor(future, dtype=torch.float32)


def generate_realistic_market_data(
    n_samples: int = 100,
    history_length: int = 100,
    future_length: int = 50,
    include_volume: bool = True,
    include_volatility: bool = True,
    seed: int = 42
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
           Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Generate more realistic market data with multiple features.
    
    Args:
        n_samples: Number of samples to generate
        history_length: Length of historical data
        future_length: Length of future data
        include_volume: Whether to include volume data
        include_volatility: Whether to include volatility estimates
        seed: Random seed
    
    Returns:
        Depending on include_volume:
            - (historical_prices, future_prices, volatility) if include_volume=False
            - (historical_prices, future_prices, volumes, volatility) if include_volume=True
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    prices = []
    volumes = []
    volatilities = []
    
    for i in range(n_samples):
        # Base price with random walk
        base_price = 100 + np.cumsum(np.random.randn(history_length + future_length) * 0.3)
        
        # Add trend component (random trend strength)
        trend_strength = np.random.uniform(-0.5, 0.5)
        trend = np.linspace(0, trend_strength * (history_length + future_length), 
                           history_length + future_length)
        
        # Add cyclical component
        cycle_freq = np.random.uniform(0.02, 0.1)
        cycle = 3 * np.sin(np.linspace(0, 4*np.pi, history_length + future_length) * cycle_freq)
        
        # Add regime changes
        if np.random.random() < 0.3:
            regime_idx = np.random.randint(history_length, history_length + future_length)
            regime_shift = np.random.uniform(-5, 5)
            base_price[regime_idx:] += regime_shift
        
        # Combine
        price = base_price + trend + cycle
        
        # Add GARCH-like volatility clustering
        if include_volatility:
            returns = np.diff(price, prepend=price[0])
            # Simple volatility model
            vol = 1.0 + 0.5 * np.abs(returns) * 2
            # Add some persistence
            vol = np.convolve(vol, np.ones(5)/5, mode='same')
            price += np.random.randn(history_length + future_length) * vol * 0.2
        
        prices.append(price)
        
        if include_volume:
            # Volume correlates with volatility and price changes
            returns = np.diff(price, prepend=price[0])
            volume_base = 1000 + np.abs(returns) * 100
            # Add some autocorrelation
            volume_noise = np.random.randn(history_length + future_length) * 100
            volume_noise = np.convolve(volume_noise, np.ones(3)/3, mode='same')
            volume = np.maximum(volume_base + volume_noise, 10)
            volumes.append(volume)
        
        # Volatility estimate (using future period)
        returns = np.diff(price, prepend=price[0])
        vol_estimate = np.std(returns[-future_length:])
        volatilities.append(vol_estimate)
    
    # Split into historical and future
    prices = np.array(prices)
    historical = prices[:, :history_length]
    future = prices[:, history_length:]
    
    # Convert to tensors
    historical = torch.tensor(historical, dtype=torch.float32)
    future = torch.tensor(future, dtype=torch.float32)
    volatility = torch.tensor(volatilities, dtype=torch.float32)
    
    if include_volume:
        volumes = torch.tensor(np.array(volumes), dtype=torch.float32)
        return historical, future, volumes, volatility
    else:
        return historical, future, volatility


def generate_batch_data(
    batch_size: int = 4,
    history_length: int = 80,
    future_length: int = 40,
    trend_types: Optional[List[str]] = None,
    noise_scale: float = 1.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of synthetic data with different trend types.
    
    Args:
        batch_size: Number of examples
        history_length: Length of historical data
        future_length: Length of future data
        trend_types: List of trend types
        noise_scale: Scale of random noise
    
    Returns:
        historical_batch, future_batch
    """
    if trend_types is None:
        trend_types = ['upward', 'downward', 'sideways', 'volatile', 'mixed', 'reversal']
        trend_types = trend_types[:batch_size]
    
    histories = []
    futures = []
    
    for i, trend_type in enumerate(trend_types):
        seed = hash(trend_type + str(i)) % 1000
        hist, fut = generate_synthetic_data(
            history_length=history_length,
            future_length=future_length,
            trend_type=trend_type,
            noise_scale=noise_scale,
            seed=seed
        )
        histories.append(hist)
        futures.append(fut)
    
    return torch.stack(histories), torch.stack(futures)


# ============================================================================
# PART 3: ENHANCED VISUALIZATION
# ============================================================================

def visualize_dynamic_horizon(
    historical_prices: torch.Tensor,
    future_prices: torch.Tensor,
    max_accum_deviation: float,
    min_horizon: int = 2,
    max_horizon: Optional[int] = None,
    title: str = "Dynamic Horizon Linear Trend",
    figsize: Tuple[int, int] = (18, 12),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, Dict]:
    """
    Enhanced visualization of the dynamic horizon selection process.
    
    Args:
        historical_prices: Past market data
        future_prices: Future price sequence
        max_accum_deviation: Maximum cumulative deviation allowed
        min_horizon: Minimum horizon to consider
        max_horizon: Maximum horizon to consider
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path
    
    Returns:
        Figure object and information dictionary
    """
    
    # Convert to numpy
    hist = historical_prices.cpu().numpy() if torch.is_tensor(historical_prices) else np.array(historical_prices)
    fut = future_prices.cpu().numpy() if torch.is_tensor(future_prices) else np.array(future_prices)
    
    if max_horizon is None:
        max_horizon = len(fut)
    else:
        max_horizon = min(max_horizon, len(fut))
    
    # Pre-compute trends and deviations for all horizons
    trends = []
    deviations = []
    cumulative_deviations = []
    slopes = []
    intercepts = []
    r_squareds = []
    residual_sum_squares = []
    
    for h in range(min_horizon, max_horizon + 1):
        x_vals = np.arange(h)
        y_vals = fut[:h]
        
        # Fit linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
        trend_line = slope * x_vals + intercept
        
        slopes.append(slope)
        intercepts.append(intercept)
        trends.append(trend_line)
        r_squareds.append(r_value ** 2)
        
        # Calculate deviations
        dev = np.abs(y_vals - trend_line)
        cum_dev = np.cumsum(dev)
        deviations.append(dev)
        cumulative_deviations.append(cum_dev[-1])
        residual_sum_squares.append(np.sum((y_vals - trend_line) ** 2))
    
    # Find the optimal horizon
    cum_devs = np.array(cumulative_deviations)
    valid_indices = np.where(cum_devs <= max_accum_deviation)[0]
    
    if len(valid_indices) > 0:
        optimal_idx = valid_indices[-1]
        optimal_horizon = optimal_idx + min_horizon
        optimal_slope = slopes[optimal_idx]
        optimal_intercept = intercepts[optimal_idx]
        optimal_trend = trends[optimal_idx]
        optimal_cum_dev = cum_devs[optimal_idx]
        optimal_r2 = r_squareds[optimal_idx]
        optimal_rss = residual_sum_squares[optimal_idx]
    else:
        optimal_horizon = min_horizon
        optimal_slope = slopes[0]
        optimal_intercept = intercepts[0]
        optimal_trend = trends[0]
        optimal_cum_dev = cum_devs[0]
        optimal_r2 = r_squareds[0]
        optimal_rss = residual_sum_squares[0]
    
    # Create enhanced figure with 5 subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # --- Plot 1: Full price history with optimal trend ---
    ax1 = fig.add_subplot(gs[0, :2])
    
    full_prices = np.concatenate([hist, fut])
    time_steps = np.arange(len(full_prices))
    split_idx = len(hist)
    
    # Plot historical
    ax1.plot(time_steps[:split_idx], hist, 'b-', linewidth=2, label='Historical', alpha=0.8)
    ax1.plot(time_steps[split_idx:], fut, 'g-', linewidth=2, label='Future', alpha=0.8)
    
    # Plot optimal trend
    trend_time = np.arange(optimal_horizon)
    trend_prices = optimal_slope * trend_time + fut[0]
    ax1.plot(
        time_steps[split_idx:split_idx + optimal_horizon],
        trend_prices,
        'r-', linewidth=3, label=f'Optimal Trend (m={optimal_slope:.4f})'
    )
    
    # Mark horizon boundary
    ax1.axvline(
        x=split_idx + optimal_horizon - 1,
        color='red', linestyle='-', linewidth=2, alpha=0.8,
        label=f'H = {optimal_horizon}'
    )
    
    # Fill valid region
    ax1.axvspan(
        split_idx, split_idx + optimal_horizon - 1,
        alpha=0.15, color='green', label='Valid Trend Region'
    )
    
    # Extended trend (invalid)
    if optimal_horizon < len(fut):
        extended_time = np.arange(optimal_horizon, len(fut))
        extended_prices = optimal_slope * extended_time + fut[0]
        ax1.plot(
            time_steps[split_idx + optimal_horizon:],
            extended_prices,
            'r:', linewidth=1.5, alpha=0.4, label='Invalid Extension'
        )
    
    ax1.axvline(x=split_idx - 0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.text(split_idx - 1, ax1.get_ylim()[1] * 0.95, 'NOW', 
             horizontalalignment='center', fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.set_title(f'Dynamic Horizon: H={optimal_horizon} (CumDev={optimal_cum_dev:.2f} ≤ {max_accum_deviation})', fontsize=13)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Cumulative Deviation vs Horizon ---
    ax2 = fig.add_subplot(gs[0, 2])
    
    horizons_range = np.arange(min_horizon, max_horizon + 1)
    ax2.plot(horizons_range, cum_devs, 'b-', linewidth=2, label='Cumulative Deviation')
    ax2.axhline(y=max_accum_deviation, color='red', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'MaxAccumDeviation = {max_accum_deviation}')
    
    # Mark optimal horizon
    ax2.scatter(optimal_horizon, optimal_cum_dev, color='red', s=100, zorder=5)
    ax2.axvline(x=optimal_horizon, color='red', linestyle=':', alpha=0.5)
    
    # Fill valid region
    ax2.fill_between(horizons_range, 0, cum_devs, 
                     where=(cum_devs <= max_accum_deviation),
                     alpha=0.2, color='green', label='Valid Horizons')
    
    ax2.set_xlabel('Horizon (h)', fontsize=12)
    ax2.set_ylabel('Cumulative Deviation', fontsize=12)
    ax2.set_title('Dynamic Horizon Selection', fontsize=13)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: R² vs Horizon ---
    ax3 = fig.add_subplot(gs[1, 0])
    
    ax3.plot(horizons_range, r_squareds, 'purple', linewidth=2)
    ax3.scatter(optimal_horizon, optimal_r2, color='red', s=80, zorder=5)
    ax3.axvline(x=optimal_horizon, color='red', linestyle=':', alpha=0.5)
    
    ax3.set_xlabel('Horizon (h)', fontsize=12)
    ax3.set_ylabel('R²', fontsize=12)
    ax3.set_title('Goodness of Fit', fontsize=13)
    ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Future with Deviation Bars ---
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Plot future prices
    ax4.plot(fut, 'g-', linewidth=2, label='Actual Future Prices', alpha=0.8)
    ax4.scatter(np.arange(len(fut)), fut, color='green', s=20, alpha=0.5)
    
    # Plot optimal trend
    if optimal_horizon > 0:
        ax4.plot(trend_time, trend_prices, 'r--', linewidth=2.5, 
                label=f'Trend: m={optimal_slope:.4f}')
        
        # Show deviation bars
        x_vals = np.arange(optimal_horizon)
        y_vals = fut[:optimal_horizon]
        trend_vals = optimal_slope * x_vals + fut[0]
        deviations_vals = np.abs(y_vals - trend_vals)
        
        # Individual deviation bars
        ax4.bar(x_vals, deviations_vals, bottom=np.minimum(y_vals, trend_vals),
                alpha=0.3, color='orange', width=0.6, label='Deviations')
        
        # Mark horizon
        ax4.axvline(x=optimal_horizon - 1, color='red', linestyle='-', 
                   linewidth=2, alpha=0.7)
        
        # Fill valid region
        ax4.axvspan(0, optimal_horizon - 1, alpha=0.1, color='green')
    
    ax4.set_xlabel('Future Time Steps', fontsize=12)
    ax4.set_ylabel('Price', fontsize=12)
    ax4.set_title('Future Prices with Deviation Bars', fontsize=13)
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # --- Plot 5: Per-Step Deviation Analysis ---
    ax5 = fig.add_subplot(gs[1, 2])
    
    if optimal_horizon > 0:
        x_vals = np.arange(optimal_horizon)
        y_vals = fut[:optimal_horizon]
        trend_vals = optimal_slope * x_vals + fut[0]
        per_step_dev = np.abs(y_vals - trend_vals)
        
        # Bar chart of per-step deviations
        bars = ax5.bar(x_vals, per_step_dev, alpha=0.6, color='steelblue')
        
        # Color bars that exceed average by 1.5x
        avg_dev = per_step_dev.mean()
        for i, bar in enumerate(bars):
            if per_step_dev[i] > avg_dev * 1.5:
                bar.set_color('red')
                bar.set_alpha(0.8)
        
        ax5.axhline(y=avg_dev, color='red', linestyle='--', 
                   label=f'Avg Dev = {avg_dev:.3f}')
        ax5.axhline(y=avg_dev * 1.5, color='orange', linestyle=':', 
                   label=f'1.5× Avg Dev')
        
        ax5.set_xlabel('Time Step', fontsize=12)
        ax5.set_ylabel('Per-Step Deviation', fontsize=12)
        ax5.set_title(f'Per-Step Deviations (Total: {optimal_cum_dev:.2f})', fontsize=13)
        ax5.legend(loc='best', fontsize=10)
        ax5.grid(True, alpha=0.3)
    
    # --- Bottom: Metrics Summary ---
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Create metrics table
    metrics_text = f"""
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                          LABELING METRICS SUMMARY                          │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │ Optimal Horizon (H)    : {optimal_horizon:>4}                               │
    │ Tangent (Slope)        : {optimal_slope:>10.6f}                            │
    │ Intercept (b)          : {optimal_intercept:>10.3f}                        │
    │ R²                     : {optimal_r2:>10.4f}                               │
    │ Max Accum Deviation    : {max_accum_deviation:>10.2f}                      │
    │ Actual Accum Deviation : {optimal_cum_dev:>10.2f}                          │
    │ Residual Sum Squares   : {optimal_rss:>10.2f}                              │
    │ Min Horizon            : {min_horizon:>4}                                  │
    │ Max Horizon            : {max_horizon:>4}                                  │
    │ Valid Horizons Found   : {len(valid_indices):>4} out of {len(horizons_range)}│
    └─────────────────────────────────────────────────────────────────────────────┘
    """
    
    ax6.text(0.5, 0.5, metrics_text, fontsize=11, verticalalignment='center',
             horizontalalignment='center', family='monospace', transform=ax6.transAxes)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    info_dict = {
        'optimal_horizon': optimal_horizon,
        'optimal_slope': optimal_slope,
        'optimal_intercept': optimal_intercept,
        'optimal_cum_dev': optimal_cum_dev,
        'optimal_r2': optimal_r2,
        'optimal_rss': optimal_rss,
        'all_horizons': horizons_range,
        'all_cum_devs': cum_devs,
        'all_slopes': np.array(slopes),
        'all_r2': np.array(r_squareds),
        'all_rss': np.array(residual_sum_squares)
    }
    
    return fig, info_dict


def visualize_batch_with_trends(
    historical_batch: torch.Tensor,
    future_batch: torch.Tensor,
    tangents: torch.Tensor,
    horizons: torch.Tensor,
    max_accum_deviation: float,
    n_examples: int = 4,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Visualize multiple examples from a batch with their trends.
    
    Args:
        historical_batch: Batch of historical prices, shape (batch_size, history_length)
        future_batch: Batch of future prices, shape (batch_size, future_length)
        tangents: Batch of tangents, shape (batch_size,)
        horizons: Batch of horizons, shape (batch_size,)
        max_accum_deviation: Maximum cumulative deviation
        n_examples: Number of examples to show
        figsize: Figure size
    
    Returns:
        Figure object
    """
    batch_size = min(n_examples, historical_batch.shape[0])
    
    # Calculate grid size
    cols = min(2, batch_size)
    rows = (batch_size + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single subplot case
    if batch_size == 1:
        axes = np.array([axes])
    elif rows == 1 and cols > 1:
        # axes is already a 1D array
        pass
    else:
        axes = axes.flatten()
    
    for i in range(batch_size):
        ax = axes[i]
        
        # Extract data
        hist = historical_batch[i].cpu().numpy()
        fut = future_batch[i].cpu().numpy()
        tangent = tangents[i].item()
        horizon = horizons[i].item()
        
        # Combine
        full = np.concatenate([hist, fut])
        time = np.arange(len(full))
        split_idx = len(hist)
        
        # Plot
        ax.plot(time[:split_idx], hist, 'b-', linewidth=1.5, alpha=0.7)
        ax.plot(time[split_idx:], fut, 'g-', linewidth=1.5, alpha=0.7)
        
        # Trend
        if horizon > 0:
            trend_time = np.arange(horizon)
            trend_prices = tangent * trend_time + fut[0]
            ax.plot(
                time[split_idx:split_idx + horizon],
                trend_prices,
                'r-', linewidth=2.5, label=f'm={tangent:.4f}'
            )
            
            # Mark horizon
            ax.axvline(x=split_idx + horizon - 1, color='red', 
                      linestyle='-', linewidth=1.5, alpha=0.6)
        
        ax.axvline(x=split_idx - 0.5, color='black', linestyle='--', 
                  linewidth=1, alpha=0.4)
        
        ax.set_title(f'Example {i+1}: m={tangent:.4f}, h={horizon}', fontsize=11)
        ax.grid(True, alpha=0.2)
        
        # Add info
        ax.text(0.02, 0.98, f'h={horizon}', transform=ax.transAxes,
               fontsize=9, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Hide empty subplots
    for i in range(batch_size, rows * cols):
        axes[i].set_visible(False)
    
    fig.suptitle(f'Batch Examples with Linear Trends (MaxAccumDev={max_accum_deviation})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig


def visualize_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Visualize training history with enhanced plots.
    
    Args:
        history: Dictionary with training metrics
        figsize: Figure size
    
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Total loss
    axes[0].plot(history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    if 'val_total_loss' in history and history['val_total_loss']:
        axes[0].plot(history['val_total_loss'], 'b--', linewidth=2, label='Val Total Loss')
    axes[0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Tangent loss
    axes[1].plot(history['tangent_loss'], 'r-', linewidth=2, label='Tangent Loss')
    if 'val_tangent_loss' in history and history['val_tangent_loss']:
        axes[1].plot(history['val_tangent_loss'], 'r--', linewidth=2, label='Val Tangent Loss')
    axes[1].set_title('Tangent Loss', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Horizon loss
    axes[2].plot(history['horizon_loss'], 'g-', linewidth=2, label='Horizon Loss')
    if 'val_horizon_loss' in history and history['val_horizon_loss']:
        axes[2].plot(history['val_horizon_loss'], 'g--', linewidth=2, label='Val Horizon Loss')
    axes[2].set_title('Horizon Loss', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.suptitle('Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig


def visualize_threshold_analysis(
    results: Dict[float, Dict[str, float]],
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Visualize threshold analysis results.
    
    Args:
        results: Dictionary from find_optimal_threshold
        figsize: Figure size
    
    Returns:
        Figure object
    """
    thresholds = sorted(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Horizon vs Threshold
    ax1 = axes[0, 0]
    mean_horizons = [results[t]['mean_horizon'] for t in thresholds]
    std_horizons = [results[t]['std_horizon'] for t in thresholds]
    
    ax1.plot(thresholds, mean_horizons, 'b-', linewidth=2, label='Mean Horizon')
    ax1.fill_between(thresholds, 
                     np.array(mean_horizons) - np.array(std_horizons),
                     np.array(mean_horizons) + np.array(std_horizons),
                     alpha=0.2, color='blue')
    ax1.set_xlabel('Threshold (Max Accum Deviation)', fontsize=11)
    ax1.set_ylabel('Horizon', fontsize=11)
    ax1.set_title('Horizon vs Threshold', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Tangent vs Threshold
    ax2 = axes[0, 1]
    mean_tangents = [results[t]['mean_tangent'] for t in thresholds]
    std_tangents = [results[t]['std_tangent'] for t in thresholds]
    
    ax2.plot(thresholds, mean_tangents, 'r-', linewidth=2, label='Mean Tangent')
    ax2.fill_between(thresholds,
                     np.array(mean_tangents) - np.array(std_tangents),
                     np.array(mean_tangents) + np.array(std_tangents),
                     alpha=0.2, color='red')
    ax2.set_xlabel('Threshold (Max Accum Deviation)', fontsize=11)
    ax2.set_ylabel('Tangent', fontsize=11)
    ax2.set_title('Tangent vs Threshold', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Horizon CV vs Threshold
    ax3 = axes[1, 0]
    horizon_cv = [results[t]['horizon_cv'] for t in thresholds]
    
    ax3.plot(thresholds, horizon_cv, 'g-', linewidth=2)
    ax3.scatter(thresholds, horizon_cv, color='green', s=30)
    ax3.set_xlabel('Threshold (Max Accum Deviation)', fontsize=11)
    ax3.set_ylabel('Horizon CV (Std/Mean)', fontsize=11)
    ax3.set_title('Horizon Stability', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Tangent CV vs Threshold
    ax4 = axes[1, 1]
    tangent_cv = [results[t]['tangent_cv'] for t in thresholds]
    
    ax4.plot(thresholds, tangent_cv, 'purple', linewidth=2)
    ax4.scatter(thresholds, tangent_cv, color='purple', s=30)
    ax4.set_xlabel('Threshold (Max Accum Deviation)', fontsize=11)
    ax4.set_ylabel('Tangent CV (Std/Mean)', fontsize=11)
    ax4.set_title('Tangent Stability', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Threshold Analysis for Dynamic Horizon Labeling', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig


# ============================================================================
# PART 4: FEATURE ENGINEERING
# ============================================================================

def extract_features(
    historical: torch.Tensor,
    include_technical: bool = True,
    include_statistical: bool = True,
    include_trend: bool = True
) -> torch.Tensor:
    """
    Extract meaningful features from historical data.
    
    Args:
        historical: Historical prices, shape (batch_size, history_length)
        include_technical: Include technical indicators
        include_statistical: Include statistical features
        include_trend: Include trend features
    
    Returns:
        Features tensor, shape (batch_size, n_features)
    """
    features = []
    batch_size, history_length = historical.shape
    device = historical.device
    
    # Ensure historical is float
    historical = historical.float()
    
    if include_statistical:
        # Basic statistics
        features.append(historical.mean(dim=1, keepdim=True))
        features.append(historical.std(dim=1, keepdim=True))
        
        # Max and min - extract values only
        max_vals, _ = historical.max(dim=1, keepdim=True)
        min_vals, _ = historical.min(dim=1, keepdim=True)
        features.append(max_vals)
        features.append(min_vals)
        features.append((max_vals - min_vals))
        
        # Returns statistics
        returns = torch.diff(historical, dim=1)
        features.append(returns.mean(dim=1, keepdim=True))
        features.append(returns.std(dim=1, keepdim=True))
        
        # Skewness and kurtosis (approximated with torch)
        returns_centered = returns - returns.mean(dim=1, keepdim=True)
        var = (returns_centered ** 2).mean(dim=1, keepdim=True) + 1e-10
        skewness = (returns_centered ** 3).mean(dim=1, keepdim=True) / (var ** 1.5)
        features.append(skewness.clamp(-5, 5))
        
        # Kurtosis: E[(X-μ)^4]/σ^4 - 3
        kurt = (returns_centered ** 4).mean(dim=1, keepdim=True) / (var ** 2) - 3
        features.append(kurt.clamp(-5, 5))
    
    if include_trend:
        # Simple OLS slope
        t = torch.arange(history_length, device=device, dtype=torch.float32)
        t_centered = t - t.mean()
        returns = torch.diff(historical, dim=1)
        
        # Cumulative returns trend
        cum_returns = torch.cumsum(returns, dim=1)
        slope_numerator = torch.sum(t_centered[:history_length-1] * cum_returns, dim=1)
        slope_denominator = torch.sum(t_centered[:history_length-1] ** 2)
        slope = slope_numerator / (slope_denominator + 1e-10)
        features.append(slope.unsqueeze(1))
        
        # Moving average crossover (short vs long)
        short_window = max(5, history_length // 20)
        long_window = max(20, history_length // 10)
        
        # Simple moving averages using cumulative sum
        cumsum = torch.cumsum(historical, dim=1)
        
        # Handle edge cases where window might be too large
        if short_window < history_length:
            short_ma = (cumsum[:, short_window:] - cumsum[:, :-short_window]) / short_window
            ma_diff = short_ma[:, -1]
        else:
            ma_diff = torch.zeros(batch_size, device=device)
        
        if long_window < history_length:
            long_ma = (cumsum[:, long_window:] - cumsum[:, :-long_window]) / long_window
            ma_diff = ma_diff - long_ma[:, -1]
        
        features.append(ma_diff.unsqueeze(1))
        
        # Rate of change
        roc = (historical[:, -1] - historical[:, 0]) / (historical[:, 0] + 1e-10)
        features.append(roc.unsqueeze(1))
    
    if include_technical:
        # RSI approximation
        returns = torch.diff(historical, dim=1)
        gains = torch.clamp(returns, min=0)
        losses = torch.clamp(-returns, min=0)
        
        period = min(14, history_length - 1)
        if period > 0:
            avg_gain = gains[:, -period:].mean(dim=1)
            avg_loss = losses[:, -period:].mean(dim=1) + 1e-10
            rsi = 100 - 100 / (1 + avg_gain / avg_loss)
            features.append(rsi.unsqueeze(1))
        
        # Price position in recent range
        if period > 0:
            recent_min, _ = historical[:, -period:].min(dim=1)
            recent_max, _ = historical[:, -period:].max(dim=1)
            price_position = (historical[:, -1] - recent_min) / (recent_max - recent_min + 1e-10)
            features.append(price_position.unsqueeze(1))
    
    # If no features were added, return the original data
    if len(features) == 0:
        return historical
    
    return torch.cat(features, dim=1)

# ============================================================================
# PART 5: ENHANCED NEURAL NETWORK MODELS
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out + residual)
        return out


class TrendPredictor(nn.Module):
    """
    Neural network that predicts tangent and horizon from historical data.
    Enhanced with residual connections and batch normalization.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.use_residual = use_residual
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers with residual connections
        for i in range(num_layers - 1):
            if use_residual:
                layers.append(ResidualBlock(hidden_dim, dropout))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*layers)
        
        # Tangent output (regression)
        self.tangent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Horizon output (regression, positive values)
        self.horizon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive horizon
        )
        
        # Optional: Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
            nn.Softplus()  # Positive variances
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], 
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Historical market data, shape (batch_size, input_dim)
            return_uncertainty: Whether to return uncertainty estimates
        
        Returns:
            If return_uncertainty=False:
                tangent: Predicted slope, shape (batch_size,)
                horizon: Predicted horizon, shape (batch_size,)
            If return_uncertainty=True:
                tangent: Predicted slope, shape (batch_size,)
                horizon: Predicted horizon, shape (batch_size,)
                tangent_var: Variance of tangent prediction
                horizon_var: Variance of horizon prediction
        """
        features = self.encoder(x)
        tangent = self.tangent_head(features).squeeze(-1)
        horizon = self.horizon_head(features).squeeze(-1)
        
        if return_uncertainty:
            uncertainty = self.uncertainty_head(features)
            tangent_var = uncertainty[:, 0]
            horizon_var = uncertainty[:, 1]
            return tangent, horizon, tangent_var, horizon_var
        
        return tangent, horizon


class MultiScaleTrendPredictor(nn.Module):
    """
    Multi-scale trend predictor that processes data at different time scales.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_scales: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        # Scale-specific encoders
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim // (2 ** i), hidden_dim // (2 ** i)),
                nn.BatchNorm1d(hidden_dim // (2 ** i)),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_scales)
        ])
        
        # Attention mechanism for combining scales
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Final prediction layers
        self.tangent_head = nn.Linear(hidden_dim, 1)
        self.horizon_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-scale processing.
        
        Args:
            x: Input data, shape (batch_size, input_dim)
        
        Returns:
            tangent: Predicted slope
            horizon: Predicted horizon
        """
        batch_size = x.shape[0]
        
        # Process at different scales
        scale_features = []
        for i, encoder in enumerate(self.scale_encoders):
            # Downsample for coarser scales
            if i > 0:
                scale_input = F.avg_pool1d(
                    x.unsqueeze(1), 
                    kernel_size=2**i, 
                    stride=2**i
                ).squeeze(1)
            else:
                scale_input = x
            
            # Encode
            features = encoder(scale_input)
            
            # Pad to same dimension if needed
            if features.shape[1] < self.scale_encoders[0][0].out_features:
                pad_size = self.scale_encoders[0][0].out_features - features.shape[1]
                features = F.pad(features, (0, pad_size))
            elif features.shape[1] > self.scale_encoders[0][0].out_features:
                features = features[:, :self.scale_encoders[0][0].out_features]
            
            scale_features.append(features)
        
        # Stack features for attention
        stacked = torch.stack(scale_features, dim=1)  # (batch, num_scales, hidden_dim)
        
        # Apply attention
        attended, _ = self.attention(stacked, stacked, stacked)
        attended = attended.mean(dim=1)  # Average over scales
        
        # Final predictions
        tangent = self.tangent_head(attended).squeeze(-1)
        horizon = self.horizon_head(attended).squeeze(-1)
        
        return tangent, horizon


# ============================================================================
# PART 6: TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss: float, model: Optional[nn.Module] = None):
        if self.best_loss is None:
            self.best_loss = val_loss
            if model is not None:
                self.best_model_state = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None:
                self.best_model_state = model.state_dict()


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    config: TrainingConfig = None,
    device: str = 'cpu'
) -> Dict[str, List[float]]:
    """
    Enhanced training function with validation and early stopping.
    
    Args:
        model: Neural network model
        train_loader: DataLoader with (x, tangent_labels, horizon_labels)
        val_loader: Optional validation DataLoader
        config: Training configuration
        device: Device for training
    
    Returns:
        Dictionary of training metrics
    """
    if config is None:
        config = TrainingConfig()
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=config.scheduler_patience, 
        factor=config.scheduler_factor
    )
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    
    history = {
        'total_loss': [],
        'tangent_loss': [],
        'horizon_loss': [],
        'val_total_loss': [],
        'val_tangent_loss': [],
        'val_horizon_loss': [],
        'learning_rates': []
    }
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_tangent_loss = 0.0
        epoch_horizon_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}', leave=False):
            x, tangent_labels, horizon_labels = batch
            x = x.to(device)
            tangent_labels = tangent_labels.to(device)
            horizon_labels = horizon_labels.to(device)
            
            # Forward pass
            pred_tangent, pred_horizon = model(x)
            
            # Calculate losses
            tangent_loss = nn.MSELoss()(pred_tangent, tangent_labels)
            horizon_loss = nn.MSELoss()(pred_horizon, horizon_labels.float())
            total_loss = config.tangent_weight * tangent_loss + config.horizon_weight * horizon_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            
            # Track metrics
            epoch_loss += total_loss.item()
            epoch_tangent_loss += tangent_loss.item()
            epoch_horizon_loss += horizon_loss.item()
            num_batches += 1
        
        # Average training losses
        avg_loss = epoch_loss / num_batches
        avg_tangent_loss = epoch_tangent_loss / num_batches
        avg_horizon_loss = epoch_horizon_loss / num_batches
        
        history['total_loss'].append(avg_loss)
        history['tangent_loss'].append(avg_tangent_loss)
        history['horizon_loss'].append(avg_horizon_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_tangent_loss = 0.0
            val_horizon_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    x, tangent_labels, horizon_labels = batch
                    x = x.to(device)
                    tangent_labels = tangent_labels.to(device)
                    horizon_labels = horizon_labels.to(device)
                    
                    pred_tangent, pred_horizon = model(x)
                    
                    tangent_loss = nn.MSELoss()(pred_tangent, tangent_labels)
                    horizon_loss = nn.MSELoss()(pred_horizon, horizon_labels.float())
                    total_loss = config.tangent_weight * tangent_loss + config.horizon_weight * horizon_loss
                    
                    val_loss += total_loss.item()
                    val_tangent_loss += tangent_loss.item()
                    val_horizon_loss += horizon_loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            avg_val_tangent_loss = val_tangent_loss / val_batches
            avg_val_horizon_loss = val_horizon_loss / val_batches
            
            history['val_total_loss'].append(avg_val_loss)
            history['val_tangent_loss'].append(avg_val_tangent_loss)
            history['val_horizon_loss'].append(avg_val_horizon_loss)
            
            # Early stopping
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                # Load best model
                if early_stopping.best_model_state is not None:
                    model.load_state_dict(early_stopping.best_model_state)
                break
            
            # Scheduler
            scheduler.step(avg_val_loss)
        else:
            scheduler.step(avg_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            val_str = f", Val={avg_val_loss:.4f}" if val_loader is not None else ""
            print(f"Epoch {epoch+1}/{config.num_epochs}: Loss={avg_loss:.4f}, "
                  f"Tangent={avg_tangent_loss:.4f}, Horizon={avg_horizon_loss:.4f}{val_str}")
    
    return history


def create_labeled_dataset(
    historical_data: torch.Tensor,
    future_data: torch.Tensor,
    max_accum_deviation: float,
    min_horizon: int = 2,
    max_horizon: Optional[int] = None,
    differentiable: bool = False,
    temperature: float = 1.0,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a labeled dataset for training.
    
    Args:
        historical_data: Historical prices, shape (n_samples, history_length)
        future_data: Future prices, shape (n_samples, future_length)
        max_accum_deviation: Maximum cumulative deviation
        min_horizon: Minimum horizon
        max_horizon: Maximum horizon
        differentiable: Use differentiable labeling
        temperature: Temperature for differentiable labeling
        device: Device for output tensors
    
    Returns:
        x: Historical data, shape (n_samples, history_length)
        tangents: Tangent labels, shape (n_samples,)
        horizons: Horizon labels, shape (n_samples,)
    """
    if differentiable:
        tangents, horizons = generate_labels_dynamic_horizon_differentiable(
            y=future_data,
            max_accum_deviation=max_accum_deviation,
            min_horizon=min_horizon,
            max_horizon=max_horizon,
            temperature=temperature,
            device=device
        )
    else:
        tangents, horizons = generate_labels_dynamic_horizon(
            y=future_data,
            max_accum_deviation=max_accum_deviation,
            min_horizon=min_horizon,
            max_horizon=max_horizon,
            device=device
        )
    
    return historical_data, tangents, horizons


def process_large_dataset(
    data_loader: torch.utils.data.DataLoader,
    max_accum_deviation: float,
    min_horizon: int = 2,
    max_horizon: Optional[int] = None,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process large datasets in batches to avoid memory issues.
    
    Args:
        data_loader: DataLoader with (historical, future) or just future
        max_accum_deviation: Maximum cumulative deviation
        min_horizon: Minimum horizon
        max_horizon: Maximum horizon
        device: Device for computation
    
    Returns:
        tangents: Tangent labels, shape (n_samples,)
        horizons: Horizon labels, shape (n_samples,)
    """
    all_tangents = []
    all_horizons = []
    
    for batch in tqdm(data_loader, desc="Processing batches"):
        # Assuming batch contains (historical, future)
        if isinstance(batch, (tuple, list)):
            _, future = batch[0], batch[1]
        else:
            future = batch
        
        # Move to device
        if torch.is_tensor(future):
            future = future.to(device)
        
        tangents, horizons = generate_labels_dynamic_horizon(
            y=future,
            max_accum_deviation=max_accum_deviation,
            min_horizon=min_horizon,
            max_horizon=max_horizon,
            device=device
        )
        
        all_tangents.append(tangents.cpu())
        all_horizons.append(horizons.cpu())
    
    return torch.cat(all_tangents), torch.cat(all_horizons)


# ============================================================================
# PART 7: MAIN DEMONSTRATION
# ============================================================================

def main_demo():
    """
    Complete demonstration of the enhanced pipeline.
    """
    print("=" * 80)
    print("ENHANCED DYNAMIC HORIZON LINEAR TREND LABELING - COMPLETE DEMONSTRATION")
    print("=" * 80)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # ========== PART 1: Single Example Visualization ==========
    print("\n1. Generating single example with enhanced visualization...")
    print("-" * 50)
    
    # Generate data
    history, future = generate_synthetic_data(
        history_length=100,
        future_length=50,
        trend_type='upward',
        noise_scale=1.5
    )
    
    # Set parameters
    max_accum_deviation = 15.0
    min_horizon = 3
    max_horizon = 45
    
    # Generate labels
    tangents, horizons = generate_labels_dynamic_horizon(
        y=future.unsqueeze(0),
        max_accum_deviation=max_accum_deviation,
        min_horizon=min_horizon,
        max_horizon=max_horizon
    )
    
    print(f"Generated labels:")
    print(f"  Tangent: {tangents[0].item():.6f}")
    print(f"  Horizon: {horizons[0].item()}")
    print(f"  Max Accum Deviation: {max_accum_deviation}")
    
    # Visualize
    fig, info = visualize_dynamic_horizon(
        historical_prices=history,
        future_prices=future,
        max_accum_deviation=max_accum_deviation,
        min_horizon=min_horizon,
        max_horizon=max_horizon,
        title="Enhanced Visualization: Upward Trend with Dynamic Horizon"
    )
    
    # ========== PART 2: Batch Visualization ==========
    print("\n2. Generating batch with multiple trend types...")
    print("-" * 50)
    
    # Generate batch data
    trend_types = ['upward', 'downward', 'sideways', 'volatile']
    hist_batch, fut_batch = generate_batch_data(
        batch_size=4,
        history_length=80,
        future_length=40,
        trend_types=trend_types
    )
    
    # Generate labels for batch
    tangents_batch, horizons_batch = generate_labels_dynamic_horizon(
        y=fut_batch,
        max_accum_deviation=12.0,
        min_horizon=3,
        max_horizon=35
    )
    
    print("Generated batch labels:")
    for i, (tan, hor) in enumerate(zip(tangents_batch, horizons_batch)):
        print(f"  Example {i+1} ({trend_types[i]}): tangent={tan.item():.4f}, horizon={hor.item()}")
    
    # Visualize batch
    fig_batch = visualize_batch_with_trends(
        historical_batch=hist_batch,
        future_batch=fut_batch,
        tangents=tangents_batch,
        horizons=horizons_batch,
        max_accum_deviation=12.0,
        n_examples=4
    )
    
    # ========== PART 3: Multiple Thresholds ==========
    print("\n3. Analyzing effect of different thresholds...")
    print("-" * 50)
    
    thresholds = [5.0, 10.0, 15.0, 20.0]
    results = generate_labels_with_multiple_thresholds(
        y=future.unsqueeze(0),
        thresholds=thresholds,
        min_horizon=3,
        max_horizon=45
    )
    
    print("Threshold effects:")
    for threshold, (tans, hors) in results.items():
        print(f"  Threshold {threshold}: horizon={hors[0].item()}, tangent={tans[0].item():.6f}")
    
    # Plot threshold comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    horizons_for_threshold = [hors[0].item() for hors in results.values()]
    tangents_for_threshold = [tans[0].item() for tans in results.values()]
    
    ax.plot(thresholds, horizons_for_threshold, 'bo-', linewidth=2, markersize=10, label='Horizon')
    ax.set_xlabel('Max Accum Deviation (C)', fontsize=12)
    ax.set_ylabel('Horizon', fontsize=12, color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.grid(True, alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(thresholds, tangents_for_threshold, 'ro-', linewidth=2, markersize=10, label='Tangent')
    ax2.set_ylabel('Tangent', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax.set_title('Effect of Deviation Threshold on Horizon and Tangent', fontsize=14)
    plt.show()
    
    # ========== PART 4: Label Statistics ==========
    print("\n4. Computing label statistics...")
    print("-" * 50)
    
    stats = calculate_label_statistics(
        tangents=tangents_batch,
        horizons=horizons_batch,
        future_prices=fut_batch,
        max_accum_deviation=12.0
    )
    
    print("Label Statistics:")
    print(f"  Tangent: mean={stats['tangent_mean']:.4f}, std={stats['tangent_std']:.4f}")
    print(f"  Horizon: mean={stats['horizon_mean']:.1f}, std={stats['horizon_std']:.1f}")
    print(f"  Cumulative Deviation: mean={stats['cumulative_deviation_mean']:.2f}, "
          f"std={stats['cumulative_deviation_std']:.2f}")
    print(f"  Valid samples: {stats['samples_with_valid_horizon']}/{len(tangents_batch)}")
    
    # ========== PART 5: Feature Engineering ==========
    print("\n5. Feature engineering demonstration...")
    print("-" * 50)
    
    features = extract_features(
        historical=hist_batch,
        include_technical=True,
        include_statistical=True,
        include_trend=True
    )
    
    print(f"Feature shape: {features.shape}")
    print(f"Feature statistics:")
    print(f"  Mean: {features.mean(dim=0).tolist()[:5]}...")  # Show first 5
    print(f"  Std: {features.std(dim=0).tolist()[:5]}...")
    
    # ========== PART 6: Training Example ==========
    print("\n6. Training enhanced neural network...")
    print("-" * 50)
    
    # Generate training data
    n_samples = 200  # Reduced for faster demo
    history_length = 50
    future_length = 30
    
    print("Generating training data...")
    all_hist = []
    all_fut = []
    
    for i in range(n_samples):
        trend_type = np.random.choice(['upward', 'downward', 'sideways', 'volatile', 'mixed'])
        seed = i * 42 + 7
        hist, fut = generate_synthetic_data(
            history_length=history_length,
            future_length=future_length,
            trend_type=trend_type,
            noise_scale=np.random.uniform(0.5, 2.0),
            seed=seed
        )
        all_hist.append(hist)
        all_fut.append(fut)
    
    X_train = torch.stack(all_hist)
    Y_train = torch.stack(all_fut)
    
    # Create labels
    max_dev = 10.0
    tangents, horizons = generate_labels_dynamic_horizon(
        y=Y_train,
        max_accum_deviation=max_dev,
        min_horizon=3,
        max_horizon=future_length
    )
    
    # Split into train/val
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X_train[:split_idx], X_train[split_idx:]
    Y_train, Y_val = Y_train[:split_idx], Y_train[split_idx:]
    tangents_train, tangents_val = tangents[:split_idx], tangents[split_idx:]
    horizons_train, horizons_val = horizons[:split_idx], horizons[split_idx:]
    
    # Extract features
    print("Extracting features...")
    X_train_features = extract_features(X_train)
    X_val_features = extract_features(X_val)
    
    print(f"Training data shapes:")
    print(f"  X_train: {X_train_features.shape}")
    print(f"  Y_train: {Y_train.shape}")
    print(f"  Tangents: {tangents_train.shape}")
    print(f"  Horizons: {horizons_train.shape}")
    
    # Create model
    model = MultiScaleTrendPredictor(
        input_dim=X_train_features.shape[1],
        hidden_dim=128,
        num_scales=3,
        dropout=0.1
    )
    
    print(f"\nModel architecture: {model.__class__.__name__}")
    print(f"  Input dimension: {X_train_features.shape[1]}")
    print(f"  Hidden dimension: 128")
    print(f"  Number of scales: 3")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(
        X_train_features, tangents_train, horizons_train.float()
    )
    val_dataset = torch.utils.data.TensorDataset(
        X_val_features, tangents_val, horizons_val.float()
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False
    )
    
    # Train model
    config = TrainingConfig(
        num_epochs=30,  # Reduced for demo
        learning_rate=1e-3,
        patience=10
    )
    
    print("\nTraining...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device='cpu'
    )
    
    # Plot training history
    fig_train = visualize_training_history(history)
    
    # ========== PART 7: Model Evaluation ==========
    print("\n7. Model evaluation on test examples...")
    print("-" * 50)
    
    model.eval()
    
    # Generate test examples
    test_hist, test_fut = generate_synthetic_data(
        history_length=history_length,
        future_length=future_length,
        trend_type='upward',
        noise_scale=1.0,
        seed=999
    )
    
    # Extract features
    test_features = extract_features(test_hist.unsqueeze(0))
    
    # Get true labels
    true_tangent, true_horizon = generate_labels_dynamic_horizon(
        y=test_fut.unsqueeze(0),
        max_accum_deviation=max_dev,
        min_horizon=3,
        max_horizon=future_length
    )
    
    # Model prediction
    with torch.no_grad():
        pred_tangent, pred_horizon = model(test_features)
    
    print(f"Test Example (Upward Trend):")
    print(f"  True:  tangent={true_tangent[0].item():.4f}, horizon={true_horizon[0].item()}")
    print(f"  Pred:  tangent={pred_tangent[0].item():.4f}, horizon={pred_horizon[0].item():.1f}")
    
    # Visualize the test example
    fig, info = visualize_dynamic_horizon(
        historical_prices=test_hist,
        future_prices=test_fut,
        max_accum_deviation=max_dev,
        min_horizon=3,
        max_horizon=future_length,
        title=f"Test Example: True (m={true_tangent[0].item():.4f}, h={true_horizon[0].item()}) "
              f"vs Pred (m={pred_tangent[0].item():.4f}, h={pred_horizon[0].item():.1f})"
    )
    
    # ========== PART 8: Threshold Optimization ==========
    print("\n8. Finding optimal threshold...")
    print("-" * 50)
    
    # Generate data for threshold analysis
    n_samples_analysis = 30  # Reduced for faster demo
    analysis_hist = []
    analysis_fut = []
    
    for i in range(n_samples_analysis):
        trend_type = np.random.choice(['upward', 'downward', 'sideways', 'volatile'])
        seed = i * 100 + 42
        hist, fut = generate_synthetic_data(
            history_length=50,
            future_length=30,
            trend_type=trend_type,
            noise_scale=1.5,
            seed=seed
        )
        analysis_hist.append(hist)
        analysis_fut.append(fut)
    
    analysis_fut_tensor = torch.stack(analysis_fut)
    
    # Find optimal threshold
    threshold_results = find_optimal_threshold(
        future_prices=analysis_fut_tensor,
        horizon_range=(3, 30),
        threshold_range=(3.0, 20.0),
        n_thresholds=10,  # Reduced for faster demo
        min_horizon=3
    )
    
    # Visualize threshold analysis
    fig_threshold = visualize_threshold_analysis(threshold_results)
    
    # ========== PART 9: Differentiable Labeling Demo ==========
    print("\n9. Demonstrating differentiable labeling...")
    print("-" * 50)
    
    # Create a small batch
    test_batch = analysis_fut_tensor[:5]
    
    # Standard labels
    tangents_standard, horizons_standard = generate_labels_dynamic_horizon(
        y=test_batch,
        max_accum_deviation=10.0,
        min_horizon=3,
        max_horizon=30
    )
    
    # Differentiable labels (requires gradients)
    tangents_diff, horizons_diff = generate_labels_dynamic_horizon_differentiable(
        y=test_batch,
        max_accum_deviation=10.0,
        min_horizon=3,
        max_horizon=30,
        temperature=0.5
    )
    
    print("Standard vs Differentiable labels:")
    for i in range(len(test_batch)):
        print(f"  Sample {i+1}: Standard (m={tangents_standard[i]:.4f}, h={horizons_standard[i]}) "
              f"vs Differentiable (m={tangents_diff[i]:.4f}, h={horizons_diff[i]:.2f})")
    
    print("\n" + "=" * 80)
    print("ENHANCED DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print("\nSummary of enhancements:")
    print("  ✓ Differentiable labeling for end-to-end training")
    print("  ✓ Comprehensive label statistics and validation")
    print("  ✓ Enhanced feature engineering pipeline")
    print("  ✓ Advanced neural network architectures (Multi-scale, Residual)")
    print("  ✓ Training with validation and early stopping")
    print("  ✓ Threshold optimization and analysis")
    print("  ✓ Realistic market data generation")
    print("  ✓ Memory-efficient batch processing")
    print("  ✓ Enhanced visualizations with metrics")
    
    return {
        'model': model,
        'training_history': history,
        'threshold_results': threshold_results,
        'label_stats': stats
    }


# ============================================================================
# RUN THE DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    results = main_demo()