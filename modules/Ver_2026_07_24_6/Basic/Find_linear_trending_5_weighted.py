"""
Dynamic Horizon Linear Trend Labeling - Weighted Shortest Horizon
===================================================================
Find the shortest horizon where WEIGHTED cumulative deviation reaches max_accum_deviation.
NEAR FUTURE time slots get higher weights, FAR FUTURE get lower weights.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# GPU SETUP
# ============================================================================

def get_device() -> str:
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Apple MPS available")
    else:
        device = 'cpu'
        print("Using CPU")
    return device

DEVICE = get_device()

# ============================================================================
# WEIGHTING FUNCTIONS - NEAR FUTURE GETS HIGHER WEIGHTS
# ============================================================================

def get_weights(horizon: int, weight_type: str = 'exponential', decay: float = 0.9) -> torch.Tensor:
    """
    Generate weights for different time slots.
    NEAR FUTURE (closer to t=0) gets higher weights.
    FAR FUTURE (closer to t=horizon-1) gets lower weights.
    
    Args:
        horizon: Number of time steps
        weight_type: 'exponential', 'linear', 'sqrt', 'sigmoid', 'none'
        decay: Decay factor for exponential weighting (0-1)
    
    Returns:
        weights: Tensor of shape (horizon,)
    """
    t = torch.arange(horizon, dtype=torch.float32)
    
    if weight_type == 'exponential':
        # Exponential decay: t=0 gets highest weight, t=horizon-1 gets lowest
        weights = torch.exp(-decay * t)
        weights = weights / weights.sum()
        
    elif weight_type == 'linear':
        # Linear: t=0 gets highest weight, t=horizon-1 gets lowest
        weights = torch.linspace(1, 0.1, horizon)
        weights = weights / weights.sum()
        
    elif weight_type == 'sqrt':
        # Square root: less aggressive decay
        weights = 1 / torch.sqrt(t + 1)
        weights = weights / weights.sum()
        
    elif weight_type == 'sigmoid':
        # Sigmoid: smooth transition from high to low
        mid = horizon / 2
        weights = 1 - 1 / (1 + torch.exp(-(t - mid) / (horizon / 6)))
        weights = weights + 0.1  # Ensure some weight for far future
        weights = weights / weights.sum()
        
    elif weight_type == 'step':
        # Step function: high weight for first half, low for second half
        weights = torch.ones(horizon)
        mid = horizon // 2
        weights[mid:] = 0.2
        weights = weights / weights.sum()
        
    elif weight_type == 'none' or weight_type == 'uniform':
        # Equal weights
        weights = torch.ones(horizon, dtype=torch.float32)
        weights = weights / weights.sum()
        
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")
    
    return weights


def visualize_weights(horizon: int = 30, figsize: Tuple[int, int] = (14, 10)):
    """
    Visualize different weighting schemes - NEAR FUTURE gets higher weights.
    """
    weight_types = ['exponential', 'linear', 'sqrt', 'sigmoid', 'step', 'none']
    decay_factors = [0.3, 0.5, 0.7, 0.9]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, wt in enumerate(weight_types):
        ax = axes[idx]
        
        if wt == 'exponential':
            for decay in decay_factors:
                weights = get_weights(horizon, wt, decay)
                ax.plot(weights.numpy(), label=f'decay={decay}')
            ax.set_title(f'{wt.upper()} Weighting (Near Future High)')
            ax.legend()
        else:
            weights = get_weights(horizon, wt)
            ax.plot(weights.numpy(), 'b-', linewidth=2)
            ax.fill_between(range(horizon), weights.numpy(), alpha=0.3)
            ax.set_title(f'{wt.upper()} Weighting (Near Future High)')
        
        ax.set_xlabel('Time Step (0 = NEAR FUTURE, far = FAR FUTURE)')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(weights.numpy()) * 1.1)
    
    axes[5].axis('off')
    plt.suptitle('Weighting Schemes - NEAR FUTURE Gets Higher Weights', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
# CORE FUNCTION: Find the Shortest Horizon with Weighting
# ============================================================================

def find_shortest_horizon_weighted(
    future_prices: torch.Tensor,
    max_accum_deviation: float,
    min_horizon: int = 2,
    max_horizon: Optional[int] = None,
    weight_type: str = 'exponential',
    decay: float = 0.9,
    device: str = DEVICE
) -> Tuple[int, float, float, float, List[float], List[float], List[float], List[float], List[float]]:
    """
    Find the SHORTEST horizon where WEIGHTED cumulative deviation reaches max_accum_deviation.
    NEAR FUTURE time slots get higher weights.
    
    Args:
        future_prices: Future price sequence, shape (future_length,)
        max_accum_deviation: Threshold for weighted cumulative absolute deviation
        min_horizon: Minimum horizon to consider
        max_horizon: Maximum horizon to consider
        weight_type: 'exponential', 'linear', 'sqrt', 'sigmoid', 'step', 'none'
        decay: Decay factor for exponential weighting (higher = more weight to near future)
    
    Returns:
        horizon: The shortest horizon where weighted cumulative deviation >= threshold
        slope: Slope (m) at that horizon
        intercept: Intercept (b) at that horizon
        cumulative_deviation: The weighted cumulative deviation at this horizon
        all_horizons: List of all horizons tested
        all_cum_devs: List of weighted cumulative deviations for each horizon
        all_weights: The weight vector used at each horizon
        all_slopes: Slopes for all horizons
        all_intercepts: Intercepts for all horizons
    """
    
    # Move to device and ensure 1D
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
    
    # Search for the shortest horizon
    for h in range(min_horizon, max_horizon + 1):
        # Get the segment
        segment = prices[:h]
        
        # Time indices starting from 0 (NEAR FUTURE)
        t = torch.arange(h, dtype=torch.float32, device=device)
        
        # Get weights for this horizon - NEAR FUTURE gets higher weights
        weights = get_weights(h, weight_type, decay)
        all_weights.append(weights.cpu().numpy())
        
        # Fit weighted linear trend using WLS: y = m*t + b
        w = weights.to(device)
        
        # Weighted means
        w_sum = w.sum()
        t_mean_w = torch.sum(w * t) / w_sum
        y_mean_w = torch.sum(w * segment) / w_sum
        
        # Weighted covariance and variance
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
        
        # Calculate weighted cumulative absolute deviation
        predicted = slope * t + intercept
        deviations = torch.abs(segment - predicted)
        weighted_deviations = w * deviations
        cumulative_deviation = torch.sum(weighted_deviations).item()
        
        # Store for analysis
        all_horizons.append(h)
        all_cum_devs.append(cumulative_deviation)
        all_slopes.append(slope)
        all_intercepts.append(intercept)
        
        # If we've reached the threshold, return this horizon
        if cumulative_deviation >= max_accum_deviation:
            return h, slope, intercept, cumulative_deviation, all_horizons, all_cum_devs, all_weights, all_slopes, all_intercepts
    
    # If we never reach the threshold, return the max horizon
    h = max_horizon
    segment = prices[:h]
    t = torch.arange(h, dtype=torch.float32, device=device)
    weights = get_weights(h, weight_type, decay)
    
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
    weighted_deviations = w * deviations
    cumulative_deviation = torch.sum(weighted_deviations).item()
    
    print(f"WARNING: Threshold {max_accum_deviation} NOT reached!")
    print(f"  Max weighted cumulative deviation at H={h}: {cumulative_deviation:.2f}")
    print(f"  Returning horizon={h} with best fit")
    
    return h, slope, intercept, cumulative_deviation, all_horizons, all_cum_devs, all_weights, all_slopes, all_intercepts


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_synthetic_data(
    history_length: int = 100,
    future_length: int = 50,
    trend_type: str = 'upward',
    noise_scale: float = 2.0,
    seed: int = 42,
    device: str = DEVICE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic price data with various trend types.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate historical data (random walk)
    historical = np.cumsum(np.random.randn(history_length) * 0.5) + 100
    
    # Generate future with different trends
    if trend_type == 'upward':
        future = np.cumsum(np.random.randn(future_length) * 0.5 + 0.4) + historical[-1]
    elif trend_type == 'strong_upward':
        future = np.cumsum(np.random.randn(future_length) * 0.3 + 0.8) + historical[-1]
    elif trend_type == 'weak_upward':
        future = np.cumsum(np.random.randn(future_length) * 0.5 + 0.15) + historical[-1]
    elif trend_type == 'downward':
        future = np.cumsum(np.random.randn(future_length) * 0.5 - 0.4) + historical[-1]
    elif trend_type == 'strong_downward':
        future = np.cumsum(np.random.randn(future_length) * 0.3 - 0.8) + historical[-1]
    elif trend_type == 'sideways':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
    elif trend_type == 'volatile':
        future = np.cumsum(np.random.randn(future_length) * 1.5) + historical[-1]
    elif trend_type == 'mixed':
        trend = np.sin(np.linspace(0, 2*np.pi, future_length)) * 5
        future = np.cumsum(np.random.randn(future_length) * 0.3) + trend + historical[-1]
    elif trend_type == 'reversal':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
        reversal_point = future_length // 2
        future[reversal_point:] -= np.linspace(0, 10, future_length - reversal_point)
    elif trend_type == 'breakout':
        future = np.cumsum(np.random.randn(future_length) * 0.2) + historical[-1]
        breakout_point = future_length // 3
        future[breakout_point:] += np.linspace(0, 15, future_length - breakout_point)
    else:  # 'random_walk'
        future = np.cumsum(np.random.randn(future_length) * 0.5) + historical[-1]
    
    return (torch.tensor(historical, dtype=torch.float32, device=device), 
            torch.tensor(future, dtype=torch.float32, device=device))


# ============================================================================
# ENHANCED VISUALIZATION
# ============================================================================

def visualize_weighted_result(
    historical_prices: torch.Tensor,
    future_prices: torch.Tensor,
    horizon: int,
    slope: float,
    intercept: float,
    cumulative_deviation: float,
    max_accum_deviation: float,
    all_horizons: List[float],
    all_cum_devs: List[float],
    all_weights: List[np.ndarray],
    all_slopes: List[float],
    all_intercepts: List[float],
    weight_type: str = 'exponential',
    decay: float = 0.9,
    min_horizon: int = 2,
    title: str = "Weighted Shortest Horizon Labeling",
    figsize: Tuple[int, int] = (20, 14)
) -> plt.Figure:
    """
    Enhanced visualization with weighting information.
    NEAR FUTURE gets higher weights.
    """
    
    # Move to CPU for plotting
    hist = historical_prices.cpu().numpy()
    fut = future_prices.cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # --- Plot 1: Full price history with weighted trend ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    full_prices = np.concatenate([hist, fut])
    time_steps = np.arange(len(full_prices))
    split_idx = len(hist)
    
    # Plot historical and future
    ax1.plot(time_steps[:split_idx], hist, 'b-', linewidth=2, label='Historical', alpha=0.7)
    ax1.plot(time_steps[split_idx:], fut, 'g-', linewidth=2, label='Future Prices', alpha=0.8)
    
    # Plot the weighted trend line
    if horizon > 0:
        trend_time = np.arange(horizon)
        trend_prices = slope * trend_time + intercept
        ax1.plot(
            time_steps[split_idx:split_idx + horizon],
            trend_prices,
            'r-', linewidth=3, label=f'Weighted OLS: y = {slope:.4f}*t + {intercept:.2f}'
        )
        
        # Mark horizon boundary
        ax1.axvline(
            x=split_idx + horizon - 1,
            color='red', linestyle='-', linewidth=2, alpha=0.8,
            label=f'H = {horizon} (SHORTEST)'
        )
        
        # Fill selected region
        ax1.axvspan(
            split_idx, split_idx + horizon - 1,
            alpha=0.15, color='orange', label='Selected Region'
        )
    
    ax1.axvline(x=split_idx - 0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.text(split_idx - 1, ax1.get_ylim()[1] * 0.95, 'NOW', 
             horizontalalignment='center', fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('Time Steps', fontsize=11)
    ax1.set_ylabel('Price', fontsize=11)
    ax1.set_title('Price History with Weighted OLS Trend', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Weighting Scheme ---
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Show the weights
    if all_weights and horizon > 0:
        idx = horizon - min_horizon
        if idx < len(all_weights):
            weights = all_weights[idx]
            ax2.plot(weights, 'b-', linewidth=2, label=f'Weights ({weight_type})')
            ax2.fill_between(range(len(weights)), weights, alpha=0.3)
            
            # Annotate near vs far future
            ax2.axvline(x=0, color='green', linestyle=':', alpha=0.5, label='NEAR FUTURE (high weight)')
            ax2.axvline(x=horizon-1, color='red', linestyle=':', alpha=0.5, label='FAR FUTURE (low weight)')
            
            ax2.set_xlabel('Time Step (0 = NEAR FUTURE)', fontsize=11)
            ax2.set_ylabel('Weight', fontsize=11)
            ax2.set_title(f'Weighting Scheme: Near Future High (decay={decay})', fontsize=12, fontweight='bold')
            ax2.legend(loc='best', fontsize=8)
            ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Cumulative Deviation Growth ---
    ax3 = fig.add_subplot(gs[0, 2])
    
    if all_horizons and all_cum_devs:
        ax3.plot(all_horizons, all_cum_devs, 'b-', linewidth=2.5, label='Weighted Cumulative Deviation')
        ax3.axhline(y=max_accum_deviation, color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'Threshold = {max_accum_deviation}')
        
        # Mark the selected horizon
        ax3.scatter(horizon, cumulative_deviation, color='red', s=100, zorder=5,
                   label=f'Selected: H={horizon}')
        ax3.axvline(x=horizon, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
        
        # Fill below threshold
        cum_devs_array = np.array(all_cum_devs)
        below_threshold = cum_devs_array < max_accum_deviation
        ax3.fill_between(all_horizons, 0, cum_devs_array, 
                         where=below_threshold,
                         alpha=0.2, color='green', label='Below Threshold')
        
        ax3.set_xlabel('Horizon (h)', fontsize=11)
        ax3.set_ylabel('Weighted Cumulative Deviation', fontsize=11)
        ax3.set_title('Weighted Cumulative Deviation Growth', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Future prices with deviations ---
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Plot future prices
    ax4.plot(fut, 'g-', linewidth=2, label='Actual Prices', alpha=0.7)
    ax4.scatter(np.arange(len(fut)), fut, color='green', s=30, alpha=0.5)
    
    # Plot weighted trend
    if horizon > 0:
        x_vals = np.arange(horizon)
        y_vals = fut[:horizon]
        trend_vals = slope * x_vals + intercept
        
        ax4.plot(x_vals, trend_vals, 'r-', linewidth=2.5, 
                label=f'Weighted OLS: y={slope:.4f}t+{intercept:.2f}')
        
        # Show weighted deviation bars
        deviations = np.abs(y_vals - trend_vals)
        
        # Weighted deviations for display
        if all_weights and horizon > 0:
            idx = horizon - min_horizon
            if idx < len(all_weights):
                weights = all_weights[idx]
                weighted_dev = deviations * weights
                ax4.bar(x_vals, weighted_dev, bottom=np.minimum(y_vals, trend_vals),
                        alpha=0.3, color='orange', width=0.6, 
                        label='Weighted Deviations')
        
        # Mark horizon
        ax4.axvline(x=horizon - 1, color='red', linestyle='-', 
                   linewidth=2, alpha=0.7)
        
        # Annotations
        total_dev = np.sum(deviations)
        weighted_total = cumulative_deviation
        ax4.text(0.02, 0.98, f'Total Dev: {total_dev:.2f}\nWeighted Dev: {weighted_total:.2f}',
                transform=ax4.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax4.set_xlabel('Future Time Steps (0 = NEAR FUTURE)', fontsize=11)
    ax4.set_ylabel('Price', fontsize=11)
    ax4.set_title(f'Weighted Deviations - Near Future Gets Higher Weight (H={horizon})', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # --- Plot 5: Weighted vs Unweighted Comparison ---
    ax5 = fig.add_subplot(gs[1, 1])
    
    if all_horizons and all_cum_devs and all_slopes and all_intercepts:
        # Compute unweighted deviations for comparison
        prices = future_prices.cpu().numpy()
        unweighted_cum = []
        weighted_cum = all_cum_devs
        
        for i, h in enumerate(all_horizons):
            segment = prices[:h]
            t = np.arange(h)
            m = all_slopes[i]
            b = all_intercepts[i]
            pred = m * t + b
            unweighted_cum.append(np.sum(np.abs(segment - pred)))
        
        ax5.plot(all_horizons, unweighted_cum, 'b-', linewidth=2, label='Unweighted')
        ax5.plot(all_horizons, weighted_cum, 'r-', linewidth=2, label='Weighted (Near Future High)')
        ax5.axhline(y=max_accum_deviation, color='black', linestyle='--', 
                   alpha=0.5, label='Threshold')
        
        # Mark selected horizon
        ax5.axvline(x=horizon, color='green', linestyle=':', alpha=0.5)
        
        ax5.set_xlabel('Horizon (h)', fontsize=11)
        ax5.set_ylabel('Cumulative Deviation', fontsize=11)
        ax5.set_title('Weighted vs Unweighted Comparison', fontsize=12, fontweight='bold')
        ax5.legend(loc='best', fontsize=9)
        ax5.grid(True, alpha=0.3)
    
    # --- Plot 6: Per-step weighted deviations ---
    ax6 = fig.add_subplot(gs[1, 2])
    
    if horizon > 0:
        x_vals = np.arange(horizon)
        y_vals = fut[:horizon]
        trend_vals = slope * x_vals + intercept
        per_step_dev = np.abs(y_vals - trend_vals)
        
        if all_weights and horizon > 0:
            idx = horizon - min_horizon
            if idx < len(all_weights):
                weights = all_weights[idx]
                weighted_per_step = per_step_dev * weights
                
                # Bar chart of weighted deviations
                bars = ax6.bar(x_vals, weighted_per_step, alpha=0.6, color='steelblue')
                
                # Color significant deviations
                avg_dev = weighted_per_step.mean()
                for i, bar in enumerate(bars):
                    if weighted_per_step[i] > avg_dev * 1.5:
                        bar.set_color('red')
                        bar.set_alpha(0.8)
                
                ax6.axhline(y=avg_dev, color='red', linestyle='--', 
                           label=f'Avg Weighted Dev = {avg_dev:.3f}')
                
                # Show original deviations as line
                if per_step_dev.max() > 0:
                    scaled_original = per_step_dev / per_step_dev.max() * weighted_per_step.max()
                    ax6.plot(x_vals, scaled_original, 'g--', linewidth=1.5, alpha=0.5, 
                            label='Original Dev (scaled)')
                
                # Mark near vs far future
                ax6.axvline(x=horizon//2, color='orange', linestyle=':', alpha=0.5, 
                           label='Midpoint')
        
        ax6.set_xlabel('Time Step (0 = NEAR FUTURE)', fontsize=11)
        ax6.set_ylabel('Weighted Deviation', fontsize=11)
        ax6.set_title(f'Per-Step Weighted Deviations - Near Future Higher', fontsize=12, fontweight='bold')
        ax6.legend(loc='best', fontsize=9)
        ax6.grid(True, alpha=0.3)
    
    # --- Plot 7: Weighted Residuals ---
    ax7 = fig.add_subplot(gs[2, 0])
    
    if horizon > 0:
        x_vals = np.arange(horizon)
        y_vals = fut[:horizon]
        trend_vals = slope * x_vals + intercept
        residuals = y_vals - trend_vals
        
        if all_weights and horizon > 0:
            idx = horizon - min_horizon
            if idx < len(all_weights):
                weights = all_weights[idx]
                # Size proportional to weight
                sizes = weights * 100 + 20
                
                scatter = ax7.scatter(x_vals, residuals, s=sizes, c=weights, 
                                     alpha=0.6, cmap='viridis')
                ax7.axhline(y=0, color='red', linestyle='-', linewidth=1.5)
                
                # Colorbar
                cbar = plt.colorbar(scatter, ax=ax7)
                cbar.set_label('Weight (High = Near Future)')
                
                ax7.set_xlabel('Time Step (0 = NEAR FUTURE)', fontsize=11)
                ax7.set_ylabel('Residual', fontsize=11)
                ax7.set_title(f'Weighted Residuals (size = weight - near future bigger)', fontsize=12, fontweight='bold')
                ax7.grid(True, alpha=0.3)
    
    # --- Plot 8: Metrics Summary ---
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.axis('off')
    
    # Calculate R² for the selected horizon
    if horizon > 0:
        x_vals = np.arange(horizon)
        y_vals = fut[:horizon]
        trend_vals = slope * x_vals + intercept
        
        ss_res = np.sum((y_vals - trend_vals) ** 2)
        ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
    else:
        r2 = 0
    
    # Calculate additional metrics
    if horizon > 0:
        x_vals = np.arange(horizon)
        y_vals = fut[:horizon]
        trend_vals = slope * x_vals + intercept
        deviations = np.abs(y_vals - trend_vals)
        mae = np.mean(deviations)
        rmse = np.sqrt(np.mean(deviations ** 2))
        max_dev = np.max(deviations)
    else:
        mae = rmse = max_dev = 0
    
    threshold_reached = cumulative_deviation >= max_accum_deviation
    
    metrics_text = f"""
    ┌─────────────────────────────────────────────────────────────┐
    │        WEIGHTED SHORTEST HORIZON - RESULTS                  │
    │     (Near Future Gets Higher Weights)                       │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  WEIGHTING SCHEME     : {weight_type:>12}                    │
    │  DECAY FACTOR         : {decay:>12.2f}                       │
    │  HORIZON (H)          : {horizon:>4}                        │
    │  SLOPE (m)            : {slope:>10.6f}                      │
    │  INTERCEPT (b)        : {intercept:>10.3f}                  │
    │  R²                   : {r2:>10.4f}                         │
    │                                                             │
    │  THRESHOLD            : {max_accum_deviation:>10.2f}        │
    │  WEIGHTED CUM DEV     : {cumulative_deviation:>10.2f}       │
    │  THRESHOLD REACHED?   : {str(threshold_reached):>10}        │
    │                                                             │
    │  MAE                  : {mae:>10.3f}                        │
    │  RMSE                 : {rmse:>10.3f}                       │
    │  MAX DEVIATION        : {max_dev:>10.3f}                    │
    │                                                             │
    │  MIN HORIZON          : {min_horizon:>4}                    │
    │  MAX HORIZON          : {horizon:>4} (selected)             │
    └─────────────────────────────────────────────────────────────┘
    """
    
    ax8.text(0.5, 0.5, metrics_text, fontsize=10, verticalalignment='center',
             horizontalalignment='center', family='monospace', transform=ax8.transAxes)
    
    # --- Plot 9: Legend ---
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    legend_text = """
    WEIGHTING EXPLANATION:
    
    • NEAR FUTURE (t=0) gets HIGHER weights
    • FAR FUTURE (t=horizon-1) gets LOWER weights
    
    WHY?
    • Near future prices are more certain/relevant
    • Far future prices are more uncertain
    • Trading decisions depend more on near-term trends
    
    EFFECTS:
    • Weighted OLS fits near future prices more closely
    • Shortest horizon found earlier (near future matters more)
    • More responsive to immediate trend changes
    • Less influenced by distant, uncertain price movements
    
    RECOMMENDATIONS:
    • High decay (0.9): Very short-term focused
    • Low decay (0.3): More balanced
    • Linear: Smooth decrease in importance
    """
    
    ax9.text(0.5, 0.5, legend_text, fontsize=10, verticalalignment='center',
             horizontalalignment='center', family='monospace', transform=ax9.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig


# ============================================================================
# COMPARISON FUNCTION
# ============================================================================

def compare_weighting_schemes(
    future_prices: torch.Tensor,
    max_accum_deviation: float,
    min_horizon: int = 2,
    max_horizon: Optional[int] = None,
    figsize: Tuple[int, int] = (18, 12)
) -> plt.Figure:
    """
    Compare different weighting schemes on the same data.
    """
    
    weight_schemes = [
        ('none', 0.0, 'Unweighted'),
        ('linear', 0.0, 'Linear (Near High)'),
        ('sqrt', 0.0, 'Sqrt (Near High)'),
        ('sigmoid', 0.0, 'Sigmoid'),
        ('step', 0.0, 'Step (Near High)'),
        ('exponential', 0.5, 'Exp (decay=0.5)'),
        ('exponential', 0.7, 'Exp (decay=0.7)'),
        ('exponential', 0.9, 'Exp (decay=0.9)'),
    ]
    
    results = []
    
    for wt, decay, label in weight_schemes:
        h, m, b, cum, all_h, all_c, all_w, all_s, all_i = find_shortest_horizon_weighted(
            future_prices=future_prices,
            max_accum_deviation=max_accum_deviation,
            min_horizon=min_horizon,
            max_horizon=max_horizon,
            weight_type=wt,
            decay=decay,
            device=DEVICE
        )
        
        results.append({
            'label': label,
            'horizon': h,
            'slope': m,
            'intercept': b,
            'cum_dev': cum,
            'all_horizons': all_h,
            'all_cum_devs': all_c,
            'reached': cum >= max_accum_deviation
        })
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        
        ax.plot(result['all_horizons'], result['all_cum_devs'], 'b-', linewidth=2)
        ax.axhline(y=max_accum_deviation, color='red', linestyle='--', alpha=0.7)
        
        if result['reached']:
            ax.scatter(result['horizon'], result['cum_dev'], color='red', s=80, zorder=5)
            ax.axvline(x=result['horizon'], color='red', linestyle=':', alpha=0.5)
        
        ax.set_title(f"{result['label']}\nH={result['horizon']}, m={result['slope']:.3f}", 
                    fontsize=10)
        ax.set_xlabel('Horizon')
        ax.set_ylabel('Cum Dev')
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(['Cum Dev', f'Threshold={max_accum_deviation}'], fontsize=8)
    
    plt.suptitle('Comparison of Weighting Schemes (Near Future Gets Higher Weights)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main_demo():
    """
    Comprehensive demonstration of weighted shortest horizon labeling.
    NEAR FUTURE gets higher weights.
    """
    print("=" * 80)
    print("WEIGHTED SHORTEST HORIZON LABELING - Near Future Gets Higher Weights")
    print("=" * 80)
    print(f"Using device: {DEVICE}")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # ========== PART 1: Visualize Weighting Schemes ==========
    print("\n" + "="*80)
    print("PART 1: Weighting Schemes Visualization")
    print("="*80)
    print("\nNEAR FUTURE (t=0) gets HIGHEST weight")
    print("FAR FUTURE (t=horizon-1) gets LOWEST weight")
    
    visualize_weights(horizon=30)
    
    # ========== PART 2: Compare Weighting Schemes ==========
    print("\n" + "="*80)
    print("PART 2: Comparing Weighting Schemes on Same Data")
    print("="*80)
    
    # Generate test data
    _, future = generate_synthetic_data(
        history_length=80,
        future_length=40,
        trend_type='mixed',
        noise_scale=1.5,
        seed=42,
        device=DEVICE
    )
    
    threshold = 10.0
    
    print(f"\nData: Mixed trend with threshold={threshold}")
    print("-" * 80)
    print(f"{'Weighting':>20} | {'H':>4} | {'Slope':>10} | {'Cum Dev':>10} | {'Status':>12}")
    print("-" * 80)
    
    weight_schemes = [
        ('none', 0.0, 'Unweighted'),
        ('linear', 0.0, 'Linear (Near High)'),
        ('sqrt', 0.0, 'Sqrt (Near High)'),
        ('sigmoid', 0.0, 'Sigmoid'),
        ('step', 0.0, 'Step (Near High)'),
        ('exponential', 0.5, 'Exp(0.5)'),
        ('exponential', 0.7, 'Exp(0.7)'),
        ('exponential', 0.9, 'Exp(0.9)'),
    ]
    
    for wt, decay, label in weight_schemes:
        h, m, b, cum, _, _, _, _, _ = find_shortest_horizon_weighted(
            future_prices=future,
            max_accum_deviation=threshold,
            min_horizon=3,
            max_horizon=35,
            weight_type=wt,
            decay=decay,
            device=DEVICE
        )
        
        status = "✓ REACHED" if cum >= threshold else "✗ NOT REACHED"
        print(f"  {label:>18} | {h:>4} | {m:>10.4f} | {cum:>10.2f} | {status:>12}")
    
    # Visualize comparison
    print("\nGenerating comparison visualization...")
    compare_weighting_schemes(
        future_prices=future,
        max_accum_deviation=threshold,
        min_horizon=3,
        max_horizon=35
    )
    
    # ========== PART 3: Different Decay Factors ==========
    print("\n" + "="*80)
    print("PART 3: Effect of Decay Factor on Exponential Weighting")
    print("="*80)
    print("\nHigher decay = More weight on NEAR FUTURE")
    
    decay_factors = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    
    print(f"\nData: Mixed trend with threshold={threshold}")
    print("-" * 70)
    print(f"{'Decay':>10} | {'H':>4} | {'Slope':>10} | {'Cum Dev':>10} | {'Near Future Weight':>15}")
    print("-" * 70)
    
    for decay in decay_factors:
        h, m, b, cum, _, _, all_w, _, _ = find_shortest_horizon_weighted(
            future_prices=future,
            max_accum_deviation=threshold,
            min_horizon=3,
            max_horizon=35,
            weight_type='exponential',
            decay=decay,
            device=DEVICE
        )
        
        # Get weight at t=0 (near future) for this horizon
        if all_w and h >= 3:
            idx = h - 3
            if idx < len(all_w):
                near_weight = all_w[idx][0]
            else:
                near_weight = 0
        else:
            near_weight = 0
        
        status = "✓" if cum >= threshold else "✗"
        print(f"  {decay:>10.2f} | {h:>4} | {m:>10.4f} | {cum:>10.2f} | {near_weight:>15.4f}")
    
    # ========== PART 4: Detailed Example ==========
    print("\n" + "="*80)
    print("PART 4: Detailed Weighted Example")
    print("="*80)
    
    # Generate a more interesting example
    _, future = generate_synthetic_data(
        history_length=80,
        future_length=40,
        trend_type='reversal',
        noise_scale=1.5,
        seed=123,
        device=DEVICE
    )
    
    threshold = 8.0
    
    print(f"\nData: Reversal pattern with threshold={threshold}")
    print(f"Weighting: Exponential (decay=0.9) - Near Future gets higher weights")
    print("\nFinding shortest weighted horizon...")
    
    h, m, b, cum, all_h, all_c, all_w, all_s, all_i = find_shortest_horizon_weighted(
        future_prices=future,
        max_accum_deviation=threshold,
        min_horizon=3,
        max_horizon=35,
        weight_type='exponential',
        decay=0.9,
        device=DEVICE
    )
    
    print(f"\nResults:")
    print(f"  Horizon: {h}")
    print(f"  Slope: {m:.4f}")
    print(f"  Intercept: {b:.2f}")
    print(f"  Weighted Cumulative Deviation: {cum:.2f}")
    print(f"  Threshold Reached: {cum >= threshold}")
    
    # Show near vs far weights
    if all_w and h >= 3:
        idx = h - 3
        if idx < len(all_w):
            weights = all_w[idx]
            print(f"\n  Near Future Weight (t=0): {weights[0]:.4f}")
            print(f"  Far Future Weight (t={h-1}): {weights[-1]:.4f}")
            print(f"  Weight Ratio: {weights[0]/weights[-1]:.2f}x")
    
    # Create dummy historical data for visualization
    dummy_hist = torch.zeros(1, device=DEVICE)
    
    visualize_weighted_result(
        historical_prices=dummy_hist,
        future_prices=future,
        horizon=h,
        slope=m,
        intercept=b,
        cumulative_deviation=cum,
        max_accum_deviation=threshold,
        all_horizons=all_h,
        all_cum_devs=all_c,
        all_weights=all_w,
        all_slopes=all_s,
        all_intercepts=all_i,
        weight_type='exponential',
        decay=0.9,
        min_horizon=3,
        title=f"Reversal Pattern - Near Future Weighted (H={h})"
    )
    
    # ========== PART 5: Compare Weighted vs Unweighted ==========
    print("\n" + "="*80)
    print("PART 5: Weighted vs Unweighted Comparison on Multiple Trends")
    print("="*80)
    
    test_trends = ['upward', 'downward', 'volatile', 'reversal', 'breakout']
    threshold = 12.0
    
    print(f"\nThreshold: {threshold}")
    print("-" * 80)
    print(f"{'Trend':>12} | {'Unweighted H':>12} | {'Weighted H':>12} | {'Difference':>12}")
    print("-" * 80)
    
    for trend in test_trends:
        _, future = generate_synthetic_data(
            history_length=80,
            future_length=40,
            trend_type=trend,
            noise_scale=1.5,
            seed=hash(trend) % 1000,
            device=DEVICE
        )
        
        # Unweighted
        h_u, m_u, b_u, cum_u, _, _, _, _, _ = find_shortest_horizon_weighted(
            future_prices=future,
            max_accum_deviation=threshold,
            min_horizon=3,
            max_horizon=35,
            weight_type='none',
            device=DEVICE
        )
        
        # Weighted (exponential - near future high)
        h_w, m_w, b_w, cum_w, _, _, _, _, _ = find_shortest_horizon_weighted(
            future_prices=future,
            max_accum_deviation=threshold,
            min_horizon=3,
            max_horizon=35,
            weight_type='exponential',
            decay=0.9,
            device=DEVICE
        )
        
        diff = h_w - h_u
        print(f"  {trend:>10} | {h_u:>12} | {h_w:>12} | {diff:>+12}")
    
    # ========== SUMMARY ==========
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nKey Findings:")
    print("  ✓ NEAR FUTURE time slots get HIGHER weights")
    print("  ✓ FAR FUTURE time slots get LOWER weights")
    print("  ✓ This makes the labeling more sensitive to near-term price behavior")
    print("  ✓ Weighted OLS fits near future prices more closely")
    print("  ✓ Shorter horizons are found because near future matters more")
    print("  ✓ More responsive to immediate trend changes")
    
    print("\nWeighting Options (Near Future High):")
    print("  • 'exponential': Smooth decay from near to far (recommended)")
    print("    - decay=0.5: moderate near-future emphasis")
    print("    - decay=0.9: strong near-future emphasis")
    print("    - decay=0.95: very strong near-future emphasis")
    print("  • 'linear': Linear decrease from near to far")
    print("  • 'sqrt': Less aggressive than linear")
    print("  • 'sigmoid': S-shaped transition")
    print("  • 'step': Step function (high for first half, low for second)")
    print("  • 'none' or 'uniform': Equal weights (unweighted)")
    
    print("\nUse Cases:")
    print("  • High decay (0.9-0.95): Very short-term focused, scalping")
    print("  • Medium decay (0.5-0.7): Short-term trading")
    print("  • Low decay (0.3): More balanced near/far")
    print("  • Unweighted: When all time periods are equally important")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main_demo()