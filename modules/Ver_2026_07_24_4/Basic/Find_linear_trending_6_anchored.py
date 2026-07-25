"""
Dynamic Horizon Linear Trend Labeling - Find Single Shortest Horizon
===================================================================
Find the single shortest horizon where cumulative deviation reaches max_accum_deviation.
The trend line is constrained to pass through the first future price (f[0]).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Any
from scipy import stats
import warnings
from collections import defaultdict

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
# CORE FUNCTION: Find the Shortest Horizon (Constrained through f[0])
# ============================================================================

def find_shortest_horizon(
    future_prices: torch.Tensor,
    max_accum_deviation: float,
    min_horizon: int = 2,
    max_horizon: Optional[int] = None,
    device: str = DEVICE,
    verbose: bool = False
) -> Tuple[int, float, float, float, List[float], List[float], List[float]]:
    """
    Find the SINGLE SHORTEST horizon where cumulative deviation reaches max_accum_deviation.
    The trend line is CONSTRAINED to pass through the first price point f[0].
    
    Args:
        future_prices: Future price sequence, shape (future_length,)
        max_accum_deviation: Threshold for cumulative absolute deviation
        min_horizon: Minimum horizon to consider
        max_horizon: Maximum horizon to consider
        device: Device for computation
        verbose: Print detailed progress
    
    Returns:
        horizon: The shortest horizon where cumulative deviation >= threshold
        slope: Slope (m) at that horizon (trend line: y = m*t + f[0])
        intercept: Always equals f[0] (the first future price)
        cumulative_deviation: The cumulative deviation at this horizon
        all_horizons: List of all horizons tested
        all_cum_devs: List of cumulative deviations for each horizon
        all_slopes: List of slopes for each horizon
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
    
    # Get the first price point (f[0]) - this is where the trend line starts
    f0 = prices[0].item()
    
    all_horizons = []
    all_cum_devs = []
    all_slopes = []
    
    if verbose:
        print(f"\n{'H':>4} | {'Slope':>12} | {'Cum Dev':>12} | {'Status':>15}")
        print("-" * 50)
    
    # Search for the shortest horizon
    for h in range(min_horizon, max_horizon + 1):
        # Get the segment
        segment = prices[:h]
        
        # Time indices starting from 0
        t = torch.arange(h, dtype=torch.float32, device=device)
        
        # Fit linear trend CONSTRAINED through f[0]: y = m*t + f[0]
        y_centered = segment - f0
        numerator = torch.sum(t * y_centered)
        denominator = torch.sum(t ** 2)
        
        if denominator < 1e-10:
            slope = 0.0
        else:
            slope = (numerator / denominator).item()
        
        # Intercept is always f[0]
        intercept = f0
        
        # Calculate cumulative absolute deviation
        predicted = slope * t + intercept
        deviations = torch.abs(segment - predicted)
        cumulative_deviation = torch.sum(deviations).item()
        
        # Store for analysis
        all_horizons.append(h)
        all_cum_devs.append(cumulative_deviation)
        all_slopes.append(slope)
        
        if verbose:
            status = "✓ REACHED" if cumulative_deviation >= max_accum_deviation else "..."
            print(f"{h:>4} | {slope:>12.6f} | {cumulative_deviation:>12.4f} | {status:>15}")
        
        # If we've reached the threshold, return this horizon
        if cumulative_deviation >= max_accum_deviation:
            return h, slope, intercept, cumulative_deviation, all_horizons, all_cum_devs, all_slopes
    
    # If we never reach the threshold, return the max horizon
    h = max_horizon
    segment = prices[:h]
    t = torch.arange(h, dtype=torch.float32, device=device)
    
    y_centered = segment - f0
    numerator = torch.sum(t * y_centered)
    denominator = torch.sum(t ** 2)
    
    if denominator < 1e-10:
        slope = 0.0
    else:
        slope = (numerator / denominator).item()
    
    intercept = f0
    
    predicted = slope * t + intercept
    deviations = torch.abs(segment - predicted)
    cumulative_deviation = torch.sum(deviations).item()
    
    print(f"WARNING: Threshold {max_accum_deviation} NOT reached!")
    print(f"  Max cumulative deviation at H={h}: {cumulative_deviation:.2f}")
    print(f"  Returning horizon={h} with best fit")
    
    return h, slope, intercept, cumulative_deviation, all_horizons, all_cum_devs, all_slopes


# ============================================================================
# DATA GENERATION - EXTENDED WITH MORE PATTERNS
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
    elif trend_type == 'upward_strong':
        future = np.cumsum(np.random.randn(future_length) * 0.3 + 0.8) + historical[-1]
    elif trend_type == 'upward_weak':
        future = np.cumsum(np.random.randn(future_length) * 0.5 + 0.15) + historical[-1]
    elif trend_type == 'upward_steep':
        future = np.cumsum(np.random.randn(future_length) * 0.2 + 1.2) + historical[-1]
    elif trend_type == 'upward_shallow':
        future = np.cumsum(np.random.randn(future_length) * 0.6 + 0.1) + historical[-1]
    elif trend_type == 'downward':
        future = np.cumsum(np.random.randn(future_length) * 0.5 - 0.4) + historical[-1]
    elif trend_type == 'downward_strong':
        future = np.cumsum(np.random.randn(future_length) * 0.3 - 0.8) + historical[-1]
    elif trend_type == 'downward_weak':
        future = np.cumsum(np.random.randn(future_length) * 0.5 - 0.15) + historical[-1]
    elif trend_type == 'downward_steep':
        future = np.cumsum(np.random.randn(future_length) * 0.2 - 1.2) + historical[-1]
    elif trend_type == 'sideways':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
    elif trend_type == 'sideways_tight':
        future = np.cumsum(np.random.randn(future_length) * 0.1) + historical[-1]
    elif trend_type == 'sideways_wide':
        future = np.cumsum(np.random.randn(future_length) * 0.8) + historical[-1]
    elif trend_type == 'volatile':
        future = np.cumsum(np.random.randn(future_length) * 1.5) + historical[-1]
    elif trend_type == 'volatile_extreme':
        future = np.cumsum(np.random.randn(future_length) * 3.0) + historical[-1]
    elif trend_type == 'volatile_mild':
        future = np.cumsum(np.random.randn(future_length) * 0.8) + historical[-1]
    elif trend_type == 'mixed_sine':
        trend = np.sin(np.linspace(0, 2*np.pi, future_length)) * 5
        future = np.cumsum(np.random.randn(future_length) * 0.3) + trend + historical[-1]
    elif trend_type == 'mixed_square':
        trend = np.sign(np.sin(np.linspace(0, 2*np.pi, future_length))) * 5
        future = np.cumsum(np.random.randn(future_length) * 0.3) + trend + historical[-1]
    elif trend_type == 'mixed_sawtooth':
        trend = np.linspace(0, 5, future_length) % 5 - 2.5
        future = np.cumsum(np.random.randn(future_length) * 0.3) + trend + historical[-1]
    elif trend_type == 'reversal_v':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
        reversal_point = future_length // 2
        future[reversal_point:] -= np.linspace(0, 10, future_length - reversal_point)
    elif trend_type == 'reversal_inverted_v':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
        reversal_point = future_length // 2
        future[reversal_point:] += np.linspace(0, 10, future_length - reversal_point)
    elif trend_type == 'reversal_double':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
        p1 = future_length // 3
        p2 = 2 * future_length // 3
        future[p1:p2] += np.linspace(0, 8, p2 - p1)
        future[p2:] -= np.linspace(0, 8, future_length - p2)
    elif trend_type == 'breakout_up':
        future = np.cumsum(np.random.randn(future_length) * 0.2) + historical[-1]
        breakout_point = future_length // 3
        future[breakout_point:] += np.linspace(0, 15, future_length - breakout_point)
    elif trend_type == 'breakout_down':
        future = np.cumsum(np.random.randn(future_length) * 0.2) + historical[-1]
        breakout_point = future_length // 3
        future[breakout_point:] -= np.linspace(0, 15, future_length - breakout_point)
    elif trend_type == 'breakout_late':
        future = np.cumsum(np.random.randn(future_length) * 0.2) + historical[-1]
        breakout_point = 2 * future_length // 3
        future[breakout_point:] += np.linspace(0, 15, future_length - breakout_point)
    elif trend_type == 'sawtooth':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
        for i in range(0, future_length, 8):
            future[i:min(i+4, future_length)] += np.linspace(0, 5, min(4, future_length-i))
            if i+4 < future_length:
                future[i+4:min(i+8, future_length)] -= np.linspace(0, 5, min(4, future_length-(i+4)))
    elif trend_type == 'step_up':
        future = np.cumsum(np.random.randn(future_length) * 0.2) + historical[-1]
        step_points = [future_length//4, future_length//2, 3*future_length//4]
        for sp in step_points:
            future[sp:] += np.random.uniform(3, 7)
    elif trend_type == 'step_down':
        future = np.cumsum(np.random.randn(future_length) * 0.2) + historical[-1]
        step_points = [future_length//4, future_length//2, 3*future_length//4]
        for sp in step_points:
            future[sp:] -= np.random.uniform(3, 7)
    elif trend_type == 'accelerating':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
        acceleration = np.linspace(0, 0.8, future_length)
        future += np.cumsum(acceleration)
    elif trend_type == 'decelerating':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
        deceleration = np.linspace(0.8, 0, future_length)
        future += np.cumsum(deceleration)
    elif trend_type == 'parabolic':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
        parabola = np.linspace(0, 1, future_length) ** 2 * 20
        future += parabola
    elif trend_type == 'exponential':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
        exp_growth = np.exp(np.linspace(0, 1, future_length)) * 5
        future += exp_growth
    elif trend_type == 'logarithmic':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
        log_growth = np.log(np.linspace(1, 20, future_length)) * 3
        future += log_growth
    elif trend_type == 'random_walk':
        future = np.cumsum(np.random.randn(future_length) * 0.5) + historical[-1]
    elif trend_type == 'random_walk_drift_up':
        future = np.cumsum(np.random.randn(future_length) * 0.5 + 0.2) + historical[-1]
    elif trend_type == 'random_walk_drift_down':
        future = np.cumsum(np.random.randn(future_length) * 0.5 - 0.2) + historical[-1]
    else:
        # Default: upward with noise
        future = np.cumsum(np.random.randn(future_length) * 0.5 + 0.3) + historical[-1]
    
    return (torch.tensor(historical, dtype=torch.float32, device=device), 
            torch.tensor(future, dtype=torch.float32, device=device))


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_result(
    historical_prices: torch.Tensor,
    future_prices: torch.Tensor,
    horizon: int,
    slope: float,
    intercept: float,
    cumulative_deviation: float,
    max_accum_deviation: float,
    all_horizons: List[float],
    all_cum_devs: List[float],
    all_slopes: List[float],
    min_horizon: int = 2,
    title: str = "Shortest Horizon Labeling (Constrained through f[0])",
    figsize: Tuple[int, int] = (18, 12)
) -> plt.Figure:
    """
    Enhanced visualization showing the trend line constrained through f[0].
    """
    
    # Move to CPU for plotting
    hist = historical_prices.cpu().numpy()
    fut = future_prices.cpu().numpy()
    
    # Get f[0] value
    f0 = fut[0]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # --- Plot 1: Full price history with constrained trend ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    full_prices = np.concatenate([hist, fut])
    time_steps = np.arange(len(full_prices))
    split_idx = len(hist)
    
    # Plot historical and future
    ax1.plot(time_steps[:split_idx], hist, 'b-', linewidth=2, label='Historical', alpha=0.7)
    ax1.plot(time_steps[split_idx:], fut, 'g-', linewidth=2, label='Future Prices', alpha=0.8)
    
    # Plot the trend line constrained through f[0]
    if horizon > 0:
        trend_time = np.arange(horizon)
        trend_prices = slope * trend_time + intercept
        ax1.plot(
            time_steps[split_idx:split_idx + horizon],
            trend_prices,
            'r-', linewidth=3, label=f'Constrained Trend: y = {slope:.4f}*t + {f0:.2f}'
        )
        
        # Mark the starting point f[0]
        ax1.scatter(split_idx, f0, color='red', s=100, zorder=5, 
                   label=f'f[0] = {f0:.2f} (fixed anchor)')
        
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
    ax1.set_title(f'Trend Line Constrained Through f[0]={f0:.2f}', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Future prices with constrained trend ---
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Plot future prices
    ax2.plot(fut, 'g-', linewidth=2, label='Actual Prices', alpha=0.7)
    ax2.scatter(np.arange(len(fut)), fut, color='green', s=30, alpha=0.5)
    
    # Plot constrained trend
    if horizon > 0:
        x_vals = np.arange(horizon)
        y_vals = fut[:horizon]
        trend_vals = slope * x_vals + intercept
        
        ax2.plot(x_vals, trend_vals, 'r-', linewidth=2.5, 
                label=f'Constrained: y={slope:.4f}t+{f0:.2f}')
        
        # Mark the anchor point f[0]
        ax2.scatter(0, f0, color='red', s=100, zorder=5, label=f'f[0]={f0:.2f}')
        
        # Show deviation bars
        deviations = np.abs(y_vals - trend_vals)
        ax2.bar(x_vals, deviations, bottom=np.minimum(y_vals, trend_vals),
                alpha=0.3, color='orange', width=0.6, label='Deviations')
        
        # Mark horizon
        ax2.axvline(x=horizon - 1, color='red', linestyle='-', 
                   linewidth=2, alpha=0.7)
        
        # Annotations
        total_dev = np.sum(deviations)
        avg_dev = np.mean(deviations)
        ax2.text(0.02, 0.98, f'Total Dev: {total_dev:.2f}\nAvg Dev: {avg_dev:.3f}',
                transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Future Time Steps', fontsize=11)
    ax2.set_ylabel('Price', fontsize=11)
    ax2.set_title(f'Constrained Trend Through f[0] (H={horizon})', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Cumulative Deviation Growth ---
    ax3 = fig.add_subplot(gs[0, 2])
    
    if all_horizons and all_cum_devs:
        ax3.plot(all_horizons, all_cum_devs, 'b-', linewidth=2.5, label='Cumulative Deviation')
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
        ax3.set_ylabel('Cumulative Absolute Deviation', fontsize=11)
        ax3.set_title('Cumulative Deviation Growth', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Slope vs Horizon ---
    ax4 = fig.add_subplot(gs[1, 0])
    
    if all_horizons and all_slopes:
        ax4.plot(all_horizons, all_slopes, 'purple', linewidth=2.5, label='Slope')
        ax4.scatter(horizon, slope, color='red', s=100, zorder=5,
                   label=f'Selected: m={slope:.4f}')
        ax4.axvline(x=horizon, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=1)
        
        ax4.set_xlabel('Horizon (h)', fontsize=11)
        ax4.set_ylabel('Slope', fontsize=11)
        ax4.set_title('Slope Evolution', fontsize=12, fontweight='bold')
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)
    
    # --- Plot 5: Residuals Analysis ---
    ax5 = fig.add_subplot(gs[1, 1])
    
    if horizon > 0:
        x_vals = np.arange(horizon)
        y_vals = fut[:horizon]
        trend_vals = slope * x_vals + intercept
        residuals = y_vals - trend_vals
        
        ax5.scatter(x_vals, residuals, color='blue', s=30, alpha=0.6)
        ax5.axhline(y=0, color='red', linestyle='-', linewidth=1.5)
        ax5.axhline(y=np.std(residuals), color='orange', linestyle='--', alpha=0.5)
        ax5.axhline(y=-np.std(residuals), color='orange', linestyle='--', alpha=0.5)
        
        # Mark f[0] anchor
        ax5.axvline(x=0, color='green', linestyle=':', alpha=0.5, label='f[0] anchor')
        ax5.scatter(0, 0, color='green', s=50, zorder=5, label='residual=0 at f[0]')
        
        ax5.set_xlabel('Time Step', fontsize=11)
        ax5.set_ylabel('Residual', fontsize=11)
        ax5.set_title(f'Residuals (Std: {np.std(residuals):.3f})', fontsize=12, fontweight='bold')
        ax5.legend(loc='best', fontsize=8)
        ax5.grid(True, alpha=0.3)
    
    # --- Plot 6: Metrics Summary ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
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
    │     SHORTEST HORIZON - CONSTRAINED THROUGH f[0]            │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  f[0] (ANCHOR)        : {intercept:>10.2f}                  │
    │  HORIZON (H)          : {horizon:>4}                        │
    │  SLOPE (m)            : {slope:>10.6f}                      │
    │  TREND LINE           : y = {slope:.4f}*t + {intercept:.2f} │
    │  R²                   : {r2:>10.4f}                         │
    │                                                             │
    │  THRESHOLD            : {max_accum_deviation:>10.2f}        │
    │  CUMULATIVE DEVIATION : {cumulative_deviation:>10.2f}       │
    │  THRESHOLD REACHED?   : {str(threshold_reached):>10}        │
    │                                                             │
    │  MAE                   : {mae:>10.3f}                       │
    │  RMSE                  : {rmse:>10.3f}                       │
    │  MAX DEVIATION         : {max_dev:>10.3f}                   │
    │                                                             │
    │  MIN HORIZON           : {min_horizon:>4}                   │
    │  MAX HORIZON           : {horizon:>4} (selected)            │
    └─────────────────────────────────────────────────────────────┘
    """
    
    ax6.text(0.5, 0.5, metrics_text, fontsize=10, verticalalignment='center',
             horizontalalignment='center', family='monospace', transform=ax6.transAxes)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig


# ============================================================================
# BATCH TESTING AND ANALYSIS
# ============================================================================

def analyze_multiple_trends(
    trend_types: List[str],
    threshold: float = 12.0,
    min_horizon: int = 3,
    max_horizon: int = 35,
    n_samples: int = 5,
    device: str = DEVICE
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze multiple trend types with multiple samples each.
    """
    results = defaultdict(lambda: {'horizons': [], 'slopes': [], 'cum_devs': [], 'reached': []})
    
    for trend in trend_types:
        for seed in range(n_samples):
            _, future = generate_synthetic_data(
                history_length=80,
                future_length=40,
                trend_type=trend,
                noise_scale=1.5,
                seed=seed * 100 + hash(trend) % 1000,
                device=device
            )
            
            h, m, b, cum, _, _, _ = find_shortest_horizon(
                future_prices=future,
                max_accum_deviation=threshold,
                min_horizon=min_horizon,
                max_horizon=max_horizon,
                device=device
            )
            
            results[trend]['horizons'].append(h)
            results[trend]['slopes'].append(m)
            results[trend]['cum_devs'].append(cum)
            results[trend]['reached'].append(cum >= threshold)
    
    # Compute statistics
    summary: Dict[str, Dict[str, Any]] = {}
    for trend, data in results.items():
        summary[trend] = {
            'horizon_mean': float(np.mean(data['horizons'])),
            'horizon_std': float(np.std(data['horizons'])),
            'horizon_min': int(np.min(data['horizons'])),
            'horizon_max': int(np.max(data['horizons'])),
            'slope_mean': float(np.mean(data['slopes'])),
            'slope_std': float(np.std(data['slopes'])),
            'cum_dev_mean': float(np.mean(data['cum_devs'])),
            'reached_pct': float(np.mean(data['reached']) * 100),
            'raw_data': data
        }
    
    return summary


def print_analysis_summary(summary: Dict[str, Dict[str, Any]], title: str = "Analysis Summary"):
    """Print a formatted summary of the analysis."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"\n{'Trend':>18} | {'H mean':>8} | {'H std':>8} | {'Slope mean':>12} | {'Reached %':>10}")
    print("-" * 70)
    
    for trend, stats in sorted(summary.items()):
        print(f"  {trend:>16} | {stats['horizon_mean']:>8.2f} | {stats['horizon_std']:>8.2f} | {stats['slope_mean']:>12.4f} | {stats['reached_pct']:>10.1f}%")


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main_demo():
    """
    Comprehensive demonstration with many examples.
    """
    print("=" * 80)
    print("SHORTEST HORIZON LABELING - CONSTRAINED THROUGH f[0]")
    print("=" * 80)
    print(f"Using device: {DEVICE}")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # ========== EXAMPLE 1: Basic Trend Types ==========
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Trend Types")
    print("="*80)
    
    basic_trends = ['upward', 'downward', 'sideways', 'volatile']
    threshold = 12.0
    
    print(f"\nTesting basic trend types (threshold={threshold})")
    print("-" * 70)
    print(f"{'Trend':>12} | {'H':>4} | {'Slope':>10} | {'Cum Dev':>10} | {'Status':>10}")
    print("-" * 70)
    
    for trend in basic_trends:
        _, future = generate_synthetic_data(
            history_length=80,
            future_length=40,
            trend_type=trend,
            noise_scale=1.5,
            seed=hash(trend) % 1000,
            device=DEVICE
        )
        
        h, m, b, cum, _, _, _ = find_shortest_horizon(
            future_prices=future,
            max_accum_deviation=threshold,
            min_horizon=3,
            max_horizon=35,
            device=DEVICE
        )
        
        reached = "✓" if cum >= threshold else "✗"
        print(f"  {trend:>10} | {h:>4} | {m:>10.4f} | {cum:>10.2f} | {reached:>10}")
        
        # Visualize first example
        if trend == 'upward':
            history, future = generate_synthetic_data(
                history_length=80,
                future_length=40,
                trend_type=trend,
                noise_scale=1.5,
                seed=42,
                device=DEVICE
            )
            h, m, b, cum, all_h, all_c, all_s = find_shortest_horizon(
                future_prices=future,
                max_accum_deviation=threshold,
                min_horizon=3,
                max_horizon=35,
                device=DEVICE,
                verbose=True
            )
            
            visualize_result(
                historical_prices=history,
                future_prices=future,
                horizon=h,
                slope=m,
                intercept=b,
                cumulative_deviation=cum,
                max_accum_deviation=threshold,
                all_horizons=all_h,
                all_cum_devs=all_c,
                all_slopes=all_s,
                min_horizon=3,
                title=f"Basic: {trend.upper()} Trend - H={h}, m={m:.4f}"
            )
    
    # ========== EXAMPLE 2: Strength Variations ==========
    print("\n" + "="*80)
    print("EXAMPLE 2: Trend Strength Variations")
    print("="*80)
    
    strength_trends = ['upward_weak', 'upward', 'upward_strong', 'upward_steep']
    threshold = 12.0
    
    print(f"\nComparing trend strengths (threshold={threshold})")
    print("-" * 70)
    print(f"{'Strength':>14} | {'H':>4} | {'Slope':>10} | {'Cum Dev':>10} | {'Status':>10}")
    print("-" * 70)
    
    for trend in strength_trends:
        _, future = generate_synthetic_data(
            history_length=80,
            future_length=40,
            trend_type=trend,
            noise_scale=1.5,
            seed=hash(trend) % 1000,
            device=DEVICE
        )
        
        h, m, b, cum, _, _, _ = find_shortest_horizon(
            future_prices=future,
            max_accum_deviation=threshold,
            min_horizon=3,
            max_horizon=35,
            device=DEVICE
        )
        
        reached = "✓" if cum >= threshold else "✗"
        label = trend.replace('upward_', '')
        print(f"  {label:>12} | {h:>4} | {m:>10.4f} | {cum:>10.2f} | {reached:>10}")
    
    # ========== EXAMPLE 3: Volatility Variations ==========
    print("\n" + "="*80)
    print("EXAMPLE 3: Volatility Variations")
    print("="*80)
    
    volatility_trends = ['sideways_tight', 'sideways', 'sideways_wide', 'volatile_mild', 'volatile', 'volatile_extreme']
    threshold = 12.0
    
    print(f"\nComparing volatility levels (threshold={threshold})")
    print("-" * 70)
    print(f"{'Volatility':>14} | {'H':>4} | {'Slope':>10} | {'Cum Dev':>10} | {'Status':>10}")
    print("-" * 70)
    
    for trend in volatility_trends:
        _, future = generate_synthetic_data(
            history_length=80,
            future_length=40,
            trend_type=trend,
            noise_scale=1.5,
            seed=hash(trend) % 1000,
            device=DEVICE
        )
        
        h, m, b, cum, _, _, _ = find_shortest_horizon(
            future_prices=future,
            max_accum_deviation=threshold,
            min_horizon=3,
            max_horizon=35,
            device=DEVICE
        )
        
        reached = "✓" if cum >= threshold else "✗"
        label = trend.replace('sideways_', '').replace('volatile_', '')
        print(f"  {label:>12} | {h:>4} | {m:>10.4f} | {cum:>10.2f} | {reached:>10}")
    
    # ========== EXAMPLE 4: Pattern Types ==========
    print("\n" + "="*80)
    print("EXAMPLE 4: Pattern Types")
    print("="*80)
    
    pattern_trends = ['reversal_v', 'reversal_inverted_v', 'reversal_double', 'breakout_up', 'breakout_down']
    threshold = 12.0
    
    print(f"\nTesting pattern types (threshold={threshold})")
    print("-" * 70)
    print(f"{'Pattern':>16} | {'H':>4} | {'Slope':>10} | {'Cum Dev':>10} | {'Status':>10}")
    print("-" * 70)
    
    for trend in pattern_trends:
        _, future = generate_synthetic_data(
            history_length=80,
            future_length=40,
            trend_type=trend,
            noise_scale=1.5,
            seed=hash(trend) % 1000,
            device=DEVICE
        )
        
        h, m, b, cum, _, _, _ = find_shortest_horizon(
            future_prices=future,
            max_accum_deviation=threshold,
            min_horizon=3,
            max_horizon=35,
            device=DEVICE
        )
        
        reached = "✓" if cum >= threshold else "✗"
        label = trend.replace('_', ' ').title()
        print(f"  {label:>14} | {h:>4} | {m:>10.4f} | {cum:>10.2f} | {reached:>10}")
        
        # Visualize reversal
        if trend == 'reversal_v':
            history, future = generate_synthetic_data(
                history_length=80,
                future_length=40,
                trend_type=trend,
                noise_scale=1.5,
                seed=123,
                device=DEVICE
            )
            h, m, b, cum, all_h, all_c, all_s = find_shortest_horizon(
                future_prices=future,
                max_accum_deviation=threshold,
                min_horizon=3,
                max_horizon=35,
                device=DEVICE
            )
            
            visualize_result(
                historical_prices=history,
                future_prices=future,
                horizon=h,
                slope=m,
                intercept=b,
                cumulative_deviation=cum,
                max_accum_deviation=threshold,
                all_horizons=all_h,
                all_cum_devs=all_c,
                all_slopes=all_s,
                min_horizon=3,
                title=f"Pattern: V-Reversal - H={h}, m={m:.4f}"
            )
    
    # ========== EXAMPLE 5: Complex Mathematical Trends ==========
    print("\n" + "="*80)
    print("EXAMPLE 5: Complex Mathematical Trends")
    print("="*80)
    
    math_trends = ['accelerating', 'decelerating', 'parabolic', 'exponential', 'logarithmic']
    threshold = 12.0
    
    print(f"\nTesting mathematical trends (threshold={threshold})")
    print("-" * 70)
    print(f"{'Math Type':>14} | {'H':>4} | {'Slope':>10} | {'Cum Dev':>10} | {'Status':>10}")
    print("-" * 70)
    
    for trend in math_trends:
        _, future = generate_synthetic_data(
            history_length=80,
            future_length=40,
            trend_type=trend,
            noise_scale=1.0,
            seed=hash(trend) % 1000,
            device=DEVICE
        )
        
        h, m, b, cum, _, _, _ = find_shortest_horizon(
            future_prices=future,
            max_accum_deviation=threshold,
            min_horizon=3,
            max_horizon=35,
            device=DEVICE
        )
        
        reached = "✓" if cum >= threshold else "✗"
        label = trend.title()
        print(f"  {label:>12} | {h:>4} | {m:>10.4f} | {cum:>10.2f} | {reached:>10}")
    
    # ========== EXAMPLE 6: Random Walk Variations ==========
    print("\n" + "="*80)
    print("EXAMPLE 6: Random Walk Variations")
    print("="*80)
    
    random_trends = ['random_walk', 'random_walk_drift_up', 'random_walk_drift_down']
    threshold = 12.0
    
    print(f"\nTesting random walks (threshold={threshold})")
    print("-" * 70)
    print(f"{'Random Walk':>16} | {'H':>4} | {'Slope':>10} | {'Cum Dev':>10} | {'Status':>10}")
    print("-" * 70)
    
    for trend in random_trends:
        _, future = generate_synthetic_data(
            history_length=80,
            future_length=40,
            trend_type=trend,
            noise_scale=1.5,
            seed=42,
            device=DEVICE
        )
        
        h, m, b, cum, _, _, _ = find_shortest_horizon(
            future_prices=future,
            max_accum_deviation=threshold,
            min_horizon=3,
            max_horizon=35,
            device=DEVICE
        )
        
        reached = "✓" if cum >= threshold else "✗"
        label = trend.replace('_', ' ').title()
        print(f"  {label:>14} | {h:>4} | {m:>10.4f} | {cum:>10.2f} | {reached:>10}")
    
    # ========== EXAMPLE 7: Different Thresholds ==========
    print("\n" + "="*80)
    print("EXAMPLE 7: Effect of Different Thresholds")
    print("="*80)
    
    _, future = generate_synthetic_data(
        history_length=80,
        future_length=40,
        trend_type='upward',
        noise_scale=1.5,
        seed=42,
        device=DEVICE
    )
    
    thresholds = [3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]
    
    print(f"\nUpward trend - effect of threshold")
    print("-" * 70)
    print(f"{'Threshold':>10} | {'H':>4} | {'Slope':>10} | {'Cum Dev':>10} | {'Status':>10}")
    print("-" * 70)
    
    for thresh in thresholds:
        h, m, b, cum, _, _, _ = find_shortest_horizon(
            future_prices=future,
            max_accum_deviation=thresh,
            min_horizon=3,
            max_horizon=35,
            device=DEVICE
        )
        
        reached = "✓" if cum >= thresh else "✗"
        print(f"  {thresh:>8.1f} | {h:>4} | {m:>10.4f} | {cum:>10.2f} | {reached:>10}")
    
    # ========== EXAMPLE 8: Statistical Analysis ==========
    print("\n" + "="*80)
    print("EXAMPLE 8: Statistical Analysis (Multiple Samples)")
    print("="*80)
    
    trend_groups = {
        'Upward Trends': ['upward', 'upward_strong', 'upward_weak'],
        'Downward Trends': ['downward', 'downward_strong', 'downward_weak'],
        'Sideways Trends': ['sideways', 'sideways_tight', 'sideways_wide'],
        'Volatile Trends': ['volatile', 'volatile_mild', 'volatile_extreme'],
        'Patterns': ['reversal_v', 'breakout_up', 'breakout_down'],
        'Math Trends': ['accelerating', 'decelerating', 'parabolic']
    }
    
    for group_name, trends in trend_groups.items():
        print(f"\n{group_name}:")
        summary = analyze_multiple_trends(
            trend_types=trends,
            threshold=12.0,
            min_horizon=3,
            max_horizon=35,
            n_samples=10
        )
        print_analysis_summary(summary, f"Analysis: {group_name}")
    
    # ========== EXAMPLE 9: Visual Comparison ==========
    print("\n" + "="*80)
    print("EXAMPLE 9: Visual Comparison of Different Trends")
    print("="*80)
    
    compare_trends = ['upward', 'downward', 'volatile', 'reversal_v', 'breakout_up']
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for idx, trend in enumerate(compare_trends):
        ax = axes[idx]
        
        _, future = generate_synthetic_data(
            history_length=60,
            future_length=30,
            trend_type=trend,
            noise_scale=1.5,
            seed=hash(trend) % 1000,
            device=DEVICE
        )
        
        fut_np = future.cpu().numpy()
        
        h, m, b, cum, _, _, _ = find_shortest_horizon(
            future_prices=future,
            max_accum_deviation=10.0,
            min_horizon=3,
            max_horizon=30,
            device=DEVICE
        )
        
        # Plot future prices
        ax.plot(fut_np, 'g-', linewidth=2, alpha=0.7)
        
        # Plot trend
        if h > 0:
            trend_vals = m * np.arange(h) + b
            ax.plot(trend_vals, 'r-', linewidth=2.5)
            ax.axvline(x=h-1, color='red', linestyle='-', alpha=0.7)
            ax.axvline(x=0, color='green', linestyle=':', alpha=0.5)
        
        ax.set_title(f'{trend.replace("_", " ").title()}\nH={h}, m={m:.3f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
    
    plt.suptitle('Comparison of Different Trend Types (Constrained Through f[0])', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\nKey Findings:")
    print("  ✓ Trend line is CONSTRAINED through f[0] (current price)")
    print("  ✓ Slope is calculated as: m = Σ(t * (y - f[0])) / Σ(t²)")
    print("  ✓ Stronger trends → shorter horizons")
    print("  ✓ Higher volatility → shorter horizons (more deviation)")
    print("  ✓ Higher thresholds → longer horizons")
    print("  ✓ Patterns with reversals may reach threshold earlier")
    print("  ✓ Random walks have unpredictable horizons")
    
    print("\nUse Cases:")
    print("  • Finding minimum lookback period for trend identification")
    print("  • Dynamic stop-loss placement based on deviation")
    print("  • Early warning system for trend changes")
    print("  • Risk management based on cumulative deviation")
    print("  • Feature engineering for ML models")
    
    print("\nTrend Line Formula:")
    print("  y = m * t + f[0]")
    print("  where:")
    print("    • y = predicted price at time t")
    print("    • m = slope (trend strength)")
    print("    • t = time step (0, 1, 2, ...)")
    print("    • f[0] = current price (fixed anchor)")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main_demo()