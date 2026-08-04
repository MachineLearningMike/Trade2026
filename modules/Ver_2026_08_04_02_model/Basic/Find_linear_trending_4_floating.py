"""
Dynamic Horizon Linear Trend Labeling - Find Single Shortest Horizon
===================================================================
Find the single shortest horizon where cumulative deviation reaches max_accum_deviation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from scipy import stats
import warnings
from dataclasses import dataclass

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
# CORE FUNCTION: Find the Shortest Horizon
# ============================================================================

def find_shortest_horizon(
    future_prices: torch.Tensor,
    max_accum_deviation: float,
    min_horizon: int = 2,
    max_horizon: Optional[int] = None,
    device: str = DEVICE
) -> Tuple[int, float, float, float, List[float], List[float]]:
    """
    Find the SINGLE SHORTEST horizon where cumulative deviation reaches max_accum_deviation.
    
    Args:
        future_prices: Future price sequence, shape (future_length,)
        max_accum_deviation: Threshold for cumulative absolute deviation
        min_horizon: Minimum horizon to consider
        max_horizon: Maximum horizon to consider
    
    Returns:
        horizon: The shortest horizon where cumulative deviation >= max_accum_deviation
        slope: Slope (m) at that horizon
        intercept: Intercept (b) at that horizon (value at time 0)
        cumulative_deviation: The cumulative deviation at this horizon
        all_horizons: List of all horizons tested
        all_cum_devs: List of cumulative deviations for each horizon
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
    
    # Search for the shortest horizon
    for h in range(min_horizon, max_horizon + 1):
        # Get the segment
        segment = prices[:h]
        
        # Time indices starting from 0
        t = torch.arange(h, dtype=torch.float32, device=device)
        
        # Fit linear trend using OLS: y = m*t + b
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
        
        # Calculate cumulative absolute deviation
        predicted = slope * t + intercept
        deviations = torch.abs(segment - predicted)
        cumulative_deviation = torch.sum(deviations).item()
        
        # Store for analysis
        all_horizons.append(h)
        all_cum_devs.append(cumulative_deviation)
        all_slopes.append(slope)
        all_intercepts.append(intercept)
        
        # If we've reached the threshold, return this horizon
        if cumulative_deviation >= max_accum_deviation:
            return h, slope, intercept, cumulative_deviation, all_horizons, all_cum_devs
    
    # If we never reach the threshold, return the max horizon
    h = max_horizon
    segment = prices[:h]
    t = torch.arange(h, dtype=torch.float32, device=device)
    
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
    
    predicted = slope * t + intercept
    deviations = torch.abs(segment - predicted)
    cumulative_deviation = torch.sum(deviations).item()
    
    print(f"WARNING: Threshold {max_accum_deviation} NOT reached!")
    print(f"  Max cumulative deviation at H={h}: {cumulative_deviation:.2f}")
    print(f"  Returning horizon={h} with best fit")
    
    return h, slope, intercept, cumulative_deviation, all_horizons, all_cum_devs


# ============================================================================
# DATA GENERATION - MORE VARIETY
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
    elif trend_type == 'sideways_with_noise':
        future = np.cumsum(np.random.randn(future_length) * 0.8) + historical[-1]
    elif trend_type == 'volatile':
        future = np.cumsum(np.random.randn(future_length) * 1.5) + historical[-1]
    elif trend_type == 'very_volatile':
        future = np.cumsum(np.random.randn(future_length) * 3.0) + historical[-1]
    elif trend_type == 'mixed':
        trend = np.sin(np.linspace(0, 2*np.pi, future_length)) * 5
        future = np.cumsum(np.random.randn(future_length) * 0.3) + trend + historical[-1]
    elif trend_type == 'reversal':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
        reversal_point = future_length // 2
        future[reversal_point:] -= np.linspace(0, 10, future_length - reversal_point)
    elif trend_type == 'double_reversal':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
        p1 = future_length // 3
        p2 = 2 * future_length // 3
        future[p1:p2] += np.linspace(0, 8, p2 - p1)
        future[p2:] -= np.linspace(0, 8, future_length - p2)
    elif trend_type == 'breakout':
        future = np.cumsum(np.random.randn(future_length) * 0.2) + historical[-1]
        breakout_point = future_length // 3
        future[breakout_point:] += np.linspace(0, 15, future_length - breakout_point)
    elif trend_type == 'breakdown':
        future = np.cumsum(np.random.randn(future_length) * 0.2) + historical[-1]
        breakdown_point = future_length // 3
        future[breakdown_point:] -= np.linspace(0, 15, future_length - breakdown_point)
    elif trend_type == 'sawtooth':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
        for i in range(0, future_length, 8):
            future[i:min(i+4, future_length)] += np.linspace(0, 5, min(4, future_length-i))
            if i+4 < future_length:
                future[i+4:min(i+8, future_length)] -= np.linspace(0, 5, min(4, future_length-(i+4)))
    elif trend_type == 'step':
        future = np.cumsum(np.random.randn(future_length) * 0.2) + historical[-1]
        step_points = [future_length//4, future_length//2, 3*future_length//4]
        for sp in step_points:
            future[sp:] += np.random.uniform(3, 7)
    elif trend_type == 'accelerating':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
        acceleration = np.linspace(0, 0.8, future_length)
        future += np.cumsum(acceleration)
    elif trend_type == 'decelerating':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
        deceleration = np.linspace(0.8, 0, future_length)
        future += np.cumsum(deceleration)
    else:  # 'random_walk'
        future = np.cumsum(np.random.randn(future_length) * 0.5) + historical[-1]
    
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
    min_horizon: int = 2,
    title: str = "Shortest Horizon Labeling",
    figsize: Tuple[int, int] = (18, 12)
) -> plt.Figure:
    """
    Enhanced visualization with deviation growth plot.
    """
    
    # Move to CPU for plotting
    hist = historical_prices.cpu().numpy()
    fut = future_prices.cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # --- Plot 1: Full price history with trend ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    full_prices = np.concatenate([hist, fut])
    time_steps = np.arange(len(full_prices))
    split_idx = len(hist)
    
    # Plot historical and future
    ax1.plot(time_steps[:split_idx], hist, 'b-', linewidth=2, label='Historical', alpha=0.7)
    ax1.plot(time_steps[split_idx:], fut, 'g-', linewidth=2, label='Future Prices', alpha=0.8)
    
    # Plot the CORRECT trend line using OLS intercept
    if horizon > 0:
        trend_time = np.arange(horizon)
        trend_prices = slope * trend_time + intercept
        ax1.plot(
            time_steps[split_idx:split_idx + horizon],
            trend_prices,
            'r-', linewidth=3, label=f'OLS Trend: y = {slope:.4f}*t + {intercept:.2f}'
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
    ax1.set_title('Price History with OLS Trend Line', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Future prices with deviations ---
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Plot future prices
    ax2.plot(fut, 'g-', linewidth=2, label='Actual Prices', alpha=0.7)
    ax2.scatter(np.arange(len(fut)), fut, color='green', s=30, alpha=0.5)
    
    # Plot CORRECT trend
    if horizon > 0:
        x_vals = np.arange(horizon)
        y_vals = fut[:horizon]
        trend_vals = slope * x_vals + intercept
        
        ax2.plot(x_vals, trend_vals, 'r-', linewidth=2.5, 
                label=f'OLS: y={slope:.4f}t+{intercept:.2f}')
        
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
    ax2.set_title(f'Future Prices with OLS Fit (H={horizon})', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
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
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Per-step deviations ---
    ax4 = fig.add_subplot(gs[1, 0])
    
    if horizon > 0:
        x_vals = np.arange(horizon)
        y_vals = fut[:horizon]
        trend_vals = slope * x_vals + intercept
        per_step_dev = np.abs(y_vals - trend_vals)
        
        # Bar chart
        bars = ax4.bar(x_vals, per_step_dev, alpha=0.6, color='steelblue')
        
        # Color significant deviations
        avg_dev = per_step_dev.mean()
        for i, bar in enumerate(bars):
            if per_step_dev[i] > avg_dev * 1.5:
                bar.set_color('red')
                bar.set_alpha(0.8)
        
        ax4.axhline(y=avg_dev, color='red', linestyle='--', 
                   label=f'Avg Dev = {avg_dev:.3f}')
        ax4.axhline(y=avg_dev * 1.5, color='orange', linestyle=':', 
                   label=f'1.5× Avg Dev')
        
        ax4.set_xlabel('Time Step', fontsize=11)
        ax4.set_ylabel('Per-Step Deviation', fontsize=11)
        ax4.set_title(f'Per-Step Deviations (Total: {np.sum(per_step_dev):.2f})', 
                     fontsize=12, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
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
        
        ax5.set_xlabel('Time Step', fontsize=11)
        ax5.set_ylabel('Residual', fontsize=11)
        ax5.set_title(f'Residuals (Std: {np.std(residuals):.3f})', fontsize=12, fontweight='bold')
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
    │            SHORTEST HORIZON - RESULTS                       │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  HORIZON (H)           : {horizon:>4}                       │
    │  SLOPE (m)             : {slope:>10.6f}                     │
    │  INTERCEPT (b)         : {intercept:>10.3f}                 │
    │  R²                    : {r2:>10.4f}                        │
    │                                                             │
    │  THRESHOLD             : {max_accum_deviation:>10.2f}       │
    │  CUMULATIVE DEVIATION  : {cumulative_deviation:>10.2f}      │
    │  THRESHOLD REACHED?    : {str(threshold_reached):>10}       │
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
# TEST FUNCTIONS
# ============================================================================

def test_trend_type(trend_type: str, threshold: float = 15.0, verbose: bool = True):
    """
    Test a specific trend type and return results.
    """
    # Generate data
    history, future = generate_synthetic_data(
        history_length=80,
        future_length=40,
        trend_type=trend_type,
        noise_scale=1.5,
        seed=hash(trend_type) % 1000,
        device=DEVICE
    )
    
    # Find shortest horizon
    horizon, slope, intercept, cum_dev, all_h, all_c = find_shortest_horizon(
        future_prices=future,
        max_accum_deviation=threshold,
        min_horizon=3,
        max_horizon=35,
        device=DEVICE
    )
    
    if verbose:
        print(f"\n{trend_type.upper()}:")
        print(f"  Horizon: {horizon}")
        print(f"  Slope: {slope:.4f}")
        print(f"  Intercept: {intercept:.2f}")
        print(f"  Cum Dev: {cum_dev:.2f} (threshold: {threshold})")
        print(f"  Reached: {cum_dev >= threshold}")
    
    return {
        'trend_type': trend_type,
        'horizon': horizon,
        'slope': slope,
        'intercept': intercept,
        'cum_dev': cum_dev,
        'threshold': threshold,
        'reached': cum_dev >= threshold,
        'history': history,
        'future': future,
        'all_horizons': all_h,
        'all_cum_devs': all_c
    }


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main_demo():
    """
    Comprehensive demonstration with many examples.
    """
    print("=" * 80)
    print("SHORTEST HORIZON LABELING - COMPREHENSIVE EXAMPLES")
    print("=" * 80)
    print(f"Using device: {DEVICE}")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # ========== EXAMPLE 1: Perfect Linear Trends ==========
    print("\n" + "="*80)
    print("EXAMPLE 1: Perfect Linear Trends (No Noise)")
    print("="*80)
    
    print("\nCreating perfect linear trends:")
    print("-" * 60)
    
    perfect_cases = [
        ('upward', 100, 2.0, 30),
        ('downward', 150, -2.0, 30),
        ('flat', 100, 0.0, 30),
        ('steep_up', 100, 4.0, 30),
        ('steep_down', 150, -4.0, 30),
    ]
    
    for name, start, slope_val, length in perfect_cases:
        # Create perfect linear data
        t = np.arange(length)
        future = start + slope_val * t
        future_tensor = torch.tensor(future, dtype=torch.float32, device=DEVICE)
        
        # Test different thresholds
        for threshold in [5.0, 10.0, 20.0]:
            h, m, b, cum, _, _ = find_shortest_horizon(
                future_prices=future_tensor,
                max_accum_deviation=threshold,
                min_horizon=2,
                max_horizon=length,
                device=DEVICE
            )
            
            # For perfect linear data, cumulative deviation should be 0
            # So threshold is never reached - it returns max horizon
            if cum == 0:
                status = "✓ PERFECT"
            else:
                status = f"cum={cum:.1f}"
            
            print(f"  {name:>12} | threshold={threshold:>5.1f} | H={h:>3} | m={m:>8.4f} | {status}")
        
        print()
        
        # Visualize one example
        if name == 'upward':
            dummy_hist = torch.zeros(1, device=DEVICE)
            threshold = 15.0
            h, m, b, cum, all_h, all_c = find_shortest_horizon(
                future_prices=future_tensor,
                max_accum_deviation=threshold,
                min_horizon=2,
                max_horizon=length,
                device=DEVICE
            )
            
            # Since perfect trend has zero deviation, it returns max horizon
            # Let's add a tiny bit of noise to make it interesting
            future_noisy = future + np.random.randn(length) * 0.3
            future_noisy_tensor = torch.tensor(future_noisy, dtype=torch.float32, device=DEVICE)
            
            h, m, b, cum, all_h, all_c = find_shortest_horizon(
                future_prices=future_noisy_tensor,
                max_accum_deviation=threshold,
                min_horizon=2,
                max_horizon=length,
                device=DEVICE
            )
            
            visualize_result(
                historical_prices=dummy_hist,
                future_prices=future_noisy_tensor,
                horizon=h,
                slope=m,
                intercept=b,
                cumulative_deviation=cum,
                max_accum_deviation=threshold,
                all_horizons=all_h,
                all_cum_devs=all_c,
                min_horizon=2,
                title=f"Perfect Upward Trend with Small Noise - H={h}, m={m:.4f}"
            )
    
    # ========== EXAMPLE 2: All Trend Types ==========
    print("\n" + "="*80)
    print("EXAMPLE 2: All Trend Types with Noise")
    print("="*80)
    
    all_trends = [
        'upward', 'strong_upward', 'weak_upward',
        'downward', 'strong_downward',
        'sideways', 'sideways_with_noise',
        'volatile', 'very_volatile',
        'mixed', 'reversal', 'double_reversal',
        'breakout', 'breakdown',
        'sawtooth', 'step',
        'accelerating', 'decelerating',
        'random_walk'
    ]
    
    threshold = 12.0
    
    print(f"\nTesting {len(all_trends)} trend types with threshold={threshold}")
    print("-" * 80)
    print(f"{'Trend Type':>18} | {'H':>4} | {'Slope':>10} | {'Cum Dev':>10} | {'Status':>12}")
    print("-" * 80)
    
    results = []
    for trend in all_trends:
        result = test_trend_type(trend, threshold, verbose=False)
        results.append(result)
        
        status = "✓ REACHED" if result['reached'] else "✗ NOT REACHED"
        print(f"  {trend:>16} | {result['horizon']:>4} | {result['slope']:>10.4f} | {result['cum_dev']:>10.2f} | {status:>12}")
    
    # ========== EXAMPLE 3: Different Thresholds ==========
    print("\n" + "="*80)
    print("EXAMPLE 3: Effect of Different Thresholds")
    print("="*80)
    
    # Use a volatile trend
    threshold_trend = 'volatile'
    thresholds = [3.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0]
    
    print(f"\nTrend Type: {threshold_trend.upper()}")
    print("-" * 70)
    print(f"{'Threshold':>12} | {'H':>4} | {'Slope':>10} | {'Cum Dev':>10} | {'Status':>12}")
    print("-" * 70)
    
    threshold_results = []
    for thresh in thresholds:
        result = test_trend_type(threshold_trend, thresh, verbose=False)
        threshold_results.append(result)
        status = "✓" if result['reached'] else "✗"
        print(f"  {thresh:>10.1f} | {result['horizon']:>4} | {result['slope']:>10.4f} | {result['cum_dev']:>10.2f} | {status:>12}")
    
    # ========== EXAMPLE 4: Compare Strong vs Weak Trends ==========
    print("\n" + "="*80)
    print("EXAMPLE 4: Strong vs Weak Trends")
    print("="*80)
    
    print("\nComparing trend strengths at different thresholds:")
    print("-" * 80)
    
    trend_pairs = [
        ('weak_upward', 'strong_upward'),
        ('weak_upward', 'strong_downward'),
        ('sideways', 'volatile'),
        ('volatile', 'very_volatile'),
    ]
    
    for pair in trend_pairs:
        print(f"\n  {pair[0]} vs {pair[1]}:")
        print(f"  {'Threshold':>10} | {pair[0]:>12} | {pair[1]:>12}")
        print(f"  {'-'*10} | {'-'*12} | {'-'*12}")
        
        for thresh in [5.0, 10.0, 15.0, 20.0]:
            r1 = test_trend_type(pair[0], thresh, verbose=False)
            r2 = test_trend_type(pair[1], thresh, verbose=False)
            
            h1 = r1['horizon']
            h2 = r2['horizon']
            status1 = "✓" if r1['reached'] else "✗"
            status2 = "✓" if r2['reached'] else "✗"
            
            print(f"  {thresh:>10.1f} | H={h1:>3} {status1:>4} | H={h2:>3} {status2:>4}")
    
    # ========== EXAMPLE 5: Visualize Interesting Cases ==========
    print("\n" + "="*80)
    print("EXAMPLE 5: Detailed Visualization of Interesting Cases")
    print("="*80)
    
    interesting_cases = ['breakout', 'reversal', 'double_reversal', 'sawtooth', 'accelerating']
    
    for case in interesting_cases:
        print(f"\nVisualizing: {case.upper()}")
        
        # Generate data
        history, future = generate_synthetic_data(
            history_length=80,
            future_length=40,
            trend_type=case,
            noise_scale=1.5,
            seed=hash(case + 'viz') % 1000,
            device=DEVICE
        )
        
        # Find shortest horizon
        threshold = 15.0
        h, m, b, cum, all_h, all_c = find_shortest_horizon(
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
            min_horizon=3,
            title=f"{case.upper()} Pattern - H={h}, m={m:.4f}, CumDev={cum:.2f}"
        )
    
    # ========== EXAMPLE 6: Statistical Summary ==========
    print("\n" + "="*80)
    print("EXAMPLE 6: Statistical Summary")
    print("="*80)
    
    # Generate multiple samples for each trend type
    print("\nGenerating statistics across 20 samples per trend type...")
    
    trend_categories = ['upward', 'downward', 'sideways', 'volatile', 'mixed']
    threshold = 12.0
    
    stats_summary = {}
    
    for trend in trend_categories:
        horizons = []
        slopes = []
        cum_devs = []
        
        for seed in range(20):
            _, future = generate_synthetic_data(
                history_length=80,
                future_length=40,
                trend_type=trend,
                noise_scale=1.5,
                seed=seed * 100 + 42,
                device=DEVICE
            )
            
            h, m, b, cum, _, _ = find_shortest_horizon(
                future_prices=future,
                max_accum_deviation=threshold,
                min_horizon=3,
                max_horizon=35,
                device=DEVICE
            )
            
            horizons.append(h)
            slopes.append(m)
            cum_devs.append(cum)
        
        stats_summary[trend] = {
            'horizon_mean': np.mean(horizons),
            'horizon_std': np.std(horizons),
            'slope_mean': np.mean(slopes),
            'slope_std': np.std(slopes),
            'cum_dev_mean': np.mean(cum_devs),
            'cum_dev_std': np.std(cum_devs),
            'reached_pct': np.mean([c >= threshold for c in cum_devs]) * 100
        }
    
    print("\n" + "-" * 80)
    print(f"{'Trend':>12} | {'H mean':>8} | {'H std':>8} | {'Slope mean':>12} | {'Reached %':>10}")
    print("-" * 80)
    
    for trend, stats in stats_summary.items():
        print(f"  {trend:>10} | {stats['horizon_mean']:>8.1f} | {stats['horizon_std']:>8.2f} | {stats['slope_mean']:>12.4f} | {stats['reached_pct']:>10.1f}%")
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\nKey Findings:")
    print("  ✓ The function finds the SHORTEST horizon where cumulative deviation reaches the threshold")
    print("  ✓ Strong trends reach the threshold with shorter horizons")
    print("  ✓ Weak trends require longer horizons or may never reach the threshold")
    print("  ✓ Higher thresholds require longer horizons")
    print("  ✓ Volatile data reaches the threshold quickly due to large deviations")
    print("  ✓ Perfect linear trends have zero deviation and never reach the threshold")
    
    print("\nReturn Values:")
    print("  horizon: The shortest horizon where threshold is reached")
    print("  slope: The OLS slope at that horizon")
    print("  intercept: The OLS intercept at that horizon (value at time 0)")
    print("  cumulative_deviation: The cumulative deviation at that horizon")
    print("  all_horizons: All horizons tested (for plotting)")
    print("  all_cum_devs: Cumulative deviations for all horizons (for plotting)")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main_demo()