"""
Dynamic Horizon Linear Trend Labeling - Find Single Shortest Horizon
===================================================================
Find the single shortest horizon where cumulative deviation reaches max_accum_deviation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
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
# CORRECTED: Find the Shortest Horizon
# ============================================================================

def find_shortest_horizon(
    future_prices: torch.Tensor,
    max_accum_deviation: float,
    min_horizon: int = 2,
    max_horizon: Optional[int] = None,
    device: str = DEVICE
) -> Tuple[int, float, float, float]:
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
    
    # Search for the shortest horizon
    for h in range(min_horizon, max_horizon + 1):
        # Get the segment
        segment = prices[:h]
        
        # Time indices starting from 0
        t = torch.arange(h, dtype=torch.float32, device=device)
        
        # Fit linear trend using OLS: y = m*t + b
        # m = sum((t - t_mean) * (y - y_mean)) / sum((t - t_mean)^2)
        # b = y_mean - m * t_mean
        
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
        # Predicted values using the OLS line: y_pred = slope * t + intercept
        predicted = slope * t + intercept
        deviations = torch.abs(segment - predicted)
        cumulative_deviation = torch.sum(deviations).item()
        
        # If we've reached the threshold, return this horizon
        if cumulative_deviation >= max_accum_deviation:
            return h, slope, intercept, cumulative_deviation
    
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
    
    return h, slope, intercept, cumulative_deviation


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
    Generate synthetic price data.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate historical data (random walk)
    historical = np.cumsum(np.random.randn(history_length) * 0.5) + 100
    
    # Generate future with different trends
    if trend_type == 'upward':
        # Strong upward trend with noise
        future = np.cumsum(np.random.randn(future_length) * 0.5 + 0.4) + historical[-1]
    elif trend_type == 'downward':
        future = np.cumsum(np.random.randn(future_length) * 0.5 - 0.4) + historical[-1]
    elif trend_type == 'sideways':
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
    elif trend_type == 'volatile':
        future = np.cumsum(np.random.randn(future_length) * 1.5) + historical[-1]
    elif trend_type == 'mixed':
        trend = np.sin(np.linspace(0, 2*np.pi, future_length)) * 5
        future = np.cumsum(np.random.randn(future_length) * 0.3) + trend + historical[-1]
    elif trend_type == 'reversal':
        # Starts going up then reverses down
        future = np.cumsum(np.random.randn(future_length) * 0.3) + historical[-1]
        reversal_point = future_length // 2
        future[reversal_point:] -= np.linspace(0, 10, future_length - reversal_point)
    else:  # 'breakout'
        future = np.cumsum(np.random.randn(future_length) * 0.2) + historical[-1]
        breakout_point = future_length // 3
        future[breakout_point:] += np.linspace(0, 15, future_length - breakout_point)
    
    return (torch.tensor(historical, dtype=torch.float32, device=device), 
            torch.tensor(future, dtype=torch.float32, device=device))


# ============================================================================
# ENHANCED VISUALIZATION
# ============================================================================

def visualize_result(
    historical_prices: torch.Tensor,
    future_prices: torch.Tensor,
    horizon: int,
    slope: float,
    intercept: float,
    cumulative_deviation: float,
    max_accum_deviation: float,
    min_horizon: int = 2,
    title: str = "Shortest Horizon Labeling",
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Visualize the single shortest horizon result with CORRECT trend line.
    """
    
    # Move to CPU for plotting
    hist = historical_prices.cpu().numpy()
    fut = future_prices.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # --- Plot 1: Full price history with trend ---
    ax1 = axes[0, 0]
    
    full_prices = np.concatenate([hist, fut])
    time_steps = np.arange(len(full_prices))
    split_idx = len(hist)
    
    # Plot historical and future
    ax1.plot(time_steps[:split_idx], hist, 'b-', linewidth=2, label='Historical', alpha=0.7)
    ax1.plot(time_steps[split_idx:], fut, 'g-', linewidth=2, label='Future Prices', alpha=0.8)
    
    # Plot the CORRECT trend line using OLS intercept
    if horizon > 0:
        trend_time = np.arange(horizon)
        # Use the OLS intercept, NOT fut[0]
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
    ax2 = axes[0, 1]
    
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
    
    # --- Plot 3: Per-step deviations ---
    ax3 = axes[1, 0]
    
    if horizon > 0:
        x_vals = np.arange(horizon)
        y_vals = fut[:horizon]
        trend_vals = slope * x_vals + intercept
        per_step_dev = np.abs(y_vals - trend_vals)
        
        # Bar chart
        bars = ax3.bar(x_vals, per_step_dev, alpha=0.6, color='steelblue')
        
        # Color significant deviations
        avg_dev = per_step_dev.mean()
        for i, bar in enumerate(bars):
            if per_step_dev[i] > avg_dev * 1.5:
                bar.set_color('red')
                bar.set_alpha(0.8)
        
        ax3.axhline(y=avg_dev, color='red', linestyle='--', 
                   label=f'Avg Dev = {avg_dev:.3f}')
        ax3.axhline(y=avg_dev * 1.5, color='orange', linestyle=':', 
                   label=f'1.5× Avg Dev')
        
        ax3.set_xlabel('Time Step', fontsize=11)
        ax3.set_ylabel('Per-Step Deviation', fontsize=11)
        ax3.set_title(f'Per-Step Deviations (Total: {np.sum(per_step_dev):.2f})', 
                     fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Metrics Summary ---
    ax4 = axes[1, 1]
    ax4.axis('off')
    
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
    │  MIN HORIZON           : {min_horizon:>4}                   │
    │  MAX HORIZON           : {horizon:>4} (selected)            │
    └─────────────────────────────────────────────────────────────┘
    """
    
    ax4.text(0.5, 0.5, metrics_text, fontsize=11, verticalalignment='center',
             horizontalalignment='center', family='monospace', transform=ax4.transAxes)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main_demo():
    """
    Demonstrate finding the single shortest horizon.
    """
    print("=" * 80)
    print("SINGLE SHORTEST HORIZON LABELING - CORRECTED VERSION")
    print("=" * 80)
    print(f"Using device: {DEVICE}")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # ========== Test a perfect upward trend ==========
    print("\n" + "="*80)
    print("TEST 1: Perfect Upward Trend")
    print("="*80)
    
    # Create a perfect linear trend with some noise
    np.random.seed(42)
    future_length = 30
    perfect_trend = np.linspace(100, 130, future_length) + np.random.randn(future_length) * 0.5
    future_tensor = torch.tensor(perfect_trend, dtype=torch.float32, device=DEVICE)
    
    max_accum_deviation = 5.0
    min_horizon = 2
    max_horizon = 25
    
    print(f"\nData: Perfect upward trend with small noise")
    print(f"Threshold: {max_accum_deviation}")
    print(f"Expected: Short horizon should be found quickly")
    
    horizon, slope, intercept, cum_dev = find_shortest_horizon(
        future_prices=future_tensor,
        max_accum_deviation=max_accum_deviation,
        min_horizon=min_horizon,
        max_horizon=max_horizon,
        device=DEVICE
    )
    
    print(f"\nResults:")
    print(f"  Horizon: {horizon}")
    print(f"  Slope: {slope:.6f}")
    print(f"  Intercept: {intercept:.6f}")
    print(f"  Cumulative Deviation: {cum_dev:.2f}")
    print(f"  Threshold: {max_accum_deviation}")
    print(f"  Threshold reached: {cum_dev >= max_accum_deviation}")
    
    # Visualize
    dummy_hist = torch.zeros(1, device=DEVICE)
    visualize_result(
        historical_prices=dummy_hist,
        future_prices=future_tensor,
        horizon=horizon,
        slope=slope,
        intercept=intercept,
        cumulative_deviation=cum_dev,
        max_accum_deviation=max_accum_deviation,
        min_horizon=min_horizon,
        title=f"Perfect Upward Trend - H={horizon}, m={slope:.4f}"
    )
    
    # ========== Test different trend types ==========
    print("\n" + "="*80)
    print("TEST 2: Different Trend Types")
    print("="*80)
    
    trend_types = ['upward', 'downward', 'volatile', 'mixed', 'reversal']
    max_accum_deviation = 15.0
    min_horizon = 3
    max_horizon = 45
    
    print(f"\nParameters:")
    print(f"  Threshold: {max_accum_deviation}")
    print(f"  Min Horizon: {min_horizon}")
    print(f"  Max Horizon: {max_horizon}")
    print("\n" + "-" * 80)
    print(f"{'Trend Type':>12} | {'Horizon':>10} | {'Slope':>12} | {'Intercept':>12} | {'Cum Dev':>12} | {'Reached':>10}")
    print("-" * 80)
    
    for trend_type in trend_types:
        # Generate data
        history, future = generate_synthetic_data(
            history_length=100,
            future_length=50,
            trend_type=trend_type,
            noise_scale=1.5,
            seed=hash(trend_type) % 1000,
            device=DEVICE
        )
        
        # Find the single shortest horizon
        horizon, slope, intercept, cum_dev = find_shortest_horizon(
            future_prices=future,
            max_accum_deviation=max_accum_deviation,
            min_horizon=min_horizon,
            max_horizon=max_horizon,
            device=DEVICE
        )
        
        reached = cum_dev >= max_accum_deviation
        status = "✓ YES" if reached else "✗ NO"
        
        print(f"  {trend_type:>10} | {horizon:>10} | {slope:>12.4f} | {intercept:>12.3f} | {cum_dev:>12.2f} | {status:>10}")
        
        # Visualize a few examples
        if trend_type in ['upward', 'volatile', 'reversal']:
            print(f"\n  Visualizing {trend_type}...")
            visualize_result(
                historical_prices=history,
                future_prices=future,
                horizon=horizon,
                slope=slope,
                intercept=intercept,
                cumulative_deviation=cum_dev,
                max_accum_deviation=max_accum_deviation,
                min_horizon=min_horizon,
                title=f"{trend_type.upper()} Trend - H={horizon}, m={slope:.4f}"
            )
    
    # ========== Summary ==========
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nThe OLS trend line is calculated correctly using:")
    print("  y = m * t + b")
    print("  where m = slope, b = intercept (value at time 0)")
    print("\nThe cumulative deviation is:")
    print("  sum(|actual_price - trend_line|) for each time step")
    print("\nThe function finds the SHORTEST horizon where:")
    print("  cumulative_deviation >= max_accum_deviation")
    print("\nReturns: (horizon, slope, intercept, cumulative_deviation)")

if __name__ == "__main__":
    main_demo()