import matplotlib.pyplot as plt
import numpy as np

epsilon = 1e-12

def get_back_mean_std(arr, anchor, back, axis=0):   # [anchor - back + 1, anchor], n = back
    v = arr[max(0, anchor - back + 1): anchor + 1]      
    return v.mean(axis=axis), v.std(axis=axis)

def get_zscore(arr, anchor, back, axis=0):  # [anchor - back + 1, anchor], n = back
    sma, std = get_back_mean_std(arr, anchor, back, axis=axis)
    z_score = (arr[anchor] - sma) / (std + epsilon)
    return z_score

def get_level(arr, anchor, back, axis=0):   # [anchor - (back - 1), anchor], n = back
    return get_zscore(arr, anchor, back, axis=axis)

def get_change(arr, anchor, back, axis=0):    # [anchor - 2 * (back - 1), anchor], n = 2 * back - 1, safe init anchor = 2 * (back - 1)
    pcs = []
    for idx in range(max(0, anchor - back + 1), anchor + 1):
       sma = arr[max(0, idx - back + 1) : idx + 1].mean()   # [anchor - 2 * back + 2, anchor]
       pc = (arr[idx] / sma) - 1
       pcs.append(pc)
    pcs = np.array(pcs)
    assert len(pcs) <= back     # = is required
    price_change = get_zscore(pcs, len(pcs) - 1, back, axis=axis)
    return price_change

def normalize(arr, axis=0):
    return (arr - arr.mean(axis=axis)) / (arr.std(axis=axis) + epsilon)
    
def ema(arr, span): # [0, len(arr) - 1 ], n = len(arr)
    """
    Simple function to get final EMA value
    """
    alpha = 2 / (span + 1)
    ema = arr[0]
    for value in arr[1:]:
        ema = alpha * value + (1 - alpha) * ema
    return ema
    
def get_volatility(arr, anchor, back, axis=0, span=20):     # [anchor - 3 * back + 2, anchor], n = 3 * back - 1, safe init anchor = 3 * back - 2
    abs_volatilities = []
    for idx in range(max(0, anchor - back + 1), anchor + 1):    # [anchor - back + 1, anchor], n = back
        span_curr = arr[max(0, idx - back + 1) : max(0, idx + 1)]       # [anchor - 2 * back + 2, anchor]
        ema_curr = ema(span_curr, span) if span_curr.shape[0] > 0 else arr[idx]
        span_past = arr[max(0, idx - 2 * back + 1) : max(0, idx - back + 1)]    # [anchor - 3 * back + 2, anchor - back]
        ema_past =  ema(span_past, span) if span_past.shape[0] > 0 else arr[idx]
        volatility = (ema_curr - ema_past) / (ema_past + epsilon)
        abs_volatilities.append(volatility)
    assert len(abs_volatilities) <= back    # = is required
    abs_volatilities = np.array(abs_volatilities)
    # print(f"abs_volatilities: {len(abs_volatilities)}")
    volatility = get_zscore(abs_volatilities, len(abs_volatilities) - 1, back, axis=axis)
    return volatility
        
    
def plot_array1(arr, title="Price History", xlabel="Time", ylabel="Price", figsize=(18, 3), flip=False, save_path=None, vlines=None, recolors=None):
    """
    Plot a 1D numpy array
    
    Parameters:
    - arr: numpy array of shape (n,)
    - title: str - plot title
    - xlabel: str - x-axis label
    - ylabel: str - y-axis label
    - figsize: tuple - figure size (width, height)
    - flip: bool - if True, reverse the x-axis (plot from right to left)
    - save_path: str - if provided, saves plot to this path
    """
    if arr is None or len(arr) == 0:
        print("❌ No data to plot")
        return
    
    plt.figure(figsize=figsize)
    if vlines:
        plt.axvline(x=vlines, color='yellow', linewidth=0.5, alpha=0.7)
        # y_min, y_max = plt.ylim()  # Get current y-axis limits
        # plt.vlines(x=vlines, ymin=y_min, ymax=y_max, colors='red', linewidth=0.5, alpha=0.7)
    plt.plot(arr, linewidth=1, color='blue')
    if recolors:
        plt.plot(arr[:recolors], linewidth=1, color='red')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    
    if flip:
        plt.gca().invert_xaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()
        
def plot_array(arr, title="Price History", xlabel="Time", ylabel="Price", figsize=(18, 3), flip=False, save_path=None, vlines=None, recolors=None, show_stats=True):
    """
    Plot a 1D numpy array
    
    Parameters:
    - arr: numpy array of shape (n,)
    - title: str - plot title
    - xlabel: str - x-axis label
    - ylabel: str - y-axis label
    - figsize: tuple - figure size (width, height)
    - flip: bool - if True, reverse the x-axis (plot from right to left)
    - save_path: str - if provided, saves plot to this path
    - vlines: int or list - x-coordinate(s) for vertical line(s)
    - recolors: int - index to recolor the plot from that point
    - show_stats: bool - if True, display mean and std as text annotation
    """
    if arr is None or len(arr) == 0:
        print("❌ No data to plot")
        return
    
    plt.figure(figsize=figsize)
    
    # Plot main line
    plt.plot(arr, linewidth=1, color='blue', label='Value')
    
    # Handle recolor if specified
    if recolors:
        plt.plot(arr[:recolors], linewidth=1, color='red') # , label='Recolored section')
    
    # Handle vertical lines
    if vlines is not None:
        if isinstance(vlines, (int, float)):
            vlines = [vlines]
        for vline in vlines:
            plt.axvline(x=vline, color='yellow', linewidth=0.5, alpha=0.7)
    
    # Calculate statistics
    if show_stats:
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        
        # Create statistics text
        stats_text = f'μ: {mean_val:.5f} ± {std_val:.5f}\nMin: {min_val:.5f}, Max: {max_val:.5f}'
        
        # Add text annotation in the top right corner
        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Optionally add horizontal line for mean
        plt.axhline(y=mean_val, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Mean: {mean_val:.5f}')
        
        # Add std deviation bands
        plt.axhspan(mean_val - std_val, mean_val + std_val, alpha=0.2, color='gray', label=f'±1σ [{mean_val-std_val:.5f}, {mean_val+std_val:.5f}]')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    if flip:
        plt.gca().invert_xaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_arrays(arrays, labels=None, title="Price History", xlabel="Time", ylabel="Price", figsize=(18, 2), flip=False, save_path=None, lw=1):
    """
    Plot multiple 1D numpy arrays on the same graph
    
    Parameters:
    - arrays: list of numpy arrays, each of shape (n,)
    - labels: list of str - labels for each array
    - title: str - plot title
    - xlabel: str - x-axis label
    - ylabel: str - y-axis label
    - figsize: tuple - figure size (width, height)
    - flip: bool - if True, reverse the x-axis (plot from right to left)
    - save_path: str - if provided, saves plot to this path
    """
    if not arrays:
        print("❌ No data to plot")
        return
    
    plt.figure(figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, arr in enumerate(arrays):
        if arr is None or len(arr) == 0:
            continue
        
        color = colors[i % len(colors)]
        label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
        plt.plot(arr, linewidth=lw, color=color, label=label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if flip:
        plt.gca().invert_xaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()