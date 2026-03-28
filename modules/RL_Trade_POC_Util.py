import matplotlib.pyplot as plt
import numpy as np

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


def plot_arrays(arrays, labels=None, title="Price History", xlabel="Time", ylabel="Price", figsize=(18, 1), flip=False, save_path=None):
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
        plt.plot(arr, linewidth=1, color=color, label=label)
    
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