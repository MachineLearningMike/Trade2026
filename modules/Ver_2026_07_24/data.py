# data.py
# Dataset and DataLoader with per-window normalization

import math
import torch
from torch.utils.data import Dataset
import threading
import queue
import random
import time
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any
from config import get_config


# ═══════════════════════════════════════════════════════════
# SHORTEST HORIZON LABELING FUNCTION
# ═══════════════════════════════════════════════════════════

def find_shortest_horizon(
    future_prices: torch.Tensor,
    max_accum_deviation: float,
    min_horizon: int = 2,
    max_horizon: Optional[int] = None,
    device: str = 'cpu',
    verbose: bool = False
) -> Tuple[int, float, float, float, List[float], List[float], List[float]]:
    """
    Find the SINGLE SHORTEST horizon where cumulative deviation reaches max_accum_deviation.
    The trend line is CONSTRAINED to pass through the first price point f[0].
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
    
    return h, slope, intercept, cumulative_deviation, all_horizons, all_cum_devs, all_slopes


# ═══════════════════════════════════════════════════════════
# TRADING DATASET WITH PER-WINDOW NORMALIZATION
# ═══════════════════════════════════════════════════════════

class TradingDataset(Dataset):
    """Trading dataset with per-window normalization."""
    
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
        normalize: bool = True
    ):
        """
        Initialize the trading dataset with per-window normalization.
        """
        self.past_days = past_days
        self.future_minutes = future_minutes
        self.price_feature_idx = price_feature_idx
        self.step = step
        self.max_accum_deviation = max_accum_deviation
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.device = device
        self.normalize = normalize
        
        # Stack features into (F, T) tensor
        self.Total = self._stack_features(features)
        self.F, self.T = self.Total.shape
        
        # Calculate window sizes in minutes
        self.past_window = 60 * 24 * past_days  # minutes in past_days
        self.future_window = future_minutes  # Direct minutes
        self.total_window = self.past_window + self.future_window
        
        # Set max_horizon if not provided
        if self.max_horizon is None:
            self.max_horizon = self.future_window
        
        # Calculate valid start indices with step
        self.valid_starts = self._get_valid_starts()
        
        # Pre-compute labels and normalization parameters
        print("Pre-computing horizon labels and normalization parameters...")
        self.labels, self.norm_params = self._compute_labels_and_norm_params()
        print(f"TradingDataset initialized:")
        print(f"  Features: {self.F}, Time steps: {self.T}")
        print(f"  Past window: {self.past_window} minutes ({past_days} days)")
        print(f"  Future window: {self.future_window} minutes")
        print(f"  Step: {step}")
        print(f"  Max accum deviation: {max_accum_deviation}")
        print(f"  Min horizon: {min_horizon}, Max horizon: {self.max_horizon}")
        print(f"  Normalize: {normalize}")
        print(f"  Total samples: {len(self)}")
        print(f"  Labels computed: {len(self.labels)}")
        
    def _stack_features(self, features: List[np.ndarray]) -> torch.Tensor:
        """Stack numpy arrays into a torch tensor."""
        shapes = [arr.shape for arr in features]
        if len(set(shapes)) != 1:
            raise ValueError(f"All features must have the same shape. Got: {shapes}")
        
        stacked = np.stack(features, axis=0)
        return torch.from_numpy(stacked).float()
    
    def _get_valid_starts(self) -> List[int]:
        """Get all valid start indices with step."""
        max_start = self.T - self.total_window
        if max_start < 0:
            raise ValueError(
                f"Time series length {self.T} is too short for "
                f"past {self.past_window} + future {self.future_window} = {self.total_window} minutes"
            )
        
        # Get starts from 0 to max_start with the specified step
        starts = list(range(0, max_start + 1, self.step))
        
        if not starts:
            raise ValueError(f"No valid starts with step {self.step}")
        
        return starts
    
    def _normalize_window(self, window: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Normalize a window by its mean and std.
        
        Args:
            window: Tensor of shape (F, W) where W is window length
            
        Returns:
            normalized_window: Normalized tensor
            norm_params: Dictionary with mean and std for each feature
        """
        # Compute mean and std across time dimension (dim=1)
        mean = window.mean(dim=1, keepdim=True)  # (F, 1)
        std = window.std(dim=1, keepdim=True)    # (F, 1)
        std[std == 0] = 1  # Avoid division by zero
        
        # Normalize
        normalized = (window - mean) / std
        
        # Store normalization parameters
        norm_params = {
            'mean': mean.squeeze().cpu().numpy(),  # (F,)
            'std': std.squeeze().cpu().numpy()     # (F,)
        }
        
        return normalized, norm_params
    
    def _denormalize_price(self, price: float, norm_params: Dict[str, float]) -> float:
        """Denormalize a price value using the price feature's normalization parameters."""
        price_feature_mean = norm_params['mean'][self.price_feature_idx]
        price_feature_std = norm_params['std'][self.price_feature_idx]
        return price * price_feature_std + price_feature_mean
    
    def _compute_labels_and_norm_params(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Pre-compute horizon labels and normalization parameters for all samples."""
        labels = []
        norm_params_list = []
        
        for idx in tqdm(range(len(self)), desc="Computing labels and norm params"):
            start = self.valid_starts[idx]
            
            # Extract x (past data) - shape (F, past_window)
            x = self.Total[:, start:start + self.past_window]
            
            # Normalize x if requested
            if self.normalize:
                x_normalized, norm_params = self._normalize_window(x)
            else:
                x_normalized = x
                norm_params = {
                    'mean': torch.ones(self.F).numpy(),
                    'std': torch.ones(self.F).numpy()
                }
            
            # Extract future prices for labeling
            y_start = start + self.past_window
            y_end = y_start + self.future_window
            future_prices = self.Total[self.price_feature_idx, y_start:y_end]
            
            # Normalize future prices using the same price feature parameters
            if self.normalize:
                price_mean = norm_params['mean'][self.price_feature_idx]
                price_std = norm_params['std'][self.price_feature_idx]
                future_prices_normalized = (future_prices - price_mean) / price_std
            else:
                future_prices_normalized = future_prices
            
            # Compute horizon label on normalized prices
            h, m, b, cum, _, _, _ = find_shortest_horizon(
                future_prices=future_prices_normalized,
                max_accum_deviation=self.max_accum_deviation,
                min_horizon=self.min_horizon,
                max_horizon=self.max_horizon,
                device=self.device,
                verbose=False
            )
            
            labels.append({
                'horizon': h,
                'slope': m,  # Slope in normalized space
                'intercept': b,  # Intercept in normalized space (should be near 0 after normalization)
                'cumulative_deviation': cum,
                'threshold_reached': cum >= self.max_accum_deviation,
            })
            
            norm_params_list.append(norm_params)
        
        return labels, norm_params_list
    
    def __len__(self) -> int:
        """Number of valid samples."""
        return len(self.valid_starts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Get a single sample (x, y, norm_params) for a given index.
        
        Args:
            idx: Index into valid_starts
            
        Returns:
            x: Normalized tensor of shape (F, past_window)
            y: Tensor of shape (2,) - [horizon, slope] in normalized space
            norm_params: Dictionary with mean and std for each feature
        """
        start = self.valid_starts[idx]
        
        # Extract x (past data)
        x = self.Total[:, start:start + self.past_window]
        
        # Normalize x if requested
        if self.normalize:
            x_normalized, norm_params = self._normalize_window(x)
        else:
            x_normalized = x
            norm_params = {
                'mean': torch.ones(self.F).numpy(),
                'std': torch.ones(self.F).numpy()
            }
        
        # Get pre-computed labels
        label = self.labels[idx]
        y = torch.tensor([label['horizon'], label['slope']], dtype=torch.float32)
        
        return x_normalized, y, norm_params


# ═══════════════════════════════════════════════════════════
# THREADED DATA LOADER
# ═══════════════════════════════════════════════════════════

class ThreadedDataLoader:
    """DataLoader that pre-fetches next batch in a separate thread."""
    
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.current_epoch_indices = None
        self.current_batch_start = 0
        
        self.iterator = None
        self.prefetch_queue = queue.Queue(maxsize=1)
        self.prefetch_thread = None
        self.load_start_time = None
        
    def start_epoch(self, epoch, resume_indices=None, resume_batch_start=0):
        self.epoch = epoch
        
        if resume_indices is not None:
            self.current_epoch_indices = resume_indices
        else:
            self.current_epoch_indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(self.current_epoch_indices)
        
        self.current_batch_start = resume_batch_start
        self._create_iterator()
        self._start_prefetch()
    
    def _create_iterator(self):
        start = self.current_batch_start * self.batch_size
        remaining_indices = self.current_epoch_indices[start:]
        self.iterator = self._batch_generator(remaining_indices)
    
    def _batch_generator(self, indices):
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            X_list = []
            Y_list = []
            
            for idx in batch_indices:
                x, y, _ = self.dataset[idx]  # Ignore norm_params for training
                X_list.append(x)
                Y_list.append(y)
            
            X_batch = torch.stack(X_list)
            Y_batch = torch.stack(Y_list)
            
            yield X_batch, Y_batch
    
    def get_total_batches_in_epoch(self):
        """Return total number of batches in current epoch (including partial last batch)."""
        if self.current_epoch_indices is None:
            return math.ceil(len(self.dataset) / self.batch_size)
        return math.ceil(len(self.current_epoch_indices) / self.batch_size)
    
    def get_state(self):
        return {
            'epoch_indices': self.current_epoch_indices,
            'batch_start': self.current_batch_start,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
        }
    
    def restore_state(self, state):
        self.current_epoch_indices = state['epoch_indices']
        self.current_batch_start = state['batch_start']
    
    def _start_prefetch(self):
        self.load_start_time = time.time()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
        
    def _prefetch_worker(self):
        try:
            batch = next(self.iterator)
            load_time = time.time() - self.load_start_time
            self.prefetch_queue.put(('data', batch, load_time))
        except StopIteration:
            self.prefetch_queue.put(('stop', None, 0))
        except Exception as e:
            self.prefetch_queue.put(('error', e, 0))
    
    def get_next_batch(self):
        wait_start = time.time()
        status, data, load_time = self.prefetch_queue.get()
        wait_time = time.time() - wait_start
        
        if status == 'stop':
            return None, wait_time, load_time
        elif status == 'error':
            raise data
        
        self.current_batch_start += 1
        self._start_prefetch()
        return data, wait_time, load_time


# ═══════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("DATA MODULE DEMO")
    print("=" * 60)
    
    config = get_config()
    device = config['device']
    
    # Generate synthetic data
    T = 60 * 24 * config['total_days']
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
    
    # Create dataset
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
        normalize=config['normalize']
    )
    
    # Test dataloader
    dataloader = ThreadedDataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    dataloader.start_epoch(0)
    
    print("\nTesting batch iteration...")
    batch, wait_time, load_time = dataloader.get_next_batch()
    if batch is not None:
        X, Y = batch
        print(f"Batch X shape: {X.shape}")
        print(f"Batch Y shape: {Y.shape}")
        print(f"Sample X[0] shape: {X[0].shape}")
        print(f"Sample Y[0]: {Y[0]}")
    
    print("\nData module demo complete!")