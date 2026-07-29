import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import threading
import queue
from tqdm import tqdm
import time
import os
import json
from pathlib import Path
import random
import signal
import sys
import atexit
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


# ═══════════════════════════════════════════════════════════
# CONFIGURATION - CENTRAL HYPER-PARAMETER DEFINITION
# ═══════════════════════════════════════════════════════════

config = {
    # ============================================================
    # DATA CONFIGURATION
    # ============================================================
    
    # Number of days of historical data used as input (X)
    # Determines past_window = 60 * 24 * past_days (minutes)
    # Larger values give more context but increase memory/compute
    'past_days': 7,
    
    # Number of minutes of future data used for labeling (Y)
    # This is the maximum horizon that can be predicted
    # Must be >= max_horizon (labeling config)
    'future_minutes': 60,
    
    # Step size for sliding window sampling (minutes)
    # Step = 1 means every minute, Step = 60 means every hour
    # Affects dataset size: total_samples = (T - total_window) // step
    # Trade-off: smaller step = more samples, larger step = less correlation
    'step': 60,
    
    # Number of features (F) in the input data
    # Determines embed_dim = F * patch_size (transformer config)
    # Must match the actual feature dimension of your data
    'num_features': 5,
    
    # Total days of synthetic data to generate (for testing)
    # Determines total time steps T = 60 * 24 * total_days
    'total_synthetic_days': 30,
    
    # Index of the price feature in the features list (0-based)
    # Used for:
    #   1. Extracting future prices for horizon labeling
    #   2. Normalizing y using the same stats as x's price feature
    # Must be a valid index in range(0, num_features)
    'price_feature_idx': 0,
    
    
    # ============================================================
    # HORIZON LABELING CONFIGURATION
    # ============================================================
    
    # Threshold for cumulative absolute deviation
    # The horizon is the first point where cumulative deviation >= this value
    # Smaller threshold = shorter horizons, larger = longer horizons
    # Related to min_horizon, max_horizon (must be within [min, max])
    'max_accum_deviation': 12.0,
    
    # Minimum horizon to consider (in minutes)
    # Skips very short horizons that might be noisy
    # Must be >= 2 (need at least 2 points for a trend line)
    # Must be < max_horizon
    'min_horizon': 3,
    
    # Maximum horizon to consider (in minutes)
    # Caps the search to avoid looking too far into the future
    # Must be <= future_minutes (data config)
    # If None, uses future_minutes as max
    'max_horizon': 35,
    
    
    # ============================================================
    # NORMALIZATION CONFIGURATION
    # ============================================================
    
    # Whether to normalize each window by its own mean/std
    # True: Per-window normalization (recommended for non-stationary data)
    # False: No normalization (use if data is already stationary)
    # When True: each (x, y) pair is normalized using x's stats
    # Slope labels are computed in normalized space
    'normalize': True,
    
    
    # ============================================================
    # TRANSFORMER CONFIGURATION
    # ============================================================
    
    # Patch size for temporal patching (P)
    # Divides the sequence into patches of length P
    # Number of patches N = past_window // patch_size
    # Must divide past_window evenly (no remainder)
    # Related to embed_dim: embed_dim = num_features * patch_size
    # Larger patch = fewer patches, more compression, less temporal resolution
    'patch_size': 30,
    
    # Embedding dimension (D) - AUTO-COMPUTED
    # D = num_features * patch_size (then adjusted for num_heads divisibility)
    # This is the dimension of the transformer's internal representation
    # Must be divisible by num_heads (transformer constraint)
    # Larger D = more expressive power, more memory/compute
    'embed_dim': None,  # Will be computed automatically
    
    # Number of transformer encoder layers
    # More layers = deeper network, more capacity, longer training
    # Trade-off: 4-8 for small data, 12-24 for large data
    'num_layers': 4,
    
    # Number of attention heads
    # Must divide embed_dim (critical constraint!)
    # More heads = more parallel attention patterns
    # Common values: 4, 8, 16 (must be <= embed_dim)
    'num_heads': 4,
    
    # Feed-forward dimension in transformer
    # Typically 2x-4x of embed_dim
    # Controls the capacity of the feed-forward network
    'ff_dim': 256,
    
    # Dropout rate for regularization
    # Higher = more regularization, less overfitting
    # Typical range: 0.1 - 0.3
    'dropout': 0.1,
    
    # Whether to use causal (masked) attention
    # True: Each position only attends to previous positions (good for forecasting)
    # False: Full bidirectional attention (good for reconstruction/autoencoding)
    'causal': True,
    
    # Maximum sequence length (N = past_window // patch_size)
    # AUTO-COMPUTED from past_days and patch_size
    # Used for positional encoding buffer size
    'max_seq_len': None,  # Will be computed automatically
    
    
    # ============================================================
    # HEAD CONFIGURATION
    # ============================================================
    
    # Hidden dimension for prediction heads
    # Intermediate size between embed_dim and output_dim
    # Larger = more capacity for prediction, more parameters
    'head_hidden_dim': 64,
    
    # Output dimension (must be 2 for horizon, slope)
    # Position 0: horizon (time to reach threshold)
    # Position 1: slope (trend strength in normalized space)
    'output_dim': 2,
    
    
    # ============================================================
    # TRAINING CONFIGURATION
    # ============================================================
    
    # Number of training epochs
    # One epoch = one pass through the entire dataset
    'epochs': 3,
    
    # Batch size for training
    # Number of samples processed together
    # Larger = more stable gradients, more memory usage
    'batch_size': 16,
    
    # Learning rate for Adam optimizer
    # Controls step size during gradient descent
    # Too high = unstable, too low = slow convergence
    'learning_rate': 0.001,
    
    # Save checkpoint every N minibatches
    # More frequent = safer but slower (disk I/O)
    # Related to max_checkpoints (disk space management)
    'save_every_n_minibatches': 50,
    
    # Directory to save checkpoints
    'checkpoint_dir': 'trading_checkpoints',
    
    # Maximum number of checkpoints to keep
    # Older checkpoints are automatically deleted
    # Related to save_every_n_minibatches (frequency affects count)
    'max_checkpoints': 3,
    
    
    # ============================================================
    # DEVICE CONFIGURATION
    # ============================================================
    
    # Device for computation: 'cuda', 'mps', or 'cpu'
    # AUTO-DETECTED: uses GPU if available, otherwise CPU
    # Related: model.to(device), tensor.to(device)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    
    # ============================================================
    # RESUME CONFIGURATION
    # ============================================================
    
    # Whether to resume from checkpoint if available
    # True: Loads latest checkpoint and continues training
    # False: Starts fresh (ignores existing checkpoints)
    'resume': True,
    
    # Whether to delete old checkpoints before starting
    # True: Removes all existing checkpoints (fresh start)
    # False: Keeps existing checkpoints (may resume or overwrite)
    'clean': False,
    
    
    # ============================================================
    # LOSS CONFIGURATION
    # ============================================================
    
    # Weight for slope loss relative to horizon loss
    # Total loss = horizon_loss + slope_weight * slope_loss
    # If slope is less important: use smaller value (e.g., 0.1)
    # If slope is equally important: use 1.0
    'slope_loss_weight': 0.1,
    
    # Weight for reconstruction loss (auxiliary task)
    # Total loss = pred_loss + recon_weight * recon_loss
    # Reconstruction helps learn better representations
    # Set to 0 to disable reconstruction loss
    'reconstruction_loss_weight': 0.01,
}


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
# POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════════

def get_sinusoidal_encoding(seq_len: int, embed_dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding (1, seq_len, embed_dim)."""
    pe = torch.zeros(seq_len, embed_dim)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


# ═══════════════════════════════════════════════════════════
# PATCH EMBEDDING AND DEPATCHING
# ═══════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    """
    Patches the temporal dimension and projects to embed_dim.
    Input:  (B, T, F)
    Output: (B, N, D) where N = T // patch_size
    """
    def __init__(self, in_features: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * in_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        P = self.patch_size
        # (B, T, F) -> (B, N, F, P)
        x = x.unfold(dimension=1, size=P, step=P)
        # (B, N, F, P) -> (B, N, F * P)
        x = x.reshape(B, -1, F * P)
        # (B, N, F*P) -> (B, N, D)
        return self.proj(x)


class Depatch(nn.Module):
    """
    Reverses the patching process to reconstruct the original sequence.
    Input:  (B, N, D)
    Output: (B, T, F) where T = N * patch_size
    """
    def __init__(self, out_features: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_features = out_features
        self.proj = nn.Linear(embed_dim, patch_size * out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        P = self.patch_size
        F = self.out_features
        # (B, N, D) -> (B, N, F * P)
        x = self.proj(x)
        # (B, N, F * P) -> (B, N, F, P)
        x = x.reshape(B, N, F, P)
        # (B, N, F, P) -> (B, N, P, F) -> (B, N*P, F)
        x = x.permute(0, 1, 3, 2).reshape(B, N * P, F)
        return x


# ═══════════════════════════════════════════════════════════
# PREDICTION HEADS
# ═══════════════════════════════════════════════════════════

class PredictionHead(nn.Module):
    """Prediction head for horizon and slope."""
    
    def __init__(self, embed_dim: int, hidden_dim: int, output_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, embed_dim) - aggregated representation
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # (B, 2) -> [horizon, slope]


# ═══════════════════════════════════════════════════════════
# TRANSFORMER MODEL WITH PREDICTION HEADS
# ═══════════════════════════════════════════════════════════

class MarketTransformerWithHeads(nn.Module):
    """
    Transformer model with prediction heads for horizon, slope, and optional reconstruction.
    
    Input:  (B, T, F) — raw market data.
    Output: (B, 2) — [horizon, slope] predictions
    """
    
    def __init__(
        self,
        in_features: int,          # F
        patch_size: int,           # P
        embed_dim: int,            # D
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        causal: bool = True,
        head_hidden_dim: int = 128,
        output_dim: int = 2,
        use_reconstruction: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_features = in_features
        self.causal = causal
        self.max_seq_len = max_seq_len
        self.use_reconstruction = use_reconstruction

        # -------- Encoder --------
        self.patch_embed = PatchEmbed(in_features, patch_size, embed_dim)

        # Sinusoidal positional encoding
        self.register_buffer(
            'pos_embed',
            get_sinusoidal_encoding(max_seq_len, embed_dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # -------- Prediction Head --------
        self.prediction_head = PredictionHead(
            embed_dim=embed_dim,
            hidden_dim=head_hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )

        # -------- Reconstruction Decoder (auxiliary) --------
        if use_reconstruction:
            self.depatch = Depatch(in_features, patch_size, embed_dim)
        else:
            self.depatch = None

        # Causal mask buffer
        self.register_buffer('causal_mask', None)

    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        mask = mask.float().masked_fill(mask, float('-inf'))
        mask = mask.masked_fill(~mask.bool(), 0.0)
        return mask

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, F) — raw market data. T must be divisible by patch_size.

        Returns:
            predictions: (B, 2) — [horizon, slope]
            patch_map: (B, N, D) — latent representation
            recon: (B, T, F) — reconstructed sequence (if use_reconstruction)
        """
        B, T, F = x.shape
        device = x.device
        P = self.patch_size

        assert T % P == 0, f"T ({T}) must be divisible by patch_size ({P})"
        N = T // P

        if N > self.max_seq_len:
            raise ValueError(f"N ({N}) exceeds max_seq_len ({self.max_seq_len})")

        # -------- 1. Patch Embedding --------
        # (B, T, F) -> (B, N, D)
        x = self.patch_embed(x)

        # -------- 2. Add Positional Encoding --------
        x = x + self.pos_embed[:, :N, :]

        # -------- 3. Transformer --------
        if self.causal:
            if self.causal_mask is None or self.causal_mask.shape[0] != N:
                self.causal_mask = self._create_causal_mask(N).to(device)
            attn_mask = self.causal_mask
        else:
            attn_mask = None

        x = self.transformer(x, mask=attn_mask)  # (B, N, D)
        patch_map = self.norm(x)                 # (B, N, D) — latent

        # -------- 4. Aggregate for prediction --------
        # Use the last token or mean pooling
        # For causal, using the last token makes sense
        aggregated = patch_map[:, -1, :]  # (B, D)

        # -------- 5. Prediction Head --------
        predictions = self.prediction_head(aggregated)  # (B, 2)

        # -------- 6. Reconstruction (auxiliary) --------
        if self.use_reconstruction:
            recon = self.depatch(patch_map)  # (B, T, F)
        else:
            recon = None

        return predictions, patch_map, recon


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
# TRAINING STATE MANAGER
# ═══════════════════════════════════════════════════════════

class TrainingState:
    """Manages saving and loading of complete training state with atomic writes."""
    
    def __init__(self, save_dir='checkpoints', save_every_n_minibatches=100, 
                 max_checkpoints=5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_minibatches = save_every_n_minibatches
        self.max_checkpoints = max_checkpoints
        self.state_file = self.save_dir / 'training_state.json'
        
    def save_checkpoint(self, model, optimizer, dataloader_state, epoch, batch_idx, 
                        total_minibatches_processed, loss, is_emergency=False):
        """Save checkpoint with atomic write to prevent corruption."""
        checkpoint = {
            'version': 4,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'total_minibatches_processed': total_minibatches_processed,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dataloader_state': dataloader_state,
            'timestamp': time.time(),
            'is_emergency': is_emergency,
        }
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        prefix = 'emergency' if is_emergency else 'checkpoint'
        checkpoint_file = self.save_dir / f'{prefix}_minibatch{total_minibatches_processed}_{timestamp}.pt'
        
        # Atomic save: write to temp file, then rename
        temp_file = checkpoint_file.with_suffix('.tmp')
        torch.save(checkpoint, temp_file)
        
        # Retry rename with exponential backoff
        max_retries = 5
        for attempt in range(max_retries):
            try:
                os.replace(temp_file, checkpoint_file)
                break
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                else:
                    # If rename fails, try copy as fallback
                    try:
                        import shutil
                        shutil.copy2(temp_file, checkpoint_file)
                        os.remove(temp_file)
                    except:
                        raise e
        
        # Save state info with retry
        state_info = {
            'latest_checkpoint': str(checkpoint_file),
            'version': 4,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'total_minibatches_processed': total_minibatches_processed,
            'loss': loss,
            'timestamp': time.time(),
            'is_emergency': is_emergency,
        }
        
        # Atomic state file update with retry
        temp_state = self.state_file.with_suffix('.tmp')
        with open(temp_state, 'w') as f:
            json.dump(state_info, f, indent=2)
        
        for attempt in range(max_retries):
            try:
                os.replace(temp_state, self.state_file)
                break
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    # If rename fails, write directly with a different name
                    fallback_file = self.save_dir / f'training_state_{timestamp}.json'
                    with open(fallback_file, 'w') as f:
                        json.dump(state_info, f, indent=2)
                    print(f"⚠ State file saved as fallback: {fallback_file}")
                    break
        
        self._cleanup_old_checkpoints()
        
        if is_emergency:
            print(f"\n{'!'*60}")
            print(f"EMERGENCY CHECKPOINT SAVED at minibatch {total_minibatches_processed}")
            print(f"  File: {checkpoint_file.name}")
            print(f"  Epoch: {epoch+1}, Batch: {batch_idx}")
            print(f"  Loss: {loss:.4f}")
            print(f"{'!'*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Checkpoint saved at minibatch {total_minibatches_processed}")
            print(f"  File: {checkpoint_file.name}")
            print(f"  Epoch: {epoch+1}, Batch: {batch_idx}")
            print(f"  Loss: {loss:.4f}")
            print(f"{'='*60}")
        
    def load_latest_checkpoint(self, model, optimizer, dataloader):
        """Load the latest checkpoint. Returns None if nothing to resume."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                state_info = json.load(f)
        except (json.JSONDecodeError, IOError):
            print("⚠ State file corrupted. Starting fresh.")
            return None
        
        latest_checkpoint = Path(state_info['latest_checkpoint'])
        
        if not latest_checkpoint.exists():
            print(f"⚠ Checkpoint file not found. Starting fresh.")
            return None
        
        print(f"Found checkpoint: {latest_checkpoint.name}")
        
        try:
            checkpoint = torch.load(latest_checkpoint)
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {e}. Starting fresh.")
            return None
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        batch_idx = checkpoint['batch_idx']
        total_minibatches_processed = checkpoint['total_minibatches_processed']
        loss = checkpoint['loss']
        is_emergency = checkpoint.get('is_emergency', False)
        
        if 'dataloader_state' in checkpoint:
            dataloader_state = checkpoint['dataloader_state']
            dataloader.restore_state(dataloader_state)
            
            if is_emergency:
                print(f"⚠ Resuming from EMERGENCY checkpoint")
            else:
                print(f"Resuming from checkpoint")
            print(f"  Minibatch: {total_minibatches_processed}")
            print(f"  Epoch {epoch+1}, batch {batch_idx}")
            print(f"  Loss: {loss:.4f}")
            
            return epoch, batch_idx, total_minibatches_processed, loss, dataloader_state, is_emergency
        
        return None
    
    def _cleanup_old_checkpoints(self):
        all_checkpoints = sorted(
            list(self.save_dir.glob('checkpoint_minibatch*.pt')) + 
            list(self.save_dir.glob('emergency_minibatch*.pt')),
            key=os.path.getmtime,
            reverse=True
        )
        
        for old_file in all_checkpoints[self.max_checkpoints:]:
            try:
                old_file.unlink()
            except:
                pass


# ═══════════════════════════════════════════════════════════
# RESUMABLE TRAINER
# ═══════════════════════════════════════════════════════════

class ResumableTrainer:
    """
    Trainer with robust handling of all interruption scenarios.
    """
    
    def __init__(self, model, loss_fn, optimizer, dataloader, 
                 save_dir='checkpoints', device='cpu',
                 save_every_n_minibatches=100):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        
        self.state_manager = TrainingState(
            save_dir=save_dir,
            save_every_n_minibatches=save_every_n_minibatches
        )
        
        # Control flags
        self.stop_requested = False
        self.is_running = False
        self.ctrl_c_count = 0
        
        # Training state
        self.total_minibatches_processed = 0
        self.start_epoch = 0
        self.start_batch = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.current_loss = 0.0
        
        # Register handlers
        self._setup_signal_handlers()
        atexit.register(self._at_exit_handler)
        
    def _setup_signal_handlers(self):
        """Setup handlers for all interruption scenarios."""
        
        def sigint_handler(signum, frame):
            """Ctrl+C: once = graceful, twice = force quit"""
            self.ctrl_c_count += 1
            
            if self.ctrl_c_count == 1:
                print(f"\n\n⚠ Ctrl+C pressed once. Will save after current batch.")
                print("Press Ctrl+C again to force quit WITHOUT saving.")
                self.stop_requested = True
            else:
                print(f"\n\n⚠ Ctrl+C pressed twice. Force quitting NOW (no save).")
                sys.exit(1)
        
        def sigterm_handler(signum, frame):
            """SIGTERM: Cloud preemption - save immediately"""
            print(f"\n\n⚠ CLOUD PREEMPTION / SIGTERM received!")
            print("Saving emergency checkpoint NOW...")
            self._emergency_save()
            print("Exiting.")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, sigint_handler)
        signal.signal(signal.SIGTERM, sigterm_handler)
    
    def _at_exit_handler(self):
        """Called on Python exit - last resort for power failure."""
        if self.is_running and self.total_minibatches_processed > 0:
            try:
                print(f"\n⚠ ATEXIT: Attempting emergency save...")
                self._emergency_save()
                print("Emergency checkpoint saved.")
            except Exception as e:
                print(f"Could not save on exit: {e}")
    
    def _emergency_save(self):
        """Save checkpoint immediately for emergencies."""
        try:
            dataloader_state = self.dataloader.get_state()
            
            self.state_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                dataloader_state=dataloader_state,
                epoch=self.current_epoch,
                batch_idx=self.current_batch,
                total_minibatches_processed=self.total_minibatches_processed,
                loss=self.current_loss,
                is_emergency=True
            )
        except Exception as e:
            print(f"Emergency save failed: {e}")
            try:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': self.current_epoch,
                    'total_minibatches_processed': self.total_minibatches_processed,
                }, self.state_manager.save_dir / 'last_resort.pt')
                print("Bare minimum checkpoint saved.")
            except:
                print("CRITICAL: Could not save anything!")
    
    def train(self, num_epochs, resume=True):
        """Train with full interruption resilience."""
        
        # Try to resume from checkpoint
        if resume:
            loaded = self.state_manager.load_latest_checkpoint(
                self.model, self.optimizer, self.dataloader
            )
            
            if loaded:
                (self.start_epoch, self.start_batch, 
                 self.total_minibatches_processed, _, _, is_emergency) = loaded
                
                # Calculate total batches in the saved epoch
                total_batches_in_epoch = self.dataloader.get_total_batches_in_epoch()
                
                # Check if training is actually complete
                training_complete = False
                
                # Case 1: All epochs finished (start_epoch is 0-indexed)
                if self.start_epoch >= num_epochs:
                    training_complete = True
                
                # Case 2: Last epoch had all batches processed
                elif self.start_epoch == num_epochs - 1 and self.start_batch >= total_batches_in_epoch:
                    training_complete = True
                
                if training_complete:
                    print("\n" + "="*60)
                    print("✓ TRAINING ALREADY COMPLETE")
                    print("="*60)
                    print(f"All {num_epochs} epochs finished in previous run.")
                    print(f"Last checkpoint: minibatch {self.total_minibatches_processed}")
                    print(f"Epoch {self.start_epoch+1}/{num_epochs}")
                    print(f"Loss: {self.current_loss:.4f}")
                    print("="*60)
                    print("Nothing to train.")
                    print("Use --clean for fresh start, or increase --epochs.")
                    print("="*60)
                    return
                
                if is_emergency:
                    print(f"\n⚠ Resuming from emergency save.")
                    # Emergency save happens AFTER batch completion (SIGTERM, Ctrl+C)
                    # So the saved batch was already fully processed
                    self.start_batch += 1
                    print(f"  Batch {self.start_batch-1} was completed before save.")
                    print(f"  Starting from batch {self.start_batch}.")
            else:
                print("\nStarting fresh training.")
        
        self.is_running = True
        self.ctrl_c_count = 0
        
        print(f"\n{'='*60}")
        print(f"TRAINING STARTED")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs} (starting from {self.start_epoch+1})")
        print(f"Total minibatches processed so far: {self.total_minibatches_processed}")
        print(f"Save checkpoint every: {self.state_manager.save_every_n_minibatches} minibatches")
        print(f"Checkpoint directory: {self.state_manager.save_dir}/")
        print(f"{'='*60}")
        print(f"Interruption handling:")
        print(f"  Ctrl+C once  → Graceful stop (save & exit)")
        print(f"  Ctrl+C twice → Force quit (no save)")
        print(f"  Cloud preempt → Emergency save & exit")
        print(f"  Power failure → Emergency save on exit")
        print(f"{'='*60}\n")
        
        try:
            for epoch in range(self.start_epoch, num_epochs):
                if not self.is_running:
                    break
                    
                self.current_epoch = epoch
                avg_loss = self.train_epoch(epoch)
                
                if not self.is_running:
                    break
                    
                print(f'Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}')
                
        except KeyboardInterrupt:
            self._emergency_save()
            
        except Exception as e:
            print(f"\n⚠ Unexpected error: {e}")
            self._emergency_save()
            raise
        
        finally:
            self.is_running = False
            
            if self.stop_requested:
                print(f"\n⏸ Training stopped at minibatch {self.total_minibatches_processed}")
                print(f"   Epoch {self.current_epoch+1}, batch {self.current_batch}")
                print(f"   Resume with: python train.py")
            elif self.current_epoch + 1 >= num_epochs:
                print(f"\n✓ Training complete!")
                print(f"   Total minibatches: {self.total_minibatches_processed}")
                print(f"   All {num_epochs} epochs finished.")
            else:
                print(f"\n⚠ Training ended unexpectedly at minibatch {self.total_minibatches_processed}")
                print(f"   Epoch {self.current_epoch+1}/{num_epochs}")
    
    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        
        # Resume from checkpoint if needed
        if epoch == self.start_epoch and self.start_batch > 0:
            resume_indices = self.dataloader.current_epoch_indices
            resume_batch_start = self.start_batch
        else:
            resume_indices = None
            resume_batch_start = 0
            self.start_batch = 0
        
        self.dataloader.start_epoch(
            epoch, 
            resume_indices=resume_indices,
            resume_batch_start=resume_batch_start
        )
        
        total_batches = self.dataloader.get_total_batches_in_epoch()
        total_samples = len(self.dataloader.current_epoch_indices)
        
        pbar = tqdm(total=total_batches, desc=f'Epoch {epoch+1}',
                    initial=resume_batch_start)
        
        batch_idx = resume_batch_start
        total_loss = 0.0
        batches_in_epoch = 0
        
        while True:
            # Check for stop request (Ctrl+C once)
            if self.stop_requested:
                self._emergency_save()
                pbar.close()
                self.is_running = False
                return total_loss / batches_in_epoch if batches_in_epoch > 0 else 0
            
            # Get batch
            batch, wait_time, load_time = self.dataloader.get_next_batch()
            
            if batch is None:
                break
            
            # Compute
            compute_start = time.time()
            
            X, Y = batch
            X, Y = X.to(self.device), Y.to(self.device)
            
            # Model expects (B, T, F) but we have (B, F, T)
            # Transpose to (B, T, F)
            X = X.transpose(1, 2)  # (B, F, T) -> (B, T, F)
            
            self.optimizer.zero_grad()
            P = self.model(X)
            L = self.loss_fn(P, X, Y)
            L.backward()
            self.optimizer.step()
            
            compute_time = time.time() - compute_start
            
            # Update counters
            batch_idx += 1
            batches_in_epoch += 1
            self.total_minibatches_processed += 1
            total_loss += L.item()
            avg_loss = total_loss / batches_in_epoch
            self.current_batch = batch_idx
            self.current_loss = avg_loss
            
            # Progress bar - cap sample count at total
            samples_used = min(batch_idx * self.dataloader.batch_size, total_samples)
            
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'wait': f'{wait_time*1000:.0f}ms',
                'comp': f'{compute_time*1000:.0f}ms',
                'data': f'{samples_used}/{total_samples}',
                'total': self.total_minibatches_processed
            })
            
            # Periodic checkpoint
            if (self.total_minibatches_processed % 
                self.state_manager.save_every_n_minibatches == 0):
                
                dataloader_state = self.dataloader.get_state()
                
                self.state_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    dataloader_state=dataloader_state,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    total_minibatches_processed=self.total_minibatches_processed,
                    loss=avg_loss,
                    is_emergency=False
                )
        
        # Ensure progress bar shows completed
        pbar.update(total_batches - pbar.n)
        pbar.close()
        
        return total_loss / batches_in_epoch if batches_in_epoch > 0 else 0


# ═══════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════

def custom_loss(P, X, Y, slope_weight=0.1, recon_weight=0.01):
    """
    Custom loss function for multi-output prediction with optional reconstruction loss.
    
    Args:
        P: Tuple of (predictions, patch_map, recon) from model
        X: Input features (batch, T, F) after transpose
        Y: Target labels (batch, 2) -> [horizon, slope]
    
    Returns:
        Total loss
    """
    predictions, patch_map, recon = P
    
    # Unpack predictions and targets
    pred_horizon = predictions[:, 0]
    pred_slope = predictions[:, 1]
    true_horizon = Y[:, 0]
    true_slope = Y[:, 1]
    
    # Horizon loss (MSE)
    horizon_loss = nn.functional.mse_loss(pred_horizon, true_horizon)
    
    # Slope loss (MSE)
    slope_loss = nn.functional.mse_loss(pred_slope, true_slope)
    
    # Prediction loss
    pred_loss = horizon_loss + slope_weight * slope_loss
    
    # Reconstruction loss (auxiliary)
    if recon is not None:
        # X is (batch, T, F), recon is (batch, T, F)
        recon_loss = nn.functional.mse_loss(recon, X)
        total_loss = pred_loss + recon_weight * recon_loss
    else:
        total_loss = pred_loss
    
    return total_loss


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    # Override config with command-line arguments
    parser = argparse.ArgumentParser(description='Resilient Trading Training with Transformer')
    parser.add_argument('--epochs', type=int, default=config['epochs'], help='Number of epochs')
    parser.add_argument('--save-every', type=int, default=config['save_every_n_minibatches'], 
                       help='Save every N minibatches')
    parser.add_argument('--clean', action='store_true', default=config['clean'], 
                       help='Delete old checkpoints')
    parser.add_argument('--past-days', type=int, default=config['past_days'], 
                       help='Past days for input')
    parser.add_argument('--future-minutes', type=int, default=config['future_minutes'], 
                       help='Future minutes for prediction')
    parser.add_argument('--batch-size', type=int, default=config['batch_size'], 
                       help='Batch size')
    parser.add_argument('--num-features', type=int, default=config['num_features'], 
                       help='Number of features')
    parser.add_argument('--total-days', type=int, default=config['total_synthetic_days'], 
                       help='Total days of data')
    parser.add_argument('--step', type=int, default=config['step'], 
                       help='Step size for start indices')
    parser.add_argument('--max-accum-deviation', type=float, default=config['max_accum_deviation'], 
                       help='Threshold for cumulative absolute deviation')
    parser.add_argument('--min-horizon', type=int, default=config['min_horizon'], 
                       help='Minimum horizon to consider')
    parser.add_argument('--max-horizon', type=int, default=config['max_horizon'], 
                       help='Maximum horizon to consider')
    parser.add_argument('--no-normalize', action='store_true', 
                       help='Disable per-window normalization')
    parser.add_argument('--lr', type=float, default=config['learning_rate'], 
                       help='Learning rate')
    parser.add_argument('--slope-weight', type=float, default=config['slope_loss_weight'], 
                       help='Weight for slope loss')
    parser.add_argument('--recon-weight', type=float, default=config['reconstruction_loss_weight'], 
                       help='Weight for reconstruction loss')
    
    # Transformer-specific arguments
    parser.add_argument('--patch-size', type=int, default=config['patch_size'], 
                       help='Patch size for temporal patching')
    parser.add_argument('--num-layers', type=int, default=config['num_layers'], 
                       help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=config['num_heads'], 
                       help='Number of attention heads')
    parser.add_argument('--ff-dim', type=int, default=config['ff_dim'], 
                       help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=config['dropout'], 
                       help='Dropout rate')
    parser.add_argument('--head-hidden-dim', type=int, default=config['head_hidden_dim'], 
                       help='Hidden dimension for prediction heads')
    parser.add_argument('--no-causal', action='store_true', 
                       help='Disable causal attention')
    parser.add_argument('--no-reconstruction', action='store_true', 
                       help='Disable reconstruction auxiliary loss')
    
    args = parser.parse_args()
    
    # Update config with command-line arguments
    config['epochs'] = args.epochs
    config['save_every_n_minibatches'] = args.save_every
    config['clean'] = args.clean
    config['past_days'] = args.past_days
    config['future_minutes'] = args.future_minutes
    config['batch_size'] = args.batch_size
    config['num_features'] = args.num_features
    config['total_synthetic_days'] = args.total_days
    config['step'] = args.step
    config['max_accum_deviation'] = args.max_accum_deviation
    config['min_horizon'] = args.min_horizon
    config['max_horizon'] = args.max_horizon
    config['learning_rate'] = args.lr
    config['slope_loss_weight'] = args.slope_weight
    config['reconstruction_loss_weight'] = args.recon_weight
    
    # Update transformer config
    config['patch_size'] = args.patch_size
    config['num_layers'] = args.num_layers
    config['num_heads'] = args.num_heads
    config['ff_dim'] = args.ff_dim
    config['dropout'] = args.dropout
    config['head_hidden_dim'] = args.head_hidden_dim
    config['causal'] = not args.no_causal
    config['use_reconstruction'] = not args.no_reconstruction
    
    if args.no_normalize:
        config['normalize'] = False
    
    # Clean start if requested
    if config['clean'] and Path(config['checkpoint_dir']).exists():
        import shutil
        shutil.rmtree(config['checkpoint_dir'])
        print("Cleaned previous checkpoints.\n")
    
    # Create synthetic trading data
    T = 60 * 24 * config['total_synthetic_days']  # total_days of minute data
    F = config['num_features']
    
    print(f"Generating synthetic trading data: {T} minutes, {F} features")
    print(f"Using device: {config['device']}")
    
    # Create features with realistic price behavior
    features = []
    for i in range(F):
        if i == config['price_feature_idx']:  # Price feature - random walk with drift
            price = np.cumsum(np.random.randn(T) * 0.5 + 0.1)
            features.append(price.astype(np.float32))
        elif i == 1:  # Volume - positive
            volume = np.abs(np.random.randn(T) * 100 + 1000)
            features.append(volume.astype(np.float32))
        else:  # Other features
            features.append(np.random.randn(T).astype(np.float32))
    
    # Create dataset with per-window normalization
    dataset = TradingDataset(
        features=features,
        price_feature_idx=config['price_feature_idx'],
        past_days=config['past_days'],
        future_minutes=config['future_minutes'],
        step=config['step'],
        max_accum_deviation=config['max_accum_deviation'],
        min_horizon=config['min_horizon'],
        max_horizon=config['max_horizon'],
        device=config['device'],
        normalize=config['normalize']
    )
    
    # Calculate dimensions
    past_window = 60 * 24 * config['past_days']
    future_window = config['future_minutes']
    F = len(features)
    
    # Compute transformer dimensions - ensure embed_dim is divisible by num_heads
    base_embed_dim = F * config['patch_size']
    # Make sure embed_dim is divisible by num_heads
    while base_embed_dim % config['num_heads'] != 0:
        base_embed_dim += 1
    config['embed_dim'] = base_embed_dim
    config['max_seq_len'] = past_window // config['patch_size']
    
    print(f"\nData shapes:")
    print(f"  Input x: (F={F}, past_window={past_window})")
    print(f"  Output: (horizon, slope) - 2 values")
    print(f"  Total samples: {len(dataset)}")
    print(f"\nTransformer config:")
    print(f"  Patch size: {config['patch_size']}")
    print(f"  Embed dim: {config['embed_dim']} (adjusted for divisibility by num_heads)")
    print(f"  Max seq len: {config['max_seq_len']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  Num heads: {config['num_heads']}")
    print(f"  FF dim: {config['ff_dim']}")
    print(f"  Causal: {config['causal']}")
    print(f"  Use reconstruction: {config['use_reconstruction']}")
    
    # Create model
    model = MarketTransformerWithHeads(
        in_features=F,
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        causal=config['causal'],
        head_hidden_dim=config['head_hidden_dim'],
        output_dim=config['output_dim'],
        use_reconstruction=config['use_reconstruction']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Create dataloader
    dataloader = ThreadedDataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Create trainer
    trainer = ResumableTrainer(
        model=model,
        loss_fn=lambda P, X, Y: custom_loss(
            P, X, Y, 
            slope_weight=config['slope_loss_weight'],
            recon_weight=config['reconstruction_loss_weight']
        ),
        optimizer=optimizer,
        dataloader=dataloader,
        save_dir=config['checkpoint_dir'],
        device=config['device'],
        save_every_n_minibatches=config['save_every_n_minibatches']
    )
    
    # Show status
    print("\n" + "=" * 60)
    print("TRADING MODEL TRAINING WITH TRANSFORMER")
    print("=" * 60)
    print(f"Dataset: {len(dataset)} samples, batch size: {config['batch_size']}")
    print(f"Past days: {config['past_days']} ({past_window} minutes)")
    print(f"Future minutes: {config['future_minutes']}")
    print(f"Step: {config['step']} minute(s)")
    print(f"Max accum deviation: {config['max_accum_deviation']}")
    print(f"Min horizon: {config['min_horizon']}, Max horizon: {config['max_horizon']}")
    print(f"Normalize: {config['normalize']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Minibatches per epoch: {math.ceil(len(dataset) / config['batch_size'])}")
    print(f"Model: Transformer with {config['num_layers']} layers, {config['num_heads']} heads")
    print(f"Device: {trainer.device}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Slope loss weight: {config['slope_loss_weight']}")
    print(f"Reconstruction loss weight: {config['reconstruction_loss_weight']}")
    print(f"Save every: {config['save_every_n_minibatches']} minibatches")
    print(f"Checkpoints: {config['checkpoint_dir']}/")
    print("=" * 60)
    
    # Check for existing checkpoint
    state_file = Path(config['checkpoint_dir']) / 'training_state.json'
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            print(f"Found previous checkpoint at minibatch {state['total_minibatches_processed']}")
            print(f"Epoch {state['epoch']+1}, batch {state['batch_idx']}")
        except:
            pass
    else:
        print("No previous checkpoint - fresh start")
    
    print()
    
    # Train
    trainer.train(num_epochs=config['epochs'], resume=config['resume'])