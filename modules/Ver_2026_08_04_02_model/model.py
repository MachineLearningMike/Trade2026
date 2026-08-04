# model.py
# Transformer model with prediction heads

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from config import get_config


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
        x = x.unfold(dimension=1, size=P, step=P)
        x = x.reshape(B, -1, F * P)
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
        x = self.proj(x)
        x = x.reshape(B, N, F, P)
        x = x.permute(0, 1, 3, 2).reshape(B, N * P, F)
        return x


# ═══════════════════════════════════════════════════════════
# PREDICTION HEADS
# ═══════════════════════════════════════════════════════════

class PredictionHead(nn.Module):
    """Prediction head for horizon, slope, and intercept."""
    
    def __init__(self, embed_dim: int, hidden_dim: int, output_dim: int = 3, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # (B, 3) -> [horizon, slope, intercept]


# ═══════════════════════════════════════════════════════════
# TRANSFORMER MODEL WITH PREDICTION HEADS
# ═══════════════════════════════════════════════════════════

class MarketTransformerWithHeads(nn.Module):
    """
    Transformer model with prediction heads for horizon, slope, intercept, and optional reconstruction.
    
    Input:  (B, T, F) — raw market data.
    Output: (B, 3) — [horizon, slope, intercept] predictions
    """
    
    def __init__(
        self,
        in_features: int,
        n_market_features: int,
        n_macro_features: int,
        n_latent_macro_features: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        ff_dim_multiplier: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        causal: bool = True,
        head_hidden_dim_multiplier: int = 4,
        output_dim: int = 3,
        use_reconstruction: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_features = in_features
        self.n_market_features = n_market_features
        self.n_macro_features = n_macro_features
        self.n_latent_macro_features = n_latent_macro_features
        self.causal = causal
        self.max_seq_len = max_seq_len
        self.use_reconstruction = use_reconstruction

        #--------- get latent macro features --------
        self.latent_macro_transform_x = nn.Linear(n_macro_features, n_latent_macro_features)
        self.latent_macro_transform_mean = nn.Linear(n_market_features+n_macro_features, n_latent_macro_features)
        self.latent_macro_transform_std = nn.Linear(n_market_features+n_macro_features, n_latent_macro_features)
        self.latent_macro_transform_shape = nn.Linear(n_market_features+n_macro_features, n_latent_macro_features)

        # -------- Encoder --------
        n_features_after_latent_macro = n_market_features + n_latent_macro_features
        embed_dim = n_features_after_latent_macro * patch_size
        self.patch_embed = PatchEmbed(n_features_after_latent_macro, patch_size, embed_dim)

        # Sinusoidal positional encoding
        self.register_buffer(
            'pos_embed',
            get_sinusoidal_encoding(max_seq_len, embed_dim)
        )

        # Transformer encoder
        ff_dim = ff_dim_multiplier * embed_dim
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
            hidden_dim=head_hidden_dim_multiplier * embed_dim,
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

    def forward(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, shape: torch.Tensor):
        """
        Args:
            x: (B, T, F) — raw market data. T must be divisible by patch_size.
            mean: (B, F, 1)
            std: (B, F, 1)
            shape: (B, F, num_shapes)
        """

        x_macro = x[:, :, self.n_market_features:]  # (B, T, n_macro_features)
        x_macro = self.latent_macro_transform_x(x_macro)
        x = torch.cat([x[:, :, :self.n_market_features], x_macro], dim=-1)  # (B, T, n_market_features + n_latent_macro_features)
        mean = self.latent_macro_transform_mean(mean)
        std = self.latent_macro_transform_std(std)
        shape = self.latent_macro_transform_shape(shape)

        B, T, F = x.shape
        device = x.device
        P = self.patch_size

        assert T % P == 0, f"T ({T}) must be divisible by patch_size ({P})"
        N = T // P

        if N > self.max_seq_len:
            raise ValueError(f"N ({N}) exceeds max_seq_len ({self.max_seq_len})")

        # -------- 1. Patch Embedding --------
        x = self.patch_embed(x)

        # -------- 2. Add Positional Encoding --------
        x = x + self.pos_embed[:, :N, :]    # (B, N, D)

        # -------- 3. Transformer --------
        if self.causal:
            if self.causal_mask is None or self.causal_mask.shape[0] != N:
                self.causal_mask = self._create_causal_mask(N).to(device)
            attn_mask = self.causal_mask
        else:
            attn_mask = None

        x = self.transformer(x, mask=attn_mask)     # (B, N, D)
        patch_map = self.norm(x)    # (B, N, D)

        # -------- 4. Aggregate for prediction --------
        aggregated = patch_map[:, -1, :]  # (B, D)

        # -------- 5. Prediction Head --------
        predictions = self.prediction_head(aggregated)  # (B, 3)

        # -------- 6. Reconstruction (auxiliary) --------
        if self.use_reconstruction:
            recon = self.depatch(patch_map)  # (B, T, F)
        else:
            recon = None

        return predictions, patch_map, recon


# ═══════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════

def custom_loss(P, X, Y, slope_weight=0.1, intercept_weight=0.01, recon_weight=0.01):
    """
    Custom loss function for multi-output prediction with optional reconstruction loss.
    """
    predictions, patch_map, recon = P
    
    pred_horizon = predictions[:, 0]
    pred_slope = predictions[:, 1]
    pred_intercept = predictions[:, 2]
    
    true_horizon = Y[:, 0]
    true_slope = Y[:, 1]
    true_intercept = Y[:, 2]
    
    horizon_loss = nn.functional.mse_loss(pred_horizon, true_horizon)
    slope_loss = nn.functional.mse_loss(pred_slope, true_slope)
    intercept_loss = nn.functional.mse_loss(pred_intercept, true_intercept)
    
    pred_loss = horizon_loss + slope_weight * slope_loss + intercept_weight * intercept_loss
    
    if recon is not None:
        recon_loss = nn.functional.mse_loss(recon, X)
        return pred_loss + recon_weight * recon_loss
    
    return pred_loss


# ═══════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL MODULE DEMO")
    print("=" * 60)
    
    config = get_config()
    device = config['device']
    
    past_window = 60 * 24 * config['past_days']
    F = config['num_features']
    
    base_embed_dim = F * config['patch_size']
    while base_embed_dim % config['num_heads'] != 0:
        base_embed_dim += 1
    embed_dim = base_embed_dim
    max_seq_len = past_window // config['patch_size']
    
    print(f"Creating model with:")
    print(f"  in_features: {F}")
    print(f"  patch_size: {config['patch_size']}")
    print(f"  embed_dim: {embed_dim}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  num_layers: {config['num_layers']}")
    print(f"  num_heads: {config['num_heads']}")
    print(f"  output_dim: {config['output_dim']}")
    
    model = MarketTransformerWithHeads(
        in_features=F,
        patch_size=config['patch_size'],
        embed_dim=embed_dim,
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim_multiplier=config['ff_dim_multiplier'],
        dropout=config['dropout'],
        max_seq_len=max_seq_len,
        causal=config['causal'],
        head_hidden_dim_multiplier=config['head_hidden_dim_multiplier'],
        output_dim=config['output_dim'],
        use_reconstruction=config['use_reconstruction']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    B = 4
    T = (past_window // config['patch_size']) * config['patch_size']
    x = torch.randn(B, T, F).to(device)
    
    print(f"\nTesting forward pass with input shape: {x.shape}")
    predictions, patch_map, recon = model(x)
    
    print(f"Predictions shape: {predictions.shape}")
    if recon is not None:
        print(f"Reconstruction shape: {recon.shape}")
    
    print("\nModel module demo complete!")