import math
import torch
import torch.nn as nn


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


class MarketVideoTransformer(nn.Module):
    """
    Temporal autoencoder for fixed-length market history.
    This is designed to be a modular component of a larger model.

    - Input:  (B, T, F) — raw market data.
    - Encoder: PatchEmbed → Transformer → patch_map (latent low-res sequence).
    - Decoder: Depatch → reconstructs (B, T, F).
    """
    def __init__(
        self,
        in_features: int,      # F
        patch_size: int,       # P
        embed_dim: int,        # D
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        causal: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_features = in_features
        self.causal = causal
        self.max_seq_len = max_seq_len

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

        # -------- Decoder --------
        self.depatch = Depatch(in_features, patch_size, embed_dim)

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
            patch_map: (B, N, D) — low‑resolution latent sequence (for downstream tasks).
            recon: (B, T, F) — reconstructed high‑resolution sequence (for auxiliary loss).
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
        patch_map = self.norm(x)                 # (B, N, D) — final latent

        # -------- 4. Depatch --------
        # (B, N, D) -> (B, T, F)
        recon = self.depatch(patch_map)

        return patch_map, recon


# ========== Usage Example ==========
if __name__ == "__main__":
    # Detect available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model configuration
    F = 16          # features
    T = 10080        # ticks (fixed)
    P = 30          # patch size -> N = 32
    D = F * P         # embedding dim = 480
    max_seq_len = int(T / P)

    # Instantiate model and move to device
    model = MarketVideoTransformer(
        in_features=F,
        patch_size=P,
        embed_dim=D,
        num_layers=16,
        num_heads=16,
        ff_dim=512,
        dropout=0.1,
        max_seq_len=max_seq_len,
        causal=True,
    ).to(device)

    # Create batch and move to device
    B = 8
    batch = torch.randn(B, T, F).to(device)

    # Forward pass
    for i in range(300):
        patch_map, recon = model(batch)

    print(f"Input shape:          {batch.shape}")          # (4, 1024, 16)
    print(f"Patch Map shape:      {patch_map.shape}")     # (4, 32, 256)
    print(f"Reconstruction shape: {recon.shape}")         # (4, 1024, 16)

    # Compute loss on the same device
    loss = nn.MSELoss()(recon, batch)
    print(f"Reconstruction MSE:   {loss.item():.6f}")