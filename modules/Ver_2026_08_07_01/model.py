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
# MODEL INSPECTION UTILITIES
# ═══════════════════════════════════════════════════════════

def report_model_size(model: nn.Module, name: str = "Model") -> dict:
    """
    Compute and display model size (parameter counts + memory footprint).

    Returns a dict with total/trainable/frozen counts and byte sizes.
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    param_bytes = 0
    trainable_bytes = 0

    for p in model.parameters():
        n = p.numel()
        b = n * p.element_size()
        total_params += n
        param_bytes += b
        if p.requires_grad:
            trainable_params += n
            trainable_bytes += b
        else:
            frozen_params += n

    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = param_bytes + buffer_bytes

    def _fmt_bytes(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024.0:
                return f"{n:.2f} {unit}"
            n /= 1024.0
        return f"{n:.2f} TB"

    info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "param_bytes": param_bytes,
        "trainable_bytes": trainable_bytes,
        "buffer_bytes": buffer_bytes,
        "total_bytes": total_bytes,
    }

    print(f"\n{name} size:")
    print(f"  Parameters total:     {total_params:,}")
    print(f"  Parameters trainable: {trainable_params:,}")
    print(f"  Parameters frozen:    {frozen_params:,}")
    print(f"  Param memory:         {_fmt_bytes(param_bytes)}")
    print(f"  Buffer memory:        {_fmt_bytes(buffer_bytes)}")
    print(f"  Total footprint:      {_fmt_bytes(total_bytes)}")
    return info


def check_all_weights_updated(
    model: nn.Module,
    sample_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    loss_fn=None,
    name: str = "Model",
) -> bool:
    """
    Run one forward + backward and verify every trainable parameter gets a gradient.

    Args:
        model:        module under test
        sample_batch: (x, mean, std, shape) matching model.forward
        loss_fn:      optional callable(P, X, Y) -> scalar; if None, uses
                      predictions.sum() (+ recon.sum() if present)
        name:         label for log lines

    Returns:
        True if all trainable weights received a non-None gradient.
    """
    was_training = model.training
    model.train()

    # Snapshot so this diagnostic does not permanently alter weights
    snapped = {n: p.detach().clone() for n, p in model.named_parameters()}

    model.zero_grad(set_to_none=True)
    x, mean, std, shape = sample_batch
    device = next(model.parameters()).device
    x = x.to(device)
    mean = mean.to(device)
    std = std.to(device)
    shape = shape.to(device)

    P = model(x, mean, std, shape)
    predictions, patch_map, recon = P

    if loss_fn is not None:
        # Dummy targets matching prediction layout (2 or 3 dims)
        Y = torch.zeros(predictions.shape[0], predictions.shape[1], device=device)
        loss = loss_fn(P, x, Y)
    else:
        loss = predictions.sum()
        if recon is not None:
            loss = loss + recon.sum()

    loss.backward()

    missing = []
    zero_grad = []
    updated = 0
    trainable = 0

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        trainable += 1
        if p.grad is None:
            missing.append(n)
        elif torch.count_nonzero(p.grad).item() == 0:
            zero_grad.append(n)
        else:
            updated += 1

    # Restore original weights (grads are left for inspection if needed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(snapped[n])
    model.zero_grad(set_to_none=True)

    if not was_training:
        model.eval()

    print(f"\n{name} weight-update check (one backward pass):")
    print(f"  Trainable tensors: {trainable}")
    print(f"  Received grad:     {updated}")
    print(f"  Missing grad:      {len(missing)}")
    print(f"  All-zero grad:     {len(zero_grad)}")

    if missing:
        print("  ⚠ Parameters with NO gradient (not on the loss path):")
        for n in missing:
            print(f"      - {n}")
    if zero_grad:
        print("  ⚠ Parameters with all-zero gradient:")
        for n in zero_grad:
            print(f"      - {n}")

    ok = len(missing) == 0
    if ok and len(zero_grad) == 0:
        print("  ✓ All trainable weights received a non-zero gradient.")
    elif ok:
        print("  ~ All trainable weights received a gradient (some were zero this step).")
    else:
        print("  ✗ Some trainable weights were NOT updated (missing gradient).")
    return ok


# ═══════════════════════════════════════════════════════════
# PREDICTION HEADS
# ═══════════════════════════════════════════════════════════

class PredictionHead(nn.Module):
    """
    Prediction head.

    output_dim=3 → (horizon, slope, intercept)
    output_dim=2 → (slope, intercept)
    """

    def __init__(self, embed_dim: int, hidden_dim: int, output_dim: int = 3, dropout: float = 0.1):
        super().__init__()
        if output_dim not in (2, 3):
            raise ValueError(f"output_dim must be 2 or 3, got {output_dim}")
        self.output_dim = output_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # (B, output_dim)


# ═══════════════════════════════════════════════════════════
# TRANSFORMER MODEL WITH PREDICTION HEADS
# ═══════════════════════════════════════════════════════════

class MarketTransformerWithHeads(nn.Module):
    """
    Transformer model with a prediction head for trend labels and optional
    reconstruction.

    output_dim controls the head layout:
      3 → (horizon, slope, intercept)
      2 → (slope, intercept)

    Window statistics (mean / std / shape moments) condition the decoder via
    x_stat tokens.
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
        num_shapes: int = 4,
    ):
        super().__init__()
        if output_dim not in (2, 3):
            raise ValueError(f"output_dim must be 2 or 3, got {output_dim}")
        self.patch_size = patch_size
        self.in_features = in_features
        self.n_market_features = n_market_features
        self.n_macro_features = n_macro_features
        self.n_latent_macro_features = n_latent_macro_features
        self.causal = causal
        self.max_seq_len = max_seq_len
        self.use_reconstruction = use_reconstruction
        self.output_dim = output_dim
        self.num_shapes = num_shapes
        # x_stat tokens: mean (1) + std (1) + shape moments (num_shapes)
        self.n_stat_tokens = 1 + 1 + num_shapes

        F = n_market_features + n_macro_features
        L = n_latent_macro_features
        self.F = F

        # ---- compress raw macro time series ----
        self.latent_macro_transform_x = nn.Linear(n_macro_features, L)

        # ---- project window stats into latent space (used by forward / _encode_stats) ----
        # mean/std: (B, F) → (B, L)
        # shape:    (B, num_shapes, F) → (B, num_shapes, L), then pooled
        # self.latent_macro_transform_mean = nn.Linear(F, L)
        # self.latent_macro_transform_std = nn.Linear(F, L)
        # self.latent_macro_transform_shape = nn.Linear(F, L)

        # ---- Encoder ----
        n_features_after_latent_macro = n_market_features + L
        embed_dim = n_features_after_latent_macro * patch_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(n_features_after_latent_macro, patch_size, embed_dim)

        # Fuse mean+std+shape latents → embed_dim, then residual-add to all patches
        # self.stats_to_embed = nn.Sequential(
        #     nn.Linear(3 * L, embed_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(embed_dim, embed_dim),
        # )
        # self.stats_norm = nn.LayerNorm(embed_dim)

        self.register_buffer(
            'pos_embed',
            get_sinusoidal_encoding(max_seq_len, embed_dim)
        )

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

        # ---- Encoder–decoder: x patches as memory, x_stat tokens as tgt ----
        # memory = encoder(x)  → Key & Value for decoder cross-attn
        # tgt    = x_stat      → decoder self-attn; Query for cross-attn
        # After decode: flatten (B, N_stat, D) → FFN → (B, output_dim)
        self.stat_token_proj = nn.Linear(F, embed_dim)
        self.stat_pos_embed = nn.Parameter(
            torch.zeros(1, self.n_stat_tokens, embed_dim)
        )
        nn.init.normal_(self.stat_pos_embed, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # ---- Prediction Head ----
        # Flatten decoder tokens (B, N_stat, D) → (B, N_stat * D) then FFN → (B, output_dim)
        pred_in_dim = self.n_stat_tokens * embed_dim
        self.prediction_head = PredictionHead(
            embed_dim=pred_in_dim,
            hidden_dim=head_hidden_dim_multiplier * embed_dim,
            output_dim=output_dim,
            dropout=dropout
        )

        # ---- Reconstruction Decoder (auxiliary) ----
        if use_reconstruction:
            self.depatch = Depatch(in_features, patch_size, embed_dim)
        else:
            self.depatch = None

        self.register_buffer('causal_mask', None)

    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        mask = mask.float().masked_fill(mask, float('-inf'))
        mask = mask.masked_fill(~mask.bool(), 0.0)
        return mask

    def _encode_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build a single conditioning vector from window statistics.

        Args:
            mean:  (B, F)
            std:   (B, F)
            shape: (B, num_shapes, F)

        Returns:
            stats_embed: (B, D)
        """
        mean_lat = self.latent_macro_transform_mean(mean)          # (B, L)
        std_lat = self.latent_macro_transform_std(std)             # (B, L)
        shape_lat = self.latent_macro_transform_shape(shape)       # (B, S, L)
        shape_lat = shape_lat.mean(dim=1)                          # (B, L) pool moments

        stats = torch.cat([mean_lat, std_lat, shape_lat], dim=-1)  # (B, 3L)
        return self.stats_norm(self.stats_to_embed(stats))         # (B, D)

    # ───────────────────────────────────────────────────────
    # forward2 helpers
    # ───────────────────────────────────────────────────────

    def _build_x_stat(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stack window statistics into a short token sequence.

        mean:  (B, F)            → token 0
        std:   (B, F)            → token 1
        shape: (B, num_shapes, F) → tokens 2 .. 1+num_shapes

        Returns:
            x_stat: (B, N_stat, F) with N_stat = 1 + 1 + num_shapes
        """
        if mean.dim() != 2 or std.dim() != 2:
            raise ValueError(f"mean/std must be (B, F); got {tuple(mean.shape)}, {tuple(std.shape)}")
        if shape.dim() != 3:
            raise ValueError(f"shape must be (B, num_shapes, F); got {tuple(shape.shape)}")

        mean_tok = mean.unsqueeze(1)   # (B, 1, F)
        std_tok = std.unsqueeze(1)     # (B, 1, F)
        # shape already (B, S, F)
        x_stat = torch.cat([mean_tok, std_tok, shape], dim=1)  # (B, 2+S, F)

        expected = self.n_stat_tokens
        if x_stat.shape[1] != expected:
            raise ValueError(
                f"x_stat length {x_stat.shape[1]} != expected {expected} "
                f"(1 mean + 1 std + {self.num_shapes} shapes)"
            )
        return x_stat

    def _encode_x_patches(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Macro-compress + patch-embed + pos-encode + transformer encoder.

        Returns:
            patch_map: (B, N, D) encoder output
            N: number of patches
        """
        x_macro = x[:, :, self.n_market_features:]                 # (B, T, n_macro)
        x_macro = self.latent_macro_transform_x(x_macro)           # (B, T, L)
        x = torch.cat(
            [x[:, :, :self.n_market_features], x_macro], dim=-1
        )                                                          # (B, T, n_market + L)

        B, T, _ = x.shape
        device = x.device
        P = self.patch_size

        assert T % P == 0, f"T ({T}) must be divisible by patch_size ({P})"
        N = T // P
        if N > self.max_seq_len:
            raise ValueError(f"N ({N}) exceeds max_seq_len ({self.max_seq_len})")

        x = self.patch_embed(x)                                    # (B, N, D)
        x = x + self.pos_embed[:, :N, :]

        if self.causal:
            if self.causal_mask is None or self.causal_mask.shape[0] != N:
                self.causal_mask = self._create_causal_mask(N).to(device)
            attn_mask = self.causal_mask
        else:
            attn_mask = None

        x = self.transformer(x, mask=attn_mask)                    # (B, N, D)
        patch_map = self.norm(x)                                   # (B, N, D)
        return patch_map, N

    def forward(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        shape: torch.Tensor,
    ):
        """
        Encoder–decoder mix of the time series and window statistics.

        1. Build x_stat (B, N_stat, F), N_stat = 1 + 1 + num_shapes:
              [mean, std, shape_0, ..., shape_{S-1}]
        2. Encoder on x (patches) → memory
        3. Decoder on x_stat:
              - project x_stat → (B, N_stat, D)
              - TransformerDecoder(tgt=x_stat, memory=encoder(x))
              - LayerNorm → (B, N_stat, D)
        4. Flatten (B, N_stat, D) → (B, N_stat * D), FFN → (B, 3)

        Args:
            x:     (B, T, F) — window features. T must be divisible by patch_size.
                    x = normalize(feature_window, stat(feature_window))
            mean:  (B, F) — normalize(mean_feature_window, train-set mean/std of means)
            std:   (B, F) — normalize(std_feature_window, train-set mean/std of stds)
                    NOTE: Do NOT multiply x with std or mean; some channels are ~0.
            shape: (B, num_shapes, F)

        Returns:
            predictions, patch_map, recon
        """
        # ---- 0. Statistics token sequence ----
        x_stat = self._build_x_stat(mean, std, shape)              # (B, N_stat, F)

        # ---- 1. Encode x ----
        patch_map, N = self._encode_x_patches(x)                   # (B, N, D)

        # ---- 2. Embed x_stat as decoder tokens ----
        stat_tok = self.stat_token_proj(x_stat)                    # (B, N_stat, D)
        n_stat = stat_tok.shape[1]
        stat_tok = stat_tok + self.stat_pos_embed[:, :n_stat, :]

        # Transformer decoder: tgt = stats, memory = encoder(x)
        # Cross-attention inside: Query ← tgt (stats), Key/Value ← memory (x)
        dec_out = self.transformer_decoder(
            tgt=stat_tok,
            memory=patch_map,
        )                                                          # (B, N_stat, D)
        dec_out = self.decoder_norm(dec_out)

        # ---- 3. Flatten decoder tokens + FFN → (B, 3) ----
        B = dec_out.shape[0]
        flat = dec_out.reshape(B, n_stat * self.embed_dim)         # (B, N_stat * D)
        predictions = self.prediction_head(flat)                   # (B, output_dim)

        # ---- 4. Reconstruction from encoder map (optional) ----
        if self.use_reconstruction:
            recon = self.depatch(patch_map)
        else:
            recon = None

        return predictions, patch_map, recon


# ═══════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════

def custom_loss(P, X, Y, slope_weight=0.1, intercept_weight=10, recon_weight=0.01):
    """
    Multi-output prediction loss with optional reconstruction term.

    Prediction / target layout (last dim must match):
      output_dim=3 → (horizon, slope, intercept)
      output_dim=2 → (slope, intercept)  — no horizon term
    """
    predictions, patch_map, recon = P
    out_dim = predictions.shape[-1]
    if Y.shape[-1] != out_dim:
        raise ValueError(
            f"Target dim Y.shape[-1]={Y.shape[-1]} != predictions dim {out_dim}"
        )

    if out_dim == 3:
        pred_horizon, pred_slope, pred_intercept = (
            predictions[:, 0], predictions[:, 1], predictions[:, 2]
        )
        true_horizon, true_slope, true_intercept = Y[:, 0], Y[:, 1], Y[:, 2]
        horizon_loss = nn.functional.mse_loss(pred_horizon, true_horizon)
        slope_loss = nn.functional.mse_loss(pred_slope, true_slope)
        intercept_loss = nn.functional.mse_loss(pred_intercept, true_intercept)
        pred_loss = (
            horizon_loss
            + slope_weight * slope_loss
            + intercept_weight * intercept_loss
        )
    elif out_dim == 2:
        pred_slope, pred_intercept = predictions[:, 0], predictions[:, 1]
        true_slope, true_intercept = Y[:, 0], Y[:, 1]
        slope_loss = nn.functional.mse_loss(pred_slope, true_slope)
        intercept_loss = nn.functional.mse_loss(pred_intercept, true_intercept)
        pred_loss = slope_weight * slope_loss + intercept_weight * intercept_loss
    else:
        raise ValueError(f"Unsupported prediction dim {out_dim}; expected 2 or 3")

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
    n_market = 4
    n_macro = 10
    n_latent = config['n_latent_macro_features']
    F = n_market + n_macro
    num_shapes = config['num_shapes']
    max_seq_len = past_window // config['patch_size']
    embed_dim = (n_market + n_latent) * config['patch_size']

    print(f"Creating model with:")
    print(f"  in_features: {F} (market={n_market}, macro={n_macro})")
    print(f"  n_latent_macro_features: {n_latent}")
    print(f"  patch_size: {config['patch_size']}")
    print(f"  embed_dim: {embed_dim}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  num_layers: {config['num_layers']}")
    print(f"  num_heads: {config['num_heads']}")
    print(f"  output_dim: {config['output_dim']}")

    model = MarketTransformerWithHeads(
        in_features=F,
        n_market_features=n_market,
        n_macro_features=n_macro,
        n_latent_macro_features=n_latent,
        patch_size=config['patch_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim_multiplier=config['ff_dim_multiplier'],
        dropout=config['dropout'],
        max_seq_len=max_seq_len,
        causal=config['causal'],
        head_hidden_dim_multiplier=config['head_hidden_dim_multiplier'],
        output_dim=config['output_dim'],
        use_reconstruction=config['use_reconstruction'],
        num_shapes=num_shapes,
    ).to(device)

    # ---- Size + weight-update diagnostics at start ----
    report_model_size(model, name="MarketTransformerWithHeads")

    B = 4
    T = (past_window // config['patch_size']) * config['patch_size']
    x = torch.randn(B, T, F, device=device)
    mean = torch.randn(B, F, device=device)
    std = torch.randn(B, F, device=device).abs() + 1e-3
    shape = torch.randn(B, num_shapes, F, device=device)

    check_all_weights_updated(
        model,
        sample_batch=(x, mean, std, shape),
        loss_fn=lambda P, X, Y: custom_loss(
            P, X, Y, slope_weight=0.1, intercept_weight=0.01, recon_weight=0.01
        ),
        name="MarketTransformerWithHeads",
    )

    print(f"\nTesting forward() with input shape: {x.shape}")
    predictions, patch_map, recon = model(x, mean, std, shape)
    print(f"  predictions: {tuple(predictions.shape)}")
    print(f"  patch_map:   {tuple(patch_map.shape)}")
    if recon is not None:
        print(f"  recon:       {tuple(recon.shape)}")

    x_stat = model._build_x_stat(mean, std, shape)
    print(f"  x_stat:      {tuple(x_stat.shape)}  (N_stat={model.n_stat_tokens})")
    print(f"  pred_in_dim: {model.n_stat_tokens * model.embed_dim}  (N_stat * D)")

    print("\nModel module demo complete!")
