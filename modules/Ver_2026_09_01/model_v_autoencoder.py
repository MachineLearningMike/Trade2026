# model_v_autoencoder.py
# Variational Auto-Encoder drop-in for MarketRegressionWithHeads.
#
# The VAE operates on x only. mean/std define the meaning of x and condition
# encoder/decoder via FiLM (not parallel time tokens). momentum is accepted for
# API compatibility but unused.
#
# Encoder and decoder are symmetric Transformers over N patch tokens
# (depth/width from num_layers, num_heads, ff_dim_multiplier).

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config
from task_base import resolve_model_past_len
from model_regression import (
    Depatch,
    PatchEmbed,
    PredictionHead,
    check_all_weights_updated,
    get_sinusoidal_encoding,
    report_model_size,
)


def resolve_vae_max_seq_len(past_window: int, patch_size: int) -> int:
    """Patch count for window length T (zero-padded to patch_size)."""
    T = int(past_window)
    P = int(patch_size)
    padded = ((T + P - 1) // P) * P
    return padded // P


def resolve_max_seq_len(config: dict, past_window: int) -> int:
    """Seq length for the active model_type."""
    P = int(config['patch_size'])
    if str(config.get('model_type', 'regression')).lower() == 'vae':
        return resolve_vae_max_seq_len(past_window, P)
    return int(past_window) // P


def build_market_model(
    config: dict,
    *,
    in_features: int,
    n_market_features: int,
    n_macro_features: int,
    n_max_past_horizon: int,
    max_seq_len: int,
):
    """Construct regression or VAE from ``config['model_type']``."""
    model_type = str(config.get('model_type', 'regression')).lower()
    common = dict(
        in_features=in_features,
        n_market_features=n_market_features,
        n_macro_features=n_macro_features,
        n_latent_macros=config['n_latent_macros'],
        n_max_past_horizon=n_max_past_horizon,
        patch_size=config['patch_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim_multiplier=config['ff_dim_multiplier'],
        dropout=config['dropout'],
        max_seq_len=max_seq_len,
        causal=config['causal'],
        head_hidden_dim_multiplier=config['head_hidden_dim_multiplier'],
        output_dim=config['output_dim'],
        use_reconstruction=config.get('use_reconstruction', False),
        emit_reconstruction=config.get(
            'emit_reconstruction', config.get('use_reconstruction', False)
        ),
        num_momentum=config.get('num_momentum', 4),
        look_price=config.get('look_price', True),
    )
    if model_type == 'vae':
        return MarketVAEWithHeads(
            **common,
            vae_latent_dim=int(config.get('vae_latent_dim', 64)),
        )
    if model_type != 'regression':
        raise ValueError(
            f"Unknown model_type={model_type!r}; expected 'regression' or 'vae'"
        )
    from model_regression import MarketRegressionWithHeads
    return MarketRegressionWithHeads(**common)


def unpack_model_input(sample_or_batch):
    """ModelSpec.unpack_input — see sample_batch.unpack_model_input."""
    from data import unpack_model_input as _unpack
    return _unpack(sample_or_batch)


def wrap_loss(model: nn.Module, label_loss_fn, config: dict, objective: Optional[str] = None):
    """
    ModelSpec.wrap_loss — compose train loss from COMPAT objective.

    objective:
      'supervised'   → label_loss_fn only (regression × true label)
      'unsupervised' → VAE recon+KL only; y ignored (vae × none)
    If objective is None, resolved from config via compat/COMPAT.
    """
    from task_base import assert_config_compatible
    from task_base import normalize_model_id

    if objective is None:
        objective = assert_config_compatible(config)

    recon_w = float(config.get('reconstruction_loss_weight', 1.0))
    kl_w = float(config.get('vae_kl_weight', 0.1))
    model_id = normalize_model_id(config.get('model_type', 'regression'))

    if objective == 'supervised':
        def loss_fn(x, pred, y):
            return label_loss_fn(x, pred, y)

        loss_fn.label_cls = getattr(label_loss_fn, 'label_cls', None)
        loss_fn.objective = objective  # type: ignore[attr-defined]
        loss_fn.auxiliary = getattr(label_loss_fn, 'auxiliary', None)
        return loss_fn

    if objective == 'unsupervised':
        if model_id != 'vae':
            raise ValueError(
                f"objective='unsupervised' expects model_type=vae, got {model_id!r}"
            )

        def loss_fn(x, pred, y):
            del y  # unused by design for (vae, none)
            x_hat = getattr(model, 'last_x_hat', None)
            x_tgt = getattr(model, 'last_x', None)
            if x_hat is None or x_tgt is None:
                raise RuntimeError(
                    "VAE loss requires model.last_x and model.last_x_hat; "
                    "was forward() called?"
                )
            loss = recon_w * F.mse_loss(x_hat, x_tgt)
            kl = getattr(model, 'last_kl', None)
            if kl is not None:
                loss = loss + kl_w * kl
            # Keep prediction head on the graph (zero contribution).
            if isinstance(pred, (tuple, list)):
                loss = loss + pred[0].sum() * 0.0
            elif torch.is_tensor(pred):
                loss = loss + pred.sum() * 0.0
            return loss

        loss_fn.label_cls = getattr(label_loss_fn, 'label_cls', None)
        loss_fn.objective = objective  # type: ignore[attr-defined]
        loss_fn.auxiliary = getattr(label_loss_fn, 'auxiliary', None)
        return loss_fn

    raise ValueError(
        f"Unknown COMPAT objective {objective!r}; expected "
        f"'supervised' or 'unsupervised'"
    )


def wrap_model_loss(model: nn.Module, base_loss_fn, config: dict):
    """Backward-compatible alias for ``wrap_loss``."""
    return wrap_loss(model, base_loss_fn, config)


class MeanStdConditioner(nn.Module):
    """
    mean‖std → shared cond, plus FiLM (γ, β) for encoder tokens and for z.

    mean/std are context for x, not parallel sequence tokens. Callers apply
    ``film`` instead of concatenating cond onto features.
    """

    def __init__(
        self,
        feature_dim: int,
        cond_dim: int,
        enc_film_dim: int,
        latent_film_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.enc_film_dim = int(enc_film_dim)
        self.latent_film_dim = int(latent_film_dim)
        self.to_cond = nn.Sequential(
            nn.Linear(2 * feature_dim, cond_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cond_dim, cond_dim),
        )
        # γ, β for FiLM: h' = (1+γ)⊙h + β
        self.to_film_enc = nn.Linear(cond_dim, 2 * enc_film_dim)
        self.to_film_z = nn.Linear(cond_dim, 2 * latent_film_dim)

    def forward(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mean.dim() != 2 or std.dim() != 2:
            raise ValueError(
                f"mean/std must be (B, F); got {tuple(mean.shape)}, {tuple(std.shape)}"
            )
        cond = self.to_cond(torch.cat([mean, std], dim=-1))  # (B, C)
        gamma_e, beta_e = self.to_film_enc(cond).chunk(2, dim=-1)
        gamma_z, beta_z = self.to_film_z(cond).chunk(2, dim=-1)
        return cond, gamma_e, beta_e, gamma_z, beta_z

    @staticmethod
    def film(h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Apply FiLM to (B, N, D) or (B, D) features."""
        if h.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return h * (1.0 + gamma) + beta


class MarketVAEWithHeads(nn.Module):
    """
    VAE on x, conditioned on mean/std.

    forward(x, p, mean, std, momentum) → (predictions, patch_map, recon)
      predictions: (B, output_dim)
      patch_map:   FiLM-modulated encoder tokens (B, N, D)
      recon:       x_hat (B, T, F) when use_reconstruction else None

    Decoder mirrors encoder: FiLM(z) seeds N patch tokens, same
    Transformer depth/width, then Depatch → x_hat.
    momentum is accepted for call-signature compatibility and ignored.
    """

    def __init__(
        self,
        in_features: int,
        n_market_features: int,
        n_macro_features: int,
        n_latent_macros: int,
        n_max_past_horizon: int,
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
        emit_reconstruction: Optional[bool] = None,
        num_momentum: int = 4,
        look_price: bool = True,
        vae_latent_dim: int = 64,
    ):
        super().__init__()
        output_dim = int(output_dim)
        if output_dim < 1:
            raise ValueError(f"output_dim must be a positive int, got {output_dim}")

        self.patch_size = int(patch_size)
        self.in_features = int(in_features)
        self.n_market_features = int(n_market_features)
        self.n_macro_features = int(n_macro_features)
        self.n_latent_macros = int(n_latent_macros)
        self.causal = bool(causal)
        self.max_seq_len = int(max_seq_len)
        if emit_reconstruction is None:
            emit_reconstruction = use_reconstruction
        self.use_reconstruction = bool(emit_reconstruction)
        self.emit_reconstruction = self.use_reconstruction
        self.output_dim = output_dim
        self.num_momentum = int(num_momentum)
        self.n_max_past_horizon = int(n_max_past_horizon)
        self.look_price = bool(look_price)
        self.vae_latent_dim = int(vae_latent_dim)
        self.n_stat_tokens = 1 + 1 + self.num_momentum  # API parity only
        self.F = int(in_features)

        L = self.n_latent_macros
        self.latent_macro_transform_x = nn.Linear(self.n_macro_features, L)

        n_feat_enc = self.n_market_features + L
        embed_dim = n_feat_enc * self.patch_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(n_feat_enc, self.patch_size, embed_dim)

        self.register_buffer(
            'pos_embed',
            get_sinusoidal_encoding(self.max_seq_len, embed_dim),
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
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Learnable pool over patch tokens (replaces mean(dim=1)).
        self.pool_score = nn.Linear(embed_dim, 1)

        # mean‖std → FiLM for encoder tokens and for z (no feature concat).
        self.conditioner = MeanStdConditioner(
            feature_dim=self.F,
            cond_dim=embed_dim,
            enc_film_dim=embed_dim,
            latent_film_dim=self.vae_latent_dim,
            dropout=dropout,
        )

        self.to_mu = nn.Linear(embed_dim, self.vae_latent_dim)
        self.to_logvar = nn.Linear(embed_dim, self.vae_latent_dim)

        # ---- Decoder (mirrors encoder complexity) ----
        # FiLM(z) → seed each of N patch tokens → TransformerEncoder
        # (same num_layers / num_heads / ff_dim) → Depatch → (B, T, F).
        # max_seq_len is patch count; max time steps = max_seq_len * patch_size.
        self.max_time = self.max_seq_len * self.patch_size
        self.z_to_token = nn.Linear(self.vae_latent_dim, embed_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.depatch = Depatch(self.F, self.patch_size, embed_dim)

        head_in = self.vae_latent_dim
        if self.look_price:
            head_in = head_in + self.n_max_past_horizon
        self.prediction_head = PredictionHead(
            input_dim=head_in,
            hidden_dim=head_hidden_dim_multiplier * head_in,
            output_dim=output_dim,
            dropout=dropout,
        )

        self.register_buffer('causal_mask', None)
        self.last_kl: Optional[torch.Tensor] = None
        self.last_kl_per: Optional[torch.Tensor] = None  # (B,)
        self.last_mu: Optional[torch.Tensor] = None
        self.last_logvar: Optional[torch.Tensor] = None
        self.last_x: Optional[torch.Tensor] = None
        self.last_x_hat: Optional[torch.Tensor] = None

    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        mask = mask.float().masked_fill(mask, float('-inf'))
        mask = mask.masked_fill(~mask.bool(), 0.0)
        return mask

    def _macro_compress(self, x_seq: torch.Tensor) -> torch.Tensor:
        """(B, T, F) → (B, T, n_market+L_macro)."""
        x_macro = x_seq[:, :, self.n_market_features :]
        x_macro = self.latent_macro_transform_x(x_macro)
        return torch.cat(
            [x_seq[:, :, : self.n_market_features], x_macro], dim=-1
        )

    def _pad_time(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Zero-pad time to a multiple of patch_size. Returns (padded, raw_len)."""
        _B, L, _F = x_seq.shape
        P = self.patch_size
        rem = L % P
        if rem == 0:
            return x_seq, L
        pad = P - rem
        x_seq = torch.nn.functional.pad(x_seq, (0, 0, 0, pad))
        return x_seq, L

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    @staticmethod
    def kl_divergence_per(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Per-sample KL(q(z|x)||N(0,I)), momentum (B,)."""
        return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    @classmethod
    def kl_divergence(cls, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return torch.mean(cls.kl_divergence_per(mu, logvar))

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        momentum: torch.Tensor,
    ):
        """
        Args:
            x:     (B, T, F) — primary VAE signal
            p:     (B, n_max_past_horizon) — prediction head only
            mean:  (B, F) — conditions encoder/decoder (with std)
            std:   (B, F) — conditions encoder/decoder (with mean)
            momentum: (B, S, F) — ignored (API compatibility)

        Returns:
            predictions, patch_map, recon(=x_hat or None)
        """
        del momentum  # explicit: not used in this VAE design
        B, T, Feat = x.shape
        if Feat != self.F:
            raise ValueError(f"x feature dim {Feat} != model.F {self.F}")
        if T > self.max_time:
            raise ValueError(
                f"T={T} exceeds decoder capacity {self.max_time} "
                f"(increase max_seq_len / patch_size)."
            )

        _cond, gamma_e, beta_e, gamma_z, beta_z = self.conditioner(mean, std)

        x_enc = self._macro_compress(x)
        x_enc, _raw_len = self._pad_time(x_enc)
        N = x_enc.shape[1] // self.patch_size
        if N > self.max_seq_len:
            raise ValueError(
                f"N={N} patches exceeds max_seq_len={self.max_seq_len}"
            )

        tokens = self.patch_embed(x_enc)  # (B, N, D)
        tokens = tokens + self.pos_embed[:, :N, :]

        if self.causal:
            if self.causal_mask is None or self.causal_mask.shape[0] != N:
                self.causal_mask = self._create_causal_mask(N).to(tokens.device)
            attn_mask = self.causal_mask
        else:
            attn_mask = None

        patch_map = self.norm(self.transformer(tokens, mask=attn_mask))
        # FiLM: mean/std modulate encoder features (not concatenated as tokens)
        patch_map = MeanStdConditioner.film(patch_map, gamma_e, beta_e)
        # Learnable weighted pool over N (not mean)
        pool_w = torch.softmax(self.pool_score(patch_map), dim=1)  # (B, N, 1)
        pooled = (patch_map * pool_w).sum(dim=1)  # (B, D)

        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled).clamp(-20.0, 20.0)
        z = self.reparameterize(mu, logvar)
        self.last_mu = mu
        self.last_logvar = logvar
        self.last_kl_per = self.kl_divergence_per(mu, logvar)
        self.last_kl = torch.mean(self.last_kl_per)

        # Decoder: FiLM(z) → N patch tokens → Transformer (same depth as enc)
        # → Depatch → (B, T_pad, F) → slice to T
        z_dec = MeanStdConditioner.film(z, gamma_z, beta_z)
        seed = self.z_to_token(z_dec).unsqueeze(1).expand(B, N, self.embed_dim)
        seed = seed + self.pos_embed[:, :N, :]
        seed = MeanStdConditioner.film(seed, gamma_e, beta_e)
        dec_tokens = self.decoder_norm(
            self.transformer_decoder(seed, mask=attn_mask)
        )
        x_hat_pad = self.depatch(dec_tokens)  # (B, N*P, F)
        x_hat = x_hat_pad[:, :T, :]
        self.last_x = x
        self.last_x_hat = x_hat

        head_in = z
        if self.look_price:
            if p is None:
                p = torch.zeros(
                    B, self.n_max_past_horizon, device=z.device, dtype=z.dtype
                )
            else:
                p = p.to(device=z.device, dtype=z.dtype)
                if p.dim() == 1:
                    p = p.unsqueeze(0)
                if p.shape[-1] != self.n_max_past_horizon:
                    raise ValueError(
                        f"p last dim {p.shape[-1]} != n_max_past_horizon "
                        f"{self.n_max_past_horizon}"
                    )
            head_in = torch.cat([z, p], dim=1)
        predictions = self.prediction_head(head_in)

        recon = x_hat if self.emit_reconstruction else None
        return predictions, patch_map, recon


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL_VAE MODULE DEMO (x | mean,std FiLM)")
    print("=" * 60)

    config = get_config()
    device = torch.device(config.get('device', 'cpu'))
    from telepred import resolve_past_window
    past_window = int(resolve_past_window(config))
    P = int(config['patch_size'])
    T = (past_window // P) * P
    L = int(config['n_latent_macros'])
    n_heads = int(config['num_heads'])
    n_market = 2
    while ((n_market + L) * P) % n_heads != 0:
        n_market += 1
        if n_market > 64:
            raise RuntimeError("Could not find n_market making embed_dim % num_heads == 0")
    n_macro = 10
    feat_dim = n_market + n_macro
    S = int(config.get('num_momentum', 4))
    max_seq_len = resolve_vae_max_seq_len(T, P)
    n_max_past = int(resolve_model_past_len(config))

    model = MarketVAEWithHeads(
        in_features=feat_dim,
        n_market_features=n_market,
        n_macro_features=n_macro,
        n_latent_macros=L,
        n_max_past_horizon=n_max_past,
        patch_size=P,
        num_layers=config['num_layers'],
        num_heads=n_heads,
        ff_dim_multiplier=config['ff_dim_multiplier'],
        dropout=config['dropout'],
        max_seq_len=max_seq_len,
        causal=config['causal'],
        head_hidden_dim_multiplier=config['head_hidden_dim_multiplier'],
        output_dim=config['output_dim'],
        use_reconstruction=True,
        num_momentum=S,
        look_price=config.get('look_price', True),
        vae_latent_dim=int(config.get('vae_latent_dim', 64)),
    ).to(device)

    report_model_size(model, name="MarketVAEWithHeads")

    B = 2
    x = torch.randn(B, T, feat_dim, device=device)
    p = torch.randn(B, n_max_past, device=device)
    mean = torch.randn(B, feat_dim, device=device)
    std = torch.randn(B, feat_dim, device=device).abs() + 1e-3
    momentum = torch.randn(B, S, feat_dim, device=device)

    pred, patch_map, recon = model(x, p, mean, std, momentum)
    print(f"\npredictions: {tuple(pred.shape)}")
    print(f"patch_map:   {tuple(patch_map.shape)}")
    print(f"recon:       {None if recon is None else tuple(recon.shape)}")
    print(f"last_x:      {tuple(model.last_x.shape)}")
    print(f"last_x_hat:  {tuple(model.last_x_hat.shape)}")
    print(f"last_kl:     {float(model.last_kl):.6f}")

    def _loss(x_, y_pred, y_truth):
        head = y_pred[0] if isinstance(y_pred, (tuple, list)) else y_pred
        return (
            torch.nn.functional.mse_loss(head, y_truth)
            + 1.0 * torch.nn.functional.mse_loss(model.last_x_hat, model.last_x)
            + 0.1 * model.last_kl
        )

    check_all_weights_updated(
        model,
        sample_batch=(x, p, mean, std, momentum),
        loss_fn=_loss,
        name="MarketVAEWithHeads",
    )
    print("\n✅ model_v_autoencoder demo complete.")
