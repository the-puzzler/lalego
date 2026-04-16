import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import EncodecModel


def modulate(x, shift, scale):
    """AdaLN-zero modulation."""
    return x * (1 + scale) + shift


def rotate_half(x):
    """Rotate pairs of features for RoPE."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)


def apply_rope(q, k):
    """Apply rotary position embeddings to query/key tensors."""
    _, _, t, d = q.shape
    if d % 2 != 0:
        raise ValueError("RoPE requires an even dim_head")

    positions = torch.arange(t, device=q.device, dtype=q.dtype)
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, d, 2, device=q.device, dtype=q.dtype) / d)
    )
    freqs = torch.outer(positions, inv_freq)
    cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)[None, None, :, :]
    sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)[None, None, :, :]
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class FeedForward(nn.Module):
    """Feed-forward network used in the encoder transformer."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product self-attention with causal masking support."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x: (B, T, D)
        """
        x = self.norm(x)
        dropout_p = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        q, k = apply_rope(q, k)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=causal,
        )
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class IDAttention(nn.Module):
    """Self-attention that can see the current token and one step ahead."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        _, sequence_length, _ = x.shape
        x = self.norm(x)
        dropout_p = self.dropout if self.training else 0.0
        peek_mask = torch.tril(
            torch.ones(
                (sequence_length, sequence_length),
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        q, k = apply_rope(q, k)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=False,
            attn_mask=peek_mask,
        )
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class Block(nn.Module):
    """Standard transformer block used by the encoder."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class IDBlock(nn.Module):
    """Transformer block for inverse dynamics with one-step peek attention."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = IDAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

        nn.init.constant_(self.ada_ln_modulation[-1].weight, 0)
        nn.init.constant_(self.ada_ln_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.ada_ln_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    """Transformer encoder stack."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.is_conditional = block_class is ConditionalBlock
        self.norm = nn.LayerNorm(hidden_dim)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        if self.is_conditional:
            self.cond_proj = (
                nn.Linear(input_dim, hidden_dim)
                if input_dim != hidden_dim
                else nn.Identity()
            )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )

    def forward(self, x, c=None):
        x = self.input_proj(x)
        if self.is_conditional:
            if c is None:
                raise ValueError("conditional transformer requires conditioning tensor c")
            c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c) if self.is_conditional else block(x)
        x = self.norm(x)
        return self.output_proj(x)


class SignalPatchEncoder(nn.Module):
    """Encode each temporal chunk into a single latent via internal patchification."""

    def __init__(
        self,
        *,
        num_channels: int,
        patch_size: int,
        hidden_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        output_dim: int | None = None,
        projector_hidden_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        conv_channels = [
            max(hidden_dim // 4, 32),
            max(hidden_dim // 2, 64),
            hidden_dim,
            hidden_dim,
        ]
        # A deeper strided conv frontend extracts richer local audio features before tokenization.
        self.frontend = nn.Sequential(
            nn.Conv1d(num_channels, conv_channels[0], kernel_size=15, stride=5, padding=7),
            nn.GELU(),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=15, stride=4, padding=7),
            nn.GELU(),
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=9, stride=2, padding=4),
            nn.GELU(),
            nn.Conv1d(conv_channels[2], conv_channels[3], kernel_size=9, stride=1, padding=4),
            nn.GELU(),
        )
        self.transformer = Transformer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim or hidden_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        projector_output_dim = output_dim or hidden_dim
        projector_hidden_dim = projector_hidden_dim or hidden_dim
        self.projector = nn.Sequential(
            nn.Linear(projector_output_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.GELU(),
            nn.Linear(projector_hidden_dim, projector_output_dim),
        )

    def forward(self, signal_values: torch.Tensor) -> torch.Tensor:
        """
        signal_values: (B, T, C, S)
        returns: (B, T, D), one latent per chunk
        """
        batch_size, num_chunks, num_channels, num_samples = signal_values.shape
        flat_chunks = signal_values.reshape(batch_size * num_chunks, num_channels, num_samples)
        features = self.frontend(flat_chunks)
        num_patches = max(1, num_samples // self.patch_size)
        pooled = F.adaptive_avg_pool1d(features, num_patches)
        tokens = pooled.transpose(1, 2).contiguous()
        cls_tokens = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([tokens, cls_tokens], dim=1)
        encoded = self.transformer(tokens)
        cls_latents = encoded[:, -1]
        projected = self.projector(cls_latents)
        return projected.reshape(batch_size, num_chunks, -1)


class EncodecPatchEncoder(nn.Module):
    """Use continuous EnCodec encoder embeddings as patch tokens for chunk encoding."""

    def __init__(
        self,
        *,
        model_name: str,
        input_sample_rate: int,
        hidden_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        output_dim: int | None = None,
        projector_hidden_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        freeze_encodec: bool = True,
    ):
        super().__init__()
        self.input_sample_rate = int(input_sample_rate)
        self.encodec = EncodecModel.from_pretrained(model_name)
        self.encodec.eval()
        self.encodec_sample_rate = int(self.encodec.config.sampling_rate)
        self.encodec_channels = int(self.encodec.config.audio_channels)
        if freeze_encodec:
            for parameter in self.encodec.parameters():
                parameter.requires_grad_(False)

        encodec_dim = int(self.encodec.config.hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.input_proj = (
            nn.Linear(encodec_dim, hidden_dim) if encodec_dim != hidden_dim else nn.Identity()
        )
        self.transformer = Transformer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim or hidden_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        projector_output_dim = output_dim or hidden_dim
        projector_hidden_dim = projector_hidden_dim or hidden_dim
        self.projector = nn.Sequential(
            nn.Linear(projector_output_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.GELU(),
            nn.Linear(projector_hidden_dim, projector_output_dim),
        )

    def _prepare_waveform(self, flat_chunks: torch.Tensor) -> torch.Tensor:
        waveform = flat_chunks
        if waveform.shape[1] != self.encodec_channels:
            if self.encodec_channels == 1:
                waveform = waveform.mean(dim=1, keepdim=True)
            else:
                waveform = waveform.expand(-1, self.encodec_channels, -1)
        if self.input_sample_rate != self.encodec_sample_rate:
            target_num_samples = max(
                1,
                int(round(waveform.shape[-1] * self.encodec_sample_rate / self.input_sample_rate)),
            )
            waveform = F.interpolate(
                waveform,
                size=target_num_samples,
                mode="linear",
                align_corners=False,
            )
        return waveform

    def forward(self, signal_values: torch.Tensor) -> torch.Tensor:
        """
        signal_values: (B, T, C, S)
        returns: (B, T, D), one latent per chunk
        """
        batch_size, num_chunks, num_channels, num_samples = signal_values.shape
        flat_chunks = signal_values.reshape(batch_size * num_chunks, num_channels, num_samples)
        waveform = self._prepare_waveform(flat_chunks)
        encodec_context = torch.no_grad() if not any(
            parameter.requires_grad for parameter in self.encodec.parameters()
        ) else torch.enable_grad()
        with encodec_context:
            features = self.encodec.encoder(waveform)
        tokens = features.transpose(1, 2).contiguous()
        tokens = self.input_proj(tokens)
        cls_tokens = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([tokens, cls_tokens], dim=1)
        encoded = self.transformer(tokens)
        cls_latents = encoded[:, -1]
        projected = self.projector(cls_latents)
        return projected.reshape(batch_size, num_chunks, -1)


def build_frame_encoder_from_config(cfg) -> nn.Module:
    encoder_type = getattr(cfg, "frame_encoder_type", "waveform")
    if encoder_type == "waveform":
        return SignalPatchEncoder(
            num_channels=cfg.audio_num_channels,
            patch_size=cfg.audio_patch_samples,
            hidden_dim=cfg.frame_hidden_dim,
            depth=cfg.frame_depth,
            heads=cfg.frame_heads,
            mlp_dim=cfg.frame_mlp_dim,
            output_dim=cfg.latent_dim,
            projector_hidden_dim=cfg.frame_projector_hidden_dim,
            dim_head=cfg.dim_head,
            dropout=cfg.dropout,
        )
    if encoder_type == "encodec":
        return EncodecPatchEncoder(
            model_name=cfg.encodec_model_name,
            input_sample_rate=cfg.audio_sample_rate,
            hidden_dim=cfg.frame_hidden_dim,
            depth=cfg.frame_depth,
            heads=cfg.frame_heads,
            mlp_dim=cfg.frame_mlp_dim,
            output_dim=cfg.latent_dim,
            projector_hidden_dim=cfg.frame_projector_hidden_dim,
            dim_head=cfg.dim_head,
            dropout=cfg.dropout,
            freeze_encodec=getattr(cfg, "freeze_encodec", True),
        )
    raise ValueError(f"Unsupported frame_encoder_type={encoder_type!r}")


class InverseDynamicsTransformer(nn.Module):
    """Infer latent actions with one-step peek attention and direct quantization output."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(emb_dropout)
        action_dim = output_dim or input_dim
        self.transformer = Transformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            block_class=IDBlock,
        )
        self.final_norm = nn.LayerNorm(action_dim)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = self.dropout(x)
        return self.final_norm(self.transformer(x)[:, :-1])


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim or input_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, D)
        c: (B, T, D)
        """
        x = self.dropout(x)
        return self.transformer(x, c)


class VectorQuantizer(nn.Module):
    """VQ-VAE style codebook with straight-through estimation."""

    def __init__(self, num_codes: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, x):
        """
        x: (B, T, D)
        returns:
            quantized: (B, T, D)
            indices: (B, T)
            codebook_loss: scalar
            commitment_loss: scalar
        """
        flat_x = x.reshape(-1, self.code_dim)
        codebook = self.codebook.weight

        distances = (
            flat_x.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_x @ codebook.t()
            + codebook.pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        quantized = self.codebook(indices).view_as(x)

        codebook_loss = F.mse_loss(quantized, x.detach())
        commitment_loss = F.mse_loss(x, quantized.detach())

        # Forward uses quantized values while gradients flow through x.
        quantized = x + (quantized - x).detach()

        return {
            "quantized": quantized,
            "indices": indices.view(*x.shape[:-1]),
            "codebook_loss": codebook_loss,
            "commitment_loss": self.beta * commitment_loss,
        }
