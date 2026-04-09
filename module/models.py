import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


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
    """Attention variant used by the inverse dynamics transformer."""

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
        _, t, _ = x.shape
        x = self.norm(x)
        attn_mask = torch.tril(
            torch.ones((t, t), device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        dropout_p = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(chunk, "b t (h d) -> b h t d", h=self.heads) for chunk in qkv)
        q, k = apply_rope(q, k)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=False,
            attn_mask=attn_mask,
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


class IDBlock(nn.Module):
    """Transformer block used by the inverse dynamics model."""

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


class Embedder(nn.Module):
    """Projects input features into encoder embeddings."""

    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.embed(x)


class InverseDynamicsTransformer(nn.Module):
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
        self.num_frames = num_frames
        self.dropout = nn.Dropout(emb_dropout)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [IDBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim or input_dim)
            if hidden_dim != (output_dim or input_dim)
            else nn.Identity()
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = self.dropout(x)
        x = self.input_proj(x)
        for block in self.layers:
            x = block(x)
        x = self.norm(x)
        return self.output_proj(x)


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
        self.num_frames = num_frames
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
