from __future__ import annotations

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader

import config as cfg
from module.dataset import build_audio_dataset, collate_audio_windows
from module.models import Transformer, VectorQuantizer, build_frame_encoder_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the older causal-context inverse-dynamics design in isolation: "
            "causal transformer over z, concat h_t with z_{t+1}, then a small MLP "
            "optionally using LayerNorm before quantization."
        )
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=None,
        help="Hidden dim for the small action MLP head. Defaults to latent_dim.",
    )
    return parser.parse_args()


def build_loader(split: str, batch_size: int) -> DataLoader:
    dataset = build_audio_dataset(
        dataset_backend=cfg.dataset_backend,
        dataset_root=cfg.dataset_root,
        dataset_cache_root=cfg.dataset_cache_root,
        splits=(split,),
        sample_rate=cfg.audio_sample_rate,
        patch_size=cfg.audio_patch_samples,
        clip_seconds=cfg.audio_clip_seconds,
        clip_stride_seconds=cfg.audio_clip_stride_seconds,
        sequence_length=cfg.audio_sequence_length,
        sequence_stride=cfg.audio_sequence_stride,
        mono=cfg.audio_mono,
        normalization=cfg.audio_normalization,
        max_cached_payloads=cfg.dataset_max_cached_payloads,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_audio_windows,
        num_workers=0,
    )


def summarize_quantizer_geometry(
    *,
    action_logits: torch.Tensor,
    quantized: torch.Tensor,
    indices: torch.Tensor,
    codebook_weight: torch.Tensor,
    num_codes: int,
) -> dict[str, object]:
    flat_logits = action_logits.reshape(-1, action_logits.shape[-1])
    flat_quantized = quantized.reshape(-1, quantized.shape[-1])
    flat_distance = flat_logits - flat_quantized
    distances = (
        flat_logits.pow(2).sum(dim=1, keepdim=True)
        - 2 * flat_logits @ codebook_weight.t()
        + codebook_weight.pow(2).sum(dim=1)
    )
    nearest = distances.min(dim=1).values.sqrt()
    code_counts = torch.bincount(indices.reshape(-1).cpu(), minlength=num_codes)
    probabilities = code_counts.float() / code_counts.sum().clamp_min(1)
    nonzero = probabilities[probabilities > 0]
    entropy = -(nonzero * nonzero.log()).sum().item() if len(nonzero) > 0 else 0.0
    perplexity = float(torch.exp(torch.tensor(entropy)))
    return {
        "action_logits_norm": flat_logits.norm(dim=-1).mean().item(),
        "action_logits_std": action_logits.std(dim=tuple(range(action_logits.ndim - 1))).mean().item(),
        "quantized_norm": flat_quantized.norm(dim=-1).mean().item(),
        "quantized_std": quantized.std(dim=tuple(range(quantized.ndim - 1))).mean().item(),
        "distance_norm": flat_distance.norm(dim=-1).mean().item(),
        "distance_mse": torch.mean(flat_distance.pow(2)).item(),
        "nearest_distance_norm": nearest.mean().item(),
        "unique_codes": int((code_counts > 0).sum().item()),
        "code_perplexity": perplexity,
        "code_counts": code_counts.tolist(),
    }


class CausalConcatInverseDynamics(nn.Module):
    """Older ID design: causal context, concat with next latent, small MLP head."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        head_hidden_dim: int,
        use_layernorm: bool,
        use_post_layernorm: bool,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.context_transformer = Transformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.next_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        head_layers: list[nn.Module] = []
        if use_layernorm:
            head_layers.append(nn.LayerNorm(hidden_dim * 2))
        head_layers.extend(
            [
                nn.Linear(hidden_dim * 2, head_hidden_dim),
                nn.GELU(),
                nn.Linear(head_hidden_dim, input_dim),
            ]
        )
        self.action_head = nn.Sequential(*head_layers)
        self.residual_proj = (
            nn.Linear(hidden_dim, input_dim) if hidden_dim != input_dim else nn.Identity()
        )
        self.post_layernorm = nn.LayerNorm(input_dim) if use_post_layernorm else nn.Identity()

    def forward(self, z: torch.Tensor, *, use_residual: bool = False) -> torch.Tensor:
        past_context = self.context_transformer(z)
        next_latent = self.next_proj(z[:, 1:])
        transition_inputs = torch.cat(
            [past_context[:, :-1], next_latent],
            dim=-1,
        )
        action_logits = self.action_head(transition_inputs)
        if use_residual:
            action_logits = action_logits + self.residual_proj(next_latent)
        return self.post_layernorm(action_logits)


def build_encoder(device: str):
    return build_frame_encoder_from_config(cfg).to(device)


def build_variants(
    device: str, head_hidden_dim: int
) -> list[tuple[str, nn.Module, VectorQuantizer, bool]]:
    variants: list[tuple[str, nn.Module, VectorQuantizer, bool]] = []
    for use_layernorm, use_residual, label in (
        (False, False, "causal_concat_no_ln"),
        (True, False, "causal_concat_with_ln"),
        (False, True, "causal_concat_no_ln_residual"),
        (True, True, "causal_concat_with_ln_residual"),
        (False, False, "causal_concat_no_ln_post_ln"),
        (True, False, "causal_concat_with_ln_post_ln"),
        (False, True, "causal_concat_no_ln_residual_post_ln"),
        (True, True, "causal_concat_with_ln_residual_post_ln"),
    ):
        use_post_layernorm = label.endswith("post_ln")
        inverse_dynamics = CausalConcatInverseDynamics(
            input_dim=cfg.latent_dim,
            hidden_dim=cfg.id_hidden_dim,
            depth=cfg.id_depth,
            heads=cfg.heads,
            dim_head=cfg.dim_head,
            mlp_dim=cfg.mlp_dim,
            head_hidden_dim=head_hidden_dim,
            use_layernorm=use_layernorm,
            use_post_layernorm=use_post_layernorm,
            dropout=cfg.dropout,
        ).to(device)
        codebook = VectorQuantizer(
            num_codes=cfg.num_codes,
            code_dim=cfg.latent_dim,
            beta=cfg.codebook_beta,
        ).to(device)
        variants.append((label, inverse_dynamics, codebook, use_residual))
    return variants


def main() -> int:
    args = parse_args()
    head_hidden_dim = args.head_hidden_dim or cfg.latent_dim
    encoder = build_encoder(args.device)
    encoder.eval()
    variants = build_variants(args.device, head_hidden_dim)
    loader = build_loader(args.split, args.batch_size)

    with torch.no_grad():
        batch = next(iter(loader))
        signal_values = batch["signal_values"].to(args.device)
        z = encoder(signal_values)

    print(f"split: {args.split}")
    print(f"latent_dim: {cfg.latent_dim}")
    print(f"id_hidden_dim: {cfg.id_hidden_dim}")
    print(f"head_hidden_dim: {head_hidden_dim}")
    print(f"z_norm: {z.norm(dim=-1).mean().item():.8f}")
    print(f"z_std: {z.std(dim=(0, 1)).mean().item():.8f}")

    for name, inverse_dynamics, codebook, use_residual in variants:
        inverse_dynamics.eval()
        codebook.eval()
        stats_accum = {
            "action_logits_norm": 0.0,
            "action_logits_std": 0.0,
            "quantized_norm": 0.0,
            "quantized_std": 0.0,
            "distance_norm": 0.0,
            "distance_mse": 0.0,
            "nearest_distance_norm": 0.0,
        }
        code_counts = torch.zeros(cfg.num_codes, dtype=torch.long)
        batches = 0

        with torch.no_grad():
            for batch in loader:
                if batches >= args.max_batches:
                    break
                signal_values = batch["signal_values"].to(args.device)
                z = encoder(signal_values)
                action_logits = inverse_dynamics(z, use_residual=use_residual)
                quantizer_outputs = codebook(action_logits)
                quantized = quantizer_outputs["quantized"]
                indices = quantizer_outputs["indices"]
                stats = summarize_quantizer_geometry(
                    action_logits=action_logits,
                    quantized=quantized,
                    indices=indices,
                    codebook_weight=codebook.codebook.weight,
                    num_codes=cfg.num_codes,
                )
                for key in stats_accum:
                    stats_accum[key] += float(stats[key])
                code_counts += torch.bincount(indices.reshape(-1).cpu(), minlength=cfg.num_codes)
                batches += 1

        for key in stats_accum:
            stats_accum[key] /= batches
        probabilities = code_counts.float() / code_counts.sum().clamp_min(1)
        nonzero = probabilities[probabilities > 0]
        entropy = -(nonzero * nonzero.log()).sum().item() if len(nonzero) > 0 else 0.0
        perplexity = float(torch.exp(torch.tensor(entropy)))
        print(name + ":")
        print(f"  batches_evaluated: {batches}")
        print(f"  action_logits_norm: {stats_accum['action_logits_norm']:.8f}")
        print(f"  action_logits_std: {stats_accum['action_logits_std']:.8f}")
        print(f"  quantized_norm: {stats_accum['quantized_norm']:.8f}")
        print(f"  quantized_std: {stats_accum['quantized_std']:.8f}")
        print(f"  distance_norm: {stats_accum['distance_norm']:.8f}")
        print(f"  distance_mse: {stats_accum['distance_mse']:.8f}")
        print(f"  nearest_distance_norm: {stats_accum['nearest_distance_norm']:.8f}")
        print(f"  unique_codes: {int((code_counts > 0).sum().item())}")
        print(f"  code_perplexity: {perplexity:.8f}")
        print(f"  code_counts: {code_counts.tolist()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
