from __future__ import annotations

import argparse
import types
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config as cfg
from module.dataset import build_audio_dataset, collate_audio_windows
from module.models import InverseDynamicsTransformer, SignalPatchEncoder, VectorQuantizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the geometry of state latents, inverse-dynamics outputs, and "
            "codebook matches for either random initialization or a saved checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to load. If omitted, uses random initialization.",
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
        "--compare-lewm",
        action="store_true",
        help="Also load https://github.com/the-puzzler/le-wm and report its random-init "
        "ID/codebook geometry on synthetic inputs.",
    )
    parser.add_argument(
        "--synthetic-batch-size",
        type=int,
        default=256,
        help="Batch size for synthetic comparison runs.",
    )
    parser.add_argument(
        "--override-num-codes",
        type=int,
        default=None,
        help="Optional temporary override for this repo's num_codes during analysis only.",
    )
    parser.add_argument(
        "--override-id-hidden-dim",
        type=int,
        default=None,
        help="Optional temporary override for this repo's inverse-dynamics hidden dim during analysis only.",
    )
    parser.add_argument(
        "--override-id-depth",
        type=int,
        default=None,
        help="Optional temporary override for this repo's inverse-dynamics depth during analysis only.",
    )
    parser.add_argument(
        "--override-id-mlp-dim",
        type=int,
        default=None,
        help="Optional temporary override for this repo's inverse-dynamics MLP dim during analysis only.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    return parser.parse_args()


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        normalized[key] = value
    return normalized


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


def build_models(
    *,
    device: str,
    num_codes: int,
    id_hidden_dim: int,
    id_depth: int,
    id_mlp_dim: int,
) -> tuple[SignalPatchEncoder, InverseDynamicsTransformer, VectorQuantizer]:
    encoder = SignalPatchEncoder(
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
    ).to(device)
    inverse_dynamics = InverseDynamicsTransformer(
        num_frames=cfg.audio_sequence_length,
        input_dim=cfg.latent_dim,
        hidden_dim=id_hidden_dim,
        output_dim=cfg.latent_dim,
        depth=id_depth,
        heads=cfg.heads,
        dim_head=cfg.dim_head,
        mlp_dim=id_mlp_dim,
        dropout=cfg.dropout,
    ).to(device)
    codebook = VectorQuantizer(
        num_codes=num_codes,
        code_dim=cfg.latent_dim,
        beta=cfg.codebook_beta,
    ).to(device)
    return encoder, inverse_dynamics, codebook


def load_remote_python_module(url: str, module_name: str) -> types.ModuleType:
    source = urllib.request.urlopen(url, timeout=30).read().decode("utf-8")
    module = types.ModuleType(module_name)
    module.__file__ = url
    exec(compile(source, url, "exec"), module.__dict__)
    return module


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


def compare_with_lewm(
    *,
    device: str,
    sequence_length: int,
    synthetic_batch_size: int,
    local_inverse_dynamics: InverseDynamicsTransformer,
    local_codebook: VectorQuantizer,
) -> None:
    module = load_remote_python_module(
        "https://raw.githubusercontent.com/the-puzzler/le-wm/main/module.py",
        "lewm_module",
    )
    remote_cfg = load_remote_python_module(
        "https://raw.githubusercontent.com/the-puzzler/le-wm/main/config.py",
        "lewm_config",
    )

    inverse_dynamics = module.InverseDynamicsTransformer(
        num_frames=sequence_length,
        input_dim=remote_cfg.EMBED_DIM,
        hidden_dim=remote_cfg.EMBED_DIM,
        output_dim=remote_cfg.EMBED_DIM,
        depth=remote_cfg.INVERSE_DYNAMICS_DEPTH,
        heads=remote_cfg.INVERSE_DYNAMICS_HEADS,
        dim_head=remote_cfg.INVERSE_DYNAMICS_DIM_HEAD,
        mlp_dim=remote_cfg.INVERSE_DYNAMICS_MLP_DIM,
        dropout=remote_cfg.INVERSE_DYNAMICS_DROPOUT,
        emb_dropout=remote_cfg.INVERSE_DYNAMICS_EMB_DROPOUT,
    ).to(device)
    codebook = module.VectorQuantizer(
        num_codes=remote_cfg.NUM_CODES,
        code_dim=remote_cfg.EMBED_DIM,
        beta=remote_cfg.CODEBOOK_BETA,
    ).to(device)
    inverse_dynamics.eval()
    codebook.eval()

    synthetic_inputs = {
        "zeros": torch.zeros(
            synthetic_batch_size,
            sequence_length,
            remote_cfg.EMBED_DIM,
            device=device,
        ),
        "gaussian": torch.randn(
            synthetic_batch_size,
            sequence_length,
            remote_cfg.EMBED_DIM,
            device=device,
        ),
    }

    print("synthetic_id_comparison:")
    for name, x in synthetic_inputs.items():
        local_x = x[..., : cfg.latent_dim]
        with torch.no_grad():
            local_action_logits = local_inverse_dynamics(local_x)
            local_outputs = local_codebook(local_action_logits)
        local_stats = summarize_quantizer_geometry(
            action_logits=local_action_logits,
            quantized=local_outputs["quantized"],
            indices=local_outputs["indices"],
            codebook_weight=local_codebook.codebook.weight,
            num_codes=local_codebook.num_codes,
        )
        print(
            {
                "repo": "this_repo",
                "input": name,
                "embed_dim": cfg.latent_dim,
                "num_codes": local_codebook.num_codes,
                **{k: round(v, 8) if isinstance(v, float) else v for k, v in local_stats.items()},
            }
        )
        with torch.no_grad():
            action_logits = inverse_dynamics(x)
            outputs = codebook(action_logits)
        stats = summarize_quantizer_geometry(
            action_logits=action_logits,
            quantized=outputs["quantized"],
            indices=outputs["indices"],
            codebook_weight=codebook.codebook.weight,
            num_codes=remote_cfg.NUM_CODES,
        )
        print(
            {
                "repo": "lewm",
                "input": name,
                "embed_dim": remote_cfg.EMBED_DIM,
                "num_codes": remote_cfg.NUM_CODES,
                **{k: round(v, 8) if isinstance(v, float) else v for k, v in stats.items()},
            }
        )


def main() -> int:
    args = parse_args()
    num_codes = args.override_num_codes or cfg.num_codes
    id_hidden_dim = args.override_id_hidden_dim or cfg.id_hidden_dim
    id_depth = args.override_id_depth or cfg.id_depth
    id_mlp_dim = args.override_id_mlp_dim or cfg.mlp_dim
    encoder, inverse_dynamics, codebook = build_models(
        device=args.device,
        num_codes=num_codes,
        id_hidden_dim=id_hidden_dim,
        id_depth=id_depth,
        id_mlp_dim=id_mlp_dim,
    )
    checkpoint_step = None

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        encoder.load_state_dict(normalize_state_dict(checkpoint["frame_encoder"]))
        inverse_dynamics.load_state_dict(normalize_state_dict(checkpoint["inverse_dynamics"]))
        codebook.load_state_dict(normalize_state_dict(checkpoint["codebook"]))
        checkpoint_step = int(checkpoint["step"])

    encoder.eval()
    inverse_dynamics.eval()
    codebook.eval()
    loader = build_loader(args.split, args.batch_size)

    stats = {
        "z_norm": 0.0,
        "z_std": 0.0,
        "action_logits_norm": 0.0,
        "action_logits_std": 0.0,
        "quantized_norm": 0.0,
        "quantized_std": 0.0,
        "distance_norm": 0.0,
        "distance_mse": 0.0,
        "nearest_distance_norm": 0.0,
    }
    code_counts = torch.zeros(num_codes, dtype=torch.long)
    batches = 0

    with torch.no_grad():
        for batch in loader:
            if batches >= args.max_batches:
                break
            signal_values = batch["signal_values"].to(args.device)
            z = encoder(signal_values)
            action_logits = inverse_dynamics(z)
            quantizer_outputs = codebook(action_logits)
            quantized = quantizer_outputs["quantized"]
            indices = quantizer_outputs["indices"]

            stats["z_norm"] += z.norm(dim=-1).mean().item()
            stats["z_std"] += z.std(dim=(0, 1)).mean().item()
            quantizer_stats = summarize_quantizer_geometry(
                action_logits=action_logits,
                quantized=quantized,
                indices=indices,
                codebook_weight=codebook.codebook.weight,
                num_codes=num_codes,
            )
            stats["action_logits_norm"] += float(quantizer_stats["action_logits_norm"])
            stats["action_logits_std"] += float(quantizer_stats["action_logits_std"])
            stats["quantized_norm"] += float(quantizer_stats["quantized_norm"])
            stats["quantized_std"] += float(quantizer_stats["quantized_std"])
            stats["distance_norm"] += float(quantizer_stats["distance_norm"])
            stats["distance_mse"] += float(quantizer_stats["distance_mse"])
            stats["nearest_distance_norm"] += float(quantizer_stats["nearest_distance_norm"])
            code_counts += torch.bincount(indices.reshape(-1).cpu(), minlength=num_codes)
            batches += 1

    if batches == 0:
        raise RuntimeError("No batches were available for action geometry analysis.")

    for key in stats:
        stats[key] /= batches

    probabilities = code_counts.float() / code_counts.sum().clamp_min(1)
    nonzero = probabilities[probabilities > 0]
    entropy = -(nonzero * nonzero.log()).sum().item() if len(nonzero) > 0 else 0.0
    perplexity = float(torch.exp(torch.tensor(entropy)))

    print(f"mode: {'checkpoint' if args.checkpoint else 'random_init'}")
    print(f"num_codes: {num_codes}")
    print(f"id_hidden_dim: {id_hidden_dim}")
    print(f"id_depth: {id_depth}")
    print(f"id_mlp_dim: {id_mlp_dim}")
    if args.checkpoint is not None:
        print(f"checkpoint: {args.checkpoint}")
    if checkpoint_step is not None:
        print(f"step: {checkpoint_step}")
    print(f"split: {args.split}")
    print(f"batches_evaluated: {batches}")
    print(f"z_norm: {stats['z_norm']:.8f}")
    print(f"z_std: {stats['z_std']:.8f}")
    print(f"action_logits_norm: {stats['action_logits_norm']:.8f}")
    print(f"action_logits_std: {stats['action_logits_std']:.8f}")
    print(f"quantized_norm: {stats['quantized_norm']:.8f}")
    print(f"quantized_std: {stats['quantized_std']:.8f}")
    print(f"distance_norm: {stats['distance_norm']:.8f}")
    print(f"distance_mse: {stats['distance_mse']:.8f}")
    print(f"nearest_distance_norm: {stats['nearest_distance_norm']:.8f}")
    print(f"unique_codes: {int((code_counts > 0).sum().item())}")
    print(f"code_perplexity: {perplexity:.8f}")
    print(f"code_counts: {code_counts.tolist()}")
    if args.compare_lewm:
        compare_with_lewm(
            device=args.device,
            sequence_length=cfg.audio_sequence_length,
            synthetic_batch_size=args.synthetic_batch_size,
            local_inverse_dynamics=inverse_dynamics,
            local_codebook=codebook,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
