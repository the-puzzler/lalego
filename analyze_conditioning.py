from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config as cfg
from module.dataset import build_audio_dataset, collate_audio_windows
from module.models import ARPredictor, InverseDynamicsTransformer, SignalPatchEncoder, VectorQuantizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether the predictor is using inferred latent actions by "
            "comparing checkpoint performance under action ablations."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/20260414_161558/latest.pt"),
        help="Path to a training checkpoint.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for the analysis loader.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=32,
        help="Maximum number of batches to evaluate.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use.",
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


def build_models(device: str) -> tuple[SignalPatchEncoder, InverseDynamicsTransformer, VectorQuantizer, ARPredictor]:
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
        hidden_dim=cfg.id_hidden_dim,
        output_dim=cfg.latent_dim,
        depth=cfg.id_depth,
        heads=cfg.heads,
        dim_head=cfg.dim_head,
        mlp_dim=cfg.mlp_dim,
        dropout=cfg.dropout,
    ).to(device)
    codebook = VectorQuantizer(
        num_codes=cfg.num_codes,
        code_dim=cfg.latent_dim,
        beta=cfg.codebook_beta,
    ).to(device)
    predictor = ARPredictor(
        num_frames=cfg.audio_sequence_length - 1,
        input_dim=cfg.latent_dim,
        hidden_dim=cfg.predictor_hidden_dim,
        output_dim=cfg.latent_dim,
        depth=cfg.predictor_depth,
        heads=cfg.heads,
        dim_head=cfg.dim_head,
        mlp_dim=cfg.mlp_dim,
        dropout=cfg.dropout,
    ).to(device)
    return encoder, inverse_dynamics, codebook, predictor


def load_checkpoint(
    checkpoint_path: Path,
    *,
    device: str,
) -> tuple[SignalPatchEncoder, InverseDynamicsTransformer, VectorQuantizer, ARPredictor, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder, inverse_dynamics, codebook, predictor = build_models(device)
    encoder.load_state_dict(normalize_state_dict(checkpoint["frame_encoder"]))
    inverse_dynamics.load_state_dict(normalize_state_dict(checkpoint["inverse_dynamics"]))
    codebook.load_state_dict(normalize_state_dict(checkpoint["codebook"]))
    predictor.load_state_dict(normalize_state_dict(checkpoint["predictor"]))
    encoder.eval()
    inverse_dynamics.eval()
    codebook.eval()
    predictor.eval()
    return encoder, inverse_dynamics, codebook, predictor, int(checkpoint["step"])


def summarize_conditioning(
    *,
    loader: DataLoader,
    device: str,
    encoder: SignalPatchEncoder,
    inverse_dynamics: InverseDynamicsTransformer,
    codebook: VectorQuantizer,
    predictor: ARPredictor,
    max_batches: int,
) -> dict[str, float | list[int]]:
    stats = {
        "mse_real": 0.0,
        "mse_zero": 0.0,
        "mse_random": 0.0,
        "pred_diff_zero": 0.0,
        "pred_diff_random": 0.0,
        "action_norm": 0.0,
        "gate_msa_abs_mean": 0.0,
        "gate_mlp_abs_mean": 0.0,
    }
    code_counts = torch.zeros(cfg.num_codes, dtype=torch.long)
    batches = 0

    with torch.no_grad():
        for batch in loader:
            if batches >= max_batches:
                break

            signal_values = batch["signal_values"].to(device)
            latents = encoder(signal_values)
            quantizer_outputs = codebook(inverse_dynamics(latents))
            actions = quantizer_outputs["quantized"]
            action_indices = quantizer_outputs["indices"]
            targets = latents[:, 1:]

            predictions_real = predictor(latents[:, :-1], actions)
            predictions_zero = predictor(latents[:, :-1], torch.zeros_like(actions))
            random_indices = torch.randint(
                cfg.num_codes,
                action_indices.shape,
                device=device,
            )
            random_actions = codebook.codebook(random_indices)
            predictions_random = predictor(latents[:, :-1], random_actions)

            stats["mse_real"] += F.mse_loss(predictions_real, targets).item()
            stats["mse_zero"] += F.mse_loss(predictions_zero, targets).item()
            stats["mse_random"] += F.mse_loss(predictions_random, targets).item()
            stats["pred_diff_zero"] += F.mse_loss(predictions_real, predictions_zero).item()
            stats["pred_diff_random"] += F.mse_loss(predictions_real, predictions_random).item()
            stats["action_norm"] += actions.norm(dim=-1).mean().item()

            block = predictor.transformer.layers[0]
            conditioning_proj = predictor.transformer.cond_proj(actions)
            modulation = block.ada_ln_modulation(conditioning_proj)
            _, _, gate_msa, _, _, gate_mlp = modulation.chunk(6, dim=-1)
            stats["gate_msa_abs_mean"] += gate_msa.abs().mean().item()
            stats["gate_mlp_abs_mean"] += gate_mlp.abs().mean().item()

            code_counts += torch.bincount(action_indices.reshape(-1).cpu(), minlength=cfg.num_codes)
            batches += 1

    if batches == 0:
        raise RuntimeError("No batches were available for conditioning analysis.")

    for key in stats:
        stats[key] /= batches

    probabilities = code_counts.float() / code_counts.sum().clamp_min(1)
    nonzero = probabilities[probabilities > 0]
    entropy = -(nonzero * nonzero.log()).sum().item() if len(nonzero) > 0 else 0.0
    perplexity = float(torch.exp(torch.tensor(entropy)))
    unique_codes = int((code_counts > 0).sum().item())

    return {
        "batches": batches,
        "mse_real": stats["mse_real"],
        "mse_zero": stats["mse_zero"],
        "mse_random": stats["mse_random"],
        "delta_mse_zero": stats["mse_zero"] - stats["mse_real"],
        "delta_mse_random": stats["mse_random"] - stats["mse_real"],
        "pred_diff_zero": stats["pred_diff_zero"],
        "pred_diff_random": stats["pred_diff_random"],
        "action_norm": stats["action_norm"],
        "gate_msa_abs_mean": stats["gate_msa_abs_mean"],
        "gate_mlp_abs_mean": stats["gate_mlp_abs_mean"],
        "unique_codes": unique_codes,
        "code_perplexity": perplexity,
        "code_counts": code_counts.tolist(),
    }


def main() -> int:
    args = parse_args()
    loader = build_loader(args.split, args.batch_size)
    encoder, inverse_dynamics, codebook, predictor, step = load_checkpoint(
        args.checkpoint,
        device=args.device,
    )
    results = summarize_conditioning(
        loader=loader,
        device=args.device,
        encoder=encoder,
        inverse_dynamics=inverse_dynamics,
        codebook=codebook,
        predictor=predictor,
        max_batches=args.max_batches,
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"step: {step}")
    print(f"split: {args.split}")
    print(f"batches_evaluated: {results['batches']}")
    print(f"mse_real: {results['mse_real']:.8f}")
    print(f"mse_zero: {results['mse_zero']:.8f}")
    print(f"mse_random: {results['mse_random']:.8f}")
    print(f"delta_mse_zero: {results['delta_mse_zero']:.8f}")
    print(f"delta_mse_random: {results['delta_mse_random']:.8f}")
    print(f"pred_diff_zero: {results['pred_diff_zero']:.8f}")
    print(f"pred_diff_random: {results['pred_diff_random']:.8f}")
    print(f"action_norm: {results['action_norm']:.8f}")
    print(f"gate_msa_abs_mean: {results['gate_msa_abs_mean']:.8f}")
    print(f"gate_mlp_abs_mean: {results['gate_mlp_abs_mean']:.8f}")
    print(f"unique_codes: {results['unique_codes']}")
    print(f"code_perplexity: {results['code_perplexity']:.8f}")
    print(f"code_counts: {results['code_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
