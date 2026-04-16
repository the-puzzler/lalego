from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config as cfg
from module.dataset import build_audio_dataset, collate_audio_windows
from module.models import build_frame_encoder_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure whether chunk latents are overly smooth or nearly constant."
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


def build_encoder(device: str):
    return build_frame_encoder_from_config(cfg).to(device)


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    encoder = build_encoder(args.device)
    encoder.load_state_dict(normalize_state_dict(checkpoint["frame_encoder"]))
    encoder.eval()
    loader = build_loader(args.split, args.batch_size)

    stats = {
        "adj_mse": 0.0,
        "adj_cos": 0.0,
        "rand_mse": 0.0,
        "rand_cos": 0.0,
        "delta_norm": 0.0,
        "latent_std": 0.0,
        "seq_var": 0.0,
    }
    batches = 0

    with torch.no_grad():
        for batch in loader:
            if batches >= args.max_batches:
                break

            signal_values = batch["signal_values"].to(args.device)
            latents = encoder(signal_values)
            adjacent_a = latents[:, :-1]
            adjacent_b = latents[:, 1:]
            random_perm = torch.randperm(latents.shape[0], device=latents.device)
            random_b = latents[random_perm, 1:]
            deltas = adjacent_b - adjacent_a

            stats["adj_mse"] += F.mse_loss(adjacent_a, adjacent_b).item()
            stats["adj_cos"] += F.cosine_similarity(adjacent_a, adjacent_b, dim=-1).mean().item()
            stats["rand_mse"] += F.mse_loss(adjacent_a, random_b).item()
            stats["rand_cos"] += F.cosine_similarity(adjacent_a, random_b, dim=-1).mean().item()
            stats["delta_norm"] += deltas.norm(dim=-1).mean().item()
            stats["latent_std"] += latents.std(dim=(0, 1)).mean().item()
            stats["seq_var"] += latents.var(dim=1).mean().item()
            batches += 1

    if batches == 0:
        raise RuntimeError("No batches were available for latent smoothness analysis.")

    for key in stats:
        stats[key] /= batches

    print(f"checkpoint: {args.checkpoint}")
    print(f"step: {int(checkpoint['step'])}")
    print(f"split: {args.split}")
    print(f"batches_evaluated: {batches}")
    print(f"adj_mse: {stats['adj_mse']:.8f}")
    print(f"adj_cos: {stats['adj_cos']:.8f}")
    print(f"rand_mse: {stats['rand_mse']:.8f}")
    print(f"rand_cos: {stats['rand_cos']:.8f}")
    print(f"delta_norm: {stats['delta_norm']:.8f}")
    print(f"latent_std: {stats['latent_std']:.8f}")
    print(f"seq_var: {stats['seq_var']:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
