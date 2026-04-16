from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config as cfg
from module.dataset import build_audio_dataset, collate_audio_windows
from module.models import (
    InverseDynamicsTransformer,
    VectorQuantizer,
    build_frame_encoder_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize where discrete action codes are used and dump representative "
            "transition examples for each code."
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
        "--examples-per-code",
        type=int,
        default=3,
        help="How many low/median/high-delta examples to print per code.",
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


def build_models(device: str):
    encoder = build_frame_encoder_from_config(cfg).to(device)
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
    return encoder, inverse_dynamics, codebook


def select_examples(
    entries: list[dict[str, object]],
    *,
    examples_per_code: int,
) -> list[dict[str, object]]:
    if not entries:
        return []

    sorted_entries = sorted(entries, key=lambda entry: float(entry["delta_norm"]))
    positions = {0, len(sorted_entries) // 2, len(sorted_entries) - 1}
    if examples_per_code > 3:
        step = max(1, len(sorted_entries) // examples_per_code)
        positions.update(range(0, len(sorted_entries), step))
    selected = [sorted_entries[index] for index in sorted(positions)[:examples_per_code]]
    return selected


def to_float(value: object) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    encoder, inverse_dynamics, codebook = build_models(args.device)
    encoder.load_state_dict(normalize_state_dict(checkpoint["frame_encoder"]))
    inverse_dynamics.load_state_dict(normalize_state_dict(checkpoint["inverse_dynamics"]))
    codebook.load_state_dict(normalize_state_dict(checkpoint["codebook"]))
    encoder.eval()
    inverse_dynamics.eval()
    codebook.eval()

    loader = build_loader(args.split, args.batch_size)

    code_counts = Counter()
    position_counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
    code_examples: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    code_composers: defaultdict[int, Counter[str]] = defaultdict(Counter)
    delta_norms: defaultdict[int, list[float]] = defaultdict(list)
    delta_sum: dict[int, torch.Tensor] = {}
    batches = 0

    with torch.no_grad():
        for batch in loader:
            if batches >= args.max_batches:
                break

            signal_values = batch["signal_values"].to(args.device)
            latents = encoder(signal_values)
            action_outputs = codebook(inverse_dynamics(latents))
            action_indices = action_outputs["indices"]
            quantized_actions = action_outputs["quantized"]
            deltas = latents[:, 1:] - latents[:, :-1]

            for batch_index, metadata in enumerate(batch["metadata"]):
                for transition_index in range(action_indices.shape[1]):
                    code = int(action_indices[batch_index, transition_index].item())
                    delta = deltas[batch_index, transition_index]
                    delta_norm = float(delta.norm().item())
                    mean_delta = delta.detach().cpu()

                    code_counts[code] += 1
                    position_counts[transition_index][code] += 1
                    code_composers[code][str(metadata["composer"])] += 1
                    delta_norms[code].append(delta_norm)
                    if code not in delta_sum:
                        delta_sum[code] = mean_delta.clone()
                    else:
                        delta_sum[code] += mean_delta

                    clip_starts = metadata["clip_start_seconds"]
                    code_examples[code].append(
                        {
                            "key": batch["key"][batch_index],
                            "composer": metadata["composer"],
                            "title": metadata["title"],
                            "year": metadata["year"],
                            "transition_index": transition_index,
                            "state_start_seconds": to_float(clip_starts[transition_index]),
                            "next_state_start_seconds": to_float(
                                clip_starts[transition_index + 1]
                            ),
                            "delta_norm": delta_norm,
                            "action_norm": float(
                                quantized_actions[batch_index, transition_index].norm().item()
                            ),
                        }
                    )
            batches += 1

    if batches == 0:
        raise RuntimeError("No batches were available for code example analysis.")

    print(f"checkpoint: {args.checkpoint}")
    print(f"step: {int(checkpoint['step'])}")
    print(f"split: {args.split}")
    print(f"batches_evaluated: {batches}")
    print(f"code_counts: {dict(sorted(code_counts.items()))}")
    print("position_distributions:")
    for position in sorted(position_counts):
        print(
            f"  transition_{position}: "
            f"{position_counts[position].most_common(8)}"
        )

    print("per_code_summary:")
    for code, count in code_counts.most_common():
        mean_delta = delta_sum[code] / count
        norms = torch.tensor(delta_norms[code])
        print(
            {
                "code": code,
                "count": count,
                "delta_norm_mean": round(float(norms.mean().item()), 4),
                "delta_norm_std": round(float(norms.std(unbiased=False).item()), 4),
                "mean_delta_norm": round(float(mean_delta.norm().item()), 4),
                "top_composers": code_composers[code].most_common(4),
            }
        )

    codes = sorted(code_counts)
    print("mean_delta_cosine_matrix:")
    means = {code: delta_sum[code] / code_counts[code] for code in codes}
    for code_a in codes:
        row = []
        for code_b in codes:
            similarity = F.cosine_similarity(
                means[code_a].unsqueeze(0),
                means[code_b].unsqueeze(0),
            ).item()
            row.append(round(similarity, 3))
        print(f"  {code_a}: {row}")

    print("representative_examples:")
    for code, _ in code_counts.most_common():
        print(f"  code_{code}:")
        for entry in select_examples(
            code_examples[code],
            examples_per_code=args.examples_per_code,
        ):
            print(
                "   ",
                {
                    "composer": entry["composer"],
                    "title": entry["title"],
                    "year": entry["year"],
                    "transition_index": entry["transition_index"],
                    "state_start_seconds": round(float(entry["state_start_seconds"]), 2),
                    "next_state_start_seconds": round(
                        float(entry["next_state_start_seconds"]),
                        2,
                    ),
                    "delta_norm": round(float(entry["delta_norm"]), 4),
                    "action_norm": round(float(entry["action_norm"]), 4),
                    "key": entry["key"],
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
