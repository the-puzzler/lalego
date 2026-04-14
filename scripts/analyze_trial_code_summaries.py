from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402
from module.dataset import EEGTokenDataset, collate_eeg_windows  # noqa: E402
from module.models import EEGPatchEncoder, InverseDynamicsTransformer, VectorQuantizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-trial action-code usage for a checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "runs" / sorted((ROOT / "runs").iterdir())[-1] / "latest.pt",
        help="Path to checkpoint.",
    )
    return parser.parse_args()


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def entropy_from_counts(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p + 1e-12)
    return entropy


def main() -> None:
    args = parse_args()
    if cfg.action_source != "inferred":
        raise ValueError("This analysis is intended for inferred-action checkpoints only.")

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    dataset = EEGTokenDataset(
        data_root=cfg.dataset_root,
        data_files=cfg.data_files,
        sample_rate=cfg.dataset_fps,
        num_channels=cfg.eeg_num_channels,
        subject_ids=cfg.train_subject_ids + cfg.val_subject_ids,
        train_session_suffixes=cfg.train_session_suffixes,
        patch_size=cfg.eeg_patch_size,
        bandpass_low_hz=cfg.eeg_bandpass_low_hz,
        bandpass_high_hz=cfg.eeg_bandpass_high_hz,
        epoch_start_seconds=cfg.eeg_epoch_start_seconds,
        epoch_end_seconds=cfg.eeg_epoch_end_seconds,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_eeg_windows,
        num_workers=0,
    )

    first_batch = next(iter(loader))
    num_tokens = int(first_batch["eeg_values"].shape[1])
    frame_encoder = EEGPatchEncoder(
        num_channels=cfg.eeg_num_channels,
        patch_size=cfg.eeg_patch_size,
        hidden_dim=cfg.frame_hidden_dim,
        depth=cfg.frame_depth,
        heads=cfg.frame_heads,
        mlp_dim=cfg.frame_mlp_dim,
        output_dim=cfg.latent_dim,
        dim_head=cfg.dim_head,
        dropout=cfg.dropout,
    ).to(device)
    codebook = VectorQuantizer(
        num_codes=cfg.num_codes,
        code_dim=cfg.latent_dim,
        beta=cfg.codebook_beta,
    ).to(device)
    inverse_dynamics = InverseDynamicsTransformer(
        num_frames=num_tokens,
        input_dim=cfg.latent_dim,
        hidden_dim=cfg.id_hidden_dim,
        output_dim=cfg.latent_dim,
        depth=cfg.id_depth,
        heads=cfg.heads,
        dim_head=cfg.dim_head,
        mlp_dim=cfg.mlp_dim,
        dropout=cfg.dropout,
    ).to(device)

    checkpoint = torch.load(args.checkpoint.resolve(), map_location=device)
    frame_encoder.load_state_dict(normalize_state_dict(checkpoint["frame_encoder"]))
    codebook.load_state_dict(normalize_state_dict(checkpoint["codebook"]))
    inverse_dynamics.load_state_dict(normalize_state_dict(checkpoint["inverse_dynamics"]))
    frame_encoder.eval()
    codebook.eval()
    inverse_dynamics.eval()

    by_label_majority: dict[str, Counter[int]] = defaultdict(Counter)
    by_label_unique_count: defaultdict[str, list[int]] = defaultdict(list)
    by_label_majority_fraction: defaultdict[str, list[float]] = defaultdict(list)
    by_label_entropy: defaultdict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            eeg_values = batch["eeg_values"].to(device)
            latents = frame_encoder(eeg_values)
            action_logits = inverse_dynamics(latents)
            indices = codebook(action_logits)["indices"].cpu()

            for codes_tensor, label_name in zip(indices, batch["label_name"]):
                codes = codes_tensor.tolist()
                counts = Counter(codes)
                majority_code, majority_count = counts.most_common(1)[0]
                by_label_majority[label_name][majority_code] += 1
                by_label_unique_count[label_name].append(len(counts))
                by_label_majority_fraction[label_name].append(majority_count / len(codes))
                by_label_entropy[label_name].append(entropy_from_counts(counts))

    print(f"checkpoint: {args.checkpoint.resolve()}")
    print("per_label_trial_summary:")
    for label_name in sorted(by_label_majority):
        total_trials = sum(by_label_majority[label_name].values())
        majority_mix = " ".join(
            f"{code}:{by_label_majority[label_name][code] / total_trials:.3f}"
            for code in range(cfg.num_codes)
        )
        unique_counts = by_label_unique_count[label_name]
        majority_fractions = by_label_majority_fraction[label_name]
        entropies = by_label_entropy[label_name]
        print(f"  {label_name} trials={total_trials}")
        print(f"    majority_code_dist {majority_mix}")
        print(f"    mean_unique_codes {sum(unique_counts) / len(unique_counts):.3f}")
        print(f"    mean_majority_fraction {sum(majority_fractions) / len(majority_fractions):.3f}")
        print(f"    mean_entropy {sum(entropies) / len(entropies):.3f}")


if __name__ == "__main__":
    main()
