from __future__ import annotations

import argparse
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
        description="Analyze alignment between learned action codes and class labels."
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

    label_code_counts: dict[str, Counter[int]] = defaultdict(Counter)
    label_trial_majority: dict[str, Counter[int]] = defaultdict(Counter)
    global_counts: Counter[int] = Counter()

    with torch.no_grad():
        for batch in loader:
            eeg_values = batch["eeg_values"].to(device)
            latents = frame_encoder(eeg_values)
            action_logits = inverse_dynamics(latents)
            indices = codebook(action_logits)["indices"].cpu()

            for sample_indices, label_name in zip(indices, batch["label_name"]):
                codes = sample_indices.tolist()
                label_code_counts[label_name].update(codes)
                global_counts.update(codes)
                majority_code = Counter(codes).most_common(1)[0][0]
                label_trial_majority[label_name][majority_code] += 1

    total_actions = sum(global_counts.values())
    print(f"checkpoint: {args.checkpoint.resolve()}")
    print(f"total_actions: {total_actions}")
    print("global_code_usage:")
    for code in range(cfg.num_codes):
        fraction = global_counts[code] / total_actions if total_actions else 0.0
        print(f"  code {code}: {fraction:.4f}")

    print("label_code_usage:")
    for label_name in sorted(label_code_counts):
        total = sum(label_code_counts[label_name].values())
        usage = " ".join(
            f"{code}:{label_code_counts[label_name][code] / total:.3f}"
            for code in range(cfg.num_codes)
        )
        print(f"  {label_name} total={total} {usage}")

    print("label_majority_code_per_trial:")
    for label_name in sorted(label_trial_majority):
        total = sum(label_trial_majority[label_name].values())
        usage = " ".join(
            f"{code}:{label_trial_majority[label_name][code] / total:.3f}"
            for code in range(cfg.num_codes)
        )
        print(f"  {label_name} total={total} {usage}")


if __name__ == "__main__":
    main()
