from __future__ import annotations

import argparse
import sys
from collections import Counter
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
        description="Analyze action-code usage for a trained checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "runs" / sorted((ROOT / "runs").iterdir())[-1] / "latest.pt",
        help="Path to the checkpoint to analyze.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=cfg.batch_size,
        help="Batch size for analysis.",
    )
    return parser.parse_args()


def build_models(device: str, num_tokens: int) -> tuple[EEGPatchEncoder, VectorQuantizer, InverseDynamicsTransformer]:
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
    return frame_encoder, codebook, inverse_dynamics


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
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
        batch_size=args.batch_size,
        collate_fn=collate_eeg_windows,
        num_workers=0,
    )

    first_batch = next(iter(loader))
    num_tokens = int(first_batch["eeg_values"].shape[1])
    frame_encoder, codebook, inverse_dynamics = build_models(device, num_tokens)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    frame_encoder.load_state_dict(normalize_state_dict(checkpoint["frame_encoder"]))
    codebook.load_state_dict(normalize_state_dict(checkpoint["codebook"]))
    inverse_dynamics.load_state_dict(normalize_state_dict(checkpoint["inverse_dynamics"]))

    frame_encoder.eval()
    codebook.eval()
    inverse_dynamics.eval()

    counts: Counter[int] = Counter()
    subject_counts: dict[str, Counter[int]] = {}
    total_actions = 0
    latent_sum = None
    latent_sq_sum = None
    latent_count = 0
    trial_means = []

    with torch.no_grad():
        for batch in loader:
            eeg_values = batch["eeg_values"].to(device)
            latents = frame_encoder(eeg_values)
            action_logits = inverse_dynamics(latents)
            indices = codebook(action_logits)["indices"].cpu()

            flat_latents = latents.detach().cpu().reshape(-1, latents.shape[-1])
            if latent_sum is None:
                latent_sum = flat_latents.sum(dim=0)
                latent_sq_sum = flat_latents.square().sum(dim=0)
            else:
                latent_sum += flat_latents.sum(dim=0)
                latent_sq_sum += flat_latents.square().sum(dim=0)
            latent_count += flat_latents.shape[0]
            trial_means.append(latents.detach().cpu().mean(dim=1))

            metadata = batch["metadata"]
            for sample_indices, sample_meta in zip(indices, metadata):
                subject_counter = subject_counts.setdefault(
                    sample_meta["subject_id"],
                    Counter(),
                )
                flat_indices = sample_indices.tolist()
                counts.update(flat_indices)
                subject_counter.update(flat_indices)
                total_actions += len(flat_indices)

    print(f"checkpoint: {checkpoint_path}")
    print(f"num_codes: {cfg.num_codes}")
    print(f"total_actions: {total_actions}")
    print("global_usage:")
    for code in range(cfg.num_codes):
        count = counts[code]
        fraction = (count / total_actions) if total_actions else 0.0
        print(f"  code {code}: count={count} fraction={fraction:.4f}")

    print("subject_usage:")
    for subject_id in sorted(subject_counts):
        total_subject = sum(subject_counts[subject_id].values())
        usage = " ".join(
            f"{code}:{subject_counts[subject_id][code] / total_subject:.3f}"
            for code in range(cfg.num_codes)
        )
        print(f"  {subject_id} total={total_subject} {usage}")

    if latent_sum is not None and latent_sq_sum is not None and latent_count > 0:
        latent_mean = latent_sum / latent_count
        latent_var = (latent_sq_sum / latent_count) - latent_mean.square()
        latent_std = latent_var.clamp_min(0.0).sqrt()
        print("latent_stats:")
        print(f"  token_count: {latent_count}")
        print(f"  mean_abs_mean: {latent_mean.abs().mean().item():.6f}")
        print(f"  mean_std: {latent_std.mean().item():.6f}")
        print(f"  min_std: {latent_std.min().item():.6f}")
        print(f"  max_std: {latent_std.max().item():.6f}")

        trial_mean_tensor = torch.cat(trial_means, dim=0)
        centered = trial_mean_tensor - trial_mean_tensor.mean(dim=0, keepdim=True)
        trial_norms = centered.norm(dim=1)
        print("trial_embedding_stats:")
        print(f"  num_trials: {trial_mean_tensor.shape[0]}")
        print(f"  mean_centered_norm: {trial_norms.mean().item():.6f}")
        print(f"  min_centered_norm: {trial_norms.min().item():.6f}")
        print(f"  max_centered_norm: {trial_norms.max().item():.6f}")


if __name__ == "__main__":
    main()
