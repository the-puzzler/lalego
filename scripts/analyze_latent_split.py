from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402
from module.dataset import EEGTokenDataset, collate_eeg_windows  # noqa: E402
from module.models import EEGPatchEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare latent spread for train vs validation subject splits."
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


def build_loader(subject_ids: tuple[str, ...]) -> DataLoader:
    dataset = EEGTokenDataset(
        data_root=cfg.dataset_root,
        data_files=cfg.data_files,
        sample_rate=cfg.dataset_fps,
        num_channels=cfg.eeg_num_channels,
        subject_ids=subject_ids,
        train_session_suffixes=cfg.train_session_suffixes,
        patch_size=cfg.eeg_patch_size,
        bandpass_low_hz=cfg.eeg_bandpass_low_hz,
        bandpass_high_hz=cfg.eeg_bandpass_high_hz,
        epoch_start_seconds=cfg.eeg_epoch_start_seconds,
        epoch_end_seconds=cfg.eeg_epoch_end_seconds,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_eeg_windows,
        num_workers=0,
    )


def summarize_split(name: str, loader: DataLoader, encoder: EEGPatchEncoder, device: str) -> None:
    token_sum = None
    token_sq_sum = None
    token_count = 0
    trial_means = []

    with torch.no_grad():
        for batch in loader:
            eeg_values = batch["eeg_values"].to(device)
            latents = encoder(eeg_values).cpu()
            flat = latents.reshape(-1, latents.shape[-1])
            if token_sum is None:
                token_sum = flat.sum(dim=0)
                token_sq_sum = flat.square().sum(dim=0)
            else:
                token_sum += flat.sum(dim=0)
                token_sq_sum += flat.square().sum(dim=0)
            token_count += flat.shape[0]
            trial_means.append(latents.mean(dim=1))

    if token_sum is None or token_sq_sum is None or token_count == 0:
        print(f"{name}: no samples")
        return

    token_mean = token_sum / token_count
    token_var = (token_sq_sum / token_count) - token_mean.square()
    token_std = token_var.clamp_min(0.0).sqrt()

    trial_mean_tensor = torch.cat(trial_means, dim=0)
    centered = trial_mean_tensor - trial_mean_tensor.mean(dim=0, keepdim=True)
    trial_norms = centered.norm(dim=1)

    print(name)
    print(f"  token_count: {token_count}")
    print(f"  mean_abs_mean: {token_mean.abs().mean().item():.6f}")
    print(f"  mean_std: {token_std.mean().item():.6f}")
    print(f"  min_std: {token_std.min().item():.6f}")
    print(f"  max_std: {token_std.max().item():.6f}")
    print(f"  num_trials: {trial_mean_tensor.shape[0]}")
    print(f"  mean_centered_norm: {trial_norms.mean().item():.6f}")
    print(f"  min_centered_norm: {trial_norms.min().item():.6f}")
    print(f"  max_centered_norm: {trial_norms.max().item():.6f}")


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu")

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    encoder = EEGPatchEncoder(
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
    encoder.load_state_dict(normalize_state_dict(checkpoint["frame_encoder"]))
    encoder.eval()

    print(f"checkpoint: {args.checkpoint.resolve()}")
    summarize_split("train_split", build_loader(cfg.train_subject_ids), encoder, device)
    summarize_split("val_split", build_loader(cfg.val_subject_ids), encoder, device)


if __name__ == "__main__":
    main()
