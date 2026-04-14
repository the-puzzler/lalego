from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402
from module.dataset import EEGTokenDataset, collate_eeg_windows  # noqa: E402
from module.models import EEGPatchEncoder, InverseDynamicsTransformer, VectorQuantizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/test a linear probe on per-trial action-code histograms."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "runs" / sorted((ROOT / "runs").iterdir())[-1] / "latest.pt",
        help="Path to checkpoint.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Max iterations for logistic regression.",
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


def build_dataset(subject_ids: tuple[str, ...]) -> EEGTokenDataset:
    return EEGTokenDataset(
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


def extract_histograms(
    *,
    dataset: EEGTokenDataset,
    frame_encoder: EEGPatchEncoder,
    inverse_dynamics: InverseDynamicsTransformer,
    codebook: VectorQuantizer,
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_eeg_windows,
        num_workers=0,
    )

    features = []
    labels = []
    keys = []

    with torch.no_grad():
        for batch in loader:
            eeg_values = batch["eeg_values"].to(device)
            latents = frame_encoder(eeg_values)
            action_logits = inverse_dynamics(latents)
            indices = codebook(action_logits)["indices"].cpu()

            for sample_indices, label_id, key in zip(indices, batch["label_id"], batch["key"]):
                counts = torch.bincount(sample_indices, minlength=cfg.num_codes).float()
                histogram = counts / counts.sum().clamp_min(1.0)
                features.append(histogram.numpy())
                labels.append(int(label_id))
                keys.append(key)

    return np.stack(features), np.array(labels), keys


def main() -> None:
    args = parse_args()
    if cfg.action_source != "inferred":
        raise ValueError("This probe is intended for inferred-action checkpoints only.")

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    train_dataset = build_dataset(cfg.train_subject_ids)
    val_dataset = build_dataset(cfg.val_subject_ids)

    sample = train_dataset.examples[0]
    num_tokens = int(sample["eeg_values"].shape[0])
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

    x_train, y_train, _ = extract_histograms(
        dataset=train_dataset,
        frame_encoder=frame_encoder,
        inverse_dynamics=inverse_dynamics,
        codebook=codebook,
        device=device,
    )
    x_val, y_val, _ = extract_histograms(
        dataset=val_dataset,
        frame_encoder=frame_encoder,
        inverse_dynamics=inverse_dynamics,
        codebook=codebook,
        device=device,
    )

    probe = LogisticRegression(
        max_iter=args.max_iter,
        solver="lbfgs",
    )
    probe.fit(x_train, y_train)
    train_pred = probe.predict(x_train)
    val_pred = probe.predict(x_val)

    print(f"checkpoint: {args.checkpoint.resolve()}")
    print(f"train_trials: {len(y_train)}")
    print(f"val_trials: {len(y_val)}")
    print(f"train_accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"val_accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print("val_confusion_matrix:")
    print(confusion_matrix(y_val, val_pred))
    print("predicted_class_counts_val:")
    print(Counter(val_pred.tolist()))


if __name__ == "__main__":
    main()
