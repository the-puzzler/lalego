from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from module.dataset import build_audio_dataset, collate_audio_windows
from module.models import build_frame_encoder_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a trained chunk encoder and train a lightweight decoder from chunk "
            "latents back to waveform. This is a probe for how much waveform detail "
            "the state embedding retains."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Training checkpoint containing frame_encoder weights. Defaults to latest run.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=64)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=128,
        help="Decoder channel width.",
    )
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--decoder-heads", type=int, default=4)
    return parser.parse_args()


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        normalized[key] = value
    return normalized


class ChunkWaveformDecoder(nn.Module):
    """Direct latent-to-waveform decoder probe using only learned upsampling convs."""

    def __init__(
        self,
        *,
        latent_dim: int,
        patch_size: int,
        output_samples: int,
        hidden_channels: int = 64,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
    ) -> None:
        super().__init__()
        self.output_samples = int(output_samples)
        self.base_steps = max(1, self.output_samples // (10 * 10 * 10 * 6 * 2))
        if self.output_samples % self.base_steps != 0:
            raise ValueError("output_samples must be divisible by base_steps")
        total_upsample = self.output_samples // self.base_steps
        self.deconv_factors = self._factorize_upsample(total_upsample)
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels * self.base_steps),
            nn.GELU(),
        )

        channels = [hidden_channels]
        current = hidden_channels
        for _ in range(len(self.deconv_factors) - 1):
            current = max(current // 2, 32)
            channels.append(current)
        channels.append(1)
        layers: list[nn.Module] = []
        in_channels = channels[0]
        for stage_index, factor in enumerate(self.deconv_factors):
            is_last_upsample = stage_index == len(self.deconv_factors) - 1
            out_channels = (
                1 if is_last_upsample else channels[stage_index + 1]
            )
            layers.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=factor,
                    stride=factor,
                )
            )
            if not is_last_upsample:
                layers.append(nn.GELU())
            in_channels = out_channels
        self.deconv = nn.Sequential(*layers)

    @staticmethod
    def _factorize_upsample(total_upsample: int) -> list[int]:
        remaining = int(total_upsample)
        factors: list[int] = []
        preferred = (10, 8, 6, 5, 4, 3, 2)
        while remaining > 1:
            chosen = None
            for factor in preferred:
                if remaining % factor == 0 and remaining != factor:
                    chosen = factor
                    break
            if chosen is None:
                factors.append(remaining)
                break
            factors.append(chosen)
            remaining //= chosen
        return factors

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: (N, D)
        returns: (N, 1, S)
        """
        base = self.latent_proj(latents).reshape(latents.shape[0], -1, self.base_steps)
        waveform = self.deconv(base)
        if waveform.shape[-1] != self.output_samples:
            raise ValueError(
                f"Decoder produced {waveform.shape[-1]} samples, expected {self.output_samples}"
            )
        return waveform


def build_loader(
    *,
    splits: tuple[str, ...],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = build_audio_dataset(
        dataset_backend=cfg.dataset_backend,
        dataset_root=cfg.dataset_root,
        dataset_cache_root=cfg.dataset_cache_root,
        splits=splits,
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
        shuffle=shuffle,
        collate_fn=collate_audio_windows,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=cfg.prefetch_factor if num_workers > 0 else None,
    )


def build_encoder(device: str):
    return build_frame_encoder_from_config(cfg).to(device)


def find_latest_checkpoint() -> Path:
    runs = sorted((Path(__file__).resolve().parent / cfg.runs_dir).glob("*"))
    if not runs:
        raise FileNotFoundError("No runs found under runs/")
    return runs[-1] / "latest.pt"


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = F.mse_loss(prediction, target)
    return mse, {"mse": float(mse.item())}


def evaluate(
    *,
    loader: DataLoader,
    device: str,
    encoder: torch.nn.Module,
    decoder: ChunkWaveformDecoder,
    max_batches: int,
    use_amp: bool,
) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    totals = {"loss": 0.0, "mse": 0.0}
    batches = 0

    with torch.no_grad():
        for batch in loader:
            if max_batches > 0 and batches >= max_batches:
                break
            signal_values = batch["signal_values"].to(device, non_blocking=True)
            flat_chunks = signal_values.reshape(-1, signal_values.shape[-2], signal_values.shape[-1])
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            )
            with autocast_context:
                latents = encoder(signal_values).reshape(-1, cfg.latent_dim)
                predictions = decoder(latents)
                loss, parts = reconstruction_loss(
                    predictions,
                    flat_chunks,
                )

            totals["loss"] += float(loss.item())
            totals["mse"] += parts["mse"]
            batches += 1

    if batches == 0:
        raise RuntimeError("No validation batches were available for decoder probe.")
    return {key: value / batches for key, value in totals.items()}


def main() -> int:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    use_amp = bool(cfg.amp and device == "cuda")

    checkpoint_path = args.checkpoint or find_latest_checkpoint()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    train_loader = build_loader(
        splits=cfg.dataset_train_splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = build_loader(
        splits=cfg.dataset_val_splits,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
    )

    first_batch = next(iter(train_loader))
    output_samples = int(first_batch["signal_values"].shape[-1])

    encoder = build_encoder(device)
    encoder.load_state_dict(normalize_state_dict(checkpoint["frame_encoder"]))
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    decoder = ChunkWaveformDecoder(
        latent_dim=cfg.latent_dim,
        patch_size=cfg.audio_patch_samples,
        output_samples=output_samples,
        hidden_channels=args.hidden_channels,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"checkpoint: {checkpoint_path}")
    print(f"probe_output_samples: {output_samples}")
    print(f"probe_base_steps: {decoder.base_steps}")
    print(f"probe_deconv_factors: {decoder.deconv_factors}")
    print(f"decoder_params: {sum(p.numel() for p in decoder.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        progress = tqdm(train_loader, desc=f"decoder epoch {epoch}", dynamic_ncols=True)
        running = {"loss": 0.0, "mse": 0.0}
        batches = 0

        for batch in progress:
            if args.max_train_batches > 0 and batches >= args.max_train_batches:
                break

            signal_values = batch["signal_values"].to(device, non_blocking=True)
            flat_chunks = signal_values.reshape(-1, signal_values.shape[-2], signal_values.shape[-1])

            with torch.no_grad():
                latents = encoder(signal_values).reshape(-1, cfg.latent_dim)

            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context:
                predictions = decoder(latents)
                loss, parts = reconstruction_loss(
                    predictions,
                    flat_chunks,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running["loss"] += float(loss.item())
            running["mse"] += parts["mse"]
            batches += 1

            progress.set_postfix(
                loss=f"{running['loss'] / batches:.4f}",
                mse=f"{running['mse'] / batches:.4f}",
            )

        progress.close()
        train_metrics = {key: value / max(batches, 1) for key, value in running.items()}
        val_metrics = evaluate(
            loader=val_loader,
            device=device,
            encoder=encoder,
            decoder=decoder,
            max_batches=args.max_val_batches,
            use_amp=use_amp,
        )
        print(
            {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "train_mse": round(train_metrics["mse"], 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_mse": round(val_metrics["mse"], 6),
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
