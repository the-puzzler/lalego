from __future__ import annotations

import csv
import math
import shutil
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext

import matplotlib
import torch
import torch.nn.functional as F
from torch import nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from module.dataset import (
    MarioWindowDataset,
    collate_video_windows,
)
from module.models import ARPredictor, InverseDynamicsTransformer, VectorQuantizer, ViTFrameEncoder
from module.sigreg import SIGReg

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def lr_multiplier(step: int) -> float:
    if cfg.max_steps <= 0:
        return 1.0

    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return float(step + 1) / float(cfg.warmup_steps)

    if cfg.max_steps <= cfg.warmup_steps:
        return 1.0

    progress = (step - cfg.warmup_steps) / float(cfg.max_steps - cfg.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def create_run_dir() -> Path:
    runs_dir = ROOT / cfg.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT / "config.py", run_dir / "config.py")
    return run_dir


def append_metrics_rows(metrics_path: Path, rows: list[dict[str, float]], start_index: int) -> int:
    if start_index >= len(rows):
        return start_index

    file_exists = metrics_path.exists()
    with metrics_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if not file_exists:
            writer.writerow(
                ["step", "loss", "mse", "sigreg", "codebook", "commitment", "lr"]
            )
        for row in rows[start_index:]:
            writer.writerow(
                [
                    row["step"],
                    f"{row['loss']:.8f}",
                    f"{row['mse']:.8f}",
                    f"{row['sigreg']:.8f}",
                    f"{row['codebook']:.8f}",
                    f"{row['commitment']:.8f}",
                    f"{row['lr']:.10f}",
                ]
            )
    return len(rows)


def save_metrics_plot(run_dir: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return

    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    series = [
        ("loss", "Total Loss"),
        ("mse", "MSE Loss"),
        ("sigreg", "SIGReg Loss"),
        ("codebook", "Codebook Loss"),
        ("commitment", "Commitment Loss"),
        ("lr", "Learning Rate"),
    ]

    for axis, (key, title) in zip(axes.flat, series):
        values = [row[key] for row in rows]
        axis.plot(steps, values, linewidth=2)
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(run_dir / "metrics.png", dpi=160)
    plt.close(fig)


def save_latest_checkpoint(
    checkpoint_path: Path,
    *,
    step: int,
    frame_encoder: nn.Module,
    codebook: nn.Module,
    inverse_dynamics: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
) -> None:
    torch.save(
        {
            "step": step,
            "frame_encoder": frame_encoder.state_dict(),
            "codebook": codebook.state_dict(),
            "inverse_dynamics": inverse_dynamics.state_dict(),
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        checkpoint_path,
    )


def describe_tensor_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def main() -> None:
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    use_amp = bool(cfg.amp and device == "cuda")
    use_compile = bool(cfg.compile and hasattr(torch, "compile"))

    run_dir = create_run_dir()
    metrics_path = run_dir / "metrics.tsv"
    checkpoint_path = run_dir / "latest.pt"

    dataset = MarioWindowDataset(
        data_root=cfg.dataset_root,
        data_files=cfg.data_files,
        frames_per_window=cfg.frames_per_window,
        window_stride=cfg.window_stride,
        skip_n=cfg.skip_n,
        image_size=cfg.image_size,
        fps=cfg.dataset_fps,
        max_windows_per_sequence=cfg.max_windows_per_sequence,
    )

    try:
        first_sample = next(iter(dataset))
    except StopIteration as exc:
        raise RuntimeError(
            "Mario dataset produced no training windows. "
            "Check dataset_root/data_files and your temporal sampling settings."
        ) from exc

    print(
        "first sample:",
        {
            "key": first_sample["key"],
            "pixel_values": describe_tensor_shape(first_sample["pixel_values"]),
            "frame_indices": describe_tensor_shape(first_sample["frame_indices"]),
        },
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_video_windows,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        pin_memory=device == "cuda",
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )

    try:
        first_batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError(
            "DataLoader produced no batches. "
            "Check batch_size, num_workers, and dataset contents."
        ) from exc

    print(
        "first batch:",
        {
            "pixel_values": describe_tensor_shape(first_batch["pixel_values"]),
            "frame_indices": describe_tensor_shape(first_batch["frame_indices"]),
            "batch_size": len(first_batch["key"]),
        },
    )

    frame_encoder = ViTFrameEncoder(
        image_size=cfg.image_size,
        patch_size=cfg.frame_patch_size,
        hidden_dim=cfg.frame_hidden_dim,
        depth=cfg.frame_depth,
        heads=cfg.frame_heads,
        mlp_dim=cfg.frame_mlp_dim,
        output_dim=cfg.latent_dim,
        dropout=cfg.dropout,
    ).to(device)
    codebook = VectorQuantizer(
        num_codes=cfg.num_codes,
        code_dim=cfg.latent_dim,
        beta=cfg.codebook_beta,
    ).to(device)
    inverse_dynamics = InverseDynamicsTransformer(
        num_frames=cfg.frames_per_window,
        input_dim=cfg.latent_dim,
        hidden_dim=cfg.id_hidden_dim,
        output_dim=cfg.latent_dim,
        depth=cfg.id_depth,
        heads=cfg.heads,
        dim_head=cfg.dim_head,
        mlp_dim=cfg.mlp_dim,
        dropout=cfg.dropout,
    ).to(device)
    predictor = ARPredictor(
        num_frames=cfg.frames_per_window - 1,
        input_dim=cfg.latent_dim,
        hidden_dim=cfg.predictor_hidden_dim,
        output_dim=cfg.latent_dim,
        depth=cfg.predictor_depth,
        heads=cfg.heads,
        dim_head=cfg.dim_head,
        mlp_dim=cfg.mlp_dim,
        dropout=cfg.dropout,
    ).to(device)
    sigreg = SIGReg().to(device)
    if use_compile:
        frame_encoder = torch.compile(frame_encoder)
        codebook = torch.compile(codebook)
        inverse_dynamics = torch.compile(inverse_dynamics)
        predictor = torch.compile(predictor)

    frame_encoder_params = count_parameters(frame_encoder)
    codebook_params = count_parameters(codebook)
    inverse_dynamics_params = count_parameters(inverse_dynamics)
    predictor_params = count_parameters(predictor)
    print(f"run dir: {run_dir}")
    print(f"device: {device}")
    print(f"frame encoder params: {frame_encoder_params:,}")
    print(f"codebook params: {codebook_params:,}")
    print(f"inverse dynamics params: {inverse_dynamics_params:,}")
    print(f"predictor params: {predictor_params:,}")
    print(f"amp: {use_amp}")
    print(f"compile: {use_compile}")

    optimizer = torch.optim.AdamW(
        list(frame_encoder.parameters())
        + list(codebook.parameters())
        + list(inverse_dynamics.parameters())
        + list(predictor.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    frame_encoder.train()
    codebook.train()
    inverse_dynamics.train()
    predictor.train()

    metrics_history: list[dict[str, float]] = []
    flushed_metrics = 0

    print("starting training loop")
    progress = tqdm(loader, total=cfg.max_steps, desc="train", dynamic_ncols=True)
    for step, batch in enumerate(progress, start=1):
        if step > cfg.max_steps:
            progress.close()
            break

        pixel_values = batch["pixel_values"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
        )
        with autocast_context:
            latents = frame_encoder(pixel_values)
            action_logits = inverse_dynamics(latents)
            quantizer_outputs = codebook(action_logits)
            quantized_actions = quantizer_outputs["quantized"]
            predictions = predictor(latents[:, :-1], quantized_actions[:, :-1])
            targets = latents[:, 1:]

            mse_loss = F.mse_loss(predictions, targets)
            sigreg_loss = sigreg(latents.transpose(0, 1))
            codebook_loss = quantizer_outputs["codebook_loss"]
            commitment_loss = quantizer_outputs["commitment_loss"]
            loss = (
                mse_loss
                + (cfg.sigreg_weight * sigreg_loss)
                + (cfg.codebook_loss_weight * codebook_loss)
                + (cfg.commitment_loss_weight * commitment_loss)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        metrics_history.append(
            {
                "step": float(step),
                "loss": float(loss.item()),
                "mse": float(mse_loss.item()),
                "sigreg": float(sigreg_loss.item()),
                "codebook": float(codebook_loss.item()),
                "commitment": float(commitment_loss.item()),
                "lr": float(current_lr),
            }
        )

        if step % cfg.checkpoint_every_steps == 0:
            save_latest_checkpoint(
                checkpoint_path,
                step=step,
                frame_encoder=frame_encoder,
                codebook=codebook,
                inverse_dynamics=inverse_dynamics,
                predictor=predictor,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )

        if step % cfg.metrics_every_steps == 0:
            flushed_metrics = append_metrics_rows(metrics_path, metrics_history, flushed_metrics)
            save_metrics_plot(run_dir, metrics_history)

        progress.set_postfix(
            loss=f"{loss.item():.6f}",
            mse=f"{mse_loss.item():.6f}",
            sigreg=f"{sigreg_loss.item():.6f}",
            codebook=f"{codebook_loss.item():.6f}",
            commitment=f"{commitment_loss.item():.6f}",
            lr=f"{current_lr:.2e}",
            pixel_shape=str(tuple(pixel_values.shape)),
        )

    save_latest_checkpoint(
        checkpoint_path,
        step=min(cfg.max_steps, len(metrics_history)),
        frame_encoder=frame_encoder,
        codebook=codebook,
        inverse_dynamics=inverse_dynamics,
        predictor=predictor,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    append_metrics_rows(metrics_path, metrics_history, flushed_metrics)
    save_metrics_plot(run_dir, metrics_history)


if __name__ == "__main__":
    main()
