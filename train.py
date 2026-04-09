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
from module.dataset import Egocentric10KWindowDataset, collate_video_windows
from module.models import ARPredictor, Transformer
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
            writer.writerow(["step", "loss", "mse", "sigreg", "lr"])
        for row in rows[start_index:]:
            writer.writerow(
                [
                    row["step"],
                    f"{row['loss']:.8f}",
                    f"{row['mse']:.8f}",
                    f"{row['sigreg']:.8f}",
                    f"{row['lr']:.10f}",
                ]
            )
    return len(rows)


def save_metrics_plot(run_dir: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return

    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    series = [
        ("loss", "Total Loss"),
        ("mse", "MSE Loss"),
        ("sigreg", "SIGReg Loss"),
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
    encoder: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
) -> None:
    torch.save(
        {
            "step": step,
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        checkpoint_path,
    )


def main() -> None:
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    use_amp = bool(cfg.amp and device == "cuda")
    use_compile = bool(cfg.compile and hasattr(torch, "compile"))

    run_dir = create_run_dir()
    metrics_path = run_dir / "metrics.tsv"
    checkpoint_path = run_dir / "latest.pt"

    input_dim = 3 * cfg.image_size * cfg.image_size
    dataset = Egocentric10KWindowDataset(
        data_files=cfg.data_files,
        frames_per_window=cfg.frames_per_window,
        window_stride=cfg.window_stride,
        skip_n=cfg.skip_n,
        image_size=cfg.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_video_windows,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        pin_memory=device == "cuda",
    )

    encoder = Transformer(
        input_dim=input_dim,
        hidden_dim=cfg.encoder_hidden_dim,
        output_dim=cfg.latent_dim,
        depth=cfg.encoder_depth,
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
        encoder = torch.compile(encoder)
        predictor = torch.compile(predictor)

    encoder_params = count_parameters(encoder)
    predictor_params = count_parameters(predictor)
    print(f"run dir: {run_dir}")
    print(f"device: {device}")
    print(f"encoder params: {encoder_params:,}")
    print(f"predictor params: {predictor_params:,}")
    print(f"amp: {use_amp}")
    print(f"compile: {use_compile}")

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    encoder.train()
    predictor.train()

    metrics_history: list[dict[str, float]] = []
    flushed_metrics = 0

    progress = tqdm(loader, total=cfg.max_steps, desc="train", dynamic_ncols=True)
    for step, batch in enumerate(progress, start=1):
        if step > cfg.max_steps:
            progress.close()
            break

        tokens = batch["tokens"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
        )
        with autocast_context:
            latents = encoder(tokens)
            predictions = predictor(latents[:, :-1], latents[:, :-1])
            targets = latents[:, 1:]

            mse_loss = F.mse_loss(predictions, targets)
            sigreg_loss = sigreg(latents.transpose(0, 1))
            loss = mse_loss + (cfg.sigreg_weight * sigreg_loss)

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
                "lr": float(current_lr),
            }
        )

        if step % cfg.checkpoint_every_steps == 0:
            save_latest_checkpoint(
                checkpoint_path,
                step=step,
                encoder=encoder,
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
            lr=f"{current_lr:.2e}",
            token_shape=str(tuple(tokens.shape)),
        )

    save_latest_checkpoint(
        checkpoint_path,
        step=min(cfg.max_steps, len(metrics_history)),
        encoder=encoder,
        predictor=predictor,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    append_metrics_rows(metrics_path, metrics_history, flushed_metrics)
    save_metrics_plot(run_dir, metrics_history)


if __name__ == "__main__":
    main()
