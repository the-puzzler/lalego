from __future__ import annotations

import csv
import math
import shutil
from datetime import datetime
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def lr_multiplier(step: int, *, max_steps: int, warmup_steps: int) -> float:
    if max_steps <= 0:
        return 1.0

    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)

    if max_steps <= warmup_steps:
        return 1.0

    progress = (step - warmup_steps) / float(max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def create_run_dir(*, root: Path, runs_dir: str, config_path: Path) -> Path:
    runs_path = root / runs_dir
    runs_path.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_path / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config.py")
    return run_dir


def append_metrics_rows(metrics_path: Path, rows: list[dict[str, float]], start_index: int) -> int:
    if start_index >= len(rows):
        return start_index

    file_exists = metrics_path.exists()
    fieldnames = [
        "step",
        "loss",
        "mse",
        "sigreg",
        "codebook",
        "commitment",
        "lr",
        "val_loss",
        "val_mse",
        "val_sigreg",
        "val_codebook",
        "val_commitment",
    ]
    with metrics_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if not file_exists:
            writer.writerow(fieldnames)
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
                    f"{row.get('val_loss', float('nan')):.8f}",
                    f"{row.get('val_mse', float('nan')):.8f}",
                    f"{row.get('val_sigreg', float('nan')):.8f}",
                    f"{row.get('val_codebook', float('nan')):.8f}",
                    f"{row.get('val_commitment', float('nan')):.8f}",
                ]
            )
    return len(rows)


def save_metrics_plot(run_dir: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return

    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    series = [
        ("loss", "Total Loss"),
        ("mse", "MSE Loss"),
        ("sigreg", "SIGReg Loss"),
        ("codebook", "Codebook Loss"),
        ("commitment", "Commitment Loss"),
        ("lr", "Learning Rate"),
    ]
    val_keys = {
        "loss": "val_loss",
        "mse": "val_mse",
        "sigreg": "val_sigreg",
        "codebook": "val_codebook",
        "commitment": "val_commitment",
    }

    for axis, (key, title) in zip(axes.flat, series):
        values = [row[key] for row in rows]
        axis.plot(steps, values, linewidth=2, label="train")
        if key in val_keys:
            val_series = [row.get(val_keys[key], float("nan")) for row in rows]
            if any(not math.isnan(value) for value in val_series):
                axis.plot(steps, val_series, linewidth=2, linestyle="--", label="val")
                axis.legend()
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
