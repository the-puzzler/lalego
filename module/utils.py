from __future__ import annotations

import csv
import math
import multiprocessing as mp
import shutil
from datetime import datetime
from pathlib import Path

import torch
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
        "adj_mse",
        "adj_cos",
        "rand_mse",
        "rand_cos",
        "delta_norm",
        "latent_std",
        "seq_var",
        *[f"code_count_{index}" for index in range(8)],
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
                    f"{row.get('adj_mse', float('nan')):.8f}",
                    f"{row.get('adj_cos', float('nan')):.8f}",
                    f"{row.get('rand_mse', float('nan')):.8f}",
                    f"{row.get('rand_cos', float('nan')):.8f}",
                    f"{row.get('delta_norm', float('nan')):.8f}",
                    f"{row.get('latent_std', float('nan')):.8f}",
                    f"{row.get('seq_var', float('nan')):.8f}",
                    *[f"{row.get(f'code_count_{index}', float('nan')):.0f}" for index in range(8)],
                    f"{row.get('val_loss', float('nan')):.8f}",
                    f"{row.get('val_mse', float('nan')):.8f}",
                    f"{row.get('val_sigreg', float('nan')):.8f}",
                    f"{row.get('val_codebook', float('nan')):.8f}",
                    f"{row.get('val_commitment', float('nan')):.8f}",
                ]
            )
    return len(rows)


def load_metrics_rows(metrics_path: Path) -> list[dict[str, float]]:
    if not metrics_path.exists():
        return []
    rows: list[dict[str, float]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def save_metrics_plot(run_dir: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return

    from matplotlib import pyplot as plt

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

    latest = rows[-1]
    code_axis = axes[3, 0]
    code_counts = [latest.get(f"code_count_{index}", 0.0) for index in range(8)]
    code_axis.bar(range(len(code_counts)), code_counts, width=0.8)
    code_axis.set_title("Latest Code Usage")
    code_axis.set_xlabel("Code")
    code_axis.set_ylabel("Count")
    code_axis.set_xticks(range(len(code_counts)))
    code_axis.grid(True, axis="y", alpha=0.3)

    collapse_axis = axes[3, 1]
    collapse_axis.axis("off")
    collapse_text = "\n".join(
        [
            f"adj_mse: {latest.get('adj_mse', float('nan')):.4f}",
            f"adj_cos: {latest.get('adj_cos', float('nan')):.4f}",
            f"rand_mse: {latest.get('rand_mse', float('nan')):.4f}",
            f"rand_cos: {latest.get('rand_cos', float('nan')):.4f}",
            f"delta_norm: {latest.get('delta_norm', float('nan')):.4f}",
            f"latent_std: {latest.get('latent_std', float('nan')):.4f}",
            f"seq_var: {latest.get('seq_var', float('nan')):.4f}",
        ]
    )
    collapse_axis.text(
        0.02,
        0.98,
        collapse_text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        transform=collapse_axis.transAxes,
    )
    collapse_axis.set_title("Latest Collapse Stats")

    fig.tight_layout()
    fig.savefig(run_dir / "metrics.png", dpi=160)
    plt.close(fig)


def _plot_worker_main(queue: mp.Queue) -> None:
    latest: tuple[Path, Path] | None = None
    while True:
        item = queue.get()
        if item is None:
            if latest is not None:
                run_dir, metrics_path = latest
                rows = load_metrics_rows(metrics_path)
                save_metrics_plot(run_dir, rows)
            return
        latest = item
        while True:
            try:
                item = queue.get_nowait()
            except Exception:
                break
            if item is None:
                run_dir, metrics_path = latest
                rows = load_metrics_rows(metrics_path)
                save_metrics_plot(run_dir, rows)
                return
            latest = item
        run_dir, metrics_path = latest
        rows = load_metrics_rows(metrics_path)
        save_metrics_plot(run_dir, rows)


class AsyncMetricsPlotter:
    """Render metrics plots in a background process so training does not block."""

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._queue: mp.Queue = self._ctx.Queue(maxsize=1)
        self._process = self._ctx.Process(
            target=_plot_worker_main,
            args=(self._queue,),
            daemon=True,
        )
        self._process.start()

    def submit(self, run_dir: Path, metrics_path: Path) -> None:
        payload = (run_dir, metrics_path)
        try:
            self._queue.put_nowait(payload)
            return
        except Exception:
            pass
        try:
            _ = self._queue.get_nowait()
        except Exception:
            pass
        try:
            self._queue.put_nowait(payload)
        except Exception:
            pass

    def close(self) -> None:
        if not self._process.is_alive():
            return
        try:
            self._queue.put(None, timeout=1.0)
        except Exception:
            pass
        self._process.join(timeout=30.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)


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
