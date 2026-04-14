from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext

import matplotlib
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from module.dataset import (
    EEGTokenDataset,
    collate_eeg_windows,
)
from module.models import ARPredictor, EEGPatchEncoder, InverseDynamicsTransformer, VectorQuantizer
from module.sigreg import SIGReg
from module.utils import (
    append_metrics_rows,
    count_parameters,
    create_run_dir,
    describe_tensor_shape,
    lr_multiplier,
    save_latest_checkpoint,
    save_metrics_plot,
)

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent


def compute_action_outputs(
    *,
    latents: torch.Tensor,
    batch: dict[str, object],
    device: str,
    inverse_dynamics: nn.Module,
    codebook: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cfg.action_source == "inferred":
        action_logits = inverse_dynamics(latents)
        quantizer_outputs = codebook(action_logits)
        return (
            quantizer_outputs["quantized"],
            quantizer_outputs["codebook_loss"],
            quantizer_outputs["commitment_loss"],
        )

    if cfg.action_source == "label":
        label_ids = batch["label_id"].to(device, non_blocking=True)
        quantized_actions = inverse_dynamics(label_ids).unsqueeze(1).expand(
            -1,
            latents.shape[1] - 1,
            -1,
        )
    else:
        quantized_actions = torch.zeros(
            latents.shape[0],
            latents.shape[1] - 1,
            cfg.latent_dim,
            device=device,
            dtype=latents.dtype,
        )

    zero = torch.zeros((), device=device, dtype=latents.dtype)
    return quantized_actions, zero, zero


def evaluate(
    *,
    loader: DataLoader,
    device: str,
    frame_encoder: nn.Module,
    inverse_dynamics: nn.Module,
    codebook: nn.Module,
    predictor: nn.Module,
    sigreg: nn.Module,
) -> dict[str, float]:
    frame_encoder.eval()
    inverse_dynamics.eval()
    codebook.eval()
    predictor.eval()

    totals = {
        "loss": 0.0,
        "mse": 0.0,
        "sigreg": 0.0,
        "codebook": 0.0,
        "commitment": 0.0,
    }
    batches = 0

    with torch.no_grad():
        for batch in loader:
            eeg_values = batch["eeg_values"].to(device, non_blocking=True)
            latents = frame_encoder(eeg_values)
            quantized_actions, codebook_loss, commitment_loss = compute_action_outputs(
                latents=latents,
                batch=batch,
                device=device,
                inverse_dynamics=inverse_dynamics,
                codebook=codebook,
            )
            predictions = predictor(latents[:, :-1], quantized_actions)
            targets = latents[:, 1:]

            mse_loss = F.mse_loss(predictions, targets)
            sigreg_loss = sigreg(latents.transpose(0, 1))
            loss = (
                mse_loss
                + (cfg.sigreg_weight * sigreg_loss)
                + (cfg.codebook_loss_weight * codebook_loss)
                + (cfg.commitment_loss_weight * commitment_loss)
            )

            totals["loss"] += float(loss.item())
            totals["mse"] += float(mse_loss.item())
            totals["sigreg"] += float(sigreg_loss.item())
            totals["codebook"] += float(codebook_loss.item())
            totals["commitment"] += float(commitment_loss.item())
            batches += 1

    frame_encoder.train()
    inverse_dynamics.train()
    codebook.train()
    predictor.train()

    if batches == 0:
        return {key: float("nan") for key in totals}
    return {key: value / batches for key, value in totals.items()}


def main() -> None:
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    use_amp = bool(cfg.amp and device == "cuda")
    use_compile = bool(cfg.compile and hasattr(torch, "compile"))

    run_dir = create_run_dir(root=ROOT, runs_dir=cfg.runs_dir, config_path=ROOT / "config.py")
    metrics_path = run_dir / "metrics.tsv"
    checkpoint_path = run_dir / "latest.pt"

    dataset = EEGTokenDataset(
        data_root=cfg.dataset_root,
        data_files=cfg.data_files,
        sample_rate=cfg.dataset_fps,
        num_channels=cfg.eeg_num_channels,
        subject_ids=cfg.train_subject_ids,
        train_session_suffixes=cfg.train_session_suffixes,
        patch_size=cfg.eeg_patch_size,
        bandpass_low_hz=cfg.eeg_bandpass_low_hz,
        bandpass_high_hz=cfg.eeg_bandpass_high_hz,
        epoch_start_seconds=cfg.eeg_epoch_start_seconds,
        epoch_end_seconds=cfg.eeg_epoch_end_seconds,
    )
    val_dataset = EEGTokenDataset(
        data_root=cfg.dataset_root,
        data_files=cfg.data_files,
        sample_rate=cfg.dataset_fps,
        num_channels=cfg.eeg_num_channels,
        subject_ids=cfg.val_subject_ids,
        train_session_suffixes=cfg.train_session_suffixes,
        patch_size=cfg.eeg_patch_size,
        bandpass_low_hz=cfg.eeg_bandpass_low_hz,
        bandpass_high_hz=cfg.eeg_bandpass_high_hz,
        epoch_start_seconds=cfg.eeg_epoch_start_seconds,
        epoch_end_seconds=cfg.eeg_epoch_end_seconds,
    )

    try:
        first_sample = next(iter(dataset))
    except StopIteration as exc:
        raise RuntimeError(
            "EEG dataset produced no training trials. "
            "Check dataset_root/data_files and your temporal sampling settings."
        ) from exc

    print(
        "first sample:",
        {
            "key": first_sample["key"],
            "eeg_values": describe_tensor_shape(first_sample["eeg_values"]),
            "frame_indices": describe_tensor_shape(first_sample["frame_indices"]),
        },
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_eeg_windows,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        pin_memory=device == "cuda",
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_eeg_windows,
        num_workers=0,
        pin_memory=device == "cuda",
    )

    try:
        first_batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError(
            "DataLoader produced no batches. "
            "Check batch_size, num_workers, and dataset contents."
        ) from exc

    num_tokens = int(first_batch["eeg_values"].shape[1])
    print(
        "first batch:",
        {
            "eeg_values": describe_tensor_shape(first_batch["eeg_values"]),
            "frame_indices": describe_tensor_shape(first_batch["frame_indices"]),
            "batch_size": len(first_batch["key"]),
        },
    )
    print(f"train examples: {len(dataset.examples)}")
    print(f"val examples: {len(val_dataset.examples)}")

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
    if cfg.action_source == "inferred":
        codebook: nn.Module = VectorQuantizer(
            num_codes=cfg.num_codes,
            code_dim=cfg.latent_dim,
            beta=cfg.codebook_beta,
        ).to(device)
        inverse_dynamics: nn.Module = InverseDynamicsTransformer(
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
    elif cfg.action_source == "label":
        codebook = nn.Identity().to(device)
        inverse_dynamics = nn.Embedding(cfg.num_action_classes, cfg.latent_dim).to(device)
    elif cfg.action_source == "none":
        codebook = nn.Identity().to(device)
        inverse_dynamics = nn.Identity().to(device)
    else:
        raise ValueError(f"Unsupported action_source={cfg.action_source}")
    predictor = ARPredictor(
        num_frames=num_tokens - 1,
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
    print(f"action source: {cfg.action_source}")
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
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_multiplier(
            step,
            max_steps=cfg.max_steps,
            warmup_steps=cfg.warmup_steps,
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    frame_encoder.train()
    codebook.train()
    inverse_dynamics.train()
    predictor.train()

    metrics_history: list[dict[str, float]] = []
    flushed_metrics = 0

    print("starting training loop")
    progress = tqdm(total=cfg.max_steps, desc="train", dynamic_ncols=True)
    step = 0
    epoch = 0
    while step < cfg.max_steps:
        epoch += 1
        for batch in loader:
            if step >= cfg.max_steps:
                break

            step += 1
            eeg_values = batch["eeg_values"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            )
            with autocast_context:
                latents = frame_encoder(eeg_values)
                quantized_actions, codebook_loss, commitment_loss = compute_action_outputs(
                    latents=latents,
                    batch=batch,
                    device=device,
                    inverse_dynamics=inverse_dynamics,
                    codebook=codebook,
                )
                predictions = predictor(latents[:, :-1], quantized_actions)
                targets = latents[:, 1:]

                mse_loss = F.mse_loss(predictions, targets)
                sigreg_loss = sigreg(latents.transpose(0, 1))
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
                val_metrics = evaluate(
                    loader=val_loader,
                    device=device,
                    frame_encoder=frame_encoder,
                    inverse_dynamics=inverse_dynamics,
                    codebook=codebook,
                    predictor=predictor,
                    sigreg=sigreg,
                )
                metrics_history[-1].update(
                    {
                        "val_loss": val_metrics["loss"],
                        "val_mse": val_metrics["mse"],
                        "val_sigreg": val_metrics["sigreg"],
                        "val_codebook": val_metrics["codebook"],
                        "val_commitment": val_metrics["commitment"],
                    }
                )
                flushed_metrics = append_metrics_rows(metrics_path, metrics_history, flushed_metrics)
                save_metrics_plot(run_dir, metrics_history)

            progress.update(1)
            progress.set_postfix(
                epoch=epoch,
                loss=f"{loss.item():.6f}",
                val_loss=(
                    f"{metrics_history[-1]['val_loss']:.6f}"
                    if "val_loss" in metrics_history[-1]
                    else "na"
                ),
                mse=f"{mse_loss.item():.6f}",
                sigreg=f"{sigreg_loss.item():.6f}",
                codebook=f"{codebook_loss.item():.6f}",
                commitment=f"{commitment_loss.item():.6f}",
                lr=f"{current_lr:.2e}",
                eeg_shape=str(tuple(eeg_values.shape)),
            )

    progress.close()

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
