from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from module.dataset import (
    build_audio_dataset,
    collate_audio_windows,
)
from module.models import (
    ARPredictor,
    InverseDynamicsTransformer,
    VectorQuantizer,
    build_frame_encoder_from_config,
)
from module.sigreg import SIGReg
from module.utils import (
    AsyncMetricsPlotter,
    append_metrics_rows,
    count_parameters,
    create_run_dir,
    describe_tensor_shape,
    lr_multiplier,
    save_latest_checkpoint,
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
 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if cfg.action_source == "inferred":
        action_logits = inverse_dynamics(latents)
        quantizer_outputs = codebook(action_logits)
        return (
            quantizer_outputs["quantized"],
            quantizer_outputs["codebook_loss"],
            quantizer_outputs["commitment_loss"],
            quantizer_outputs["indices"],
        )

    if cfg.action_source == "label":
        label_ids = batch["label_id"].to(device, non_blocking=True)
        if torch.any(label_ids < 0):
            raise ValueError("action_source='label' requires dataset labels, but label_id < 0 found")
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
    return quantized_actions, zero, zero, None


def summarize_latent_geometry(latents: torch.Tensor) -> dict[str, float]:
    adjacent_a = latents[:, :-1]
    adjacent_b = latents[:, 1:]
    random_perm = torch.randperm(latents.shape[0], device=latents.device)
    random_b = latents[random_perm, 1:]
    deltas = adjacent_b - adjacent_a
    return {
        "adj_mse": float(F.mse_loss(adjacent_a, adjacent_b).item()),
        "adj_cos": float(F.cosine_similarity(adjacent_a, adjacent_b, dim=-1).mean().item()),
        "rand_mse": float(F.mse_loss(adjacent_a, random_b).item()),
        "rand_cos": float(F.cosine_similarity(adjacent_a, random_b, dim=-1).mean().item()),
        "delta_norm": float(deltas.norm(dim=-1).mean().item()),
        "latent_std": float(latents.std(dim=(0, 1)).mean().item()),
        "seq_var": float(latents.var(dim=1).mean().item()),
    }


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
            signal_values = batch["signal_values"].to(device, non_blocking=True)
            latents = frame_encoder(signal_values)
            quantized_actions, codebook_loss, commitment_loss, _ = compute_action_outputs(
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
    plotter = AsyncMetricsPlotter()

    dataset = build_audio_dataset(
        dataset_backend=cfg.dataset_backend,
        dataset_root=cfg.dataset_root,
        dataset_cache_root=cfg.dataset_cache_root,
        splits=cfg.dataset_train_splits,
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
    val_dataset = build_audio_dataset(
        dataset_backend=cfg.dataset_backend,
        dataset_root=cfg.dataset_root,
        dataset_cache_root=cfg.dataset_cache_root,
        splits=cfg.dataset_val_splits,
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

    try:
        first_sample = next(iter(dataset))
    except StopIteration as exc:
        raise RuntimeError(
            "Audio dataset produced no training clips. "
            "Check dataset_root, MAESTRO extraction, and your clip settings."
        ) from exc

    print(
        "first sample:",
        {
            "key": first_sample["key"],
            "signal_values": describe_tensor_shape(first_sample["signal_values"]),
            "frame_indices": describe_tensor_shape(first_sample["frame_indices"]),
        },
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_audio_windows,
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

    num_tokens = int(first_batch["signal_values"].shape[1])
    print(
        "first batch:",
        {
            "signal_values": describe_tensor_shape(first_batch["signal_values"]),
            "frame_indices": describe_tensor_shape(first_batch["frame_indices"]),
            "batch_size": len(first_batch["key"]),
        },
    )
    print(f"train examples: {len(dataset.examples)}")
    print(f"val examples: {len(val_dataset.examples)}")

    frame_encoder = build_frame_encoder_from_config(cfg).to(device)
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
    optimizer.zero_grad(set_to_none=True)

    metrics_history: list[dict[str, float]] = []
    flushed_metrics = 0

    def finalize_optimizer_step() -> None:
        nonlocal step, flushed_metrics
        step += 1
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

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
                **latent_geometry_stats,
                **{f"code_count_{index}": float(value) for index, value in enumerate(code_count_values)},
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
            plotter.submit(run_dir, metrics_path)

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
            signal_shape=str(tuple(signal_values.shape)),
        )

    print("starting training loop")
    progress = tqdm(total=cfg.max_steps, desc="train", dynamic_ncols=True)
    step = 0
    micro_step = 0
    epoch = 0
    while step < cfg.max_steps:
        epoch += 1
        for batch in loader:
            if step >= cfg.max_steps:
                break

            micro_step += 1
            signal_values = batch["signal_values"].to(device, non_blocking=True)
            is_accum_boundary = micro_step % cfg.grad_accum_steps == 0

            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            )
            with autocast_context:
                latents = frame_encoder(signal_values)
                quantized_actions, codebook_loss, commitment_loss, action_indices = compute_action_outputs(
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
                latent_geometry_stats = summarize_latent_geometry(latents)
                if action_indices is not None:
                    code_count_values = (
                        torch.bincount(action_indices.reshape(-1).cpu(), minlength=cfg.num_codes).tolist()
                    )
                else:
                    code_count_values = [0] * cfg.num_codes

            scaled_loss = loss / cfg.grad_accum_steps
            scaler.scale(scaled_loss).backward()

            if not is_accum_boundary:
                continue

            finalize_optimizer_step()

        if micro_step % cfg.grad_accum_steps != 0:
            finalize_optimizer_step()

            if step >= cfg.max_steps:
                break

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
    plotter.submit(run_dir, metrics_path)
    plotter.close()


if __name__ == "__main__":
    main()
