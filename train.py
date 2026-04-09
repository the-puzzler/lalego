from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from module.dataset import Egocentric10KWindowDataset, collate_video_windows
from module.models import ARPredictor, Transformer
from module.sigreg import SIGReg


def count_parameters(model: torch.nn.Module) -> int:
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


def main() -> None:
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

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
        num_workers=0,
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

    encoder_params = count_parameters(encoder)
    predictor_params = count_parameters(predictor)
    print(f"encoder params: {encoder_params:,}")
    print(f"predictor params: {predictor_params:,}")

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)

    encoder.train()
    predictor.train()

    progress = tqdm(loader, total=cfg.max_steps, desc="train", dynamic_ncols=True)
    for step, batch in enumerate(progress, start=1):
        if step > cfg.max_steps:
            progress.close()
            break

        tokens = batch["tokens"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        latents = encoder(tokens)
        predictions = predictor(latents[:, :-1], latents[:, :-1])
        targets = latents[:, 1:]

        mse_loss = F.mse_loss(predictions, targets)
        sigreg_loss = sigreg(latents.transpose(0, 1))
        loss = mse_loss + (cfg.sigreg_weight * sigreg_loss)

        loss.backward()
        optimizer.step()
        scheduler.step()

        progress.set_postfix(
            loss=f"{loss.item():.6f}",
            mse=f"{mse_loss.item():.6f}",
            sigreg=f"{sigreg_loss.item():.6f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            token_shape=str(tuple(tokens.shape)),
        )


if __name__ == "__main__":
    main()
