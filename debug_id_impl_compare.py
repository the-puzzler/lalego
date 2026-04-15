from __future__ import annotations

import argparse
import types
import urllib.request

import torch

import config as cfg
from module.models import InverseDynamicsTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare this repo's inverse-dynamics implementation against le-wm "
            "stage by stage on the same synthetic inputs."
        )
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument(
        "--input-kind",
        choices=("zeros", "gaussian"),
        default="zeros",
        help="Synthetic input family for the comparison.",
    )
    return parser.parse_args()


def load_remote_python_module(url: str, module_name: str) -> types.ModuleType:
    source = urllib.request.urlopen(url, timeout=30).read().decode("utf-8")
    module = types.ModuleType(module_name)
    module.__file__ = url
    exec(compile(source, url, "exec"), module.__dict__)
    return module


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "norm": float(x.norm(dim=-1).mean().item()),
        "absmax": float(x.abs().max().item()),
    }


def print_stats(prefix: str, x: torch.Tensor) -> None:
    stats = tensor_stats(x)
    print(
        {
            "stage": prefix,
            "shape": tuple(int(d) for d in x.shape),
            **{key: round(value, 8) for key, value in stats.items()},
        }
    )


def trace_local(model: InverseDynamicsTransformer, x: torch.Tensor) -> None:
    print("local_trace:")
    print_stats("input", x)
    y = model.dropout(x)
    print_stats("after_dropout", y)
    y = model.transformer.input_proj(y)
    print_stats("after_input_proj", y)
    for index, block in enumerate(model.transformer.layers):
        y = block(y)
        print_stats(f"after_block_{index}", y)
    y = model.transformer.norm(y)
    print_stats("after_final_norm", y)
    y = model.transformer.output_proj(y)
    print_stats("after_output_proj", y)
    y = y[:, :-1]
    print_stats("after_trim", y)


def trace_remote(module: types.ModuleType, config_module: types.ModuleType, x: torch.Tensor) -> None:
    model = module.InverseDynamicsTransformer(
        num_frames=x.size(1),
        input_dim=config_module.EMBED_DIM,
        hidden_dim=config_module.EMBED_DIM,
        output_dim=config_module.EMBED_DIM,
        depth=config_module.INVERSE_DYNAMICS_DEPTH,
        heads=config_module.INVERSE_DYNAMICS_HEADS,
        dim_head=config_module.INVERSE_DYNAMICS_DIM_HEAD,
        mlp_dim=config_module.INVERSE_DYNAMICS_MLP_DIM,
        dropout=config_module.INVERSE_DYNAMICS_DROPOUT,
        emb_dropout=config_module.INVERSE_DYNAMICS_EMB_DROPOUT,
    ).to(x.device)
    model.eval()

    print("lewm_trace:")
    print_stats("input", x)
    y = x + model.pos_embedding[:, : x.size(1)]
    print_stats("after_pos", y)
    y = model.dropout(y)
    print_stats("after_dropout", y)
    y = model.transformer.input_proj(y)
    print_stats("after_input_proj", y)
    for index, block in enumerate(model.transformer.layers):
        y = block(y)
        print_stats(f"after_block_{index}", y)
    y = model.transformer.norm(y)
    print_stats("after_final_norm", y)
    y = model.transformer.output_proj(y)
    print_stats("after_output_proj", y)


def build_input(*, kind: str, batch_size: int, sequence_length: int, dim: int, device: str) -> torch.Tensor:
    if kind == "zeros":
        return torch.zeros(batch_size, sequence_length, dim, device=device)
    if kind == "gaussian":
        return torch.randn(batch_size, sequence_length, dim, device=device)
    raise ValueError(f"Unsupported input kind: {kind}")


def main() -> int:
    args = parse_args()
    device = args.device

    remote_module = load_remote_python_module(
        "https://raw.githubusercontent.com/the-puzzler/le-wm/main/module.py",
        "lewm_module",
    )
    remote_cfg = load_remote_python_module(
        "https://raw.githubusercontent.com/the-puzzler/le-wm/main/config.py",
        "lewm_config",
    )

    local_model = InverseDynamicsTransformer(
        num_frames=args.sequence_length,
        input_dim=cfg.latent_dim,
        hidden_dim=cfg.id_hidden_dim,
        output_dim=cfg.latent_dim,
        depth=cfg.id_depth,
        heads=cfg.heads,
        dim_head=cfg.dim_head,
        mlp_dim=cfg.mlp_dim,
        dropout=cfg.dropout,
    ).to(device)
    local_model.eval()

    print(
        {
            "input_kind": args.input_kind,
            "local_dim": cfg.latent_dim,
            "local_id_hidden_dim": cfg.id_hidden_dim,
            "local_id_depth": cfg.id_depth,
            "local_id_mlp_dim": cfg.mlp_dim,
            "lewm_dim": remote_cfg.EMBED_DIM,
            "lewm_id_depth": remote_cfg.INVERSE_DYNAMICS_DEPTH,
            "lewm_id_mlp_dim": remote_cfg.INVERSE_DYNAMICS_MLP_DIM,
        }
    )

    local_input = build_input(
        kind=args.input_kind,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        dim=cfg.latent_dim,
        device=device,
    )
    remote_input = build_input(
        kind=args.input_kind,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        dim=remote_cfg.EMBED_DIM,
        device=device,
    )

    with torch.no_grad():
        trace_local(local_model, local_input)
        trace_remote(remote_module, remote_cfg, remote_input)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
