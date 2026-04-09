from __future__ import annotations

import io
import sys
from pathlib import Path

import av
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from module.dataset import load_egocentric10k_stream


OUTPUT_DIR = ROOT / "outputs" / "frame_inspect"
NUM_FRAMES = 4
WINDOW_FRAMES = 32


def frame_to_pil(frame: torch.Tensor) -> Image.Image:
    array = frame.permute(1, 2, 0).contiguous().cpu().numpy()
    return Image.fromarray(array)


def make_horizontal_grid(frames: torch.Tensor) -> Image.Image:
    images = [frame_to_pil(frame) for frame in frames]
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height))

    x_offset = 0
    for image in images:
        canvas.paste(image, (x_offset, 0))
        x_offset += image.width
    return canvas


def make_tiled_grid(frames: torch.Tensor, *, columns: int) -> Image.Image:
    images = [frame_to_pil(frame) for frame in frames]
    rows = (len(images) + columns - 1) // columns
    tile_width = images[0].width
    tile_height = images[0].height
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height))

    for index, image in enumerate(images):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        canvas.paste(image, (x, y))
    return canvas


def decode_first_n_frames(video_bytes: bytes, n: int) -> torch.Tensor:
    frames = []
    with av.open(io.BytesIO(video_bytes), mode="r", format="mp4") as container:
        for frame in container.decode(video=0):
            frame_array = frame.to_ndarray(format="rgb24")
            frames.append(torch.from_numpy(frame_array).permute(2, 0, 1))
            if len(frames) >= n:
                break

    if len(frames) < n:
        raise RuntimeError(f"expected at least {n} frames, got {len(frames)}")

    return torch.stack(frames, dim=0)


def resize_frames(frames: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        frames.float(),
        size=(cfg.image_size, cfg.image_size),
        mode="bilinear",
        align_corners=False,
    ).round().clamp(0, 255).to(torch.uint8)


def main() -> None:
    stream = load_egocentric10k_stream(data_files=cfg.data_files)
    sample = next(iter(stream))
    metadata = sample["json"]

    first_four = decode_first_n_frames(sample["mp4"], NUM_FRAMES)
    resized_four = resize_frames(first_four)

    first_window = decode_first_n_frames(sample["mp4"], WINDOW_FRAMES)
    resized_window = resize_frames(first_window)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_path = OUTPUT_DIR / "original_4_frames.png"
    resized_path = OUTPUT_DIR / "resized_4_frames.png"
    window_path = OUTPUT_DIR / "resized_32_frame_window.png"

    make_horizontal_grid(first_four).save(original_path)
    make_horizontal_grid(resized_four).save(resized_path)
    make_tiled_grid(resized_window, columns=4).save(window_path)

    print(f"saved {original_path}")
    print(f"saved {resized_path}")
    print(f"saved {window_path}")
    print(f"original_shape={tuple(first_four.shape)}")
    print(f"resized_shape={tuple(resized_four.shape)}")
    print(f"window_shape={tuple(resized_window.shape)}")
    print(f"metadata={metadata}")


if __name__ == "__main__":
    main()
