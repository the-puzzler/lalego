from __future__ import annotations

import io
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import av
import torch
import torch.nn.functional as F
from datasets import Features, Value, load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import IterableDataset


EGOCENTRIC_10K_REPO = "builddotai/Egocentric-10K"
EGOCENTRIC_10K_FEATURES = Features(
    {
        "mp4": Value("binary"),
        "json": {
            "factory_id": Value("string"),
            "worker_id": Value("string"),
            "video_index": Value("int64"),
            "duration_sec": Value("float64"),
            "width": Value("int64"),
            "height": Value("int64"),
            "fps": Value("float64"),
            "size_bytes": Value("int64"),
            "codec": Value("string"),
        },
        "__key__": Value("string"),
        "__url__": Value("string"),
    }
)


@dataclass(frozen=True)
class WindowSpec:
    frames_per_window: int = 32
    stride: int = 1
    min_frames: int = 32

    def __post_init__(self):
        if self.frames_per_window <= 0:
            raise ValueError("frames_per_window must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.min_frames <= 0:
            raise ValueError("min_frames must be positive")


def make_raw_window_spec(
    *,
    frames_per_window: int,
    window_stride: int,
    skip_n: int,
    min_frames: int | None = None,
) -> WindowSpec:
    if skip_n <= 0:
        raise ValueError("skip_n must be positive")

    raw_frames_per_window = 1 + (frames_per_window - 1) * skip_n
    raw_stride = window_stride * skip_n
    raw_min_frames = (
        1 + (min_frames - 1) * skip_n if min_frames is not None else raw_frames_per_window
    )
    return WindowSpec(
        frames_per_window=raw_frames_per_window,
        stride=raw_stride,
        min_frames=raw_min_frames,
    )


def load_egocentric10k_stream(
    *,
    data_files: list[str] | None = None,
    token: str | None = None,
):
    """Return the streaming train split for Egocentric-10K."""
    dataset = load_dataset(
        EGOCENTRIC_10K_REPO,
        split="train",
        streaming=True,
        data_files=data_files,
        features=EGOCENTRIC_10K_FEATURES,
        token=token,
    )
    return dataset


def load_worker_intrinsics(
    *,
    factory_id: str,
    worker_id: str,
    token: str | None = None,
) -> dict[str, Any]:
    """Download and parse the camera intrinsics for one worker."""
    path = hf_hub_download(
        repo_id=EGOCENTRIC_10K_REPO,
        repo_type="dataset",
        filename=f"{factory_id}/workers/{worker_id}/intrinsics.json",
        token=token,
    )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def decode_video_bytes(video_bytes: bytes) -> tuple[torch.Tensor, float]:
    """
    Decode an MP4 payload into a tensor of shape (T, C, H, W) in uint8.
    Returns frames and the average decoded fps when available.
    """
    with av.open(io.BytesIO(video_bytes), mode="r", format="mp4") as container:
        stream = container.streams.video[0]
        average_rate = float(stream.average_rate) if stream.average_rate is not None else 0.0
        frames = []
        for frame in container.decode(video=0):
            frame_array = frame.to_ndarray(format="rgb24")
            frames.append(torch.from_numpy(frame_array).permute(2, 0, 1))

    if not frames:
        raise ValueError("decoded video contains no frames")

    return torch.stack(frames, dim=0), average_rate


def iter_video_windows_from_bytes(
    video_bytes: bytes,
    spec: WindowSpec,
    *,
    max_windows: int | None = None,
    max_decode_frames: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, float]]:
    """
    Decode video bytes lazily and yield consecutive windows without loading the whole clip.
    """
    with av.open(io.BytesIO(video_bytes), mode="r", format="mp4") as container:
        stream = container.streams.video[0]
        average_rate = float(stream.average_rate) if stream.average_rate is not None else 0.0
        frame_buffer: deque[torch.Tensor] = deque(maxlen=spec.frames_per_window)
        index_buffer: deque[int] = deque(maxlen=spec.frames_per_window)
        yielded = 0

        for frame_index, frame in enumerate(container.decode(video=0)):
            if max_decode_frames is not None and frame_index >= max_decode_frames:
                break

            frame_array = frame.to_ndarray(format="rgb24")
            frame_tensor = torch.from_numpy(frame_array).permute(2, 0, 1)
            frame_buffer.append(frame_tensor)
            index_buffer.append(frame_index)

            if len(frame_buffer) < spec.frames_per_window:
                continue

            start_index = index_buffer[0]
            if start_index % spec.stride != 0:
                continue

            yield (
                torch.stack(tuple(frame_buffer), dim=0),
                torch.tensor(index_buffer, dtype=torch.long),
                average_rate,
            )
            yielded += 1

            if max_windows is not None and yielded >= max_windows:
                break


def iter_consecutive_windows(
    frames: torch.Tensor,
    spec: WindowSpec,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield consecutive frame windows and their source indices."""
    num_frames = frames.shape[0]
    if num_frames < max(spec.frames_per_window, spec.min_frames):
        return

    last_start = num_frames - spec.frames_per_window
    for start in range(0, last_start + 1, spec.stride):
        end = start + spec.frames_per_window
        yield frames[start:end], torch.arange(start, end, dtype=torch.long)


def tokenize_frame_window(
    frames: torch.Tensor,
    *,
    skip_n: int = 1,
    image_size: int,
) -> torch.Tensor:
    """
    Convert a frame window into simple per-frame tokens.

    frames: (T, C, H, W) uint8 or float
    returns: (T, 3 * image_size * image_size) float32 in [0, 1]
    """
    if skip_n <= 0:
        raise ValueError("skip_n must be positive")

    frames = frames[::skip_n]
    frames = frames.float() / 255.0
    frames = F.interpolate(
        frames,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return frames.reshape(frames.shape[0], -1)


class Egocentric10KWindowDataset(IterableDataset):
    """
    Stream Egocentric-10K and emit fixed-length consecutive frame windows.

    Each yielded item is a dict with:
    - frames: (window, C, H, W) uint8 by default
    - frame_indices: (window,)
    - fps: decoded fps if available, otherwise metadata fps
    - metadata: original JSON metadata
    - key/url: WebDataset identifiers
    """

    def __init__(
        self,
        *,
        data_files: list[str] | None = None,
        frames_per_window: int = 32,
        window_stride: int = 1,
        min_frames: int | None = None,
        max_windows_per_video: int | None = None,
        max_decode_frames: int | None = None,
        skip_n: int = 1,
        image_size: int | None = None,
        token: str | None = None,
        transform=None,
    ):
        super().__init__()
        self.data_files = data_files
        self.spec = make_raw_window_spec(
            frames_per_window=frames_per_window,
            window_stride=window_stride,
            skip_n=skip_n,
            min_frames=min_frames or frames_per_window,
        )
        self.max_windows_per_video = max_windows_per_video
        self.max_decode_frames = max_decode_frames
        self.skip_n = skip_n
        self.image_size = image_size
        self.token = token
        self.transform = transform

    def _iter_source(self) -> Iterable[dict[str, Any]]:
        return load_egocentric10k_stream(data_files=self.data_files, token=self.token)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for sample in self._iter_source():
            metadata = sample["json"]
            metadata_fps = float(metadata.get("fps", 0.0))

            for window, frame_indices, decoded_fps in iter_video_windows_from_bytes(
                sample["mp4"],
                self.spec,
                max_windows=self.max_windows_per_video,
                max_decode_frames=self.max_decode_frames,
            ):
                if self.transform is not None:
                    window = self.transform(window)

                sample_out = {
                    "frames": window,
                    "frame_indices": frame_indices,
                    "fps": decoded_fps or metadata_fps,
                    "metadata": metadata,
                    "key": sample["__key__"],
                    "url": sample["__url__"],
                }

                if self.image_size is not None:
                    sample_out["tokens"] = tokenize_frame_window(
                        window,
                        skip_n=self.skip_n,
                        image_size=self.image_size,
                    )

                yield sample_out


def collate_video_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack frame windows into a training batch."""
    collated = {
        "frames": torch.stack([item["frames"] for item in batch], dim=0),
        "frame_indices": torch.stack([item["frame_indices"] for item in batch], dim=0),
        "fps": torch.tensor([item["fps"] for item in batch], dtype=torch.float32),
        "metadata": [item["metadata"] for item in batch],
        "key": [item["key"] for item in batch],
        "url": [item["url"] for item in batch],
    }
    if "tokens" in batch[0]:
        collated["tokens"] = torch.stack([item["tokens"] for item in batch], dim=0)
    return collated
