from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import av
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info


VIT_IMAGE_MEAN = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
VIT_IMAGE_STD = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)


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
    raw_min_frames = (
        1 + (min_frames - 1) * skip_n if min_frames is not None else raw_frames_per_window
    )
    return WindowSpec(
        frames_per_window=raw_frames_per_window,
        stride=window_stride,
        min_frames=raw_min_frames,
    )


def resolve_local_video_files(*, data_root: str | Path, data_files: list[str]) -> list[str]:
    root = Path(data_root).expanduser()
    matched: list[Path] = []
    for pattern in data_files:
        matched.extend(path for path in root.glob(pattern) if path.is_file())

    unique_paths = sorted({path.resolve() for path in matched})
    if not unique_paths:
        raise ValueError(f"no local video files matched data_root={root} data_files={data_files}")
    return [str(path) for path in unique_paths]


def iter_video_windows_from_path(
    video_path: str | Path,
    spec: WindowSpec,
    *,
    max_windows: int | None = None,
    max_decode_frames: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, float]]:
    with av.open(str(video_path), mode="r") as container:
        stream = container.streams.video[0]
        try:
            stream.thread_type = "AUTO"
        except Exception:
            pass
        average_rate = float(stream.average_rate) if stream.average_rate is not None else 0.0
        frame_buffer: deque[torch.Tensor] = deque(maxlen=spec.frames_per_window)
        index_buffer: deque[int] = deque(maxlen=spec.frames_per_window)
        yielded = 0

        for frame_index, frame in enumerate(container.decode(video=0)):
            if max_decode_frames is not None and frame_index >= max_decode_frames:
                break

            frame_tensor = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
            frame_buffer.append(frame_tensor)
            index_buffer.append(frame_index)

            if len(frame_buffer) < spec.frames_per_window:
                continue

            if index_buffer[0] % spec.stride != 0:
                continue

            yield (
                torch.stack(tuple(frame_buffer), dim=0),
                torch.tensor(index_buffer, dtype=torch.long),
                average_rate,
            )
            yielded += 1

            if max_windows is not None and yielded >= max_windows:
                break


def preprocess_frame_window(
    frames: torch.Tensor,
    *,
    skip_n: int,
    image_size: int,
) -> torch.Tensor:
    frames = frames[::skip_n]
    frames = frames.float() / 255.0
    frames = F.interpolate(
        frames,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return (frames - VIT_IMAGE_MEAN) / VIT_IMAGE_STD


class EpicKitchensWindowDataset(IterableDataset):
    def __init__(
        self,
        *,
        data_root: str | Path,
        data_files: list[str],
        frames_per_window: int,
        window_stride: int,
        skip_n: int,
        image_size: int,
        max_windows_per_video: int | None = None,
        max_decode_frames: int | None = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.data_files = data_files
        self.spec = make_raw_window_spec(
            frames_per_window=frames_per_window,
            window_stride=window_stride,
            skip_n=skip_n,
            min_frames=frames_per_window,
        )
        self.skip_n = skip_n
        self.image_size = image_size
        self.max_windows_per_video = max_windows_per_video
        self.max_decode_frames = max_decode_frames

    def __iter__(self) -> Iterator[dict[str, Any]]:
        video_paths = resolve_local_video_files(
            data_root=self.data_root,
            data_files=self.data_files,
        )

        worker_info = get_worker_info()
        if worker_info is not None:
            video_paths = video_paths[worker_info.id :: worker_info.num_workers]

        for video_path in video_paths:
            video_file = Path(video_path)
            metadata = {
                "participant_id": video_file.parent.name,
                "narration_id": video_file.stem,
                "video_path": str(video_file),
            }
            for window, frame_indices, decoded_fps in iter_video_windows_from_path(
                video_file,
                self.spec,
                max_windows=self.max_windows_per_video,
                max_decode_frames=self.max_decode_frames,
            ):
                yield {
                    "frame_indices": frame_indices,
                    "fps": decoded_fps,
                    "metadata": metadata,
                    "key": video_file.stem,
                    "url": str(video_file),
                    "pixel_values": preprocess_frame_window(
                        window,
                        skip_n=self.skip_n,
                        image_size=self.image_size,
                    ),
                }


def collate_video_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "frame_indices": torch.stack([item["frame_indices"] for item in batch], dim=0),
        "fps": torch.tensor([item["fps"] for item in batch], dtype=torch.float32),
        "metadata": [item["metadata"] for item in batch],
        "key": [item["key"] for item in batch],
        "url": [item["url"] for item in batch],
        "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
    }
