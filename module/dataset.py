from __future__ import annotations

import io
import json
import tarfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import av
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info


EGOCENTRIC_10K_REPO = "builddotai/Egocentric-10K"
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
    raw_stride = window_stride
    raw_min_frames = (
        1 + (min_frames - 1) * skip_n if min_frames is not None else raw_frames_per_window
    )
    return WindowSpec(
        frames_per_window=raw_frames_per_window,
        stride=raw_stride,
        min_frames=raw_min_frames,
    )


def resolve_local_data_files(
    *,
    data_root: str | Path,
    data_files: list[str] | None,
) -> list[str]:
    root = Path(data_root).expanduser()
    patterns = data_files or ["**/*.tar"]
    matched: list[Path] = []

    for pattern in patterns:
        matched.extend(path for path in root.glob(pattern) if path.is_file())

    unique_paths = sorted({path.resolve() for path in matched})
    if not unique_paths:
        raise ValueError(f"no local tar files matched data_root={root} data_files={patterns}")
    return [str(path) for path in unique_paths]


def iter_local_tar_samples(*, shard_paths: list[str]) -> Iterator[dict[str, Any]]:
    for shard_path in shard_paths:
        pending: dict[str, dict[str, Any]] = {}
        with tarfile.open(shard_path, mode="r") as archive:
            for member in archive:
                if not member.isfile():
                    continue

                member_path = Path(member.name)
                suffix = member_path.suffix
                if suffix not in {".mp4", ".json"}:
                    continue

                file_obj = archive.extractfile(member)
                if file_obj is None:
                    continue

                key = member_path.stem
                entry = pending.setdefault(
                    key,
                    {
                        "__key__": key,
                        "__url__": shard_path,
                    },
                )
                payload = file_obj.read()
                if suffix == ".mp4":
                    entry["mp4"] = payload
                else:
                    entry["json"] = json.loads(payload.decode("utf-8"))

                if "mp4" in entry and "json" in entry:
                    yield pending.pop(key)


def iter_video_windows_from_bytes(
    video_bytes: bytes,
    spec: WindowSpec,
    *,
    max_windows: int | None = None,
    max_decode_frames: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, float]]:
    with av.open(io.BytesIO(video_bytes), mode="r", format="mp4") as container:
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
    skip_n: int = 1,
    image_size: int,
) -> torch.Tensor:
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
    return (frames - VIT_IMAGE_MEAN) / VIT_IMAGE_STD


class Egocentric10KWindowDataset(IterableDataset):
    def __init__(
        self,
        *,
        data_root: str | Path,
        data_files: list[str] | None = None,
        frames_per_window: int = 32,
        window_stride: int = 1,
        min_frames: int | None = None,
        max_windows_per_video: int | None = None,
        max_decode_frames: int | None = None,
        skip_n: int = 1,
        image_size: int | None = None,
        transform=None,
        include_frames: bool = True,
    ):
        super().__init__()
        self.data_root = data_root
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
        self.transform = transform
        self.include_frames = include_frames

    def __iter__(self) -> Iterator[dict[str, Any]]:
        shard_paths = resolve_local_data_files(
            data_root=self.data_root,
            data_files=self.data_files,
        )

        worker_info = get_worker_info()
        if worker_info is not None:
            shard_paths = shard_paths[worker_info.id :: worker_info.num_workers]
        if not shard_paths:
            return

        for sample in iter_local_tar_samples(shard_paths=shard_paths):
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
                    "frame_indices": frame_indices,
                    "fps": decoded_fps or metadata_fps,
                    "metadata": metadata,
                    "key": sample["__key__"],
                    "url": sample["__url__"],
                }

                if self.include_frames:
                    sample_out["frames"] = window

                if self.image_size is not None:
                    sample_out["pixel_values"] = preprocess_frame_window(
                        window,
                        skip_n=self.skip_n,
                        image_size=self.image_size,
                    )

                yield sample_out


def collate_video_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated = {
        "frame_indices": torch.stack([item["frame_indices"] for item in batch], dim=0),
        "fps": torch.tensor([item["fps"] for item in batch], dtype=torch.float32),
        "metadata": [item["metadata"] for item in batch],
        "key": [item["key"] for item in batch],
        "url": [item["url"] for item in batch],
    }
    if "frames" in batch[0]:
        collated["frames"] = torch.stack([item["frames"] for item in batch], dim=0)
    if "pixel_values" in batch[0]:
        collated["pixel_values"] = torch.stack([item["pixel_values"] for item in batch], dim=0)
    return collated
