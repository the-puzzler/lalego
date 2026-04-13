from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.io import ImageReadMode, read_image


VIT_IMAGE_MEAN = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
VIT_IMAGE_STD = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)

MARIO_FRAME_PATTERN = re.compile(
    r"^(?P<player>.+)_f(?P<frame_id>\d+)_a(?P<action>\d+)_nt(?P<alive>[01])$"
)


@dataclass(frozen=True)
class WindowSpec:
    frames_per_window: int
    stride: int

    def __post_init__(self) -> None:
        if self.frames_per_window <= 0:
            raise ValueError("frames_per_window must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")


@dataclass(frozen=True)
class MarioFrameRecord:
    path: Path
    run_id: str
    player_id: str
    frame_id: int
    action_code: int
    alive_flag: int


def resolve_image_files(data_root: str | Path, data_files: list[str]) -> list[Path]:
    root = Path(data_root).expanduser()
    matched: list[Path] = []
    for pattern in data_files:
        matched.extend(path for path in root.glob(pattern) if path.is_file())

    image_paths = sorted({path.resolve() for path in matched})
    if not image_paths:
        raise ValueError(f"no Mario frames matched data_root={root} data_files={data_files}")
    return image_paths


def parse_mario_frame(path: Path) -> MarioFrameRecord:
    match = MARIO_FRAME_PATTERN.match(path.stem)
    if match is None:
        raise ValueError(
            "unexpected Mario frame name format. "
            f"Expected '<player>_f<frame>_a<action>_nt<0|1>.png', got {path.name}"
        )

    return MarioFrameRecord(
        path=path,
        run_id=path.parent.name,
        player_id=match.group("player"),
        frame_id=int(match.group("frame_id")),
        action_code=int(match.group("action")),
        alive_flag=int(match.group("alive")),
    )


def build_window_spec(
    *,
    frames_per_window: int,
    window_stride: int,
    skip_n: int,
) -> WindowSpec:
    if skip_n <= 0:
        raise ValueError("skip_n must be positive")
    raw_frames_per_window = 1 + (frames_per_window - 1) * skip_n
    return WindowSpec(frames_per_window=raw_frames_per_window, stride=window_stride)


def preprocess_frames(
    frames: list[torch.Tensor],
    *,
    skip_n: int,
    image_size: int,
) -> torch.Tensor:
    window = torch.stack(frames, dim=0)[::skip_n]
    window = window.float() / 255.0
    window = F.interpolate(
        window,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return (window - VIT_IMAGE_MEAN) / VIT_IMAGE_STD


def iter_sequences(image_paths: list[Path]) -> Iterator[list[MarioFrameRecord]]:
    current: list[MarioFrameRecord] = []
    previous: MarioFrameRecord | None = None

    for image_path in image_paths:
        record = parse_mario_frame(image_path)
        starts_new_sequence = (
            previous is None
            or record.run_id != previous.run_id
            or record.frame_id != previous.frame_id + 1
            or previous.alive_flag == 0
        )

        if starts_new_sequence:
            if current:
                yield current
            current = [record]
        else:
            current.append(record)

        previous = record

    if current:
        yield current


class MarioWindowDataset(IterableDataset):
    def __init__(
        self,
        *,
        data_root: str | Path,
        data_files: list[str],
        frames_per_window: int,
        window_stride: int,
        skip_n: int,
        image_size: int,
        fps: float = 30.0,
        max_windows_per_sequence: int | None = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.data_files = data_files
        self.spec = build_window_spec(
            frames_per_window=frames_per_window,
            window_stride=window_stride,
            skip_n=skip_n,
        )
        self.skip_n = skip_n
        self.image_size = image_size
        self.fps = fps
        self.max_windows_per_sequence = max_windows_per_sequence

    def __iter__(self) -> Iterator[dict[str, Any]]:
        image_paths = resolve_image_files(self.data_root, self.data_files)
        sequences = list(iter_sequences(image_paths))

        worker_info = get_worker_info()
        if worker_info is not None:
            sequences = sequences[worker_info.id :: worker_info.num_workers]

        for sequence in sequences:
            if len(sequence) < self.spec.frames_per_window:
                continue

            windows_yielded = 0
            for start in range(
                0,
                len(sequence) - self.spec.frames_per_window + 1,
                self.spec.stride,
            ):
                window_records = sequence[start : start + self.spec.frames_per_window]
                frames = [
                    read_image(str(record.path), mode=ImageReadMode.RGB)
                    for record in window_records
                ]
                yield {
                    "frame_indices": torch.tensor(
                        [record.frame_id for record in window_records],
                        dtype=torch.long,
                    ),
                    "fps": self.fps,
                    "metadata": {
                        "run_id": window_records[0].run_id,
                        "player_id": window_records[0].player_id,
                        "action_codes": [record.action_code for record in window_records],
                        "alive_flags": [record.alive_flag for record in window_records],
                        "source_paths": [str(record.path) for record in window_records],
                    },
                    "key": f"{window_records[0].run_id}:{window_records[0].frame_id}",
                    "url": str(window_records[0].path.parent),
                    "pixel_values": preprocess_frames(
                        frames,
                        skip_n=self.skip_n,
                        image_size=self.image_size,
                    ),
                }
                windows_yielded += 1
                if (
                    self.max_windows_per_sequence is not None
                    and windows_yielded >= self.max_windows_per_sequence
                ):
                    break


def collate_video_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "frame_indices": torch.stack([item["frame_indices"] for item in batch], dim=0),
        "fps": torch.tensor([item["fps"] for item in batch], dtype=torch.float32),
        "metadata": [item["metadata"] for item in batch],
        "key": [item["key"] for item in batch],
        "url": [item["url"] for item in batch],
        "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
    }
