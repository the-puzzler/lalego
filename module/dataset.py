from __future__ import annotations

import csv
import wave
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class MaestroPerformanceRecord:
    audio_path: Path
    split: str
    composer: str
    title: str
    year: int
    source_sample_rate: int
    num_source_frames: int


@dataclass(frozen=True)
class AudioClipRecord:
    performance_index: int
    clip_index: int
    clip_frame_offset: int
    clip_num_source_frames: int


@dataclass(frozen=True)
class AudioSequenceRecord:
    performance_index: int
    sequence_index: int
    clip_records: tuple[AudioClipRecord, ...]


@dataclass(frozen=True)
class CachedPerformanceRecord:
    split: str
    year: int
    composer: str
    title: str
    audio_path: str
    cache_path: Path
    num_chunks: int
    num_channels: int
    num_samples: int
    dtype: str


@dataclass(frozen=True)
class CachedSequenceRecord:
    performance_index: int
    sequence_index: int
    chunk_start_index: int


def _canonical_split_name(split: str) -> str:
    normalized = split.strip().lower()
    aliases = {
        "val": "validation",
        "valid": "validation",
        "dev": "validation",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VALID_SPLITS:
        raise ValueError(f"Unsupported split={split!r}; expected one of {sorted(VALID_SPLITS)}")
    return normalized


def _normalize_splits(
    *,
    splits: tuple[str, ...] | None = None,
    subject_ids: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if splits is not None:
        requested = splits
    elif subject_ids is not None and all(candidate.lower() in VALID_SPLITS for candidate in subject_ids):
        requested = subject_ids
    else:
        requested = tuple(sorted(VALID_SPLITS))
    return tuple(_canonical_split_name(split) for split in requested)


def resolve_maestro_root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser()
    candidates = (
        root,
        root / "maestro-v3.0.0",
    )
    for candidate in candidates:
        if (candidate / "maestro-v3.0.0.csv").is_file():
            return candidate
    raise ValueError(
        "Could not resolve MAESTRO root. Expected `maestro-v3.0.0.csv` under "
        f"{root} or {root / 'maestro-v3.0.0'}."
    )


def normalize_audio_clip(waveform: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return waveform
    if mode != "per_clip":
        raise ValueError(f"Unsupported audio normalization mode={mode!r}")

    mean = waveform.mean(dim=-1, keepdim=True)
    std = waveform.std(dim=-1, keepdim=True).clamp_min(1e-4)
    return (waveform - mean) / std


def read_wav_metadata(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as handle:
        return handle.getframerate(), handle.getnframes()


def load_wav_clip(path: Path, *, frame_offset: int, num_frames: int) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        num_channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, found sample_width={sample_width} for {path}")

        safe_offset = min(max(frame_offset, 0), handle.getnframes())
        handle.setpos(safe_offset)
        frame_bytes = handle.readframes(num_frames)

    if not frame_bytes:
        return torch.zeros(num_channels, 0, dtype=torch.float32), sample_rate

    samples = torch.frombuffer(bytearray(frame_bytes), dtype=torch.int16)
    waveform = samples.to(torch.float32).reshape(-1, num_channels).transpose(0, 1)
    waveform = waveform / 32768.0
    return waveform, sample_rate


def resample_waveform(
    waveform: torch.Tensor,
    *,
    source_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    if source_sample_rate == target_sample_rate or waveform.shape[-1] == 0:
        return waveform

    target_num_samples = max(
        1,
        int(round(waveform.shape[-1] * target_sample_rate / source_sample_rate)),
    )
    return F.interpolate(
        waveform.unsqueeze(0),
        size=target_num_samples,
        mode="linear",
        align_corners=False,
    ).squeeze(0)


class AudioTokenDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        data_root: str | Path,
        splits: tuple[str, ...] | None = None,
        sample_rate: float,
        patch_size: int,
        clip_seconds: float,
        clip_stride_seconds: float | None = None,
        sequence_length: int = 4,
        sequence_stride: int = 1,
        mono: bool = True,
        normalization: str = "per_clip",
        subject_ids: tuple[str, ...] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        if clip_seconds <= 0:
            raise ValueError("clip_seconds must be positive")

        self.root = resolve_maestro_root(data_root)
        self.sample_rate = int(sample_rate)
        self.patch_size = int(patch_size)
        self.clip_seconds = float(clip_seconds)
        self.clip_stride_seconds = float(clip_stride_seconds or clip_seconds)
        self.sequence_length = int(sequence_length)
        self.sequence_stride = int(sequence_stride)
        self.mono = bool(mono)
        self.normalization = normalization
        self.splits = _normalize_splits(splits=splits, subject_ids=subject_ids)
        self.clip_num_samples = int(round(self.sample_rate * self.clip_seconds))
        if self.clip_num_samples <= 0:
            raise ValueError("clip_seconds produced an empty clip")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2 for dynamics learning")
        if self.sequence_stride <= 0:
            raise ValueError("sequence_stride must be positive")

        self.performances = self._build_performances()
        self.examples = self._build_examples()

    def _build_performances(self) -> list[MaestroPerformanceRecord]:
        manifest_path = self.root / "maestro-v3.0.0.csv"
        performances: list[MaestroPerformanceRecord] = []
        missing_audio = 0

        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                split = _canonical_split_name(row["split"])
                if split not in self.splits:
                    continue

                audio_path = self.root / row["audio_filename"]
                if not audio_path.is_file():
                    missing_audio += 1
                    continue

                source_sample_rate, num_source_frames = read_wav_metadata(audio_path)
                performances.append(
                    MaestroPerformanceRecord(
                        audio_path=audio_path,
                        split=split,
                        composer=row["canonical_composer"],
                        title=row["canonical_title"],
                        year=int(row["year"]),
                        source_sample_rate=int(source_sample_rate),
                        num_source_frames=int(num_source_frames),
                    )
                )

        if not performances:
            raise ValueError(
                "No MAESTRO audio files were available for the requested split selection. "
                "Check that extraction completed and dataset_root points at the extracted tree."
            )
        if missing_audio > 0:
            print(f"warning: skipped {missing_audio} MAESTRO rows whose audio files were not extracted yet")
        return performances

    def _build_examples(self) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        for performance_index, record in enumerate(self.performances):
            clip_num_source_frames = int(round(record.source_sample_rate * self.clip_seconds))
            clip_stride_source_frames = int(round(record.source_sample_rate * self.clip_stride_seconds))
            clip_num_source_frames = max(1, clip_num_source_frames)
            clip_stride_source_frames = max(1, clip_stride_source_frames)

            if record.num_source_frames <= clip_num_source_frames:
                starts = [0]
            else:
                max_start = record.num_source_frames - clip_num_source_frames
                starts = list(range(0, max_start + 1, clip_stride_source_frames))
                if starts[-1] != max_start:
                    starts.append(max_start)

            clip_records = [
                AudioClipRecord(
                    performance_index=performance_index,
                    clip_index=clip_index,
                    clip_frame_offset=int(clip_frame_offset),
                    clip_num_source_frames=clip_num_source_frames,
                )
                for clip_index, clip_frame_offset in enumerate(starts)
            ]

            if len(clip_records) < self.sequence_length:
                continue

            for sequence_index, start_index in enumerate(
                range(0, len(clip_records) - self.sequence_length + 1, self.sequence_stride)
            ):
                sequence_clips = tuple(clip_records[start_index : start_index + self.sequence_length])
                key = (
                    f"{record.split}:{record.year}:"
                    f"{record.audio_path.stem}:seq{sequence_index:05d}"
                )
                examples.append(
                    {
                        "record": AudioSequenceRecord(
                            performance_index=performance_index,
                            sequence_index=sequence_index,
                            clip_records=sequence_clips,
                        ),
                        "key": key,
                    }
                )
        return examples

    def _resample(self, waveform: torch.Tensor, *, source_sample_rate: int) -> torch.Tensor:
        return resample_waveform(
            waveform,
            source_sample_rate=source_sample_rate,
            target_sample_rate=self.sample_rate,
        )

    def _load_audio_clip(self, clip_record: AudioClipRecord) -> tuple[torch.Tensor, MaestroPerformanceRecord]:
        performance = self.performances[clip_record.performance_index]
        waveform, source_sample_rate = load_wav_clip(
            performance.audio_path,
            frame_offset=clip_record.clip_frame_offset,
            num_frames=clip_record.clip_num_source_frames,
        )
        if int(source_sample_rate) != performance.source_sample_rate:
            raise ValueError(
                f"Sample rate changed unexpectedly for {performance.audio_path}: "
                f"{source_sample_rate} != {performance.source_sample_rate}"
            )

        if self.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif not self.mono and waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        waveform = self._resample(waveform, source_sample_rate=performance.source_sample_rate)
        if waveform.shape[-1] < self.clip_num_samples:
            pad = self.clip_num_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        waveform = waveform[:, : self.clip_num_samples]
        waveform = normalize_audio_clip(waveform, self.normalization)
        return waveform, performance

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        sequence_record: AudioSequenceRecord = example["record"]
        clip_waveforms: list[torch.Tensor] = []
        performance: MaestroPerformanceRecord | None = None
        clip_start_seconds: list[float] = []
        for clip_record in sequence_record.clip_records:
            waveform, performance = self._load_audio_clip(clip_record)
            clip_waveforms.append(waveform)
            clip_start_seconds.append(clip_record.clip_frame_offset / performance.source_sample_rate)

        assert performance is not None
        chunks = torch.stack(clip_waveforms, dim=0)
        frame_indices = torch.arange(chunks.shape[0], dtype=torch.long)
        fps = 1.0 / self.clip_seconds
        metadata = {
            "split": performance.split,
            "composer": performance.composer,
            "title": performance.title,
            "year": performance.year,
            "sequence_index": sequence_record.sequence_index,
            "clip_start_seconds": clip_start_seconds,
            "source_path": str(performance.audio_path),
        }

        return {
            "signal_values": chunks,
            "eeg_values": chunks,
            "frame_indices": frame_indices,
            "fps": fps,
            "label_id": -1,
            "label_name": "",
            "metadata": metadata,
            "key": example["key"],
            "url": str(performance.audio_path),
        }


class CachedAudioTokenDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        cache_root: str | Path,
        splits: tuple[str, ...] | None = None,
        sequence_length: int = 4,
        sequence_stride: int = 1,
        max_cached_payloads: int = 2,
        subject_ids: tuple[str, ...] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.cache_root = Path(cache_root).expanduser()
        self.splits = _normalize_splits(splits=splits, subject_ids=subject_ids)
        self.sequence_length = int(sequence_length)
        self.sequence_stride = int(sequence_stride)
        self.max_cached_payloads = max(0, int(max_cached_payloads))
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2 for dynamics learning")
        if self.sequence_stride <= 0:
            raise ValueError("sequence_stride must be positive")

        self.performances = self._build_performances()
        self.examples = self._build_examples()
        self._cache_payloads: OrderedDict[Path, dict[str, Any]] = OrderedDict()

    def _build_performances(self) -> list[CachedPerformanceRecord]:
        manifest_path = self.cache_root / "manifest.tsv"
        if not manifest_path.is_file():
            raise ValueError(
                f"Cached MAESTRO manifest not found at {manifest_path}. "
                "Run scripts/precompute_maestro_chunks.py first."
            )

        performances: list[CachedPerformanceRecord] = []
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                split = _canonical_split_name(row["split"])
                if split not in self.splits:
                    continue
                cache_path = Path(row["cache_path"])
                if not cache_path.is_absolute():
                    cache_path = (self.cache_root / cache_path).resolve()
                if not cache_path.is_file():
                    continue
                performances.append(
                    CachedPerformanceRecord(
                        split=split,
                        year=int(row["year"]),
                        composer=row["composer"],
                        title=row["title"],
                        audio_path=row["audio_path"],
                        cache_path=cache_path,
                        num_chunks=int(row["num_chunks"]),
                        num_channels=int(row["num_channels"]),
                        num_samples=int(row["num_samples"]),
                        dtype=row["dtype"],
                    )
                )

        if not performances:
            raise ValueError(
                "No cached MAESTRO performances were found for the requested split selection. "
                f"Check cache_root={self.cache_root} and preprocessing completion."
            )
        return performances

    def _build_examples(self) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        for performance_index, record in enumerate(self.performances):
            if record.num_chunks < self.sequence_length:
                continue
            for sequence_index, start_index in enumerate(
                range(0, record.num_chunks - self.sequence_length + 1, self.sequence_stride)
            ):
                key = (
                    f"{record.split}:{record.year}:"
                    f"{record.cache_path.stem}:seq{sequence_index:05d}"
                )
                examples.append(
                    {
                        "record": CachedSequenceRecord(
                            performance_index=performance_index,
                            sequence_index=sequence_index,
                            chunk_start_index=start_index,
                        ),
                        "key": key,
                    }
                )
        return examples

    def _load_payload(self, cache_path: Path) -> dict[str, Any]:
        payload = self._cache_payloads.get(cache_path)
        if payload is None:
            payload = torch.load(cache_path, map_location="cpu")
            if self.max_cached_payloads > 0:
                self._cache_payloads[cache_path] = payload
                self._cache_payloads.move_to_end(cache_path)
                while len(self._cache_payloads) > self.max_cached_payloads:
                    self._cache_payloads.popitem(last=False)
        elif self.max_cached_payloads > 0:
            self._cache_payloads.move_to_end(cache_path)
        return payload

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        sequence_record: CachedSequenceRecord = example["record"]
        performance = self.performances[sequence_record.performance_index]
        payload = self._load_payload(performance.cache_path)

        start = sequence_record.chunk_start_index
        end = start + self.sequence_length
        chunks = payload["chunks"][start:end].to(torch.float32)
        clip_start_seconds = payload["clip_start_seconds"][start:end]
        frame_indices = torch.arange(chunks.shape[0], dtype=torch.long)
        fps = 1.0 / float(payload["clip_seconds"])
        metadata = {
            "split": performance.split,
            "composer": performance.composer,
            "title": performance.title,
            "year": performance.year,
            "sequence_index": sequence_record.sequence_index,
            "clip_start_seconds": clip_start_seconds,
            "source_path": performance.audio_path,
            "cache_path": str(performance.cache_path),
        }

        return {
            "signal_values": chunks,
            "eeg_values": chunks,
            "frame_indices": frame_indices,
            "fps": fps,
            "label_id": -1,
            "label_name": "",
            "metadata": metadata,
            "key": example["key"],
            "url": performance.audio_path,
        }


def collate_audio_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    signal_values = torch.stack([item["signal_values"] for item in batch], dim=0)
    return {
        "signal_values": signal_values,
        "eeg_values": signal_values,
        "frame_indices": torch.stack([item["frame_indices"] for item in batch], dim=0),
        "fps": torch.tensor([item["fps"] for item in batch], dtype=torch.float32),
        "label_id": torch.tensor([item["label_id"] for item in batch], dtype=torch.long),
        "label_name": [item["label_name"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
        "key": [item["key"] for item in batch],
        "url": [item["url"] for item in batch],
    }


def build_audio_dataset(
    *,
    dataset_backend: str,
    dataset_root: str | Path,
    dataset_cache_root: str | Path | None,
    splits: tuple[str, ...],
    sample_rate: float,
    patch_size: int,
    clip_seconds: float,
    clip_stride_seconds: float,
    sequence_length: int,
    sequence_stride: int,
    mono: bool,
    normalization: str,
    max_cached_payloads: int = 2,
) -> Dataset[dict[str, Any]]:
    if dataset_backend == "raw":
        return AudioTokenDataset(
            data_root=dataset_root,
            splits=splits,
            sample_rate=sample_rate,
            patch_size=patch_size,
            clip_seconds=clip_seconds,
            clip_stride_seconds=clip_stride_seconds,
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
            mono=mono,
            normalization=normalization,
        )
    if dataset_backend == "cached":
        if dataset_cache_root is None:
            raise ValueError("dataset_backend='cached' requires dataset_cache_root")
        return CachedAudioTokenDataset(
            cache_root=dataset_cache_root,
            splits=splits,
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
            max_cached_payloads=max_cached_payloads,
        )
    raise ValueError(f"Unsupported dataset_backend={dataset_backend!r}")
