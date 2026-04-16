from __future__ import annotations

import csv
import os
import sys
import wave
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402
from module.dataset import normalize_audio_clip, resolve_maestro_root  # noqa: E402


VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class PerformanceJob:
    split: str
    year: int
    composer: str
    title: str
    audio_path: str
    source_sample_rate: int


@dataclass(frozen=True)
class WorkerConfig:
    output_root: str
    sample_rate: int
    clip_seconds: float
    clip_stride_seconds: float
    mono: bool
    normalization: str
    dtype: str


DEFAULT_DTYPE = "float16"
DEFAULT_OVERWRITE = False


def canonical_split_name(split: str) -> str:
    normalized = split.strip().lower()
    aliases = {"val": "validation", "valid": "validation", "dev": "validation"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in VALID_SPLITS:
        raise ValueError(f"Unsupported split={split!r}; expected one of {sorted(VALID_SPLITS)}")
    return normalized


def read_manifest(root: Path, splits: tuple[str, ...]) -> list[PerformanceJob]:
    manifest_path = root / "maestro-v3.0.0.csv"
    jobs: list[PerformanceJob] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = canonical_split_name(row["split"])
            if split not in splits:
                continue
            audio_path = root / row["audio_filename"]
            if not audio_path.is_file():
                continue
            jobs.append(
                PerformanceJob(
                    split=split,
                    year=int(row["year"]),
                    composer=row["canonical_composer"],
                    title=row["canonical_title"],
                    audio_path=str(audio_path),
                    source_sample_rate=read_wav_metadata(audio_path),
                )
            )
    return jobs


def read_wav_metadata(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        return int(handle.getframerate())


def load_full_wav(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = int(handle.getframerate())
        num_channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, found sample_width={sample_width} for {path}")
        frame_bytes = handle.readframes(handle.getnframes())

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


def compute_clip_starts(num_samples: int, clip_num_samples: int, stride_samples: int) -> list[int]:
    if num_samples <= clip_num_samples:
        return [0]

    max_start = num_samples - clip_num_samples
    starts = list(range(0, max_start + 1, stride_samples))
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def sanitize_stem(path: Path) -> str:
    safe = []
    for char in path.stem:
        if char.isalnum() or char in ("-", "_"):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe)


def worker_init() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def process_performance(task: tuple[PerformanceJob, WorkerConfig, bool]) -> dict[str, object]:
    job, cfg, overwrite = task
    output_root = Path(cfg.output_root)
    split_dir = output_root / job.split
    split_dir.mkdir(parents=True, exist_ok=True)
    output_path = split_dir / f"{job.year}_{sanitize_stem(Path(job.audio_path))}.pt"

    if output_path.exists() and not overwrite:
        saved = torch.load(output_path, map_location="cpu")
        return {
            "split": job.split,
            "year": job.year,
            "composer": job.composer,
            "title": job.title,
            "audio_path": job.audio_path,
            "cache_path": str(output_path),
            "num_chunks": int(saved["chunks"].shape[0]),
            "num_channels": int(saved["chunks"].shape[1]),
            "num_samples": int(saved["chunks"].shape[2]),
            "dtype": str(saved["chunks"].dtype),
        }

    waveform, source_sample_rate = load_full_wav(Path(job.audio_path))
    if cfg.mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = resample_waveform(
        waveform,
        source_sample_rate=source_sample_rate,
        target_sample_rate=cfg.sample_rate,
    )

    clip_num_samples = int(round(cfg.sample_rate * cfg.clip_seconds))
    stride_samples = int(round(cfg.sample_rate * cfg.clip_stride_seconds))
    stride_samples = max(1, stride_samples)
    starts = compute_clip_starts(waveform.shape[-1], clip_num_samples, stride_samples)

    clips: list[torch.Tensor] = []
    clip_start_seconds: list[float] = []
    for start in starts:
        clip = waveform[:, start : start + clip_num_samples]
        if clip.shape[-1] < clip_num_samples:
            clip = F.pad(clip, (0, clip_num_samples - clip.shape[-1]))
        clip = normalize_audio_clip(clip, cfg.normalization)
        if cfg.dtype == "float16":
            clip = clip.to(torch.float16)
        else:
            clip = clip.to(torch.float32)
        clips.append(clip.contiguous())
        clip_start_seconds.append(start / cfg.sample_rate)

    chunk_tensor = torch.stack(clips, dim=0)
    torch.save(
        {
            "chunks": chunk_tensor,
            "clip_start_seconds": clip_start_seconds,
            "split": job.split,
            "year": job.year,
            "composer": job.composer,
            "title": job.title,
            "audio_path": job.audio_path,
            "sample_rate": cfg.sample_rate,
            "clip_seconds": cfg.clip_seconds,
            "clip_stride_seconds": cfg.clip_stride_seconds,
            "mono": cfg.mono,
            "normalization": cfg.normalization,
        },
        output_path,
    )
    return {
        "split": job.split,
        "year": job.year,
        "composer": job.composer,
        "title": job.title,
        "audio_path": job.audio_path,
        "cache_path": str(output_path),
        "num_chunks": int(chunk_tensor.shape[0]),
        "num_channels": int(chunk_tensor.shape[1]),
        "num_samples": int(chunk_tensor.shape[2]),
        "dtype": str(chunk_tensor.dtype),
    }


def write_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    manifest_path = output_root / "manifest.tsv"
    fieldnames = [
        "split",
        "year",
        "composer",
        "title",
        "audio_path",
        "cache_path",
        "num_chunks",
        "num_channels",
        "num_samples",
        "dtype",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    split_names = set(cfg.dataset_train_splits) | set(cfg.dataset_val_splits)
    if hasattr(cfg, "dataset_test_splits"):
        split_names |= set(cfg.dataset_test_splits)
    splits = tuple(canonical_split_name(split) for split in sorted(split_names))
    root = resolve_maestro_root(Path(cfg.dataset_root))
    output_root = Path(cfg.dataset_cache_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = read_manifest(root, splits)
    if not jobs:
        raise RuntimeError(f"No MAESTRO performances found under {root} for splits={splits}")

    worker_cfg = WorkerConfig(
        output_root=str(output_root),
        sample_rate=cfg.audio_sample_rate,
        clip_seconds=cfg.audio_clip_seconds,
        clip_stride_seconds=cfg.audio_clip_stride_seconds,
        mono=cfg.audio_mono,
        normalization=cfg.audio_normalization,
        dtype=DEFAULT_DTYPE,
    )

    workers = max(1, (os.cpu_count() or 1) - 1)
    overwrite = DEFAULT_OVERWRITE
    tasks = [(job, worker_cfg, overwrite) for job in jobs]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
        iterator = executor.map(process_performance, tasks, chunksize=1)
        for row in tqdm(iterator, total=len(tasks), desc="precompute", dynamic_ncols=True):
            rows.append(row)

    rows.sort(key=lambda row: (str(row["split"]), str(row["year"]), str(row["cache_path"])))
    write_manifest(output_root, rows)

    split_counts: dict[str, int] = {}
    chunk_counts: dict[str, int] = {}
    for row in rows:
        split = str(row["split"])
        split_counts[split] = split_counts.get(split, 0) + 1
        chunk_counts[split] = chunk_counts.get(split, 0) + int(row["num_chunks"])

    print(f"cached performances: {len(rows)}")
    print(f"workers: {workers}")
    for split in sorted(split_counts):
        print(
            f"{split}: performances={split_counts[split]} chunks={chunk_counts[split]}"
        )
    print(f"manifest: {output_root / 'manifest.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
