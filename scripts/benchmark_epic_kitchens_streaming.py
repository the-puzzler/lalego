from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_REPO_ID = "lightly-ai/epic-kitchens-100-clips"
DEFAULT_DATA_ROOT = "data/epic_kitchens_100_clips_full"
DEFAULT_DATA_FILES = ["clips/**/*.mp4"]


@dataclass(frozen=True)
class SourceFile:
    name: str
    uri: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark EPIC-KITCHENS clip decoding from local disk and/or direct Hugging Face URLs."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("local", "remote", "both"),
        default="both",
        help="Which source to benchmark.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo id used for remote streaming.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision for remote streaming URLs.",
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Local dataset root.",
    )
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=DEFAULT_DATA_FILES,
        help="Glob patterns relative to --data-root.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=8,
        help="Number of clips to benchmark per source.",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=("first", "random"),
        default="random",
        help="How to select the benchmarked clips.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --sample-strategy=random.",
    )
    parser.add_argument(
        "--frames-per-window",
        type=int,
        default=32,
        help="Training-time frames per window after temporal downsampling.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=128,
        help="Stride in raw decoded frames.",
    )
    parser.add_argument(
        "--skip-n",
        type=int,
        default=4,
        help="Temporal downsampling factor used by training.",
    )
    parser.add_argument(
        "--max-windows-per-video",
        type=int,
        default=2,
        help="Maximum windows to decode from each clip.",
    )
    parser.add_argument(
        "--max-decode-frames",
        type=int,
        default=512,
        help="Maximum raw frames to decode from each clip.",
    )
    return parser.parse_args()


def choose_files(paths: list[str], *, sample_size: int, strategy: str, seed: int) -> list[str]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not paths:
        raise ValueError("no files available to sample")

    if sample_size >= len(paths):
        return list(paths)
    if strategy == "first":
        return paths[:sample_size]

    rng = random.Random(seed)
    return sorted(rng.sample(paths, sample_size))


def resolve_remote_video_files(*, repo_id: str, revision: str, data_files: list[str]) -> list[str]:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    matches: set[str] = set()
    for pattern in data_files:
        for path in fs.glob(f"datasets/{repo_id}/{pattern}", revision=revision):
            if path.endswith(".mp4"):
                matches.add(path.removeprefix(f"datasets/{repo_id}/"))

    if not matches:
        raise ValueError(f"no remote video files matched repo_id={repo_id} data_files={data_files}")

    return sorted(matches)


def benchmark_source(
    *,
    label: str,
    files: list[SourceFile],
    frames_per_window: int,
    window_stride: int,
    skip_n: int,
    max_windows_per_video: int,
    max_decode_frames: int,
) -> None:
    from module.dataset import iter_video_windows_from_path, make_raw_window_spec

    spec = make_raw_window_spec(
        frames_per_window=frames_per_window,
        window_stride=window_stride,
        skip_n=skip_n,
        min_frames=frames_per_window,
    )

    clip_times: list[float] = []
    first_window_times: list[float] = []
    decoded_windows = 0
    decoded_frames = 0
    failures = 0

    print(f"\n[{label}] benchmarking {len(files)} clips")
    total_started = time.perf_counter()

    for index, source_file in enumerate(files, start=1):
        clip_started = time.perf_counter()
        first_window_elapsed: float | None = None
        clip_window_count = 0
        clip_frame_count = 0
        average_fps = 0.0

        try:
            for window, frame_indices, average_fps in iter_video_windows_from_path(
                source_file.uri,
                spec,
                max_windows=max_windows_per_video,
                max_decode_frames=max_decode_frames,
            ):
                clip_window_count += 1
                clip_frame_count += int(window.shape[0])
                if first_window_elapsed is None:
                    first_window_elapsed = time.perf_counter() - clip_started
        except Exception as exc:
            failures += 1
            print(f"[{label}] {index:03d}/{len(files):03d} FAIL {source_file.name} :: {exc}")
            continue

        clip_elapsed = time.perf_counter() - clip_started
        clip_times.append(clip_elapsed)
        decoded_windows += clip_window_count
        decoded_frames += clip_frame_count
        if first_window_elapsed is not None:
            first_window_times.append(first_window_elapsed)
        first_window_display = (
            f"{first_window_elapsed:.2f}s" if first_window_elapsed is not None else "n/a"
        )

        print(
            f"[{label}] {index:03d}/{len(files):03d} "
            f"{source_file.name} | {clip_window_count} windows | "
            f"{clip_frame_count} raw frames | clip {clip_elapsed:.2f}s | "
            f"first window {first_window_display} | fps {average_fps:.2f}"
        )

    total_elapsed = time.perf_counter() - total_started
    if not clip_times:
        print(f"[{label}] no successful decodes")
        return

    print(f"\n[{label}] summary")
    print(f"clips: {len(files)}")
    print(f"successes: {len(clip_times)}")
    print(f"failures: {failures}")
    print(f"total wall time: {total_elapsed:.2f}s")
    print(f"decoded windows: {decoded_windows}")
    print(f"decoded raw frames inside windows: {decoded_frames}")
    print(f"avg clip time: {statistics.mean(clip_times):.2f}s")
    print(f"median clip time: {statistics.median(clip_times):.2f}s")
    if first_window_times:
        print(f"avg first-window latency: {statistics.mean(first_window_times):.2f}s")
    else:
        print("avg first-window latency: n/a")
    print(f"clips/sec: {len(clip_times) / total_elapsed:.2f}")
    print(f"windows/sec: {decoded_windows / total_elapsed:.2f}")
    print(f"raw frames/sec: {decoded_frames / total_elapsed:.2f}")


def main() -> None:
    args = parse_args()

    if args.mode in {"local", "both"}:
        from module.dataset import resolve_local_video_files

        local_paths = resolve_local_video_files(
            data_root=args.data_root,
            data_files=args.data_files,
        )
        selected_local = choose_files(
            local_paths,
            sample_size=args.sample_size,
            strategy=args.sample_strategy,
            seed=args.seed,
        )
        benchmark_source(
            label="local",
            files=[
                SourceFile(name=Path(path).name, uri=path)
                for path in selected_local
            ],
            frames_per_window=args.frames_per_window,
            window_stride=args.window_stride,
            skip_n=args.skip_n,
            max_windows_per_video=args.max_windows_per_video,
            max_decode_frames=args.max_decode_frames,
        )

    if args.mode in {"remote", "both"}:
        from huggingface_hub import hf_hub_url

        remote_paths = resolve_remote_video_files(
            repo_id=args.repo_id,
            revision=args.revision,
            data_files=args.data_files,
        )
        selected_remote = choose_files(
            remote_paths,
            sample_size=args.sample_size,
            strategy=args.sample_strategy,
            seed=args.seed,
        )
        benchmark_source(
            label="remote",
            files=[
                SourceFile(
                    name=Path(path).name,
                    uri=hf_hub_url(
                        repo_id=args.repo_id,
                        filename=path,
                        repo_type="dataset",
                        revision=args.revision,
                    ),
                )
                for path in selected_remote
            ],
            frames_per_window=args.frames_per_window,
            window_stride=args.window_stride,
            skip_n=args.skip_n,
            max_windows_per_video=args.max_windows_per_video,
            max_decode_frames=args.max_decode_frames,
        )


if __name__ == "__main__":
    main()
