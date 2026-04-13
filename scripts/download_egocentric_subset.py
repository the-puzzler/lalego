from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download, list_repo_files

from module.dataset import EGOCENTRIC_10K_REPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a local Egocentric-10K subset.")
    parser.add_argument("--factory", default="032", help="Factory id, e.g. 032")
    parser.add_argument(
        "--workers",
        default="*",
        help="Worker selector, e.g. '*', '001,002,003', or '001-012'",
    )
    parser.add_argument(
        "--dest",
        default="data/egocentric10k",
        help="Local dataset root.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel download workers. Defaults to all matched tar files.",
    )
    return parser.parse_args()


def worker_patterns(worker_selector: str) -> list[str]:
    if worker_selector == "*":
        return ["worker_*"]

    patterns: list[str] = []
    for chunk in worker_selector.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"invalid worker range {token}")
            patterns.extend(f"worker_{index:03d}" for index in range(start, end + 1))
        else:
            patterns.append(f"worker_{int(token):03d}")
    return patterns


def matching_repo_files(factory: str, worker_selector: str) -> list[str]:
    factory_name = f"factory_{int(factory):03d}"
    worker_globs = worker_patterns(worker_selector)
    repo_files = list_repo_files(EGOCENTRIC_10K_REPO, repo_type="dataset")

    matched = []
    for repo_file in repo_files:
        if not repo_file.endswith(".tar"):
            continue
        if not repo_file.startswith(f"{factory_name}/workers/"):
            continue
        worker_name = repo_file.split("/")[2]
        if any(fnmatch.fnmatch(worker_name, pattern) for pattern in worker_globs):
            matched.append(repo_file)

    matched = sorted(dict.fromkeys(matched))
    if not matched:
        raise ValueError(
            f"no tar files matched factory={factory_name} workers={worker_selector}"
        )
    return matched


def download_one(repo_file: str, dest: Path) -> str:
    local_path = hf_hub_download(
        repo_id=EGOCENTRIC_10K_REPO,
        repo_type="dataset",
        filename=repo_file,
        local_dir=str(dest),
    )
    return str(local_path)


def main() -> None:
    args = parse_args()
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    repo_files = matching_repo_files(args.factory, args.workers)
    jobs = args.jobs if args.jobs is not None else max(1, len(repo_files))
    print(f"factory: {int(args.factory):03d}")
    print(f"workers: {args.workers}")
    print(f"dest: {dest}")
    print(f"matched_tar_files: {len(repo_files)}")
    print(f"parallel_jobs: {jobs}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(download_one, repo_file, dest): repo_file for repo_file in repo_files
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            local_path = future.result()
            completed += 1
            print(f"[{completed}/{len(repo_files)}] {local_path}")


if __name__ == "__main__":
    main()
