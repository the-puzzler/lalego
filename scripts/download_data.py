from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path


KAGGLE_DATASET = "aymanmostafa11/eeg-motor-imagery-bciciv-2a"
ARCHIVE_NAME = "eeg-motor-imagery-bciciv-2a.zip"
OUTPUT_DIR_NAME = "bciciv2a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the BCI Competition IV 2a EEG dataset from Kaggle."
    )
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parents[1] / "data",
        type=Path,
        help="Directory where the archive and extracted dataset will be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite existing files.",
    )
    parser.add_argument(
        "--dataset",
        default=KAGGLE_DATASET,
        help="Kaggle dataset slug in the form owner/name.",
    )
    return parser.parse_args()


def ensure_kaggle_cli() -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI not found. Install it with `pip install kaggle` and configure "
            "`~/.kaggle/kaggle.json` before downloading."
        )


def download_archive(dataset: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"Archive already exists: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(destination.parent),
    ]
    if force:
        command.append("--force")

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)

    downloaded_archive = destination.parent / f"{dataset.rsplit('/', 1)[-1]}.zip"
    if downloaded_archive != destination:
        downloaded_archive.replace(destination)


def extract_archive(archive_path: Path, output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        print(f"Dataset already exists: {output_dir}")
        return

    if force and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(output_dir)


def main() -> int:
    args = parse_args()
    ensure_kaggle_cli()

    output_root = args.output_dir.resolve()
    archive_path = output_root / ARCHIVE_NAME
    dataset_path = output_root / OUTPUT_DIR_NAME

    download_archive(args.dataset, archive_path, force=args.force)
    extract_archive(archive_path, dataset_path, force=args.force)

    print(f"Dataset ready at {dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
