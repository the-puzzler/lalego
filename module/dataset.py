from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torchaudio.functional as AF
from torch.utils.data import IterableDataset, get_worker_info


EEG_FILE_PATTERN = re.compile(r"BCICIV_2a_(?P<subject>\d+)\.csv", re.IGNORECASE)
LABEL_TO_ID = {
    "left": 0,
    "right": 1,
    "foot": 2,
    "tongue": 3,
}


@dataclass(frozen=True)
class EEGFileRecord:
    path: Path
    subject_id: str


def resolve_eeg_files(data_root: str | Path, data_files: list[str]) -> list[EEGFileRecord]:
    root = Path(data_root).expanduser()
    matched: list[Path] = []
    for pattern in data_files:
        matched.extend(path for path in root.glob(pattern) if path.is_file())

    records: list[EEGFileRecord] = []
    for path in sorted({path.resolve() for path in matched}):
        match = EEG_FILE_PATTERN.search(path.name)
        if match is None:
            continue
        records.append(
            EEGFileRecord(
                path=path,
                subject_id=f"A{int(match.group('subject')):02d}",
            )
        )

    if not records:
        raise ValueError(f"no EEG csv files matched data_root={root} data_files={data_files}")
    return records


def bandpass_filter_trials(
    trials: torch.Tensor,
    *,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
) -> torch.Tensor:
    filtered = AF.highpass_biquad(
        trials.reshape(-1, trials.shape[-1]),
        sample_rate=sample_rate,
        cutoff_freq=low_hz,
    )
    filtered = AF.lowpass_biquad(
        filtered,
        sample_rate=sample_rate,
        cutoff_freq=high_hz,
    )
    return filtered.reshape_as(trials)


def load_eeg_csv(
    path: Path,
    *,
    num_channels: int,
    train_session_suffixes: tuple[str, ...] = ("T",),
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    allowed_sessions = {suffix.upper() for suffix in train_session_suffixes}
    trial_rows: dict[str, list[list[float]]] = defaultdict(list)
    trial_times: dict[str, list[float]] = defaultdict(list)
    trial_labels: dict[str, str] = {}

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"EEG file is missing headers: {path}")

        eeg_columns = [column for column in reader.fieldnames if column.startswith("EEG-")]
        if len(eeg_columns) != num_channels:
            raise ValueError(
                f"Expected {num_channels} EEG columns in {path}, found {len(eeg_columns)}"
            )

        for row in reader:
            label = row.get("label", "").strip().lower()
            if label == "unknown":
                session_id = "E"
            else:
                session_id = "T"

            if session_id not in allowed_sessions:
                continue

            epoch_id = row["epoch"]
            trial_labels.setdefault(epoch_id, label)
            trial_times[epoch_id].append(float(row["time"]))
            trial_rows[epoch_id].append([float(row[column]) for column in eeg_columns])

    if not trial_rows:
        raise ValueError(f"EEG file produced no trials after filtering sessions: {path}")

    ordered_epochs = sorted(trial_rows, key=lambda epoch: int(epoch))
    trials = []
    times = None
    labels = []
    for epoch_id in ordered_epochs:
        epoch_rows = trial_rows[epoch_id]
        epoch_tensor = torch.tensor(epoch_rows, dtype=torch.float32).transpose(0, 1)
        trials.append(epoch_tensor)
        labels.append(trial_labels[epoch_id])

        epoch_times = torch.tensor(trial_times[epoch_id], dtype=torch.float32)
        if times is None:
            times = epoch_times
        elif epoch_times.shape != times.shape or not torch.allclose(epoch_times, times):
            raise ValueError(f"Inconsistent time axis across epochs in {path}")

    return torch.stack(trials, dim=0), times, labels


def crop_trials(
    trials: torch.Tensor,
    *,
    times: torch.Tensor,
    start_seconds: float,
    end_seconds: float,
) -> torch.Tensor:
    mask = (times >= start_seconds) & (times <= end_seconds)
    if not torch.any(mask):
        raise ValueError(
            f"Invalid EEG crop [{start_seconds}, {end_seconds}] for available range "
            f"[{times.min().item():.3f}, {times.max().item():.3f}]"
        )
    return trials[..., mask]


def tokenize_eeg_trial(trial: torch.Tensor, *, patch_size: int) -> torch.Tensor:
    num_channels, num_samples = trial.shape
    usable_samples = (num_samples // patch_size) * patch_size
    if usable_samples == 0:
        raise ValueError("patch_size is larger than the available EEG samples")

    trimmed = trial[:, :usable_samples]
    num_patches = usable_samples // patch_size
    return trimmed.reshape(num_channels, num_patches, patch_size).transpose(0, 1).contiguous()


class EEGTokenDataset(IterableDataset):
    def __init__(
        self,
        *,
        data_root: str | Path,
        data_files: list[str],
        sample_rate: float,
        num_channels: int,
        subject_ids: tuple[str, ...] | None = None,
        train_session_suffixes: tuple[str, ...] = ("T",),
        patch_size: int = 25,
        bandpass_low_hz: float = 8.0,
        bandpass_high_hz: float = 30.0,
        epoch_start_seconds: float = 0.5,
        epoch_end_seconds: float = 4.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.records = resolve_eeg_files(data_root, data_files)
        if subject_ids is not None:
            allowed_subjects = {subject_id.upper() for subject_id in subject_ids}
            self.records = [
                record for record in self.records if record.subject_id in allowed_subjects
            ]
        if not self.records:
            raise ValueError("No EEG files were found for the requested subject split")

        self.examples = self._build_examples(
            train_session_suffixes=train_session_suffixes,
            bandpass_low_hz=bandpass_low_hz,
            bandpass_high_hz=bandpass_high_hz,
            epoch_start_seconds=epoch_start_seconds,
            epoch_end_seconds=epoch_end_seconds,
        )

    def _build_examples(
        self,
        *,
        train_session_suffixes: tuple[str, ...],
        bandpass_low_hz: float,
        bandpass_high_hz: float,
        epoch_start_seconds: float,
        epoch_end_seconds: float,
    ) -> list[dict[str, Any]]:
        trials_by_subject: dict[str, list[tuple[EEGFileRecord, torch.Tensor, str]]] = {}
        for record in self.records:
            trials, times, labels = load_eeg_csv(
                record.path,
                num_channels=self.num_channels,
                train_session_suffixes=train_session_suffixes,
            )
            trials = bandpass_filter_trials(
                trials,
                sample_rate=self.sample_rate,
                low_hz=bandpass_low_hz,
                high_hz=bandpass_high_hz,
            )
            trials = crop_trials(
                trials,
                times=times,
                start_seconds=epoch_start_seconds,
                end_seconds=epoch_end_seconds,
            )
            subject_trials = trials_by_subject.setdefault(record.subject_id, [])
            subject_trials.extend((record, trial, label) for trial, label in zip(trials, labels))

        examples: list[dict[str, Any]] = []
        for subject_id in sorted(trials_by_subject):
            subject_trials = trials_by_subject[subject_id]
            stacked = torch.stack([trial for _, trial, _ in subject_trials], dim=0)
            mean = stacked.mean(dim=(0, 2), keepdim=True)
            std = stacked.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)

            for trial_index, (record, trial, label) in enumerate(subject_trials):
                normalized_trial = ((trial.unsqueeze(0) - mean) / std).squeeze(0)
                tokens = tokenize_eeg_trial(normalized_trial, patch_size=self.patch_size)
                examples.append(
                    {
                        "eeg_values": tokens,
                        "frame_indices": torch.arange(tokens.shape[0], dtype=torch.long),
                        "fps": self.sample_rate / self.patch_size,
                        "label_id": LABEL_TO_ID[label],
                        "label_name": label,
                        "metadata": {
                            "subject_id": record.subject_id,
                            "session_id": "T",
                            "trial_index": trial_index,
                            "label_name": label,
                            "source_path": str(record.path),
                        },
                        "key": f"{record.subject_id}:trial{trial_index:03d}",
                        "url": str(record.path),
                    }
                )

        return examples

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = get_worker_info()
        if worker_info is None:
            yield from self.examples
            return

        yield from self.examples[worker_info.id :: worker_info.num_workers]


def collate_eeg_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "eeg_values": torch.stack([item["eeg_values"] for item in batch], dim=0),
        "frame_indices": torch.stack([item["frame_indices"] for item in batch], dim=0),
        "fps": torch.tensor([item["fps"] for item in batch], dtype=torch.float32),
        "label_id": torch.tensor([item["label_id"] for item in batch], dtype=torch.long),
        "label_name": [item["label_name"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
        "key": [item["key"] for item in batch],
        "url": [item["url"] for item in batch],
    }
