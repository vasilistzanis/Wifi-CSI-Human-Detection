#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class SeqStats:
    first_seq: int | None = None
    last_seq: int | None = None
    received_count: int = 0
    missing_count: int = 0
    gap_events: int = 0
    duplicate_count: int = 0
    reset_count: int = 0

    def update(self, seq: int) -> None:
        if self.first_seq is None:
            self.first_seq = seq
        elif self.last_seq is not None:
            if seq > self.last_seq + 1:
                self.missing_count += seq - self.last_seq - 1
                self.gap_events += 1
            elif seq == self.last_seq:
                self.duplicate_count += 1
            elif seq < self.last_seq:
                self.reset_count += 1

        self.last_seq = seq
        self.received_count += 1

    @property
    def unique_count(self) -> int:
        return self.received_count - self.duplicate_count

    @property
    def expected_count(self) -> int:
        return self.unique_count + self.missing_count

    @property
    def loss_percent(self) -> float:
        if self.expected_count == 0:
            return 0.0
        return (self.missing_count / self.expected_count) * 100.0


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CSI amplitude and phase from a dataset")
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset file to open. If omitted, the newest file in datasets/ is used.",
    )
    parser.add_argument(
        "-d",
        "--datasets-dir",
        default="datasets",
        help="Dataset directory used when no file is provided.",
    )
    parser.add_argument(
        "--unwrap-phase",
        action="store_true",
        help="Apply numpy.unwrap() along the time axis.",
    )
    return parser.parse_args()


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def get_latest_dataset(datasets_dir: Path):
    files = list(datasets_dir.glob("*.txt"))
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def extract_payload(line: str):
    start_idx = line.find("[")
    end_idx = line.find("]")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx + 1:
        return None
    return line[start_idx + 1:end_idx].strip()


def extract_seq(line: str):
    parts = line.split(",", 3)
    if len(parts) < 2:
        return None

    try:
        return int(parts[1])
    except ValueError:
        return None


def parse_csi_line(line: str):
    if not line.startswith("CSI_DATA"):
        return None

    payload = extract_payload(line)
    if not payload:
        return None

    token_count = payload.count(",") + 1
    values = np.fromstring(payload, sep=",", dtype=np.float32)

    if values.size != token_count or values.size < 2 or values.size % 2 != 0:
        return None

    imag = values[0::2]
    real = values[1::2]
    return (real + 1j * imag).astype(np.complex64)


def load_csi_matrix(dataset_path: Path):
    frames = []
    dropped_frames = 0
    expected_subcarriers = None
    seq_stats = SeqStats()

    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("CSI_DATA"):
                continue

            seq = extract_seq(line)
            if seq is not None:
                seq_stats.update(seq)

            frame = parse_csi_line(line)
            if frame is None:
                dropped_frames += 1
                continue

            if expected_subcarriers is None:
                expected_subcarriers = frame.shape[0]

            if frame.shape[0] != expected_subcarriers:
                dropped_frames += 1
                continue

            frames.append(frame)

    if not frames:
        return np.empty((0, 0), dtype=np.complex64), dropped_frames, seq_stats

    return np.vstack(frames), dropped_frames, seq_stats


def plot_all(complex_matrix: np.ndarray, dataset_path: Path, unwrap_phase: bool = False):
    if complex_matrix.size == 0:
        print("No valid CSI frames were found.")
        return

    amplitude = np.abs(complex_matrix)
    phase = np.angle(complex_matrix)
    if unwrap_phase:
        phase = np.unwrap(phase, axis=0)

    active_mask = np.any(amplitude > 0, axis=0)
    if not np.any(active_mask):
        print("All subcarriers are zero.")
        return

    active_indices = np.flatnonzero(active_mask)
    amplitude_active = amplitude[:, active_mask]
    phase_active = phase[:, active_mask]
    mean_amplitude = amplitude_active.mean(axis=0)

    title = (
        f"{dataset_path.name}\n"
        f"{complex_matrix.shape[0]} frames x {complex_matrix.shape[1]} subcarriers"
    )

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title("Amplitude Heatmap\n" + title)
    im1 = ax1.imshow(
        amplitude_active.T,
        aspect="auto",
        cmap="jet",
        interpolation="nearest",
        origin="lower",
    )
    ax1.set_xlabel("Frame Index")
    ax1.set_ylabel("Active Subcarrier")
    fig1.colorbar(im1, ax=ax1, label="Amplitude")
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.set_title("Mean Amplitude per Active Subcarrier\n" + title)
    ax2.plot(active_indices, mean_amplitude, color="#f7b731", linewidth=1.5)
    ax2.fill_between(active_indices, mean_amplitude, alpha=0.25, color="#f7b731")
    ax2.set_xlabel("Subcarrier Index")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    phase_title = "Phase Heatmap"
    if unwrap_phase:
        phase_title += " (Unwrapped)"
    ax3.set_title(phase_title + "\n" + title)
    im3 = ax3.imshow(
        phase_active.T,
        aspect="auto",
        cmap="hsv",
        interpolation="nearest",
        origin="lower",
        vmin=None if unwrap_phase else -np.pi,
        vmax=None if unwrap_phase else np.pi,
    )
    ax3.set_xlabel("Frame Index")
    ax3.set_ylabel("Active Subcarrier")
    fig3.colorbar(im3, ax=ax3, label="Phase (rad)")
    fig3.tight_layout()

    plt.show()


def main():
    args = parse_args()

    if args.dataset:
        dataset_path = resolve_path(args.dataset)
    else:
        datasets_dir = resolve_path(args.datasets_dir)
        dataset_path = get_latest_dataset(datasets_dir)

    if dataset_path is None:
        print("No dataset file was found.")
        return

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    print(f"Reading: {dataset_path}")
    complex_matrix, dropped_frames, seq_stats = load_csi_matrix(dataset_path)
    print(f"Valid frames  : {complex_matrix.shape[0]}")
    print(f"Unique frames : {seq_stats.unique_count}")
    print(f"Dropped frames: {dropped_frames}")
    print(f"Seq start/end : {seq_stats.first_seq} -> {seq_stats.last_seq}")
    print(f"Missing seq   : {seq_stats.missing_count} in {seq_stats.gap_events} gap(s)")
    print(f"Loss rate     : {seq_stats.loss_percent:.2f}%")
    print(f"Duplicate seq : {seq_stats.duplicate_count}")
    print(f"True resets   : {seq_stats.reset_count}")

    plot_all(complex_matrix, dataset_path, unwrap_phase=args.unwrap_phase)


if __name__ == "__main__":
    main()
