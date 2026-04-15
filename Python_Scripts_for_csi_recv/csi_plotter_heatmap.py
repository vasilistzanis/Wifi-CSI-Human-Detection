#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSI Amplitude & Phase Plotter (Thesis Grade)
Reads raw serial dump (.txt) or CSV (.csv) from the ESP32-C6 recv.

Usage:
  python csi_plotter_heatmap.py                          # latest file in datasets/
  python csi_plotter_heatmap.py path/to/file.txt
  python csi_plotter_heatmap.py path/to/file.csv --unwrap-phase
  python csi_plotter_heatmap.py path/to/file.txt --save  # saves PNG alongside file
"""

from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib


def configure_console_output() -> None:
    """Avoid UnicodeEncodeError on legacy Windows console encodings."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass


configure_console_output()


try:
    matplotlib.use("Qt5Agg")
except Exception:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

import matplotlib.pyplot as plt
plt.ioff()  # Disable interactive mode for faster background rendering
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RECV_FIELD_COUNT = 15

# ════════════════════════════════════════════════════════════════════════
# SEQUENCE STATS
# ════════════════════════════════════════════════════════════════════════

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
            diff = seq - self.last_seq
            if diff > 1:
                self.missing_count += diff - 1
                self.gap_events += 1
            elif diff == 0:
                self.duplicate_count += 1
                self.received_count += 1
                return                  # do NOT update last_seq for duplicate
            elif diff < 0:
                self.reset_count += 1  # sequence counter reset or reorder

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


# ════════════════════════════════════════════════════════════════════════
# PATH HELPERS
# ════════════════════════════════════════════════════════════════════════

def resolve_path(path_arg: str) -> Path:
    """Resolve relative path from script directory."""
    path = Path(path_arg)
    return path if path.is_absolute() else BASE_DIR / path


def get_latest_dataset(datasets_dir: Path) -> Path | None:
    """Return newest .txt or .csv file in datasets_dir."""
    files = list(datasets_dir.glob("*.txt")) + list(datasets_dir.glob("*.csv"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def split_recv_fields(line: str) -> list[str] | None:
    """Split one recv/logger line and reject malformed concatenated records."""
    if not line.startswith("CSI_DATA"):
        return None

    parts = [part.strip() for part in line.strip().split(",", RECV_FIELD_COUNT - 1)]
    if len(parts) != RECV_FIELD_COUNT:
        return None

    for idx in (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13):
        try:
            int(parts[idx])
        except ValueError:
            return None

    return parts


# ════════════════════════════════════════════════════════════════════════
# PARSING
# ════════════════════════════════════════════════════════════════════════

def extract_seq(line: str) -> int | None:
    """Extract sequence number from CSI_DATA line (field index 1)."""
    parts = split_recv_fields(line)
    return int(parts[1]) if parts is not None else None


def parse_csi_line(line: str) -> np.ndarray | None:
    """
    Parse one CSI_DATA text line into a complex64 array.

    ESP32 CSI buf layout: [imag0, real0, imag1, real1, ...]
    So values[0::2] = imaginary, values[1::2] = real.
    complex(i) = real[i] + j*imag[i]
    """
    parts = split_recv_fields(line)
    if parts is None:
        return None

    payload = parts[14].strip().strip('"')
    if not payload.startswith("[") or not payload.endswith("]"):
        return None

    payload = payload[1:-1].strip()
    if not payload:
        return None

    # Fast CSV parsing (np.fromstring with sep is NOT deprecated).
    # Fallback to list comprehension for future-proofing.
    try:
        values = np.fromstring(payload, sep=",", dtype=np.float32)
    except Exception:
        try:
            values = np.array([float(x) for x in payload.split(',')],
                              dtype=np.float32)
        except (ValueError, AttributeError):
            return None

    token_count = payload.count(",") + 1
    if values.size != token_count or values.size < 2 or values.size % 2 != 0:
        return None

    first_word_invalid = int(parts[13]) != 0
    if first_word_invalid and values.size >= 4:
        values = values.copy()
        values[:4] = 0.0

    imag = values[0::2]
    real = values[1::2]
    return (real + 1j * imag).astype(np.complex64)


def load_csi_matrix(dataset_path: Path) -> tuple[np.ndarray, int, SeqStats]:
    """
    Load CSI data from a .txt or .csv file.

    Returns:
      complex_matrix  : (N_frames, N_subcarriers) complex64
      dropped_frames  : count of unparseable lines
      seq_stats       : SeqStats object with gap/loss metrics
    
    Raises:
      FileNotFoundError: if file doesn't exist
      PermissionError: if file can't be read
      ValueError: if file is empty or contains no valid CSI data
    """
    # File validation
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    if dataset_path.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty: {dataset_path}")

    frames: list[np.ndarray] = []
    dropped_frames = 0
    expected_subcarriers: int | None = None
    seq_stats = SeqStats()

    # File I/O with error handling
    try:
        with open(dataset_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
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
    except PermissionError as e:
        raise PermissionError(f"Cannot read file {dataset_path}: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"File encoding error in {dataset_path}: {e}")

    if not frames:
        # Validation empty check
        raise ValueError(
            f"No valid CSI frames found in {dataset_path}. "
            f"Dropped {dropped_frames} malformed lines. "
            f"Check that the file contains valid CSI_DATA lines."
        )

    return np.vstack(frames), dropped_frames, seq_stats


# ════════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════════

def plot_all(complex_matrix: np.ndarray, dataset_path: Path,
             unwrap_phase: bool = False, save: bool = False) -> None:
    """
    Plot amplitude heatmap, mean amplitude, and phase heatmap.
    Optionally save as PNG next to the dataset file.
    """
    if complex_matrix.size == 0:
        print("⚠️  No valid CSI frames to plot.")
        return

    amplitude = np.abs(complex_matrix)
    phase = np.angle(complex_matrix)
    if unwrap_phase:
        phase = np.unwrap(phase, axis=0)

    active_mask = np.any(amplitude > 0, axis=0)
    if not np.any(active_mask):
        print("⚠️  All subcarriers are zero — nothing to plot.")
        return

    active_indices = np.flatnonzero(active_mask)
    amp_active   = amplitude[:, active_mask]
    phase_active = phase[:, active_mask]
    mean_amp     = amp_active.mean(axis=0)

    title = (f"{dataset_path.name}  —  "
             f"{complex_matrix.shape[0]} frames × "
             f"{int(active_mask.sum())} active subcarriers")

    # ── Amplitude Heatmap ─────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title("Amplitude Heatmap\n" + title)
    vmin = np.percentile(amp_active, 2)
    vmax = np.percentile(amp_active, 98)
    im1 = ax1.imshow(amp_active.T, aspect="auto", cmap="viridis",
                     interpolation="nearest", origin="lower", vmin=vmin, vmax=vmax)
    ax1.set_xlabel("Frame Index")
    ax1.set_ylabel("Active Subcarrier")
    fig1.colorbar(im1, ax=ax1, label="Amplitude")
    fig1.tight_layout()

    # ── Mean Amplitude per Subcarrier ─────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.set_title("Mean Amplitude per Active Subcarrier\n" + title)
    ax2.plot(active_indices, mean_amp, color="#f7b731", linewidth=1.5)
    ax2.fill_between(active_indices, mean_amp, alpha=0.25, color="#f7b731")
    ax2.set_xlabel("Subcarrier Index")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    # ── Phase Heatmap ─────────────────────────────────────────────────────
    phase_title = "Phase Heatmap" + (" (Unwrapped)" if unwrap_phase else "")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.set_title(phase_title + "\n" + title)
    im3 = ax3.imshow(
        phase_active.T, aspect="auto", cmap="hsv",
        interpolation="nearest", origin="lower",
        vmin=None if unwrap_phase else -np.pi,
        vmax=None if unwrap_phase else  np.pi,
    )
    ax3.set_xlabel("Frame Index")
    ax3.set_ylabel("Active Subcarrier")
    fig3.colorbar(im3, ax=ax3, label="Phase (rad)")
    fig3.tight_layout()

    if save:
        stem = dataset_path.stem
        parent = dataset_path.parent
        for fig, suffix in [(fig1, "_amp_heatmap"), (fig2, "_mean_amp"), (fig3, "_phase")]:
            out = parent / f"{stem}{suffix}.png"
            try:
                fig.savefig(out, dpi=150, bbox_inches="tight")
                print(f"💾 Saved: {out}")
            except (PermissionError, OSError) as e:
                print(f"⚠️  Could not save {out}: {e}")

    plt.show()


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot CSI amplitude and phase from a dataset file"
    )
    parser.add_argument(
        "dataset", nargs="?",
        help="Dataset file (.txt or .csv). Omit to use newest file in datasets/."
    )
    parser.add_argument(
        "-d", "--datasets-dir", default="datasets",
        help="Dataset directory (default: datasets/)"
    )
    parser.add_argument(
        "--unwrap-phase", action="store_true",
        help="Apply numpy.unwrap() along the time axis for phase plot"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save plots as PNG files next to the dataset"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset:
        dataset_path = resolve_path(args.dataset)
    else:
        datasets_dir = resolve_path(args.datasets_dir)
        dataset_path = get_latest_dataset(datasets_dir)

    if dataset_path is None:
        print("❌ No dataset file found in datasets/ directory.")
        print("   Use: python csi_plotter_heatmap.py <file.txt>")
        return 1

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return 1

    print(f"📂 Reading: {dataset_path}")
    
    # Execute safely
    try:
        complex_matrix, dropped_frames, seq_stats = load_csi_matrix(dataset_path)
    except (FileNotFoundError, PermissionError, ValueError) as e:
        print(f"❌ Error loading dataset: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

    print(f"✅ Loaded successfully")
    print(f"Valid frames  : {complex_matrix.shape[0]}")
    print(f"Subcarriers   : {complex_matrix.shape[1]}")
    print(f"Unique frames : {seq_stats.unique_count}")
    print(f"Dropped frames: {dropped_frames}")
    print(f"Seq range     : {seq_stats.first_seq} → {seq_stats.last_seq}")
    print(f"Missing seq   : {seq_stats.missing_count} "
          f"in {seq_stats.gap_events} gap(s)")
    print(f"Loss rate     : {seq_stats.loss_percent:.2f}%")
    print(f"Duplicates    : {seq_stats.duplicate_count}")
    print(f"Resets        : {seq_stats.reset_count}")

    plot_all(complex_matrix, dataset_path,
             unwrap_phase=args.unwrap_phase, save=args.save)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
