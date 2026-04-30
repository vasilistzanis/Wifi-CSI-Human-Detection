#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSI Parser - Shared Parsing & Loading Utilities
================================================
Central module for all CSI data parsing, loading, and path resolution.

Used by:
  * csi_plotter_heatmap.py   (plotting)
  * live_predict.py           (real-time inference)
  * live_data_visualization.py (live PyQt viewer)
  * plot_lines_data_preprocessing.py
  * visualize_all_steps_heatmap_data_preprocessing.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ========================================================================
# CONSTANTS
# ========================================================================

BASE_DIR = Path(__file__).resolve().parent
RECV_FIELD_COUNT = 15


# ========================================================================
# CONSOLE HELPER
# ========================================================================

def configure_console_output() -> None:
    """Avoid UnicodeEncodeError on legacy Windows console encodings."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass


# ========================================================================
# SEQUENCE STATS
# ========================================================================

@dataclass(frozen=True)
class SeqTransition:
    missing_count: int = 0
    gap_event: bool = False
    duplicate: bool = False
    reset: bool = False


def analyze_seq_transition(previous_seq: int | None, current_seq: int) -> SeqTransition:
    """
    Classify one sequence-number transition without assuming a fixed wrap modulus.

    The recorded datasets in this project use large monotonic sequence values,
    so negative jumps are treated as resets/reordering rather than wrapped
    counters.
    """
    if previous_seq is None:
        return SeqTransition()

    diff = current_seq - previous_seq
    if diff > 1:
        return SeqTransition(missing_count=diff - 1, gap_event=True)
    if diff == 0:
        return SeqTransition(duplicate=True)
    if diff < 0:
        return SeqTransition(reset=True)
    return SeqTransition()


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
            transition = analyze_seq_transition(self.last_seq, seq)
            if transition.gap_event:
                self.missing_count += transition.missing_count
                self.gap_events += 1
            elif transition.duplicate:
                self.duplicate_count += 1
                self.received_count += 1
                return                  # do NOT update last_seq for duplicate
            elif transition.reset:
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


# ========================================================================
# PATH HELPERS
# ========================================================================

def resolve_path(path_arg: str) -> Path:
    """Resolve relative path from script directory."""
    path = Path(path_arg)
    return path if path.is_absolute() else BASE_DIR / path


def get_latest_dataset(datasets_dir: Path) -> Path | None:
    """Return newest .txt or .csv file in datasets_dir."""
    files = list(datasets_dir.glob("*.txt")) + list(datasets_dir.glob("*.csv"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


# ========================================================================
# LINE-LEVEL PARSING
# ========================================================================

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


def extract_seq(line: str) -> int | None:
    """Extract sequence number from CSI_DATA line (field index 1)."""
    parts = split_recv_fields(line)
    return int(parts[1]) if parts is not None else None


def parse_csi_line(line: str, expected_subcarriers: int | None = None) -> np.ndarray | None:
    """
    Parse one CSI_DATA text line into a complex64 array.

    ESP32 CSI buf layout: [imag0, real0, imag1, real1, ...]
    So values[0::2] = imaginary, values[1::2] = real.
    complex(i) = real[i] + j*imag[i]

    Parameters
    ----------
    line : str
        Raw CSI_DATA line from serial or file.
    expected_subcarriers : int or None
        If provided, only accept frames with exactly this many subcarriers.
        Used by live_data_visualization for strict validation.
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

    # Optional strict subcarrier count validation
    n_subcarriers = values.size // 2
    if expected_subcarriers is not None and n_subcarriers != expected_subcarriers:
        return None

    first_word_invalid = int(parts[13]) != 0
    if first_word_invalid and values.size >= 4:
        values = values.copy()
        values[:4] = 0.0

    imag = values[0::2]
    real = values[1::2]
    return (real + 1j * imag).astype(np.complex64)


# ========================================================================
# FILE-LEVEL LOADING
# ========================================================================

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
