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
from pathlib import Path

import matplotlib

# ── Shared parsing & loading from csi_parser ──────────────────────────────────
from csi_parser import (
    configure_console_output,
    resolve_path,
    get_latest_dataset,
    load_csi_matrix,
)

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
