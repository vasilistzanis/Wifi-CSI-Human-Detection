#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSI Line Plotter — Thesis / Paper Grade
Visualizes the full signal processing pipeline as 2D line plots.

Shows 5 stages side-by-side so the reader can see exactly what each
processing step does to the CSI signal. Designed for thesis figures.

Usage:
  python plot_lines.py                          # latest file in datasets/
  python plot_lines.py path/to/file.txt
  python plot_lines.py file.txt --save
  python plot_lines.py file.txt --n-subcarriers 6 --pca-components 5
  python plot_lines.py file.txt --no-diff --no-bg
"""

import sys
import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from csi_plotter import load_csi_matrix, resolve_path, get_latest_dataset
from data_preprocessing import CSIPipeline


# ════════════════════════════════════════════════════════════════════════
# ARGS
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="CSI Line Plotter — Thesis / Paper Grade"
    )
    p.add_argument(
        "file", nargs="?", default=None,
        help="TXT or CSV file (default: latest in datasets/)"
    )
    p.add_argument(
        "--save", action="store_true",
        help="Save figure as PNG next to the dataset file"
    )
    p.add_argument(
        "--n-subcarriers", type=int, default=5,
        help="Number of subcarriers to overlay (default: 5)"
    )
    p.add_argument(
        "--pca-components", type=int, default=3,
        help="Number of PCA components to show (default: 3)"
    )
    p.add_argument(
        "--background-frames", type=int, default=100,
        help="Frames used as static background (default: 100 = 1 s)"
    )
    p.add_argument(
        "--cutoff", type=float, default=12.0,
        help="Butterworth cutoff in Hz (default: 12)"
    )
    p.add_argument(
        "--no-bg", action="store_true",
        help="Disable background subtraction"
    )
    p.add_argument(
        "--no-diff", action="store_true",
        help="Disable temporal difference"
    )
    p.add_argument(
        "--fs", type=float, default=100.0,
        help="Sampling frequency in Hz (default: 100)"
    )
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════

def select_subcarriers(n_active: int, n_wanted: int) -> list[int]:
    """
    Select n_wanted subcarrier indices spread across the active spectrum.
    Uses linspace so we always get indices within bounds.
    """
    if n_wanted >= n_active:
        return list(range(n_active))
    # Spread evenly: avoid index 0 and last (often guard)
    margin = max(1, n_active // (n_wanted * 2))
    indices = np.linspace(margin, n_active - 1 - margin, n_wanted, dtype=int)
    return list(dict.fromkeys(indices.tolist()))  # deduplicate, keep order


def make_time_axis(n_frames: int, fs: float) -> np.ndarray:
    return np.arange(n_frames) / fs


def annotate_bg_region(ax, bg_frames: int, fs: float, alpha: float = 0.12):
    """Shade the background calibration period on an axis."""
    bg_end_s = bg_frames / fs
    ax.axvspan(0, bg_end_s, color="#ffcc00", alpha=alpha, zorder=0)
    ax.axvline(bg_end_s, color="#ffcc00", linewidth=1.0,
               linestyle="--", alpha=0.7, zorder=1)


def style_ax(ax, title: str, ylabel: str, show_xlabel: bool = False,
             fs: float = 100.0):
    """Apply consistent styling to a single axis."""
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6,
                 color="#222222")
    ax.set_ylabel(ylabel, fontsize=9, color="#333333")
    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=9, color="#333333")
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5, linestyle='-')
    ax.spines[['top', 'right']].set_visible(False)


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── File resolution ───────────────────────────────────────────────────
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            sys.exit(1)
    else:
        default_dir = resolve_path("datasets")
        if not default_dir.exists():
            print(f"❌ datasets/ directory not found — pass a file explicitly")
            sys.exit(1)
        file_path = get_latest_dataset(default_dir)
        if file_path is None:
            print(f"❌ No TXT/CSV files found in {default_dir}")
            sys.exit(1)

    print(f"\n📂 Loading: {file_path.name}")
    complex_matrix, dropped, seq_stats = load_csi_matrix(file_path)

    if complex_matrix.size == 0:
        print("❌ No valid frames — exiting")
        sys.exit(1)

    n_frames, n_sub = complex_matrix.shape
    print(f"   {n_frames} frames × {n_sub} subcarriers | "
          f"loss={seq_stats.loss_percent:.2f}% | "
          f"seq {seq_stats.first_seq}→{seq_stats.last_seq}")

    # ── Pipeline — step by step to capture every stage ───────────────────
    bg_frames = 0 if args.no_bg else args.background_frames
    pipeline = CSIPipeline(
        fs=args.fs,
        background_frames=bg_frames,
        use_diff=not args.no_diff,
    )

    # Stage 0: Raw amplitude (alle subcarriers including nulls)
    amp_raw = np.abs(complex_matrix)                               # (N, n_sub)

    # Stage 1: Null removal + Hampel + Butterworth
    amp_null = pipeline.remove_null_subcarriers(complex_matrix, fit=True)
    amp_hamp = pipeline.apply_hampel_filter(amp_null)
    amp_filt = pipeline.apply_lowpass_filter(amp_hamp, cutoff=args.cutoff)

    # Stage 2: Background subtraction
    amp_bg   = pipeline.apply_background_subtraction(amp_filt, fit=True)
    bg_enabled = bg_frames > 0 and pipeline.background_mean is not None

    # Stage 3: Temporal diff
    amp_diff = pipeline.apply_temporal_diff(amp_bg)
    diff_enabled = not args.no_diff and amp_diff.shape[0] < amp_bg.shape[0]

    # Stage 4: PCA
    n_pca = min(args.pca_components, amp_diff.shape[0] - 1, amp_diff.shape[1])
    pca = PCA(n_components=n_pca)
    pca_data = pca.fit_transform(amp_diff)
    explained = pca.explained_variance_ratio_ * 100

    n_active = amp_filt.shape[1]
    sc_indices = select_subcarriers(n_active, args.n_subcarriers)

    print(f"   Active subcarriers: {n_active} | "
          f"Plotting {len(sc_indices)} subcarriers: {sc_indices}")
    print(f"   PCA {n_pca} components explaining "
          f"{explained.sum():.1f}% total variance")

    # ── Time axes ─────────────────────────────────────────────────────────
    t_full = make_time_axis(amp_filt.shape[0], args.fs)
    t_diff = make_time_axis(amp_diff.shape[0], args.fs)
    t_pca  = make_time_axis(pca_data.shape[0], args.fs)
    duration = t_full[-1]

    # ════════════════════════════════════════════════════════════════════
    # FIGURE SETUP
    # ════════════════════════════════════════════════════════════════════

    # Try IEEE-style or fallback to seaborn-whitegrid
    for style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot']:
        try:
            plt.style.use(style)
            break
        except Exception:
            continue

    # Additional overrides for clean paper look
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "axes.facecolor":   "#fafafa",
        "figure.facecolor": "#ffffff",
        "axes.grid":        True,
        "grid.alpha":       0.4,
        "grid.linewidth":   0.4,
    })

    # 5 rows: Raw | Filtered | Background-sub | Temporal diff | PCA
    fig, axes = plt.subplots(5, 1, figsize=(16, 14),
                             gridspec_kw={'hspace': 0.55})

    # ── Title ─────────────────────────────────────────────────────────────
    rssi_info = f"RSSI: — dBm"  # will be overwritten if metadata available
    fig.suptitle(
        f"CSI Signal Processing Pipeline  ·  {file_path.name}\n"
        f"{n_frames} frames  ·  {n_active} active subcarriers  ·  "
        f"duration ≈{duration:.1f} s  ·  "
        f"packet loss {seq_stats.loss_percent:.2f}%  ·  "
        f"seq {seq_stats.first_seq}→{seq_stats.last_seq}",
        fontsize=12, fontweight='bold', y=0.98, color="#111111"
    )

    # Color palette — distinct colors for subcarriers / PCA components
    SC_COLORS  = plt.cm.tab10(np.linspace(0, 0.9, max(len(sc_indices), n_pca)))
    PCA_COLORS = ['#e63946', '#2a9d8f', '#e9c46a', '#457b9d', '#f4a261']

    # ════════════════════════════════════════════════════════════════════
    # PANEL 0 — Raw Amplitude (all active subcarriers, no filtering)
    # ════════════════════════════════════════════════════════════════════
    ax = axes[0]
    for i, sc in enumerate(sc_indices):
        ax.plot(t_full, amp_null[:, sc],
                color=SC_COLORS[i], linewidth=0.9, alpha=0.85,
                label=f"SC {sc}")
    if bg_enabled:
        annotate_bg_region(ax, bg_frames, args.fs)
    style_ax(ax,
             "① Raw Amplitude  (null bands removed, no temporal filtering)",
             "Amplitude (a.u.)")
    ax.legend(loc="upper right", fontsize=8, ncol=min(len(sc_indices), 5),
              framealpha=0.7)

    # ════════════════════════════════════════════════════════════════════
    # PANEL 1 — Filtered Amplitude (Hampel + Butterworth)
    # ════════════════════════════════════════════════════════════════════
    ax = axes[1]
    for i, sc in enumerate(sc_indices):
        ax.plot(t_full, amp_filt[:, sc],
                color=SC_COLORS[i], linewidth=1.2, alpha=0.9,
                label=f"SC {sc}")
    if bg_enabled:
        annotate_bg_region(ax, bg_frames, args.fs)
    style_ax(ax,
             f"② Hampel + Butterworth Low-Pass ({args.cutoff} Hz)  —  noise removed",
             "Amplitude (a.u.)")
    ax.legend(loc="upper right", fontsize=8, ncol=min(len(sc_indices), 5),
              framealpha=0.7)

    # ════════════════════════════════════════════════════════════════════
    # PANEL 2 — After Background Subtraction
    # Static room → ≈ 0 everywhere. Motion → visible deviations.
    # ════════════════════════════════════════════════════════════════════
    ax = axes[2]
    if bg_enabled:
        for i, sc in enumerate(sc_indices):
            ax.plot(t_full, amp_bg[:, sc],
                    color=SC_COLORS[i], linewidth=1.2, alpha=0.9,
                    label=f"SC {sc}")
        annotate_bg_region(ax, bg_frames, args.fs)
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
        style_ax(ax,
                 f"③ Background Subtraction  (first {bg_frames} frames = static room removed)",
                 "Δ Amplitude")
        ax.legend(loc="upper right", fontsize=8, ncol=min(len(sc_indices), 5),
                  framealpha=0.7)
    else:
        ax.text(0.5, 0.5, "Background subtraction DISABLED (--no-bg)",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color="#888888", style='italic')
        style_ax(ax, "③ Background Subtraction  [DISABLED]", "Δ Amplitude")

    # ════════════════════════════════════════════════════════════════════
    # PANEL 3 — Temporal Difference
    # Static room → ≈ 0. Motion event → sharp peaks.
    # Note: N-1 frames due to diff.
    # ════════════════════════════════════════════════════════════════════
    ax = axes[3]
    if diff_enabled:
        for i, sc in enumerate(sc_indices):
            ax.plot(t_diff, amp_diff[:, sc],
                    color=SC_COLORS[i], linewidth=1.0, alpha=0.85,
                    label=f"SC {sc}")
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
        style_ax(ax,
                 f"④ Temporal Difference  [frame(t+1) − frame(t)]  →  "
                 f"motion events visible  ({amp_diff.shape[0]} frames)",
                 "Δ Amplitude / frame")
        ax.legend(loc="upper right", fontsize=8, ncol=min(len(sc_indices), 5),
                  framealpha=0.7)
    else:
        ax.text(0.5, 0.5, "Temporal difference DISABLED (--no-diff)",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color="#888888", style='italic')
        style_ax(ax, "④ Temporal Difference  [DISABLED]", "Δ Amplitude / frame")

    # ════════════════════════════════════════════════════════════════════
    # PANEL 4 — PCA Components
    # Dimensionality: N_active_subcarriers → n_pca
    # This is what the AI model receives.
    # ════════════════════════════════════════════════════════════════════
    ax = axes[4]
    for i in range(n_pca):
        color = PCA_COLORS[i % len(PCA_COLORS)]
        label = (f"PC{i+1}  ({explained[i]:.1f}% var)")
        ax.plot(t_pca, pca_data[:, i],
                color=color, linewidth=1.3, alpha=0.9, label=label)
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    style_ax(ax,
             f"⑤ PCA  ({n_pca} components · {explained.sum():.1f}% variance explained)  "
             f"—  Final AI Input",
             "Component Value",
             show_xlabel=True)
    ax.legend(loc="upper right", fontsize=8, ncol=min(n_pca, 5),
              framealpha=0.8)

    # ════════════════════════════════════════════════════════════════════
    # SHARED LEGEND — background calibration annotation
    # ════════════════════════════════════════════════════════════════════
    if bg_enabled:
        bg_patch = mpatches.Patch(
            facecolor="#ffcc00", alpha=0.4,
            label=f"Background calibration period (first {bg_frames} frames = "
                  f"{bg_frames / args.fs:.1f} s  ·  empty room)"
        )
        fig.legend(handles=[bg_patch], loc="lower center",
                   fontsize=9, framealpha=0.9,
                   bbox_to_anchor=(0.5, 0.005))

    # ── Save / Show ───────────────────────────────────────────────────────
    if args.save:
        out_path = file_path.parent / (file_path.stem + "_lines.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight",
                    facecolor="#ffffff")
        print(f"\n💾 Saved: {out_path}")

    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)


if __name__ == "__main__":
    main()