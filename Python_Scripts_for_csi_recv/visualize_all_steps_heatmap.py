#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSI Step-by-Step Filter Visualization (Thesis Grade)
Visualizes each DSP stage of the preprocessing pipeline.

Usage:
  python visualize_all_steps.py                       # latest file in datasets/
  python visualize_all_steps.py path/to/file.txt
  python visualize_all_steps.py path/to/file.csv
  python visualize_all_steps.py file.txt --save       # save PNG alongside file
  python visualize_all_steps.py file.txt --unwrap-phase
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from csi_plotter_heatmap import load_csi_matrix, resolve_path, get_latest_dataset
from data_preprocessing import CSIPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="CSI Step-by-Step Filter Visualization"
    )
    parser.add_argument(
        "file", nargs="?", default=None,
        help="TXT or CSV file (default: latest in datasets/)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save output PNG next to the dataset file"
    )
    parser.add_argument(
        "--pca-components", type=int, default=10,
        help="Number of PCA components (default: 10)"
    )
    parser.add_argument(
        "--cutoff", type=float, default=12.0,
        help="Butterworth cutoff frequency in Hz (default: 12)"
    )
    parser.add_argument(
        "--background-frames", type=int, default=100,
        help="Frames for background estimation (default: 100 = 1 s). Set 0 to disable."
    )
    parser.add_argument(
        "--no-diff", action="store_true",
        help="Disable temporal difference step"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Εύρεση αρχείου ───────────────────────────────────────────────────
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            sys.exit(1)
    else:
        default_dir = resolve_path("datasets")
        if not default_dir.exists():
            print(f"❌ Directory not found: {default_dir}")
            print("   Pass a file: python visualize_all_steps.py file.txt")
            sys.exit(1)
        file_path = get_latest_dataset(default_dir)
        if file_path is None:
            print(f"❌ No TXT/CSV files in {default_dir}")
            sys.exit(1)

    print(f"\n📂 Loading: {file_path.name}")

    # ── Φόρτωση ──────────────────────────────────────────────────────────
    complex_matrix, dropped_frames, seq_stats = load_csi_matrix(file_path)

    if complex_matrix.size == 0:
        print("❌ No valid frames found.")
        sys.exit(1)

    n_frames, n_sub = complex_matrix.shape
    print(f"   Frames: {n_frames} | Subcarriers: {n_sub} | "
          f"Dropped: {dropped_frames} | Loss: {seq_stats.loss_percent:.2f}%")

    # ── Pipeline step-by-step ─────────────────────────────────────────────
    # We call each method manually to capture intermediate results for plotting.
    # This mirrors exactly what CSIPipeline.fit_transform() does internally.

    pipeline = CSIPipeline(
        fs=100.0,
        background_frames=args.background_frames,
        use_diff=not args.no_diff,
    )

    # Step 0: Raw amplitude (με nulls — πριν από οτιδήποτε)
    amp_step0 = np.abs(complex_matrix)

    # Step 1: Null subcarrier removal
    amp_step1 = pipeline.remove_null_subcarriers(complex_matrix, fit=True)

    # Step 2: Hampel filter
    amp_step2 = pipeline.apply_hampel_filter(amp_step1, window_size=11, n_sigmas=3.0)

    # Step 3: Butterworth low-pass
    amp_step3 = pipeline.apply_lowpass_filter(amp_step2, cutoff=args.cutoff)

    # Step 4: Background subtraction
    amp_step4 = pipeline.apply_background_subtraction(amp_step3, fit=True)

    # Step 5: Temporal difference
    amp_step5 = pipeline.apply_temporal_diff(amp_step4)

    # Step 6: PCA
    n_components = min(args.pca_components,
                       amp_step5.shape[0] - 1,
                       amp_step5.shape[1])
    pca = PCA(n_components=n_components)
    amp_step6 = pca.fit_transform(amp_step5)
    explained_var = pca.explained_variance_ratio_.sum() * 100

    # Step 7: Standard scaler (Z-score)
    scaler = StandardScaler()
    amp_step7 = scaler.fit_transform(amp_step6)

    # ── Stats ─────────────────────────────────────────────────────────────
    active_count   = amp_step1.shape[1]
    null_count     = n_sub - active_count
    bg_enabled     = args.background_frames > 0
    diff_enabled   = not args.no_diff
    diff_info      = f"N→{amp_step5.shape[0]} frames" if diff_enabled else "disabled"

    print(f"\n📊 Pipeline stats:")
    print(f"   [0] Raw:            {amp_step0.shape}  "
          f"range [{amp_step0.min():.1f}, {amp_step0.max():.1f}]")
    print(f"   [1] Null removed:   {amp_step1.shape}  "
          f"({null_count} null bands removed)")
    print(f"   [2] Hampel:         {amp_step2.shape}")
    print(f"   [3] Butterworth:    {amp_step3.shape}  "
          f"(cutoff={args.cutoff} Hz)")
    print(f"   [4] Background sub: {amp_step4.shape}  "
          f"({'bg=' + str(args.background_frames) + ' frames' if bg_enabled else 'DISABLED'})")
    print(f"   [5] Temporal diff:  {amp_step5.shape}  ({diff_info})")
    print(f"   [6] PCA:            {amp_step6.shape}  "
          f"({explained_var:.1f}% variance)")
    print(f"   [7] StandardScaler: {amp_step7.shape}  "
          f"mean={amp_step7.mean():.3f} std={amp_step7.std():.3f}")

    # ════════════════════════════════════════════════════════════════════
    # 8-PANEL VISUALIZATION (2 rows × 4 columns)
    # Rules for correct colormaps:
    #   - Amplitude data (steps 0-3):   'jet'     — non-negative, no center
    #   - Deviation data (steps 4-5):   'RdBu_r'  — ONLY when step is active
    #     (centered at 0: negative=blue, zero=white, positive=red)
    #     If step is DISABLED, the data is still amplitude → use 'jet'
    #   - Reduced data (steps 6-7):     'viridis' — can be negative after PCA
    # ════════════════════════════════════════════════════════════════════

    # Determine actual colormaps based on whether steps are enabled
    step4_title = (
        f"4. Background Subtraction\n(≈{args.background_frames / 100:.1f}s static room removed)"
        if bg_enabled else
        "4. Background Subtraction\n⚠ DISABLED"
    )
    step4_cmap = "RdBu_r" if bg_enabled else "jet"

    step5_title = (
        f"5. Temporal Difference\n(Rate of change  {amp_step5.shape[0]} frames)"
        if diff_enabled else
        "5. Temporal Difference\n⚠ DISABLED"
    )
    step5_cmap = "RdBu_r" if diff_enabled else ("RdBu_r" if bg_enabled else "jet")

    plots_config = [
        (amp_step0, "0. Raw Amplitude\n(with guard/null bands)",          "jet"),
        (amp_step1, "1. Null Bands Removed\n(active subcarriers only)",   "jet"),
        (amp_step2, "2. Hampel Filter\n(spike / outlier removal)",        "jet"),
        (amp_step3, f"3. Butterworth Low-Pass\n({args.cutoff} Hz cutoff)", "jet"),
        (amp_step4, step4_title,                                           step4_cmap),
        (amp_step5, step5_title,                                           step5_cmap),
        (amp_step6, f"6. PCA\n({n_components} components · {explained_var:.0f}% variance)", "viridis"),
        (amp_step7, "7. StandardScaler (Z-score)\nFinal AI Input",         "viridis"),
    ]

    # ── Figure setup ─────────────────────────────────────────────────────
    STYLE_BG    = "#1a1a2e"
    STYLE_PANEL = "#16213e"
    STYLE_TEXT  = "#e0e0e0"
    STYLE_GRID  = "#2a2a4a"

    plt.rcParams.update({
        "figure.facecolor":  STYLE_BG,
        "axes.facecolor":    STYLE_PANEL,
        "axes.edgecolor":    STYLE_GRID,
        "axes.labelcolor":   STYLE_TEXT,
        "xtick.color":       STYLE_TEXT,
        "ytick.color":       STYLE_TEXT,
        "text.color":        STYLE_TEXT,
        "grid.color":        STYLE_GRID,
    })

    fig, axes = plt.subplots(2, 4, figsize=(24, 9))
    fig.patch.set_facecolor(STYLE_BG)

    fig.suptitle(
        f"CSI Preprocessing Pipeline  ·  {file_path.name}\n"
        f"{n_frames} frames  ·  {n_sub} subcarriers ({active_count} active)  ·  "
        f"packet loss {seq_stats.loss_percent:.2f}%",
        fontsize=13, fontweight='bold', color=STYLE_TEXT, y=0.98
    )

    axes_flat = axes.flatten()

    for ax, (data, title, cmap) in zip(axes_flat, plots_config):
        ax.set_facecolor(STYLE_PANEL)

        # ── Robust color limits (percentile clipping) ─────────────────
        vmin = np.percentile(data, 2)
        vmax = np.percentile(data, 98)

        # ✅ FIX: diverging colormap ONLY when data is actually centered at 0
        # (i.e., when background subtraction or temporal diff is active)
        if cmap == "RdBu_r":
            abs_max = max(abs(float(vmin)), abs(float(vmax)))
            if abs_max == 0:
                abs_max = 1.0   # prevent vmin == vmax == 0 → blank plot
            vmin, vmax = -abs_max, abs_max

        im = ax.imshow(
            data.T,
            aspect="auto",
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest"
        )

        # ── Title with step number highlighted ────────────────────────
        step_num = title.split(".")[0]
        ax.set_title(title, fontsize=9, pad=5,
                     color=STYLE_TEXT, fontweight='normal')

        ax.set_xlabel("Time (Frame Index)", fontsize=8, color=STYLE_TEXT)
        ax.tick_params(labelsize=7)

        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=7, colors=STYLE_TEXT)
        cb.outline.set_edgecolor(STYLE_GRID)

    # Y-axis labels for leftmost column only
    axes_flat[0].set_ylabel("Subcarrier Index", fontsize=8, color=STYLE_TEXT)
    axes_flat[4].set_ylabel("Subcarrier / Component", fontsize=8, color=STYLE_TEXT)

    # ── Vertical separator between raw (left 4) and processed (right 4) ─
    # Actually panels are: 0-3 = amplitude, 4-7 = processed
    # Add subtle divider between row 0 col 3 and row 1 col 0 area

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if args.save:
        out_path = file_path.parent / (file_path.stem + "_pipeline.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor=STYLE_BG)
        print(f"\n💾 Saved: {out_path}")

    # Reset rcParams after show to avoid affecting other plots
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)


if __name__ == "__main__":
    main()