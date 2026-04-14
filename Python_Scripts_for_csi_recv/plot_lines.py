#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSI Line Plotter — Thesis / Paper Grade (Improved)
Visualizes the full signal processing pipeline as 2D line plots.
Shows 5 stages in SEPARATE WINDOWS so the reader can see exactly what each
processing step does to the CSI signal. Designed for thesis figures.

Improvements:
  - Better color array handling for large numbers of subcarriers
  - Better error handling for imports and file loading
  - Validation of intermediate results
  - More informative error messages

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
    print("⚠️  Qt5Agg backend not available, falling back to TkAgg")
    try:
        matplotlib.use("TkAgg")
    except Exception:
        print("⚠️  TkAgg backend not available, using default")
        pass

import matplotlib.pyplot as plt
plt.ioff()  # Disable interactive mode for faster background rendering
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

# ✅ IMPROVED: Better import error handling
try:
    from csi_plotter_heatmap import load_csi_matrix, resolve_path, get_latest_dataset
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   Make sure csi_plotter_heatmap.py is in the same directory")
    sys.exit(1)

try:
    from data_preprocessing import CSIPipeline
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   Make sure data_preprocessing.py is in the same directory")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════
# ARGS
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="CSI Line Plotter — Thesis / Paper Grade (Separate Windows)"
    )
    p.add_argument(
        "file", nargs="?", default=None,
        help="TXT or CSV file (default: latest in datasets/)"
    )
    p.add_argument(
        "--save", action="store_true",
        help="Save figure as PNG next to the dataset file (creates 5 files)"
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
    """Create time axis in seconds."""
    return np.arange(n_frames) / fs





def style_ax(ax, title: str, ylabel: str, show_xlabel: bool = True):
    """Apply consistent styling to a single axis."""
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8,
                 color="#222222")
    ax.set_ylabel(ylabel, fontsize=10, color="#333333")
    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=10, color="#333333")
    ax.tick_params(labelsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5, linestyle='-')
    ax.spines[['top', 'right']].set_visible(False)


def get_color_palette(n_colors: int):
    """
    Get appropriate color palette for n_colors.
    
    Returns:
      numpy array of RGB colors, shape (n_colors, 3 or 4)
    """
    # ✅ IMPROVED: Better color handling for large n
    if n_colors <= 10:
        return plt.cm.tab10(np.linspace(0, 0.9, n_colors))
    elif n_colors <= 20:
        return plt.cm.tab20(np.linspace(0, 0.95, n_colors))
    else:
        # Use continuous colormap for very large numbers
        return plt.cm.viridis(np.linspace(0, 0.95, n_colors))


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
            print("   Use: python plot_lines.py <file.txt>")
            sys.exit(1)
        file_path = get_latest_dataset(default_dir)
        if file_path is None:
            print(f"❌ No TXT/CSV files found in {default_dir}")
            print("   Run csi_logger.py first to capture data")
            sys.exit(1)

    print(f"\n📂 Loading: {file_path.name}")
    
    # ✅ IMPROVED: Better error handling
    try:
        complex_matrix, _, seq_stats = load_csi_matrix(file_path)
    except (FileNotFoundError, PermissionError, ValueError) as e:
        print(f"❌ Error loading file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    n_frames, n_sub = complex_matrix.shape
    print(f"   {n_frames} frames × {n_sub} subcarriers | "
          f"loss={seq_stats.loss_percent:.2f}% | "
          f"seq {seq_stats.first_seq}→{seq_stats.last_seq}")

    # ── Pipeline — step by step to capture every stage ───────────────────
    pipeline = CSIPipeline(
        fs=args.fs,
        use_diff=not args.no_diff,
    )

    # ✅ IMPROVED: Validate each step
    try:
        # Stage 0: Raw amplitude (all subcarriers including nulls)
        # Stage 1: Null removal + Hampel + Butterworth
        amp_null = pipeline.remove_null_subcarriers(complex_matrix, fit=True)
        
        # ✅ NEW: Validate active subcarriers
        if amp_null.shape[1] == 0:
            print("❌ No active subcarriers after null removal!")
            sys.exit(1)
        
        amp_hamp = pipeline.apply_hampel_filter(amp_null)
        amp_filt = pipeline.apply_lowpass_filter(amp_hamp, cutoff=args.cutoff)

        # Stage 2: Temporal diff
        amp_diff = pipeline.apply_temporal_diff(amp_filt)
        diff_enabled = not args.no_diff and amp_diff.shape[0] < amp_filt.shape[0]

        # ✅ NEW: Validate PCA inputs
        if amp_diff.shape[0] < 2:
            print(f"❌ Too few frames ({amp_diff.shape[0]}) for PCA")
            sys.exit(1)

        # Stage 4: PCA
        n_pca = min(args.pca_components, amp_diff.shape[0] - 1, amp_diff.shape[1])
        if n_pca < 1:
            print(f"❌ Cannot perform PCA: shape {amp_diff.shape} too small")
            sys.exit(1)

        pca = PCA(n_components=n_pca)
        pca_data = pca.fit_transform(amp_diff)
        explained = pca.explained_variance_ratio_ * 100

    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

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

    # ── Global Title & Color Helpers ──────────────────────────────────────
    global_suptitle = (
        f"CSI Signal Processing Pipeline  ·  {file_path.name}\n"
        f"{n_frames} frames  ·  {n_active} active subcarriers  ·  "
        f"duration ≈{duration:.1f} s  ·  "
        f"packet loss {seq_stats.loss_percent:.2f}%"
    )

    # ✅ IMPROVED: Better color palette selection
    SC_COLORS  = get_color_palette(max(len(sc_indices), n_pca))
    PCA_COLORS = ['#e63946', '#2a9d8f', '#e9c46a', '#457b9d', '#f4a261']

    def create_window():
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(global_suptitle, fontsize=12, fontweight='bold', 
                    y=0.96, color="#111111")
        return fig, ax

    def add_shared_legend(fig):
        pass # Removed background subtraction legend

    # ════════════════════════════════════════════════════════════════════
    # CREATE PLOTS WITH ERROR HANDLING
    # ════════════════════════════════════════════════════════════════════
    
    try:
        # ── PANEL 0 — Raw Amplitude (Window 1) ────────────────────────────
        fig0, ax0 = create_window()
        for i, sc in enumerate(sc_indices):
            ax0.plot(t_full, amp_null[:, sc],
                    color=SC_COLORS[i], linewidth=1.2, alpha=0.85,
                    label=f"SC {sc}")


        style_ax(ax0,
                 "① Raw Amplitude  (null bands removed, no temporal filtering)",
                 "Amplitude (a.u.)")
        ax0.legend(loc="upper right", fontsize=9, ncol=min(len(sc_indices), 5), framealpha=0.7)
        fig0.tight_layout(rect=[0, 0.05, 1, 0.92])
        
        if args.save:
            out_path = file_path.parent / f"{file_path.stem}_line_0.png"
            fig0.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"💾 Saved: {out_path}")

        # ── PANEL 1 — Filtered Amplitude (Window 2) ───────────────────────
        fig1, ax1 = create_window()
        for i, sc in enumerate(sc_indices):
            ax1.plot(t_full, amp_filt[:, sc],
                    color=SC_COLORS[i], linewidth=1.5, alpha=0.9,
                    label=f"SC {sc}")
        style_ax(ax1,
                 f"② Hampel + Butterworth Low-Pass ({args.cutoff} Hz)  —  noise removed",
                 "Amplitude (a.u.)")
        ax1.legend(loc="upper right", fontsize=9, ncol=min(len(sc_indices), 5), framealpha=0.7)
        fig1.tight_layout(rect=[0, 0.05, 1, 0.92])

        if args.save:
            out_path = file_path.parent / f"{file_path.stem}_line_1.png"
            fig1.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"💾 Saved: {out_path}")

        # ── PANEL 2 — Temporal Difference (Window 3) ──────────────────────
        fig2, ax2 = create_window()
        if diff_enabled:
            for i, sc in enumerate(sc_indices):
                ax2.plot(t_diff, amp_diff[:, sc],
                        color=SC_COLORS[i], linewidth=1.2, alpha=0.85,
                        label=f"SC {sc}")
            ax2.axhline(0, color="#999999", linewidth=1.0, linestyle="--")
            style_ax(ax2,
                     f"③ Temporal Difference  [frame(t+1) − frame(t)]  →  "
                     f"motion events visible  ({amp_diff.shape[0]} frames)",
                     "Δ Amplitude / frame")
            ax2.legend(loc="upper right", fontsize=9, ncol=min(len(sc_indices), 5), framealpha=0.7)
        else:
            ax2.text(0.5, 0.5, "Temporal difference DISABLED (--no-diff)",
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=13, color="#888888", style='italic')
            style_ax(ax2, "③ Temporal Difference  [DISABLED]", "Δ Amplitude / frame")
        
        fig2.tight_layout(rect=[0, 0.05, 1, 0.92])

        if args.save:
            out_path = file_path.parent / f"{file_path.stem}_line_2.png"
            fig2.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"💾 Saved: {out_path}")

        # ── PANEL 3 — PCA Components (Window 4) ───────────────────────────
        fig3, ax3 = create_window()
        for i in range(n_pca):
            color = PCA_COLORS[i % len(PCA_COLORS)]
            label = (f"PC{i+1}  ({explained[i]:.1f}% var)")
            ax3.plot(t_pca, pca_data[:, i],
                    color=color, linewidth=1.5, alpha=0.9, label=label)
        ax3.axhline(0, color="#999999", linewidth=1.0, linestyle="--")
        style_ax(ax3,
                 f"④ PCA  ({n_pca} components · {explained.sum():.1f}% variance explained)  "
                 f"—  Final AI Input",
                 "Component Value")
        ax3.legend(loc="upper right", fontsize=9, ncol=min(n_pca, 5), framealpha=0.8)
        
        fig3.tight_layout(rect=[0, 0.05, 1, 0.92])

        if args.save:
            out_path = file_path.parent / f"{file_path.stem}_line_3.png"
            fig3.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"💾 Saved: {out_path}")

    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        print("   Try installing: pip install matplotlib python-tk")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ Created 4 separate windows! (Close them all to end the script)")
    
    # ── Show All Windows ──────────────────────────────────────────────────
    try:
        plt.show()
    except Exception as e:
        print(f"⚠️  Error displaying plots: {e}")
        print("   Plots were created but may not display properly.")
    
    plt.rcParams.update(plt.rcParamsDefault)


if __name__ == "__main__":
    main()
