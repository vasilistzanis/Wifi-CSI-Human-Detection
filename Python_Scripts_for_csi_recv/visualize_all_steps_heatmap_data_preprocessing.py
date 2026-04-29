#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
CSI Step-by-Step Filter Visualization (Thesis Grade - Improved)
Visualizes each DSP stage of the preprocessing pipeline in SEPARATE WINDOWS.


Improvements:
  - Better error handling for imports and file loading
  - Validation of intermediate results
  - Graceful matplotlib backend fallback
  - More informative error messages
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
    print("[WARNING]  Qt5Agg backend not available, falling back to TkAgg")
    try:
        matplotlib.use("TkAgg")
    except Exception:
        print("[WARNING]  TkAgg backend not available, using default")
        pass


import matplotlib.pyplot as plt
plt.ioff()  # Disable interactive mode for faster background rendering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Improved: Better import error handling
try:
    from csi_parser import load_csi_matrix, resolve_path, get_latest_dataset
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("   Make sure csi_parser.py is in the same directory")
    sys.exit(1)


try:
    from data_preprocessing import CSIPipeline
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("   Make sure data_preprocessing.py is in the same directory")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CSI Step-by-Step Filter Visualization (Separate Windows)"
    )
    parser.add_argument(
        "file", nargs="?", default=None,
        help="TXT or CSV file (default: latest in datasets/)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save output PNG next to the dataset file (Will save 7 files!)"
    )
    parser.add_argument(
        "--pca-components", type=int, default=10,
        help="Number of PCA components (default: 10)"
    )
    parser.add_argument(
        "--cutoff", type=float, default=10.0,
        help="Butterworth cutoff frequency in Hz (default: 10)"
    )
    parser.add_argument(
        "--no-diff", action="store_true",
        help="Disable temporal difference step"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # -- File discovery -------------------------------------------------------
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"[ERROR] File not found: {file_path}")
            sys.exit(1)
    else:
        default_dir = resolve_path("datasets")
        if not default_dir.exists():
            print(f"[ERROR] Directory not found: {default_dir}")
            print("   Create a 'datasets/' directory or specify a file with: python script.py <file.txt>")
            sys.exit(1)
        file_path = get_latest_dataset(default_dir)
        if file_path is None:
            print(f"[ERROR] No TXT/CSV files in {default_dir}")
            print("   Run csi_logger.py first to capture data")
            sys.exit(1)

    print(f"\n[FILE] Loading: {file_path.name}")

    # -- Data Loading ----------------------------------------------------------
    try:
        complex_matrix, dropped_frames, seq_stats = load_csi_matrix(file_path)
    except (FileNotFoundError, PermissionError, ValueError) as e:
        print(f"[ERROR] Error loading file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error loading file: {e}")
        sys.exit(1)

    n_frames, n_sub = complex_matrix.shape
    print(f"   Frames: {n_frames} | Subcarriers: {n_sub} | "
          f"Dropped: {dropped_frames} | Loss: {seq_stats.loss_percent:.2f}%")

    # -- Pipeline step-by-step ---------------------------------------------
    pipeline = CSIPipeline(
        fs=100.0,
        use_diff=not args.no_diff,
    )

    # Improved: Validate each step
    try:
        amp_step0 = np.abs(complex_matrix)
        amp_step1 = pipeline.remove_null_subcarriers(complex_matrix, fit=True)
        
        # Check if we have any active subcarriers
        if amp_step1.shape[1] == 0:
            print("[ERROR] No active subcarriers after null removal!")
            print("   All subcarriers appear to be zero. Check your ESP32 configuration.")
            sys.exit(1)
        
        amp_step2 = pipeline.apply_hampel_filter(amp_step1, window_size=11, n_sigmas=3.0)
        amp_step3 = pipeline.apply_lowpass_filter(amp_step2, cutoff=args.cutoff)
        amp_step4 = pipeline.apply_temporal_diff(amp_step3)

        # Validate PCA inputs
        if amp_step4.shape[0] < 2:
            print(f"[ERROR] Too few frames ({amp_step4.shape[0]}) for PCA after temporal diff")
            print("   Need at least 2 frames. Try capturing more data.")
            sys.exit(1)

        n_components = min(args.pca_components, amp_step4.shape[0] - 1, amp_step4.shape[1])
        if n_components < 1:
            print(f"[ERROR] Cannot perform PCA: shape {amp_step4.shape} too small")
            sys.exit(1)

        pca = PCA(n_components=n_components)
        amp_step5 = pca.fit_transform(amp_step4)
        explained_var = pca.explained_variance_ratio_.sum() * 100

        scaler = StandardScaler()
        amp_step6 = scaler.fit_transform(amp_step5)

    except Exception as e:
        print(f"[ERROR] Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # -- Stats -------------------------------------------------------------
    active_count   = amp_step1.shape[1]
    null_count     = n_sub - active_count
    diff_enabled   = not args.no_diff
    print(f"\n[STATS] Pipeline results:")
    print(f"   [0] Raw:            {amp_step0.shape}")
    print(f"   [1] Null removed:   {amp_step1.shape} ({null_count} nulls)")
    print(f"   [2] Hampel:         {amp_step2.shape}")
    print(f"   [3] Butterworth:    {amp_step3.shape}")
    print(f"   [4] Temporal diff:  {amp_step4.shape}")
    print(f"   [5] PCA:            {amp_step5.shape} ({explained_var:.1f}% variance)")
    print(f"   [6] StandardScaler: {amp_step6.shape}")

    # ====================================================================
    # VISUALIZATION
    # ====================================================================

    step4_title = (
        f"4. Temporal Difference\n(Rate of change  {amp_step4.shape[0]} frames)"
        if diff_enabled else "4. Temporal Difference\n[WARNING] DISABLED"
    )
    step4_cmap = "RdBu_r" if diff_enabled else "jet"

    plots_config = [
        (amp_step0, "0. Raw Amplitude\n(with guard/null bands)",          "jet"),
        (amp_step1, "1. Null Bands Removed\n(active subcarriers only)",   "jet"),
        (amp_step2, "2. Hampel Filter\n(spike / outlier removal)",        "jet"),
        (amp_step3, f"3. Butterworth Low-Pass\n({args.cutoff} Hz cutoff)", "jet"),
        (amp_step4, step4_title,                                           step4_cmap),
        (amp_step5, f"5. PCA\n({n_components} components - {explained_var:.0f}% variance)", "viridis"),
        (amp_step6, "6. StandardScaler (Z-score)\nFinal AI Input",         "viridis"),
    ]

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

    global_suptitle = (
        f"CSI Preprocessing Pipeline - {file_path.name}\n"
        f"{n_frames} frames - {n_sub} subcarriers ({active_count} active) - "
        f"packet loss {seq_stats.loss_percent:.2f}%"
    )

    # Create a separate window (figure) for each plot
    try:
        for step_num, (data, title, cmap) in enumerate(plots_config):
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor(STYLE_BG)
            ax.set_facecolor(STYLE_PANEL)

            fig.suptitle(global_suptitle, fontsize=11, fontweight='bold', 
                        color=STYLE_TEXT, y=0.96)

            vmin = np.percentile(data, 2)
            vmax = np.percentile(data, 98)

            if cmap == "RdBu_r":
                abs_max = max(abs(float(vmin)), abs(float(vmax)))
                if abs_max == 0:
                    abs_max = 1.0
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

            ax.set_title(title, fontsize=11, pad=10, color=STYLE_TEXT, fontweight='normal')
            ax.set_xlabel("Time (Frame Index)", fontsize=9, color=STYLE_TEXT)
            
            if "PCA" in title or "StandardScaler" in title:
                ax.set_ylabel("Component Index", fontsize=9, color=STYLE_TEXT)
            else:
                ax.set_ylabel("Subcarrier Index", fontsize=9, color=STYLE_TEXT)

            ax.tick_params(labelsize=8)

            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.ax.tick_params(labelsize=8, colors=STYLE_TEXT)
            cb.outline.set_edgecolor(STYLE_GRID)

            plt.tight_layout(rect=[0, 0, 1, 0.90])

            if args.save:
                out_path = file_path.parent / f"{file_path.stem}_step{step_num}.png"
                try:
                    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=STYLE_BG)
                    print(f"[FILE] Saved: {out_path}")
                except (PermissionError, OSError) as e:
                    print(f"[WARNING]  Could not save {out_path}: {e}")

    except Exception as e:
        print(f"[ERROR] Error during plotting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n[OK] Created 7 separate windows! (Close them all to end the script)")
    
    try:
        plt.show()
    except Exception as e:
        print(f"[WARNING]  Error displaying plots: {e}")
    
    plt.rcParams.update(plt.rcParamsDefault)

if __name__ == "__main__":
    main()
