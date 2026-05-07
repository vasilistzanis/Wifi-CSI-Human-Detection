#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
CSI Line Plotter - Thesis / Paper Grade
Visualizes the full signal processing pipeline as 2D line plots.
Shows all 7 stages (0-6) in SEPARATE WINDOWS, matching exactly the
6-step CSIPipeline defined in data_preprocessing.py.


Pipeline steps visualized:
  0. Raw Amplitude          (np.abs, all subcarriers incl. nulls)
  1. Null Subcarrier Removal
  2. Hampel Filter
  3. Butterworth Low-Pass
  4. Temporal Difference
  5. PCA
  6. StandardScaler         - Final AI Input


Usage:
  python plot_lines_data_preprocessing.py                          # latest file in datasets/
  python plot_lines_data_preprocessing.py path/to/file.txt
  python plot_lines_data_preprocessing.py file.txt --save
  python plot_lines_data_preprocessing.py file.txt --n-subcarriers 6 --pca-components 5
  python plot_lines_data_preprocessing.py file.txt --no-diff
"""


import sys
import argparse
from pathlib import Path


import numpy as np
import matplotlib


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler




from csi_parser import configure_console_output
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
plt.ioff()


try:
    import config
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




# ========================================================================
# ARGS
# ========================================================================


def parse_args():
    defaults = config.get_script_defaults("plot_lines_data_preprocessing")
    p = argparse.ArgumentParser(
        description="CSI Line Plotter - Thesis / Paper Grade (7 Separate Windows)"
    )
    p.add_argument(
        "file", nargs="?", default=defaults["file"],
        help="TXT or CSV file (default: latest in datasets/)"
    )
    config.add_bool_argument(
        p,
        dest="save",
        default=defaults["save"],
        help="Save figures as PNG next to the dataset file (creates 7 files)",
        positive_flags=["--save"],
        negative_flags=["--no-save"],
    )
    p.add_argument(
        "--n-subcarriers", type=int, default=defaults["n_subcarriers"],
        help=f"Number of subcarriers to overlay (default: {config.MAX_SUBCARRIERS})"
    )
    p.add_argument(
        "--pca-components", type=int, default=defaults["pca_components"],
        help=f"Number of PCA components to show (default: {config.N_PCA_COMPONENTS})"
    )
    p.add_argument(
        "--cutoff", type=float, default=defaults["cutoff"],
        help="Butterworth cutoff in Hz (default: 10)"
    )
    config.add_bool_argument(
        p,
        dest="use_diff",
        default=defaults["use_diff"],
        help="Enable temporal difference",
        positive_flags=["--diff"],
        negative_flags=["--no-diff"],
    )
    p.add_argument(
        "--fs", type=float, default=defaults["fs"],
        help=f"Sampling frequency in Hz (default: {config.SAMPLING_RATE})"
    )
    return p.parse_args()




# ========================================================================
# HELPERS
# ========================================================================


def select_subcarriers(n_active: int, n_wanted: int) -> list[int]:
    """Select n_wanted subcarrier indices spread evenly across active spectrum."""
    if n_wanted >= n_active:
        return list(range(n_active))
    margin = max(1, n_active // (n_wanted * 2))
    indices = np.linspace(margin, n_active - 1 - margin, n_wanted, dtype=int)
    return list(dict.fromkeys(indices.tolist()))




def make_time_axis(n_frames: int, fs: float) -> np.ndarray:
    """Create time axis in seconds."""
    return np.arange(n_frames) / fs




def style_ax(ax, title: str, ylabel: str):
    """Apply consistent styling to a single axis."""
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8, color="#222222")
    ax.set_ylabel(ylabel, fontsize=10, color="#333333")
    ax.set_xlabel("Time (s)", fontsize=10, color="#333333")
    ax.tick_params(labelsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5, linestyle='-')
    ax.spines[['top', 'right']].set_visible(False)




def get_color_palette(n_colors: int):
    """Get appropriate color palette for n_colors."""
    if n_colors <= 10:
        return plt.cm.tab10(np.linspace(0, 0.9, n_colors))
    elif n_colors <= 20:
        return plt.cm.tab20(np.linspace(0, 0.95, n_colors))
    else:
        return plt.cm.viridis(np.linspace(0, 0.95, n_colors))




# ========================================================================
# MAIN
# ========================================================================


def main():
    args = parse_args()


    # -- File resolution ---------------------------------------------------
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"[ERROR] File not found: {file_path}")
            sys.exit(1)
    else:
        default_dir = resolve_path(config.DATASETS_DIR)
        if not default_dir.exists():
            print(f"[ERROR] datasets/ directory not found - pass a file explicitly")
            print("   Use: python plot_lines_data_preprocessing.py <file.txt>")
            sys.exit(1)
        file_path = get_latest_dataset(default_dir)
        if file_path is None:
            print(f"[ERROR] No TXT/CSV files found in {default_dir}")
            print("   Run csi_logger.py first to capture data")
            sys.exit(1)


    print(f"\n[FILE] Loading: {file_path.name}")


    try:
        complex_matrix, _, seq_stats = load_csi_matrix(file_path)
    except (FileNotFoundError, PermissionError, ValueError) as e:
        print(f"[ERROR] Error loading file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)


    n_frames, n_sub = complex_matrix.shape
    print(f"   {n_frames} frames x {n_sub} subcarriers | "
          f"loss={seq_stats.loss_percent:.2f}% | "
          f"seq {seq_stats.first_seq}->{seq_stats.last_seq}")


    # -- Pipeline - step by step (mirrors CSIPipeline.fit_transform) -------
    pipeline = CSIPipeline(
        fs=args.fs,
        use_diff=args.use_diff,
    )


    try:
        # [0] Raw amplitude (before any processing, includes null bands)
        amp_step0 = np.abs(complex_matrix)


        # [1] Null subcarrier removal
        amp_step1 = pipeline.remove_null_subcarriers(complex_matrix, fit=True)


        if amp_step1.shape[1] == 0:
            print("[ERROR] No active subcarriers after null removal!")
            print("   All subcarriers appear to be zero. Check your ESP32 configuration.")
            sys.exit(1)


        # [2] Hampel filter (outlier/spike removal)
        amp_step2 = pipeline.apply_hampel_filter(amp_step1, window_size=11, n_sigmas=3.0)


        # [3] Butterworth low-pass filter
        amp_step3 = pipeline.apply_lowpass_filter(amp_step2, cutoff=args.cutoff)


        # [4] Temporal difference
        amp_step4 = pipeline.apply_temporal_diff(amp_step3)
        diff_enabled = args.use_diff


        # [5] PCA
        if amp_step4.shape[0] < 2:
            print(f"[ERROR] Too few frames ({amp_step4.shape[0]}) for PCA after temporal diff")
            sys.exit(1)


        n_pca = min(args.pca_components, amp_step4.shape[0] - 1, amp_step4.shape[1])
        if n_pca < 1:
            print(f"[ERROR] Cannot perform PCA: shape {amp_step4.shape} too small")
            sys.exit(1)


        pca = PCA(n_components=n_pca)
        amp_step5 = pca.fit_transform(amp_step4)
        explained = pca.explained_variance_ratio_ * 100
        explained_total = explained.sum()


        # [6] StandardScaler (Z-score) - Final AI Input
        scaler = StandardScaler()
        amp_step6 = scaler.fit_transform(amp_step5)


    except Exception as e:
        print(f"[ERROR] Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # -- Stats -------------------------------------------------------------
    n_active   = amp_step1.shape[1]
    null_count = n_sub - n_active
    sc_indices = select_subcarriers(n_active, args.n_subcarriers)


    print(f"\n[INFO] Pipeline stats:")
    print(f"   [0] Raw:            {amp_step0.shape}")
    print(f"   [1] Null removed:   {amp_step1.shape} ({null_count} nulls)")
    print(f"   [2] Hampel:         {amp_step2.shape}")
    print(f"   [3] Butterworth:    {amp_step3.shape}")
    print(f"   [4] Temporal diff:  {amp_step4.shape}")
    print(f"   [5] PCA:            {amp_step5.shape} ({explained_total:.1f}% variance)")
    print(f"   [6] StandardScaler: {amp_step6.shape}")
    print(f"   Plotting {len(sc_indices)} subcarriers: {sc_indices}")


    # -- Time axes ---------------------------------------------------------
    # Steps 0-3 share the same frame count (no frames lost yet)
    t_full = make_time_axis(amp_step3.shape[0], args.fs)
    # Steps 4-6: temporal diff removes 1 frame
    t_diff = make_time_axis(amp_step4.shape[0], args.fs)
    duration = t_full[-1]


    # ====================================================================
    # FIGURE SETUP
    # ====================================================================


    for style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot']:
        try:
            plt.style.use(style)
            break
        except Exception:
            continue


    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "axes.facecolor":   "#fafafa",
        "figure.facecolor": "#ffffff",
        "axes.grid":        True,
        "grid.alpha":       0.4,
        "grid.linewidth":   0.4,
    })


    global_suptitle = (
        f"CSI Signal Processing Pipeline  -  {file_path.name}\n"
        f"{n_frames} frames  -  {n_active} active subcarriers  -  "
        f"duration ~{duration:.1f} s  -  "
        f"packet loss {seq_stats.loss_percent:.2f}%"
    )


    raw_indices = select_subcarriers(n_sub, args.n_subcarriers)
    SC_COLORS  = get_color_palette(max(len(raw_indices), len(sc_indices), n_pca))
    PCA_COLORS = ['#e63946', '#2a9d8f', '#e9c46a', '#457b9d', '#f4a261']


    def create_window():
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(global_suptitle, fontsize=12, fontweight='bold',
                     y=0.96, color="#111111")
        return fig, ax


    def save_fig(fig, step_num: int):
        if args.save:
            out_path = file_path.parent / f"{file_path.stem}_line_{step_num}.png"
            try:
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                print(f"[SAVE] Saved: {out_path}")
            except (PermissionError, OSError) as e:
                print(f"[WARNING]  Could not save {out_path}: {e}")


    # ====================================================================
    # PLOTS
    # ====================================================================


    try:
        # -- PANEL 0 - Raw Amplitude (with null bands) ------------------
        fig0, ax0 = create_window()
        t_raw = make_time_axis(amp_step0.shape[0], args.fs)
        # For raw, pick same sc_indices mapped to all subcarriers (not just active)
        for i, sc in enumerate(raw_indices):
            ax0.plot(t_raw, amp_step0[:, sc],
                     color=SC_COLORS[i], linewidth=1.0, alpha=0.75,
                     label=f"SC {sc}")
        style_ax(ax0,
                 "[INFO] Raw Amplitude  (all subcarriers incl. guard/null bands)",
                 "Amplitude (a.u.)")
        if len(raw_indices) <= 20:
            ax0.legend(loc="upper right", fontsize=9, ncol=min(len(raw_indices), 5),
                       framealpha=0.7)
        else:
            ax0.text(0.99, 0.96, f"All {len(raw_indices)} Subcarriers",
                     transform=ax0.transAxes, ha='right', va='top', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.8))
        fig0.tight_layout(rect=[0, 0.05, 1, 0.92])
        save_fig(fig0, 0)


        # -- PANEL 1 - Null Subcarrier Removal -------------------------
        fig1, ax1 = create_window()
        for i, sc in enumerate(sc_indices):
            ax1.plot(t_full, amp_step1[:, sc],
                     color=SC_COLORS[i], linewidth=1.0, alpha=0.75,
                     label=f"SC {sc}")
        style_ax(ax1,
                 f"1. Null Subcarrier Removal  "
                 f"({null_count} nulls removed - {n_active} active subcarriers)",
                 "Amplitude (a.u.)")
        if len(sc_indices) <= 20:
            ax1.legend(loc="upper right", fontsize=9, ncol=min(len(sc_indices), 5),
                       framealpha=0.7)
        else:
            ax1.text(0.99, 0.96, f"All {len(sc_indices)} Active Subcarriers",
                     transform=ax1.transAxes, ha='right', va='top', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.8))
        fig1.tight_layout(rect=[0, 0.05, 1, 0.92])
        save_fig(fig1, 1)


        # -- PANEL 2 - Hampel Filter ------------------------------------
        fig2, ax2 = create_window()
        for i, sc in enumerate(sc_indices):
            ax2.plot(t_full, amp_step2[:, sc],
                     color=SC_COLORS[i], linewidth=1.0, alpha=0.75,
                     label=f"SC {sc}")
        style_ax(ax2,
                 "2. Hampel Filter  (spike / outlier removal, window=11, 3sigma)",
                 "Amplitude (a.u.)")
        if len(sc_indices) <= 20:
            ax2.legend(loc="upper right", fontsize=9, ncol=min(len(sc_indices), 5),
                       framealpha=0.7)
        else:
            ax2.text(0.99, 0.96, f"All {len(sc_indices)} Active Subcarriers",
                     transform=ax2.transAxes, ha='right', va='top', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.8))
        fig2.tight_layout(rect=[0, 0.05, 1, 0.92])
        save_fig(fig2, 2)


        # -- PANEL 3 - Butterworth Low-Pass ----------------------------
        fig3, ax3 = create_window()
        for i, sc in enumerate(sc_indices):
            ax3.plot(t_full, amp_step3[:, sc],
                     color=SC_COLORS[i], linewidth=1.5, alpha=0.9,
                     label=f"SC {sc}")
        style_ax(ax3,
                 f"3. Butterworth Low-Pass  ({args.cutoff} Hz, 4th order, zero-phase)"
                 f"  -  noise removed",
                 "Amplitude (a.u.)")
        if len(sc_indices) <= 20:
            ax3.legend(loc="upper right", fontsize=9, ncol=min(len(sc_indices), 5),
                       framealpha=0.7)
        else:
            ax3.text(0.99, 0.96, f"All {len(sc_indices)} Active Subcarriers",
                     transform=ax3.transAxes, ha='right', va='top', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.8))
        fig3.tight_layout(rect=[0, 0.05, 1, 0.92])
        save_fig(fig3, 3)


        # -- PANEL 4 - Temporal Difference -----------------------------
        fig4, ax4 = create_window()
        if diff_enabled:
            for i, sc in enumerate(sc_indices):
                ax4.plot(t_diff, amp_step4[:, sc],
                         color=SC_COLORS[i], linewidth=1.2, alpha=0.85,
                         label=f"SC {sc}")
            ax4.axhline(0, color="#999999", linewidth=1.0, linestyle="--")
            style_ax(ax4,
                     f"4. Temporal Difference  [frame(t+1) - frame(t)]  ->  "
                     f"motion events visible  ({amp_step4.shape[0]} frames)",
                     "Delta Amplitude / frame")
            if len(sc_indices) <= 20:
                ax4.legend(loc="upper right", fontsize=9, ncol=min(len(sc_indices), 5),
                           framealpha=0.7)
            else:
                ax4.text(0.99, 0.96, f"All {len(sc_indices)} Active Subcarriers",
                         transform=ax4.transAxes, ha='right', va='top', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.8))
        else:
            ax4.text(0.5, 0.5, "Temporal difference DISABLED (--no-diff)",
                     ha='center', va='center', transform=ax4.transAxes,
                     fontsize=13, color="#888888", style='italic')
            style_ax(ax4, "4. Temporal Difference  [DISABLED]", "Delta Amplitude / frame")
        fig4.tight_layout(rect=[0, 0.05, 1, 0.92])
        save_fig(fig4, 4)


        # -- PANEL 5 - PCA ---------------------------------------------
        fig5, ax5 = create_window()
        for i in range(n_pca):
            color = PCA_COLORS[i % len(PCA_COLORS)]
            ax5.plot(t_diff, amp_step5[:, i],
                     color=color, linewidth=1.5, alpha=0.9,
                     label=f"PC{i+1}  ({explained[i]:.1f}%)")
        ax5.axhline(0, color="#999999", linewidth=1.0, linestyle="--")
        style_ax(ax5,
                 f"5. PCA  ({n_pca} components - {explained_total:.1f}% variance explained)"
                 "\n(file-local PCA — for illustration)",
                 "Component Value")
        ax5.legend(loc="upper right", fontsize=9, ncol=min(n_pca, 5), framealpha=0.8)
        fig5.tight_layout(rect=[0, 0.05, 1, 0.92])
        save_fig(fig5, 5)


        # -- PANEL 6 - StandardScaler - Final AI Input -----------------
        fig6, ax6 = create_window()
        for i in range(n_pca):
            color = PCA_COLORS[i % len(PCA_COLORS)]
            ax6.plot(t_diff, amp_step6[:, i],
                     color=color, linewidth=1.5, alpha=0.9,
                     label=f"PC{i+1}")
        ax6.axhline(0, color="#999999", linewidth=1.0, linestyle="--")
        style_ax(ax6,
                 f"6. StandardScaler (Z-score)  -  mean~{amp_step6.mean():.3f}"
                 f"  std~{amp_step6.std():.3f}  -  Final AI Input"
                 "\n(file-local PCA — for illustration)",
                 "Z-score")
        ax6.legend(loc="upper right", fontsize=9, ncol=min(n_pca, 5), framealpha=0.8)
        fig6.tight_layout(rect=[0, 0.05, 1, 0.92])
        save_fig(fig6, 6)


    except Exception as e:
        print(f"[ERROR] Error during plotting: {e}")
        print("   Try installing: pip install matplotlib python-tk")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    print("\n[OK] Created 7 separate windows! (Close them all to end the script)")


    try:
        plt.show()
    except Exception as e:
        print(f"[WARNING]  Error displaying plots: {e}")
        print("   Plots were created but may not display properly.")


    plt.rcParams.update(plt.rcParamsDefault)




if __name__ == "__main__":
    main()
