#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Pipeline View — uses the ACTUAL saved preprocessing model
=============================================================
Visualizes every preprocessing stage by calling the same fitted
transforms that are deployed at inference time.  All PCA/scaler
numbers match what the trained model sees — not a fresh re-fit.

Modes
-----
  (default)       7-step pipeline + PCA scree plot for one data file
  --compare       PC1 vs PC2 scatter for all classes (one file per class)
  --features      140-dim feature vector bar chart

Usage
-----
  python visualize_ml_pipeline_view.py
  python visualize_ml_pipeline_view.py --file datasets/walk_activity/walk_activity_01_vasilis_.txt
  python visualize_ml_pipeline_view.py --compare
  python visualize_ml_pipeline_view.py --features --file datasets/no_activity/no_activity_livroom_01_1776088773.txt
  python visualize_ml_pipeline_view.py --save --no-show
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np

from csi_parser import configure_console_output
configure_console_output()

from plot_window_utils import setup_matplotlib, show_all
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import joblib
    from data_preprocessing import CSIPipeline, load_csi_csv
    from csi_ml_pipeline import extract_features_from_window, _get_feature_names, N_STATS
    import config
except ImportError as e:
    print(f"[ERROR] Missing module: {e}")
    print("        Run this script from the Python_Scripts_for_csi_recv directory.")
    sys.exit(1)


# -----------------------------------------------------------------------
# STYLE  (matches thesis palette)
# -----------------------------------------------------------------------

STYLE = {
    "bg":      "#ffffff",
    "panel":   "#fafafa",
    "text":    "#1a1a1a",
    "grid":    "#e0e0e0",
    "a1":      "#2563eb",
    "a2":      "#f59e0b",
    "a3":      "#10b981",
    "a4":      "#ef4444",
    "a5":      "#8b5cf6",
    "a6":      "#06b6d4",
}

PCA_COLORS = ["#e63946", "#2a9d8f", "#e9c46a", "#457b9d", "#f4a261",
              "#6a4c93", "#1982c4", "#8ac926", "#ff595e", "#ffca3a"]

CLASS_COLORS = [STYLE["a1"], STYLE["a3"], STYLE["a4"], STYLE["a2"],
                STYLE["a5"], STYLE["a6"]]


def _apply_style():
    for s in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"]:
        try:
            plt.style.use(s)
            break
        except Exception:
            continue
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.facecolor":    STYLE["panel"],
        "figure.facecolor":  STYLE["bg"],
        "axes.grid":         True,
        "grid.alpha":        0.35,
        "grid.linewidth":    0.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def _save_fig(fig, save_dir: Path, name: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=STYLE["bg"])
    print(f"  [SAVE] {out}")


# -----------------------------------------------------------------------
# ARTIFACT LOADING
# -----------------------------------------------------------------------

def load_artifacts(models_dir: Path):
    """Load pipeline, label encoder, and experiment config from models_dir."""
    pipeline_path = models_dir / "csi_pipeline.joblib"
    le_path       = models_dir / "label_encoder.joblib"
    cfg_path      = models_dir / "experiment_config.json"

    if not pipeline_path.exists():
        print(f"[ERROR] Pipeline not found: {pipeline_path}")
        print("        Run: python csi_ml_pipeline.py --classes walk_activity no_activity --save_model")
        sys.exit(1)

    pipeline = joblib.load(pipeline_path)
    print(f"  [LOAD] {pipeline_path}")

    le = None
    if le_path.exists():
        le = joblib.load(le_path)
        print(f"  [LOAD] {le_path}")

    cfg = None
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"  [LOAD] {cfg_path}")

    return pipeline, le, cfg


# -----------------------------------------------------------------------
# PIPELINE STAGE RUNNER
# -----------------------------------------------------------------------

def _run_all_stages(pipeline: CSIPipeline, complex_matrix: np.ndarray):
    """
    Apply each stage of the SAVED pipeline individually and return
    intermediate arrays for visualization.  The PCA and scaler are
    the actual fitted objects — not a fresh re-fit.
    """
    # Stage 0: raw amplitude
    amp_raw = np.abs(complex_matrix)

    # Stage 1: null removal (uses saved active_mask — no re-fit)
    amp_s1 = pipeline.remove_null_subcarriers(complex_matrix, fit=False)

    # Stage 2: Hampel filter
    amp_s2 = pipeline.apply_hampel_filter(amp_s1)

    # Stage 3: Butterworth low-pass
    amp_s3 = pipeline.apply_lowpass_filter(amp_s2, cutoff=pipeline.cutoff)

    # Stage 4: temporal difference (optional, controlled by pipeline.use_diff)
    amp_s4 = pipeline.apply_temporal_diff(amp_s3)

    # Stage 5: PCA — ACTUAL trained transform
    if pipeline.pca is not None:
        amp_s5 = pipeline.pca.transform(amp_s4)
        explained = pipeline.pca.explained_variance_ratio_ * 100
    else:
        amp_s5 = amp_s4
        explained = np.array([100.0])

    # Stage 6: scaler — ACTUAL trained transform
    amp_s6 = pipeline.scaler.transform(amp_s5)

    return amp_raw, amp_s1, amp_s2, amp_s3, amp_s4, amp_s5, amp_s6, explained


# -----------------------------------------------------------------------
# MODE 1: PIPELINE STEP VIEW
# -----------------------------------------------------------------------

def plot_pipeline_steps(
    pipeline: CSIPipeline, data_path: Path,
    window_size: int, start_frame: int,
    save: bool, save_dir: Path, no_show: bool,
):
    """Show all 7 preprocessing stages using the trained pipeline."""
    print(f"\n[INFO] Loading: {data_path}")
    complex_matrix, _ = load_csi_csv(data_path)

    n_frames = complex_matrix.shape[0]
    start = min(start_frame, max(0, n_frames - window_size))
    window = complex_matrix[start:start + window_size]

    (amp_raw, amp_s1, amp_s2, amp_s3,
     amp_s4, amp_s5, amp_s6, explained) = _run_all_stages(pipeline, window)

    n_pca      = amp_s5.shape[1]
    total_var  = explained.sum()
    diff_label = "(after temporal diff)" if pipeline.use_diff else "(no temporal diff)"
    t          = np.arange(amp_s4.shape[0])
    t_raw      = np.arange(window_size)

    _apply_style()
    fig = plt.figure(figsize=config.VISUALIZE_ML_PIPELINE_SIZE)
    fig.patch.set_facecolor(STYLE["bg"])

    # 3-column, 3-row grid: panels 0-6 for pipeline steps, panel 7 for scree
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

    # Map (grid_row, grid_col) to each pipeline stage
    panel_defs = [
        (gs[0, 0], amp_raw,  t_raw, "0. Raw Amplitude",
         STYLE["a1"], False, "Amplitude", "All subcarriers (incl. nulls)"),
        (gs[0, 1], amp_s1,   t_raw, "1. Null Subcarrier Removal",
         STYLE["a2"], False, "Amplitude", f"{amp_s1.shape[1]} active SC"),
        (gs[0, 2], amp_s2,   t_raw, "2. Hampel Filter (Outlier Removal)",
         STYLE["a3"], False, "Amplitude", "Median ± 3σ threshold"),
        (gs[1, 0], amp_s3,   t_raw, "3. Butterworth Low-Pass (10 Hz / 4th-order)",
         STYLE["a4"], False, "Amplitude", "Zero-phase forward-backward"),
        (gs[1, 1], amp_s4,   t,     f"4. Temporal Difference {diff_label}",
         STYLE["a5"], False, "Δ Amplitude", "Motion-focused signal"),
        (gs[1, 2], amp_s5,   t,     f"5. PCA — {n_pca} components  ({total_var:.1f}% var)\n"
                                     "ACTUAL trained model transform",
         None,        True,  "PC Value", "Trained PCA (not re-fitted)"),
        (gs[2, 0], amp_s6,   t,     f"6. StandardScaler — Final AI Input\n"
                                     "ACTUAL trained model transform",
         None,        True,  "Z-score", "Trained scaler (not re-fitted)"),
    ]

    for spec, data, taxis, title, color, is_pca, ylabel, subtitle in panel_defs:
        ax = fig.add_subplot(spec)
        n_cols = data.shape[1]

        if is_pca:
            for i in range(min(n_cols, len(PCA_COLORS))):
                label = f"PC{i+1}"
                if explained is not None and i < len(explained):
                    label += f" ({explained[i]:.1f}%)"
                ax.plot(taxis, data[:, i],
                        color=PCA_COLORS[i], linewidth=1.5, alpha=0.9, label=label)
            ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
            ax.legend(fontsize=7, ncol=min(n_cols, 5), loc="upper right",
                      framealpha=0.8)
        else:
            # Show up to 8 subcarriers to keep the plot readable
            step = max(1, n_cols // 8)
            for i in range(0, min(n_cols, n_cols), step):
                ax.plot(taxis, data[:, i],
                        color=color, linewidth=1.0, alpha=0.35)
            # Highlight the first subcarrier
            ax.plot(taxis, data[:, 0],
                    color=color, linewidth=2.0, alpha=0.95, label="SC 0")

        ax.set_title(f"{title}\n{subtitle}", fontsize=9.5, fontweight="bold",
                     color=STYLE["text"], pad=6)
        ax.set_xlabel("Time (frames)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)

    # Panel 8: PCA explained variance (scree)
    ax_scree = fig.add_subplot(gs[2, 1:])
    ax_scree.bar(
        range(1, len(explained) + 1), explained,
        color=PCA_COLORS[:len(explained)], edgecolor="white", linewidth=0.5,
    )
    ax_scree.set_xticks(range(1, len(explained) + 1))
    ax_scree.set_xticklabels([f"PC{i}" for i in range(1, len(explained) + 1)],
                              fontsize=9, fontweight="bold")
    ax_scree.set_xlabel("Principal Component", fontsize=9, fontweight="bold")
    ax_scree.set_ylabel("Explained Variance (%)", fontsize=9, fontweight="bold")
    ax_scree.set_title(
        f"PCA Explained Variance — Trained Model\n"
        f"Total: {total_var:.1f}%  |  {len(explained)} components",
        fontsize=10, fontweight="bold",
    )
    # Cumulative line
    ax_cum = ax_scree.twinx()
    ax_cum.plot(range(1, len(explained) + 1), np.cumsum(explained),
                color=STYLE["a4"], linewidth=2, marker="o", markersize=5,
                label="Cumulative")
    ax_cum.set_ylabel("Cumulative (%)", fontsize=9, color=STYLE["a4"])
    ax_cum.tick_params(axis="y", labelcolor=STYLE["a4"])
    ax_cum.set_ylim(0, 105)

    # Annotate each bar
    for i, v in enumerate(explained):
        ax_scree.text(i + 1, v + 0.5, f"{v:.1f}%", ha="center", va="bottom",
                      fontsize=8, fontweight="bold")

    file_label = data_path.stem if data_path else "synthetic"
    fig.suptitle(
        f"CSI ML Pipeline — Trained Model View\n"
        f"File: {file_label}   |   Frames {start}–{start+window_size}   |   "
        f"PCA + Scaler: ACTUAL saved transforms",
        fontsize=13, fontweight="bold", y=0.98,
    )

    if save:
        _save_fig(fig, save_dir, f"ml_pipeline_steps_{file_label}")
    if not no_show:
        show_all(figs=[fig])
    plt.close(fig)


# -----------------------------------------------------------------------
# MODE 2: COMPARE CLASSES IN PCA SPACE
# -----------------------------------------------------------------------

def plot_compare_classes(
    pipeline: CSIPipeline, le, cfg: dict,
    data_dir: Path, window_size: int, step: int,
    save: bool, save_dir: Path, no_show: bool,
):
    """
    Plot all windows from each class in the ACTUAL trained PCA space.
    Uses one recording per class (first test file listed in experiment_config.json).
    """
    if le is None or cfg is None:
        print("[ERROR] --compare requires label_encoder.joblib and experiment_config.json")
        sys.exit(1)

    classes    = cfg.get("classes", le.classes_.tolist())
    test_files = cfg.get("test_files", {})

    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=config.VISUALIZE_ML_PIPELINE_SIZE)
    fig.patch.set_facecolor(STYLE["bg"])

    all_pc1, all_pc2 = [], []
    scatter_handles = []

    for cls_idx, cls_name in enumerate(classes):
        files = test_files.get(cls_name, [])
        if not files:
            print(f"  [SKIP] No test file listed for class '{cls_name}'")
            continue

        fpath = data_dir / files[0]
        if not fpath.exists():
            print(f"  [SKIP] File not found: {fpath}")
            continue

        print(f"  [LOAD] {cls_name}: {fpath}")
        complex_matrix, _ = load_csi_csv(fpath)
        n = complex_matrix.shape[0]

        # Slide windows and transform each through the saved pipeline
        pca_windows = []
        for s in range(0, n - window_size + 1, step):
            win = complex_matrix[s:s + window_size]
            # Use pipeline.transform which applies the full chain
            processed = pipeline.transform(win, use_pca=True).astype(np.float64)
            pca_windows.append(processed.mean(axis=0))   # mean over time → 1 point per window

        if not pca_windows:
            continue

        pts = np.array(pca_windows)
        color = CLASS_COLORS[cls_idx % len(CLASS_COLORS)]

        # PC1 vs PC2
        sc = axes[0].scatter(pts[:, 0], pts[:, 1],
                             c=color, alpha=0.55, s=25, label=cls_name,
                             edgecolors="none")
        scatter_handles.append(sc)

        # PC1 vs PC3 (if available)
        if pts.shape[1] >= 3:
            axes[1].scatter(pts[:, 0], pts[:, 2],
                            c=color, alpha=0.55, s=25, label=cls_name,
                            edgecolors="none")

        all_pc1.extend(pts[:, 0].tolist())
        all_pc2.extend(pts[:, 1].tolist())

    if pipeline.pca is None:
        print("[ERROR] Pipeline has no fitted PCA — re-run csi_ml_pipeline.py first.")
        plt.close(fig)
        return
    ev = pipeline.pca.explained_variance_ratio_ * 100
    axes[0].set_xlabel(f"PC1  ({ev[0]:.1f}% var)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel(f"PC2  ({ev[1]:.1f}% var)", fontsize=10, fontweight="bold")
    axes[0].set_title("PC1 vs PC2  (window mean, trained PCA space)",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=10, framealpha=0.9)

    if ev.shape[0] >= 3:
        axes[1].set_xlabel(f"PC1  ({ev[0]:.1f}% var)", fontsize=10, fontweight="bold")
        axes[1].set_ylabel(f"PC3  ({ev[2]:.1f}% var)", fontsize=10, fontweight="bold")
        axes[1].set_title("PC1 vs PC3  (window mean, trained PCA space)",
                          fontsize=11, fontweight="bold")
        axes[1].legend(fontsize=10, framealpha=0.9)
    else:
        axes[1].axis("off")

    fig.suptitle(
        "Class Separation in Trained PCA Space\n"
        "Each point = mean of one sliding window through the saved model transforms",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        _save_fig(fig, save_dir, "ml_pipeline_compare_pca_space")
    if not no_show:
        show_all(figs=[fig])
    plt.close(fig)


# -----------------------------------------------------------------------
# MODE 3: FEATURE VECTOR
# -----------------------------------------------------------------------

def plot_feature_vector(
    pipeline: CSIPipeline, data_path: Path,
    window_size: int, start_frame: int,
    save: bool, save_dir: Path, no_show: bool,
):
    """Show the full feature vector extracted by the trained pipeline."""
    print(f"\n[INFO] Loading: {data_path}")
    complex_matrix, _ = load_csi_csv(data_path)

    n_frames = complex_matrix.shape[0]
    start = min(start_frame, max(0, n_frames - window_size))
    window = complex_matrix[start:start + window_size]

    # Full pipeline transform (saved PCA + scaler)
    processed = pipeline.transform(window, use_pca=True).astype(np.float64)
    feat = extract_features_from_window(processed, fs=pipeline.fs, cutoff_hz=pipeline.cutoff)

    n_pca = processed.shape[1]
    feature_names = _get_feature_names(n_pca)

    # Normalise to show relative magnitude
    abs_feat = np.abs(feat)
    if abs_feat.max() > 0:
        rel_feat = abs_feat / abs_feat.max()
    else:
        rel_feat = abs_feat

    # Color-code by PCA component
    colors = [PCA_COLORS[(i // N_STATS) % len(PCA_COLORS)] for i in range(len(feat))]

    _apply_style()
    fig, ax = plt.subplots(figsize=(14, max(6, len(feat) * 0.18)))
    fig.patch.set_facecolor(STYLE["bg"])

    y_pos = range(len(feature_names))
    ax.barh(y_pos, rel_feat, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Relative Magnitude (|value| / max)", fontweight="bold", fontsize=10)
    ax.set_title(
        f"Feature Vector — {len(feat)} features  ({n_pca} PCA × {N_STATS} stats)\n"
        f"File: {data_path.stem}   |   Frames {start}–{start+window_size}   |   "
        "Trained pipeline transforms",
        fontsize=11, fontweight="bold",
    )

    # Legend for PCA components
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=PCA_COLORS[i % len(PCA_COLORS)], label=f"PC{i+1}")
        for i in range(n_pca)
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8,
              framealpha=0.9, ncol=min(n_pca, 5))

    fig.tight_layout()

    if save:
        _save_fig(fig, save_dir, f"ml_feature_vector_{data_path.stem}")
    if not no_show:
        show_all(figs=[fig])
    plt.close(fig)


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    setup_matplotlib()
    defaults = config.get_script_defaults("visualize_ml_pipeline_view")
    parser = argparse.ArgumentParser(
        description="Visualize CSI ML pipeline using ACTUAL saved model transforms"
    )
    parser.add_argument("--models-dir",  type=str, default=defaults["models_dir"],
                        help="Directory with saved .joblib files (default: models)")
    parser.add_argument("--file", type=str, default=defaults["file"],
                        help="CSI data file to visualize (default: first test file from experiment_config.json)")
    parser.add_argument("--start-frame", type=int, default=defaults["start_frame"],
                        help="Frame index to start the analysis window (default: 500)")
    parser.add_argument("--window-size", type=int, default=defaults["window_size"],
                        help=f"Frames per window (default: {config.WINDOW_SIZE})")
    parser.add_argument("--step",        type=int, default=defaults["step"],
                        help=f"Sliding-window step for --compare (default: {config.PIPELINE_STEP_SIZE})")
    config.add_bool_argument(
        parser,
        dest="compare",
        default=defaults["compare"],
        help="Overlay all classes in the trained PCA space",
        positive_flags=["--compare"],
        negative_flags=["--no-compare"],
    )
    config.add_bool_argument(
        parser,
        dest="features",
        default=defaults["features"],
        help="Show extracted feature vector for the given file",
        positive_flags=["--features"],
        negative_flags=["--no-features"],
    )
    config.add_bool_argument(
        parser,
        dest="save",
        default=defaults["save"],
        help="Save figures to --out-dir (default: models/plots/)",
        positive_flags=["--save"],
        negative_flags=["--no-save"],
    )
    parser.add_argument("--out-dir",     type=str, default=defaults["out_dir"],
                        help="Output directory for saved figures")
    config.add_bool_argument(
        parser,
        dest="show",
        default=defaults["show"],
        help="Open plot windows when possible.",
        positive_flags=["--show"],
        negative_flags=["--no-show"],
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    save_dir   = Path(args.out_dir) if args.out_dir else models_dir / "plots"

    print("\n" + "=" * 60)
    print("  ML PIPELINE VIEW  (Trained Model)")
    print("=" * 60)

    pipeline, le, cfg = load_artifacts(models_dir)

    # Resolve data directory from config or default
    data_dir = Path(cfg["data_dir"]) if cfg else Path(config.DATASETS_DIR)

    # Resolve default data file
    if args.file:
        data_path = Path(args.file)
    elif cfg:
        # Use first test file of the first class
        first_class = list(cfg.get("test_files", {}).keys())
        if first_class:
            rel = cfg["test_files"][first_class[0]][0]
            data_path = data_dir / rel
        else:
            data_path = None
    else:
        data_path = None

    print(f"\n  Pipeline  : fitted={pipeline.is_fitted}  "
          f"use_diff={pipeline.use_diff}  cutoff={pipeline.cutoff} Hz")
    if pipeline.pca is not None:
        ev = pipeline.pca.explained_variance_ratio_ * 100
        print(f"  PCA       : {len(ev)} components  total_var={ev.sum():.1f}%")
    if le is not None:
        print(f"  Classes   : {list(le.classes_)}")
    print()

    if args.compare:
        plot_compare_classes(
            pipeline, le, cfg, data_dir,
            args.window_size, args.step,
            args.save, save_dir, not args.show,
        )
    elif args.features:
        if data_path is None or not data_path.exists():
            print(f"[ERROR] No valid data file for --features.  Use --file <path>.")
            sys.exit(1)
        plot_feature_vector(
            pipeline, data_path,
            args.window_size, args.start_frame,
            args.save, save_dir, not args.show,
        )
    else:
        if data_path is None or not data_path.exists():
            print(f"[WARNING] Data file not found: {data_path}")
            print("          Use --file <path> or ensure experiment_config.json is present.")
            sys.exit(1)
        plot_pipeline_steps(
            pipeline, data_path,
            args.window_size, args.start_frame,
            args.save, save_dir, not args.show,
        )


if __name__ == "__main__":
    main()
