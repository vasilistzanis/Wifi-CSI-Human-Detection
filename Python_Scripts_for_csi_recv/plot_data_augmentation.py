#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSI Data Augmentation Visualization
=====================================
Visualizes the four augmentation techniques by calling the exact same
functions used during ML training (from csi_ml_pipeline.py).

Figures produced here can be cited in the thesis as accurate representations
of what the training pipeline actually does — not an approximation.

Usage
-----
  python plot_data_augmentation.py
  python plot_data_augmentation.py --file datasets/walk_activity/walk_activity_01_vasilis_.txt
  python plot_data_augmentation.py --class-label fall
  python plot_data_augmentation.py --save --no-show
"""

import numpy as np
import seaborn as sns
from pathlib import Path
import sys
import argparse

try:
    from data_preprocessing import load_csi_csv, CSIPipeline
except ImportError:
    print("[ERROR] data_preprocessing.py not found.")
    sys.exit(1)

try:
    from csi_ml_pipeline import _aug_noise, _aug_shift, _aug_scale, _aug_time_warp
except ImportError:
    print("[ERROR] csi_ml_pipeline.py not found.")
    sys.exit(1)

import config
from plot_window_utils import setup_matplotlib, show_figure
import matplotlib.pyplot as plt

RANDOM_SEED    = config.RANDOM_SEED
MIN_FRAMES     = config.PLOT_DATA_AUGMENTATION_MIN_FRAMES
SEGMENT_LEN    = config.WINDOW_SIZE
SUBCARRIER_IDX = config.PLOT_DATA_AUGMENTATION_SUBCARRIER

# -----------------------------------------------------------------------
# Parameter annotation strings — mirrors the logic in the _aug_* functions
# of csi_ml_pipeline.py so each subplot is self-documenting.
# -----------------------------------------------------------------------
_PARAM_LABELS = {
    "noise": {
        "walk_activity": "σ_signal × Uniform(0.3%, 1.0%)",
        "no_activity": "σ_signal × Uniform(0.3%, 1.0%)",
        "sit":  "σ_signal × 0.7 × Uniform(0.3%, 1.0%)",
        "fall": "σ_signal × 0.5 × Uniform(0.3%, 1.0%)",
    },
    "shift": {
        "walk_activity": "1–3 frames, edge-padded",
        "no_activity": "1–3 frames, edge-padded",
        "sit":  "1–3 frames, edge-padded",
        "fall": "1–3 frames, edge-padded",
    },
    "scale": {
        "walk_activity": "× Uniform(0.90, 1.10)",
        "no_activity": "× Uniform(0.90, 1.10)",
        "sit":  "× Uniform(0.95, 1.05)",
        "fall": "× Uniform(0.97, 1.03)",
    },
    "warp": {
        "walk_activity": "factor ∈ Uniform(0.90, 1.10)",
        "no_activity": "factor ∈ Uniform(0.98, 1.02)",
        "sit":  "factor ∈ Uniform(0.95, 1.05)",
        "fall": "Disabled (fall physics constraint)",
    },
}


# -----------------------------------------------------------------------
# TRAINING AUGMENTATION BRIDGE
# -----------------------------------------------------------------------

def _apply_aug(fn, signal_1d: np.ndarray, rng, class_label: str) -> np.ndarray:
    """
    Call a training augmentation function on a 1-D signal.

    The training functions expect (window_size, n_subcarriers); we wrap
    to (N, 1), call fn, and extract column 0 on return.  The result is
    identical to what the function produces on a single-subcarrier window
    during training.
    """
    w2d = signal_1d[:, np.newaxis].astype(np.float32)
    aug = fn(w2d, rng, class_label=class_label)
    return aug[:, 0].astype(np.float64)


# -----------------------------------------------------------------------
# DATA HELPERS 
# -----------------------------------------------------------------------

def _make_synthetic(segment_len: int = SEGMENT_LEN) -> np.ndarray:
    """Return a normalised synthetic 'walking' signal of length segment_len."""
    t        = np.linspace(0, 4 * np.pi, segment_len)
    original = np.sin(t) * np.sin(0.5 * t) + 1.0
    mn, mx   = original.min(), original.max()
    return (original - mn) / (mx - mn + 1e-9)


def _load_real(data_path: Path,
               min_frames: int = MIN_FRAMES,
               segment_len: int = SEGMENT_LEN,
               subcarrier_idx: int = SUBCARRIER_IDX) -> tuple:
    """
    Load a real CSI recording, apply standard filtering, return (signal_1d, info_str).
    Returns (None, error_str) on any failure so the caller can fall back.
    """
    complex_matrix, _ = load_csi_csv(data_path)

    if complex_matrix is None or complex_matrix.size == 0:
        return None, "Loaded matrix is empty"

    n_frames, _ = complex_matrix.shape
    if n_frames < min_frames:
        return None, (f"Recording too short: {n_frames} frames "
                      f"(need >= {min_frames})")

    pipeline   = CSIPipeline(fs=config.SAMPLING_RATE, use_diff=False)
    amp_active = pipeline.remove_null_subcarriers(complex_matrix, fit=True)
    amp_clean  = pipeline.apply_hampel_filter(amp_active)
    amp_clean  = pipeline.apply_lowpass_filter(amp_clean)

    n_frames_clean, n_active_sc = amp_clean.shape
    if n_active_sc == 0:
        return None, "No active subcarriers after filtering"

    sc_idx = min(subcarrier_idx, n_active_sc - 1)
    if sc_idx != subcarrier_idx:
        print(f"  [WARNING] subcarrier {subcarrier_idx} out of range "
              f"({n_active_sc} active) — using SC {sc_idx}")

    start     = max(0, min(500, n_frames_clean - segment_len))
    end       = start + segment_len
    amplitude = amp_clean[start:end, sc_idx]

    mn, mx = amplitude.min(), amplitude.max()
    signal = (amplitude - mn) / (mx - mn + 1e-9)

    info = (f"Filtered SC {sc_idx}  |  frames {start}–{end}  "
            f"|  {n_frames_clean} frames  |  {n_active_sc} active SC")
    return signal, info


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    setup_matplotlib()
    defaults = config.get_script_defaults("plot_data_augmentation")
    parser = argparse.ArgumentParser(
        description="Visualize the exact augmentation techniques used during training"
    )
    config.add_bool_argument(
        parser,
        dest="simulate",
        default=defaults["simulate"],
        help="Use synthetic data instead of real",
        positive_flags=["--simulate"],
        negative_flags=["--no-simulate"],
    )
    config.add_bool_argument(
        parser,
        dest="save",
        default=defaults["save"],
        help="Save the plot as PNG (300 DPI)",
        positive_flags=["--save"],
        negative_flags=["--no-save"],
    )
    parser.add_argument("--output-dir", type=str, default=defaults["output_dir"],
                        help="Directory to save plots (default: models/plots)")
    config.add_bool_argument(
        parser,
        dest="show",
        default=defaults["show"],
        help="Open a plot window when possible.",
        positive_flags=["--show"],
        negative_flags=["--no-show"],
    )
    parser.add_argument("--min-frames", type=int, default=defaults["min_frames"],
                        help=f"Minimum frames required (default: {MIN_FRAMES})")
    parser.add_argument("--file", type=str, default=defaults["file"],
                        help="Path to real CSI data file")
    parser.add_argument("--subcarrier", type=int, default=defaults["subcarrier"],
                        help=f"Subcarrier index to visualize (default: {SUBCARRIER_IDX})")
    parser.add_argument("--segment-len", type=int, default=defaults["segment_len"],
                        help=f"Signal segment length in frames (default: {SEGMENT_LEN})")
    _aug_choices = sorted(set(config.get_known_training_classes()))
    parser.add_argument("--class-label", type=str, default=defaults["class_label"],
                        choices=_aug_choices,
                        help="Activity class — controls class-aware aug constraints (default: walk_activity)")
    args = parser.parse_args()

    rng = np.random.default_rng(seed=RANDOM_SEED)
    cls = args.class_label

    # -- 1. Load data --------------------------------------------------
    title_prefix = "Synthetic"
    original     = None

    if not args.simulate:
        data_path = Path(args.file)
        if not data_path.exists():
            print(f"[ERROR] File not found: {data_path} — falling back to synthetic data")
        else:
            print(f"[FILE] Loading: {data_path}")
            original, info = _load_real(data_path,
                                        min_frames=args.min_frames,
                                        segment_len=args.segment_len,
                                        subcarrier_idx=args.subcarrier)
            if original is None:
                print(f"[ERROR] {info} — falling back to synthetic data")
            else:
                print(f"[OK] {info}")
                title_prefix = "Real"

    if original is None:
        print("[INFO] Mode: SYNTHETIC")
        original     = _make_synthetic(segment_len=args.segment_len)
        title_prefix = "Synthetic"

    # -- 2. Apply the exact training augmentation functions ------------
    # _aug_* are imported from csi_ml_pipeline and are identical to what
    # build_dataset() calls on every training window.
    noisy   = _apply_aug(_aug_noise,     original, rng, cls)
    shifted = _apply_aug(_aug_shift,     original, rng, cls)
    scaled  = _apply_aug(_aug_scale,     original, rng, cls)
    warped  = _apply_aug(_aug_time_warp, original, rng, cls)

    # time_warp is excluded for fall by augment_window() — gray it out.
    warp_disabled = (cls == "fall")

    # -- 3. Plot -------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=config.PLOT_DATA_AUGMENTATION_SIZE, sharey=True)
    axes_flat  = axes.flatten()

    fig.suptitle(
        f"Data Augmentation — {title_prefix} CSI Signal  |  class: {cls}\n"
        "Exact functions and class-aware parameters from csi_ml_pipeline.py",
        fontsize=14, fontweight="bold", y=0.97,
    )

    plot_configs = [
        ("original", original, f"Original {title_prefix} Signal",    "#00e5ff"),
        ("noise",    noisy,    "Gaussian Noise\n(channel jitter)",    "#ff3366"),
        ("shift",    shifted,  "Temporal Shift\n(start delay)",       "#9b59b6"),
        ("scale",    scaled,   "Magnitude Scaling\n(distance sim.)",  "#26de81"),
        ("warp",     warped,   "Time Warp\n(speed variation)",        "#f7b731"),
    ]

    for i, (key, data, title, color) in enumerate(plot_configs):
        ax     = axes_flat[i]
        grayed = (key == "warp" and warp_disabled)
        c      = "#aaaaaa" if grayed else color

        ax.plot(data, color=c, linewidth=2)
        ax.fill_between(range(len(data)), data, alpha=0.15, color=c)
        ax.set_title(title, fontweight="bold", fontsize=11,
                     color="#888888" if grayed else "black")
        ax.set_xlabel("Time (Frames)", fontsize=9)
        ax.set_ylim(-0.1, 1.2)
        if i % 3 == 0:
            ax.set_ylabel("Normalised Amplitude", fontsize=9)

        param = _PARAM_LABELS.get(key, {}).get(cls, "")
        if param:
            ax.text(0.97, 0.04, param,
                    transform=ax.transAxes,
                    fontsize=7.5, ha="right", va="bottom", color="#444444",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", alpha=0.75))

    axes_flat[-1].axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    # -- 4. Save / show ------------------------------------------------
    filename = f"{title_prefix}_Data_Augmentation_{cls}.png"
    if args.save:
        out_dir   = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE] → {save_path}")

    if args.show:
        show_figure(fig)
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
