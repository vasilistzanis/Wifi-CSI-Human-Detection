import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import argparse
from typing import Optional

# Import the loader from your existing preprocessing script
try:
    from data_preprocessing import load_csi_csv
except ImportError:
    print("Error: data_preprocessing.py not found in the current directory.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

RANDOM_SEED    = 42     # reproducible plots
MIN_FRAMES     = 200    # minimum recording length required
SEGMENT_LEN    = 200    # frames per segment used for visualization
SUBCARRIER_IDX = 30     # preferred subcarrier (bounds-checked below)

# ─────────────────────────────────────────────────────────────────────
# AUGMENTATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def augment_noise(window: np.ndarray,
                  noise_level: float = 0.04,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Add Gaussian noise to simulate wireless channel jitter.

    Args:
        window:      1-D normalised signal array [0, 1].
        noise_level: Std-dev of Gaussian noise relative to signal range.
                     0.04 (4%) is representative of real jitter.
        rng:         Optional numpy Generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    noise = rng.normal(0, noise_level, window.shape)
    return window + noise


def augment_scale(window: np.ndarray,
                  scale_factor: float = 0.6) -> np.ndarray:
    """
    Scale signal magnitude to simulate varying transmitter-receiver distance.

    Args:
        window:       1-D signal array.
        scale_factor: Multiplier applied to all samples (default 0.6 = 60%).
    """
    return window * scale_factor


def augment_time_warp(window: np.ndarray,
                      stretch: float = 1.5) -> np.ndarray:
    """
    Stretch or compress signal in time to simulate speed variation.

    The output always has the same length as the input:
      stretch > 1 → resample to longer, then crop centre.
      stretch < 1 → resample to shorter, then pad with edge value.

    Args:
        window:  1-D signal array.
        stretch: Time-stretch factor (e.g. 1.5 = 50% slower).
    """
    orig_len   = len(window)
    orig_steps = np.arange(orig_len)
    new_len    = int(orig_len * stretch)
    new_steps  = np.linspace(0, orig_len - 1, new_len)
    warped     = np.interp(new_steps, orig_steps, window)

    if len(warped) >= orig_len:
        # Crop centre to keep original length
        start  = (len(warped) - orig_len) // 2
        warped = warped[start : start + orig_len]
    else:
        # Pad right edge to restore original length
        warped = np.pad(warped, (0, orig_len - len(warped)), mode='edge')

    return warped


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _make_synthetic() -> np.ndarray:
    """Return a normalised synthetic 'walking' signal of length SEGMENT_LEN."""
    t        = np.linspace(0, 4 * np.pi, SEGMENT_LEN)
    original = np.sin(t) * np.sin(0.5 * t) + 1.0
    mn, mx   = original.min(), original.max()
    return (original - mn) / (mx - mn + 1e-9)


def _load_real(data_path: Path) -> tuple:
    """
    Load a real CSI recording and return (signal_1d, info_str).

    Returns (None, error_str) on any failure so the caller can fall back.
    """
    complex_matrix, _ = load_csi_csv(data_path)

    if complex_matrix is None or complex_matrix.size == 0:
        return None, "Loaded matrix is empty"

    n_frames, n_sc = complex_matrix.shape

    # Guard against short recordings
    if n_frames < MIN_FRAMES:
        return None, (f"Recording too short: {n_frames} frames "
                      f"(need >= {MIN_FRAMES})")

    # Bounds check: clamp subcarrier index to valid range
    sc_idx = min(SUBCARRIER_IDX, n_sc - 1)
    if sc_idx != SUBCARRIER_IDX:
        print(f"  ⚠️  subcarrier {SUBCARRIER_IDX} out of range "
              f"(matrix has {n_sc} SC) — using SC {sc_idx} instead")

    # Use max(0, ...) to guarantee non-negative start_frame
    start = max(0, min(500, n_frames - SEGMENT_LEN))
    end   = start + SEGMENT_LEN

    amplitude = np.abs(complex_matrix[start:end, sc_idx])

    # Normalise to [0, 1]
    mn, mx  = amplitude.min(), amplitude.max()
    signal  = (amplitude - mn) / (mx - mn + 1e-9)

    info = (f"SC {sc_idx}  |  frames {start}–{end}  "
            f"|  total {n_frames} frames  |  {n_sc} subcarriers")
    return signal, info


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot CSI Data Augmentation")
    parser.add_argument('--simulate', action='store_true',
                        help="Use synthetic data instead of real")
    parser.add_argument('--save', action='store_true',
                        help="Save the plot to a file")
    parser.add_argument('--output-dir', type=str, default="models/plots",
                        help="Directory to save plots (default: models/plots)")
    parser.add_argument('--no-show', action='store_true',
                        help="Do not display the plot window")
    args = parser.parse_args()

    # Create a single seeded RNG passed to all augmentation funcs
    rng = np.random.default_rng(seed=RANDOM_SEED)

    # ── 1. Select data source ─────────────────────────────────────────
    title_prefix = "Synthetic"
    filename     = "Synthetic_Data_Augmentation_Vis.png"
    original     = None

    if not args.simulate:
        data_path = Path("datasets/walk/walk_01.txt")
        if not data_path.exists():
            print(f"❌  File not found: {data_path} — falling back to synthetic data")
        else:
            print(f"📂  Loading real data from {data_path} ...")
            original, info = _load_real(data_path)
            if original is None:
                print(f"❌  {info} — falling back to synthetic data")
            else:
                print(f"✅  {info}")
                title_prefix = "Real"
                filename     = "Real_Data_Augmentation_Vis.png"

    if original is None:
        print("💡  Mode: SYNTHETIC")
        original = _make_synthetic()

    # ── 2. Apply augmentations ────────────────────────────────────────
    noisy  = augment_noise(original,     noise_level=0.04, rng=rng)
    scaled = augment_scale(original,     scale_factor=0.6)
    warped = augment_time_warp(original, stretch=1.5)

    # ── 3. Plot ───────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    fig.suptitle(
        f"Data Augmentation Techniques — {title_prefix} CSI Signal",
        fontsize=16, fontweight='bold', y=0.96,
    )

    plot_configs = [
        (original, f"Original {title_prefix} Signal",         '#00e5ff'),
        (noisy,    "Gaussian Noise (Simulated Jitter)",        '#ff3366'),
        (scaled,   "Magnitude Scaling (Simulated Distance)",   '#26de81'),
        (warped,   "Time Warping (Simulated Speed Variation)", '#f7b731'),
    ]

    for i, (data, title, color) in enumerate(plot_configs):
        ax = axes[i // 2, i % 2]
        ax.plot(data, color=color, linewidth=2)
        ax.fill_between(range(len(data)), data, alpha=0.15, color=color)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel("Time (Frames)")
        ax.set_ylim(-0.1, 1.2)

        # Only left column needs the Y label
        if i % 2 == 0:
            ax.set_ylabel("Normalised Amplitude")

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    # ── 4. Save / show ────────────────────────────────────────────────
    if args.save:
        out_dir  = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾  Saved → {save_path}")

    if not args.no_show:
        plt.show()

    plt.close(fig) # Always release memory after potentially showing


if __name__ == '__main__':
    main()