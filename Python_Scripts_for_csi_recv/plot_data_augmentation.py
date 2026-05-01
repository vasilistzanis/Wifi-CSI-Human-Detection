import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import argparse
from typing import Optional


# Import the loader from your existing preprocessing script
try:
    from data_preprocessing import load_csi_csv, CSIPipeline
except ImportError:
    print("Error: data_preprocessing.py not found in the current directory.")
    sys.exit(1)


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------


import config

RANDOM_SEED    = config.RANDOM_SEED     # reproducible plots
MIN_FRAMES     = 200    # minimum recording length required
SEGMENT_LEN    = config.WINDOW_SIZE    # frames per segment used for visualization
SUBCARRIER_IDX = 30     # preferred subcarrier (bounds-checked below)


# ---------------------------------------------------------------------
# AUGMENTATION FUNCTIONS
# ---------------------------------------------------------------------


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


def augment_shift(window: np.ndarray,
                  shift_steps: int = 15,
                  direction: int = 1) -> np.ndarray:
    """
    Non-circular temporal shift to simulate sensor start delay.

    Args:
        window:      1-D signal array.
        shift_steps: Number of frames to shift.
        direction:   1 (forward) or -1 (backward).
    """
    if direction == 1:
        pad = np.repeat(window[0], shift_steps)
        return np.concatenate([pad, window[:-shift_steps]])
    else:
        pad = np.repeat(window[-1], shift_steps)
        return np.concatenate([window[shift_steps:], pad])




def augment_time_warp(window: np.ndarray,
                      stretch: float = 1.5) -> np.ndarray:
    """
    Stretch or compress signal in time to simulate speed variation.


    The output always has the same length as the input:
      stretch > 1 -> resample to longer, then crop centre.
      stretch < 1 -> resample to shorter, then pad with edge value.


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




# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


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
    Load a real CSI recording, apply standard filtering, and return (signal_1d, info_str).

    Returns (None, error_str) on any failure so the caller can fall back.
    """
    complex_matrix, _ = load_csi_csv(data_path)

    if complex_matrix is None or complex_matrix.size == 0:
        return None, "Loaded matrix is empty"

    n_frames, n_sc = complex_matrix.shape

    # Guard against short recordings
    if n_frames < min_frames:
        return None, (f"Recording too short: {n_frames} frames "
                      f"(need >= {min_frames})")

    # --- APPLY FILTERING (CSIPipeline) ---
    pipeline = CSIPipeline(fs=config.SAMPLING_RATE, use_diff=False)
    amp_active = pipeline.remove_null_subcarriers(complex_matrix, fit=True)
    amp_clean = pipeline.apply_hampel_filter(amp_active)
    amp_clean = pipeline.apply_lowpass_filter(amp_clean)
    
    n_frames_clean, n_active_sc = amp_clean.shape
    if n_active_sc == 0:
        return None, "No active subcarriers after filtering"

    # Bounds check: clamp subcarrier index to valid range of ACTIVE subcarriers
    sc_idx = min(subcarrier_idx, n_active_sc - 1)
    if sc_idx != subcarrier_idx:
        print(f"  [WARNING]  subcarrier {subcarrier_idx} out of range "
              f"(matrix has {n_active_sc} ACTIVE SC) - using SC {sc_idx} instead")

    # Use max(0, ...) to guarantee non-negative start_frame
    start = max(0, min(500, n_frames_clean - segment_len))
    end   = start + segment_len

    amplitude = amp_clean[start:end, sc_idx]

    # Normalise to [0, 1] for visual consistency across augmentations
    mn, mx  = amplitude.min(), amplitude.max()
    signal  = (amplitude - mn) / (mx - mn + 1e-9)

    info = (f"Filtered SC {sc_idx}  |  frames {start}-{end}  "
            f"|  total {n_frames_clean} frames  |  {n_active_sc} active SC")
    return signal, info




# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


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
    parser.add_argument('--min-frames', type=int, default=MIN_FRAMES,
                        help=f"Minimum frames required (default: {MIN_FRAMES})")
    parser.add_argument('--file', type=str, default="datasets/walk/walk_01.txt",
                        help="Path to real CSI data file (default: datasets/walk/walk_01.txt)")
    parser.add_argument('--subcarrier', type=int, default=SUBCARRIER_IDX,
                        help=f"Subcarrier index to plot (default: {SUBCARRIER_IDX})")
    parser.add_argument('--segment-len', type=int, default=SEGMENT_LEN,
                        help=f"Length of signal segment to plot (default: {SEGMENT_LEN})")
    parser.add_argument('--realistic', action='store_true',
                        help="Use actual subtle ML parameters instead of exaggerated visual ones")
    args = parser.parse_args()


    # Create a single seeded RNG passed to all augmentation funcs
    rng = np.random.default_rng(seed=RANDOM_SEED)


    # -- 1. Select data source -----------------------------------------
    title_prefix = "Synthetic"
    filename     = "Synthetic_Data_Augmentation_Vis.png"
    original     = None


    if not args.simulate:
        data_path = Path(args.file)
        if not data_path.exists():
            print(f"[ERROR]  File not found: {data_path} - falling back to synthetic data")
        else:
            print(f"[FILE]  Loading real data from {data_path} ...")
            # We pass segment_len and subcarrier_idx from args
            original, info = _load_real(data_path, 
                                      min_frames=args.min_frames,
                                      segment_len=args.segment_len,
                                      subcarrier_idx=args.subcarrier)
            if original is None:
                print(f"[ERROR]  {info} - falling back to synthetic data")
            else:
                print(f"[OK]  {info}")
                title_prefix = "Real"
                filename     = "Real_Data_Augmentation_Vis.png"


    if original is None:
        print("[INFO] Mode: SYNTHETIC")
        original = _make_synthetic(segment_len=args.segment_len)
        title_prefix = "Synthetic"

    # -- 2. Set Parameters ---------------------------------------------
    if args.realistic:
        print("[INFO] Using REALISTIC parameters (ML training equivalents)")
        noise_param = 0.005
        shift_param = 2
        scale_param = 0.95
        warp_param  = 1.05
        filename    = f"{title_prefix}_Data_Aug_Realistic.png"
        title_suf   = "(Realistic ML Params)"
    else:
        print("[INFO] Using EXAGGERATED parameters (Best for visualization)")
        noise_param = 0.04
        shift_param = 15
        scale_param = 0.6
        warp_param  = 1.5
        filename    = f"{title_prefix}_Data_Aug_Exaggerated.png"
        title_suf   = "(Exaggerated Params)"

    # -- 3. Apply augmentations ----------------------------------------
    noisy   = augment_noise(original,     noise_level=noise_param, rng=rng)
    shifted = augment_shift(original,     shift_steps=shift_param, direction=1)
    scaled  = augment_scale(original,     scale_factor=scale_param)
    warped  = augment_time_warp(original, stretch=warp_param)


    # -- 4. Plot -------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharey=True)
    axes_flat = axes.flatten()
    fig.suptitle(
        f"Data Augmentation Techniques - {title_prefix} CSI Signal {title_suf}",
        fontsize=16, fontweight='bold', y=0.96,
    )


    plot_configs = [
        (original, f"Original {title_prefix} Signal",         '#00e5ff'),
        (noisy,    "Gaussian Noise (Simulated Jitter)",        '#ff3366'),
        (shifted,  "Temporal Shift (Simulated Delay)",         '#9b59b6'),
        (scaled,   "Magnitude Scaling (Simulated Distance)",   '#26de81'),
        (warped,   "Time Warping (Simulated Speed Variation)", '#f7b731'),
    ]


    for i, (data, title, color) in enumerate(plot_configs):
        ax = axes_flat[i]
        ax.plot(data, color=color, linewidth=2)
        ax.fill_between(range(len(data)), data, alpha=0.15, color=color)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel("Time (Frames)")
        ax.set_ylim(-0.1, 1.2)


        # Only left column needs the Y label
        if i % 3 == 0:
            ax.set_ylabel("Normalised Amplitude")

    # Turn off the empty 6th subplot
    axes_flat[-1].axis('off')


    plt.tight_layout(rect=[0, 0.03, 1, 0.93])


    # -- 5. Save / show ------------------------------------------------
    if args.save:
        out_dir  = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVE]  Saved -> {save_path}")


    if not args.no_show:
        plt.show()


    plt.close(fig) # Always release memory after potentially showing




if __name__ == '__main__':
    main()
