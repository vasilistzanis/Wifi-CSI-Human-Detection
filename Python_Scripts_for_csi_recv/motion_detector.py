#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSI Motion Detector & Visualizer — Thesis Grade
================================================
Automatically detects WHEN motion occurs in a CSI recording using
energy-based thresholding on the temporal difference signal.

Theory:
  1. After background subtraction + temporal diff:
       - Static room  → all subcarriers ≈ 0 → low energy
       - Human motion → subcarriers change  → high energy
  2. Per-frame energy:
       energy[t] = sqrt( mean( diff[t, :]² ) )   ← RMS across subcarriers
  3. Adaptive threshold from the background period:
       threshold = bg_mean + k × bg_std   (default k=3, 3-sigma rule)
  4. Smooth energy with a sliding window to avoid false positives
     from single-frame spikes.

Output:
  - Plot with 4 panels showing the full detection chain
  - Detected motion segments printed to console
  - Optionally saves PNG and CSV of motion timestamps

Usage:
  python motion_detector.py
  python motion_detector.py path/to/file.txt
  python motion_detector.py file.txt --save
  python motion_detector.py file.txt --threshold-k 4   # stricter
  python motion_detector.py file.txt --smooth-ms 300   # smoother detector
  python motion_detector.py file.txt --min-duration-ms 200  # min event length
"""

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from csi_plotter_heatmap import load_csi_matrix, resolve_path, get_latest_dataset
from data_preprocessing import CSIPipeline


# ════════════════════════════════════════════════════════════════════════
# ARGS
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="CSI Motion Detector — energy-based, adaptive threshold"
    )
    p.add_argument("file", nargs="?", default=None,
                   help="TXT or CSV file (default: latest in datasets/)")
    p.add_argument("--process-all", action="store_true",
                   help="Process ALL .txt files in the datasets/ folder automatically")
    p.add_argument("--save", action="store_true",
                   help="Save PNG and motion CSV next to dataset file")
    p.add_argument("--export-ml", action="store_true",
                   help="Export fixed-size numpy array (.npy) for ML")
    p.add_argument("--window-frames", type=int, default=300,
                   help="Fixed window size in frames for ML (default: 300 = 3 secs)")
    p.add_argument("--fs", type=float, default=100.0,
                   help="Sampling frequency Hz (default: 100)")
    p.add_argument("--background-frames", type=int, default=100,
                   help="Background calibration frames (default: 100 = 1 s)")
    p.add_argument("--cutoff", type=float, default=12.0,
                   help="Butterworth cutoff Hz (default: 12)")
    p.add_argument("--threshold-k", type=float, default=3.0,
                   help="Threshold multiplier k: threshold = bg_mean + k×bg_std "
                        "(default: 3.0 → 3-sigma rule)")
    p.add_argument("--smooth-ms", type=float, default=200.0,
                   help="Energy smoothing window in ms (default: 200 ms)")
    p.add_argument("--min-duration-ms", type=float, default=150.0,
                   help="Minimum motion event duration in ms (default: 150 ms)")
    p.add_argument("--merge-gap-ms", type=float, default=300.0,
                   help="Merge events closer than this gap in ms (default: 300 ms)")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════
# MOTION DETECTION LOGIC
# ════════════════════════════════════════════════════════════════════════

@dataclass
class MotionEvent:
    start_frame: int
    end_frame:   int
    fs:          float

    @property
    def start_s(self) -> float:
        return self.start_frame / self.fs

    @property
    def end_s(self) -> float:
        return self.end_frame / self.fs

    @property
    def duration_s(self) -> float:
        return (self.end_frame - self.start_frame) / self.fs


def compute_frame_energy(diff_data: np.ndarray) -> np.ndarray:
    """
    Compute RMS energy per frame across all active subcarriers.
    Shape: (N_frames,) — single scalar per time step.

    RMS = sqrt( mean( x² ) ) aggregates all subcarrier contributions
    into one robust scalar that represents "how much is changing".
    """
    return np.sqrt(np.mean(diff_data ** 2, axis=1)).astype(np.float32)


def smooth_energy(energy: np.ndarray, window_frames: int) -> np.ndarray:
    """
    Smooth energy with a uniform sliding window (moving average).
    This reduces false positives from isolated spike frames.
    Uses 'same' convolution so output length == input length.
    """
    if window_frames <= 1:
        return energy
    kernel = np.ones(window_frames, dtype=np.float32) / window_frames
    return np.convolve(energy, kernel, mode='same').astype(np.float32)


def compute_adaptive_threshold(energy: np.ndarray,
                                bg_frames: int,
                                k: float) -> tuple[float, float, float]:
    """
    Estimate threshold from the background (calibration) period.

    Returns:
      threshold : bg_mean + k * bg_std
      bg_mean   : mean energy during background
      bg_std    : std  energy during background

    Why adaptive?
      Different rooms, distances, and antenna orientations produce
      different baseline energy levels. A fixed threshold would need
      manual tuning for every new environment.
      By estimating from the first `bg_frames` (where we know nobody
      is moving), the threshold automatically adapts.
    """
    n_bg = min(bg_frames, energy.shape[0])
    if n_bg < 5:
        # Fallback: use global percentile
        bg_mean = float(np.percentile(energy, 10))
        bg_std  = float(np.std(energy))
    else:
        bg_region = energy[:n_bg]
        bg_mean   = float(bg_region.mean())
        bg_std    = float(bg_region.std())

    # ✅ Floor for bg_std: prevents threshold == bg_mean when signal is
    # perfectly flat (synthetic data / ideal conditions).
    # In real hardware, bg_std is always > 0 due to thermal noise.
    # 1% of bg_mean as minimum, or a tiny absolute floor.
    bg_std = max(bg_std, bg_mean * 0.01, 1e-6)

    threshold = bg_mean + k * bg_std
    return threshold, bg_mean, bg_std


def detect_motion_events(energy_smooth: np.ndarray,
                          threshold: float,
                          fs: float,
                          min_duration_ms: float,
                          merge_gap_ms: float) -> list[MotionEvent]:
    """
    Convert the thresholded binary signal into a list of MotionEvent objects.

    Steps:
      1. Binary mask: above_threshold[t] = energy[t] > threshold
      2. Find rising/falling edges → raw segments
      3. Remove segments shorter than min_duration_ms
      4. Merge segments with gaps shorter than merge_gap_ms

    Returns list of MotionEvent (sorted by start time).
    """
    min_frames   = max(1, int(min_duration_ms * fs / 1000))
    merge_frames = max(1, int(merge_gap_ms    * fs / 1000))

    above = (energy_smooth > threshold).astype(np.int8)

    # Find transitions: rising edge (+1), falling edge (-1)
    diff = np.diff(above, prepend=0, append=0)
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0]

    if len(starts) == 0:
        return []

    # Build raw segments
    segments = [(s, e) for s, e in zip(starts, ends)]

    # Step 3: remove short segments
    segments = [(s, e) for s, e in segments if (e - s) >= min_frames]

    if not segments:
        return []

    # Step 4: merge close segments
    merged = [segments[0]]
    for s, e in segments[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e <= merge_frames:
            merged[-1] = (prev_s, max(prev_e, e))  # extend
        else:
            merged.append((s, e))

    return [MotionEvent(s, e, fs) for s, e in merged]


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Process All Files ─────────────────────────────────────────────────
    if args.process_all:
        import subprocess
        import sys
        default_dir = resolve_path("datasets")
        all_files = sorted(default_dir.glob("*.txt"))
        if not all_files:
            print(f"❌ No .txt files found in {default_dir}")
            sys.exit(1)

        # Forward the same detector settings to every child run so batch mode
        # produces the same results as processing files one by one.
        shared_args = [
            "--fs", str(args.fs),
            "--background-frames", str(args.background_frames),
            "--cutoff", str(args.cutoff),
            "--threshold-k", str(args.threshold_k),
            "--smooth-ms", str(args.smooth_ms),
            "--min-duration-ms", str(args.min_duration_ms),
            "--merge-gap-ms", str(args.merge_gap_ms),
        ]
        if args.export_ml:
            shared_args.append("--export-ml")
            shared_args.extend(["--window-frames", str(args.window_frames)])
        if args.save:
            shared_args.append("--save")

        print(f"🚀 Batch Processing {len(all_files)} files...")
        failures: list[tuple[str, int]] = []

        for i, f in enumerate(all_files, start=1):
            print(f"\n[{i}/{len(all_files)}] Processing {f.name}...")
            cmd = [sys.executable, __file__, *shared_args, str(f)]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                failures.append((f.name, result.returncode))
                print(f"   ❌ Failed with exit code {result.returncode}")

        if failures:
            print("\n❌ Batch processing finished with failures:")
            for name, code in failures:
                print(f"   {name}: exit code {code}")
            sys.exit(1)

        print("\n✅ Batch processing completely finished!")
        return

    # ── File resolution ───────────────────────────────────────────────────
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            sys.exit(1)
    else:
        default_dir = resolve_path("datasets")
        if not default_dir.exists():
            print(f"❌ datasets/ not found — pass a file explicitly")
            sys.exit(1)
        file_path = get_latest_dataset(default_dir)
        if file_path is None:
            print(f"❌ No files in {default_dir}")
            sys.exit(1)

    print(f"\n📂 Loading: {file_path.name}")

    # load_csi_matrix raises ValueError if no valid frames found.
    # Wrap it so we get a clean error message instead of a traceback.
    try:
        complex_matrix, _, seq_stats = load_csi_matrix(file_path)
    except (FileNotFoundError, PermissionError) as e:
        print(f"❌ Cannot open file: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ No valid CSI data: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error loading file: {e}")
        sys.exit(1)

    n_frames = complex_matrix.shape[0]
    duration_s = n_frames / args.fs
    print(f"   {n_frames} frames  |  {duration_s:.1f} s  |  "
          f"loss={seq_stats.loss_percent:.2f}%")

    # ── Pipeline ──────────────────────────────────────────────────────────
    pipeline = CSIPipeline(
        fs=args.fs,
        background_frames=args.background_frames,
        use_diff=True,
    )

    amp       = pipeline.remove_null_subcarriers(complex_matrix, fit=True)
    amp       = pipeline.apply_hampel_filter(amp)
    amp_filt  = pipeline.apply_lowpass_filter(amp, cutoff=args.cutoff)
    amp_bg    = pipeline.apply_background_subtraction(amp_filt, fit=True)
    amp_diff  = pipeline.apply_temporal_diff(amp_bg)   # shape: (N-1, N_active)

    n_active  = amp_filt.shape[1]

    # ── Motion detection ──────────────────────────────────────────────────
    energy_raw = compute_frame_energy(amp_diff)

    smooth_frames = max(1, int(args.smooth_ms * args.fs / 1000))
    energy_smooth = smooth_energy(energy_raw, smooth_frames)

    threshold, bg_mean, bg_std = compute_adaptive_threshold(
        energy_smooth, args.background_frames, args.threshold_k
    )

    events = detect_motion_events(
        energy_smooth, threshold, args.fs,
        args.min_duration_ms, args.merge_gap_ms
    )

    # ── Console report ────────────────────────────────────────────────────
    print(f"\n🔍 Motion Detection Results")
    print(f"   Background: mean={bg_mean:.4f}  std={bg_std:.4f}")
    print(f"   Threshold:  {threshold:.4f}  "
          f"(bg_mean + {args.threshold_k:.1f} × bg_std)")
    print(f"   Smooth window: {smooth_frames} frames ({args.smooth_ms:.0f} ms)")
    print(f"   Min event duration: {args.min_duration_ms:.0f} ms")
    print(f"   Merge gap: {args.merge_gap_ms:.0f} ms")
    print()

    if not events:
        print("   ⚠️  No motion events detected.")
        print("   → Try reducing --threshold-k or check that motion is present")
    else:
        print(f"   ✅ {len(events)} motion event(s) detected:\n")
        for i, ev in enumerate(events):
            print(f"   [{i+1:2d}]  {ev.start_s:6.2f} s → {ev.end_s:6.2f} s  "
                  f"(duration: {ev.duration_s*1000:.0f} ms)")

    # ── Time axes ─────────────────────────────────────────────────────────
    # amp_filt has N frames, amp_diff has N-1 → offset by 1 frame
    t_filt   = np.arange(amp_filt.shape[0])  / args.fs
    t_energy = np.arange(energy_smooth.shape[0]) / args.fs  # same as diff = N-1

    # ── Figure ────────────────────────────────────────────────────────────
    for style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot']:
        try:
            plt.style.use(style)
            break
        except Exception:
            continue

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.facecolor": "#fafafa",
        "figure.facecolor": "#ffffff",
        "grid.alpha": 0.4,
        "grid.linewidth": 0.4,
    })

    # Select representative subcarriers spread across the active spectrum.
    # Use linspace to guarantee indices stay in-bounds and are unique.
    n_show = min(4, n_active)
    if n_active >= 2:
        raw_idx = np.linspace(max(1, n_active // 5),
                              n_active - 1 - max(1, n_active // 5),
                              n_show, dtype=int)
        # Deduplicate while preserving order
        sc_idx = list(dict.fromkeys(raw_idx.tolist()))
        n_show = len(sc_idx)
    else:
        sc_idx = [0]
        n_show = 1
    sc_colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd'][:n_show]

    fig, axes = plt.subplots(4, 1, figsize=(16, 12),
                              gridspec_kw={'hspace': 0.55, 'height_ratios': [2, 1.5, 2, 1]})

    fig.suptitle(
        f"CSI Motion Detection  ·  {file_path.name}\n"
        f"{n_frames} frames  ·  {n_active} active subcarriers  ·  "
        f"duration {duration_s:.1f} s  ·  "
        f"{len(events)} event(s)  ·  "
        f"threshold k={args.threshold_k}",
        fontsize=12, fontweight='bold', y=0.985, color="#111111"
    )

    # ── Helper: shade motion events on any axis ───────────────────────────
    def shade_events(ax, events, alpha=0.18):
        for ev in events:
            ax.axvspan(ev.start_s, ev.end_s,
                       color="#e63946", alpha=alpha, zorder=0)

    def shade_bg(ax):
        ax.axvspan(0, args.background_frames / args.fs,
                   color="#ffcc00", alpha=0.12, zorder=0)
        ax.axvline(args.background_frames / args.fs,
                   color="#ffcc00", linewidth=0.9, linestyle="--", alpha=0.7)

    # ════════════════════════════════════════════════════════════════════
    # PANEL 0 — Filtered CSI amplitude (subcarrier view)
    # Shows the "raw" signal the human analyst would see.
    # ════════════════════════════════════════════════════════════════════
    ax = axes[0]
    for i, sc in enumerate(sc_idx):
        ax.plot(t_filt, amp_filt[:, sc],
                color=sc_colors[i], linewidth=0.9, alpha=0.85,
                label=f"SC {sc}")
    shade_bg(ax)
    shade_events(ax, events)
    ax.set_title("① Filtered CSI Amplitude  (Butterworth low-pass · human view)",
                 fontsize=10, fontweight='bold', pad=5)
    ax.set_ylabel("Amplitude (a.u.)", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, ncol=n_show, framealpha=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)

    # ════════════════════════════════════════════════════════════════════
    # PANEL 1 — After background subtraction + temporal diff
    # Static room → ≈ 0. Motion → visible peaks.
    # ════════════════════════════════════════════════════════════════════
    ax = axes[1]
    for i, sc in enumerate(sc_idx):
        ax.plot(t_energy, amp_diff[:, sc],
                color=sc_colors[i], linewidth=0.8, alpha=0.75,
                label=f"SC {sc}")
    ax.axhline(0, color="#999999", linewidth=0.7, linestyle="--")
    shade_events(ax, events)
    ax.set_title("② Background Subtraction + Temporal Diff  "
                 "(static room removed — only motion remains)",
                 fontsize=10, fontweight='bold', pad=5)
    ax.set_ylabel("Δ Amplitude", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, ncol=n_show, framealpha=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)

    # ════════════════════════════════════════════════════════════════════
    # PANEL 2 — RMS Energy + smoothed energy + threshold
    # This is the "detector" signal. Everything above threshold = motion.
    # ════════════════════════════════════════════════════════════════════
    ax = axes[2]

    # Raw energy (thin, faded)
    ax.plot(t_energy, energy_raw,
            color="#aaaaaa", linewidth=0.7, alpha=0.6,
            label="Raw energy", zorder=2)

    # Smoothed energy (bold)
    ax.plot(t_energy, energy_smooth,
            color="#2a9d8f", linewidth=2.0, alpha=0.95,
            label=f"Smoothed energy ({args.smooth_ms:.0f} ms window)", zorder=3)

    # Threshold line
    ax.axhline(threshold, color="#e63946", linewidth=1.5,
               linestyle="--", zorder=4,
               label=f"Threshold = bg_mean + {args.threshold_k:.1f}×bg_std"
                     f" = {threshold:.4f}")

    # Background mean line
    ax.axhline(bg_mean, color="#f4a261", linewidth=0.9,
               linestyle=":", alpha=0.8, zorder=3,
               label=f"Background mean = {bg_mean:.4f}")

    # Shade detected motion regions
    shade_events(ax, events, alpha=0.15)
    shade_bg(ax)

    ax.set_title("③ Frame Energy (RMS across subcarriers)  "
                 "+  Adaptive Threshold  →  Motion Detector",
                 fontsize=10, fontweight='bold', pad=5)
    ax.set_ylabel("RMS Energy", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)

    # Annotate each event with its index and duration
    for i, ev in enumerate(events):
        mid = (ev.start_s + ev.end_s) / 2
        peak_e = float(energy_smooth[ev.start_frame:ev.end_frame].max())
        ax.annotate(
            f"#{i+1}\n{ev.duration_s*1000:.0f}ms",
            xy=(mid, peak_e),
            xytext=(0, 10), textcoords='offset points',
            ha='center', fontsize=7, color="#e63946", fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#e63946',
                            lw=0.8, connectionstyle='arc3,rad=0')
        )

    # ════════════════════════════════════════════════════════════════════
    # PANEL 3 — Binary motion mask (ground truth quality annotation)
    # This is what you'd use as label for the ML model.
    # ════════════════════════════════════════════════════════════════════
    ax = axes[3]
    binary_mask = np.zeros(len(t_energy), dtype=np.float32)
    for ev in events:
        binary_mask[ev.start_frame:ev.end_frame] = 1.0

    ax.fill_between(t_energy, binary_mask,
                    color="#e63946", alpha=0.7, step='mid',
                    label="MOTION DETECTED")
    ax.fill_between(t_energy, 1 - binary_mask,
                    color="#2a9d8f", alpha=0.25, step='mid',
                    label="NO MOTION")

    # Mark background calibration period separately
    shade_bg(ax)
    ax.set_ylim(-0.1, 1.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No Motion", "Motion"], fontsize=8)
    ax.set_title("④ Binary Motion Label  (ready for ML dataset annotation)",
                 fontsize=10, fontweight='bold', pad=5)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)

    # ── Shared x-axis formatting ──────────────────────────────────────────
    for ax in axes:
        ax.set_xlim(0, max(t_filt[-1], t_energy[-1]))
        ax.grid(True, linewidth=0.4, alpha=0.5)

    # ── Global legend for annotations ────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor="#e63946", alpha=0.5,
                       label="Motion event (detected)"),
        mpatches.Patch(facecolor="#ffcc00", alpha=0.4,
                       label=f"Background calibration "
                             f"(first {args.background_frames} frames = "
                             f"{args.background_frames/args.fs:.1f} s)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.005))

    # tight_layout warns when a figure-level legend is present, so reserve
    # the margins explicitly instead of relying on automatic layout.
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.08, hspace=0.55)

    # ── Save ──────────────────────────────────────────────────────────────
    if args.save:
        # PNG
        out_png = file_path.parent / (file_path.stem + "_motion.png")
        fig.savefig(out_png, dpi=200, bbox_inches="tight",
                    facecolor="#ffffff")
        print(f"\n💾 PNG saved: {out_png}")

        # CSV with motion timestamps (useful for labeling ML dataset)
        if events:
            import csv
            out_csv = file_path.parent / (file_path.stem + "_motion.csv")
            with open(out_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "event_id", "start_frame", "end_frame",
                    "start_s", "end_s", "duration_ms"
                ])
                for i, ev in enumerate(events):
                    writer.writerow([
                        i + 1, ev.start_frame, ev.end_frame,
                        f"{ev.start_s:.3f}", f"{ev.end_s:.3f}",
                        f"{ev.duration_s * 1000:.1f}"
                    ])
            print(f"💾 Motion CSV saved: {out_csv}")

    # ── Export for Machine Learning (Windowing) ───────────────────────────
    if args.export_ml and events:
        # Using amp_filt as the clean 'processed' data
        processed = amp_filt
        num_features = processed.shape[1]
        
        for i, ev in enumerate(events):
            # 1. Start from the center of motion
            center = (ev.start_frame + ev.end_frame) // 2
            half_win = args.window_frames // 2
            
            w_start = center - half_win
            w_end = center + (args.window_frames - half_win)
            
            # 2. Setup zero-padded window for consistent sizing
            window = np.zeros((args.window_frames, num_features), dtype=np.float32)
            
            # 3. Calculate bounded indices
            src_start = max(0, w_start)
            src_end = min(processed.shape[0], w_end)
            
            dst_start = src_start - w_start
            dst_end = dst_start + (src_end - src_start)
            
            # 4. Copy the data into the padded window
            window[dst_start:dst_end, :] = processed[src_start:src_end, :]
            
            # 5. Save as NPY
            out_npy = file_path.parent / f"{file_path.stem}_ml_ev{i+1}.npy"
            np.save(out_npy, window)
            print(f"📦 ML Window saved: {out_npy} (Shape: {window.shape})")

    if not args.export_ml:
        plt.show()
    plt.rcParams.update(plt.rcParamsDefault)


if __name__ == "__main__":
    main()
