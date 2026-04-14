#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSI Motion Detector & Visualizer — Thesis Grade

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
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import StandardScaler

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
                   help="Process ALL .txt and .csv files in the datasets/ folder automatically")
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
    p.add_argument("--threshold-k-high", type=float, default=2.5,
                   help="High threshold multiplier (trigger event) (default: 2.5)")
    p.add_argument("--threshold-k-low", type=float, default=1.2,
                   help="Low threshold multiplier (end event) (default: 1.2)")
    p.add_argument("--smooth-ms", type=float, default=350.0,
                   help="Energy smoothing window in ms (default: 350 ms)")
    p.add_argument("--min-duration-ms", type=float, default=150.0,
                   help="Minimum motion event duration in ms (default: 150 ms)")
    p.add_argument("--merge-gap-ms", type=float, default=800.0,
                   help="Merge events closer than this gap in ms (default: 800 ms)")
    p.add_argument("--min-peak-ratio", type=float, default=2.5,
                   help="Minimum peak_energy/bg_mean ratio for ML export (default: 2.5). "
                        "Events below this are weak/out-of-LOS and are skipped.")
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


def compute_adaptive_thresholds(energy: np.ndarray,
                                bg_frames: int,
                                k_high: float,
                                k_low: float) -> tuple[float, float, float, float]:
    """
    Estimate hysteresis thresholds from the background (calibration) period.

    Returns:
      thresh_high : bg_mean + k_high * bg_std
      thresh_low  : bg_mean + k_low  * bg_std
      bg_mean   : mean energy during background
      bg_std    : std  energy during background
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
    bg_std = max(bg_std, bg_mean * 0.01, 1e-6)

    thresh_high = bg_mean + k_high * bg_std
    thresh_low  = bg_mean + k_low * bg_std
    return thresh_high, thresh_low, bg_mean, bg_std


def detect_motion_events(energy_smooth: np.ndarray,
                         thresh_high: float,
                         thresh_low: float,
                         fs: float,
                         min_duration_ms: float,
                         merge_gap_ms: float) -> list[MotionEvent]:
    """
    Convert the analog energy signal into MotionEvents using Hysteresis (Double Threshold).
    """
    min_frames   = max(1, int(min_duration_ms * fs / 1000))
    merge_frames = max(1, int(merge_gap_ms    * fs / 1000))

    above_high = energy_smooth > thresh_high
    below_low = energy_smooth < thresh_low

    in_event = False
    starts = []
    ends = []

    for i in range(len(energy_smooth)):
        if not in_event:
            if above_high[i]:
                in_event = True
                starts.append(i)
        else:
            if below_low[i]:
                in_event = False
                ends.append(i)

    if in_event:
        ends.append(len(energy_smooth))

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
        all_files = sorted(
            list(default_dir.glob("*.txt")) + list(default_dir.glob("*.csv"))
        )
        if not all_files:
            print(f"❌ No .txt or .csv files found in {default_dir}")
            sys.exit(1)

        # Forward the same detector settings to every child run so batch mode
        # produces the same results as processing files one by one.
        shared_args = [
            "--fs", str(args.fs),
            "--background-frames", str(args.background_frames),
            "--cutoff", str(args.cutoff),
            "--threshold-k-high", str(args.threshold_k_high),
            "--threshold-k-low", str(args.threshold_k_low),
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

    if amp_diff.shape[0] < 2:
        print("❌ Need at least 2 frames after preprocessing for motion detection.")
        sys.exit(1)

    # NaN/Inf check: a corrupted recording or very short background period can
    # cause Butterworth or background subtraction to produce NaN/Inf.
    # If energy = NaN → threshold = NaN → detect_motion_events returns
    # wrong results silently (either all frames or no frames flagged as motion).
    if not np.all(np.isfinite(amp_diff)):
        n_bad = int(np.sum(~np.isfinite(amp_diff)))
        print(f"❌ amp_diff contains {n_bad} NaN/Inf values after preprocessing.")
        print("   Possible causes: recording too short for Butterworth filter, "
              "or all-zero subcarriers after null removal.")
        sys.exit(1)

    n_active  = amp_filt.shape[1]

    # ── Motion detection ──────────────────────────────────────────────────
    energy_raw = compute_frame_energy(amp_diff)

    smooth_frames = max(1, int(args.smooth_ms * args.fs / 1000))
    energy_smooth = smooth_energy(energy_raw, smooth_frames)

    thresh_high, thresh_low, bg_mean, bg_std = compute_adaptive_thresholds(
        energy_smooth, args.background_frames, args.threshold_k_high, args.threshold_k_low
    )

    events = detect_motion_events(
        energy_smooth, thresh_high, thresh_low, args.fs,
        args.min_duration_ms, args.merge_gap_ms
    )

    # ── Console report ────────────────────────────────────────────────────
    print(f"\n🔍 Motion Detection Results")
    print(f"   Background: mean={bg_mean:.4f}  std={bg_std:.4f}")
    print(f"   High Threshold:  {thresh_high:.4f}  (bg_mean + {args.threshold_k_high:.1f} × bg_std)")
    print(f"   Low  Threshold:  {thresh_low:.4f}  (bg_mean + {args.threshold_k_low:.1f} × bg_std)")
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
            # Compute peak ratio for this event (quality score)
            ev_energy = energy_smooth[ev.start_frame:ev.end_frame]
            ev_peak   = float(ev_energy.max()) if len(ev_energy) > 0 else 0.0
            ev_ratio  = ev_peak / bg_mean if bg_mean > 0 else 0.0
            quality_ok = ev_ratio >= args.min_peak_ratio
            quality_tag = "✅" if quality_ok else f"⚠️  WEAK (ratio={ev_ratio:.2f} < {args.min_peak_ratio})"
            print(f"   [{i+1:2d}]  {ev.start_s:6.2f} s → {ev.end_s:6.2f} s  "
                  f"(duration: {ev.duration_s*1000:.0f} ms)  "
                  f"peak_ratio={ev_ratio:.2f}x  {quality_tag}")

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
        f"thresholds: High({args.threshold_k_high}), Low({args.threshold_k_low})",
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

    # High Threshold line
    ax.axhline(thresh_high, color="#e63946", linewidth=1.5,
               linestyle="--", zorder=4,
               label=f"High Threshold: {thresh_high:.4f} (+{args.threshold_k_high}σ)")
    # Low Threshold line
    ax.axhline(thresh_low, color="#e63946", linewidth=1.2,
               linestyle=":", alpha=0.8, zorder=4,
               label=f"Low Threshold: {thresh_low:.4f} (+{args.threshold_k_low}σ)")

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
    fname_lower    = file_path.name.lower()
    is_idle_file   = "idle"  in fname_lower
    is_action_file = any(kw in fname_lower for kw in ("fall", "sit"))

    if args.export_ml and (events or is_idle_file):
        # ✅ FIX: Use amp_diff (background subtracted + temporal diff) with
        # StandardScaler normalization instead of raw amp_filt.
        #
        # Previous code used amp_filt (only Butterworth filtered) which:
        #   - Had no background subtraction → static room baseline still present
        #   - Had no temporal differencing  → absolute amplitude, not rate-of-change
        #   - Had no normalization          → values not ML-friendly
        #
        # amp_diff already has:
        #   ✅ Null subcarrier removal
        #   ✅ Hampel outlier filter
        #   ✅ Butterworth low-pass filter
        #   ✅ Background subtraction (static room removed)
        #   ✅ Temporal differencing (environment-independent)
        #
        # StandardScaler adds:
        #   ✅ Z-score normalization → mean=0, std=1 per subcarrier
        #   ✅ Consistent scale for LSTM training regardless of environment
        #
        # Shape: (N-1, 114) → same as before, compatible with train_lstm.py defaults.
        scaler = StandardScaler()
        processed = scaler.fit_transform(amp_diff)

        num_features = processed.shape[1]

        print(f"\n📦 Exporting ML windows  "
              f"(processed shape: {processed.shape}, "
              f"features: {num_features}, "
              f"window: {args.window_frames} frames)")

        if is_idle_file:
            print("   ℹ️ 'idle' label detected in filename. Extracting quiet windows from the middle of the recording, bypassing detected motion.")
            N = processed.shape[0]
            
            if N < args.window_frames:
                print("   ⚠️ File is too short for even a single idle window.")
            else:
                # Calculate safe middle region (assume first 20% and last 20% are noisy due to walking)
                safe_start = int(N * 0.2)
                safe_end = int(N * 0.8)
                safe_duration = safe_end - safe_start
                
                # If safe duration is smaller than a window, just take the absolute center
                if safe_duration <= args.window_frames:
                    centers = [N // 2]
                else:
                    # Extract up to 3 non-overlapping windows from the safe middle region
                    num_idle_windows = min(3, safe_duration // args.window_frames)
                    if num_idle_windows < 1: 
                        num_idle_windows = 1
                    
                    # Space them out evenly within the safe region
                    step = safe_duration // (num_idle_windows + 1)
                    centers = [safe_start + step * (j + 1) for j in range(num_idle_windows)]
                
                for i, center in enumerate(centers):
                    half_win = args.window_frames // 2
                    w_start = center - half_win
                    w_end = center + (args.window_frames - half_win)
                    
                    window = np.zeros((args.window_frames, num_features), dtype=np.float32)
                    src_start = max(0, w_start)
                    src_end = min(processed.shape[0], w_end)
                    dst_start = src_start - w_start
                    dst_end = dst_start + (src_end - src_start)
                    
                    window[dst_start:dst_end, :] = processed[src_start:src_end, :]
                    
                    out_npy = file_path.parent / f"{file_path.stem}_ml_ev{i+1}.npy"
                    np.save(out_npy, window)
                    print(f"   💾 {out_npy.name}  shape={window.shape}")
                    
        else:
            exported_count = 0

            # ── Safe Export Zone for fall/sit ─────────────────────────────
            # When recording fall/sit, someone must physically get up at the
            # end to stop the recording → this creates a false motion event
            # in the last portion of the file. We ignore events centered in
            # the last 20% of the recording to eliminate this artifact.
            N_frames = processed.shape[0]
            if is_action_file:
                safe_zone_end = int(N_frames * 0.80)
                print(f"   ℹ️  'fall'/'sit' label detected — Safe Export Zone: "
                      f"first 80% ({safe_zone_end}/{N_frames} frames). "
                      f"Events in the last 20% (closing motion) will be ignored.")
            else:
                safe_zone_end = N_frames  # walk: no restriction

            for i, ev in enumerate(events):
                # ── Safe Zone Check (fall/sit only) ───────────────────────
                ev_center = (ev.start_frame + ev.end_frame) // 2
                if ev_center > safe_zone_end:
                    print(f"   ⚠️  Event #{i+1} SKIPPED  "
                          f"center={ev_center} > safe_zone={safe_zone_end}  "
                          f"→ closing-motion artifact at end of recording")
                    continue

                # ── Event Quality Gate ────────────────────────────────────
                ev_energy = energy_smooth[ev.start_frame:ev.end_frame]
                ev_peak   = float(ev_energy.max()) if len(ev_energy) > 0 else 0.0
                ev_ratio  = ev_peak / bg_mean if bg_mean > 0 else 0.0

                if ev_ratio < args.min_peak_ratio:
                    print(f"   ⚠️  Event #{i+1} SKIPPED  "
                          f"peak_ratio={ev_ratio:.2f} < {args.min_peak_ratio:.2f}  "
                          f"→ weak/out-of-LOS signal, not suitable for ML")
                    continue

                # 1. Center of motion window
                center   = ev_center
                half_win = args.window_frames // 2

                w_start = center - half_win
                w_end   = center + (args.window_frames - half_win)

                # 2. Zero-padded window
                window = np.zeros((args.window_frames, num_features), dtype=np.float32)

                # 3. Bounded indices
                src_start = max(0, w_start)
                src_end   = min(processed.shape[0], w_end)
                dst_start = src_start - w_start
                dst_end   = dst_start + (src_end - src_start)

                # 4. Copy data
                window[dst_start:dst_end, :] = processed[src_start:src_end, :]

                # 5. Save
                exported_count += 1
                out_npy = file_path.parent / f"{file_path.stem}_ml_ev{exported_count}.npy"
                np.save(out_npy, window)
                print(f"   💾 {out_npy.name}  shape={window.shape}  "
                      f"peak_ratio={ev_ratio:.2f}x  ✅")

            if exported_count == 0:
                print(f"   ❌ No events passed the quality gate "
                      f"(min_peak_ratio={args.min_peak_ratio:.2f}).")
                print(f"   → Lower --min-peak-ratio or capture data closer to LOS.")
            else:
                print(f"\n   ✅ {exported_count}/{len(events)} event(s) exported to NPY.")

    if not args.export_ml:
        plt.show()
    plt.rcParams.update(plt.rcParamsDefault)

if __name__ == "__main__":
    main()