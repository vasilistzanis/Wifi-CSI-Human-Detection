#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSI Thesis Figures — Complete Publication-Ready Plot Set
========================================================
Generates 6 thesis-grade figures from a single CSI dataset file.

  🔴 CORE (3 plots):
    1. Amplitude vs Time        — human motion signature
    2. Heatmap (Time × SC)      — spatial-temporal pattern
    3. Amplitude vs Subcarriers — channel fingerprint

  🟡 SUPPORT (1 plot):
    4. Variance / Energy vs Time — motion intensity

  🔵 ADVANCED (2 plots):
    5. FFT / Spectrogram        — frequency domain analysis
    6. Phase vs Time            — multipath sensitivity

Usage:
  python plot_thesis_figures.py                              # latest file
  python plot_thesis_figures.py datasets/walk/walk_01.txt
  python plot_thesis_figures.py walk_01.txt --save
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
from scipy.signal import spectrogram

from csi_parser import (
    configure_console_output,
    resolve_path,
    get_latest_dataset,
    load_csi_matrix,
)

configure_console_output()

try:
    matplotlib.use("Qt5Agg")
except Exception:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

import matplotlib.pyplot as plt
plt.ioff()


# ════════════════════════════════════════════════════════════════════════
# STYLE
# ════════════════════════════════════════════════════════════════════════

# Professional light theme for thesis/paper printing
STYLE = {
    "bg":       "#ffffff",
    "panel":    "#fafafa",
    "text":     "#1a1a1a",
    "grid":     "#e0e0e0",
    "accent1":  "#2563eb",   # blue
    "accent2":  "#f59e0b",   # amber
    "accent3":  "#10b981",   # green
    "accent4":  "#ef4444",   # red
    "accent5":  "#8b5cf6",   # purple
    "accent6":  "#06b6d4",   # cyan
}

PALETTE = [STYLE["accent1"], STYLE["accent2"], STYLE["accent3"],
           STYLE["accent4"], STYLE["accent5"], STYLE["accent6"]]


def _apply_style():
    for style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot']:
        try:
            plt.style.use(style)
            break
        except Exception:
            continue
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.facecolor":   STYLE["panel"],
        "figure.facecolor": STYLE["bg"],
        "axes.grid":        True,
        "grid.alpha":       0.4,
        "grid.linewidth":   0.5,
        "axes.spines.top":  False,
        "axes.spines.right": False,
    })


def _save_fig(fig, save_dir: Path, name: str):
    out = save_dir / f"{name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=STYLE["bg"])
    print(f"  💾 {out.name}")


def _get_active(complex_matrix):
    """Return amplitude of active (non-zero) subcarriers and their indices."""
    amp = np.abs(complex_matrix)
    mask = np.any(amp > 0, axis=0)
    return amp[:, mask], np.flatnonzero(mask), mask


# ════════════════════════════════════════════════════════════════════════
# 🔴 PLOT 1 — Amplitude vs Time
# ════════════════════════════════════════════════════════════════════════

def plot_amplitude_vs_time(complex_matrix, fs, title_base, save_dir, save):
    amp_active, indices, _ = _get_active(complex_matrix)
    n_frames = amp_active.shape[0]
    t = np.arange(n_frames) / fs

    mean_amp = amp_active.mean(axis=1)
    std_amp = amp_active.std(axis=1)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t, mean_amp, color=STYLE["accent1"], linewidth=1.2, label="Mean Amplitude")
    ax.fill_between(t, mean_amp - std_amp, mean_amp + std_amp,
                    alpha=0.2, color=STYLE["accent1"], label="±1 σ")

    # Overlay 3 representative subcarriers
    n_sc = amp_active.shape[1]
    for i, sc_idx in enumerate([n_sc // 4, n_sc // 2, 3 * n_sc // 4]):
        if sc_idx < n_sc:
            ax.plot(t, amp_active[:, sc_idx], linewidth=0.6, alpha=0.5,
                    color=PALETTE[(i + 1) % len(PALETTE)],
                    label=f"SC {indices[sc_idx]}")

    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_ylabel("CSI Amplitude |H|", fontweight="bold")
    ax.set_title(f"① CSI Amplitude vs Time\n{title_base}", fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    fig.tight_layout()
    if save:
        _save_fig(fig, save_dir, "01_amplitude_vs_time")
    return fig


# ════════════════════════════════════════════════════════════════════════
# 🔴 PLOT 2 — Heatmap (Time × Subcarriers)
# ════════════════════════════════════════════════════════════════════════

def plot_heatmap(complex_matrix, fs, title_base, save_dir, save):
    amp_active, indices, _ = _get_active(complex_matrix)
    n_frames = amp_active.shape[0]

    vmin = np.percentile(amp_active, 2)
    vmax = np.percentile(amp_active, 98)

    fig, ax = plt.subplots(figsize=(14, 6))
    extent = [0, n_frames / fs, 0, amp_active.shape[1]]
    im = ax.imshow(amp_active.T, aspect="auto", cmap="viridis",
                   interpolation="nearest", origin="lower",
                   vmin=vmin, vmax=vmax, extent=extent)
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_ylabel("Active Subcarrier (sequential)", fontweight="bold")
    ax.set_title(f"② CSI Amplitude Heatmap (Time × Subcarriers)\n{title_base}",
                 fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, label="Amplitude |H|", shrink=0.8)
    fig.tight_layout()
    if save:
        _save_fig(fig, save_dir, "02_heatmap")
    return fig


# ════════════════════════════════════════════════════════════════════════
# 🔴 PLOT 3 — Amplitude vs Subcarriers (Channel Fingerprint)
# ════════════════════════════════════════════════════════════════════════

def plot_subcarrier_profile(complex_matrix, title_base, save_dir, save):
    amp_active, indices, _ = _get_active(complex_matrix)

    mean_amp = amp_active.mean(axis=0)
    std_amp = amp_active.std(axis=0)
    median_amp = np.median(amp_active, axis=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(indices, mean_amp, color=STYLE["accent2"], linewidth=2.0,
            label="Mean", zorder=3)
    ax.fill_between(indices, mean_amp - std_amp, mean_amp + std_amp,
                    alpha=0.2, color=STYLE["accent2"])
    ax.plot(indices, median_amp, color=STYLE["accent3"], linewidth=1.2,
            linestyle="--", alpha=0.8, label="Median")

    ax.set_xlabel("Subcarrier Index", fontweight="bold")
    ax.set_ylabel("Amplitude |H|", fontweight="bold")
    ax.set_title(f"③ Channel Frequency Response (Subcarrier Profile)\n{title_base}",
                 fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.8)
    fig.tight_layout()
    if save:
        _save_fig(fig, save_dir, "03_subcarrier_profile")
    return fig


# ════════════════════════════════════════════════════════════════════════
# 🟡 PLOT 4 — Variance / Signal Energy vs Time
# ════════════════════════════════════════════════════════════════════════

def plot_variance_energy(complex_matrix, fs, title_base, save_dir, save,
                         rolling_window=50):
    amp_active, _, _ = _get_active(complex_matrix)
    n_frames = amp_active.shape[0]
    t = np.arange(n_frames) / fs

    # Per-frame energy (sum of squared amplitudes)
    energy = np.sum(amp_active ** 2, axis=1)
    # Per-frame variance across subcarriers
    variance = np.var(amp_active, axis=1)

    # Rolling smoothing (avoids edge drops via nearest reflection)
    from scipy.ndimage import uniform_filter1d
    energy_smooth = uniform_filter1d(energy, size=rolling_window, mode="nearest")
    variance_smooth = uniform_filter1d(variance, size=rolling_window, mode="nearest")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                    gridspec_kw={"hspace": 0.12})

    # Energy
    ax1.plot(t, energy, alpha=0.3, color=STYLE["accent4"], linewidth=0.5)
    ax1.plot(t, energy_smooth, color=STYLE["accent4"], linewidth=2.0,
             label=f"Smoothed (window={rolling_window})")
    ax1.set_ylabel("Signal Energy  Σ|H|²", fontweight="bold")
    ax1.set_title(f"④ Signal Energy & Variance vs Time\n{title_base}",
                  fontweight="bold", pad=12)
    ax1.legend(loc="upper right", fontsize=9)

    # Variance
    ax2.plot(t, variance, alpha=0.3, color=STYLE["accent5"], linewidth=0.5)
    ax2.plot(t, variance_smooth, color=STYLE["accent5"], linewidth=2.0,
             label=f"Smoothed (window={rolling_window})")
    ax2.set_ylabel("Variance across SC", fontweight="bold")
    ax2.set_xlabel("Time (s)", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    if save:
        _save_fig(fig, save_dir, "04_variance_energy")
    return fig


# ════════════════════════════════════════════════════════════════════════
# 🔵 PLOT 5 — FFT / Spectrogram
# ════════════════════════════════════════════════════════════════════════

def plot_fft_spectrogram(complex_matrix, fs, title_base, save_dir, save):
    amp_active, _, _ = _get_active(complex_matrix)
    mean_amp = amp_active.mean(axis=1)

    # Remove DC component
    signal = mean_amp - mean_amp.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={"width_ratios": [1, 1.5]})

    # Left: FFT magnitude spectrum
    n = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Skip DC (index 0)
    ax1.plot(freqs[1:], fft_vals[1:], color=STYLE["accent1"], linewidth=1.2)
    ax1.fill_between(freqs[1:], fft_vals[1:], alpha=0.15, color=STYLE["accent1"])
    peak_idx = np.argmax(fft_vals[1:]) + 1
    peak_freq = freqs[peak_idx]
    ax1.axvline(peak_freq, color=STYLE["accent4"], linestyle="--", linewidth=1,
                label=f"Peak: {peak_freq:.1f} Hz")
    ax1.set_xlabel("Frequency (Hz)", fontweight="bold")
    ax1.set_ylabel("FFT Magnitude", fontweight="bold")
    ax1.set_title("FFT Spectrum", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, min(fs / 2, 25))

    # Right: Spectrogram
    nperseg = min(256, len(signal) // 4)
    if nperseg < 16:
        nperseg = 16
    noverlap = nperseg // 2

    f, t_spec, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg,
                                  noverlap=noverlap, scaling="spectrum")

    # Limit frequency range
    f_mask = f <= min(fs / 2, 25)
    im = ax2.pcolormesh(t_spec, f[f_mask], 10 * np.log10(Sxx[f_mask] + 1e-12),
                        shading="gouraud", cmap="inferno")
    ax2.set_xlabel("Time (s)", fontweight="bold")
    ax2.set_ylabel("Frequency (Hz)", fontweight="bold")
    ax2.set_title("Spectrogram (dB)", fontweight="bold")
    fig.colorbar(im, ax=ax2, label="Power (dB)", shrink=0.8)

    fig.suptitle(f"⑤ Frequency Domain Analysis\n{title_base}",
                 fontweight="bold")
    fig.tight_layout()
    if save:
        _save_fig(fig, save_dir, "05_fft_spectrogram")
    return fig


# ════════════════════════════════════════════════════════════════════════
# 🔵 PLOT 6 — Phase vs Time
# ════════════════════════════════════════════════════════════════════════

def plot_phase_vs_time(complex_matrix, fs, title_base, save_dir, save):
    amp = np.abs(complex_matrix)
    mask = np.any(amp > 0, axis=0)
    phase = np.angle(complex_matrix[:, mask])
    n_frames = phase.shape[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6),
                                    gridspec_kw={"hspace": 0.25})

    # Top: Phase heatmap (wrapped)
    extent = [0, n_frames / fs, 0, phase.shape[1]]
    im1 = ax1.imshow(phase.T, aspect="auto", cmap="hsv",
                     interpolation="nearest", origin="lower",
                     vmin=-np.pi, vmax=np.pi, extent=extent)
    ax1.set_xlabel("Time (s)", fontweight="bold")
    ax1.set_ylabel("Active Subcarrier", fontweight="bold")
    ax1.set_title(f"⑥ CSI Phase vs Time\n{title_base}", fontweight="bold", pad=12)
    fig.colorbar(im1, ax=ax1, label="Phase (rad)", shrink=0.8)

    # Bottom: Unwrapped phase for 3 representative subcarriers
    phase_unwrap = np.unwrap(phase, axis=0)
    n_sc = phase.shape[1]
    active_indices = np.flatnonzero(mask)
    t = np.arange(n_frames) / fs
    for i, sc_idx in enumerate([n_sc // 4, n_sc // 2, 3 * n_sc // 4]):
        if sc_idx < n_sc:
            ax2.plot(t, phase_unwrap[:, sc_idx], linewidth=1.0, alpha=0.8,
                     color=PALETTE[i % len(PALETTE)], label=f"SC {active_indices[sc_idx]}")

    ax2.set_xlabel("Time (s)", fontweight="bold")
    ax2.set_ylabel("Unwrapped Phase (rad)", fontweight="bold")
    ax2.set_title("Unwrapped Phase (selected subcarriers)", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    if save:
        _save_fig(fig, save_dir, "06_phase_vs_time")
    return fig


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="CSI Thesis Figures — 6 Publication-Ready Plots"
    )
    p.add_argument("file", nargs="?", default=None,
                   help="Dataset .txt or .csv (default: latest in datasets/)")
    p.add_argument("--save", action="store_true",
                   help="Save all figures as PNG (300 DPI)")
    p.add_argument("--fs", type=float, default=100.0,
                   help="Sampling frequency in Hz (default: 100)")
    p.add_argument("--out_dir", default=None,
                   help="Output directory for saved figures (default: next to dataset)")
    return p.parse_args()


def main():
    args = parse_args()
    _apply_style()

    # ── File resolution ───────────────────────────────────────────────
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            file_path = resolve_path(args.file)
    else:
        datasets_dir = resolve_path("datasets")
        file_path = get_latest_dataset(datasets_dir)

    if file_path is None or not file_path.exists():
        print("❌ No dataset file found.")
        print("   Use: python plot_thesis_figures.py <file.txt>")
        return 1

    # ── Load ──────────────────────────────────────────────────────────
    print(f"\n📂 Loading: {file_path.name}")
    try:
        complex_matrix, dropped, seq_stats = load_csi_matrix(file_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    n_frames, n_sub = complex_matrix.shape
    duration = n_frames / args.fs
    active_count = int(np.any(np.abs(complex_matrix) > 0, axis=0).sum())

    print(f"   {n_frames} frames × {n_sub} subcarriers ({active_count} active)")
    print(f"   Duration: {duration:.1f}s | Loss: {seq_stats.loss_percent:.2f}%")

    title_base = (f"{file_path.name}  ·  {n_frames} frames × {active_count} SC  ·  "
                  f"{duration:.1f}s  ·  loss {seq_stats.loss_percent:.1f}%")

    # ── Output directory ──────────────────────────────────────────────
    if args.out_dir:
        save_dir = Path(args.out_dir)
    else:
        save_dir = file_path.parent / f"{file_path.stem}_thesis_plots"
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Output: {save_dir}")

    # ── Generate all plots ────────────────────────────────────────────
    print(f"\n{'═' * 55}")
    print(f"  GENERATING 6 THESIS FIGURES")
    print(f"{'═' * 55}")

    print("\n🔴 CORE PLOTS")
    print("  ① Amplitude vs Time...")
    plot_amplitude_vs_time(complex_matrix, args.fs, title_base, save_dir, args.save)

    print("  ② Heatmap (Time × Subcarriers)...")
    plot_heatmap(complex_matrix, args.fs, title_base, save_dir, args.save)

    print("  ③ Subcarrier Profile...")
    plot_subcarrier_profile(complex_matrix, title_base, save_dir, args.save)

    print("\n🟡 SUPPORT PLOTS")
    print("  ④ Variance & Energy...")
    plot_variance_energy(complex_matrix, args.fs, title_base, save_dir, args.save)

    print("\n🔵 ADVANCED PLOTS")
    print("  ⑤ FFT & Spectrogram...")
    plot_fft_spectrogram(complex_matrix, args.fs, title_base, save_dir, args.save)

    print("  ⑥ Phase vs Time...")
    plot_phase_vs_time(complex_matrix, args.fs, title_base, save_dir, args.save)

    # ── Show ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 55}")
    if args.save:
        print(f"  ✅ All figures saved to: {save_dir}")
    print(f"  ✅ Showing 6 windows (close all to exit)")
    print(f"{'═' * 55}\n")

    try:
        plt.show()
    except Exception:
        pass

    plt.rcParams.update(plt.rcParamsDefault)
    return 0


if __name__ == "__main__":
    sys.exit(main())
