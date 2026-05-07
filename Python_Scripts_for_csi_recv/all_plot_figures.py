#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSI Thesis Figures - Complete Publication-Ready Plot Set
========================================================
Generates 7 thesis-grade figures from a single CSI dataset file.
Integrates standard CSIPipeline filtering for robust visualization.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import config
import matplotlib
from scipy.signal import spectrogram, savgol_filter, stft
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from csi_parser import (
    configure_console_output,
    resolve_path,
    get_latest_dataset,
    load_csi_matrix,
)

from data_preprocessing import CSIPipeline

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

# ========================================================================
# STYLE
# ========================================================================

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
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=STYLE["bg"])
    print(f"  [SAVE] {out.name}")

def _move_window(fig):
    """Force the window to open at a specific screen position to avoid stacking offsets."""
    try:
        manager = fig.canvas.manager
        backend = matplotlib.get_backend()
        if backend == 'TkAgg':
            manager.window.wm_geometry("+50+50")
        elif backend in ['Qt5Agg', 'QtAgg']:
            manager.window.move(50, 50)
    except Exception:
        pass

# ========================================================================
# PREPROCESSING INTEGRATION
# ========================================================================

def get_preprocessed_amplitude(complex_matrix, fs):
    """
    Applies the standard CSIPipeline to extract cleaned amplitude.
    Steps applied: Null Removal -> Hampel Filter -> Butterworth Low-pass.
    Returns cleaned amplitude and the indices of active subcarriers.
    """
    pipeline = CSIPipeline(fs=fs, use_diff=False)
    # 1. Null removal
    amp_active = pipeline.remove_null_subcarriers(complex_matrix, fit=True)
    indices = np.flatnonzero(pipeline.active_mask)
    # 2. Hampel filter
    amp_clean = pipeline.apply_hampel_filter(amp_active)
    # 3. Butterworth low-pass
    amp_clean = pipeline.apply_lowpass_filter(amp_clean)
    return amp_clean, indices

# ========================================================================
# [CORE] PLOT 1 - Amplitude vs Time
# ========================================================================

def plot_amplitude_vs_time(complex_matrix, fs, title_base, save_dir, save):
    amp_clean, indices = get_preprocessed_amplitude(complex_matrix, fs)
    n_frames = amp_clean.shape[0]
    t = np.arange(n_frames) / fs

    mean_amp = amp_clean.mean(axis=1)
    std_amp = amp_clean.std(axis=1)

    fig, ax = plt.subplots(figsize=(14, 8))
    _move_window(fig)
    ax.plot(t, mean_amp, color=STYLE["accent1"], linewidth=1.2, label="Mean Amplitude (Filtered)")
    ax.fill_between(t, mean_amp - std_amp, mean_amp + std_amp,
                    alpha=0.2, color=STYLE["accent1"], label="+/-1 sigma")

    n_sc = amp_clean.shape[1]
    for i, sc_idx in enumerate([n_sc // 4, n_sc // 2, 3 * n_sc // 4]):
        if sc_idx < n_sc:
            ax.plot(t, amp_clean[:, sc_idx], linewidth=0.6, alpha=0.5,
                    color=PALETTE[(i + 1) % len(PALETTE)],
                    label=f"SC {indices[sc_idx]}")

    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_ylabel("CSI Amplitude |H|", fontweight="bold")
    ax.set_title(f"1. CSI Amplitude vs Time (Filtered)\n{title_base}", fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    fig.tight_layout()
    if save:
        _save_fig(fig, save_dir, "01_amplitude_vs_time")
    return fig

# ========================================================================
# [CORE] PLOT 2 - Heatmap (Time x Subcarriers)
# ========================================================================

def plot_heatmap(complex_matrix, fs, title_base, save_dir, save):
    amp_clean, indices = get_preprocessed_amplitude(complex_matrix, fs)
    n_frames = amp_clean.shape[0]

    vmin = np.percentile(amp_clean, 2)
    vmax = np.percentile(amp_clean, 98)

    fig, ax = plt.subplots(figsize=(14, 8))
    _move_window(fig)
    extent = [0, n_frames / fs, 0, amp_clean.shape[1]]
    im = ax.imshow(amp_clean.T, aspect="auto", cmap="viridis",
                   interpolation="nearest", origin="lower",
                   vmin=vmin, vmax=vmax, extent=extent)
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_ylabel("Active Subcarrier (sequential)", fontweight="bold")
    ax.set_title(f"2. CSI Amplitude Heatmap (Filtered)\n{title_base}",
                 fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, label="Amplitude |H|", shrink=0.8)
    fig.tight_layout()
    if save:
        _save_fig(fig, save_dir, "02_heatmap")
    return fig

# ========================================================================
# [CORE] PLOT 3 - Amplitude vs Subcarriers (Channel Fingerprint)
# ========================================================================

def plot_subcarrier_profile(complex_matrix, fs, title_base, save_dir, save):
    amp_clean, indices = get_preprocessed_amplitude(complex_matrix, fs)

    mean_amp = amp_clean.mean(axis=0)
    std_amp = amp_clean.std(axis=0)
    median_amp = np.median(amp_clean, axis=0)

    fig, ax = plt.subplots(figsize=(14, 8))
    _move_window(fig)
    ax.plot(indices, mean_amp, color=STYLE["accent2"], linewidth=2.0,
            label="Mean", zorder=3)
    ax.fill_between(indices, mean_amp - std_amp, mean_amp + std_amp,
                    alpha=0.2, color=STYLE["accent2"])
    ax.plot(indices, median_amp, color=STYLE["accent3"], linewidth=1.2,
            linestyle="--", alpha=0.8, label="Median")

    ax.set_xlabel("Subcarrier Index", fontweight="bold")
    ax.set_ylabel("Amplitude |H|", fontweight="bold")
    ax.set_title(f"3. Channel Frequency Response (Filtered)\n{title_base}",
                 fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.8)
    fig.tight_layout()
    if save:
        _save_fig(fig, save_dir, "03_subcarrier_profile")
    return fig

# ========================================================================
# [INFO] PLOT 4 - Variance / Signal Energy vs Time
# ========================================================================

def plot_variance_energy(complex_matrix, fs, title_base, save_dir, save,
                         rolling_window=50):
    amp_clean, _ = get_preprocessed_amplitude(complex_matrix, fs)
    n_frames = amp_clean.shape[0]
    t = np.arange(n_frames) / fs

    energy = np.sum(amp_clean ** 2, axis=1)
    variance = np.var(amp_clean, axis=1)

    from scipy.ndimage import uniform_filter1d
    energy_smooth = uniform_filter1d(energy, size=rolling_window, mode="nearest")
    variance_smooth = uniform_filter1d(variance, size=rolling_window, mode="nearest")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"hspace": 0.2})
    _move_window(fig)

    ax1.plot(t, energy, alpha=0.3, color=STYLE["accent4"], linewidth=0.5)
    ax1.plot(t, energy_smooth, color=STYLE["accent4"], linewidth=2.0,
             label=f"Smoothed (window={rolling_window})")
    ax1.set_ylabel("Signal Energy  sum|H|^2", fontweight="bold")
    ax1.set_title(f"4. Signal Energy & Variance vs Time (Filtered)\n{title_base}",
                  fontweight="bold", pad=12)
    ax1.legend(loc="upper right", fontsize=9)

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

# ========================================================================
# [ADVANCED] PLOT 5 - FFT / Spectrogram
# ========================================================================

def plot_fft_spectrogram(complex_matrix, fs, title_base, save_dir, save, f_max=25.0):
    amp_clean, _ = get_preprocessed_amplitude(complex_matrix, fs)
    mean_amp = amp_clean.mean(axis=1)

    signal = mean_amp - mean_amp.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8),
                                    gridspec_kw={"width_ratios": [1, 1.5]})
    _move_window(fig)

    if signal.size == 0:
        ax1.text(0.5, 0.5, "No frames available", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=11)
        ax2.text(0.5, 0.5, "No frames available", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=11)
        ax1.set_title("FFT Spectrum", fontweight="bold")
        ax2.set_title("Spectrogram (dB)", fontweight="bold")
        fig.suptitle(f"5. Frequency Domain Analysis (Filtered)\n{title_base}",
                     fontweight="bold")
        fig.tight_layout()
        if save:
            _save_fig(fig, save_dir, "05_fft_spectrogram")
        return fig

    n = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    if len(fft_vals) > 1:
        ax1.plot(freqs[1:], fft_vals[1:], color=STYLE["accent1"], linewidth=1.2)
        ax1.fill_between(freqs[1:], fft_vals[1:], alpha=0.15, color=STYLE["accent1"])
        peak_idx = np.argmax(fft_vals[1:]) + 1
        peak_freq = freqs[peak_idx]
        ax1.axvline(peak_freq, color=STYLE["accent4"], linestyle="--", linewidth=1,
                    label=f"Peak: {peak_freq:.1f} Hz")
    else:
        ax1.text(0.5, 0.5, "Need at least 2 frames\nfor FFT analysis",
                 ha="center", va="center", transform=ax1.transAxes, fontsize=11)
    ax1.set_xlabel("Frequency (Hz)", fontweight="bold")
    ax1.set_ylabel("FFT Magnitude", fontweight="bold")
    ax1.set_title("FFT Spectrum", fontweight="bold")
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(fontsize=9)
    ax1.set_xlim(0, min(fs / 2, f_max))

    if len(signal) < 2:
        ax2.text(0.5, 0.5, "Need at least 2 frames\nfor spectrogram",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11)
        ax2.set_xlabel("Time (s)", fontweight="bold")
        ax2.set_ylabel("Frequency (Hz)", fontweight="bold")
        ax2.set_title("Spectrogram (dB)", fontweight="bold")
    else:
        nperseg = min(256, len(signal))
        noverlap = max(0, min(nperseg // 2, nperseg - 1))
        f, t_spec, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg,
                                     noverlap=noverlap, scaling="spectrum")
        f_mask = f <= min(fs / 2, f_max)
        im = ax2.pcolormesh(t_spec, f[f_mask], 10 * np.log10(Sxx[f_mask] + 1e-12),
                            shading="gouraud", cmap="inferno")
        ax2.set_xlabel("Time (s)", fontweight="bold")
        ax2.set_ylabel("Frequency (Hz)", fontweight="bold")
        ax2.set_title("Spectrogram (dB)", fontweight="bold")
        fig.colorbar(im, ax=ax2, label="Power (dB)", shrink=0.8)

    fig.suptitle(f"5. Frequency Domain Analysis (Filtered)\n{title_base}",
                 fontweight="bold")
    fig.tight_layout()
    if save:
        _save_fig(fig, save_dir, "05_fft_spectrogram")
    return fig

# ========================================================================
# [ADVANCED] PLOT 6 - Phase vs Time
# ========================================================================

def plot_phase_vs_time(complex_matrix, fs, title_base, save_dir, save):
    # Phase analysis uses raw complex values before amplitude filtering.
    # Raw CSI phase contains CFO (Carrier Frequency Offset) and STO (Sampling
    # Time Offset) hardware artifacts that dominate the unwrapped phase.
    # A per-subcarrier linear detrend (polyfit degree-1) removes the bulk
    # CFO/STO drift to expose the residual motion-induced phase variation.
    amp = np.abs(complex_matrix)
    mask = np.any(amp > 0, axis=0)
    phase = np.angle(complex_matrix[:, mask])
    n_frames = phase.shape[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"hspace": 0.3})
    _move_window(fig)

    extent = [0, n_frames / fs, 0, phase.shape[1]]
    im1 = ax1.imshow(phase.T, aspect="auto", cmap="hsv",
                     interpolation="nearest", origin="lower",
                     vmin=-np.pi, vmax=np.pi, extent=extent)
    ax1.set_xlabel("Time (s)", fontweight="bold")
    ax1.set_ylabel("Active Subcarrier", fontweight="bold")
    ax1.set_title(
        f"6. CSI Raw Phase (CFO/STO hardware artifacts present)\n{title_base}",
        fontweight="bold", pad=12,
    )
    fig.colorbar(im1, ax=ax1, label="Phase (rad)", shrink=0.8)

    # Unwrap then remove linear trend per subcarrier (standard CFO/STO removal)
    phase_unwrap = np.unwrap(phase, axis=0)
    t = np.arange(n_frames, dtype=np.float64)
    phase_sanitised = phase_unwrap.copy()
    for i in range(phase_unwrap.shape[1]):
        slope, intercept = np.polyfit(t, phase_unwrap[:, i], 1)
        phase_sanitised[:, i] = phase_unwrap[:, i] - (slope * t + intercept)

    n_sc = phase.shape[1]
    active_indices = np.flatnonzero(mask)
    t_sec = np.arange(n_frames) / fs
    for i, sc_idx in enumerate([n_sc // 4, n_sc // 2, 3 * n_sc // 4]):
        if sc_idx < n_sc:
            ax2.plot(t_sec, phase_sanitised[:, sc_idx], linewidth=1.0, alpha=0.8,
                     color=PALETTE[i % len(PALETTE)],
                     label=f"SC {active_indices[sc_idx]}")

    ax2.set_xlabel("Time (s)", fontweight="bold")
    ax2.set_ylabel("Sanitised Phase Residual (rad)", fontweight="bold")
    ax2.set_title(
        "Linear-detrended Phase Residual (CFO/STO removed per subcarrier)",
        fontweight="bold",
    )
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    if save:
        _save_fig(fig, save_dir, "06_phase_vs_time")
    return fig

# ========================================================================
# [ADVANCED] PLOT 7 - Synchronized Motion Analysis (3-Panel)
# ========================================================================

def plot_motion_analysis(complex_matrix, fs, title_base, save_dir, save,
                         f_max=10.0, stft_window=64):
    amp_clean, indices = get_preprocessed_amplitude(complex_matrix, fs)
    n_frames, n_sc = amp_clean.shape
    duration = n_frames / fs

    # --- Preprocessing ---
    # Centering (similar to Temporal Diff logic, removes static offsets)
    amp_centered = amp_clean - amp_clean.mean(axis=0, keepdims=True)

    # Per-subcarrier z-score for heatmap (StandardScaler logic)
    sigma = amp_centered.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    amp_norm = amp_centered / sigma

    # --- Figure ---
    fig = plt.figure(figsize=(14, 8))
    _move_window(fig)
    gs = GridSpec(3, 1, height_ratios=[1.2, 0.8, 1.2], hspace=0.45,
                  left=0.08, right=0.92, top=0.88, bottom=0.08)

    fig.suptitle(f"7. WiFi CSI Motion Analysis (CSIPipeline Integrated)\n{title_base}",
                 fontsize=14, fontweight="bold", y=0.98)

    # ---- (a) Heatmap ----
    ax1 = fig.add_subplot(gs[0])
    extent = [0, duration, 0, n_sc]
    vabs = max(abs(np.percentile(amp_norm, 2)),
               abs(np.percentile(amp_norm, 98)))

    im1 = ax1.imshow(amp_norm.T, aspect="auto", cmap="RdBu_r",
                     interpolation="bilinear", origin="lower",
                     vmin=-vabs, vmax=vabs, extent=extent)
    ax1.set_ylabel("Subcarrier", fontweight="bold")
    ax1.set_title("(a) CSI Amplitude Heatmap (Filtered, per-subcarrier z-score)",
                  fontweight="bold", pad=6, fontsize=11)
    ax1.set_xlim(0, duration)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.025, pad=0.015)
    cb1.set_label("Norm. Amp", fontsize=9)

    # ---- (b) Motion Energy ----
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Note: CSIPipeline apply_temporal_diff uses np.diff(amp, n=1, axis=0)
    diff = np.diff(amp_clean, axis=0)
    energy = np.sum(np.abs(diff), axis=1)
    e_min, e_max = energy.min(), energy.max()
    if e_max - e_min > 1e-8:
        energy = (energy - e_min) / (e_max - e_min)

    # Savitzky-Golay smoothing
    win = min(21, len(energy))
    if win % 2 == 0:
        win -= 1
    energy_smooth = savgol_filter(energy, win, 3) if win >= 5 else energy.copy()

    t_energy = np.arange(len(energy)) / fs
    ax2.fill_between(t_energy, energy, alpha=0.2, color=STYLE["accent1"],
                     label="Raw energy")
    ax2.plot(t_energy, energy_smooth, color=STYLE["accent4"],
             linewidth=2.0, label="Smoothed (Savitzky-Golay)")

    if len(energy_smooth) > 10:
        threshold = np.mean(energy_smooth) + 1.5 * np.std(energy_smooth)
        ax2.axhline(threshold, color=STYLE["accent2"], linewidth=1.0,
                    linestyle="--", alpha=0.7, label="Threshold (μ+1.5σ)")

    ax2.set_ylabel("Motion Energy", fontweight="bold")
    ax2.set_title("(b) Motion Energy (CSIPipeline Temporal Diff, L1 aggregation)",
                  fontweight="bold", pad=6, fontsize=11)
    ax2.legend(loc="upper right", fontsize=9, ncol=3)
    ax2.set_xlim(0, duration)
    ax2.set_ylim(bottom=0)

    # ---- (c) Doppler Spectrogram ----
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    signal = amp_centered.mean(axis=1)
    signal = signal - signal.mean()

    nperseg = min(stft_window, len(signal))
    if nperseg >= 8:
        noverlap = nperseg * 3 // 4
        f_stft, t_stft, Zxx = stft(signal, fs=fs, nperseg=nperseg,
                                    noverlap=noverlap, window='hann')
        mag_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
        f_mask = f_stft <= f_max

        vmin_db = np.percentile(mag_db[f_mask], 5)
        vmax_db = np.percentile(mag_db[f_mask], 99)

        im3 = ax3.pcolormesh(t_stft, f_stft[f_mask], mag_db[f_mask],
                             shading="gouraud", cmap="inferno",
                             vmin=vmin_db, vmax=vmax_db)
        cb3 = fig.colorbar(im3, ax=ax3, fraction=0.025, pad=0.015)
        cb3.set_label("Magnitude (dB)", fontsize=10)
    else:
        ax3.text(0.5, 0.5, "Insufficient data for STFT",
                 ha="center", va="center", transform=ax3.transAxes)

    ax3.set_xlabel("Time (s)", fontweight="bold")
    ax3.set_ylabel("Frequency (Hz)", fontweight="bold")
    ax3.set_title("(c) Doppler Spectrogram (STFT of mean Filtered CSI signal)",
                  fontweight="bold", pad=6, fontsize=11)
    ax3.set_xlim(0, duration)
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    if save:
        _save_fig(fig, save_dir, "07_motion_analysis")
    return fig

# ========================================================================
# COMPARE MODE
# ========================================================================

def run_compare(class_names, fs, save_dir, save, fft_max, rolling_window):
    """Generate all 7 plots for each class for side-by-side comparison."""
    datasets_root = resolve_path(config.DATASETS_DIR)

    for class_name in class_names:
        class_path = datasets_root / class_name
        if not class_path.exists():
            print(f"[WARNING] Directory not found: {class_path}")
            continue

        latest = get_latest_dataset(class_path)
        if latest is None:
            print(f"[WARNING] No data files in {class_path}")
            continue

        print(f"\n{'=' * 55}")
        print(f"  CLASS: {class_name.upper()} — {latest.name}")
        print(f"{'=' * 55}")

        complex_matrix, dropped, seq_stats = load_csi_matrix(latest)
        n_frames, n_sub = complex_matrix.shape
        duration = n_frames / fs
        active_count = int(np.any(np.abs(complex_matrix) > 0, axis=0).sum())

        title = (f"{class_name}/{latest.name}  -  {n_frames} frames x "
                 f"{active_count} SC  -  {duration:.1f}s  -  "
                 f"loss {seq_stats.loss_percent:.1f}%")

        class_dir = save_dir / class_name if save else save_dir
        if save:
            class_dir.mkdir(parents=True, exist_ok=True)

        plot_amplitude_vs_time(complex_matrix, fs, title, class_dir, save)
        plot_heatmap(complex_matrix, fs, title, class_dir, save)
        plot_subcarrier_profile(complex_matrix, fs, title, class_dir, save)
        plot_variance_energy(complex_matrix, fs, title, class_dir, save,
                             rolling_window=rolling_window)
        plot_fft_spectrogram(complex_matrix, fs, title, class_dir, save,
                             f_max=fft_max)
        plot_phase_vs_time(complex_matrix, fs, title, class_dir, save)
        plot_motion_analysis(complex_matrix, fs, title, class_dir, save)

# ========================================================================
# MAIN
# ========================================================================

def parse_args():
    defaults = config.get_script_defaults("all_plot_figures")
    p = argparse.ArgumentParser(
        description="CSI Thesis Figures - 7 Publication-Ready Plots"
    )
    p.add_argument("file", nargs="?", default=defaults["file"],
                   help="Dataset .txt or .csv (default: latest in datasets/)")
    config.add_bool_argument(
        p,
        dest="save",
        default=defaults["save"],
        help="Save all figures as PNG (300 DPI)",
        positive_flags=["--save"],
        negative_flags=["--no-save"],
    )
    p.add_argument("--compare", nargs="+", metavar="CLASS",
                   default=defaults["compare"],
                   help="Compare multiple classes (e.g. --compare walk idle)")
    p.add_argument("--fs", type=float, default=defaults["fs"],
                   help="Sampling frequency in Hz (default: 100)")
    p.add_argument("--fft-max", type=float, default=defaults["fft_max"],
                   help="Max frequency (Hz) to show in FFT/Spectrogram (default: 25)")
    p.add_argument("--rolling-window", type=int, default=defaults["rolling_window"],
                   help="Window for smoothing variance/energy plots")
    p.add_argument("--out_dir", default=defaults["out_dir"],
                   help="Output directory for saved figures (default: next to dataset)")
    return p.parse_args()

def main():
    args = parse_args()
    _apply_style()

    # -- Compare Mode --------------------------------------------------
    if args.compare:
        save_dir = Path(args.out_dir) if args.out_dir else resolve_path(config.DATASETS_DIR) / "compare_plots"
        run_compare(args.compare, args.fs, save_dir, args.save, args.fft_max, args.rolling_window)
        
        print(f"\n{'=' * 55}")
        print(f"  [OK] Showing all windows (close all to exit)")
        print(f"{'=' * 55}\n")
        try:
            plt.show()
        except Exception:
            pass
        plt.rcParams.update(plt.rcParamsDefault)
        return 0

    # -- File resolution -----------------------------------------------
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            file_path = resolve_path(args.file)
    else:
        datasets_dir = resolve_path(config.DATASETS_DIR)
        file_path = get_latest_dataset(datasets_dir)

    if file_path is None or not file_path.exists():
        print("[ERROR] No dataset file found.")
        print("   Use: python plot_thesis_figures.py <file.txt>")
        return 1

    # -- Load ----------------------------------------------------------
    print(f"\n[FILE] Loading: {file_path.name}")
    try:
        complex_matrix, dropped, seq_stats = load_csi_matrix(file_path)
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return 1

    n_frames, n_sub = complex_matrix.shape
    duration = n_frames / args.fs
    active_count = int(np.any(np.abs(complex_matrix) > 0, axis=0).sum())

    print(f"   {n_frames} frames x {n_sub} subcarriers ({active_count} active)")
    print(f"   Duration: {duration:.1f}s | Loss: {seq_stats.loss_percent:.2f}%")

    title_base = (f"{file_path.name}  -  {n_frames} frames x {active_count} SC  -  "
                  f"{duration:.1f}s  -  loss {seq_stats.loss_percent:.1f}%")

    # -- Output directory ----------------------------------------------
    if args.out_dir:
        save_dir = Path(args.out_dir)
    else:
        save_dir = file_path.parent / f"{file_path.stem}_thesis_plots"
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Output: {save_dir}")

    # -- Generate all plots --------------------------------------------
    print(f"\n{'=' * 55}")
    print(f"  GENERATING 7 THESIS FIGURES")
    print(f"{'=' * 55}")

    print("\n[CORE] CORE PLOTS")
    print("  1. Amplitude vs Time...")
    plot_amplitude_vs_time(complex_matrix, args.fs, title_base, save_dir, args.save)

    print("  2. Heatmap (Time x Subcarriers)...")
    plot_heatmap(complex_matrix, args.fs, title_base, save_dir, args.save)

    print("  3. Subcarrier Profile...")
    plot_subcarrier_profile(complex_matrix, args.fs, title_base, save_dir, args.save)

    print("\n[INFO] SUPPORT PLOTS")
    print("  4. Variance & Energy...")
    plot_variance_energy(complex_matrix, args.fs, title_base, save_dir, args.save,
                         rolling_window=args.rolling_window)

    print("\n[ADVANCED] ADVANCED PLOTS")
    print("  5. FFT & Spectrogram...")
    plot_fft_spectrogram(complex_matrix, args.fs, title_base, save_dir, args.save,
                         f_max=args.fft_max)

    print("  6. Phase vs Time...")
    plot_phase_vs_time(complex_matrix, args.fs, title_base, save_dir, args.save)
    
    print("  7. Synchronized Motion Analysis...")
    plot_motion_analysis(complex_matrix, args.fs, title_base, save_dir, args.save)

    # -- Show ----------------------------------------------------------
    print(f"\n{'=' * 55}")
    if args.save:
        print(f"  [OK] All figures saved to: {save_dir}")
    print(f"  [OK] Showing 7 windows (close all to exit)")
    print(f"{'=' * 55}\n")

    try:
        plt.show()
    except Exception:
        pass

    plt.rcParams.update(plt.rcParamsDefault)
    return 0

if __name__ == "__main__":
    sys.exit(main())
