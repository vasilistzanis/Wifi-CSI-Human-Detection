#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ESP32-C6 CSI Plotter (Thesis Grade) — Final (3 Windows Version)
Features:
- Auto-detect latest dataset
- Correct I/Q order for ESP32-C6
- Amplitude Heatmap (separate window)
- Mean Amplitude per Subcarrier (separate window)
- Phase Heatmap (separate window)
- Null subcarrier masking
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import glob

# 🔧 (Optional) Force separate GUI windows
matplotlib.use('TkAgg')  # ή 'Qt5Agg'

# ==============================================================================
# ΕΝΤΟΠΙΣΜΟΣ ΑΡΧΕΙΟΥ
# ==============================================================================

def get_latest_dataset():
    """Βρίσκει αυτόματα το πιο πρόσφατο αρχείο .txt στον φάκελο datasets."""
    list_of_files = glob.glob('datasets/*.txt')
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

# ==============================================================================
# PARSING
# ==============================================================================

def parse_csi_data(filepath):
    """
    Διαβάζει το αρχείο γραμμή-γραμμή και εξάγει:
    - amp_matrix  : (frames × subcarriers)
    - phase_matrix: (frames × subcarriers)
    """
    print(f"📂 Διαβάζω: {filepath}")
    
    amp_list   = []
    phase_list = []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.startswith("CSI_DATA"):
                continue

            try:
                start_idx = line.find('[')
                end_idx   = line.find(']')

                if start_idx == -1 or end_idx == -1:
                    continue

                raw_str  = line[start_idx + 1 : end_idx]
                num_list = [int(x) for x in raw_str.split(',')]

                frame_amp   = []
                frame_phase = []

                for i in range(0, len(num_list), 2):
                    # ✅ ESP32-C6: [Imaginary, Real]
                    imag = num_list[i]
                    real = num_list[i + 1]

                    amp   = np.sqrt(real**2 + imag**2)
                    phase = np.arctan2(imag, real)

                    frame_amp.append(amp)
                    frame_phase.append(phase)

                amp_list.append(frame_amp)
                phase_list.append(frame_phase)

            except Exception:
                continue

    return np.array(amp_list), np.array(phase_list)

# ==============================================================================
# PLOTTING (3 WINDOWS)
# ==============================================================================

def plot_all(amp_matrix, phase_matrix, filename):
    """Δημιουργεί 3 ξεχωριστά παράθυρα plots."""

    if amp_matrix.size == 0:
        print("Δεν βρέθηκαν δεδομένα!")
        return

    print(f"Frames × Subcarriers: {amp_matrix.shape}")

    # Mask null subcarriers
    non_null_mask = np.any(amp_matrix > 0, axis=0)
    amp_clean     = amp_matrix[:, non_null_mask]
    phase_clean   = phase_matrix[:, non_null_mask]

    title = (
        f"CSI Analysis — {os.path.basename(filename)}\n"
        f"{amp_matrix.shape[0]} frames @ 100Hz"
    )

    # ─────────────────────────────────────────
    # 1. Amplitude Heatmap
    # ─────────────────────────────────────────
    plt.figure(figsize=(12, 6))
    plt.title("Amplitude Heatmap\n" + title)

    im1 = plt.imshow(
        amp_clean.T,
        aspect='auto',
        cmap='jet',
        interpolation='nearest',
        origin='lower'
    )
    plt.xlabel("Time (Frames)")
    plt.ylabel("Subcarrier Index")
    plt.colorbar(im1, label='Amplitude')

    # ─────────────────────────────────────────
    # 2. Mean Amplitude
    # ─────────────────────────────────────────
    plt.figure(figsize=(12, 6))
    mean_amp = np.mean(amp_clean, axis=0)

    plt.title("Mean Amplitude Spectrum\n" + title)
    plt.plot(mean_amp, color='#f7b731', linewidth=1.5)
    plt.fill_between(range(len(mean_amp)), mean_amp, alpha=0.3, color='#f7b731')
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    # ─────────────────────────────────────────
    # 3. Phase Heatmap
    # ─────────────────────────────────────────
    plt.figure(figsize=(12, 6))
    plt.title("Phase Heatmap\n" + title)

    im3 = plt.imshow(
        phase_clean.T,
        aspect='auto',
        cmap='hsv',
        interpolation='nearest',
        origin='lower',
        vmin=-np.pi,
        vmax=np.pi
    )
    plt.xlabel("Time (Frames)")
    plt.ylabel("Subcarrier Index")
    plt.colorbar(im3, label='Phase (rad)')

    print("Άνοιγμα 3 παραθύρων...")
    plt.show()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    latest_file = get_latest_dataset()

    if not latest_file:
        print("❌ Δεν βρέθηκαν αρχεία στον φάκελο 'datasets'.")
        return

    amp_matrix, phase_matrix = parse_csi_data(latest_file)
    plot_all(amp_matrix, phase_matrix, latest_file)

if __name__ == "__main__":
    main()