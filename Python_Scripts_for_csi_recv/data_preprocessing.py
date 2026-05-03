#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ESP32-C6 CSI Data Preprocessing Pipeline (Thesis Grade - Improved)
Applies: CSV/TXT Loading, Null Removal, Hampel Filtering,
Butterworth Low-pass, PCA, and Normalization.

Improvements:
  - Shape validation in transform()
  - Better PCA component warnings
  - Removed unused imports
  - More informative error messages

Compatible with Magic Header recv format:
  type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,
  channel,local_timestamp,sig_len,rx_state,len,first_word,data

IQ convention (ESP32 CSI buf layout):
  buf[0]=imag, buf[1]=real, buf[2]=imag, buf[3]=real, ...
  CSV data: [imag0, real0, imag1, real1, ...]
  complex(i) = real[i] + j*imag[i]  =  complex(raw[2i+1], raw[2i])
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from csi_parser import SeqStats


from csi_parser import configure_console_output
configure_console_output()


# ----------------------------------------------------------------------------
# LOADERS
# ----------------------------------------------------------------------------

def load_csi_csv(filepath: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load CSV or TXT file from the recv (Magic Header format).

    Uses csi_parser.parse_csi_line for frame parsing so training and live
    inference share identical IQ-decoding logic.

    Returns:
      complex_matrix : (N_frames, N_subcarriers) complex64
      metadata_df    : DataFrame with rssi, agc_gain, fft_gain, seq, etc.
    """
    from csi_parser import (
        parse_csi_line as _parse_csi_line,
        split_recv_fields as _split_recv_fields,
    )

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    rows: list[dict] = []
    frames: list[np.ndarray] = []
    seq_stats = SeqStats()

    with filepath.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            parts = _split_recv_fields(line)
            if parts is None:
                continue

            # Pass pre-split parts so parse_csi_line skips a second field-split.
            frame = _parse_csi_line(line, _parts=parts)
            if frame is None:
                continue

            try:
                seq_stats.update(int(parts[1]))
            except (ValueError, IndexError):
                pass

            rows.append({
                "seq":        parts[1],
                "rssi":       parts[3],
                "fft_gain":   parts[6],
                "agc_gain":   parts[7],
                "len":        parts[12],
                "first_word": parts[13],
            })
            frames.append(frame)

    if not frames:
        print("[WARNING] No valid CSI frames found.")
        return np.zeros((0, 0), dtype=np.complex64), pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in ["seq", "rssi", "fft_gain", "agc_gain", "len", "first_word"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if (seq_stats.missing_count > 0
            or seq_stats.reset_count > 0
            or seq_stats.duplicate_count > 0):
        summary_parts = []
        if seq_stats.missing_count > 0:
            summary_parts.append(
                f"{seq_stats.missing_count} packets lost in {seq_stats.gap_events} events"
            )
        if seq_stats.reset_count > 0:
            summary_parts.append(f"{seq_stats.reset_count} non-monotonic transitions")
        if seq_stats.duplicate_count > 0:
            summary_parts.append(f"{seq_stats.duplicate_count} duplicate packets")
        print(
            f"[WARNING] Sequence anomalies: {' | '.join(summary_parts)} "
            f"out of {seq_stats.received_count} received"
        )

    lengths = [f.shape[0] for f in frames]
    if len(set(lengths)) > 1:
        from collections import Counter
        most_common_len = Counter(lengths).most_common(1)[0][0]
        pairs = [
            (f, r) for f, r, ln in zip(frames, rows, lengths)
            if ln == most_common_len
        ]
        frames_kept, rows_kept = zip(*pairs)
        frames = list(frames_kept)
        df = pd.DataFrame(rows_kept)
        for col in ["seq", "rssi", "fft_gain", "agc_gain", "len", "first_word"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"[WARNING] Mixed frame lengths — kept {len(frames)} frames "
              f"with len={most_common_len}")

    complex_matrix = np.vstack(frames).astype(np.complex64)
    df = df.reset_index(drop=True)

    print(f"[OK] Loaded {complex_matrix.shape[0]} frames "
          f"x {complex_matrix.shape[1]} subcarriers")
    summary_parts = []
    rssi_values = df["rssi"].dropna()
    if not rssi_values.empty:
        summary_parts.append(f"RSSI: {rssi_values.mean():.1f} dBm")
    agc_mode = df["agc_gain"].dropna().mode()
    if not agc_mode.empty:
        summary_parts.append(f"AGC: {int(agc_mode.iloc[0])}")
    fft_mode = df["fft_gain"].dropna().mode()
    if not fft_mode.empty:
        summary_parts.append(f"FFT: {int(fft_mode.iloc[0])}")
    if summary_parts:
        print("   " + " | ".join(summary_parts))

    return complex_matrix, df


# ----------------------------------------------------------------------------
# PREPROCESSING PIPELINE
# ----------------------------------------------------------------------------

class CSIPipeline:
    """
    Full preprocessing pipeline for CSI amplitude data.
    Designed for Environment-Independent HAR (Human Activity Recognition).

    Steps:
      1. Null Subcarrier Removal    (guard/null band removal, saves mask)
      2. Hampel Filter              (outlier / spike removal)
      3. Butterworth Low-Pass       (noise smoothing, zero phase, 10 Hz)
      4. Temporal Difference        (first-order diff, motion focus)
      5. PCA                        (dimensionality reduction)
      6. StandardScaler             (environment-scale normalization)
    """

    def __init__(
        self,
        fs: float = 100.0,
        use_diff: bool = True,
        cutoff: float = 10.0,
    ):
        self.fs = fs
        self.use_diff = use_diff
        self.cutoff = cutoff

        # State (saved after fit_transform for reuse in transform)
        self.active_mask = None
        self.pca = None
        self.scaler = None
        self.is_fitted = False

        # Store training shape for validation
        self._fitted_n_subcarriers = None

    # -- 1. Null Subcarrier Removal ------------------------------------------
    def remove_null_subcarriers(self, complex_matrix: np.ndarray,
                                fit: bool = False) -> np.ndarray:
        amp = np.abs(complex_matrix)

        if fit:
            self.active_mask = np.any(amp > 1e-3, axis=0)
            n_active = int(self.active_mask.sum())
            if n_active == 0:
                raise ValueError("All subcarriers are zero - check your data!")

        if self.active_mask is None:
            raise RuntimeError("remove_null_subcarriers: fit=True was never called")

        return amp[:, self.active_mask]

    # -- 2. Hampel Filter (Vectorized) ---------------------------------------
    def apply_hampel_filter(self, data: np.ndarray, window_size: int = 11,
                            n_sigmas: float = 3.0) -> np.ndarray:
        from numpy.lib.stride_tricks import sliding_window_view

        half = window_size // 2
        # Reflect-pad so the sliding window is centred at every frame
        padded = np.pad(data, ((half, half), (0, 0)), mode="reflect")
        # windows: (n_frames, n_subcarriers, window_size)  — zero-copy view
        windows = sliding_window_view(padded, window_shape=window_size, axis=0)

        rolling_median = np.median(windows, axis=2)                      # (N, SC)
        # deviations creates a full (N, SC, W) copy; ~N*SC*W*4 bytes peak.
        # For window_size=11 and a 5 000-frame recording this is ~25 MB — acceptable.
        deviations = np.abs(windows - rolling_median[:, :, np.newaxis])  # (N, SC, W)
        rolling_mad = np.median(deviations, axis=2)                      # (N, SC)

        threshold = np.clip(n_sigmas * 1.4826 * rolling_mad, 1e-6, None)
        outlier_mask = np.abs(data - rolling_median) > threshold

        filtered = data.copy()
        filtered[outlier_mask] = rolling_median[outlier_mask]
        return filtered

    # -- 3. Butterworth Low-Pass (Vectorized) --------------------------------
    def apply_lowpass_filter(self, data: np.ndarray,
                             cutoff: float = 10.0) -> np.ndarray:
        from scipy.signal import sosfiltfilt

        nyquist = self.fs / 2.0
        if cutoff >= nyquist:
            print(f"[WARNING] Lowpass cutoff {cutoff} Hz >= Nyquist {nyquist:.1f} Hz "
                  f"- skipping filter")
            return data

        sos = butter(4, cutoff / nyquist, btype='low', output='sos')
        padlen = 3 * (2 * sos.shape[0] + 1)
        if data.shape[0] <= padlen:
            print(f"[WARNING] Lowpass skipped: only {data.shape[0]} frames available "
                  f"(need > {padlen})")
            return data
        return sosfiltfilt(sos, data, axis=0).astype(data.dtype)

    # -- 4. Temporal Difference ----------------------------------------------
    def apply_temporal_diff(self, data: np.ndarray) -> np.ndarray:
        if not self.use_diff:
            return data

        if data.shape[0] < 2:
            print("[WARNING] Temporal diff skipped: need at least 2 frames")
            return data

        return np.diff(data, n=1, axis=0).astype(np.float32)

    # -- fit_transform (Training) --------------------------------------------
    def fit_transform(self, complex_matrix: np.ndarray,
                      use_pca: bool = True,
                      n_components: int = 10,
                      scaler_type: str = 'standard',
                      cutoff: float | None = None) -> np.ndarray:
        if cutoff is not None:
            self.cutoff = cutoff
        print(f"[SETUP] fit_transform - input {complex_matrix.shape}")

        self._fitted_n_subcarriers = complex_matrix.shape[1]

        # [1] Null removal
        data = self.remove_null_subcarriers(complex_matrix, fit=True)
        print(f"   [1] Null removal: {complex_matrix.shape[1]} -> {data.shape[1]} subcarriers")

        # [2] Hampel
        data = self.apply_hampel_filter(data)
        print(f"   [2] Hampel [OK]")

        # [3] Butterworth
        data = self.apply_lowpass_filter(data, cutoff=self.cutoff)
        print(f"   [3] Butterworth [OK]")

        # [4] Temporal difference
        if self.use_diff:
            data = self.apply_temporal_diff(data)
            print(f"   [4] Temporal diff [OK] shape={data.shape}")
        else:
            print(f"   [4] Temporal diff: disabled")

        # [5] PCA
        if use_pca:
            if data.shape[0] < 2:
                raise ValueError("PCA requires at least 2 frames after preprocessing.")
            actual_n = min(n_components, data.shape[0] - 1, data.shape[1])
            if actual_n < n_components:
                print(f"   [WARNING] PCA: limited to {actual_n} by data shape {data.shape}")
            
            self.pca = PCA(n_components=actual_n)
            data = self.pca.fit_transform(data)
            explained = self.pca.explained_variance_ratio_.sum() * 100
            print(f"   [5] PCA: {actual_n} components, {explained:.1f}% variance [OK]")
        else:
            self.pca = None
            print(f"   [5] PCA: skipped")

        # [6] Scaler
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler_type '{scaler_type}'")

        data = self.scaler.fit_transform(data)
        print(f"   [6] {scaler_type} scaling -> output {data.shape} [OK]")

        self.is_fitted = True
        return data

    # -- fit_from_recordings (Training — per-recording DSP) ------------------
    def fit_from_recordings(
        self,
        recordings: list,
        use_pca: bool = True,
        n_components: int = 10,
        scaler_type: str = "standard",
        cutoff: float | None = None,
    ) -> "CSIPipeline":
        """
        Fit the pipeline on a list of complex CSI matrices (one per recording).

        Each recording passes independently through Hampel → Butterworth →
        temporal-diff so that filter transients and temporal-diff boundary
        artifacts never cross recording boundaries.  PCA and scaler are then
        fitted on the concatenated per-recording outputs.

        This replaces the previous fit_transform(np.vstack(recordings)) pattern.
        """
        if cutoff is not None:
            self.cutoff = cutoff
        if not recordings:
            raise ValueError("recordings list is empty")

        print(f"[SETUP] fit_from_recordings — {len(recordings)} recordings")

        # Step 1: null mask from all training frames combined (more robust than
        # single recording; no DSP applied yet so boundaries are irrelevant here)
        combined_raw = np.vstack(recordings)
        self._fitted_n_subcarriers = combined_raw.shape[1]
        self.remove_null_subcarriers(combined_raw, fit=True)
        n_active = int(self.active_mask.sum())
        print(f"   [1] Null mask: {combined_raw.shape[1]} → {n_active} active subcarriers")
        del combined_raw

        # Steps 2–4: per-recording DSP (no cross-boundary artifacts)
        pre_pca_blocks: list[np.ndarray] = []
        skipped = 0
        for cm in recordings:
            amp = self.remove_null_subcarriers(cm, fit=False)
            amp = self.apply_hampel_filter(amp)
            amp = self.apply_lowpass_filter(amp, cutoff=self.cutoff)
            if self.use_diff:
                amp = self.apply_temporal_diff(amp)
            if amp.shape[0] < 2:
                skipped += 1
                continue
            pre_pca_blocks.append(amp)

        if skipped:
            print(f"   [WARNING] {skipped}/{len(recordings)} recordings too short — skipped")
        if not pre_pca_blocks:
            raise ValueError("No recordings produced valid frames after per-recording DSP.")

        combined = np.vstack(pre_pca_blocks)
        print(f"   [2–4] Per-recording DSP: {len(pre_pca_blocks)} recordings "
              f"→ {combined.shape[0]} frames x {combined.shape[1]} features")

        # Step 5: PCA
        if use_pca:
            if combined.shape[0] < 2:
                raise ValueError("PCA requires at least 2 frames after preprocessing.")
            actual_n = min(n_components, combined.shape[0] - 1, combined.shape[1])
            if actual_n < n_components:
                print(f"   [WARNING] PCA: limited to {actual_n} by data shape {combined.shape}")
            self.pca = PCA(n_components=actual_n)
            self.pca.fit(combined)
            explained = self.pca.explained_variance_ratio_.sum() * 100
            print(f"   [5] PCA: {actual_n} components, {explained:.1f}% variance [OK]")
        else:
            self.pca = None
            print(f"   [5] PCA: skipped")

        # Step 6: scaler
        pca_out = self.pca.transform(combined) if (use_pca and self.pca is not None) else combined
        if scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler_type '{scaler_type}'")
        self.scaler.fit(pca_out)
        print(f"   [6] {scaler_type} scaler fitted on {pca_out.shape[0]} frames [OK]")

        self.is_fitted = True
        return self

    # -- transform (Inference) -----------------------------------------------
    def transform(self, complex_matrix: np.ndarray,
                  use_pca: bool = True,
                  cutoff: float | None = None) -> np.ndarray:
        eff_cutoff = cutoff if cutoff is not None else self.cutoff
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform() or fit_from_recordings() first.")

        if complex_matrix.shape[1] != self._fitted_n_subcarriers:
            raise ValueError(
                f"Shape mismatch: input has {complex_matrix.shape[1]}, "
                f"expected {self._fitted_n_subcarriers}. Check hardware config."
            )

        data = self.remove_null_subcarriers(complex_matrix, fit=False)
        data = self.apply_hampel_filter(data)
        data = self.apply_lowpass_filter(data, cutoff=eff_cutoff)

        if self.use_diff:
            data = self.apply_temporal_diff(data)

        if use_pca and self.pca is not None:
            data = self.pca.transform(data)

        data = self.scaler.transform(data)
        return data

    @property
    def n_active_subcarriers(self) -> int:
        return int(self.active_mask.sum()) if self.active_mask is not None else 0


# ----------------------------------------------------------------------------
# STANDALONE TEST
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
        print(f"\n[FILE] Loading: {csv_path}")
        complex_matrix, meta = load_csi_csv(csv_path)
    else:
        print("[INFO] Simulation mode")
        np.random.seed(42)
        r  = np.random.randn(500, 128).astype(np.float32) * 10
        im = np.random.randn(500, 128).astype(np.float32) * 10
        complex_matrix = (r + 1j * im).astype(np.complex64)
        complex_matrix[:, :6]  = 0
        complex_matrix[:, -6:] = 0
        complex_matrix[:50] += 5.0

    if complex_matrix.size == 0:
        print("[ERROR] Empty matrix - exiting")
        sys.exit(1)

    print(f"\n[STATS] Input: {complex_matrix.shape}")

    pipeline = CSIPipeline(fs=100.0, use_diff=True)
    pipeline.fit_from_recordings(
        [complex_matrix],
        use_pca=True,
        n_components=10,
        scaler_type='standard',
    )
    processed = pipeline.transform(complex_matrix, use_pca=True)
    print(f"\n[OK] Output: {processed.shape}")
    print(f"   Mean={processed.mean():.4f} | Std={processed.std():.4f}")
