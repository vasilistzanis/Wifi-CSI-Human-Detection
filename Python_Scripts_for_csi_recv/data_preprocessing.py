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

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from csi_parser import SeqStats

DATA_COLUMNS = [
    'type', 'seq', 'mac', 'rssi', 'rate', 'noise_floor',
    'fft_gain', 'agc_gain', 'channel', 'local_timestamp',
    'sig_len', 'rx_state', 'len', 'first_word', 'data'
]


def configure_console_output() -> None:
    """Avoid UnicodeEncodeError on legacy Windows console encodings."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass


configure_console_output()


# ----------------------------------------------------------------------------
# INTERNAL HELPERS
# ----------------------------------------------------------------------------

def _safe_json(s: str):
    """Parse JSON string -> list, or None on failure."""
    try:
        result = json.loads(s)
        return result if isinstance(result, list) else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _build_complex_frame(raw: list, first_word_invalid: bool):
    """
    Convert raw [imag0, real0, imag1, real1, ...] to complex64 array.
    Applies HT40 first_word_invalid hardware bug fix.
    Returns None if malformed.
    """
    n = len(raw)
    if n < 2 or n % 2 != 0:
        return None

    # Apply HT40 hardware bug fix if flagged
    if first_word_invalid and n >= 4:
        raw = list(raw)  # explicit copy
        raw[0] = 0
        raw[1] = 0
        raw[2] = 0
        raw[3] = 0

    arr = np.array(raw, dtype=np.float32)
    real = arr[1::2]   # odd  indices -> real
    imag = arr[0::2]   # even indices -> imaginary
    return (real + 1j * imag).astype(np.complex64)


def _parse_recv_row(line: str) -> dict[str, str] | None:
    """
    Parse one CSI recv/logger line into the expected metadata columns.

    The CSI payload contains many commas, so we split only the first
    14 separators and keep the full payload as the final field.
    """
    line = line.strip()
    if not line or line.startswith("type,"):
        return None
    if not line.startswith("CSI_DATA"):
        return None

    parts = line.split(",", len(DATA_COLUMNS) - 1)
    if len(parts) != len(DATA_COLUMNS):
        return None

    row = dict(zip(DATA_COLUMNS, (part.strip() for part in parts)))
    row["data"] = row["data"].strip().strip('"')
    return row


# ----------------------------------------------------------------------------
# LOADERS
# ----------------------------------------------------------------------------

def load_csi_csv(filepath: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load CSV or TXT file from the recv (Magic Header format).

    Returns:
      complex_matrix : (N_frames, N_subcarriers) complex64
      metadata_df    : DataFrame with rssi, agc_gain, fft_gain, seq, etc.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load - names=DATA_COLUMNS + default header=0 skips the logger header row
    rows = []
    with filepath.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            row = _parse_recv_row(line)
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows, columns=DATA_COLUMNS)

    if df.empty:
        print("[WARNING] No CSI_DATA rows found.")
        return np.zeros((0, 0), dtype=np.complex64), df

    # Numeric conversion
    for col in ['seq', 'rssi', 'fft_gain', 'agc_gain', 'len', 'first_word']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle NaN in first_word to prevent conversion crash
    df['first_word'] = df['first_word'].fillna(0).astype(int)

    df = df.dropna(subset=['seq', 'len']).reset_index(drop=True)

    if df.empty:
        print("[WARNING] No valid rows after dropna.")
        return np.zeros((0, 0), dtype=np.complex64), df

    # Sequence diagnostics aligned with the shared parser logic
    seq_stats = SeqStats()
    for seq in df['seq'].astype(int).tolist():
        seq_stats.update(seq)

    if (seq_stats.missing_count > 0 or
            seq_stats.reset_count > 0 or
            seq_stats.duplicate_count > 0):
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

    # Vectorized JSON parsing for better performance
    parsed = df['data'].apply(_safe_json)
    valid_mask = parsed.notna()

    n_invalid = int((~valid_mask).sum())
    if n_invalid > 0:
        print(f"[WARNING] {n_invalid} rows with unparseable data - skipped")

    df = df[valid_mask].copy().reset_index(drop=True)
    parsed = parsed[valid_mask].reset_index(drop=True)

    if df.empty:
        print("[WARNING] No frames could be parsed.")
        return np.zeros((0, 0), dtype=np.complex64), df

    first_words = df['first_word'].tolist()
    frames = []
    valid_idx = []

    for i, (raw, fw) in enumerate(zip(parsed.tolist(), first_words)):
        frame = _build_complex_frame(raw, bool(fw))
        if frame is not None:
            frames.append(frame)
            valid_idx.append(i)

    if not frames:
        print("[WARNING] No frames converted to complex.")
        return np.zeros((0, 0), dtype=np.complex64), df

    # Consistency: keep only frames with the most common length
    lengths = [len(f) for f in frames]
    if len(set(lengths)) > 1:
        from collections import Counter
        most_common_len = Counter(lengths).most_common(1)[0][0]
        kept = [(f, i) for f, i, l in zip(frames, valid_idx, lengths)
                if l == most_common_len]
        frames, valid_idx = zip(*kept)
        frames, valid_idx = list(frames), list(valid_idx)
        print(f"[WARNING] Mixed frame lengths - kept {len(frames)} "
              f"frames with len={most_common_len}")

    complex_matrix = np.vstack(frames).astype(np.complex64)
    metadata_df = df.iloc[valid_idx].reset_index(drop=True)

    print(f"[OK] Loaded {complex_matrix.shape[0]} frames "
          f"x {complex_matrix.shape[1]} subcarriers")
    summary_parts = []
    rssi_values = metadata_df['rssi'].dropna()
    if not rssi_values.empty:
        summary_parts.append(f"RSSI: {rssi_values.mean():.1f} dBm")

    agc_mode = metadata_df['agc_gain'].dropna().mode()
    if not agc_mode.empty:
        summary_parts.append(f"AGC: {int(agc_mode.iloc[0])}")

    fft_mode = metadata_df['fft_gain'].dropna().mode()
    if not fft_mode.empty:
        summary_parts.append(f"FFT: {int(fft_mode.iloc[0])}")

    if summary_parts:
        print("   " + " | ".join(summary_parts))

    return complex_matrix, metadata_df


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
        filtered = data.copy()

        for sc in range(data.shape[1]):
            s = pd.Series(data[:, sc])
            rolling_median = s.rolling(window_size, center=True, min_periods=1).median()
            deviation = (s - rolling_median).abs()
            rolling_mad = deviation.rolling(window_size, center=True, min_periods=1).median()
            threshold = n_sigmas * 1.4826 * rolling_mad
            threshold = threshold.clip(lower=1e-6)  # Prevent zero-MAD from flagging everything
            outlier_mask = deviation > threshold
            filtered[outlier_mask.values, sc] = rolling_median[outlier_mask].values

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

    # -- transform (Inference) -----------------------------------------------
    def transform(self, complex_matrix: np.ndarray,
                  use_pca: bool = True,
                  cutoff: float | None = None) -> np.ndarray:
        eff_cutoff = cutoff if cutoff is not None else self.cutoff
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform() first.")

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
    processed = pipeline.fit_transform(
        complex_matrix,
        use_pca=True,
        n_components=10,
        scaler_type='standard'
    )
    print(f"\n[OK] Output: {processed.shape}")
    print(f"   Mean={processed.mean():.4f} | Std={processed.std():.4f}")
