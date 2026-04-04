#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ESP32-C6 CSI Data Preprocessing Pipeline (Thesis Grade)
Applies: CSV/TXT Loading, Null Removal, Hampel Filtering,
Butterworth Low-pass, PCA, and Normalization.

Compatible with Magic Header recv format:
  type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,
  channel,local_timestamp,sig_len,rx_state,len,first_word,data

IQ convention (ESP32 CSI buf layout):
  buf[0]=imag, buf[1]=real, buf[2]=imag, buf[3]=real, ...
  CSV data: [imag0, real0, imag1, real1, ...]
  complex(i) = real[i] + j*imag[i]  =  complex(raw[2i+1], raw[2i])
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATA_COLUMNS = [
    'type', 'seq', 'mac', 'rssi', 'rate', 'noise_floor',
    'fft_gain', 'agc_gain', 'channel', 'local_timestamp',
    'sig_len', 'rx_state', 'len', 'first_word', 'data'
]


# ════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ════════════════════════════════════════════════════════════════════════

def _safe_json(s: str):
    """Parse JSON string → list, or None on failure."""
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

    if first_word_invalid and n >= 4:
        raw = list(raw)  # copy — do NOT mutate original
        raw[0] = 0
        raw[1] = 0
        raw[2] = 0
        raw[3] = 0

    arr = np.array(raw, dtype=np.float32)
    real = arr[1::2]   # odd  indices → real
    imag = arr[0::2]   # even indices → imaginary
    return (real + 1j * imag).astype(np.complex64)


# ════════════════════════════════════════════════════════════════════════
# LOADERS
# ════════════════════════════════════════════════════════════════════════

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

    # Load — names=DATA_COLUMNS + default header=0 skips the logger header row
    try:
        df = pd.read_csv(filepath, names=DATA_COLUMNS,
                         on_bad_lines='skip', dtype=str)
    except TypeError:
        df = pd.read_csv(filepath, names=DATA_COLUMNS,
                         error_bad_lines=False, dtype=str)

    df = df[df['type'] == 'CSI_DATA'].copy().reset_index(drop=True)

    if df.empty:
        print("⚠️  No CSI_DATA rows found.")
        return np.zeros((0, 0), dtype=np.complex64), df

    # Numeric conversion
    for col in ['seq', 'rssi', 'fft_gain', 'agc_gain', 'len', 'first_word']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ✅ FIX: first_word NaN → 0  (prevents int(NaN) ValueError crash)
    df['first_word'] = df['first_word'].fillna(0).astype(int)

    df = df.dropna(subset=['seq', 'len']).reset_index(drop=True)

    if df.empty:
        print("⚠️  No valid rows after dropna.")
        return np.zeros((0, 0), dtype=np.complex64), df

    # Sequence gap detection
    seqs = df['seq'].astype(int).values
    gaps = np.diff(seqs) - 1
    total_gaps = int(gaps[gaps > 0].sum())
    if total_gaps > 0:
        gap_events = int((gaps > 0).sum())
        print(f"⚠️  Sequence gaps: {total_gaps} packets lost "
              f"in {gap_events} events out of {len(seqs)} received")

    # ✅ FAST: vectorized JSON parsing instead of iterrows
    parsed = df['data'].apply(_safe_json)
    valid_mask = parsed.notna()

    n_invalid = int((~valid_mask).sum())
    if n_invalid > 0:
        print(f"⚠️  {n_invalid} rows with unparseable data — skipped")

    df = df[valid_mask].copy().reset_index(drop=True)
    parsed = parsed[valid_mask].reset_index(drop=True)

    if df.empty:
        print("⚠️  No frames could be parsed.")
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
        print("⚠️  No frames converted to complex.")
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
        print(f"⚠️  Mixed frame lengths — kept {len(frames)} "
              f"frames with len={most_common_len}")

    complex_matrix = np.vstack(frames).astype(np.complex64)
    metadata_df = df.iloc[valid_idx].reset_index(drop=True)

    print(f"✅ Loaded {complex_matrix.shape[0]} frames "
          f"× {complex_matrix.shape[1]} subcarriers")
    if not metadata_df['rssi'].isna().all():
        print(f"   RSSI: {metadata_df['rssi'].mean():.1f} dBm | "
              f"AGC: {int(metadata_df['agc_gain'].mode().iloc[0])} | "
              f"FFT: {int(metadata_df['fft_gain'].mode().iloc[0])}")

    return complex_matrix, metadata_df


def get_latest_csv(folder: str | Path) -> Path | None:
    """Return the most recently modified CSV or TXT file in folder."""
    folder = Path(folder)
    files = list(folder.glob("*.csv")) + list(folder.glob("*.txt"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


# ════════════════════════════════════════════════════════════════════════
# PREPROCESSING PIPELINE
# ════════════════════════════════════════════════════════════════════════

class CSIPipeline:
    """
    Full preprocessing pipeline for CSI amplitude data.
    Designed for Environment-Independent HAR (Human Activity Recognition).

    Steps:
      1. Null Subcarrier Removal    (guard/null band removal, saves mask)
      2. Hampel Filter              (outlier / spike removal)
      3. Butterworth Low-Pass       (noise smoothing, zero phase, 12 Hz)
      4. Background Subtraction     (static environment removal — NEW)
      5. Temporal Difference        (first-order diff, motion focus — NEW)
      6. PCA                        (dimensionality reduction)
      7. StandardScaler             (environment-scale normalization)

    Why steps 4 & 5 make HAR environment-independent:
      - Background subtraction zeros out static reflections (walls, furniture).
        Only dynamic changes (human motion) remain.
      - Temporal diff (np.diff) makes the model see RATE OF CHANGE instead of
        absolute amplitude. Static rooms → diff ≈ 0. Motion → large diff.
        This is the most effective single technique for cross-environment HAR
        according to Widar3.0, CrossSense, and EI (Environment-Independent) papers.

    Usage:
      pipeline = CSIPipeline(fs=100.0)
      X_train = pipeline.fit_transform(train_matrix)   # trains + transforms
      X_test  = pipeline.transform(test_matrix)         # inference only
    """

    def __init__(self, fs: float = 100.0,
                 background_frames: int = 100,
                 use_diff: bool = True):
        """
        :param fs:                Sampling frequency Hz (100 Hz for our setup)
        :param background_frames: Frames used to estimate static background
                                  (default 100 = first 1 second at 100 Hz).
                                  Set to 0 to disable background subtraction.
        :param use_diff:          If True, apply temporal difference (np.diff).
                                  Set to False if your ML model needs fixed-length
                                  input without the N→N-1 reduction.
        """
        self.fs = fs
        self.background_frames = background_frames
        self.use_diff = use_diff

        self.pca: PCA | None = None
        self.scaler = None
        self.active_mask: np.ndarray | None = None
        self.background_mean: np.ndarray | None = None   # shape: (N_active,)
        self.is_fitted = False

    # ── 1. Null Removal ───────────────────────────────────────────────────
    def remove_null_subcarriers(self, complex_matrix: np.ndarray,
                                fit: bool = False) -> np.ndarray:
        """
        Remove always-zero subcarriers (guard / null bands).

        fit=True  → compute and store mask (call during training)
        fit=False → reuse stored mask     (call during inference)

        CRITICAL: without storing the mask, test data may have different
        zero patterns, giving the PCA a different number of features → crash.
        """
        amplitude = np.abs(complex_matrix)

        if fit or self.active_mask is None:
            self.active_mask = np.any(amplitude > 0, axis=0)

        if not np.any(self.active_mask):
            raise ValueError("No active subcarriers found — check data validity.")

        return amplitude[:, self.active_mask]

    # ── 2. Hampel Filter ──────────────────────────────────────────────────
    def apply_hampel_filter(self, data: np.ndarray,
                            window_size: int = 11,
                            n_sigmas: float = 3.0) -> np.ndarray:
        """
        Vectorized 2D Hampel outlier filter along time axis.
        MAD constant 1.4826 makes threshold equivalent to σ for Gaussian.
        """
        if window_size % 2 == 0:
            window_size += 1

        rolling_median = median_filter(data, size=(window_size, 1),
                                       mode='nearest')
        mad = median_filter(np.abs(data - rolling_median),
                            size=(window_size, 1), mode='nearest')
        threshold = n_sigmas * 1.4826 * mad

        # Only flag as outlier where mad > 0 (avoids false positives on flat signal)
        outliers = (mad > 0) & (np.abs(data - rolling_median) > threshold)

        clean = data.copy()
        clean[outliers] = rolling_median[outliers]
        return clean

    # ── 3. Butterworth Low-Pass ───────────────────────────────────────────
    def apply_lowpass_filter(self, data: np.ndarray,
                             cutoff: float = 12.0,
                             order: int = 4) -> np.ndarray:
        """
        Zero-phase Butterworth low-pass (filtfilt → no phase distortion).
        12 Hz cutoff: removes thermal noise, keeps human motion (<3 Hz).
        Minimum required length: 3*order+1 samples.
        """
        min_len = 3 * order + 1
        if data.shape[0] < min_len:
            print(f"⚠️  Butterworth skipped: {data.shape[0]} frames < {min_len}")
            return data

        nyquist = 0.5 * self.fs
        normal_cutoff = cutoff / nyquist

        if normal_cutoff >= 1.0:
            print(f"⚠️  Butterworth skipped: cutoff {cutoff} Hz >= Nyquist {nyquist} Hz")
            return data

        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data, axis=0)

    # ── 4. Background Subtraction ─────────────────────────────────────────
    def apply_background_subtraction(self, data: np.ndarray,
                                     fit: bool = False) -> np.ndarray:
        """
        Remove static environment signature by subtracting the mean of
        the first `background_frames` frames.

        Recording protocol requirement:
          The FIRST `background_frames` frames MUST be captured with the
          room empty and no movement. At 100 Hz and background_frames=100,
          this means the first 1 second of each recording = static room.

        fit=True  → estimate and store background_mean (training)
        fit=False → reuse stored background_mean        (inference)

        CRITICAL for inference: background_mean comes from training data.
        Do NOT re-estimate from test data — that would subtract a different
        room's signature and corrupt the signal.

        NOTE on output: the background frames are NOT removed from output.
        The full N-frame matrix is returned. The first background_frames
        rows will be ≈ 0 (static room after subtraction). Your ML pipeline
        should use these as a "calibration" period or exclude them via
        windowing after this step.

        After subtraction:
          - Static room    → values ≈ 0 (flat line)
          - Human movement → non-zero deviations (visible peaks)
        """
        if self.background_frames <= 0:
            return data   # disabled

        if fit or self.background_mean is None:
            n_bg = min(self.background_frames, data.shape[0])

            # Not enough frames for reliable background estimation
            if n_bg < 10:
                print(f"⚠️  Background subtraction: only {n_bg} frames available "
                      f"(need ≥10) — skipping")
                self.background_mean = None
                return data

            # ✅ FIX: warn when we get significantly fewer frames than requested
            if n_bg < self.background_frames:
                pct = 100.0 * n_bg / self.background_frames
                print(f"⚠️  Background subtraction: requested {self.background_frames} "
                      f"frames but only {n_bg} available ({pct:.0f}%). "
                      f"Ensure recording starts with an empty room.")

            # Mean over first n_bg frames, shape: (N_active_subcarriers,)
            self.background_mean = data[:n_bg].mean(axis=0)
            print(f"   Background estimated: first {n_bg} frames "
                  f"(≈{n_bg / self.fs:.1f}s)")

        if self.background_mean is None:
            return data

        # Subtract static background — broadcasts over all N frames
        return data - self.background_mean

    # ── 5. Temporal Difference ────────────────────────────────────────────
    def apply_temporal_diff(self, data: np.ndarray) -> np.ndarray:
        """
        First-order temporal difference: replaces absolute amplitude with
        the RATE OF CHANGE between consecutive frames.

        Why this makes HAR environment-independent:
          - A static room: frame[t] ≈ frame[t-1] → diff ≈ 0 for all t
          - A falling person: frame[t] changes rapidly → large diff values
          - The model sees "what changed" instead of "what the room looks like"
          - This is the key technique in Widar3.0, EI, and CrossSense papers

        Output shape: (N_frames - 1, N_subcarriers)
        The first frame is lost — this is mathematically unavoidable.
        For a 1000-frame recording at 100 Hz: 999 frames remain (9.99 s).

        NOTE: both training and inference data will have N-1 frames.
        Your ML model must be designed for variable-length inputs (RNN/LSTM)
        or use fixed-length windows AFTER this step.
        """
        if not self.use_diff:
            return data

        if data.shape[0] < 2:
            print("⚠️  Temporal diff skipped: need at least 2 frames")
            return data

        # np.diff along axis=0: out[i] = data[i+1] - data[i]
        return np.diff(data, n=1, axis=0).astype(np.float32)

    # ── fit_transform (Training) ──────────────────────────────────────────
    def fit_transform(self, complex_matrix: np.ndarray,
                      use_pca: bool = True,
                      n_components: int = 10,
                      scaler_type: str = 'standard') -> np.ndarray:
        """
        Train pipeline on training data and transform it.
        Saves active_mask, background_mean, PCA, Scaler for transform().

        Default scaler changed to 'standard' (Z-score) because it normalizes
        each component to mean=0, std=1 — making the model insensitive to
        the absolute signal strength which varies between environments.
        Use 'minmax' only if your model requires [0,1] input range.
        """
        print(f"🔧 fit_transform — input {complex_matrix.shape}")

        # [1] Null removal
        data = self.remove_null_subcarriers(complex_matrix, fit=True)
        print(f"   [1] Null removal: {complex_matrix.shape[1]} → "
              f"{data.shape[1]} subcarriers")

        # [2] Hampel
        data = self.apply_hampel_filter(data)
        print(f"   [2] Hampel ✅")

        # [3] Butterworth
        data = self.apply_lowpass_filter(data)
        print(f"   [3] Butterworth ✅")

        # [4] Background subtraction (fit=True estimates background)
        if self.background_frames > 0:
            data = self.apply_background_subtraction(data, fit=True)
            print(f"   [4] Background subtraction ✅  shape={data.shape}")
        else:
            print(f"   [4] Background subtraction: disabled")

        # [5] Temporal difference
        if self.use_diff:
            data = self.apply_temporal_diff(data)
            print(f"   [5] Temporal diff ✅  shape={data.shape}  "
                  f"(N-1={data.shape[0]} frames)")
        else:
            print(f"   [5] Temporal diff: disabled")

        # [6] PCA
        if use_pca:
            actual_n = min(n_components, data.shape[0] - 1, data.shape[1])
            self.pca = PCA(n_components=actual_n)
            data = self.pca.fit_transform(data)
            explained = self.pca.explained_variance_ratio_.sum() * 100
            print(f"   [6] PCA: {actual_n} components, {explained:.1f}% variance ✅")
        else:
            self.pca = None
            print(f"   [6] PCA: skipped")

        # [7] Scaler
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler_type '{scaler_type}'")

        data = self.scaler.fit_transform(data)
        print(f"   [7] {scaler_type} scaling → output {data.shape} ✅")

        self.is_fitted = True
        return data

    # ── transform (Inference) ─────────────────────────────────────────────
    def transform(self, complex_matrix: np.ndarray,
                  use_pca: bool = True) -> np.ndarray:
        """
        Transform new data using already-trained mask, background,
        PCA, and Scaler. Does NOT retrain anything.

        IMPORTANT: background_mean comes from training data — this is
        intentional. Applying a test room's background would defeat the
        purpose of background subtraction.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform() first.")

        data = self.remove_null_subcarriers(complex_matrix, fit=False)
        data = self.apply_hampel_filter(data)
        data = self.apply_lowpass_filter(data)

        # Apply stored background (fit=False reuses background_mean)
        if self.background_frames > 0:
            data = self.apply_background_subtraction(data, fit=False)

        # Temporal diff
        if self.use_diff:
            data = self.apply_temporal_diff(data)

        if use_pca and self.pca is not None:
            data = self.pca.transform(data)

        data = self.scaler.transform(data)
        return data

    @property
    def n_active_subcarriers(self) -> int:
        return int(self.active_mask.sum()) if self.active_mask is not None else 0


# ════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
        print(f"\n📂 Loading: {csv_path}")
        complex_matrix, meta = load_csi_csv(csv_path)
    else:
        print("ℹ️  Simulation mode")
        np.random.seed(42)
        r  = np.random.randn(500, 128).astype(np.float32) * 10
        im = np.random.randn(500, 128).astype(np.float32) * 10
        complex_matrix = (r + 1j * im).astype(np.complex64)
        complex_matrix[:, :6]  = 0
        complex_matrix[:, -6:] = 0
        # Simulate static room: add constant offset to first 50 frames
        complex_matrix[:50] += 5.0

    if complex_matrix.size == 0:
        print("❌ Empty matrix — exiting")
        sys.exit(1)

    print(f"\n📊 Input: {complex_matrix.shape}")

    # Default: environment-independent pipeline
    pipeline = CSIPipeline(
        fs=100.0,
        background_frames=100,  # first 1 second = static background
        use_diff=True,           # temporal diff for environment independence
    )
    processed = pipeline.fit_transform(
        complex_matrix,
        use_pca=True,
        n_components=10,
        scaler_type='standard'  # Z-score for cross-environment robustness
    )
    print(f"\n✅ Output: {processed.shape}")
    print(f"   Mean={processed.mean():.4f} | Std={processed.std():.4f}  "
          f"(should be ~0 and ~1 for standard scaler)")
    print(f"   Active subcarriers: {pipeline.n_active_subcarriers}")
    print(f"\n   Note: output has N-1={processed.shape[0]} frames "
          f"(temporal diff reduces by 1)")