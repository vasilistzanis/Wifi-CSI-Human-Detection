#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
CSI HAR - Complete ML Pipeline
====================================
Supports: SVM, Random Forest, K-NN, Logistic Regression, Extra Trees, Naive Bayes
Compatible with: CSIPipeline from data_preprocessing.py


Features:
  - Advanced Feature Extraction (Statistical + FFT Doppler analysis)
  - Augmented Windowing to prevent data scarcity
  - GroupKFold Validation to completely eliminate data leakage
  - Probability-based voting for robust continuous inference
  - Hyperparameter tuning via GridSearchCV
  - Export of Feature Importances and Metrics


Usage:
  python csi_ml_pipeline.py --classes walk_activity no_activity
  python csi_ml_pipeline.py --classes walk_activity sit fall no_activity --save_model --tune
"""

# FEATURE VECTOR CHANGE NOTICE:
# Kurtosis now uses excess kurtosis (value - 3); fft_peak_idx is normalised to (0,1].
# All saved .joblib model files must be retrained before deployment.

# Increment this string whenever the feature extraction semantics change
# (formula, normalisation, added/removed features).  It is written to
# metrics.json at save time and checked by benchmark_latency.py at load time.
FEATURE_VECTOR_VERSION = "4"  # ΑΛΛΑΓΗ

import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
import config


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    GroupKFold, cross_val_score, GridSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline  # ΑΛΛΑΓΗ
from sklearn.preprocessing import LabelEncoder, StandardScaler  # ΑΛΛΑΓΗ


import warnings
# Suppress only well-understood non-actionable warnings; leave convergence
# warnings visible so MLP/LR training failures are not silently ignored.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*solver.*lbfgs.*", module="sklearn")


# DWT removed: 10 Hz cutoff kills d1/d2; window_size=50 only allows depth-2 decomposition.
# Re-enable by setting WINDOW_SIZE≥100 and cutoff≥25 Hz.
_DWT_STATS_PER_COMPONENT = 0
N_STATS = 22  # ΑΛΛΑΓΗ


try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None


# ========================================================================
# IMPORT PREPROCESSING PIPELINE
# ========================================================================


try:
    from data_preprocessing import CSIPipeline, load_csi_csv
    print("[OK] CSIPipeline imported successfully")
except ImportError:
    print("[WARNING]  data_preprocessing.py not found - using simulation mode")
    CSIPipeline = None
    load_csi_csv = None




# ========================================================================
# 1. DATA AUGMENTATION  (applied on RAW amplitude windows, BEFORE PCA)
# ========================================================================


ALL_AUGMENT_TECHNIQUES = list(config.AUGMENTATION_TECHNIQUES)




def _aug_noise(window: np.ndarray, rng: np.random.Generator, class_label: str = None) -> np.ndarray:
    """Scaled Gaussian noise relative to signal standard deviation. Reduces intensity for falls."""
    signal_std = np.std(window)
    noise_level = rng.uniform(0.003, 0.01) * (signal_std if signal_std > 1e-6 else 1.0)
    

    if class_label == 'fall':
        noise_level *= 0.5
    elif class_label == 'sit':
        noise_level *= 0.7
        

    return window + rng.normal(0, noise_level, window.shape)




def _aug_shift(window: np.ndarray, rng: np.random.Generator, class_label: str = None) -> np.ndarray:
    """Non-circular temporal shift (padding-based). Avoids discontinuities."""
    shift_steps = int(rng.integers(1, 4))
    direction = rng.choice([-1, 1])
    

    if direction == 1:
        # Shift forward: pad start with edge value, drop end
        pad = np.repeat(window[0:1], shift_steps, axis=0)
        return np.vstack([pad, window[:-shift_steps]])
    else:
        # Shift backward: drop start, pad end with edge value
        pad = np.repeat(window[-1:], shift_steps, axis=0)
        return np.vstack([window[shift_steps:], pad])




def _aug_scale(window: np.ndarray, rng: np.random.Generator, class_label: str = None) -> np.ndarray:
    """Physics-aware magnitude scaling. Tighter constraints for falls."""
    if class_label == 'fall':
        scale = rng.uniform(0.97, 1.03)
    elif class_label == 'sit':
        scale = rng.uniform(0.95, 1.05)
    else:
        scale = rng.uniform(0.9, 1.1)
    return window * scale




def _aug_time_warp(window: np.ndarray, rng: np.random.Generator, class_label: str = None) -> np.ndarray:
    """Advanced Time Warp with Reflect Padding to avoid artifacts."""
    T = window.shape[0]
    

    # Class-aware factor selection
    if class_label == 'walk_activity':
        factor = rng.uniform(0.9, 1.1)
    elif class_label == 'sit':
        factor = rng.uniform(0.95, 1.05)
    else:  # no_activity/static
        factor = rng.uniform(0.98, 1.02)
        

    src_indices = np.linspace(0, T - 1, T) / factor

    # Reflect padding — avoids flat tails at window boundaries
    overflow = src_indices > (T - 1)
    src_indices[overflow] = 2 * (T - 1) - src_indices[overflow]
    src_indices = np.clip(src_indices, 0, T - 1)

    warped = np.empty_like(window, dtype=np.float64)
    for c in range(window.shape[1]):
        warped[:, c] = np.interp(src_indices, np.arange(T), window[:, c])

    return warped.astype(np.float32)




_AUG_FN_MAP = {
    'noise':     _aug_noise,
    'shift':     _aug_shift,
    'scale':     _aug_scale,
    'time_warp': _aug_time_warp,
}




def augment_window(window: np.ndarray,
                   n_augments: int = config.N_AUGMENTS,
                   techniques: list = None,
                   seed: int = None,
                   class_label: str = None) -> list:
    """
    Advanced Class-Aware Multi-Stage Augmenter (BEFORE PCA).
    Applies selected techniques sequentially based on class physical constraints.
    """
    if techniques is None:
        techniques = ALL_AUGMENT_TECHNIQUES
    if not techniques:
        return []


    rng = np.random.default_rng(seed)
    augmented_windows = []


    # Fall safety: Gravity doesn't warp
    safe_techs = [t for t in techniques if t != 'time_warp'] if class_label == 'fall' else techniques


    if not safe_techs:
        warnings.warn(
            f"augment_window: class_label='{class_label}' filtered out all requested "
            f"techniques {techniques}. Falling back to 'noise' to preserve dataset size.",
            RuntimeWarning, stacklevel=2
        )
        safe_techs = ['noise']


    for _ in range(n_augments):
        # Pick 1 or 2 techniques
        n_to_apply = rng.choice([1, 2])
        chosen = rng.choice(safe_techs, size=min(n_to_apply, len(safe_techs)), replace=False)


        aug = window.copy()
        for tech in chosen:
            if tech in _AUG_FN_MAP:
                aug = _AUG_FN_MAP[tech](aug, rng, class_label=class_label)


        augmented_windows.append(aug.astype(np.float32))


    return augmented_windows


# ========================================================================
# 2. FEATURE EXTRACTION
# ========================================================================


def _dwt_features_for_col(col: np.ndarray, wavelet: str = 'db4',
                          level: int = 3) -> list:
    """
    DWT features disabled — returns empty list.

    To re-enable: increase WINDOW_SIZE to ≥100 frames AND raise the
    Butterworth cutoff to ≥25 Hz, then set _DWT_STATS_PER_COMPONENT = 2
    and return [energy_d3, std_d3] from the coarsest detail coefficient.
    """
    return []




def extract_features_from_window(window: np.ndarray, fs: float = config.SAMPLING_RATE, cutoff_hz: float = config.FILTER_CUTOFF_HZ) -> np.ndarray:  # ΑΛΛΑΓΗ
    """
    22 features per PCA component -> flat feature vector.

    Input:  (window_size, n_pca_components)  e.g. (100, 10)
    Output: (220,)  [22 features x 10 components]

    Feature breakdown (22 per component):
      mean, std, max, min, range, median, energy,
      skewness, excess_kurtosis, fft_mean, fft_std, zcr,
      fft_peak_idx, spectral_entropy,
      autocorr_peak, autocorr_dominant_lag, gait_band_ratio,
      spectral_centroid, peak_prominence, signal_mobility,
      signal_complexity, waveform_length
    """
    feats = []
    for c in range(window.shape[1]):
        col      = window[:, c].astype(np.float64)
        mean_val = col.mean()
        std_val  = col.std() + 1e-8

        # ----------------------------------------------------------------
        # FFT — υπολογίζεται ΜΙΑ ΦΟΡΑ, χρησιμοποιείται από πολλά features
        # Περιορίζεται στο active bandwidth [0, cutoff_hz] ώστε τα bins
        # πάνω από το Butterworth cutoff (≈ 0) να μην αραιώνουν τα stats.
        # ----------------------------------------------------------------
        fft_vals        = np.abs(np.fft.rfft(col))
        freqs           = np.fft.rfftfreq(len(col), d=1.0 / fs)
        fft_power       = fft_vals ** 2

        active_mask_fft  = freqs <= cutoff_hz                    # ΑΛΛΑΓΗ — e.g. bins 0–10 Hz
        fft_vals_active  = fft_vals[active_mask_fft]             # ΑΛΛΑΓΗ — e.g. 11 bins
        freqs_active     = freqs[active_mask_fft]                # ΑΛΛΑΓΗ

        fft_vals_no_dc  = fft_vals_active[1:]                    # ΑΛΛΑΓΗ — exclude DC, e.g. 10 bins
        freqs_no_dc     = freqs_active[1:]                       # ΑΛΛΑΓΗ
        n_bins_active   = len(fft_vals_no_dc)                    # ΑΛΛΑΓΗ — replaces n_bins (=50→10)

        fft_mean = float(fft_vals_active.mean())                 # ΑΛΛΑΓΗ — active band only
        fft_std  = float(fft_vals_active.std())                  # ΑΛΛΑΓΗ — active band only

        # ----------------------------------------------------------------
        # ZCR
        # ----------------------------------------------------------------
        centered = col - mean_val
        zcr = float(np.sum(np.diff(np.sign(centered)) != 0) / max(1, len(col) - 1))

        # ----------------------------------------------------------------
        # Dominant frequency index (exclude DC)
        # ----------------------------------------------------------------
        fft_peak_idx = float(np.argmax(fft_vals_no_dc) + 1) / n_bins_active  # ΑΛΛΑΓΗ — normalised within active band (0,1]

        # ----------------------------------------------------------------
        # Spectral entropy (exclude DC)
        # ----------------------------------------------------------------
        prob             = fft_vals_no_dc / (np.sum(fft_vals_no_dc) + 1e-8)
        spectral_entropy = float(-np.sum(prob * np.log2(prob + 1e-8)))

        # ----------------------------------------------------------------
        # Autocorrelation — υπολογίζεται ΜΙΑ ΦΟΡΑ, χρησιμοποιείται από 2 features
        # ----------------------------------------------------------------
        col_norm      = (col - mean_val) / std_val  # ΑΛΛΑΓΗ
        autocorr_full = np.correlate(col_norm, col_norm, mode='full')  # ΑΛΛΑΓΗ
        autocorr      = autocorr_full[len(col_norm)-1:] / (len(col_norm) * 1.0)  # ΑΛΛΑΓΗ
        autocorr_lags = autocorr[1:]  # lag=0 εξαιρείται παντού  # ΑΛΛΑΓΗ

        # ----------------------------------------------------------------
        # Diff — υπολογίζεται ΜΙΑ ΦΟΡΑ, χρησιμοποιείται από 3 features
        # ----------------------------------------------------------------
        diff1 = np.diff(col)  # ΑΛΛΑΓΗ
        diff2 = np.diff(diff1)  # ΑΛΛΑΓΗ

        # ================================================================
        # FEATURES 1–14 (υπάρχοντα — ΜΗΝ ΑΛΛΑΞΕΙΣ)
        # ================================================================
        feats.extend([
            mean_val,
            std_val,
            col.max(),
            col.min(),
            col.max() - col.min(),
            float(np.median(col)),
            float(np.sum(col ** 2)),
            float(np.mean(((col - mean_val) / std_val) ** 3)),
            float(np.mean(((col - mean_val) / std_val) ** 4)) - 3.0,
            fft_mean,
            fft_std,
            zcr,
            fft_peak_idx,
            spectral_entropy,
        ])

        # ================================================================
        # DWT (υπάρχον — ΜΗΝ ΑΛΛΑΞΕΙΣ, επιστρέφει [] αυτή τη στιγμή)
        # ================================================================
        feats.extend(_dwt_features_for_col(col, wavelet='db4', level=3))

        # ================================================================
        # FEATURES 15–22
        # ================================================================

        # 15. autocorr_peak — max periodicity strength
        autocorr_peak = float(np.max(autocorr_lags))  # ΑΛΛΑΓΗ

        # 16. autocorr_dominant_lag — gait cycle duration estimator (sec)
        autocorr_dominant_lag = float(np.argmax(autocorr_lags) + 1) / fs  # ΑΛΛΑΓΗ

        # 17. gait_band_ratio — energy ratio in 0.5–3 Hz zone
        gait_mask        = (freqs >= 0.5) & (freqs <= 3.0)  # ΑΛΛΑΓΗ
        gait_energy      = float(np.sum(fft_power[gait_mask]))  # ΑΛΛΑΓΗ
        total_energy_fft = float(np.sum(fft_power) + 1e-8)  # ΑΛΛΑΓΗ
        gait_band_ratio  = gait_energy / total_energy_fft  # ΑΛΛΑΓΗ

        # 18. spectral_centroid — weighted mean frequency
        spectral_centroid = float(  # ΑΛΛΑΓΗ
            np.sum(freqs_no_dc * fft_vals_no_dc) / (np.sum(fft_vals_no_dc) + 1e-8)
        )

        # 19. peak_prominence — sharpness of dominant FFT peak
        peak_prominence = float(np.max(fft_vals_no_dc)) - float(np.mean(fft_vals_no_dc))  # ΑΛΛΑΓΗ

        # 20. signal_mobility — Hjorth: signal velocity
        signal_mobility = float(np.std(diff1) / (np.std(col) + 1e-8))  # ΑΛΛΑΓΗ

        # 21. signal_complexity — Hjorth: waveform complexity
        mobility_diff1    = float(np.std(diff2) / (np.std(diff1) + 1e-8))  # ΑΛΛΑΓΗ
        signal_complexity = mobility_diff1 / (signal_mobility + 1e-8)  # ΑΛΛΑΓΗ

        # 22. waveform_length — total activity measure
        waveform_length = float(np.sum(np.abs(diff1)))  # ΑΛΛΑΓΗ

        feats.extend([  # ΑΛΛΑΓΗ
            autocorr_peak,
            autocorr_dominant_lag,
            gait_band_ratio,
            spectral_centroid,
            peak_prominence,
            signal_mobility,
            signal_complexity,
            waveform_length,
        ])

    return np.array(feats, dtype=np.float32)




def _get_feature_names(n_pca_components: int) -> list[str]:
    classical = ['mean', 'std', 'max', 'min', 'range', 'median',
                 'energy', 'skewness', 'excess_kurtosis', 'fft_mean', 'fft_std',
                 'zcr', 'fft_peak_idx', 'spectral_entropy',
                 'autocorr_peak', 'autocorr_dominant_lag', 'gait_band_ratio',  # ΑΛΛΑΓΗ
                 'spectral_centroid', 'peak_prominence', 'signal_mobility',     # ΑΛΛΑΓΗ
                 'signal_complexity', 'waveform_length']                        # ΑΛΛΑΓΗ
    # DWT removed — see _DWT_STATS_PER_COMPONENT comment at top of file
    all_stats = classical   # 22 total  # ΑΛΛΑΓΗ
    return [f"PC{c+1}_{s}" for c in range(n_pca_components) for s in all_stats]


# Update this dict whenever a new feature is added to _get_feature_names above.
_STAT_TO_GROUP: dict[str, str] = {
    # Statistical — time-domain descriptors
    'mean': 'Statistical', 'std': 'Statistical', 'max': 'Statistical',
    'min': 'Statistical', 'range': 'Statistical', 'median': 'Statistical',
    'energy': 'Statistical', 'skewness': 'Statistical', 'excess_kurtosis': 'Statistical',
    'zcr': 'Statistical',
    # FFT / frequency-domain
    'fft_mean': 'FFT', 'fft_std': 'FFT',
    'fft_peak_idx': 'FFT', 'spectral_entropy': 'FFT',
    'gait_band_ratio': 'FFT', 'spectral_centroid': 'FFT', 'peak_prominence': 'FFT',
    # Temporal — autocorrelation / Hjorth / morphology
    'autocorr_peak': 'Temporal', 'autocorr_dominant_lag': 'Temporal',
    'signal_mobility': 'Temporal', 'signal_complexity': 'Temporal',
    'waveform_length': 'Temporal',
}


def _classify_feature(feature_name: str) -> str:
    """Map 'PC3_spectral_entropy' → group string. Falls back to 'Other'."""
    parts = feature_name.split("_", 1)
    stat = parts[1] if len(parts) == 2 and parts[0].startswith("PC") else feature_name
    return _STAT_TO_GROUP.get(stat, "Other")


# Colors for each feature group — update here when adding a new group to _STAT_TO_GROUP.
GROUP_COLORS: dict[str, str] = {
    "Statistical": "#2563eb",
    "FFT":         "#f59e0b",
    "Temporal":    "#10b981",
    "Other":       "#8b5cf6",
}




def extract_windows(data: np.ndarray,
                    window_size: int = config.WINDOW_SIZE,
                    step: int = config.PIPELINE_STEP_SIZE) -> list[np.ndarray]:
    """Sliding window -> list of (window_size, n_components) arrays."""
    if data.shape[0] < window_size:
        return []
    return [data[s:s + window_size]
            for s in range(0, data.shape[0] - window_size + 1, step)]




def _make_group_cv(y: np.ndarray,
                   groups: np.ndarray,
                   requested_folds: int = config.CV_FOLDS,
                   random_seed: int = config.RANDOM_SEED) -> tuple:
    """
    Build a group-aware CV splitter so windows from the same recording
    never appear in both train and validation folds.
    """
    if len(y) != len(groups):
        raise ValueError("y and groups must have the same length")


    group_to_label = {}
    for label, group in zip(y, groups):
        group = int(group)
        label = int(label)
        if group in group_to_label and group_to_label[group] != label:
            raise ValueError("Each recording group must belong to exactly one class")
        group_to_label[group] = label


    if len(group_to_label) < 2:
        raise ValueError("Need at least 2 train recordings for group-based CV.")


    class_group_counts = Counter(group_to_label.values())
    max_stratified_folds = min(class_group_counts.values()) if class_group_counts else 0


    if StratifiedGroupKFold is not None and max_stratified_folds >= 2:
        n_splits = min(requested_folds, max_stratified_folds, len(group_to_label))
        try:
            # shuffle + random_state supported from sklearn 1.3
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_seed
            )
        except TypeError:
            splitter = StratifiedGroupKFold(n_splits=n_splits)
        splitter_name = "StratifiedGroupKFold"
    else:
        n_splits = min(requested_folds, len(group_to_label))
        if n_splits < 2:
            raise ValueError("Need at least 2 train recordings for GroupKFold.")
        splitter = GroupKFold(n_splits=n_splits)
        splitter_name = "GroupKFold"
        print("Warning: StratifiedGroupKFold unavailable or unsupported by"
              " class counts; falling back to GroupKFold.")


    return splitter, n_splits, splitter_name


def _resolve_split_paths(paths: list[str] | list[Path], data_dir: Path) -> list[Path]:
    """Normalize split entries from JSON into absolute Paths."""
    resolved = []
    for entry in paths:
        path = Path(entry)
        if not path.is_absolute():
            path = data_dir / path
        resolved.append(path)
    return resolved


def _serialize_split_paths(paths: list[Path], data_dir: Path) -> list[str]:
    """Store split entries relative to data_dir when possible."""
    serialized = []
    root = data_dir.resolve()
    for path in paths:
        resolved = Path(path).resolve()
        try:
            serialized.append(resolved.relative_to(root).as_posix())
        except ValueError:
            serialized.append(str(resolved))
    return serialized




# ========================================================================
# 3. DATASET BUILDER
# ========================================================================
# Loads recordings, handles train/test splitting at the file-level, 
# applies augmentation on training data only to prevent leakage,
# and returns the separated features.
# ========================================================================


def build_dataset(
    data_dir: str | Path,
    classes: list[str],
    pipeline_kwargs: dict = None,
    window_size: int = config.WINDOW_SIZE,
    step: int = config.PIPELINE_STEP_SIZE,
    augment_techniques: list = None,
    n_augments: int = config.N_AUGMENTS,
    simulation_mode: bool = False,
    test_recording_ratio: float = config.TEST_RATIO,
    random_seed: int = config.RANDOM_SEED,
    n_pca: int = config.N_PCA_COMPONENTS,
    cutoff: float = config.FILTER_CUTOFF_HZ,
    train_files_override: dict | None = None,
    test_files_override: dict | None = None,
    pipeline_override = None,
    label_encoder_override = None,
) -> tuple:
    """
    Load recordings, preprocess, extract features.
    Augmentation is applied on RAW amplitude windows BEFORE PCA projection
    (physics-aware: noise, shift, scale, time_warp with strict limits).
    Returns train/test split at recording level (no leakage).


    Args:
      augment_techniques : list of technique names to use, e.g. ['noise', 'scale'].
                           None or empty list -> no augmentation.
                           Default (when called from main): ALL_AUGMENT_TECHNIQUES.


    Returns:
      X_train      : (N, n_pca * N_STATS) augmented train features
      X_train_orig : (N_orig, n_pca * N_STATS) non-augmented train features for clean CV
      X_test       : (M, n_pca * N_STATS) test features (no augmentation)
      y_train      : (N,) labels for X_train
      y_train_orig : (N_orig,) labels for X_train_orig
      y_test       : (M,) labels for X_test
      train_groups_orig : (N_orig,) recording ids for X_train_orig
      le           : fitted LabelEncoder
      pipeline     : fitted CSIPipeline
      dataset_info : exact split / preprocessing metadata
    """
    if pipeline_kwargs is None:
        pipeline_kwargs = {'fs': config.SAMPLING_RATE, 'use_diff': True}

    _fs     = float(pipeline_kwargs.get('fs', config.SAMPLING_RATE))  # ΑΛΛΑΓΗ
    _cutoff = float(cutoff)                            # ΑΛΛΑΓΗ — passed to bandwidth-limited FFT features

    do_augment = bool(augment_techniques)  # empty list / None -> no augmentation
    if do_augment:
        print(f"   Augmentation techniques: {augment_techniques}")


    data_dir = Path(data_dir)
    requested_classes = list(classes)
    classes, class_dirs = config.resolve_training_classes(
        requested_classes,
        data_dir=data_dir,
        require_existing=not (simulation_mode or CSIPipeline is None),
    )

    if label_encoder_override is not None:
        encoder_classes = set(label_encoder_override.classes_)
        skipped_encoder = [cls for cls in classes if cls not in encoder_classes]
        for cls in skipped_encoder:
            print(f"[WARNING] Training class '{cls}' is not present in the loaded label encoder - skipped.")
        classes = [cls for cls in classes if cls in encoder_classes]

    if not classes:
        raise ValueError(
            "No training classes remain after applying config enable/disable rules "
            "and dataset-folder validation."
        )

    if label_encoder_override is not None:
        le = label_encoder_override
    else:
        le = LabelEncoder()
        le.fit(classes)

    dataset_info = {
        'data_dir': str(data_dir.resolve()),
        'requested_classes': list(requested_classes),
        'classes': list(classes),
        'pipeline_kwargs': dict(pipeline_kwargs),
        'window_size': int(window_size),
        'step': int(step),
        'test_recording_ratio': float(test_recording_ratio),
        'random_seed': int(random_seed),
        'n_pca': int(n_pca),
        'cutoff': float(cutoff),
        'simulation_mode': bool(simulation_mode or CSIPipeline is None),
        'train_files': {},
        'test_files': {},
    }


    # -- Simulation Mode --------------------------------------------------
    if simulation_mode or CSIPipeline is None:
        print("\n[INFO] SIMULATION MODE")
        sim_rng = np.random.default_rng(random_seed)
        X_tr, y_tr = [], []
        X_tr_orig, y_tr_orig = [], []
        train_groups_orig = []
        X_te, y_te = [], []
        recording_group_id = 0
        global_window_idx = 0


        # Collect all synthetic CMs first
        rec_data = []


        for label_idx, cls in enumerate(classes):
            n_recs = 20
            n_test = max(1, int(n_recs * test_recording_ratio))
            print(f"   [{cls}] {n_recs} synthetic recordings "
                  f"(train={n_recs-n_test}, test={n_test})")


            for rec_i in range(n_recs):
                t    = np.linspace(0, 5, 500)
                freq = 1.0 + label_idx * 0.5
                r    = (np.outer(np.sin(2*np.pi*freq*t), np.ones(128))
                        + sim_rng.standard_normal((500, 128)) * 0.3)
                im   = (np.outer(np.cos(2*np.pi*freq*t), np.ones(128))
                        + sim_rng.standard_normal((500, 128)) * 0.3)
                cm   = (r + 1j*im).astype(np.complex64)
                cm[:, :6]  = 0
                cm[:, -6:] = 0
                

                is_test = (rec_i >= n_recs - n_test)
                rec_data.append((cm, label_idx, is_test))


        pp = CSIPipeline(**pipeline_kwargs) if CSIPipeline else None
        if pp:
            print("   Fitting single CSIPipeline for simulation...")
            train_cms = [r[0] for r in rec_data if not r[2]]
            if train_cms:
                pp.fit_from_recordings(train_cms, use_pca=True, n_components=n_pca,
                                      scaler_type='standard', cutoff=cutoff)


        for cm, label_idx, is_test in rec_data:
            # -- Get RAW amplitude (pre-PCA) for augmentation --------------
            if pp:
                # Replicate pipeline steps up to (but not including) PCA+scaler
                amp = pp.remove_null_subcarriers(cm, fit=False)
                amp = pp.apply_hampel_filter(amp)
                amp = pp.apply_lowpass_filter(amp, cutoff=cutoff)
                if pp.use_diff:
                    amp = pp.apply_temporal_diff(amp)
                raw_pre_pca = amp  # shape: (N_frames, n_active_subcarriers)
            else:
                raw_pre_pca = sim_rng.standard_normal((499, 114)).astype(np.float32)


            for w_raw in extract_windows(raw_pre_pca, window_size, step):
                if is_test:
                    # Test: project through PCA+scaler, then extract features
                    if pp:
                        w_proj = pp.pca.transform(w_raw)
                        w_proj = pp.scaler.transform(w_proj)
                    else:
                        w_proj = w_raw[:, :n_pca]
                    X_te.append(extract_features_from_window(w_proj, fs=_fs, cutoff_hz=_cutoff))  # ΑΛΛΑΓΗ
                    y_te.append(label_idx)
                else:
                    # Train original (no augmentation)
                    if pp:
                        w_proj = pp.pca.transform(w_raw)
                        w_proj = pp.scaler.transform(w_proj)
                    else:
                        w_proj = w_raw[:, :n_pca]
                    feat_orig = extract_features_from_window(w_proj, fs=_fs, cutoff_hz=_cutoff)  # ΑΛΛΑΓΗ
                    X_tr_orig.append(feat_orig)
                    y_tr_orig.append(label_idx)
                    train_groups_orig.append(recording_group_id)
                    X_tr.append(feat_orig)
                    y_tr.append(label_idx)


                    # Augmentation on RAW window, THEN project
                    if do_augment:
                        cls_name = classes[label_idx]
                        for aw_raw in augment_window(
                                w_raw, n_augments,
                                techniques=augment_techniques,
                                seed=random_seed + global_window_idx,
                                class_label=cls_name):
                            if not np.isfinite(aw_raw).all():
                                continue
                            if pp:
                                aw_proj = pp.pca.transform(aw_raw)
                                aw_proj = pp.scaler.transform(aw_proj)
                            else:
                                aw_proj = aw_raw[:, :n_pca]
                            X_tr.append(extract_features_from_window(aw_proj, fs=_fs, cutoff_hz=_cutoff))  # ΑΛΛΑΓΗ
                            y_tr.append(label_idx)
                    global_window_idx += 1
            recording_group_id += 1


        X_train      = np.array(X_tr,      dtype=np.float32)
        X_train_orig = np.array(X_tr_orig, dtype=np.float32)
        X_test       = np.array(X_te,      dtype=np.float32)
        y_train      = np.array(y_tr,      dtype=np.int32)
        y_train_orig = np.array(y_tr_orig, dtype=np.int32)
        y_test       = np.array(y_te,      dtype=np.int32)
        train_groups_orig = np.array(train_groups_orig, dtype=np.int32)


        print(f"\n[OK] Train={len(X_train)} (orig={len(X_train_orig)}) "
              f"| Test={len(X_test)} samples")
        return (X_train, X_train_orig, X_test,
                y_train, y_train_orig, y_test, train_groups_orig, le, None,
                dataset_info)


    # -- Real Data Mode ---------------------------------------------------
    print(f"\n[FILE] Loading data from: {data_dir}")


    train_files_all = {}
    test_files_all  = {}


    use_explicit_split = (
        train_files_override is not None or test_files_override is not None
    )

    if use_explicit_split:
        print("\n[INFO] Using explicit recording split from saved experiment metadata...")
        train_files_override = train_files_override or {}
        test_files_override = test_files_override or {}
        for cls in classes:
            train_files_all[cls] = _resolve_split_paths(
                train_files_override.get(cls, []), data_dir
            )
            test_files_all[cls] = _resolve_split_paths(
                test_files_override.get(cls, []), data_dir
            )
    else:
        for cls in classes:
            class_dir = class_dirs.get(cls, data_dir / config.get_training_class_folder(cls))
            files = (sorted(class_dir.glob("*.csv")) +
                     sorted(class_dir.glob("*.txt")))
            if not files:
                print(f"[WARNING]  No files found for class '{cls}'")
                train_files_all[cls] = []
                test_files_all[cls]  = []
                continue
            # Shuffle: recordings are independent sessions, not time-dependent.
            # Seeded shuffle ensures reproducibility while removing ordering bias.
            random.Random(random_seed).shuffle(files)
            n_test = max(1, int(len(files) * test_recording_ratio))
            train_files_all[cls] = files[:-n_test]
            test_files_all[cls]  = files[-n_test:]

    dataset_info['train_files'] = {
        cls: _serialize_split_paths(train_files_all.get(cls, []), data_dir)
        for cls in classes
    }
    dataset_info['test_files'] = {
        cls: _serialize_split_paths(test_files_all.get(cls, []), data_dir)
        for cls in classes
    }


    fit_matrices = []
    for cls in classes:
        for fpath in train_files_all.get(cls, []):
            try:
                cm, _ = load_csi_csv(fpath)
                if cm.size > 0:
                    fit_matrices.append(cm)
            except Exception as e:
                print(f"   [WARNING]  Initial fit skip {fpath.name}: {e}")
                continue


    if not fit_matrices:
        raise ValueError("No valid training CSI data found.")


    if pipeline_override is not None:
        print("\n[INFO] Using pre-fitted CSIPipeline from saved artifacts...")
        pipeline = pipeline_override
    else:
        print("\n[INFO] Fitting CSIPipeline on TRAIN recordings only (per-recording DSP)...")
        pipeline = CSIPipeline(**pipeline_kwargs)
        pipeline.fit_from_recordings(
            fit_matrices,
            use_pca=True, n_components=n_pca,
            scaler_type="standard", cutoff=cutoff,
        )


    X_tr, y_tr = [], []
    X_tr_orig, y_tr_orig = [], []
    train_groups_orig = []
    X_te, y_te = [], []
    recording_group_id = 0
    global_window_idx = 0


    for cls in classes:
        label_idx   = int(le.transform([cls])[0])
        train_files = train_files_all.get(cls, [])
        test_files  = test_files_all.get(cls, [])


        print(f"\n   [{cls}]  "
              f"train={len(train_files)} | test={len(test_files)} recordings")


        tr_wins = 0
        aug_count_cls = 0
        for fpath in train_files:
            try:
                cm, _ = load_csi_csv(fpath)
                if cm.size == 0:
                    continue
                # -- Pre-PCA steps (for augmentation on raw signal) --------
                amp = pipeline.remove_null_subcarriers(cm, fit=False)
                amp = pipeline.apply_hampel_filter(amp)
                amp = pipeline.apply_lowpass_filter(amp, cutoff=cutoff)
                if pipeline.use_diff:
                    amp = pipeline.apply_temporal_diff(amp)
                raw_pre_pca = amp  # (N_frames, n_active_subcarriers)
            except Exception as e:
                print(f"   [WARNING]  {fpath.name}: {e} - skipped")
                continue


            for w_raw in extract_windows(raw_pre_pca, window_size, step):
                # Project original window through PCA+scaler
                w_proj = pipeline.pca.transform(w_raw)
                w_proj = pipeline.scaler.transform(w_proj)
                feat_orig = extract_features_from_window(w_proj, fs=_fs, cutoff_hz=_cutoff)  # ΑΛΛΑΓΗ


                X_tr_orig.append(feat_orig)
                y_tr_orig.append(label_idx)
                train_groups_orig.append(recording_group_id)
                X_tr.append(feat_orig)
                y_tr.append(label_idx)
                tr_wins += 1


                # Augmentation: on RAW window -> then project -> features
                if do_augment:
                    for aw_raw in augment_window(
                            w_raw, n_augments,
                            techniques=augment_techniques,
                            seed=random_seed + global_window_idx,
                            class_label=cls):
                        if not np.isfinite(aw_raw).all():
                            continue
                        aw_proj = pipeline.pca.transform(aw_raw)
                        aw_proj = pipeline.scaler.transform(aw_proj)
                        X_tr.append(extract_features_from_window(aw_proj, fs=_fs, cutoff_hz=_cutoff))  # ΑΛΛΑΓΗ
                        y_tr.append(label_idx)
                        aug_count_cls += 1
                global_window_idx += 1
            recording_group_id += 1


        te_wins = 0
        for fpath in test_files:
            try:
                cm, _ = load_csi_csv(fpath)
                if cm.size == 0:
                    continue
                processed = pipeline.transform(cm, use_pca=True, cutoff=cutoff)
            except Exception as e:
                print(f"   [WARNING]  {fpath.name}: {e} - skipped")
                continue
            # Test files: use full pipeline.transform (no augmentation)
            for w in extract_windows(processed, window_size, step):
                X_te.append(extract_features_from_window(w, fs=_fs, cutoff_hz=_cutoff))  # ΑΛΛΑΓΗ
                y_te.append(label_idx)
                te_wins += 1


        print(f"   -> train: {tr_wins} orig + {aug_count_cls} augmented | "
              f"test: {te_wins} windows")


    if not X_tr:
        raise ValueError("No training features extracted.")
    if not X_te:
        raise ValueError("No test features extracted.")

    # Safety: augmented samples must NOT be tracked in groups (CV uses orig only)
    assert len(X_tr_orig) == len(train_groups_orig), \
        f"Group tracking mismatch: {len(X_tr_orig)} orig samples vs {len(train_groups_orig)} groups"


    X_train      = np.array(X_tr,      dtype=np.float32)
    X_train_orig = np.array(X_tr_orig, dtype=np.float32)
    X_test       = np.array(X_te,      dtype=np.float32)
    y_train      = np.array(y_tr,      dtype=np.int32)
    y_train_orig = np.array(y_tr_orig, dtype=np.int32)
    y_test       = np.array(y_te,      dtype=np.int32)
    train_groups_orig = np.array(train_groups_orig, dtype=np.int32)


    print(f"\n[OK] Dataset ready:")
    print(f"   Train : {len(X_train)} samples "
          f"(orig={len(X_train_orig)}) | Test: {len(X_test)} samples")


    dist_tr = ", ".join(f"{cls}={int((y_train_orig==i).sum())}"
                        for i, cls in enumerate(le.classes_))
    dist_te = ", ".join(f"{cls}={int((y_test==i).sum())}"
                        for i, cls in enumerate(le.classes_))
    print(f"   Train distribution (orig): {dist_tr}")
    print(f"   Test  distribution       : {dist_te}")


    print(f"\n[INFO] Distribution Check:")
    print(f"   Train mean/std: {X_train.mean():.4f} / {X_train.std():.4f}")
    print(f"   Test  mean/std: {X_test.mean():.4f} / {X_test.std():.4f}")


    return (X_train, X_train_orig, X_test,
            y_train, y_train_orig, y_test, train_groups_orig, le, pipeline,
            dataset_info)




# ========================================================================
# 4. OPTIONAL HYPERPARAMETER TUNING
# ========================================================================


def tune_hyperparameters(X_train_orig: np.ndarray,
                         y_train_orig: np.ndarray,
                         train_groups_orig: np.ndarray,
                         cv_folds: int = config.CV_FOLDS,
                         random_seed: int = config.RANDOM_SEED) -> dict:
    """
    GridSearchCV on non-augmented train data.
    Returns best params for SVM and RF.
    """
    print(f"\n{'='*60}")
    cv, actual_folds, splitter_name = _make_group_cv(
        y_train_orig, train_groups_orig, requested_folds=cv_folds,
        random_seed=random_seed
    )
    n_recordings = len(np.unique(train_groups_orig))
    print(f" HYPERPARAMETER TUNING ({splitter_name}, {actual_folds}-fold)")
    print(f" Data: {len(X_train_orig)} non-augmented train windows "
          f"from {n_recordings} recordings")
    print(f"{'='*60}")


    best_params = {}


    svm_grid = {
        'clf__C':     [1, 10, 100],      # ΑΛΛΑΓΗ — clf__ prefix για Pipeline
        'clf__gamma': ['scale', 'auto', 0.01, 0.001],  # ΑΛΛΑΓΗ
    }
    print("\n[TUNE] Tuning SVM...")
    svm_search = GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True))]),  # ΑΛΛΑΓΗ
        svm_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
    )
    svm_search.fit(X_train_orig, y_train_orig, groups=train_groups_orig)
    best_params['SVM (RBF)'] = {k.replace('clf__', ''): v for k, v in svm_search.best_params_.items()}  # ΑΛΛΑΓΗ
    print(f"   Best SVM params : {best_params['SVM (RBF)']}")
    print(f"   Best SVM CV acc : {svm_search.best_score_*100:.2f}%")


    rf_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth':    [10, 15, 20, None],
        'min_samples_leaf': [1, 2, 4],
    }
    print("\n[TUNE] Tuning Random Forest...")
    rf_search = GridSearchCV(
        RandomForestClassifier(class_weight='balanced',
                               n_jobs=-1, random_state=random_seed),
        rf_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
    )
    rf_search.fit(X_train_orig, y_train_orig, groups=train_groups_orig)
    best_params['Random Forest'] = rf_search.best_params_
    print(f"   Best RF params  : {rf_search.best_params_}")
    print(f"   Best RF CV acc  : {rf_search.best_score_*100:.2f}%")
    et_grid = {
        'n_estimators':     [100, 200, 300],
        'max_depth':        [10, 15, 20, None],
        'min_samples_leaf': [1, 2, 4],
    }
    print("\n[SEARCH] Tuning Extra Trees...")
    et_search = GridSearchCV(
        ExtraTreesClassifier(class_weight='balanced', n_jobs=-1, random_state=random_seed),
        et_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
    )
    et_search.fit(X_train_orig, y_train_orig, groups=train_groups_orig)
    best_params['Extra Trees'] = et_search.best_params_
    print(f"   Best ET params  : {et_search.best_params_}")
    print(f"   Best ET CV acc  : {et_search.best_score_*100:.2f}%")


    knn_grid = {
        'clf__n_neighbors': [3, 5, 7, 9],        # ΑΛΛΑΓΗ
        'clf__weights':     ['uniform', 'distance'],  # ΑΛΛΑΓΗ
        'clf__metric':      ['euclidean', 'manhattan'],  # ΑΛΛΑΓΗ
    }
    print("\n[TUNE] Tuning K-NN...")
    knn_search = GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_jobs=-1))]),  # ΑΛΛΑΓΗ
        knn_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
    )
    knn_search.fit(X_train_orig, y_train_orig, groups=train_groups_orig)
    best_params['K-NN'] = {k.replace('clf__', ''): v for k, v in knn_search.best_params_.items()}  # ΑΛΛΑΓΗ
    print(f"   Best K-NN params: {best_params['K-NN']}")
    print(f"   Best K-NN CV acc: {knn_search.best_score_*100:.2f}%")


    lr_grid = {
        'clf__C': [0.1, 1.0, 10.0, 100.0],  # ΑΛΛΑΓΗ
    }
    print("\n[TUNE] Tuning Logistic Regression...")
    lr_search = GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(  # ΑΛΛΑΓΗ
            penalty='l2', solver='lbfgs', max_iter=1000,
            class_weight='balanced', random_state=random_seed))]),
        lr_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
    )
    lr_search.fit(X_train_orig, y_train_orig, groups=train_groups_orig)
    best_params['Logistic Regression'] = {k.replace('clf__', ''): v for k, v in lr_search.best_params_.items()}  # ΑΛΛΑΓΗ
    print(f"   Best LR params  : {best_params['Logistic Regression']}")
    print(f"   Best LR CV acc  : {lr_search.best_score_*100:.2f}%")


    gb_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5],
    }
    print("\n[TUNE] Tuning Gradient Boosting...")
    gb_search = GridSearchCV(
        GradientBoostingClassifier(random_state=random_seed),
        gb_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
    )
    gb_search.fit(X_train_orig, y_train_orig, groups=train_groups_orig)
    best_params['Gradient Boosting'] = gb_search.best_params_
    print(f"   Best GB params  : {gb_search.best_params_}")
    print(f"   Best GB CV acc  : {gb_search.best_score_*100:.2f}%")


    mlp_grid = {
        'clf__hidden_layer_sizes': [(100,), (100, 50), (50, 50)],
        'clf__alpha': [0.0001, 0.001, 0.01],
        'clf__learning_rate': ['constant', 'adaptive'],
    }
    print("\n[TUNE] Tuning MLP (Neural Network)...")
    mlp_search = GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(max_iter=500, random_state=random_seed))]),
        mlp_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
    )
    mlp_search.fit(X_train_orig, y_train_orig, groups=train_groups_orig)
    best_params['MLP'] = {k.replace('clf__', ''): v for k, v in mlp_search.best_params_.items()}
    print(f"   Best MLP params : {mlp_search.best_params_}")
    print(f"   Best MLP CV acc : {mlp_search.best_score_*100:.2f}%")


    return best_params




# ========================================================================
# 5. MODEL TRAINING & EVALUATION
# ========================================================================


def train_and_evaluate(
    X_train:      np.ndarray,
    X_train_orig: np.ndarray,
    X_test:       np.ndarray,
    y_train:      np.ndarray,
    y_train_orig: np.ndarray,
    y_test:       np.ndarray,
    train_groups_orig: np.ndarray,
    le: LabelEncoder,
    cv_folds: int = config.CV_FOLDS,
    best_params: dict = None,
    random_seed: int = config.RANDOM_SEED,
    target_model: str = config.MODELS_TO_TRAIN,
    n_pca: int = None,
) -> dict:
    """
    Train SVM, RF, K-NN, Logistic Regression, Extra Trees, Naive Bayes.
    CV runs on non-augmented X_train_orig.
    Final model trained on full augmented X_train.
    """
    results = {}


    print(f"\n{'='*60}")
    print(f" MODEL TRAINING & EVALUATION")
    print(f" Classes : {list(le.classes_)}")
    print(f" Train   : {len(X_train)} samples "
          f"(orig={len(X_train_orig)}) | Test: {len(X_test)} samples")
    print(f" Features: {X_train.shape[1]}")
    print(f"{'='*60}")


    svm_params = best_params.get('SVM (RBF)', {})          if best_params else {}
    rf_params  = best_params.get('Random Forest', {})      if best_params else {}
    et_params  = best_params.get('Extra Trees', {})        if best_params else {}
    knn_params = best_params.get('K-NN', {})               if best_params else {}
    lr_params  = best_params.get('Logistic Regression', {}) if best_params else {}
    gb_params  = best_params.get('Gradient Boosting', {})  if best_params else {}
    mlp_params = best_params.get('MLP', {})                if best_params else {}


    all_models = {
        'svm': Pipeline([('scaler', StandardScaler()), ('clf', SVC(  # ΑΛΛΑΓΗ
            kernel='rbf',
            C=svm_params.get('C', 10),
            gamma=svm_params.get('gamma', 'scale'),
            class_weight='balanced',
            probability=True,
        ))]),
        'rf': RandomForestClassifier(  # tree-based — no scaler needed
            n_estimators=rf_params.get('n_estimators', 200),
            max_depth=rf_params.get('max_depth', 15),
            min_samples_leaf=rf_params.get('min_samples_leaf', 2),
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_seed,
        ),
        'et': ExtraTreesClassifier(  # tree-based — no scaler needed
            n_estimators=et_params.get('n_estimators', 200),
            max_depth=et_params.get('max_depth', None),
            min_samples_leaf=et_params.get('min_samples_leaf', 1),
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_seed,
        ),
        'knn': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(  # ΑΛΛΑΓΗ
            n_neighbors=knn_params.get('n_neighbors', 5),
            weights=knn_params.get('weights', 'distance'),
            metric=knn_params.get('metric', 'euclidean'),
            n_jobs=-1,
        ))]),
        'lr': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(  # ΑΛΛΑΓΗ
            C=lr_params.get('C', 1.0),
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=random_seed,
        ))]),
        'gb': GradientBoostingClassifier(  # tree-based — no scaler needed
            n_estimators=gb_params.get('n_estimators', 100),
            learning_rate=gb_params.get('learning_rate', 0.1),
            max_depth=gb_params.get('max_depth', 3),
            random_state=random_seed,
        ),
        'mlp': Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(  # ΑΛΛΑΓΗ
            hidden_layer_sizes=mlp_params.get('hidden_layer_sizes', (100,)),
            alpha=mlp_params.get('alpha', 0.0001),
            max_iter=500,
            random_state=random_seed,
        ))]),
        'nb': GaussianNB(),  # Gaussian — no scaler needed
    }


    # Map full names if needed or filter by ID
    if target_model.lower() == 'all':
        models = all_models
    elif target_model.lower() in all_models:
        models = {target_model.lower(): all_models[target_model.lower()]}
    else:
        print(f"[WARNING] Warning: Model '{target_model}' not recognized. Training all.")
        models = all_models


    cv, actual_folds, splitter_name = _make_group_cv(
        y_train_orig, train_groups_orig, requested_folds=cv_folds,
        random_seed=random_seed
    )
    
    if n_pca is None:
        n_pca = int(X_train.shape[1] // N_STATS)


    for name, model in models.items():
        print(f"\n{'-'*50}")
        print(f"  {name}")
        print(f"{'-'*50}")


        # CV NOTE: cross_val_score runs on X_train_orig (non-augmented, N windows).
        # The final model is trained on X_train (augmented, ~N_augments×N windows).
        # CV accuracy is therefore a conservative lower-bound, not an estimate of
        # the deployed model's training regime.  Use hold-out test accuracy as the
        # primary performance measure.
        cv_scores = cross_val_score(
            model, X_train_orig, y_train_orig,
            cv=cv, scoring='accuracy', n_jobs=-1,
            groups=train_groups_orig
        )
        print(f"  {actual_folds}-Fold {splitter_name} CV "
              f"{cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")


        model.fit(X_train, y_train)
        y_pred     = model.predict(X_test)
        acc        = accuracy_score(y_test, y_pred)
        f1_mac     = f1_score(y_test, y_pred, average='macro')
        # Measured on non-augmented original training windows (not the augmented
        # X_train the model was fit on).  This avoids a near-100% trivial value
        # while remaining comparable to the CV and test distributions.
        train_acc  = accuracy_score(y_train_orig, model.predict(X_train_orig))


        print(f"  Train Accuracy (non-aug): {train_acc*100:.2f}%")
        print(f"  Hold-out Test Accuracy  : {acc*100:.2f}%")
        print(f"  Hold-out F1 (macro)     : {f1_mac*100:.2f}%")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=le.classes_, digits=3))


        cm = confusion_matrix(y_test, y_pred)
        print(f"  Confusion Matrix:")
        print("          " + "  ".join(f"{c:>8}" for c in le.classes_))
        for i, row in enumerate(cm):
            print(f"  {le.classes_[i]:>8}  " +
                  "  ".join(f"{v:>8}" for v in row))


        print(f"\n  Per-class Accuracy:")
        for i, cls in enumerate(le.classes_):
            mask    = y_test == i
            cls_acc = accuracy_score(y_test[mask], y_pred[mask]) if mask.sum() > 0 else 0.0
            print(f"    {cls:>10}: {cls_acc*100:.1f}%  ({mask.sum()} test samples)")


        results[name] = {
            'model':          model,
            'cv_mean':        cv_scores.mean(),
            'cv_std':         cv_scores.std(),
            'cv_scores':      cv_scores.tolist(),
            'train_accuracy': train_acc,
            'test_accuracy':  acc,
            'test_f1_macro':  f1_mac,
            'confusion_matrix': cm,
            'y_pred':         y_pred,
            'y_test':         y_test,
            'feature_importances': [],
            'cv_splitter': splitter_name,
        }


        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_names  = _get_feature_names(n_pca)
            top_idx     = np.argsort(importances)[::-1][:10]
            

            top_features = []
            print(f"\n  Top 10 Important Features:")
            for rank, idx in enumerate(top_idx):
                fname = feat_names[idx] if idx < len(feat_names) else f"feat_{idx}"
                importance_val = float(importances[idx])
                top_features.append({"name": fname, "importance": importance_val})
                print(f"    {rank+1:2}. {fname:30s}  {importance_val*100:.2f}%")
            

            results[name]['feature_importances'] = top_features


    print(f"\n{'='*60}")
    print(f" SUMMARY")
    print(f"{'='*60}")
    for name, res in results.items():
        print(f"  {name:20s}  "
              f"CV={res['cv_mean']*100:.1f}% +/-{res['cv_std']*100:.1f}%  "
              f"Test={res['test_accuracy']*100:.1f}%  "
              f"F1={res['test_f1_macro']*100:.1f}%")


    best = max(results.items(), key=lambda x: x[1]['cv_mean'])
    print(f"\n  [OK] Best: {best[0]} (CV {best[1]['cv_mean']*100:.1f}%)")


    return results




# ========================================================================
# 6. SAVE MODELS
# ========================================================================


def save_models(results: dict,
                pipeline,
                le: LabelEncoder,
                output_dir: str = config.MODELS_DIR,
                experiment_config: dict | None = None) -> None:
    """
    Save everything needed for inference:
      csi_pipeline.joblib    - preprocess new recordings
      label_encoder.joblib   - int -> class name
      svm.joblib, rf.joblib  - trained models
      metrics.json           - for thesis tables
      experiment_config.json - exact split / preprocessing metadata
    """
    import joblib
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)


    if pipeline is not None:
        joblib.dump(pipeline, out / "csi_pipeline.joblib")
        print(f"[SAVE] {out / 'csi_pipeline.joblib'}")


    joblib.dump(le, out / "label_encoder.joblib")
    print(f"[SAVE] {out / 'label_encoder.joblib'}")


    metrics = {}
    for name, res in results.items():
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        path = out / f"{safe}.joblib"
        joblib.dump(res['model'], path)
        print(f"[SAVE] {path}  (test={res['test_accuracy']*100:.1f}%)")


        metrics[name] = {
            'cv_accuracy_mean': round(res['cv_mean'], 4),
            'cv_accuracy_std':  round(res['cv_std'],  4),
            'cv_scores':        [round(s, 4) for s in res.get('cv_scores', [])],
            'train_accuracy':   round(res.get('train_accuracy', 0.0), 4),
            'test_accuracy':    round(res['test_accuracy'], 4),
            'test_f1_macro':    round(res['test_f1_macro'],  4),
            'confusion_matrix': res['confusion_matrix'].tolist(),
            'classes':          list(le.classes_),
            'feature_importances': res.get('feature_importances', []),
            'cv_splitter': res.get('cv_splitter', 'GroupKFold'),
            'feature_vector_version': FEATURE_VECTOR_VERSION,
        }


    json_path = out / "metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {json_path}  (metrics for thesis)")

    if experiment_config is not None:
        config_path = out / "experiment_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_config, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] {config_path}  (reproducible experiment settings)")


    best = max(results.items(), key=lambda x: x[1]['cv_mean'])[0]
    safe_best = best.replace(" ", "_").replace("(", "").replace(")", "")
    print(f"\n   Load for inference:")
    print(f"     import joblib")
    print(f"     pipeline = joblib.load('{out}/csi_pipeline.joblib')")
    print(f"     le       = joblib.load('{out}/label_encoder.joblib')")
    print(f"     model    = joblib.load('{out}/{safe_best}.joblib')")




# ========================================================================
# MAIN
# ========================================================================


def main():
    defaults = config.get_script_defaults("csi_ml_pipeline")
    parser = argparse.ArgumentParser(description="CSI HAR - ML Pipeline")
    parser.add_argument("--data_dir",    type=str,   default=defaults["data_dir"])
    parser.add_argument("--classes",     nargs="+",  default=defaults["classes"])
    parser.add_argument("--window_size", type=int,   default=defaults["window_size"])
    parser.add_argument("--step",        type=int,   default=defaults["step"])
    parser.add_argument("--fs",          type=float, default=defaults["fs"])
    parser.add_argument(
        "--augment",
        nargs="+",
        metavar="TECHNIQUE",
        default=defaults["augment"],
        help=(
            "Augmentation techniques to apply on RAW windows (BEFORE PCA). "
            f"Choices: {ALL_AUGMENT_TECHNIQUES}. "
            "Default: all 4 techniques. "
            "Use '--augment noise scale' for a subset. "
            "To disable completely use --no-augment."
        )
    )
    config.add_bool_argument(
        parser,
        dest="use_augment",
        default=defaults["use_augment"],
        help="Enable RAW-window data augmentation before PCA.",
        positive_flags=["--use-augment"],
        negative_flags=["--no-augment", "--no_augment"],
    )
    parser.add_argument("--n_augments",  type=int,   default=defaults["n_augments"])
    parser.add_argument("--pca",         type=int,   default=defaults["pca"])
    parser.add_argument("--test_ratio",  type=float, default=defaults["test_ratio"])
    config.add_bool_argument(
        parser,
        dest="use_diff",
        default=defaults["use_diff"],
        help="Enable temporal differencing in preprocessing.",
        positive_flags=["--diff"],
        negative_flags=["--no-diff", "--no_diff"],
    )
    config.add_bool_argument(
        parser,
        dest="simulate",
        default=defaults["simulate"],
        help="Use synthetic data instead of real CSI recordings.",
        positive_flags=["--simulate"],
        negative_flags=["--no-simulate"],
    )
    config.add_bool_argument(
        parser,
        dest="save_model",
        default=defaults["save_model"],
        help="Save the fitted pipeline, label encoder, models, and metrics.",
        positive_flags=["--save_model"],
        negative_flags=["--no-save_model", "--no_save_model"],
    )
    config.add_bool_argument(
        parser,
        dest="tune",
        default=defaults["tune"],
        help="Run GridSearchCV hyperparameter tuning.",
        positive_flags=["--tune"],
        negative_flags=["--no-tune"],
    )
    parser.add_argument("--model", type=str, default=defaults["model"],
                        choices=config.MODEL_CHOICES,
                        help="Specific model to train, or 'all'")
    parser.add_argument("--seed",        type=int,   default=defaults["seed"])
    parser.add_argument("--cv_folds",    type=int,   default=defaults["cv_folds"],
                        help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--cutoff",      type=float, default=defaults["cutoff"],
                        help="Butterworth filter cutoff frequency in Hz (default: 10)")
    parser.add_argument("--models_dir",  type=str,   default=defaults["models_dir"],
                        help="Directory to save/load model files (default: ./models)")
    args = parser.parse_args()


    # Validation: Step vs Window size
    if args.step > args.window_size:
        print(f"\n[WARNING]  WARNING: step ({args.step}) > window_size ({args.window_size}).")
        print("   This means some CSI frames will be skipped and not covered by any window.\n")


    # --no_augment disables everything; otherwise use the specified (or default) list
    if not args.use_augment:
        augment_techniques = []
    else:
        augment_techniques = args.augment  # list of 1+ techniques
    # Validate
    unknown = set(augment_techniques) - set(ALL_AUGMENT_TECHNIQUES)
    if unknown:
        parser.error(f"Unknown augmentation technique(s): {unknown}. "
                     f"Valid: {ALL_AUGMENT_TECHNIQUES}")


    print("=" * 60)
    print(" CSI HAR - ML Pipeline")
    print(f" Classes : requested={args.classes}")
    print(f" Data dir: {args.data_dir}")
    print(f" Window  : {args.window_size} frames @ {args.fs} Hz = "
          f"{args.window_size/args.fs:.2f}s")
    aug_label = ', '.join(augment_techniques) if augment_techniques else 'DISABLED'
    print(f" Augment : [{aug_label}] (x{args.n_augments}) | "
          f"PCA: {args.pca} | Diff: {args.use_diff}")
    print(f" Tune    : {args.tune} | Seed: {args.seed}")
    print("=" * 60)


    (X_train, X_train_orig, X_test,
     y_train, y_train_orig, y_test,
     train_groups_orig, le, pipeline, dataset_info) = build_dataset(
        data_dir=args.data_dir,
        classes=args.classes,
        pipeline_kwargs={'fs': args.fs, 'use_diff': args.use_diff},
        window_size=args.window_size,
        step=args.step,
        augment_techniques=augment_techniques,
        n_augments=args.n_augments,
        simulation_mode=args.simulate or (CSIPipeline is None),
        test_recording_ratio=args.test_ratio,
        random_seed=args.seed,
        n_pca=args.pca,
        cutoff=args.cutoff,
    )


    if X_train.shape[0] == 0:
        print("[ERROR] No samples - check data_dir and classes")
        sys.exit(1)

    print(f" Effective classes: {dataset_info.get('classes', [])}")


    print(f"\n{'-'*60}\n Step 3: Model Training\n{'-'*60}")


    best_params = None
    if args.tune:
        best_params = tune_hyperparameters(
            X_train_orig, y_train_orig, train_groups_orig,
            random_seed=args.seed, cv_folds=args.cv_folds
        )


    results = train_and_evaluate(
        X_train, X_train_orig, X_test,
        y_train, y_train_orig, y_test,
        train_groups_orig, le, best_params=best_params,
        random_seed=args.seed,
        target_model=args.model,
        cv_folds=args.cv_folds,
        n_pca=dataset_info['n_pca'],
    )


    if args.save_model:
        experiment_config = dict(dataset_info)
        experiment_config.update({
            'augment_techniques': list(augment_techniques),
            'n_augments': int(args.n_augments),
            'cv_folds': int(args.cv_folds),
            'target_model': args.model,
        })
        save_models(
            results,
            pipeline,
            le,
            output_dir=args.models_dir,
            experiment_config=experiment_config,
        )


if __name__ == "__main__":
    main()
