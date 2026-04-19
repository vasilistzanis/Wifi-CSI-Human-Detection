#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSI HAR — Complete ML Pipeline
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
  python csi_ml_pipeline.py --classes walk idle
  python csi_ml_pipeline.py --classes walk sit fall idle --save_model --tune
"""

import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

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
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

# 14 classical stats + 6 DWT (energy+std × 3 levels with db4 wavelet)
# DWT_LEVELS = 3  →  detail coeff d1,d2,d3  (approx a3 excluded — slow drift)
_DWT_STATS_PER_COMPONENT = 6          # energy_d1, std_d1, ... energy_d3, std_d3
N_STATS = 14 + _DWT_STATS_PER_COMPONENT   # = 20

try:
    import pywt as _pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _pywt = None
    _PYWT_AVAILABLE = False
    import warnings as _warnings
    _warnings.warn(
        "PyWavelets (pywt) not installed — DWT features will be zero-padded. "
        "Run: pip install PyWavelets",
        RuntimeWarning, stacklevel=1
    )

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

# ════════════════════════════════════════════════════════════════════════
# IMPORT PREPROCESSING PIPELINE
# ════════════════════════════════════════════════════════════════════════

try:
    from data_preprocessing import CSIPipeline, load_csi_csv
    print("✅ CSIPipeline imported successfully")
except ImportError:
    print("⚠️  data_preprocessing.py not found — using simulation mode")
    CSIPipeline = None
    load_csi_csv = None


# ════════════════════════════════════════════════════════════════════════
# 1. DATA AUGMENTATION  (applied on RAW amplitude windows, BEFORE PCA)
# ════════════════════════════════════════════════════════════════════════

ALL_AUGMENT_TECHNIQUES = ['noise', 'shift', 'scale', 'time_warp']


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
    if class_label == 'walk':
        factor = rng.uniform(0.9, 1.1)
    elif class_label == 'sit':
        factor = rng.uniform(0.95, 1.05)
    else:  # idle/static
        factor = rng.uniform(0.98, 1.02)
        
    src_indices = np.linspace(0, T - 1, T) / factor
    
    # Reflect Padding logic (no flat tails)
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
                   n_augments: int = 4,
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

    # If class-aware filtering left nothing, fall back to noise (safest technique)
    # to preserve dataset size consistency — do NOT return duplicates silently
    if not safe_techs:
        import warnings
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

# ════════════════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════════

def _dwt_features_for_col(col: np.ndarray, wavelet: str = 'db4',
                          level: int = 3) -> list:
    """
    Compute DWT features for a single 1-D column (one PCA component).

    Decomposition: db4 wavelet, 3 levels.
    Features extracted from DETAIL coefficients d1, d2, d3 only
    (approximation a3 is excluded — it carries slow DC drift, already
    captured by 'mean' in the classical stats block).

    For window=50 frames @ 100 Hz the frequency bands are:
      d1 : 25–50 Hz  (high-freq noise)
      d2 : 12.5–25 Hz (rapid motion transients)
      d3 :  6.25–12.5 Hz (fall / fast gestures)
      a3 :  0–6.25 Hz  (walking ~2 Hz, idle ~0 Hz)  ← excluded

    Returns 6 floats: [energy_d1, std_d1, energy_d2, std_d2, energy_d3, std_d3]
    """
    if not _PYWT_AVAILABLE:
        return [0.0] * _DWT_STATS_PER_COMPONENT

    # Clamp level to what the signal length supports
    max_level = _pywt.dwt_max_level(len(col), wavelet)
    actual_level = min(level, max_level)

    coeffs = _pywt.wavedec(col, wavelet, level=actual_level)
    # coeffs = [a_n, d_n, d_{n-1}, ..., d1]  (pywt order)
    # We want detail coefficients d1..d3 (indices [-1], [-2], [-3])
    detail_coeffs = coeffs[1:]   # drop approximation a_n

    feats = []
    for lvl in range(1, level + 1):
        # detail coeffs are stored in reversed order: coeffs[-lvl]
        if lvl <= len(detail_coeffs):
            d = detail_coeffs[-lvl].astype(np.float64)
            energy = float(np.sum(d ** 2))
            std    = float(np.std(d))
        else:
            energy, std = 0.0, 0.0
        feats.extend([energy, std])

    return feats


def extract_features_from_window(window: np.ndarray) -> np.ndarray:
    """
    20 features per PCA component → flat feature vector.

    Input:  (window_size, n_pca_components)  e.g. (50, 10)
    Output: (200,)  [20 features × 10 components]

    Feature breakdown (20 per component):
      [0–13]  Classical stats (14):
              mean, std, max, min, range, median, energy,
              skewness, kurtosis, fft_mean, fft_std, zcr,
              fft_peak_idx, spectral_entropy
      [14–19] DWT features (6) — db4 wavelet, 3 detail levels:
              energy_d1, std_d1,   (d1: 25–50 Hz)
              energy_d2, std_d2,   (d2: 12.5–25 Hz)
              energy_d3, std_d3    (d3: 6.25–12.5 Hz)
    """
    feats = []
    for c in range(window.shape[1]):
        col      = window[:, c].astype(np.float64)
        mean_val = col.mean()
        std_val  = col.std() + 1e-8

        # ── FFT features ─────────────────────────────────────────────────
        fft_vals = np.abs(np.fft.rfft(col))
        fft_mean = float(fft_vals.mean())
        fft_std  = float(fft_vals.std())

        # ── ZCR (Zero-Crossing Rate) ──────────────────────────────────────
        centered = col - mean_val
        zcr = float(np.sum(np.diff(np.sign(centered)) != 0) / max(1, len(col) - 1))

        # ── Dominant Frequency index ──────────────────────────────────────
        fft_peak_idx = float(np.argmax(fft_vals))

        # ── Spectral Entropy ──────────────────────────────────────────────
        prob = fft_vals / (np.sum(fft_vals) + 1e-8)
        spectral_entropy = float(-np.sum(prob * np.log2(prob + 1e-8)))

        # ── Classical 14 stats ────────────────────────────────────────────
        feats.extend([
            mean_val,
            std_val,
            col.max(),
            col.min(),
            col.max() - col.min(),
            float(np.median(col)),
            float(np.sum(col ** 2)),
            float(np.mean(((col - mean_val) / std_val) ** 3)),
            float(np.mean(((col - mean_val) / std_val) ** 4)),
            fft_mean,
            fft_std,
            zcr,
            fft_peak_idx,
            spectral_entropy,
        ])

        # ── DWT 6 stats (db4, 3 detail levels) ───────────────────────────
        feats.extend(_dwt_features_for_col(col, wavelet='db4', level=3))

    return np.array(feats, dtype=np.float32)


def _get_feature_names(n_pca_components: int) -> list[str]:
    classical = ['mean', 'std', 'max', 'min', 'range', 'median',
                 'energy', 'skewness', 'kurtosis', 'fft_mean', 'fft_std',
                 'zcr', 'fft_peak_idx', 'spectral_entropy']
    dwt = ['dwt_d1_energy', 'dwt_d1_std',
           'dwt_d2_energy', 'dwt_d2_std',
           'dwt_d3_energy', 'dwt_d3_std']
    all_stats = classical + dwt   # 20 total
    return [f"PC{c+1}_{s}" for c in range(n_pca_components) for s in all_stats]


def extract_windows(data: np.ndarray,
                    window_size: int = 50,
                    step: int = 25) -> list[np.ndarray]:
    """Sliding window → list of (window_size, n_components) arrays."""
    if data.shape[0] < window_size:
        return []
    return [data[s:s + window_size]
            for s in range(0, data.shape[0] - window_size + 1, step)]


def _make_group_cv(y: np.ndarray,
                   groups: np.ndarray,
                   requested_folds: int = 5,
                   random_seed: int = 42) -> tuple:
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
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_seed
        )
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


# ════════════════════════════════════════════════════════════════════════
# 3. DATASET BUILDER
# ════════════════════════════════════════════════════════════════════════
# Loads recordings, handles train/test splitting at the file-level, 
# applies augmentation on training data only to prevent leakage,
# and returns the separated features.
# ════════════════════════════════════════════════════════════════════════

def build_dataset(
    data_dir: str | Path,
    classes: list[str],
    pipeline_kwargs: dict = None,
    window_size: int = 50,
    step: int = 25,
    augment_techniques: list = None,
    n_augments: int = 4,
    simulation_mode: bool = False,
    test_recording_ratio: float = 0.2,
    random_seed: int = 42,
    n_pca: int = 10,
) -> tuple:
    """
    Load recordings, preprocess, extract features.
    Augmentation is applied on RAW amplitude windows BEFORE PCA projection
    (physics-aware: noise, shift, scale, time_warp with strict limits).
    Returns train/test split at recording level (no leakage).

    Args:
      augment_techniques : list of technique names to use, e.g. ['noise', 'scale'].
                           None or empty list → no augmentation.
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
    """
    if pipeline_kwargs is None:
        pipeline_kwargs = {'fs': 100.0, 'use_diff': True}

    do_augment = bool(augment_techniques)  # empty list / None → no augmentation
    if do_augment:
        print(f"   Augmentation techniques: {augment_techniques}")

    data_dir = Path(data_dir)
    le = LabelEncoder()
    le.fit(classes)

    # ── Simulation Mode ──────────────────────────────────────────────────
    if simulation_mode or CSIPipeline is None:
        print("\n🔬 SIMULATION MODE")
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
                pp.fit_transform(np.vstack(train_cms), use_pca=True, n_components=n_pca, scaler_type='standard')

        for cm, label_idx, is_test in rec_data:
            # ── Get RAW amplitude (pre-PCA) for augmentation ──────────────
            if pp:
                # Replicate pipeline steps up to (but not including) PCA+scaler
                amp = pp.remove_null_subcarriers(cm, fit=False)
                amp = pp.apply_hampel_filter(amp)
                amp = pp.apply_lowpass_filter(amp)
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
                    X_te.append(extract_features_from_window(w_proj))
                    y_te.append(label_idx)
                else:
                    # Train original (no augmentation)
                    if pp:
                        w_proj = pp.pca.transform(w_raw)
                        w_proj = pp.scaler.transform(w_proj)
                    else:
                        w_proj = w_raw[:, :n_pca]
                    feat_orig = extract_features_from_window(w_proj)
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
                            if pp:
                                aw_proj = pp.pca.transform(aw_raw)
                                aw_proj = pp.scaler.transform(aw_proj)
                            else:
                                aw_proj = aw_raw[:, :n_pca]
                            X_tr.append(extract_features_from_window(aw_proj))
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

        print(f"\n✅ Train={len(X_train)} (orig={len(X_train_orig)}) "
              f"| Test={len(X_test)} samples")
        return (X_train, X_train_orig, X_test,
                y_train, y_train_orig, y_test, train_groups_orig, le, None)

    # ── Real Data Mode ───────────────────────────────────────────────────
    print(f"\n📂 Loading data from: {data_dir}")

    train_files_all = {}
    test_files_all  = {}

    for cls in classes:
        files = (sorted((data_dir/cls).glob("*.csv")) +
                 sorted((data_dir/cls).glob("*.txt")))
        if not files:
            print(f"⚠️  No files found for class '{cls}'")
            train_files_all[cls] = []
            test_files_all[cls]  = []
            continue
        # Shuffle: recordings are independent sessions, not time-dependent.
        # Seeded shuffle ensures reproducibility while removing ordering bias.
        random.Random(random_seed).shuffle(files)
        n_test = max(1, int(len(files) * test_recording_ratio))
        train_files_all[cls] = files[:-n_test]
        test_files_all[cls]  = files[-n_test:]

    fit_matrices = []
    for cls in classes:
        for fpath in train_files_all.get(cls, []):
            cm, _ = load_csi_csv(fpath)
            if cm.size > 0:
                fit_matrices.append(cm)

    if not fit_matrices:
        raise ValueError("No valid training CSI data found.")

    print("\n🔧 Fitting CSIPipeline on TRAIN recordings only...")
    pipeline = CSIPipeline(**pipeline_kwargs)
    pipeline.fit_transform(np.vstack(fit_matrices),
                           use_pca=True, n_components=n_pca,
                           scaler_type='standard')

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
        for fpath in train_files:
            cm, _ = load_csi_csv(fpath)
            if cm.size == 0:
                continue
            try:
                # ── Pre-PCA steps (for augmentation on raw signal) ────────
                amp = pipeline.remove_null_subcarriers(cm, fit=False)
                amp = pipeline.apply_hampel_filter(amp)
                amp = pipeline.apply_lowpass_filter(amp)
                if pipeline.use_diff:
                    amp = pipeline.apply_temporal_diff(amp)
                raw_pre_pca = amp  # (N_frames, n_active_subcarriers)
            except ValueError as e:
                print(f"   ⚠️  {fpath.name}: {e} — skipped")
                continue

            for w_raw in extract_windows(raw_pre_pca, window_size, step):
                # Project original window through PCA+scaler
                w_proj = pipeline.pca.transform(w_raw)
                w_proj = pipeline.scaler.transform(w_proj)
                feat_orig = extract_features_from_window(w_proj)

                X_tr_orig.append(feat_orig)
                y_tr_orig.append(label_idx)
                train_groups_orig.append(recording_group_id)
                X_tr.append(feat_orig)
                y_tr.append(label_idx)
                tr_wins += 1

                # Augmentation: on RAW window → then project → features
                if do_augment:
                    for aw_raw in augment_window(
                            w_raw, n_augments,
                            techniques=augment_techniques,
                            seed=random_seed + global_window_idx,
                            class_label=cls):
                        aw_proj = pipeline.pca.transform(aw_raw)
                        aw_proj = pipeline.scaler.transform(aw_proj)
                        X_tr.append(extract_features_from_window(aw_proj))
                        y_tr.append(label_idx)
                global_window_idx += 1
            recording_group_id += 1

        te_wins = 0
        for fpath in test_files:
            cm, _ = load_csi_csv(fpath)
            if cm.size == 0:
                continue
            try:
                processed = pipeline.transform(cm, use_pca=True)
            except ValueError as e:
                print(f"   ⚠️  {fpath.name}: {e} — skipped")
                continue
            # Test files: use full pipeline.transform (no augmentation)
            for w in extract_windows(processed, window_size, step):
                X_te.append(extract_features_from_window(w))
                y_te.append(label_idx)
                te_wins += 1

        aug_count = tr_wins * n_augments if do_augment else 0
        print(f"   → train: {tr_wins} orig + {aug_count} augmented | "
              f"test: {te_wins} windows")

    if not X_tr:
        raise ValueError("No training features extracted.")
    if not X_te:
        raise ValueError("No test features extracted.")

    X_train      = np.array(X_tr,      dtype=np.float32)
    X_train_orig = np.array(X_tr_orig, dtype=np.float32)
    X_test       = np.array(X_te,      dtype=np.float32)
    y_train      = np.array(y_tr,      dtype=np.int32)
    y_train_orig = np.array(y_tr_orig, dtype=np.int32)
    y_test       = np.array(y_te,      dtype=np.int32)
    train_groups_orig = np.array(train_groups_orig, dtype=np.int32)

    print(f"\n✅ Dataset ready:")
    print(f"   Train : {len(X_train)} samples "
          f"(orig={len(X_train_orig)}) | Test: {len(X_test)} samples")

    dist_tr = ", ".join(f"{cls}={int((y_train_orig==i).sum())}"
                        for i, cls in enumerate(le.classes_))
    dist_te = ", ".join(f"{cls}={int((y_test==i).sum())}"
                        for i, cls in enumerate(le.classes_))
    print(f"   Train distribution (orig): {dist_tr}")
    print(f"   Test  distribution       : {dist_te}")

    print(f"\n📊 Distribution Check:")
    print(f"   Train mean/std: {X_train.mean():.4f} / {X_train.std():.4f}")
    print(f"   Test  mean/std: {X_test.mean():.4f} / {X_test.std():.4f}")

    return (X_train, X_train_orig, X_test,
            y_train, y_train_orig, y_test, train_groups_orig, le, pipeline)


# ════════════════════════════════════════════════════════════════════════
# 4. OPTIONAL HYPERPARAMETER TUNING
# ════════════════════════════════════════════════════════════════════════

def tune_hyperparameters(X_train_orig: np.ndarray,
                         y_train_orig: np.ndarray,
                         train_groups_orig: np.ndarray,
                         cv_folds: int = 5,
                         random_seed: int = 42) -> dict:
    """
    GridSearchCV on non-augmented train data.
    Returns best params for SVM and RF.
    """
    print(f"\n{'═'*60}")
    cv, actual_folds, splitter_name = _make_group_cv(
        y_train_orig, train_groups_orig, requested_folds=cv_folds,
        random_seed=random_seed
    )
    n_recordings = len(np.unique(train_groups_orig))
    print(f" HYPERPARAMETER TUNING ({splitter_name}, {actual_folds}-fold)")
    print(f" Data: {len(X_train_orig)} non-augmented train windows "
          f"from {n_recordings} recordings")
    print(f"{'═'*60}")

    best_params = {}

    svm_grid = {
        'C':     [1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.001],
    }
    print("\n🔍 Tuning SVM...")
    svm_search = GridSearchCV(
        SVC(kernel='rbf', class_weight='balanced', probability=True),
        svm_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
    )
    svm_search.fit(X_train_orig, y_train_orig, groups=train_groups_orig)
    best_params['SVM (RBF)'] = svm_search.best_params_
    print(f"   Best SVM params : {svm_search.best_params_}")
    print(f"   Best SVM CV acc : {svm_search.best_score_*100:.2f}%")

    rf_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth':    [10, 15, 20, None],
        'min_samples_leaf': [1, 2, 4],
    }
    print("\n🔍 Tuning Random Forest...")
    rf_search = GridSearchCV(
        RandomForestClassifier(class_weight='balanced',
                               n_jobs=-1, random_state=42),
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
    print("\n🔍 Tuning Extra Trees...")
    et_search = GridSearchCV(
        ExtraTreesClassifier(class_weight='balanced', n_jobs=-1, random_state=42),
        et_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
    )
    et_search.fit(X_train_orig, y_train_orig, groups=train_groups_orig)
    best_params['Extra Trees'] = et_search.best_params_
    print(f"   Best ET params  : {et_search.best_params_}")
    print(f"   Best ET CV acc  : {et_search.best_score_*100:.2f}%")

    knn_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights':     ['uniform', 'distance'],
        'metric':      ['euclidean', 'manhattan'],
    }
    print("\n🔍 Tuning K-NN...")
    knn_search = GridSearchCV(
        KNeighborsClassifier(n_jobs=-1),
        knn_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
    )
    knn_search.fit(X_train_orig, y_train_orig, groups=train_groups_orig)
    best_params['K-NN'] = knn_search.best_params_
    print(f"   Best K-NN params: {knn_search.best_params_}")
    print(f"   Best K-NN CV acc: {knn_search.best_score_*100:.2f}%")

    lr_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],
    }
    print("\n🔍 Tuning Logistic Regression...")
    lr_search = GridSearchCV(
        LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000,
                           class_weight='balanced', random_state=42),
        lr_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
    )
    lr_search.fit(X_train_orig, y_train_orig, groups=train_groups_orig)
    best_params['Logistic Regression'] = lr_search.best_params_
    print(f"   Best LR params  : {lr_search.best_params_}")
    print(f"   Best LR CV acc  : {lr_search.best_score_*100:.2f}%")

    gb_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5],
    }
    print("\n🔍 Tuning Gradient Boosting...")
    gb_search = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
    )
    gb_search.fit(X_train_orig, y_train_orig, groups=train_groups_orig)
    best_params['Gradient Boosting'] = gb_search.best_params_
    print(f"   Best GB params  : {gb_search.best_params_}")
    print(f"   Best GB CV acc  : {gb_search.best_score_*100:.2f}%")

    mlp_grid = {
        'hidden_layer_sizes': [(100,), (100, 50), (50, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
    }
    print("\n🔍 Tuning MLP (Neural Network)...")
    mlp_search = GridSearchCV(
        MLPClassifier(max_iter=500, random_state=42),
        mlp_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
    )
    mlp_search.fit(X_train_orig, y_train_orig, groups=train_groups_orig)
    best_params['MLP'] = mlp_search.best_params_
    print(f"   Best MLP params : {mlp_search.best_params_}")
    print(f"   Best MLP CV acc : {mlp_search.best_score_*100:.2f}%")

    return best_params


# ════════════════════════════════════════════════════════════════════════
# 5. MODEL TRAINING & EVALUATION
# ════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    X_train:      np.ndarray,
    X_train_orig: np.ndarray,
    X_test:       np.ndarray,
    y_train:      np.ndarray,
    y_train_orig: np.ndarray,
    y_test:       np.ndarray,
    train_groups_orig: np.ndarray,
    le: LabelEncoder,
    cv_folds: int = 5,
    best_params: dict = None,
    random_seed: int = 42,
) -> dict:
    """
    Train SVM, RF, K-NN, Logistic Regression, Extra Trees, Naive Bayes.
    CV runs on non-augmented X_train_orig.
    Final model trained on full augmented X_train.
    """
    results = {}

    print(f"\n{'═'*60}")
    print(f" MODEL TRAINING & EVALUATION")
    print(f" Classes : {list(le.classes_)}")
    print(f" Train   : {len(X_train)} samples "
          f"(orig={len(X_train_orig)}) | Test: {len(X_test)} samples")
    print(f" Features: {X_train.shape[1]}")
    print(f"{'═'*60}")

    svm_params = best_params.get('SVM (RBF)', {})          if best_params else {}
    rf_params  = best_params.get('Random Forest', {})      if best_params else {}
    et_params  = best_params.get('Extra Trees', {})        if best_params else {}
    knn_params = best_params.get('K-NN', {})               if best_params else {}
    lr_params  = best_params.get('Logistic Regression', {}) if best_params else {}
    gb_params  = best_params.get('Gradient Boosting', {})  if best_params else {}
    mlp_params = best_params.get('MLP', {})                if best_params else {}

    models = {
        'SVM (RBF)': SVC(
            kernel='rbf',
            C=svm_params.get('C', 10),
            gamma=svm_params.get('gamma', 'scale'),
            class_weight='balanced',
            probability=True,
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=rf_params.get('n_estimators', 200),
            max_depth=rf_params.get('max_depth', 15),
            min_samples_leaf=rf_params.get('min_samples_leaf', 2),
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=et_params.get('n_estimators', 200),
            max_depth=et_params.get('max_depth', None),
            min_samples_leaf=et_params.get('min_samples_leaf', 1),
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        ),
        'K-NN (k=5)': KNeighborsClassifier(
            n_neighbors=knn_params.get('n_neighbors', 5),
            weights=knn_params.get('weights', 'distance'),
            metric=knn_params.get('metric', 'euclidean'),
            n_jobs=-1,
        ),
        'Logistic Regression': LogisticRegression(
            C=lr_params.get('C', 1.0),
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=gb_params.get('n_estimators', 100),
            learning_rate=gb_params.get('learning_rate', 0.1),
            max_depth=gb_params.get('max_depth', 3),
            random_state=42,
        ),
        'MLP (Neural Network)': MLPClassifier(
            hidden_layer_sizes=mlp_params.get('hidden_layer_sizes', (100,)),
            alpha=mlp_params.get('alpha', 0.0001),
            learning_rate=mlp_params.get('learning_rate', 'constant'),
            max_iter=500,
            random_state=42,
        ),
        'Naive Bayes': GaussianNB(),
    }

    cv, actual_folds, splitter_name = _make_group_cv(
        y_train_orig, train_groups_orig, requested_folds=cv_folds,
        random_seed=random_seed
    )
    n_pca = X_train.shape[1] // N_STATS

    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")

        cv_scores = cross_val_score(
            model, X_train_orig, y_train_orig,
            cv=cv, scoring='accuracy', n_jobs=-1,
            groups=train_groups_orig
        )
        print(f"  {actual_folds}-Fold {splitter_name} CV "
              f"{cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average='macro')

        print(f"  Hold-out Test Accuracy : {acc*100:.2f}%")
        print(f"  Hold-out F1 (macro)    : {f1_mac*100:.2f}%")
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
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': acc,
            'test_f1_macro': f1_mac,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_test': y_test,
            'feature_importances': []
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

    print(f"\n{'═'*60}")
    print(f" SUMMARY")
    print(f"{'═'*60}")
    for name, res in results.items():
        print(f"  {name:20s}  "
              f"CV={res['cv_mean']*100:.1f}% ±{res['cv_std']*100:.1f}%  "
              f"Test={res['test_accuracy']*100:.1f}%  "
              f"F1={res['test_f1_macro']*100:.1f}%")

    best = max(results.items(), key=lambda x: x[1]['cv_mean'])
    print(f"\n  🏆 Best: {best[0]} (CV {best[1]['cv_mean']*100:.1f}%)")

    return results


# ════════════════════════════════════════════════════════════════════════
# 6. SAVE MODELS
# ════════════════════════════════════════════════════════════════════════

def save_models(results: dict,
                pipeline,
                le: LabelEncoder,
                output_dir: str = "./models") -> None:
    """
    Save everything needed for inference:
      csi_pipeline.joblib    ← preprocess new recordings
      label_encoder.joblib   ← int → class name
      SVM_RBF.joblib
      Random_Forest.joblib
      metrics.json           ← for thesis tables
    """
    import joblib
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if pipeline is not None:
        joblib.dump(pipeline, out / "csi_pipeline.joblib")
        print(f"💾 {out / 'csi_pipeline.joblib'}")

    joblib.dump(le, out / "label_encoder.joblib")
    print(f"💾 {out / 'label_encoder.joblib'}")

    metrics = {}
    for name, res in results.items():
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        path = out / f"{safe}.joblib"
        joblib.dump(res['model'], path)
        print(f"💾 {path}  (test={res['test_accuracy']*100:.1f}%)")

        metrics[name] = {
            'cv_accuracy_mean': round(res['cv_mean'], 4),
            'cv_accuracy_std':  round(res['cv_std'],  4),
            'test_accuracy':    round(res['test_accuracy'], 4),
            'test_f1_macro':    round(res['test_f1_macro'],  4),
            'confusion_matrix': res['confusion_matrix'].tolist(),
            'classes':          list(le.classes_),
            'feature_importances': res.get('feature_importances', [])
        }

    json_path = out / "metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"📊 {json_path}  (metrics for thesis)")

    best = max(results.items(), key=lambda x: x[1]['cv_mean'])[0]
    safe_best = best.replace(" ", "_").replace("(", "").replace(")", "")
    print(f"\n   Load for inference:")
    print(f"     import joblib")
    print(f"     pipeline = joblib.load('{out}/csi_pipeline.joblib')")
    print(f"     le       = joblib.load('{out}/label_encoder.joblib')")
    print(f"     model    = joblib.load('{out}/{safe_best}.joblib')")


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CSI HAR — ML Pipeline")
    parser.add_argument("--data_dir",    type=str,   default="./datasets")
    parser.add_argument("--classes",     nargs="+",  default=["walk", "idle"])
    parser.add_argument("--window_size", type=int,   default=50)
    parser.add_argument("--step",        type=int,   default=25)
    parser.add_argument("--fs",          type=float, default=100.0)
    parser.add_argument(
        "--augment",
        nargs="+",
        metavar="TECHNIQUE",
        default=ALL_AUGMENT_TECHNIQUES,
        help=(
            "Augmentation techniques to apply on RAW windows (BEFORE PCA). "
            f"Choices: {ALL_AUGMENT_TECHNIQUES}. "
            "Default: all 4 techniques. "
            "Use '--augment noise scale' for a subset. "
            "To disable completely use --no_augment."
        )
    )
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable all data augmentation."
    )
    parser.add_argument("--n_augments",  type=int,   default=4)
    parser.add_argument("--pca",         type=int,   default=10)
    parser.add_argument("--test_ratio",  type=float, default=0.2)
    parser.add_argument("--no_diff",     action="store_true")
    parser.add_argument("--simulate",    action="store_true")
    parser.add_argument("--save_model",  action="store_true")
    parser.add_argument("--tune",        action="store_true",
                        help="Run GridSearchCV hyperparameter tuning")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    # --no_augment disables everything; otherwise use the specified (or default) list
    if args.no_augment:
        augment_techniques = []
    else:
        augment_techniques = args.augment  # list of 1+ techniques
    # Validate
    unknown = set(augment_techniques) - set(ALL_AUGMENT_TECHNIQUES)
    if unknown:
        parser.error(f"Unknown augmentation technique(s): {unknown}. "
                     f"Valid: {ALL_AUGMENT_TECHNIQUES}")

    print("=" * 60)
    print(" CSI HAR — ML Pipeline")
    print(f" Classes : {args.classes}")
    print(f" Data dir: {args.data_dir}")
    print(f" Window  : {args.window_size} frames @ {args.fs} Hz = "
          f"{args.window_size/args.fs:.2f}s")
    aug_label = ', '.join(augment_techniques) if augment_techniques else 'DISABLED'
    print(f" Augment : [{aug_label}] (×{args.n_augments}) | "
          f"PCA: {args.pca} | Diff: {not args.no_diff}")
    print(f" Tune    : {args.tune} | Seed: {args.seed}")
    print("=" * 60)

    (X_train, X_train_orig, X_test,
     y_train, y_train_orig, y_test,
     train_groups_orig, le, pipeline) = build_dataset(
        data_dir=args.data_dir,
        classes=args.classes,
        pipeline_kwargs={'fs': args.fs, 'use_diff': not args.no_diff},
        window_size=args.window_size,
        step=args.step,
        augment_techniques=augment_techniques,
        n_augments=args.n_augments,
        simulation_mode=args.simulate or (CSIPipeline is None),
        test_recording_ratio=args.test_ratio,
        random_seed=args.seed,
        n_pca=args.pca,
    )

    if X_train.shape[0] == 0:
        print("❌ No samples — check data_dir and classes")
        sys.exit(1)

    print(f"\n{'-'*60}\n Step 3: Model Training\n{'-'*60}")

    best_params = None
    if args.tune:
        best_params = tune_hyperparameters(
            X_train_orig, y_train_orig, train_groups_orig,
            random_seed=args.seed
        )

    results = train_and_evaluate(
        X_train, X_train_orig, X_test,
        y_train, y_train_orig, y_test,
        train_groups_orig, le, best_params=best_params,
        random_seed=args.seed
    )

    if args.save_model:
        save_models(results, pipeline, le)




if __name__ == "__main__":
    main()
