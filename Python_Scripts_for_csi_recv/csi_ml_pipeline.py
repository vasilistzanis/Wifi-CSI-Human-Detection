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
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
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
# 1. DATA AUGMENTATION
# ════════════════════════════════════════════════════════════════════════

def augment_window(window: np.ndarray,
                   n_augments: int = 4,
                   seed: int = None) -> list:
    """
    Augment a single window to artificially increase dataset size.

    Techniques (cycling through all 4):
      noise   : Gaussian noise injection
      shift   : Time shift 1-5 frames
      scale   : Amplitude scaling +/-10%
      reverse : Time reversal

    Args:
      window    : (window_size, n_components)
      n_augments: augmented copies (default 4)
      seed      : random seed for reproducibility
    Returns:
      List of augmented windows, same shape as input
    """
    if seed is not None:
        np.random.seed(seed)

    techniques = ['noise', 'shift', 'scale', 'reverse']
    augmented  = []

    for i in range(n_augments):
        aug  = window.copy()
        tech = techniques[i % len(techniques)]

        if tech == 'noise':
            noise_level = 0.02 * aug.std()
            aug = aug + np.random.normal(0, noise_level, aug.shape)
        elif tech == 'shift':
            aug = np.roll(aug, np.random.randint(1, 6), axis=0)
        elif tech == 'scale':
            aug = aug * np.random.uniform(0.90, 1.10)
        elif tech == 'reverse':
            aug = aug[::-1].copy()

        augmented.append(aug.astype(np.float32))

    return augmented


# ════════════════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════════

def extract_features_from_window(window: np.ndarray) -> np.ndarray:
    """
    11 statistical features per PCA component → flat vector.

    Input:  (window_size, n_pca_components)  e.g. (50, 10)
    Output: (110,)  [11 stats x 10 components]

    Stats: mean, std, max, min, range, median, energy, skewness, kurtosis, fft_mean, fft_std
    """
    feats = []
    for c in range(window.shape[1]):
        col      = window[:, c].astype(np.float64)
        mean_val = col.mean()
        std_val  = col.std() + 1e-8
        
        # FFT features
        fft_vals = np.abs(np.fft.rfft(col))
        fft_mean = float(fft_vals.mean())
        fft_std  = float(fft_vals.std())
        
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
        ])
    return np.array(feats, dtype=np.float32)


def _get_feature_names(n_pca_components: int) -> list[str]:
    stats = ['mean', 'std', 'max', 'min', 'range',
             'median', 'energy', 'skewness', 'kurtosis', 'fft_mean', 'fft_std']
    return [f"PC{c+1}_{s}" for c in range(n_pca_components) for s in stats]


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
                   requested_folds: int = 5) -> tuple:
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
            n_splits=n_splits, shuffle=True, random_state=42
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
    augment: bool = True,
    n_augments: int = 4,
    simulation_mode: bool = False,
    test_recording_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple:
    """
    Load recordings, preprocess, extract features.
    Returns train/test split at recording level (no leakage).

    Returns:
      X_train      : (N, 110) augmented train features
      X_train_orig : (N_orig, 110) non-augmented train features for clean CV
      X_test       : (M, 110) test features (no augmentation)
      y_train      : (N,) labels for X_train
      y_train_orig : (N_orig,) labels for X_train_orig
      y_test       : (M,) labels for X_test
      train_groups_orig : (N_orig,) recording ids for X_train_orig
      le           : fitted LabelEncoder
      pipeline     : fitted CSIPipeline
    """
    if pipeline_kwargs is None:
        pipeline_kwargs = {'fs': 100.0, 'use_diff': True}

    data_dir = Path(data_dir)
    le = LabelEncoder()
    le.fit(classes)

    # ── Simulation Mode ──────────────────────────────────────────────────
    if simulation_mode or CSIPipeline is None:
        print("\n🔬 SIMULATION MODE")
        np.random.seed(random_seed)
        X_tr, y_tr = [], []
        X_tr_orig, y_tr_orig = [], []
        train_groups_orig = []
        X_te, y_te = [], []
        recording_group_id = 0

        for label_idx, cls in enumerate(classes):
            n_recs = 20
            n_test = max(1, int(n_recs * test_recording_ratio))
            print(f"   [{cls}] {n_recs} synthetic recordings "
                  f"(train={n_recs-n_test}, test={n_test})")

            for rec_i in range(n_recs):
                t    = np.linspace(0, 5, 500)
                freq = 1.0 + label_idx * 0.5
                r    = (np.outer(np.sin(2*np.pi*freq*t), np.ones(128))
                        + np.random.randn(500, 128) * 0.3)
                im   = (np.outer(np.cos(2*np.pi*freq*t), np.ones(128))
                        + np.random.randn(500, 128) * 0.3)
                cm   = (r + 1j*im).astype(np.complex64)
                cm[:, :6]  = 0
                cm[:, -6:] = 0

                pp = CSIPipeline(**pipeline_kwargs) if CSIPipeline else None
                processed = (pp.fit_transform(cm, use_pca=True,
                                              n_components=10,
                                              scaler_type='standard')
                             if pp else
                             np.random.randn(499, 10).astype(np.float32))

                is_test = (rec_i >= n_recs - n_test)
                for w in extract_windows(processed, window_size, step):
                    feat = extract_features_from_window(w)
                    if is_test:
                        X_te.append(feat)
                        y_te.append(label_idx)
                    else:
                        X_tr_orig.append(feat)
                        y_tr_orig.append(label_idx)
                        train_groups_orig.append(recording_group_id)
                        X_tr.append(feat)
                        y_tr.append(label_idx)
                        if augment:
                            for aw in augment_window(w, n_augments,
                                                     seed=random_seed + recording_group_id):
                                X_tr.append(extract_features_from_window(aw))
                                y_tr.append(label_idx)
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
                           use_pca=True, n_components=10,
                           scaler_type='standard')

    X_tr, y_tr = [], []
    X_tr_orig, y_tr_orig = [], []
    train_groups_orig = []
    X_te, y_te = [], []
    recording_group_id = 0

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
                processed = pipeline.transform(cm, use_pca=True)
            except ValueError as e:
                print(f"   ⚠️  {fpath.name}: {e} — skipped")
                continue
            for w_idx, w in enumerate(extract_windows(processed,
                                                       window_size, step)):
                feat = extract_features_from_window(w)
                X_tr_orig.append(feat)
                y_tr_orig.append(label_idx)
                train_groups_orig.append(recording_group_id)
                X_tr.append(feat)
                y_tr.append(label_idx)
                tr_wins += 1
                if augment:
                    for aw in augment_window(w, n_augments,
                                             seed=random_seed + w_idx):
                        X_tr.append(extract_features_from_window(aw))
                        y_tr.append(label_idx)
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
            for w in extract_windows(processed, window_size, step):
                X_te.append(extract_features_from_window(w))
                y_te.append(label_idx)
                te_wins += 1

        aug_count = tr_wins * n_augments if augment else 0
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
                         cv_folds: int = 5) -> dict:
    """
    GridSearchCV on non-augmented train data.
    Returns best params for SVM and RF.
    """
    print(f"\n{'═'*60}")
    cv, actual_folds, splitter_name = _make_group_cv(
        y_train_orig, train_groups_orig, requested_folds=cv_folds
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

    svm_params = best_params.get('SVM (RBF)', {}) if best_params else {}
    rf_params  = best_params.get('Random Forest', {}) if best_params else {}

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
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        ),
        'K-NN (k=5)': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',   # closer neighbors weigh more
            metric='euclidean',
            n_jobs=-1,
        ),
        'Logistic Regression': LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
        ),
        'Naive Bayes': GaussianNB(),
    }

    cv, actual_folds, splitter_name = _make_group_cv(
        y_train_orig, train_groups_orig, requested_folds=cv_folds
    )
    n_pca = X_train.shape[1] // 11

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

    best = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"\n  🏆 Best: {best[0]} ({best[1]['test_accuracy']*100:.1f}%)")

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

    best = max(results.items(), key=lambda x: x[1]['test_accuracy'])[0]
    safe_best = best.replace(" ", "_").replace("(", "").replace(")", "")
    print(f"\n   Load for inference:")
    print(f"     import joblib")
    print(f"     pipeline = joblib.load('{out}/csi_pipeline.joblib')")
    print(f"     le       = joblib.load('{out}/label_encoder.joblib')")
    print(f"     model    = joblib.load('{out}/{safe_best}.joblib')")


# ════════════════════════════════════════════════════════════════════════
# 7. INFERENCE HELPER
# ════════════════════════════════════════════════════════════════════════

def predict_recording(csv_path: str,
                      pipeline_path: str = "./models/csi_pipeline.joblib",
                      model_path:    str = "./models/Random_Forest.joblib",
                      le_path:       str = "./models/label_encoder.joblib",
                      window_size:   int = 50,
                      step:          int = 25) -> str:
    """
    Classify a new CSI recording using saved models.
    Uses majority vote across all windows.
    """
    import joblib

    pipeline = joblib.load(pipeline_path)
    model    = joblib.load(model_path)
    le       = joblib.load(le_path)

    cm, _ = load_csi_csv(csv_path)
    if cm.size == 0:
        return "ERROR: empty recording"

    processed = pipeline.transform(cm, use_pca=True)
    wins      = extract_windows(processed, window_size, step)

    if not wins:
        return "ERROR: recording too short"

    feats  = np.array([extract_features_from_window(w) for w in wins],
                      dtype=np.float32)
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feats)
        avg_probs = probs.mean(axis=0)
        final_idx = np.argmax(avg_probs)
        final = le.inverse_transform([final_idx])[0]
        conf = avg_probs[final_idx] * 100
    else:
        preds  = model.predict(feats)
        labels = le.inverse_transform(preds)
        final = Counter(labels).most_common(1)[0][0]
        conf  = Counter(labels)[final] / len(labels) * 100

    print(f"🎯 Predicted: {final}  ({conf:.1f}% confidence, {len(wins)} windows)")
    return final


# ════════════════════════════════════════════════════════════════════════
# 8. OPTIONAL CNN
# ════════════════════════════════════════════════════════════════════════

def build_cnn_dataset(data_dir, classes, pipeline,
                      window_size=50, step=25, augment=True, n_augments=4,
                      test_recording_ratio=0.2, random_seed=42):
    """
    Build raw window arrays for CNN (no feature extraction).
    """
    if pipeline is None:
        raise ValueError(
            "build_cnn_dataset needs a fitted pipeline. "
            "Pass the pipeline returned by build_dataset()."
        )

    le = LabelEncoder()
    le.fit(classes)
    X_tr, y_tr, X_te, y_te = [], [], [], []

    for cls in classes:
        files = (sorted((Path(data_dir)/cls).glob("*.csv")) +
                 sorted((Path(data_dir)/cls).glob("*.txt")))
        label_idx   = int(le.transform([cls])[0])
        n_test      = max(1, int(len(files) * test_recording_ratio))
        train_files = files[:-n_test]
        test_files  = files[-n_test:]

        for fpath in train_files:
            cm, _ = load_csi_csv(fpath)
            if cm.size == 0:
                continue
            try:
                processed = pipeline.transform(cm, use_pca=True)
            except ValueError:
                continue
            for w_idx, w in enumerate(extract_windows(processed,
                                                       window_size, step)):
                X_tr.append(w.T)
                y_tr.append(label_idx)
                if augment:
                    for aw in augment_window(w, n_augments,
                                             seed=random_seed + w_idx):
                        X_tr.append(aw.T)
                        y_tr.append(label_idx)

        for fpath in test_files:
            cm, _ = load_csi_csv(fpath)
            if cm.size == 0:
                continue
            try:
                processed = pipeline.transform(cm, use_pca=True)
            except ValueError:
                continue
            for w in extract_windows(processed, window_size, step):
                X_te.append(w.T)
                y_te.append(label_idx)

    return (np.array(X_tr, dtype=np.float32),
            np.array(X_te, dtype=np.float32),
            np.array(y_tr, dtype=np.int64),
            np.array(y_te, dtype=np.int64),
            le)


def train_cnn(X_train, X_test, y_train, y_test, le,
              n_epochs=50, batch_size=32):
    """1D CNN. Input shape: (N, n_components, window_size)"""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("❌ PyTorch not installed: pip install torch")
        return

    n_classes   = len(le.classes_)
    n_channels  = X_train.shape[1]
    n_timesteps = X_train.shape[2]

    class CSI_CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(4),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4, 128), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(128, n_classes),
            )
        def forward(self, x):
            return self.classifier(self.features(x))

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    model     = CSI_CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=batch_size)

    print(f"\n🧠 CNN on {device} | input ({n_channels}, {n_timesteps}) | {n_classes} classes")
    best_acc = 0.0

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    preds = model(xb.to(device)).argmax(1).cpu()
                    correct += (preds == yb).sum().item()
                    total   += len(yb)
            acc      = correct / total
            best_acc = max(best_acc, acc)
            print(f"   Epoch {epoch+1:3d}/{n_epochs}  Test: {acc*100:.2f}%")

    print(f"\n  🏆 Best CNN: {best_acc*100:.2f}%")


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
    parser.add_argument("--no_augment",  action="store_true")
    parser.add_argument("--no_diff",     action="store_true")
    parser.add_argument("--cnn",         action="store_true")
    parser.add_argument("--simulate",    action="store_true")
    parser.add_argument("--save_model",  action="store_true")
    parser.add_argument("--tune",        action="store_true",
                        help="Run GridSearchCV hyperparameter tuning")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(" CSI HAR — ML Pipeline")
    print(f" Classes : {args.classes}")
    print(f" Data dir: {args.data_dir}")
    print(f" Window  : {args.window_size} frames @ {args.fs} Hz = "
          f"{args.window_size/args.fs:.2f}s")
    print(f" Augment : {not args.no_augment} | Diff: {not args.no_diff}")
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
        augment=not args.no_augment,
        n_augments=4,
        simulation_mode=args.simulate or (CSIPipeline is None),
        random_seed=args.seed,
    )

    if X_train.shape[0] == 0:
        print("❌ No samples — check data_dir and classes")
        sys.exit(1)

    print(f"\n{'-'*60}\n Step 3: Model Training\n{'-'*60}")

    best_params = None
    if args.tune:
        best_params = tune_hyperparameters(
            X_train_orig, y_train_orig, train_groups_orig
        )

    results = train_and_evaluate(
        X_train, X_train_orig, X_test,
        y_train, y_train_orig, y_test,
        train_groups_orig, le, best_params=best_params
    )

    if args.save_model:
        save_models(results, pipeline, le)

    if args.cnn:
        print("\n" + "═" * 60)
        print(" CNN TRAINING")
        print("═" * 60)

        if args.simulate or pipeline is None:
            X_cnn_tr = np.random.randn(
                len(X_train), 10, args.window_size).astype(np.float32)
            X_cnn_te = np.random.randn(
                len(X_test), 10, args.window_size).astype(np.float32)
            y_cnn_tr, y_cnn_te = y_train, y_test
        else:
            X_cnn_tr, X_cnn_te, y_cnn_tr, y_cnn_te, _ = build_cnn_dataset(
                data_dir=args.data_dir,
                classes=args.classes,
                pipeline=pipeline,
                window_size=args.window_size,
                step=args.step,
                augment=not args.no_augment,
                random_seed=args.seed,
            )

        train_cnn(X_cnn_tr, X_cnn_te, y_cnn_tr, y_cnn_te, le)


if __name__ == "__main__":
    main()
