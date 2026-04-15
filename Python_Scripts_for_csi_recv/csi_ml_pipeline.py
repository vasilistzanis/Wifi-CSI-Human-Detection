#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSI HAR — Complete ML Pipeline
================================
Supports: SVM, Random Forest, (optional) CNN
Compatible with: CSIPipeline from csi_preprocessing.py

Workflow:
  1. Load CSI recordings per class
  2. Preprocess with CSIPipeline
  3. Extract statistical features (windowing)
  4. Train/Evaluate SVM + Random Forest
  5. (Optional) CNN with same windows

Usage:
  python csi_ml_pipeline.py --data_dir ./data --classes walk sit fall stand
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# Sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline as SklearnPipeline

import warnings
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════
# IMPORT  PREPROCESSING PIPELINE
# ════════════════════════════════════════════════════════════════════════

# Make sure data_preprocessing.py is in the same folder
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

def augment_window(window: np.ndarray, n_augments: int = 3) -> list:
    """
    Augment a single window to artificially increase dataset size.
    Essential when you have < 200 recordings.

    Techniques:
      - Gaussian noise injection
      - Time shift
      - Amplitude scaling
      - Time reversal

    Args:
      window    : (window_size, n_components) array
      n_augments: how many augmented copies to create

    Returns:
      List of augmented windows (each same shape as input)
    """
    augmented = []
    techniques = ['noise', 'shift', 'scale', 'reverse']

    for i in range(n_augments):
        tech = techniques[i % len(techniques)]
        aug = window.copy()

        if tech == 'noise':
            # Small Gaussian noise — preserves signal shape
            noise_level = 0.02 * aug.std()
            aug = aug + np.random.normal(0, noise_level, aug.shape)

        elif tech == 'shift':
            # Time shift by 1-5 frames
            shift = np.random.randint(1, 6)
            aug = np.roll(aug, shift, axis=0)

        elif tech == 'scale':
            # Amplitude scaling ±10%
            scale = np.random.uniform(0.90, 1.10)
            aug = aug * scale

        elif tech == 'reverse':
            # Time reversal — same activity, different direction
            aug = aug[::-1].copy()

        augmented.append(aug.astype(np.float32))

    return augmented


# ════════════════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION (Windowing → Statistical Features)
# ════════════════════════════════════════════════════════════════════════

def extract_features_from_window(window: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from a single window.

    Input:  (window_size, n_pca_components)  e.g. (50, 10)
    Output: (n_features,)                    e.g. (90,) flat vector

    Features per PCA component (9 stats × 10 components = 90):
      - mean, std, max, min, range, median,
        energy, skewness, kurtosis (via manual calculation)
    """
    feats = []

    for c in range(window.shape[1]):
        col = window[:, c].astype(np.float64)

        mean_val  = col.mean()
        std_val   = col.std() + 1e-8  # avoid division by zero

        feats.extend([
            mean_val,                              # 1. mean
            std_val,                               # 2. std
            col.max(),                             # 3. max
            col.min(),                             # 4. min
            col.max() - col.min(),                 # 5. range
            float(np.median(col)),                 # 6. median
            float(np.sum(col ** 2)),               # 7. energy
            float(np.mean(((col - mean_val)        # 8. skewness
                           / std_val) ** 3)),
            float(np.mean(((col - mean_val)        # 9. kurtosis
                           / std_val) ** 4)),
        ])

    return np.array(feats, dtype=np.float32)


def extract_features(
    data: np.ndarray,
    window_size: int = 50,
    step: int = 25,
    augment: bool = False,
    n_augments: int = 3
) -> tuple[np.ndarray, int]:
    """
    Slide a window over preprocessed CSI data and extract features.

    Args:
      data        : (N_frames, n_pca_components) from CSIPipeline
      window_size : frames per window (default 50 = 0.5s @ 100Hz)
      step        : hop size (default 25 = 50% overlap)
      augment     : whether to augment each window
      n_augments  : augmented copies per window

    Returns:
      features    : (N_windows, n_features) or more if augmented
      n_windows   : number of original (non-augmented) windows
    """
    if data.shape[0] < window_size:
        print(f"⚠️  Recording too short ({data.shape[0]} frames < "
              f"window_size={window_size}) — skipping")
        return np.zeros((0, 0), dtype=np.float32), 0

    windows = []
    for start in range(0, data.shape[0] - window_size + 1, step):
        w = data[start:start + window_size]
        windows.append(w)

    n_windows = len(windows)
    all_features = []

    for w in windows:
        all_features.append(extract_features_from_window(w))
        if augment:
            for aug_w in augment_window(w, n_augments):
                all_features.append(extract_features_from_window(aug_w))

    return np.array(all_features, dtype=np.float32), n_windows


# ════════════════════════════════════════════════════════════════════════
# 3. DATASET BUILDER
# ════════════════════════════════════════════════════════════════════════

def build_dataset(
    data_dir: str | Path,
    classes: list[str],
    pipeline_kwargs: dict = None,
    window_size: int = 50,
    step: int = 25,
    augment: bool = True,
    n_augments: int = 3,
    simulation_mode: bool = False
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Load all CSI recordings, preprocess, extract features.

    Expected directory structure:
      data_dir/
        walk/
          rec_001.csv
          rec_002.csv
          ...
        sit/
          rec_001.csv
          ...
        fall/
          ...

    Args:
      data_dir       : root directory
      classes        : list of class names (subfolder names)
      pipeline_kwargs: dict passed to CSIPipeline()
      window_size    : frames per window
      step           : hop size between windows
      augment        : whether to use data augmentation
      n_augments     : augmented copies per window
      simulation_mode: generate synthetic data if True

    Returns:
      X  : (N_samples, n_features) feature matrix
      y  : (N_samples,) integer labels
      le : fitted LabelEncoder (le.classes_ gives class names)
    """
    if pipeline_kwargs is None:
        pipeline_kwargs = {'fs': 100.0, 'use_diff': True}

    data_dir = Path(data_dir)
    le = LabelEncoder()
    le.fit(classes)

    X_list, y_list = [], []

    # ── Simulation Mode ──────────────────────────────────────────────────
    if simulation_mode or CSIPipeline is None:
        print("\n🔬 SIMULATION MODE — generating synthetic CSI data")
        print(f"   Classes: {classes}")
        np.random.seed(42)

        for label_idx, cls in enumerate(classes):
            print(f"\n   [{cls}] Generating 20 synthetic recordings...")
            for rec_i in range(20):
                # Each class has slightly different frequency content
                t = np.linspace(0, 5, 500)
                freq = 1.0 + label_idx * 0.5
                r  = (np.outer(np.sin(2 * np.pi * freq * t),
                               np.ones(128)) +
                      np.random.randn(500, 128) * 0.3)
                im = (np.outer(np.cos(2 * np.pi * freq * t),
                               np.ones(128)) +
                      np.random.randn(500, 128) * 0.3)
                complex_matrix = (r + 1j * im).astype(np.complex64)
                complex_matrix[:, :6]  = 0
                complex_matrix[:, -6:] = 0

                # Each recording gets its own pipeline (fit on training data)
                pp = CSIPipeline(**pipeline_kwargs) if CSIPipeline else None
                if pp:
                    processed = pp.fit_transform(
                        complex_matrix, use_pca=True,
                        n_components=10, scaler_type='standard'
                    )
                else:
                    processed = np.random.randn(499, 10).astype(np.float32)

                feats, nw = extract_features(
                    processed, window_size, step,
                    augment=(augment and rec_i < 10),
                    n_augments=n_augments
                )
                if feats.shape[0] > 0:
                    X_list.append(feats)
                    y_list.extend([label_idx] * len(feats))

        X = np.vstack(X_list)
        y = np.array(y_list, dtype=np.int32)
        print(f"\n✅ Simulation dataset: {X.shape[0]} samples, "
              f"{X.shape[1]} features, {len(classes)} classes")
        return X, y, le

    # ── Real Data Mode ───────────────────────────────────────────────────
    print(f"\n📂 Loading data from: {data_dir}")

    # Step 1: Fit pipeline on ALL training data combined (first pass)
    # We collect one representative recording per class for fitting
    first_pass_matrices = []

    for cls in classes:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            print(f"⚠️  Directory not found: {cls_dir} — skipping")
            continue
        files = sorted(cls_dir.glob("*.csv")) + sorted(cls_dir.glob("*.txt"))
        if not files:
            print(f"⚠️  No .csv/.txt files in {cls_dir}")
            continue
        # Use first file for fitting
        cm, _ = load_csi_csv(files[0])
        if cm.size > 0:
            first_pass_matrices.append(cm)

    if not first_pass_matrices:
        raise ValueError("No valid CSI data found. Check data_dir and file format.")

    # Fit pipeline on concatenated representative data
    print("\n🔧 Fitting CSIPipeline on representative data...")
    fit_matrix = np.vstack(first_pass_matrices)
    pipeline = CSIPipeline(**pipeline_kwargs)
    pipeline.fit_transform(
        fit_matrix, use_pca=True,
        n_components=10, scaler_type='standard'
    )

    # Step 2: Transform all recordings
    for cls in classes:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            continue

        files = sorted(cls_dir.glob("*.csv")) + sorted(cls_dir.glob("*.txt"))
        label_idx = le.transform([cls])[0]
        cls_windows = 0

        print(f"\n   [{cls}] {len(files)} recordings...")

        for f_idx, fpath in enumerate(files):
            cm, _ = load_csi_csv(fpath)
            if cm.size == 0:
                continue

            try:
                processed = pipeline.transform(cm, use_pca=True)
            except ValueError as e:
                print(f"   ⚠️  {fpath.name}: {e} — skipped")
                continue

            # Augment only training files (not the last 20%)
            do_aug = augment and (f_idx < int(len(files) * 0.8))
            feats, nw = extract_features(
                processed, window_size, step,
                augment=do_aug, n_augments=n_augments
            )

            if feats.shape[0] > 0:
                X_list.append(feats)
                y_list.extend([label_idx] * len(feats))
                cls_windows += len(feats)

        print(f"   → {cls_windows} windows total")

    if not X_list:
        raise ValueError("No features extracted. Check recordings.")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int32)

    print(f"\n✅ Dataset ready: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"   Class distribution: "
          + ", ".join(f"{cls}={cnt}"
                      for cls, cnt in zip(le.classes_,
                                          np.bincount(y))))
    return X, y, le


# ════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING & EVALUATION
# ════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    test_size: float = 0.2,
    cv_folds: int = 5
) -> dict:
    """
    Train SVM and Random Forest, evaluate with both hold-out and
    k-fold cross-validation.

    Returns dict with all results.
    """
    results = {}
    n_classes = len(le.classes_)

    print(f"\n{'═'*60}")
    print(f" MODEL TRAINING & EVALUATION")
    print(f" Classes: {list(le.classes_)}")
    print(f" Total samples: {len(X)}  |  Features: {X.shape[1]}")
    print(f"{'═'*60}")

    # Train/test split (stratified — keeps class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )
    print(f"\n📊 Split: {len(X_train)} train / {len(X_test)} test")

    # ── MODEL DEFINITIONS ────────────────────────────────────────────────
    models = {
        'SVM (RBF)': SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            class_weight='balanced',
            probability=True,   # enables predict_proba
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")

        # ── Cross-Validation (on all data)
        cv_scores = cross_val_score(
            model, X, y, cv=cv, scoring='accuracy', n_jobs=-1
        )
        print(f"  {cv_folds}-Fold CV Accuracy: "
              f"{cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

        # ── Hold-out evaluation
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"  Hold-out Test Accuracy: {acc*100:.2f}%")
        print(f"\n  Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            digits=3
        ))

        # Confusion Matrix (text)
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Confusion Matrix:")
        header = "         " + "  ".join(f"{c:>8}" for c in le.classes_)
        print(header)
        for i, row in enumerate(cm):
            row_str = f"  {le.classes_[i]:>8} " + "  ".join(
                f"{v:>8}" for v in row
            )
            print(row_str)

        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': acc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_test': y_test
        }

        # Feature Importance (RF only)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_k = 10
            top_idx = np.argsort(importances)[::-1][:top_k]
            feat_names = _get_feature_names(X.shape[1])
            print(f"\n  Top {top_k} Important Features:")
            for rank, idx in enumerate(top_idx):
                fname = feat_names[idx] if idx < len(feat_names) else f"feat_{idx}"
                print(f"    {rank+1:2}. {fname:30s}  {importances[idx]*100:.2f}%")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f" SUMMARY")
    print(f"{'═'*60}")
    for name, res in results.items():
        print(f"  {name:20s} "
              f"CV={res['cv_mean']*100:.1f}% ±{res['cv_std']*100:.1f}%  "
              f"Test={res['test_accuracy']*100:.1f}%")

    best = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"\n  🏆 Best: {best[0]} "
          f"({best[1]['test_accuracy']*100:.1f}% test accuracy)")

    return results


def _get_feature_names(n_features: int) -> list[str]:
    """Generate feature names for 9 stats × n_components."""
    stats = ['mean', 'std', 'max', 'min', 'range',
             'median', 'energy', 'skewness', 'kurtosis']
    n_components = n_features // len(stats)
    names = []
    for c in range(n_components):
        for s in stats:
            names.append(f"PC{c+1}_{s}")
    return names


# ════════════════════════════════════════════════════════════════════════
# 5. OPTIONAL CNN
# ════════════════════════════════════════════════════════════════════════

def build_cnn_dataset(
    data_dir: str | Path,
    classes: list[str],
    pipeline: object,    # fitted CSIPipeline
    window_size: int = 50,
    step: int = 25,
    augment: bool = True,
    n_augments: int = 3
) -> tuple:
    """
    Build raw window dataset (no feature extraction) for CNN.

    Returns:
      X : (N_samples, n_pca_components, window_size) — for Conv1d
      y : (N_samples,) labels
    """
    le = LabelEncoder()
    le.fit(classes)
    X_list, y_list = [], []

    for cls in classes:
        cls_dir = Path(data_dir) / cls
        if not cls_dir.exists():
            continue
        files = sorted(cls_dir.glob("*.csv")) + sorted(cls_dir.glob("*.txt"))
        label_idx = le.transform([cls])[0]

        for f_idx, fpath in enumerate(files):
            cm, _ = load_csi_csv(fpath)
            if cm.size == 0:
                continue
            try:
                processed = pipeline.transform(cm, use_pca=True)
            except ValueError:
                continue

            for start in range(0, processed.shape[0] - window_size + 1, step):
                w = processed[start:start + window_size]  # (50, 10)
                # CNN input: (channels, time) = (10, 50)
                X_list.append(w.T)
                y_list.append(label_idx)

                if augment and f_idx < int(len(files) * 0.8):
                    for aug_w in augment_window(w, n_augments):
                        X_list.append(aug_w.T)
                        y_list.append(label_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, le


def train_cnn(X: np.ndarray, y: np.ndarray,
              le: LabelEncoder,
              n_epochs: int = 50,
              batch_size: int = 32) -> None:
    """
    Train a 1D CNN on windowed CSI data.
    Requires: pip install torch

    Input shape: (N, n_components, window_size) = (N, 10, 50)
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("❌ PyTorch not installed. Run: pip install torch")
        return

    n_classes = len(le.classes_)
    n_channels = X.shape[1]   # 10
    n_timesteps = X.shape[2]  # 50

    class CSI_CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),                   # → (32, 25)

                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),                   # → (64, 12)

                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(4),           # → (128, 4)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, n_classes)
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    # Split
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🧠 CNN Training on {device}")
    print(f"   Input: ({n_channels}, {n_timesteps})  Classes: {n_classes}")

    model = CSI_CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5
    )
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.tensor(X_tr), torch.tensor(y_tr)
    )
    test_ds  = TensorDataset(
        torch.tensor(X_te), torch.tensor(y_te)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    best_acc = 0.0
    for epoch in range(n_epochs):
        # Train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    preds = model(xb.to(device)).argmax(1).cpu()
                    correct += (preds == yb).sum().item()
                    total += len(yb)
            acc = correct / total
            best_acc = max(best_acc, acc)
            print(f"   Epoch {epoch+1:3d}/{n_epochs}  "
                  f"Test Accuracy: {acc*100:.2f}%")

    print(f"\n  🏆 Best CNN Accuracy: {best_acc*100:.2f}%")


# ════════════════════════════════════════════════════════════════════════
# 6. SAVE / LOAD MODEL
# ════════════════════════════════════════════════════════════════════════

def save_model(results: dict, output_dir: str = "./models") -> None:
    """Save best model to disk using joblib."""
    import joblib
    os.makedirs(output_dir, exist_ok=True)

    best_name = max(results.items(),
                    key=lambda x: x[1]['test_accuracy'])[0]
    best_model = results[best_name]['model']

    safe_name = best_name.replace(" ", "_").replace("(", "").replace(")", "")
    path = Path(output_dir) / f"{safe_name}.joblib"
    joblib.dump(best_model, path)
    print(f"\n💾 Best model saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CSI HAR — SVM + Random Forest Pipeline"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./datasets",
        help="Root directory with class subfolders"
    )
    parser.add_argument(
        "--classes", nargs="+",
        default=["walk", "sit", "fall", "idle"],
        help="Class names (must match subfolder names)"
    )
    parser.add_argument(
        "--window_size", type=int, default=50,
        help="Window size in frames (default: 50 = 0.5s @ 100Hz)"
    )
    parser.add_argument(
        "--step", type=int, default=25,
        help="Window hop size (default: 25 = 50% overlap)"
    )
    parser.add_argument(
        "--fs", type=float, default=100.0,
        help="Sampling frequency Hz (default: 100)"
    )
    parser.add_argument(
        "--no_augment", action="store_true",
        help="Disable data augmentation"
    )
    parser.add_argument(
        "--no_diff", action="store_true",
        help="Disable temporal differencing"
    )
    parser.add_argument(
        "--cnn", action="store_true",
        help="Also train CNN (requires PyTorch)"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Use synthetic data (no real files needed)"
    )
    parser.add_argument(
        "--save_model", action="store_true",
        help="Save best model to ./models/"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" CSI HAR — ML Pipeline")
    print(f" Classes : {args.classes}")
    print(f" Data dir: {args.data_dir}")
    print(f" Window  : {args.window_size} frames @ {args.fs} Hz "
          f"= {args.window_size/args.fs:.2f}s")
    print(f" Augment : {not args.no_augment}")
    print(f" Diff    : {not args.no_diff}")
    print("=" * 60)

    # Build dataset
    X, y, le = build_dataset(
        data_dir=args.data_dir,
        classes=args.classes,
        pipeline_kwargs={'fs': args.fs, 'use_diff': not args.no_diff},
        window_size=args.window_size,
        step=args.step,
        augment=not args.no_augment,
        n_augments=3,
        simulation_mode=args.simulate or (CSIPipeline is None)
    )

    if X.shape[0] == 0:
        print("❌ No samples — check data_dir and classes")
        sys.exit(1)

    # Train SVM + RF
    results = train_and_evaluate(X, y, le)

    # Save best model
    if args.save_model:
        save_model(results)

    # Optional CNN
    if args.cnn:
        print("\n" + "═" * 60)
        print(" CNN TRAINING")
        print("═" * 60)
        if args.simulate or CSIPipeline is None:
            # Synthetic CNN data
            X_cnn = np.random.randn(len(X), 10, args.window_size).astype(np.float32)
            y_cnn = y
        else:
            # Build CNN dataset with raw windows
            X_cnn, y_cnn, _ = build_cnn_dataset(
                data_dir=args.data_dir,
                classes=args.classes,
                pipeline=None,     # pass your fitted pipeline here
                window_size=args.window_size,
                step=args.step,
                augment=not args.no_augment
            )
        train_cnn(X_cnn, y_cnn, le)


if __name__ == "__main__":
    main()
