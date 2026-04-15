#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSI HAR — Complete ML Pipeline
================================
Supports: SVM, Random Forest, (optional) CNN
Compatible with: CSIPipeline from data_preprocessing.py

Workflow:
  1. Load CSI recordings per class
  2. Preprocess with CSIPipeline
  3. Extract statistical features (windowing)
  4. Train/Evaluate SVM + Random Forest
  5. (Optional) CNN with same windows

Usage:
  python csi_ml_pipeline.py --classes walk idle
  python csi_ml_pipeline.py --classes walk sit fall idle --save_model
  python csi_ml_pipeline.py --classes walk idle --cnn
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

# Sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")


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

def augment_window(window: np.ndarray, n_augments: int = 4) -> list:
    """
    Augment a single window to artificially increase dataset size.

    Techniques (cycling through all 4):
      noise   : Gaussian noise injection
      shift   : Time shift 1-5 frames
      scale   : Amplitude scaling +/-10%
      reverse : Time reversal

    Args:
      window    : (window_size, n_components)
      n_augments: augmented copies (default 4 = one per technique)
    Returns:
      List of augmented windows, same shape as input
    """
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
    9 statistical features per PCA component → flat vector.

    Input:  (window_size, n_pca_components)  e.g. (50, 10)
    Output: (90,)  [9 stats x 10 components]

    Stats: mean, std, max, min, range, median, energy, skewness, kurtosis
    """
    feats = []
    for c in range(window.shape[1]):
        col      = window[:, c].astype(np.float64)
        mean_val = col.mean()
        std_val  = col.std() + 1e-8
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
        ])
    return np.array(feats, dtype=np.float32)


def _get_feature_names(n_pca_components: int) -> list[str]:
    """PC1_mean, PC1_std, ..., PC10_kurtosis"""
    stats = ['mean', 'std', 'max', 'min', 'range',
             'median', 'energy', 'skewness', 'kurtosis']
    return [f"PC{c+1}_{s}" for c in range(n_pca_components) for s in stats]


def extract_windows(data: np.ndarray,
                    window_size: int = 50,
                    step: int = 25) -> list[np.ndarray]:
    """Sliding window → list of (window_size, n_components) arrays."""
    if data.shape[0] < window_size:
        return []
    return [data[s:s + window_size]
            for s in range(0, data.shape[0] - window_size + 1, step)]


# ════════════════════════════════════════════════════════════════════════
# 3. DATASET BUILDER
# ════════════════════════════════════════════════════════════════════════
#
# KEY FIX — Augmentation leakage prevention:
#   Split is done at RECORDING level, not window level.
#   Test recordings → no augmentation, never mixed with train.
#   Train recordings → augmented freely.
#
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
) -> tuple:
    """
    Load recordings, preprocess, extract features.
    Returns train/test split at recording level (no leakage).

    Returns:
      X_train, X_test : (N, 90) feature matrices
      y_train, y_test : (N,) integer labels
      le              : fitted LabelEncoder
      pipeline        : fitted CSIPipeline  ← needed for CNN + inference
    """
    if pipeline_kwargs is None:
        pipeline_kwargs = {'fs': 100.0, 'use_diff': True}

    data_dir = Path(data_dir)
    le = LabelEncoder()
    le.fit(classes)

    # ── Simulation Mode ──────────────────────────────────────────────────
    if simulation_mode or CSIPipeline is None:
        print("\n🔬 SIMULATION MODE")
        np.random.seed(42)
        X_tr, y_tr, X_te, y_te = [], [], [], []

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
                        X_te.append(feat);  y_te.append(label_idx)
                    else:
                        X_tr.append(feat);  y_tr.append(label_idx)
                        if augment:
                            for aw in augment_window(w, n_augments):
                                X_tr.append(extract_features_from_window(aw))
                                y_tr.append(label_idx)

        X_train = np.array(X_tr, dtype=np.float32)
        X_test  = np.array(X_te, dtype=np.float32)
        y_train = np.array(y_tr, dtype=np.int32)
        y_test  = np.array(y_te, dtype=np.int32)
        print(f"\n✅ Train={len(X_train)} | Test={len(X_test)} samples")
        return X_train, X_test, y_train, y_test, le, None

    # ── Real Data Mode ───────────────────────────────────────────────────
    print(f"\n📂 Loading data from: {data_dir}")

    # Fit pipeline on first recording of each class
    fit_matrices = []
    for cls in classes:
        files = (sorted((data_dir/cls).glob("*.csv")) +
                 sorted((data_dir/cls).glob("*.txt")))
        if not files:
            print(f"⚠️  No files found for class '{cls}'")
            continue
        cm, _ = load_csi_csv(files[0])
        if cm.size > 0:
            fit_matrices.append(cm)

    if not fit_matrices:
        raise ValueError("No valid CSI data found.")

    print("\n🔧 Fitting CSIPipeline...")
    pipeline = CSIPipeline(**pipeline_kwargs)
    pipeline.fit_transform(np.vstack(fit_matrices),
                           use_pca=True, n_components=10,
                           scaler_type='standard')

    # Extract features — recording-level split
    X_tr, y_tr = [], []
    X_te, y_te = [], []

    for cls in classes:
        files = (sorted((data_dir/cls).glob("*.csv")) +
                 sorted((data_dir/cls).glob("*.txt")))
        label_idx   = int(le.transform([cls])[0])
        n_test      = max(1, int(len(files) * test_recording_ratio))
        train_files = files[:-n_test]
        test_files  = files[-n_test:]

        print(f"\n   [{cls}]  "
              f"train={len(train_files)} | test={len(test_files)} recordings")

        # Training files → augment
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
            for w in extract_windows(processed, window_size, step):
                X_tr.append(extract_features_from_window(w))
                y_tr.append(label_idx)
                tr_wins += 1
                if augment:
                    for aw in augment_window(w, n_augments):
                        X_tr.append(extract_features_from_window(aw))
                        y_tr.append(label_idx)

        # Test files → NO augmentation
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

    X_train = np.array(X_tr, dtype=np.float32)
    X_test  = np.array(X_te, dtype=np.float32)
    y_train = np.array(y_tr, dtype=np.int32)
    y_test  = np.array(y_te, dtype=np.int32)

    print(f"\n✅ Dataset ready:")
    print(f"   Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    dist = ", ".join(f"{cls}={int((y_train==i).sum())}"
                     for i, cls in enumerate(le.classes_))
    print(f"   Train class distribution: {dist}")

    return X_train, X_test, y_train, y_test, le, pipeline


# ════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING & EVALUATION
# ════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    le: LabelEncoder,
    cv_folds: int = 5,
) -> dict:
    """Train SVM + RF, evaluate with CV on train + hold-out test."""
    results = {}

    print(f"\n{'═'*60}")
    print(f" MODEL TRAINING & EVALUATION")
    print(f" Classes : {list(le.classes_)}")
    print(f" Train   : {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f" Features: {X_train.shape[1]}")
    print(f"{'═'*60}")

    models = {
        'SVM (RBF)': SVC(
            kernel='rbf', C=10, gamma='scale',
            class_weight='balanced', probability=True,
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=2,
            class_weight='balanced', n_jobs=-1, random_state=42,
        ),
    }

    cv        = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    n_pca     = X_train.shape[1] // 9

    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")

        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        print(f"  {cv_folds}-Fold CV (train): "
              f"{cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)

        print(f"  Hold-out Test Accuracy: {acc*100:.2f}%")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=le.classes_, digits=3))

        cm = confusion_matrix(y_test, y_pred)
        print(f"  Confusion Matrix:")
        print("          " + "  ".join(f"{c:>8}" for c in le.classes_))
        for i, row in enumerate(cm):
            print(f"  {le.classes_[i]:>8}  " +
                  "  ".join(f"{v:>8}" for v in row))

        results[name] = {
            'model': model, 'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(), 'test_accuracy': acc,
            'confusion_matrix': cm, 'y_pred': y_pred, 'y_test': y_test,
        }

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_names  = _get_feature_names(n_pca)
            top_idx     = np.argsort(importances)[::-1][:10]
            print(f"\n  Top 10 Important Features:")
            for rank, idx in enumerate(top_idx):
                fname = feat_names[idx] if idx < len(feat_names) else f"feat_{idx}"
                print(f"    {rank+1:2}. {fname:30s}  {importances[idx]*100:.2f}%")

    print(f"\n{'═'*60}")
    print(f" SUMMARY")
    print(f"{'═'*60}")
    for name, res in results.items():
        print(f"  {name:20s}  "
              f"CV={res['cv_mean']*100:.1f}% ±{res['cv_std']*100:.1f}%  "
              f"Test={res['test_accuracy']*100:.1f}%")

    best = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"\n  🏆 Best: {best[0]} ({best[1]['test_accuracy']*100:.1f}%)")

    return results


# ════════════════════════════════════════════════════════════════════════
# 5. SAVE MODELS  (FIX: saves pipeline + model + label_encoder)
# ════════════════════════════════════════════════════════════════════════

def save_models(results: dict,
                pipeline,
                le: LabelEncoder,
                output_dir: str = "./models") -> None:
    """
    Save everything needed for inference:
      csi_pipeline.joblib    ← preprocess new recordings
      label_encoder.joblib   ← int → class name
      SVM_RBF.joblib         ← SVM model
      Random_Forest.joblib   ← RF model
    """
    import joblib
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if pipeline is not None:
        joblib.dump(pipeline, out / "csi_pipeline.joblib")
        print(f"💾 {out / 'csi_pipeline.joblib'}")

    joblib.dump(le, out / "label_encoder.joblib")
    print(f"💾 {out / 'label_encoder.joblib'}")

    for name, res in results.items():
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        path = out / f"{safe}.joblib"
        joblib.dump(res['model'], path)
        print(f"💾 {path}  (test={res['test_accuracy']*100:.1f}%)")

    best = max(results.items(), key=lambda x: x[1]['test_accuracy'])[0]
    safe_best = best.replace(" ", "_").replace("(", "").replace(")", "")
    print(f"\n   Load for inference:")
    print(f"     import joblib")
    print(f"     pipeline = joblib.load('{out}/csi_pipeline.joblib')")
    print(f"     le       = joblib.load('{out}/label_encoder.joblib')")
    print(f"     model    = joblib.load('{out}/{safe_best}.joblib')")


# ════════════════════════════════════════════════════════════════════════
# 6. INFERENCE HELPER
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
    preds  = model.predict(feats)
    labels = le.inverse_transform(preds)

    final = Counter(labels).most_common(1)[0][0]
    conf  = Counter(labels)[final] / len(labels) * 100
    print(f"🎯 Predicted: {final}  ({conf:.1f}% confidence, {len(wins)} windows)")
    return final


# ════════════════════════════════════════════════════════════════════════
# 7. OPTIONAL CNN  (FIX: pipeline passed correctly — no longer None)
# ════════════════════════════════════════════════════════════════════════

def build_cnn_dataset(data_dir, classes, pipeline,
                      window_size=50, step=25, augment=True, n_augments=4,
                      test_recording_ratio=0.2):
    """
    Build raw window arrays for CNN (no feature extraction).
    Same recording-level split logic as build_dataset.

    Returns: X_train, X_test, y_train, y_test, le
    Shape:   (N, n_components, window_size)  — ready for Conv1d
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
            for w in extract_windows(processed, window_size, step):
                X_tr.append(w.T);  y_tr.append(label_idx)
                if augment:
                    for aw in augment_window(w, n_augments):
                        X_tr.append(aw.T);  y_tr.append(label_idx)

        for fpath in test_files:
            cm, _ = load_csi_csv(fpath)
            if cm.size == 0:
                continue
            try:
                processed = pipeline.transform(cm, use_pca=True)
            except ValueError:
                continue
            for w in extract_windows(processed, window_size, step):
                X_te.append(w.T);  y_te.append(label_idx)

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
    args = parser.parse_args()

    print("=" * 60)
    print(" CSI HAR — ML Pipeline")
    print(f" Classes : {args.classes}")
    print(f" Data dir: {args.data_dir}")
    print(f" Window  : {args.window_size} frames @ {args.fs} Hz = "
          f"{args.window_size/args.fs:.2f}s")
    print(f" Augment : {not args.no_augment} | Diff: {not args.no_diff}")
    print("=" * 60)

    X_train, X_test, y_train, y_test, le, pipeline = build_dataset(
        data_dir=args.data_dir,
        classes=args.classes,
        pipeline_kwargs={'fs': args.fs, 'use_diff': not args.no_diff},
        window_size=args.window_size,
        step=args.step,
        augment=not args.no_augment,
        n_augments=4,
        simulation_mode=args.simulate or (CSIPipeline is None),
    )

    if X_train.shape[0] == 0:
        print("❌ No samples — check data_dir and classes")
        sys.exit(1)

    results = train_and_evaluate(X_train, X_test, y_train, y_test, le)

    if args.save_model:
        save_models(results, pipeline, le)

    if args.cnn:
        print("\n" + "═" * 60)
        print(" CNN TRAINING")
        print("═" * 60)

        if args.simulate or pipeline is None:
            n = len(X_train)
            X_cnn_tr = np.random.randn(n, 10, args.window_size).astype(np.float32)
            X_cnn_te = np.random.randn(len(X_test), 10,
                                       args.window_size).astype(np.float32)
            y_cnn_tr, y_cnn_te = y_train, y_test
        else:
            X_cnn_tr, X_cnn_te, y_cnn_tr, y_cnn_te, _ = build_cnn_dataset(
                data_dir=args.data_dir,
                classes=args.classes,
                pipeline=pipeline,   # FIX: real pipeline, not None
                window_size=args.window_size,
                step=args.step,
                augment=not args.no_augment,
            )

        train_cnn(X_cnn_tr, X_cnn_te, y_cnn_tr, y_cnn_te, le)


if __name__ == "__main__":
    main()