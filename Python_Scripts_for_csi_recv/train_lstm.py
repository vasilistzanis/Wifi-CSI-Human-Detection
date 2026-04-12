#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSI Human Activity Recognition — LSTM Training Script (Thesis Grade)
=====================================================================
Trains a stacked Bi-LSTM model on preprocessed CSI windows (.npy files).

Expected dataset structure:
  dataset/
    idle_001.npy     # shape: (FRAMES, SUBCARRIERS)
    walk_001.npy
    sit_001.npy
    fall_001.npy
    ...

Each .npy file = one labeled window of CSI data.

Usage:
  python train_lstm.py
  python train_lstm.py --epochs 50 --batch-size 32
  python train_lstm.py --dataset my_dataset --no-augment
"""

import os
import sys
import glob
import json
import argparse
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info/warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive: save to file
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical


# ════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ════════════════════════════════════════════════════════════════════════

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ════════════════════════════════════════════════════════════════════════
# ARGS
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="CSI HAR — LSTM Training Script"
    )
    p.add_argument("--dataset",    default="datasets",
                   help="Dataset folder (default: datasets/)")
    p.add_argument("--classes",    nargs="+",
                   default=["idle", "walk", "sit", "fall"],
                   help="Activity class names (must appear in filename)")
    p.add_argument("--frames",     type=int, default=300,
                   help="Time steps per window (default: 300)")
    p.add_argument("--subcarriers",type=int, default=114,
                   help="Features per time step (default: 114)")
    p.add_argument("--epochs",     type=int, default=60,
                   help="Max training epochs (default: 60)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size (default: 16)")
    p.add_argument("--test-split", type=float, default=0.2,
                   help="Test set fraction (default: 0.2)")
    p.add_argument("--val-split",  type=float, default=0.15,
                   help="Validation set fraction from training (default: 0.15)")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable data augmentation")
    p.add_argument("--output-dir", default=".",
                   help="Directory for saved model and plots (default: .)")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════

def load_data(dataset_dir: str, classes: list[str],
              frames: int, subcarriers: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all .npy files from dataset_dir.
    File name must contain a class name (case-insensitive).
    Returns X: (N, frames, subcarriers), y: (N,) int labels.
    """
    print(f"\n📂 Loading data from '{dataset_dir}'...")
    files = glob.glob(os.path.join(dataset_dir, "**", "*.npy"), recursive=True)

    if not files:
        print(f"❌ No .npy files found in '{dataset_dir}'")
        sys.exit(1)

    X, y = [], []
    skipped_label = 0
    skipped_shape = 0

    for path in sorted(files):
        name = os.path.basename(path).lower()

        label_idx = next(
            (i for i, cls in enumerate(classes) if cls.lower() in name), -1
        )
        if label_idx == -1:
            skipped_label += 1
            continue

        try:
            data = np.load(path)
        except Exception as e:
            print(f"   ⚠️  Cannot load {name}: {e}")
            continue

        if data.shape != (frames, subcarriers):
            print(f"   ⚠️  Skipped {name}: shape {data.shape} "
                  f"≠ expected ({frames}, {subcarriers})")
            skipped_shape += 1
            continue

        # NaN/Inf check: motion_detector zero-pads short windows, which is fine.
        # But corrupted exports or preprocessing errors can produce NaN/Inf.
        # These would cause the LSTM to produce nan loss silently from epoch 1.
        if np.any(~np.isfinite(data)):
            n_bad = int(np.sum(~np.isfinite(data)))
            print(f"   ⚠️  Skipped {name}: contains {n_bad} NaN/Inf values "
                  f"(corrupted export or preprocessing error)")
            skipped_shape += 1
            continue

        X.append(data.astype(np.float32))
        y.append(label_idx)

    if skipped_label:
        print(f"   ℹ️  {skipped_label} files skipped (no class keyword in name)")
    if skipped_shape:
        print(f"   ⚠️  {skipped_shape} files skipped (wrong shape)")

    if not X:
        print("❌ No valid samples loaded. Check dataset structure.")
        sys.exit(1)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)

    # ── Per-class distribution ──────────────────────────────────────────
    print(f"\n✅ Loaded {len(X_arr)} samples  |  shape: {X_arr.shape}")
    print("   Class distribution:")
    for i, cls in enumerate(classes):
        count = int((y_arr == i).sum())
        bar = "█" * (count // max(1, len(X_arr) // 40))
        print(f"     [{i}] {cls:<10} : {count:4d}  {bar}")

    return X_arr, y_arr


# ════════════════════════════════════════════════════════════════════════
# DATA AUGMENTATION
# ════════════════════════════════════════════════════════════════════════

def augment(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Augmentation for time-series CSI — triples the dataset:
      Original + Gaussian noise + time-shift = 3× samples.

      1. Gaussian noise injection  (σ = 2% of each sample's own std)
      2. Time jitter (random ±5% shift with zero-padding)

    Applied ONLY to training data, never to validation or test.
    """
    rng = np.random.default_rng(SEED)

    # Per-sample noise: each sample gets noise proportional to its own intensity.
    # This ensures low-energy samples (idle) aren't overwhelmed by noise scaled
    # to high-energy samples (fall), and vice versa.
    per_sample_std = X.std(axis=(1, 2), keepdims=True) * 0.02  # (N, 1, 1)
    X_noise = X + (rng.standard_normal(X.shape) * per_sample_std).astype(np.float32)

    # Time-shift augmentation (±5% of frames) with zero-padding.
    # Zero-pad instead of np.roll wrap-around to avoid artificial discontinuities
    # at window boundaries — consistent with motion_detector's zero-padded exports.
    max_shift = max(1, int(X.shape[1] * 0.05))
    shifts = rng.integers(-max_shift, max_shift + 1, size=len(X))

    def _shift_pad(x: np.ndarray, s: int) -> np.ndarray:
        """Shift time axis by s frames, zero-pad instead of wrap-around."""
        out = np.zeros_like(x)
        if s > 0:
            out[s:] = x[:-s]       # shift right → beginning = 0
        elif s < 0:
            out[:s] = x[-s:]       # shift left  → end = 0
        else:
            out[:] = x
        return out

    X_shift = np.stack([_shift_pad(x, s) for x, s in zip(X, shifts)])

    X_aug = np.concatenate([X, X_noise, X_shift], axis=0).astype(np.float32)
    y_aug = np.concatenate([y, y, y])

    # Shuffle
    idx = rng.permutation(len(X_aug))
    return X_aug[idx], y_aug[idx]


# ════════════════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════════════════

def build_model(frames: int, subcarriers: int, n_classes: int) -> tf.keras.Model:
    """
    Stacked Bidirectional LSTM with BatchNorm and Dropout.

    Architecture rationale for CSI HAR:
      - Bidirectional: captures temporal patterns in both directions
        (useful for activities that have distinct start/end signatures)
      - BatchNorm after each LSTM: stabilizes training, allows higher LR
      - Dropout(0.4): prevents overfitting on small datasets
      - 2 stacked layers: first captures local patterns,
        second captures longer-range activity structure

    Input: (batch, frames, subcarriers)
    Output: (batch, n_classes) — softmax probabilities
    """
    model = Sequential([
        Input(shape=(frames, subcarriers)),

        # ── Layer 1: extract local temporal patterns ──────────────────
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.4),

        # ── Layer 2: capture higher-level activity structure ──────────
        Bidirectional(LSTM(64, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.4),

        # ── Classifier head ───────────────────────────────────────────
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(n_classes, activation="softmax"),
    ], name="BiLSTM_CSI_HAR")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ════════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════════

def plot_history(history, output_dir: str) -> None:
    """Plot and save training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=13, fontweight="bold")

    epochs = range(1, len(history.history["accuracy"]) + 1)

    # Accuracy
    axes[0].plot(epochs, history.history["accuracy"],
                 label="Train", color="#2a9d8f", linewidth=2)
    axes[0].plot(epochs, history.history["val_accuracy"],
                 label="Validation", color="#e63946", linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.4)

    # Mark best validation accuracy — added BEFORE legend() so it appears in it
    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
    best_val   = max(history.history["val_accuracy"])
    axes[0].axvline(best_epoch, color="#f4a261", linestyle=":", linewidth=1.5,
                    label=f"Best epoch {best_epoch} ({best_val:.3f})")
    axes[0].annotate(f"{best_val:.3f}", xy=(best_epoch, best_val),
                     xytext=(5, -15), textcoords="offset points",
                     fontsize=9, color="#f4a261")
    axes[0].legend()   # after axvline → best epoch appears in legend

    # Loss
    axes[1].plot(epochs, history.history["loss"],
                 label="Train", color="#2a9d8f", linewidth=2)
    axes[1].plot(epochs, history.history["val_loss"],
                 label="Validation", color="#e63946", linewidth=2, linestyle="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    out = os.path.join(output_dir, "training_history.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Training history saved: {out}")


def plot_confusion_matrix(y_true, y_pred, classes: list[str],
                          output_dir: str) -> None:
    """Plot both raw counts and normalized confusion matrix side by side."""
    cm      = confusion_matrix(y_true, y_pred)
    # nan_to_num: if a class has 0 test samples, division → NaN → blank heatmap cell
    cm_norm = np.nan_to_num(
        cm.astype(float) / cm.sum(axis=1, keepdims=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Confusion Matrix", fontsize=13, fontweight="bold")

    for ax, data, title, fmt in [
        (axes[0], cm,      "Counts",       "d"),
        (axes[1], cm_norm, "Normalized",   ".2f"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=classes, yticklabels=classes,
                    linewidths=0.5, ax=ax, vmin=0,
                    vmax=(1.0 if fmt == ".2f" else None))
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Actual",    fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10)

    plt.tight_layout()
    out = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Confusion matrix saved: {out}")


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────
    X, y = load_data(args.dataset, args.classes, args.frames, args.subcarriers)
    n_classes = len(args.classes)

    # ── Input validation ──────────────────────────────────────────────────
    # Validate split fractions before any splitting happens
    if args.test_split <= 0 or args.test_split >= 1:
        print(f"❌ --test-split must be between 0 and 1, got {args.test_split}")
        sys.exit(1)

    val_ratio = args.val_split / (1.0 - args.test_split)
    if val_ratio <= 0 or val_ratio >= 1:
        print(f"❌ --val-split={args.val_split} combined with "
              f"--test-split={args.test_split} gives val_ratio={val_ratio:.3f}. "
              f"Ensure val_split + test_split < 1.0")
        sys.exit(1)

    # stratified split requires ≥ 2 samples per class in every split.
    # Minimum: n_classes × (1/min_fraction) samples total.
    min_per_class = 2
    min_fraction  = min(args.test_split, val_ratio, 1 - args.test_split - args.val_split)
    min_samples   = int(np.ceil(n_classes * min_per_class / min_fraction))
    if len(X) < min_samples:
        print(f"❌ Too few samples ({len(X)}) for stratified splits with "
              f"{n_classes} classes. Need at least {min_samples} samples.")
        sys.exit(1)
    # ── Three-way split: MUST happen BEFORE augmentation ─────────────────
    # If augment() runs before the val split, shuffled clones of train samples
    # can end up in validation → the model "cheats" by recognizing near-copies
    # of training samples, inflating val_accuracy artificially.
    #
    # Correct order:
    #   1. Split off test set  (held out forever)
    #   2. Split off val set   (from remaining clean data, stratified)
    #   3. Augment ONLY X_train

    # Step 1: hold out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=SEED, stratify=y
    )

    # Step 2: split val from remaining — val_ratio already computed above
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=SEED,
        stratify=y_temp
    )

    # Step 3: augment ONLY the training set — val and test stay clean
    if not args.no_augment:
        X_train, y_train = augment(X_train, y_train)
        print(f"   After augmentation: {len(X_train)} training samples")

    # ── Class weights (handles imbalanced datasets) ───────────────────────
    # CRITICAL: use np.unique(y_train) as keys, NOT enumerate index.
    # If a class is absent from y_train, enumerate gives wrong key assignments
    # (e.g. class label 2 gets key 1) → Keras trains with silently wrong weights.
    unique_classes    = np.unique(y_train)
    class_weights_arr = compute_class_weight(
        "balanced", classes=unique_classes, y=y_train
    )
    class_weight_dict = {int(c): float(w)
                         for c, w in zip(unique_classes, class_weights_arr)}
    print(f"\n   Class weights: "
          f"{', '.join(f'{args.classes[i]}={w:.2f}' for i, w in class_weight_dict.items())}")

    # ── One-hot encode all labels ─────────────────────────────────────────
    y_train_cat = to_categorical(y_train, num_classes=n_classes)
    y_val_cat   = to_categorical(y_val,   num_classes=n_classes)
    y_test_cat  = to_categorical(y_test,  num_classes=n_classes)

    print(f"   Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print("\n🔧 Building Bidirectional LSTM model...")
    model = build_model(args.frames, args.subcarriers, n_classes)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "best_model.keras")

    callbacks = [
        # Stop if val_accuracy doesn't improve for 10 epochs.
        # restore_best_weights=True restores the epoch with best val_accuracy,
        # consistent with what ModelCheckpoint also saves.
        # NOTE: both must monitor the SAME metric to avoid inconsistency —
        # EarlyStopping on val_loss + ModelCheckpoint on val_accuracy would
        # restore different epoch weights than what was saved as "best".
        EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True,
            mode="max", verbose=1
        ),
        # Save the best model by val_accuracy (same metric as EarlyStopping)
        ModelCheckpoint(
            model_path, monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=1
        ),
        # Reduce LR by 0.5 if val_loss plateaus for 5 epochs.
        # val_loss is appropriate here (more sensitive to small changes than accuracy)
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=1
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n🚀 Training  (max {args.epochs} epochs, "
          f"batch={args.batch_size}, "
          f"val_fraction={val_ratio:.1%} of train data "
          f"[≈{args.val_split:.0%} of total])...")

    history = model.fit(
        X_train, y_train_cat,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val_cat),   # explicit stratified val set
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n📊 Evaluation on test set:")
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"   Test loss    : {loss:.4f}")
    print(f"   Test accuracy: {accuracy * 100:.2f}%")

    y_pred        = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Per-class metrics (precision, recall, F1)
    print("\n📋 Per-class metrics:")
    # labels=range(n_classes) prevents ValueError if some class absent from y_test
    print(classification_report(
        y_test, y_pred_classes,
        labels=list(range(n_classes)),
        target_names=args.classes,
        digits=3
    ))

    # ── Save model ────────────────────────────────────────────────────────
    # Best model already saved by ModelCheckpoint during training.
    # Also save final model for comparison.
    final_path = os.path.join(args.output_dir, "final_model.keras")
    model.save(final_path)
    print(f"💾 Final model saved : {final_path}")
    print(f"💾 Best model saved  : {model_path}")

    # ── Save training config (reproducibility + live inference) ────────
    config = {
        "classes":       args.classes,
        "frames":        args.frames,
        "subcarriers":   args.subcarriers,
        "epochs_trained": len(history.history["accuracy"]),
        "best_epoch":    int(np.argmax(history.history["val_accuracy"])) + 1,
        "best_val_acc":  float(max(history.history["val_accuracy"])),
        "test_accuracy":  float(accuracy),
        "test_loss":      float(loss),
        "batch_size":    args.batch_size,
        "val_split":     float(args.val_split),
        "test_split":    float(args.test_split),
        "augmentation":  not args.no_augment,
        "seed":          SEED,
    }
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"💾 Training config   : {config_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_history(history, args.output_dir)
    plot_confusion_matrix(y_test, y_pred_classes, args.classes, args.output_dir)

    print("\n✅ Training complete.")
    print(f"   Best val_accuracy : "
          f"{max(history.history['val_accuracy'])*100:.2f}%  "
          f"(epoch {int(np.argmax(history.history['val_accuracy']))+1})")
    print(f"   Test accuracy     : {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()