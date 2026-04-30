#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Explainable AI (XAI) - Model Feature Importance Analysis
=========================================================
Generates publication-ready figures explaining which CSI features
the trained ML model relies on for Human Activity Recognition.

Methods:
  1. Permutation Importance (model-agnostic, works on ANY model)
  2. Built-in Feature Importance (tree-based models: RF, ET, GB)
  3. Per-Class Permutation Importance breakdown
  4. Feature Group Importance (aggregate by statistical category)

Usage:
  python explain_model_characteristics.py
  python explain_model_characteristics.py --model svm --top 15
  python explain_model_characteristics.py --model rf --save
  python explain_model_characteristics.py --simulate
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np

# -- Console safety --------------------------------------------------------
def configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass

configure_console_output()

import matplotlib
try:
    matplotlib.use("Qt5Agg")
except Exception:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

import matplotlib.pyplot as plt
plt.ioff()

# -- Local imports ---------------------------------------------------------
try:
    from csi_ml_pipeline import (
        build_dataset, _get_feature_names, N_STATS,
        ALL_AUGMENT_TECHNIQUES,
    )
    _PIPELINE_OK = True
except ImportError:
    _PIPELINE_OK = False
    print("[ERROR] csi_ml_pipeline.py not found. Cannot proceed.")
    sys.exit(1)


# ========================================================================
# STYLE  (matches all_plot_figures.py for thesis consistency)
# ========================================================================

STYLE = {
    "bg":       "#ffffff",
    "panel":    "#fafafa",
    "text":     "#1a1a1a",
    "grid":     "#e0e0e0",
    "accent1":  "#2563eb",   # blue
    "accent2":  "#f59e0b",   # amber
    "accent3":  "#10b981",   # green
    "accent4":  "#ef4444",   # red
    "accent5":  "#8b5cf6",   # purple
    "accent6":  "#06b6d4",   # cyan
}

PALETTE = [STYLE["accent1"], STYLE["accent2"], STYLE["accent3"],
           STYLE["accent4"], STYLE["accent5"], STYLE["accent6"]]

# Feature group colors for the aggregate chart
GROUP_COLORS = {
    "Statistical":  "#2563eb",
    "FFT":          "#f59e0b",
    "DWT":          "#10b981",
    "Other":        "#8b5cf6",
}


def _apply_style():
    for style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot']:
        try:
            plt.style.use(style)
            break
        except Exception:
            continue
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.facecolor":    STYLE["panel"],
        "figure.facecolor":  STYLE["bg"],
        "axes.grid":         True,
        "grid.alpha":        0.4,
        "grid.linewidth":    0.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def _save_fig(fig, save_dir: Path, name: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=STYLE["bg"])
    print(f"  [SAVE] {out}")


def _center_figure(fig):
    """Attempt to center the matplotlib window on the screen."""
    try:
        manager = fig.canvas.manager
        backend = matplotlib.get_backend()
        
        if backend == 'TkAgg':
            manager.window.eval('tk::PlaceWindow . center')
        elif 'Qt' in backend:
            # For Qt5/Qt6 backends
            try:
                # Try PyQt5
                from PyQt5.QtWidgets import QDesktopWidget
                qr = manager.window.frameGeometry()
                cp = QDesktopWidget().availableGeometry().center()
                qr.moveCenter(cp)
                manager.window.move(qr.topLeft())
            except Exception:
                pass
    except Exception:
        pass


# ========================================================================
# FEATURE GROUPING HELPERS
# ========================================================================

# Mapping from stat suffix -> human-readable group
_STAT_TO_GROUP = {
    'mean': 'Statistical', 'std': 'Statistical', 'max': 'Statistical',
    'min': 'Statistical', 'range': 'Statistical', 'median': 'Statistical',
    'energy': 'Statistical', 'skewness': 'Statistical', 'kurtosis': 'Statistical',
    'zcr': 'Statistical',
    'fft_mean': 'FFT', 'fft_std': 'FFT',
    'fft_peak_idx': 'FFT', 'spectral_entropy': 'FFT',
    'dwt_d1_energy': 'DWT', 'dwt_d1_std': 'DWT',
    'dwt_d2_energy': 'DWT', 'dwt_d2_std': 'DWT',
    'dwt_d3_energy': 'DWT', 'dwt_d3_std': 'DWT',
}


def _classify_feature(feature_name: str) -> str:
    """Classify a feature name like 'PC3_spectral_entropy' into its group."""
    # Strip the PC prefix (e.g. 'PC1_' -> '')
    parts = feature_name.split("_", 1)
    if len(parts) == 2 and parts[0].startswith("PC"):
        stat_name = parts[1]
    else:
        stat_name = feature_name

    return _STAT_TO_GROUP.get(stat_name, "Other")


def _aggregate_group_importance(feature_names, importances):
    """Sum importance per feature group (Statistical / FFT / DWT)."""
    groups = {}
    for name, imp in zip(feature_names, importances):
        g = _classify_feature(name)
        groups[g] = groups.get(g, 0.0) + imp
    return groups


# ========================================================================
# PLOT 1 - PERMUTATION IMPORTANCE (TOP N)
# ========================================================================

def plot_permutation_importance(
    model, X_test, y_test, feature_names,
    model_name: str, top_n: int = 15,
    save_dir: Path = None, save: bool = False,
    n_repeats: int = 10, random_state: int = 42,
):
    """
    Model-agnostic feature importance via sklearn.inspection.permutation_importance.
    Works on ANY fitted model (SVM, RF, KNN, LR, MLP, etc.).
    """
    from sklearn.inspection import permutation_importance

    print(f"\n[INFO] Computing Permutation Importance for {model_name}...")
    print(f"   Test samples: {X_test.shape[0]} | Features: {X_test.shape[1]} | Repeats: {n_repeats}")

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring='accuracy',
    )

    # Sort by mean importance (descending)
    sorted_idx = result.importances_mean.argsort()[::-1][:top_n]

    names = [feature_names[i] if i < len(feature_names) else f"feat_{i}"
             for i in sorted_idx]
    means = result.importances_mean[sorted_idx]
    stds  = result.importances_std[sorted_idx]

    # Assign colors by group
    colors = [GROUP_COLORS.get(_classify_feature(n), STYLE["accent1"]) for n in names]

    fig, ax = plt.subplots(figsize=(12, 6.5))

    bars = ax.barh(
        range(len(names)), means, xerr=stds,
        color=colors, edgecolor="white", linewidth=0.5,
        capsize=3, error_kw={"linewidth": 1, "color": "#555555"},
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Accuracy Decrease", fontweight="bold", fontsize=12)
    ax.set_title(
        f"Permutation Feature Importance - {model_name}\n"
        f"Top {top_n} Features (n_repeats={n_repeats})",
        fontweight="bold", fontsize=13,
    )

    # Legend for groups
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()
                       if any(_classify_feature(n) == g for n in names)]
    if legend_elements:
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
                  framealpha=0.9)

    ax.axvline(0, color=STYLE["grid"], linewidth=1, zorder=0)
    fig.tight_layout()
    _center_figure(fig)

    if save and save_dir:
        _save_fig(fig, save_dir, f"xai_01_permutation_importance_{model_name.lower().replace(' ', '_')}")

    # Print summary
    print(f"\n   Top {min(5, len(names))} Most Important Features:")
    for i in range(min(5, len(names))):
        print(f"     {i+1}. {names[i]:35s}  {means[i]:.4f} +/- {stds[i]:.4f}")

    return fig, result


# ========================================================================
# PLOT 2 - BUILT-IN FEATURE IMPORTANCE (TREE MODELS ONLY)
# ========================================================================

def plot_builtin_importance(
    model, feature_names, model_name: str,
    top_n: int = 15, save_dir: Path = None, save: bool = False,
):
    """
    Uses .feature_importances_ (Gini / MDI).
    Only available for tree-based models: RandomForest, ExtraTrees, GradientBoosting.
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"  [SKIP] {model_name} does not have built-in feature_importances_ (not tree-based)")
        return None

    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1][:top_n]

    names = [feature_names[i] if i < len(feature_names) else f"feat_{i}"
             for i in sorted_idx]
    values = importances[sorted_idx]

    colors = [GROUP_COLORS.get(_classify_feature(n), STYLE["accent1"]) for n in names]

    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.barh(
        range(len(names)), values,
        color=colors, edgecolor="white", linewidth=0.5,
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Gini Importance (MDI)", fontweight="bold", fontsize=12)
    ax.set_title(
        f"Built-in Feature Importance (MDI) - {model_name}\n"
        f"Top {top_n} Features",
        fontweight="bold", fontsize=13,
    )

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()
                       if any(_classify_feature(n) == g for n in names)]
    if legend_elements:
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
                  framealpha=0.9)

    fig.tight_layout()
    _center_figure(fig)

    if save and save_dir:
        _save_fig(fig, save_dir, f"xai_02_builtin_importance_{model_name.lower().replace(' ', '_')}")

    return fig


# ========================================================================
# PLOT 3 - FEATURE GROUP IMPORTANCE (AGGREGATE PIE / BAR)
# ========================================================================

def plot_group_importance(
    model, X_test, y_test, feature_names, model_name: str,
    save_dir: Path = None, save: bool = False,
    n_repeats: int = 10, random_state: int = 42,
):
    """
    Aggregate feature importance by category: Statistical, FFT, DWT.
    Shows which signal processing domain the model relies on most.
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, random_state=random_state,
        n_jobs=-1, scoring='accuracy',
    )

    # Clip negative importances to 0 for aggregation
    clipped = np.clip(result.importances_mean, 0, None)
    groups = _aggregate_group_importance(feature_names, clipped)

    # Sort by importance
    sorted_groups = sorted(groups.items(), key=lambda x: x[1], reverse=True)
    group_names = [g[0] for g in sorted_groups]
    group_vals  = [g[1] for g in sorted_groups]
    total = sum(group_vals) if sum(group_vals) > 0 else 1.0
    group_pcts = [v / total * 100 for v in group_vals]
    colors = [GROUP_COLORS.get(g, STYLE["accent5"]) for g in group_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.5),
                                    gridspec_kw={"width_ratios": [1, 1.2]})

    # Left: Pie chart
    wedges, texts, autotexts = ax1.pie(
        group_pcts, labels=group_names, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={"fontsize": 12, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_color("white")
        t.set_fontweight("bold")
    ax1.set_title("Feature Domain Contribution", fontweight="bold", fontsize=13)

    # Right: Horizontal bar
    ax2.barh(
        range(len(group_names)), group_pcts,
        color=colors, edgecolor="white", linewidth=1,
        height=0.5,
    )
    ax2.set_yticks(range(len(group_names)))
    ax2.set_yticklabels(group_names, fontsize=12, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlabel("Relative Importance (%)", fontweight="bold", fontsize=12)
    ax2.set_title("Importance by Signal Domain", fontweight="bold", fontsize=13)

    # Annotate bars with percentages
    for i, (pct, val) in enumerate(zip(group_pcts, group_vals)):
        ax2.text(pct + 0.5, i, f"{pct:.1f}%", va="center", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Feature Group Analysis - {model_name}",
        fontweight="bold", fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _center_figure(fig)

    if save and save_dir:
        _save_fig(fig, save_dir, f"xai_03_group_importance_{model_name.lower().replace(' ', '_')}")

    # Print summary
    print(f"\n   Feature Group Breakdown ({model_name}):")
    for name, pct in zip(group_names, group_pcts):
        bar = "#" * int(pct / 2)
        print(f"     {name:15s}  {pct:5.1f}%  {bar}")

    return fig


# ========================================================================
# PLOT 4 - PER-CLASS IMPORTANCE
# ========================================================================

def plot_per_class_importance(
    model, X_test, y_test, feature_names, class_names,
    model_name: str, top_n: int = 10,
    save_dir: Path = None, save: bool = False,
    n_repeats: int = 10, random_state: int = 42,
):
    """
    Compute permutation importance separately for each class.
    Shows which features are most discriminative for each activity.
    """
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import make_scorer, accuracy_score

    n_classes = len(class_names)
    if n_classes < 2:
        print("  [SKIP] Per-class importance requires at least 2 classes")
        return None

    cols = min(n_classes, 3)
    rows = (n_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6.5))

    # Flatten axes for easy indexing
    if n_classes == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for cls_idx, cls_name in enumerate(class_names):
        ax = axes[cls_idx]

        # Create a binary scorer: accuracy only for this class
        mask = y_test == cls_idx
        if mask.sum() < 2:
            ax.text(0.5, 0.5, f"Not enough\nsamples for\n{cls_name}",
                    ha="center", va="center", fontsize=12, transform=ax.transAxes)
            ax.set_title(cls_name, fontweight="bold")
            continue

        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats, random_state=random_state,
            n_jobs=-1, scoring='accuracy',
        )

        sorted_idx = result.importances_mean.argsort()[::-1][:top_n]
        names  = [feature_names[i] if i < len(feature_names) else f"feat_{i}"
                  for i in sorted_idx]
        means  = result.importances_mean[sorted_idx]
        colors = [GROUP_COLORS.get(_classify_feature(n), STYLE["accent1"]) for n in names]

        ax.barh(range(len(names)), means, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Importance", fontsize=9)
        ax.set_title(f"{cls_name}", fontweight="bold", fontsize=12,
                     color=PALETTE[cls_idx % len(PALETTE)])

    # Hide unused axes
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f"Per-Class Feature Importance - {model_name}\n"
        f"(Permutation-based, Top {top_n})",
        fontweight="bold", fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _center_figure(fig)

    if save and save_dir:
        _save_fig(fig, save_dir, f"xai_04_per_class_{model_name.lower().replace(' ', '_')}")

    return fig


# ========================================================================
# MAIN
# ========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="XAI - Explain Model Feature Importance for CSI HAR"
    )
    p.add_argument("--model", type=str, default="rf",
                   choices=["svm", "rf", "et", "knn", "lr", "gb", "mlp", "nb", "all"],
                   help="Which model to explain (default: rf)")
    p.add_argument("--models_dir", type=str, default="./models",
                   help="Directory containing saved .joblib models")
    p.add_argument("--data_dir", type=str, default="./datasets",
                   help="Dataset directory")
    p.add_argument("--classes", nargs="+", default=["walk", "idle"],
                   help="Activity classes")
    p.add_argument("--top", type=int, default=15,
                   help="Number of top features to show (default: 15)")
    p.add_argument("--repeats", type=int, default=10,
                   help="Permutation repeats (default: 10, more = slower but more stable)")
    p.add_argument("--save", action="store_true",
                   help="Save figures as PNG (300 DPI)")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory for saved figures (default: models/plots/)")
    p.add_argument("--simulate", action="store_true",
                   help="Use synthetic data (no real dataset needed)")
    p.add_argument("--window_size", type=int, default=50)
    p.add_argument("--step", type=int, default=25)
    p.add_argument("--pca", type=int, default=10)
    p.add_argument("--fs", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cutoff", type=float, default=10.0)
    return p.parse_args()


def _load_model_from_disk(models_dir: Path, model_key: str):
    """Try to load a pre-trained model from disk."""
    import joblib

    safe_key = model_key.replace(" ", "_").replace("(", "").replace(")", "")
    candidates = [
        models_dir / f"{safe_key}.joblib",
        models_dir / f"{model_key}.joblib",
    ]

    for path in candidates:
        if path.exists():
            print(f"  [LOAD] {path}")
            return joblib.load(path)

    return None


def main():
    args = parse_args()
    _apply_style()

    models_dir = Path(args.models_dir)
    save_dir = Path(args.out_dir) if args.out_dir else models_dir / "plots"

    print("=" * 60)
    print(" XAI - Explainable AI for CSI HAR")
    print("=" * 60)
    print(f"  Model  : {args.model}")
    print(f"  Classes: {args.classes}")
    print(f"  Top N  : {args.top}")
    print(f"  Repeats: {args.repeats}")
    print(f"  Save   : {args.save}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # STEP 1: Build dataset to get X_test, y_test and feature names
    # ----------------------------------------------------------------
    print("\n[STEP 1] Building dataset...")

    (X_train, X_train_orig, X_test,
     y_train, y_train_orig, y_test,
     train_groups_orig, le, pipeline) = build_dataset(
        data_dir=args.data_dir,
        classes=args.classes,
        pipeline_kwargs={'fs': args.fs, 'use_diff': True},
        window_size=args.window_size,
        step=args.step,
        augment_techniques=[],      # No augmentation for XAI
        n_augments=0,
        simulation_mode=args.simulate,
        test_recording_ratio=0.2,
        random_seed=args.seed,
        n_pca=args.pca,
        cutoff=args.cutoff,
    )

    if X_test.shape[0] < 5:
        print("[ERROR] Not enough test samples for reliable importance estimation.")
        print("   Need at least 5 test samples. Check your data directory.")
        sys.exit(1)

    n_pca = X_test.shape[1] // N_STATS
    feature_names = _get_feature_names(n_pca)
    class_names = list(le.classes_)

    print(f"  Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"  PCA components: {n_pca} | Features per component: {N_STATS}")
    print(f"  Classes: {class_names}")

    # ----------------------------------------------------------------
    # STEP 2: Load or train model(s)
    # ----------------------------------------------------------------
    print("\n[STEP 2] Loading model(s)...")

    models_to_explain = {}

    if args.model == 'all':
        model_keys = ['svm', 'rf', 'et', 'knn', 'lr', 'gb', 'mlp', 'nb']
    else:
        model_keys = [args.model]

    for key in model_keys:
        loaded = _load_model_from_disk(models_dir, key)
        if loaded is not None:
            models_to_explain[key] = loaded
        else:
            print(f"  [INFO] {key}.joblib not found in {models_dir} - will train fresh")

    # If no models loaded from disk, train fresh
    if not models_to_explain:
        print("\n  [INFO] No saved models found. Training fresh models for XAI analysis...")
        from csi_ml_pipeline import train_and_evaluate
        results = train_and_evaluate(
            X_train, X_train_orig, X_test,
            y_train, y_train_orig, y_test,
            train_groups_orig, le, best_params=None,
            random_seed=args.seed,
            target_model=args.model,
        )
        for name, res in results.items():
            models_to_explain[name] = res['model']

    if not models_to_explain:
        print("[ERROR] No models available for analysis.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # STEP 3: Generate XAI plots for each model
    # ----------------------------------------------------------------
    all_figs = []

    for model_name, model in models_to_explain.items():
        display_name = model_name.upper()
        print(f"\n{'='*60}")
        print(f" Analyzing: {display_name}")
        print(f"{'='*60}")

        # Plot 1: Permutation Importance
        fig1, perm_result = plot_permutation_importance(
            model, X_test, y_test, feature_names,
            model_name=display_name, top_n=args.top,
            save_dir=save_dir, save=args.save,
            n_repeats=args.repeats, random_state=args.seed,
        )
        all_figs.append(fig1)

        # Plot 2: Built-in importance (tree models only)
        fig2 = plot_builtin_importance(
            model, feature_names, model_name=display_name,
            top_n=args.top, save_dir=save_dir, save=args.save,
        )
        if fig2:
            all_figs.append(fig2)

        # Plot 3: Feature group breakdown
        fig3 = plot_group_importance(
            model, X_test, y_test, feature_names,
            model_name=display_name,
            save_dir=save_dir, save=args.save,
            n_repeats=args.repeats, random_state=args.seed,
        )
        all_figs.append(fig3)

        # Plot 4: Per-class importance (only if > 1 class)
        if len(class_names) >= 2:
            fig4 = plot_per_class_importance(
                model, X_test, y_test, feature_names, class_names,
                model_name=display_name, top_n=min(args.top, 10),
                save_dir=save_dir, save=args.save,
                n_repeats=args.repeats, random_state=args.seed,
            )
            if fig4:
                all_figs.append(fig4)

    # ----------------------------------------------------------------
    # STEP 4: Summary
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" XAI ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Models analyzed: {list(models_to_explain.keys())}")
    print(f"  Figures generated: {len(all_figs)}")
    if args.save:
        print(f"  Saved to: {save_dir}")
    print(f"{'='*60}")

    # Show all figures
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
