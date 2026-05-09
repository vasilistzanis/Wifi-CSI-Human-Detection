#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSI HAR - Multi-Model Latency Benchmark
========================================
Measures end-to-end inference latency (preprocessing + PCA + features + model.predict)
for all trained classifiers. Saves results as CSV and a comparison bar chart.

Usage:
  python benchmark_latency.py --simulate
  python benchmark_latency.py --file datasets/walk_activity/walk_activity_01_vasilis_.txt --save
"""

import time
import numpy as np
import pandas as pd
import warnings
import platform
import sys
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import json
import joblib
import config

from csi_parser import configure_console_output
configure_console_output()

# ML Models (used as fallback when saved .joblib is not found)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Memory tracking
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

warnings.filterwarnings("ignore")

try:
    from data_preprocessing import CSIPipeline, load_csi_csv
    from csi_ml_pipeline import extract_features_from_window, FEATURE_VECTOR_VERSION
    import sklearn
except ImportError:
    print("Cannot import modules. Run this in the correct directory.")
    sys.exit(1)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


def get_memory_usage():
    """Returns current process memory usage in MB."""
    if not _HAS_PSUTIL:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def print_system_info():
    """Print environment info."""
    print("\n" + "=" * 60)
    print("  SYSTEM INFORMATION")
    print("=" * 60)
    print(f"  OS          : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"  CPU         : {platform.processor()}")
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  NumPy       : {np.__version__}")
    print(f"  Pandas      : {pd.__version__}")
    print(f"  Scikit-Learn: {sklearn.__version__}")
    print("=" * 60 + "\n")


# Maps display name → (joblib filename, fallback fresh model)
# Fallback models are used only when the saved .joblib file does not exist.
_MODEL_REGISTRY = {
    "Random Forest":     ("rf.joblib",  lambda seed: RandomForestClassifier(n_estimators=100, random_state=seed)),
    "SVM (RBF)":         ("svm.joblib", lambda seed: SVC(kernel='rbf', probability=True)),
    "MLP":               ("mlp.joblib", lambda seed: MLPClassifier(hidden_layer_sizes=(100,), max_iter=1, random_state=seed)),
    "K-NN":              ("knn.joblib", lambda seed: KNeighborsClassifier(n_neighbors=5)),
    "Logistic Reg":      ("lr.joblib",  lambda seed: LogisticRegression(max_iter=1000)),
    "Extra Trees":       ("et.joblib",  lambda seed: ExtraTreesClassifier(n_estimators=100, random_state=seed)),
    "Gradient Boosting": ("gb.joblib",  lambda seed: GradientBoostingClassifier(n_estimators=100)),
    "Naive Bayes":       ("nb.joblib",  lambda seed: GaussianNB()),
}


def load_or_build_models(models_dir: Path, seed: int = 42) -> dict:
    """
    Load trained models from models_dir if available.
    Falls back to a freshly-initialised model (fitted on tiny dummy data)
    only when the saved file does not exist.
    Returns {display_name: (model, source_label)} where source_label is
    'saved' or 'fallback'.
    """
    result = {}
    for name, (filename, fallback_fn) in _MODEL_REGISTRY.items():
        path = models_dir / filename
        if path.exists():
            try:
                model = joblib.load(path)
                result[name] = (model, "saved")
                print(f"  [LOAD] {name:20s} <- {filename}")
            except Exception as e:
                print(f"  [WARN] Could not load {filename}: {e} — using fallback")
                result[name] = (fallback_fn(seed), "fallback")
        else:
            result[name] = (fallback_fn(seed), "fallback")
            print(f"  [WARN] {name:20s}   {filename} not found — fallback model")
    return result


def plot_comparison(df_comp, output_path):
    """Generates a comparison bar chart."""
    plt.figure(figsize=config.BENCHMARK_FIGURE_SIZE)
    sns.set_theme(style="whitegrid")
    

    # Sort by Mean Latency for the plot
    df_plot = df_comp.copy()
    df_plot['Mean Latency (ms)'] = df_plot['Mean Latency'].str.replace(' ms', '').astype(float)
    df_plot = df_plot.sort_values('Mean Latency (ms)')
    

    ax = sns.barplot(x='Model', y='Mean Latency (ms)', data=df_plot, palette='viridis')
    

    plt.title("Latency Comparison Across ML Models", fontsize=16, fontweight='bold')
    plt.ylabel("Mean Latency (ms)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=45)
    

    # Add labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}ms', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', 
                   xytext=(0, 9), 
                   textcoords='offset points',
                   fontsize=10, fontweight='bold')
    

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[OK] Comparison plot saved to: {output_path}")


def main():
    defaults = config.get_script_defaults("benchmark_latency")
    parser = argparse.ArgumentParser(description="Multi-Model CSI Latency Benchmark")
    config.add_bool_argument(
        parser,
        dest="simulate",
        default=defaults["simulate"],
        help="Use synthetic data",
        positive_flags=["--simulate"],
        negative_flags=["--no-simulate"],
    )
    config.add_bool_argument(
        parser,
        dest="save",
        default=defaults["save"],
        help="Save results (CSV and Plot)",
        positive_flags=["--save"],
        negative_flags=["--no-save"],
    )
    parser.add_argument('--output-csv', type=str, default=defaults["output_csv"])
    parser.add_argument('--output-plot', type=str, default=defaults["output_plot"])
    parser.add_argument('--models-dir', type=str, default=defaults["models_dir"],
                        help="Directory containing saved .joblib model files (default: models)")
    parser.add_argument('--pca', type=int, default=defaults["pca"], help="Number of PCA components (default: 10)")
    parser.add_argument('--file', type=str, default=defaults["file"],
                        help="Path to real CSI data file (default: datasets/walk_activity/walk_activity_01_vasilis_.txt)")
    parser.add_argument('--start-frame', type=int, default=defaults["start_frame"],
                        help="Start frame index for benchmarking (default: 500)")
    parser.add_argument('--window-size', type=int, default=defaults["window_size"],
                        help="Number of frames per inference window (default: 50)")
    parser.add_argument('--n_warmup', type=int, default=defaults["n_warmup"], help="Warm-up runs (default: 10)")
    parser.add_argument('--n_benchmark', type=int, default=defaults["n_benchmark"], help="Benchmark runs (default: 50)")
    parser.add_argument('--seed', type=int, default=defaults["seed"], help="Random seed (default: 42)")
    args = parser.parse_args()


    print("\n" + "="*52)
    print(" CSI LATENCY BENCHMARK (Full Pipeline)")
    print("   Measurement includes: Preprocessing + PCA + Features + Model")
    print("="*52)


    np.random.seed(args.seed)
    print_system_info()

    models_dir = Path(args.models_dir)

    # -- 1. Setup Data -------------------------------------------------
    # live_predict.py uses window_size + FILTER_WARMUP frames per inference call;
    # the benchmark must match that to measure realistic latency.
    FILTER_WARMUP = config.FILTER_WARMUP   # authoritative value from config.py
    buf_size = args.window_size + FILTER_WARMUP

    pipeline_path = models_dir / "csi_pipeline.joblib"

    if args.simulate:
        print("[INFO] Mode: SIMULATE (Using synthetic waves)")
        rng = np.random.default_rng(seed=args.seed)
        fit_data = (rng.random((500, 128)) + 1j * rng.random((500, 128))).astype(np.complex64)
        window_data = fit_data[:buf_size, :]
        pipeline = CSIPipeline()
        pipeline.fit_transform(fit_data, use_pca=True, n_components=args.pca)
    elif pipeline_path.exists():
        print(f"[LOAD] Using saved pipeline: {pipeline_path}")
        pipeline = joblib.load(pipeline_path)
        data_path = Path(args.file)
        if data_path.exists():
            print(f"[FILE] Mode: REAL DATA (Loading {data_path})")
            complex_matrix, _ = load_csi_csv(data_path)
            start = min(args.start_frame,
                        max(0, complex_matrix.shape[0] - buf_size))
            window_data = complex_matrix[start:start + buf_size, :]
        else:
            print(f"[WARNING] Real data not found at {args.file}. Using random window.")
            rng = np.random.default_rng(seed=args.seed)
            n_sub = getattr(pipeline, "_fitted_n_subcarriers", 128)
            window_data = (rng.random((buf_size, n_sub))
                           + 1j * rng.random((buf_size, n_sub))).astype(np.complex64)
    else:
        print(f"[WARNING] Saved pipeline not found at {pipeline_path}. Fitting fresh pipeline.")
        print("          Run: python csi_ml_pipeline.py --classes walk_activity no_activity --save_model")
        data_path = Path(args.file)
        if data_path.exists():
            print(f"[FILE] Mode: REAL DATA (Loading {data_path})")
            complex_matrix, _ = load_csi_csv(data_path)
            fit_data = complex_matrix
            start = min(args.start_frame,
                        max(0, complex_matrix.shape[0] - buf_size))
            window_data = complex_matrix[start:start + buf_size, :]
        else:
            print(f"[WARNING] Real data not found at {args.file}. Falling back to simulation.")
            rng = np.random.default_rng(seed=args.seed)
            fit_data = (rng.random((500, 128)) + 1j * rng.random((500, 128))).astype(np.complex64)
            window_data = fit_data[:buf_size, :]
        pipeline = CSIPipeline()
        pipeline.fit_transform(fit_data, use_pca=True, n_components=args.pca)

    # Reference feature vector — mirrors the live_predict.py inference path:
    # transform the full buf_size buffer, then take the last window_size rows.
    processed_ref = pipeline.transform(window_data, use_pca=True).astype(np.float64)
    dummy_feat = extract_features_from_window(
        processed_ref[-args.window_size:]
    ).reshape(1, -1)
    X_dummy = np.tile(dummy_feat, (10, 1))
    y_dummy = np.array([0, 1] * 5)


    # -- 2. Load models ------------------------------------------------
    print(f"\n[INFO] Loading models from: {models_dir.resolve()}")
    loaded_models = load_or_build_models(models_dir, seed=args.seed)

    # -- Feature vector version check ------------------------------------
    # n_features_in_ only catches dimensional changes; a version tag also
    # catches semantic changes (kurtosis formula, fft_peak_idx scale, etc.)
    try:
        _metrics_path = models_dir / "metrics.json"
        if _metrics_path.exists():
            with open(_metrics_path, "r", encoding="utf-8") as _f:
                _saved_metrics = json.load(_f)
            _saved_versions = {
                v.get("feature_vector_version")
                for v in _saved_metrics.values()
                if v.get("feature_vector_version")
            }
            if _saved_versions and FEATURE_VECTOR_VERSION not in _saved_versions:
                print(f"\n  [WARN] *** FEATURE VECTOR VERSION MISMATCH ***")
                print(f"         Saved models were trained with version : {_saved_versions}")
                print(f"         Current pipeline feature version       : {FEATURE_VECTOR_VERSION}")
                print(f"         Model predictions will be WRONG — retrain first:")
                print(f"           python csi_ml_pipeline.py --classes walk_activity no_activity --save_model\n")
    except Exception:
        pass   # missing or malformed metrics.json — non-fatal

    # -- Validate feature-count compatibility for saved models. ----------
    # A mismatch means the model was trained before a feature-space change
    # (e.g. DWT removal).  Replace stale models with fresh fallbacks and warn.
    n_feat = dummy_feat.shape[1]
    stale_names = []
    for name, (model, source) in list(loaded_models.items()):
        if source != "saved":
            continue
        expected = getattr(model, "n_features_in_", None)
        if expected is not None and expected != n_feat:
            stale_names.append(name)
            print(f"  [STALE] {name}: saved model expects {expected} features, "
                  f"pipeline now produces {n_feat}. Using untrained fallback.")
            print(f"          → Re-train to fix: python csi_ml_pipeline.py "
                  f"--classes walk_activity no_activity --save_model")
            loaded_models[name] = (_MODEL_REGISTRY[name][1](args.seed), "fallback (stale)")

    # Warn if every model ended up as a fallback (no saved models found at all)
    n_saved = sum(1 for _, src in loaded_models.values() if src == "saved")
    if n_saved == 0:
        print("\n  [WARN] *** ALL models are fallbacks (untrained) ***")
        print("         Latency numbers reflect blank models, NOT real inference.")
        print("         Re-train first: python csi_ml_pipeline.py "
              "--classes walk_activity no_activity --save_model\n")

    # Fit fallback models on dummy data so they can predict
    for name, (model, source) in loaded_models.items():
        if source != "saved":
            model.fit(X_dummy, y_dummy)

    all_results = []
    comparison_data = []

    for name, (model, source) in loaded_models.items():
        print(f"\n[RUN] Benchmarking: {name}  [{source}]")

        mem_before = get_memory_usage()
        # Touch predict once to force any lazy initialisation before RAM snapshot
        _ = model.predict(dummy_feat)
        mem_after = get_memory_usage()

        # RAM usage estimated as the delta after the first predict
        ram_mb = max(0, mem_after - mem_before)

        # Warm-up
        for _ in range(args.n_warmup):
            proc = pipeline.transform(window_data, use_pca=True).astype(np.float64)
            feat = extract_features_from_window(proc[-args.window_size:]).reshape(1, -1)
            model.predict(feat)

        # Benchmark
        model_times = []
        for _ in range(args.n_benchmark):
            t_start = time.perf_counter()
            proc = pipeline.transform(window_data, use_pca=True).astype(np.float64)
            feat = extract_features_from_window(proc[-args.window_size:]).reshape(1, -1)
            model.predict(feat)
            t_end = time.perf_counter()

            elapsed_ms = (t_end - t_start) * 1000
            model_times.append(elapsed_ms)

            all_results.append({'model': name, 'source': source, 'latency_ms': elapsed_ms})

        avg_ms = np.mean(model_times)
        p95_ms = np.percentile(model_times, 95)
        fps = 1000.0 / avg_ms

        comparison_data.append({
            'Model': name,
            'Source': source,
            'Mean Latency': f"{avg_ms:.2f} ms",
            'p95 Latency': f"{p95_ms:.2f} ms",
            'Throughput': f"{fps:.0f} inf/sec",
            'RAM Usage': f"{ram_mb:.2f} MB"
        })
        print(f"   Mean: {avg_ms:.2f}ms | {fps:.0f} inf/sec | RAM: {ram_mb:.2f}MB ({source})")


    # -- 3. Report & Save ----------------------------------------------
    df_comp = pd.DataFrame(comparison_data)
    # Sort by Mean Latency (numeric extraction)
    df_comp['sort_val'] = df_comp['Mean Latency'].str.replace(' ms', '').astype(float)
    df_comp = df_comp.sort_values('sort_val').drop(columns=['sort_val'])
    # Reorder columns so Source appears right after Model
    col_order = ['Model', 'Source', 'Mean Latency', 'p95 Latency', 'Throughput', 'RAM Usage']
    df_comp = df_comp[[c for c in col_order if c in df_comp.columns]]
    

    print("\n" + "=" * 80)
    print(f"{'FINAL COMPARISON (Sorted by Speed)':^80}")
    print("=" * 80)
    print(df_comp.to_string(index=False))
    print("=" * 80)


    if args.save:
        # Save CSV
        df_raw = pd.DataFrame(all_results)
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(args.output_csv, index=False)
        print(f"[SAVE] Raw timings exported to: {args.output_csv}")
        

        # Save Plot
        Path(args.output_plot).parent.mkdir(parents=True, exist_ok=True)
        plot_comparison(df_comp, args.output_plot)
    else:
        print("[INFO] Results not saved (use --save to export CSV and Plot)")


if __name__ == '__main__':
    main()
