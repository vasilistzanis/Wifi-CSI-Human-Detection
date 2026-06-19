#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSI HAR - Multi-Model Latency Benchmark
========================================
Measures, for every trained classifier:
  * Inference latency per window (mean / p95 in ms)  -> matches live_predict.py path
  * Throughput (inferences / sec)
  * Model file size on disk (KB)                     -> from .joblib stat
  * Training time (s)                                -> real dataset re-fit
  * RAM delta on first predict (MB)

Outputs JSON + Markdown report into <models_dir>/benchmark/ (created if absent),
plus the legacy CSV + comparison plot when --save is given.

Usage:
  python benchmark_latency.py
  python benchmark_latency.py --file datasets/walk_activity/walk_activity_01_vasilis_.txt --save
  python benchmark_latency.py --no-training-time         # skip the heavy re-fit step

The script REQUIRES real data — a fitted CSIPipeline at <models-dir>/csi_pipeline.joblib
AND the CSI file passed via --file.  If either is missing, the script aborts with a
clear message instead of falling back to synthetic data.
"""

import time
import gc
import datetime
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
from sklearn.base import clone as sk_clone

from csi_parser import configure_console_output
configure_console_output()

# ML Models (used as fallback when saved .joblib is not found)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

try:
    from data_preprocessing import CSIPipeline, load_csi_csv
    from csi_ml_pipeline import (
        extract_features_from_window,
        FEATURE_VECTOR_VERSION,
        N_STATS as _N_STATS_DEFAULT,
        build_dataset,
    )
    import sklearn
except ImportError:
    print("Cannot import modules. Run this in the correct directory.")
    sys.exit(1)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


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


def load_or_build_models(models_dir: Path, seed: int = config.RANDOM_SEED) -> dict:
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


def get_model_size_kb(path: Path) -> float | None:
    """Return the .joblib file size in KB, or None if the file does not exist."""
    try:
        if path.exists() and path.is_file():
            return os.path.getsize(path) / 1024.0
    except OSError:
        pass
    return None


def measure_training_time(model, X_train, y_train, n_repeats: int = 1) -> float:
    """
    Re-fit a *fresh clone* of `model` on (X_train, y_train) and return the
    median wall-clock time in seconds.  Cloning resets any prior fitted state
    while preserving the saved hyperparameters, so the measurement reflects the
    same architecture that produced the saved estimator.
    """
    times = []
    for _ in range(max(1, n_repeats)):
        fresh = sk_clone(model)
        t0 = time.perf_counter()
        fresh.fit(X_train, y_train)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def write_json_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] JSON report -> {path}")


def write_markdown_report(path: Path, payload: dict) -> None:
    """Render the same `payload` as a human-readable Markdown report."""
    sysinfo = payload["system_info"]
    cfg = payload["config"]
    rows = payload["results"]

    def _fmt(v, suffix="", digits=2):
        if v is None:
            return "—"
        return f"{v:.{digits}f}{suffix}"

    feat_g = cfg.get("feature_extraction_global", {}) or {}
    budget_ms = cfg.get("budget_ms")

    lines = []
    lines.append("# CSI HAR — Multi-Model Benchmark Report")
    lines.append("")
    lines.append(f"_Generated: {payload['generated_at']}_")
    lines.append("")
    lines.append("## System")
    lines.append("")
    lines.append(f"- **OS**: {sysinfo['os']}")
    lines.append(f"- **CPU**: {sysinfo['cpu']}")
    lines.append(f"- **Python**: {sysinfo['python']}")
    lines.append(f"- **NumPy / Pandas / scikit-learn**: "
                 f"{sysinfo['numpy']} / {sysinfo['pandas']} / {sysinfo['sklearn']}")
    lines.append("")
    lines.append("## Benchmark configuration")
    lines.append("")
    lines.append(f"- **Window size**: {cfg['window_size']} frames "
                 f"(+ {cfg['filter_warmup']} warmup = {cfg['buffer_size']} buffered)")
    lines.append(f"- **Step size**: {cfg.get('step_size', '—')} frames "
                 f"@ {cfg.get('sampling_rate_hz', '—')} Hz "
                 f"⇒ real-time budget per step = **{_fmt(budget_ms, digits=1)} ms**")
    lines.append(f"- **PCA components**: {cfg['n_pca']}")
    lines.append(f"- **Stats per component**: {cfg['n_stats']}")
    lines.append(f"- **Feature dimension**: {cfg['feature_dim']}")
    lines.append(f"- **Warm-up runs**: {cfg['n_warmup']}")
    lines.append(f"- **Benchmark runs (N per model)**: {cfg['n_benchmark']}")
    lines.append(f"- **Data source**: `{cfg['data_source']}`")
    if cfg.get("train_samples") is not None:
        lines.append(f"- **Training samples used for fit timing**: "
                     f"{cfg['train_samples']} (classes: {', '.join(cfg.get('classes', []))})")
    lines.append(f"- **Feature vector version**: {cfg['feature_vector_version']}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| Μοντέλο | Πηγή | "
        "Χρόνος εκπαίδευσης<br/>(Train, s) | "
        "Μέγεθος μοντέλου<br/>(Size, KB) | "
        "Μέσος χρόνος πρόβλεψης<br/>(Inf mean, ms) | "
        "Χειρότερος χρόνος πρόβλεψης<br/>(Inf p95, ms) | "
        "Συνολικός χρόνος ανά παράθυρο<br/>(E2E p95, ms) | "
        "Χρήση προθεσμίας real-time<br/>(% budget, p95) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['source']} | "
            f"{_fmt(r['training_time_s'], digits=3)} | "
            f"{_fmt(r['model_size_kb'])} | "
            f"{_fmt(r['inference_mean_ms'])} | "
            f"{_fmt(r['inference_p95_ms'])} | "
            f"{_fmt(r['total_p95_ms'])} | "
            f"{_fmt(r['pct_budget_p95'], suffix='%')} |"
        )
    lines.append("")
    lines.append("### Notes")
    lines.append("")
    lines.append(
        f"- **Feature extraction (shared by every model)**: "
        f"mean = {_fmt(feat_g.get('mean_ms'))} ms, "
        f"p95 = {_fmt(feat_g.get('p95_ms'))} ms "
        f"(pooled over N = {feat_g.get('n_samples', '—')} window evaluations). "
        f"Includes `pipeline.transform` (filter + diff + PCA) and `extract_features_from_window` "
        f"({cfg['feature_dim']} statistical / FFT features)."
    )
    lines.append(
        f"- **End-to-end (E2E) latency** per window = feature extraction + model inference. "
        f"The **% budget** column reports `E2E p95 / {_fmt(budget_ms, digits=1)} ms × 100`, "
        f"where the budget is the time available before the next window starts "
        f"(`step_size / fs = {cfg.get('step_size', '—')} / {cfg.get('sampling_rate_hz', '—')} Hz`). "
        f"Anything < 100 % means the model meets the real-time deadline at p95."
    )
    lines.append(
        f"- **Inference (`Inf`)** isolates `model.predict(feat)` only, "
        f"so different model families can be compared head-to-head independent of preprocessing."
    )
    lines.append(
        f"- **Training time** is the median wall-clock of "
        f"`sklearn.base.clone(model).fit(X_train, y_train)` "
        f"on the real CSI training split (no augmentation)."
    )
    lines.append(
        f"- **Model size** is the `.joblib` file size on disk (KB)."
    )
    lines.append(
        f"- **Measurement**: `time.perf_counter`, single-threaded, "
        f"Python GC disabled during the timed loop, "
        f"{cfg['n_warmup']} warm-up runs preceding each {cfg['n_benchmark']}-run measurement loop."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVE] Markdown report -> {path}")


def plot_comparison(df_comp, output_path, budget_ms: float | None = None):
    """Bar chart of end-to-end p95 latency per model, with the real-time budget line."""
    plt.figure(figsize=config.BENCHMARK_FIGURE_SIZE)
    sns.set_theme(style="whitegrid")

    df_plot = df_comp.copy()
    df_plot['E2E p95 (ms)'] = df_plot['E2E p95 (ms)'].astype(float)
    df_plot = df_plot.sort_values('E2E p95 (ms)')

    ax = sns.barplot(x='Model', y='E2E p95 (ms)', data=df_plot, palette='viridis')

    plt.title("End-to-End Latency (p95) per Model", fontsize=16, fontweight='bold')
    plt.ylabel("E2E p95 latency (ms)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=45)

    if budget_ms is not None:
        ax.axhline(budget_ms, ls='--', color='red', label=f'Real-time budget = {budget_ms:.0f} ms')
        ax.legend()

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
                        help="Number of frames per inference window (default: 100)")
    parser.add_argument('--n_warmup', type=int, default=defaults["n_warmup"], help="Warm-up runs (default: 10)")
    parser.add_argument('--n_benchmark', type=int, default=defaults["n_benchmark"], help="Benchmark runs (default: 50)")
    parser.add_argument('--seed', type=int, default=defaults["seed"], help="Random seed (default: 42)")
    parser.add_argument('--features', type=int, default=defaults.get("features", _N_STATS_DEFAULT),
                        help=f"Stats per PCA component (22–{_N_STATS_DEFAULT}). Must match the trained models.")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Directory for the JSON+MD report (default: <models-dir>/benchmark)")
    config.add_bool_argument(
        parser,
        dest="measure_training_time",
        default=True,
        help="Re-fit every model on the real training split to measure training time.",
        positive_flags=["--training-time"],
        negative_flags=["--no-training-time"],
    )
    parser.add_argument('--training-classes', nargs="+", default=None,
                        help="Classes to use for the training-time re-fit "
                             "(default: TARGET_CLASSES from config.py).")
    parser.add_argument('--training-repeats', type=int, default=1,
                        help="Number of fit repeats per model (median is reported). Default: 1")
    parser.add_argument('--only', nargs="+", default=[],
                        metavar="MODEL_ID",
                        help="Whitelist: benchmark ONLY these model IDs. "
                             "Valid: rf, svm, mlp, knn, lr, et, gb, nb. "
                             "Example: --only rf svm gb")
    parser.add_argument('--skip', nargs="+", default=[],
                        metavar="MODEL_ID",
                        help="Blacklist: skip these model IDs. "
                             "Valid: rf, svm, mlp, knn, lr, et, gb, nb. "
                             "Example: --skip mlp nb. Ignored if --only is given.")
    args = parser.parse_args()

    # --only and --skip together is ambiguous; --only wins (clearer intent).
    if args.only and args.skip:
        print("[WARN] Both --only and --skip given; --only takes precedence, --skip ignored.")
        args.skip = []


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
    data_path = Path(args.file)

    # The benchmark requires BOTH a fitted pipeline and a real CSI file.
    # Synthetic-data fallbacks have been removed — fake numbers must never reach
    # the report.  Abort early with an actionable message instead.
    missing = []
    if not pipeline_path.exists():
        missing.append(
            f"fitted pipeline at {pipeline_path}\n"
            f"     -> train one with: python csi_ml_pipeline.py "
            f"--classes walk_activity no_activity --save_model"
        )
    if not data_path.exists():
        missing.append(
            f"CSI recording at {data_path}\n"
            f"     -> pass an existing file with: --file <path/to/recording.txt>"
        )
    if missing:
        print("\n[ERROR] Cannot run benchmark — required real data is missing:")
        for item in missing:
            print(f"   - {item}")
        print("\nThe --simulate fallback has been removed. Aborting.\n")
        sys.exit(2)

    print(f"[LOAD] Using saved pipeline: {pipeline_path}")
    pipeline = joblib.load(pipeline_path)
    print(f"[FILE] Mode: REAL DATA (Loading {data_path})")
    complex_matrix, _ = load_csi_csv(data_path)
    if complex_matrix.shape[0] < buf_size:
        print(f"\n[ERROR] Recording at {data_path} has only {complex_matrix.shape[0]} "
              f"frames, but the benchmark window needs {buf_size} "
              f"(window_size={args.window_size} + filter_warmup={FILTER_WARMUP}).")
        print("Pick a longer recording with --file. Aborting.\n")
        sys.exit(2)
    start = min(args.start_frame, complex_matrix.shape[0] - buf_size)
    window_data = complex_matrix[start:start + buf_size, :]

    # Reference feature vector — mirrors the live_predict.py inference path:
    # transform the full buf_size buffer, then take the last window_size rows.
    processed_ref = pipeline.transform(window_data, use_pca=True).astype(np.float64)
    dummy_feat = extract_features_from_window(
        processed_ref[-args.window_size:], fs=pipeline.fs, cutoff_hz=pipeline.cutoff,
        n_stats=args.features,
    ).reshape(1, -1)
    X_dummy = np.tile(dummy_feat, (10, 1))
    y_dummy = np.array([0, 1] * 5)


    # -- 2. Load models ------------------------------------------------
    print(f"\n[INFO] Loading models from: {models_dir.resolve()}")
    loaded_models = load_or_build_models(models_dir, seed=args.seed)

    # ---- Model selection filter (--only / --skip) --------------------
    # Match a CLI ID against the .joblib filename stem of each registry entry
    # (e.g. "Random Forest" -> "rf", "SVM (RBF)" -> "svm").
    if args.only or args.skip:
        valid_ids = {Path(fn).stem.lower() for fn, _ in _MODEL_REGISTRY.values()}
        if args.only:
            only_set = {s.strip().lower() for s in args.only}
            bad = only_set - valid_ids
            if bad:
                print(f"[ERROR] Unknown model IDs in --only: {sorted(bad)}. "
                      f"Valid: {sorted(valid_ids)}. Aborting.")
                sys.exit(2)
            kept, skipped = {}, []
            for name, payload in loaded_models.items():
                stem = Path(_MODEL_REGISTRY[name][0]).stem.lower()
                if stem in only_set:
                    kept[name] = payload
                else:
                    skipped.append(f"{name} (id={stem})")
            print(f"  [ONLY] Whitelisted: {sorted(only_set)}")
            if skipped:
                print(f"         Excluded: {', '.join(skipped)}")
        else:
            skip_set = {s.strip().lower() for s in args.skip}
            bad = skip_set - valid_ids
            if bad:
                print(f"[ERROR] Unknown model IDs in --skip: {sorted(bad)}. "
                      f"Valid: {sorted(valid_ids)}. Aborting.")
                sys.exit(2)
            kept, skipped = {}, []
            for name, payload in loaded_models.items():
                stem = Path(_MODEL_REGISTRY[name][0]).stem.lower()
                if stem in skip_set:
                    skipped.append(f"{name} (id={stem})")
                else:
                    kept[name] = payload
            if skipped:
                print(f"  [SKIP] Excluded by --skip: {', '.join(skipped)}")

        loaded_models = kept
        if not loaded_models:
            print("[ERROR] No models left to benchmark after filtering. Aborting.")
            sys.exit(2)

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
    for name, (model, source) in list(loaded_models.items()):
        if source != "saved":
            continue
        expected = getattr(model, "n_features_in_", None)
        if expected is not None and expected != n_feat:
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

    # -- 2b. Load REAL training data for training-time measurement -------
    # Re-uses csi_ml_pipeline.build_dataset so the train split matches what
    # produced the saved .joblib files (same window/step/PCA/augmentation).
    X_train_real = None
    y_train_real = None
    training_classes = args.training_classes or list(config.TARGET_CLASSES)
    if args.measure_training_time:
        print(f"\n[INFO] Building training split for fit-time measurement "
              f"(classes: {training_classes})")
        try:
            (_Xa, X_train_real, _Xt, _ya, y_train_real, _yt,
             _grp, _le, _pipe, _info) = build_dataset(
                data_dir=config.DATASETS_DIR,
                classes=training_classes,
                pipeline_kwargs={'fs': config.SAMPLING_RATE, 'use_diff': True},
                window_size=args.window_size,
                step=config.PIPELINE_STEP_SIZE,
                augment_techniques=None,         # no augmentation -> deterministic fit time
                n_augments=0,
                simulation_mode=False,
                test_recording_ratio=config.TEST_RATIO,
                random_seed=args.seed,
                n_pca=args.pca,
                cutoff=config.FILTER_CUTOFF_HZ,
                n_stats=args.features,
            )
            print(f"   Train samples: {X_train_real.shape[0]} | "
                  f"Feature dim: {X_train_real.shape[1]} | "
                  f"Classes: {sorted(set(y_train_real.tolist()))}")
        except Exception as e:
            print(f"  [WARN] Could not build real training split ({e}). "
                  "Training time will be reported as null.")
            X_train_real = None
            y_train_real = None

    # ---- Real-time budget (constant across models) --------------------
    # The window advances by PIPELINE_STEP_SIZE frames each inference step,
    # so the *budget* per step is step / fs seconds.  Anything that fits in
    # it under the worst case (p95) is real-time on this hardware.
    step_size = config.PIPELINE_STEP_SIZE
    fs_hz = config.SAMPLING_RATE
    budget_ms = (step_size / fs_hz) * 1000.0
    print(f"\n[INFO] Real-time budget per step: {budget_ms:.1f} ms "
          f"(step={step_size} frames @ {fs_hz:.0f} Hz)")

    all_results = []
    comparison_data = []
    report_rows = []          # structured rows for JSON+MD report
    feat_times_global = []    # pooled across models (feat work is model-agnostic)

    for name, (model, source) in loaded_models.items():
        print(f"\n[RUN] Benchmarking: {name}  [{source}]")

        # ---- Model size on disk (KB) ----------------------------------
        joblib_path = models_dir / _MODEL_REGISTRY[name][0]
        model_size_kb = get_model_size_kb(joblib_path) if source == "saved" else None
        if model_size_kb is not None:
            print(f"   Size on disk: {model_size_kb:.2f} KB  ({joblib_path.name})")

        # ---- Training time (s) on the REAL train split ----------------
        training_time_s = None
        if (args.measure_training_time and X_train_real is not None
                and y_train_real is not None):
            try:
                training_time_s = measure_training_time(
                    model, X_train_real, y_train_real, n_repeats=args.training_repeats,
                )
                print(f"   Training time: {training_time_s:.3f} s "
                      f"(median of {args.training_repeats} re-fit(s))")
            except Exception as e:
                print(f"  [WARN] Training-time measurement failed for {name}: {e}")

        # Auto-detect n_stats from this model (supports 22–N_STATS features)
        _n_pca_bench = getattr(getattr(pipeline, 'pca', None), 'n_components_', None)
        if _n_pca_bench and hasattr(model, 'n_features_in_'):
            _n_stats_bench = model.n_features_in_ // _n_pca_bench
        else:
            _n_stats_bench = _N_STATS_DEFAULT

        # ---- Warm-up: run the FULL pipeline + predict path -------------
        # Force any lazy initialisation (sklearn lazy attrs, BLAS thread pool,
        # filter coefficients) before the timed loop.
        for _ in range(args.n_warmup):
            proc = pipeline.transform(window_data, use_pca=True).astype(np.float64)
            feat = extract_features_from_window(
                proc[-args.window_size:], fs=pipeline.fs, cutoff_hz=pipeline.cutoff,
                n_stats=_n_stats_bench,
            ).reshape(1, -1)
            model.predict(feat)

        # ---- Measurement loop: feat / inference / total separately -----
        # Disable GC so collection pauses don't poison the p95 tail.
        feat_times = []
        pred_times = []
        total_times = []
        gc.collect()
        gc.disable()
        try:
            for _ in range(args.n_benchmark):
                t0 = time.perf_counter()
                proc = pipeline.transform(window_data, use_pca=True).astype(np.float64)
                feat = extract_features_from_window(
                    proc[-args.window_size:], fs=pipeline.fs, cutoff_hz=pipeline.cutoff,
                    n_stats=_n_stats_bench,
                ).reshape(1, -1)
                t1 = time.perf_counter()
                model.predict(feat)
                t2 = time.perf_counter()

                feat_ms  = (t1 - t0) * 1000.0
                pred_ms  = (t2 - t1) * 1000.0
                total_ms = (t2 - t0) * 1000.0
                feat_times.append(feat_ms)
                pred_times.append(pred_ms)
                total_times.append(total_ms)
                all_results.append({
                    'model': name, 'source': source,
                    'feature_extraction_ms': feat_ms,
                    'inference_ms': pred_ms,
                    'total_ms': total_ms,
                })
        finally:
            gc.enable()

        feat_times_global.extend(feat_times)

        inf_mean   = float(np.mean(pred_times))
        inf_p95    = float(np.percentile(pred_times, 95))
        total_mean = float(np.mean(total_times))
        total_p95  = float(np.percentile(total_times, 95))
        pct_budget = (total_p95 / budget_ms) * 100.0

        comparison_data.append({
            'Model': name,
            'Source': source,
            'Χρόνος εκπαίδευσης (Train, s)':       f"{training_time_s:.3f}" if training_time_s is not None else "—",
            'Μέγεθος μοντέλου (Size, KB)':         f"{model_size_kb:.2f}" if model_size_kb is not None else "—",
            'Μέσος χρόνος πρόβλεψης (Inf mean, ms)':       f"{inf_mean:.2f}",
            'Χειρότερος χρόνος πρόβλεψης (Inf p95, ms)':   f"{inf_p95:.2f}",
            'Συνολικός χρόνος ανά παράθυρο (E2E p95, ms)': f"{total_p95:.2f}",
            'Χρήση προθεσμίας real-time (% budget, p95)':  f"{pct_budget:.2f}%",
        })
        report_rows.append({
            'model': name,
            'source': source,
            'training_time_s': (round(training_time_s, 4)
                                if training_time_s is not None else None),
            'model_size_kb':   (round(model_size_kb, 3)
                                if model_size_kb is not None else None),
            'inference_mean_ms': round(inf_mean, 4),
            'inference_p95_ms':  round(inf_p95, 4),
            'feat_mean_ms':      round(float(np.mean(feat_times)), 4),
            'feat_p95_ms':       round(float(np.percentile(feat_times, 95)), 4),
            'total_mean_ms':     round(total_mean, 4),
            'total_p95_ms':      round(total_p95, 4),
            'pct_budget_p95':    round(pct_budget, 3),
            'joblib_path':       str(joblib_path) if source == "saved" else None,
        })
        print(f"   Inf:   mean={inf_mean:.2f}ms  p95={inf_p95:.2f}ms")
        print(f"   E2E:   mean={total_mean:.2f}ms  p95={total_p95:.2f}ms  "
              f"= {pct_budget:.2f}% of {budget_ms:.0f}ms budget")


    # -- 3. Report & Save ----------------------------------------------
    df_comp = pd.DataFrame(comparison_data)
    # Sort by E2E p95 latency (most relevant for real-time)
    e2e_col = 'Συνολικός χρόνος ανά παράθυρο (E2E p95, ms)'
    df_comp['sort_val'] = df_comp[e2e_col].astype(float)
    df_comp = df_comp.sort_values('sort_val').drop(columns=['sort_val'])
    col_order = [
        'Model', 'Source',
        'Χρόνος εκπαίδευσης (Train, s)',
        'Μέγεθος μοντέλου (Size, KB)',
        'Μέσος χρόνος πρόβλεψης (Inf mean, ms)',
        'Χειρότερος χρόνος πρόβλεψης (Inf p95, ms)',
        'Συνολικός χρόνος ανά παράθυρο (E2E p95, ms)',
        'Χρήση προθεσμίας real-time (% budget, p95)',
    ]
    df_comp = df_comp[[c for c in col_order if c in df_comp.columns]]

    # Sort the structured rows the same way so JSON/MD match the printed table.
    report_rows.sort(key=lambda r: r['total_p95_ms'])

    print("\n" + "=" * 110)
    print(f"{'FINAL COMPARISON (sorted by E2E p95)':^110}")
    print("=" * 110)
    print(df_comp.to_string(index=False))
    print("=" * 110)

    # -- 3a. Global feature-extraction stats (shared by every model) ----
    feat_global_mean = float(np.mean(feat_times_global)) if feat_times_global else None
    feat_global_p95  = float(np.percentile(feat_times_global, 95)) if feat_times_global else None
    if feat_global_mean is not None:
        print(f"\n[INFO] Feature extraction (shared by all models): "
              f"mean={feat_global_mean:.2f}ms  p95={feat_global_p95:.2f}ms "
              f"(N={len(feat_times_global)} pooled samples)")

    # -- 3b. Always emit JSON + Markdown report into models/benchmark/ --
    output_dir = Path(args.output_dir) if args.output_dir else (models_dir / "benchmark")
    payload = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "system_info": {
            "os": f"{platform.system()} {platform.release()} ({platform.architecture()[0]})",
            "cpu": platform.processor(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "config": {
            "window_size": args.window_size,
            "filter_warmup": FILTER_WARMUP,
            "buffer_size": buf_size,
            "step_size": step_size,
            "sampling_rate_hz": fs_hz,
            "budget_ms": round(budget_ms, 3),
            "n_pca": args.pca,
            "n_stats": args.features,
            "feature_dim": int(dummy_feat.shape[1]),
            "n_warmup": args.n_warmup,
            "n_benchmark": args.n_benchmark,
            "data_source": str(data_path),
            "training_repeats": args.training_repeats,
            "train_samples": (int(X_train_real.shape[0])
                              if X_train_real is not None else None),
            "classes": training_classes,
            "feature_vector_version": FEATURE_VECTOR_VERSION,
            "seed": args.seed,
            "feature_extraction_global": {
                "mean_ms": round(feat_global_mean, 4) if feat_global_mean is not None else None,
                "p95_ms":  round(feat_global_p95, 4) if feat_global_p95 is not None else None,
                "n_samples": len(feat_times_global),
            },
        },
        "results": report_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_report(output_dir / "benchmark_results.json", payload)
    write_markdown_report(output_dir / "benchmark_results.md", payload)

    if args.save:
        # Save CSV (legacy raw timings)
        df_raw = pd.DataFrame(all_results)
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(args.output_csv, index=False)
        print(f"[SAVE] Raw timings CSV   -> {args.output_csv}")

        # Save Plot
        Path(args.output_plot).parent.mkdir(parents=True, exist_ok=True)
        plot_comparison(df_comp, args.output_plot, budget_ms=budget_ms)
    else:
        print("[INFO] CSV + plot skipped (use --save to also export them).")


if __name__ == '__main__':
    main()
