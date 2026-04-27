import time
import numpy as np
import pandas as pd
import warnings
import platform
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ML Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Memory tracking
try:
    import psutil
    import os
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

warnings.filterwarnings("ignore")

try:
    from data_preprocessing import CSIPipeline, load_csi_csv
    from csi_ml_pipeline import extract_features_from_window
    import sklearn
except ImportError:
    print("Cannot import modules. Run this in the correct directory.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

N_WARMUP    = 10    # warm-up runs per model
N_BENCHMARK = 50    # runs per model
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

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

def get_models():
    """Returns a dictionary of models to benchmark."""
    return {
        "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "SVM (RBF)":          SVC(kernel='rbf', probability=True),
        "MLP (100,50)":       MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1),
        "K-NN (k=5)":         KNeighborsClassifier(n_neighbors=5),
        "Logistic Reg":       LogisticRegression(max_iter=1000),
        "Extra Trees":        ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=100),
        "Naive Bayes":        GaussianNB()
    }

def plot_comparison(df_comp, output_path):
    """Generates a comparison bar chart."""
    plt.figure(figsize=(12, 6))
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
    print(f"📊 Comparison plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Model CSI Latency Benchmark")
    parser.add_argument('--simulate', action='store_true', help="Use synthetic data")
    parser.add_argument('--save', action='store_true', help="Save results (CSV and Plot)")
    parser.add_argument('--output-csv', type=str, default="multi_model_latency.csv")
    parser.add_argument('--output-plot', type=str, default="models/plots/Latency_Comparison.png")
    args = parser.parse_args()

    print_system_info()
    
    # ── 1. Setup Data ─────────────────────────────────────────────────
    pipeline = CSIPipeline()
    
    if args.simulate:
        print("💡 Mode: SIMULATE (Using synthetic waves)")
        rng = np.random.default_rng(seed=RANDOM_SEED)
        window_data = (rng.random((50, 128)) + 1j * rng.random((50, 128))).astype(np.complex64)
    else:
        data_path = Path("datasets/walk/walk_01.txt")
        if not data_path.exists():
            print(f"⚠️ Real data not found at {data_path}. Falling back to simulation...")
            rng = np.random.default_rng(seed=RANDOM_SEED)
            window_data = (rng.random((50, 128)) + 1j * rng.random((50, 128))).astype(np.complex64)
        else:
            print(f"📂 Mode: REAL DATA (Loading {data_path})")
            complex_matrix, _ = load_csi_csv(data_path)
            window_data = complex_matrix[500:550, :] # Take a 50-frame slice
    
    # Fit pipeline
    pipeline.fit_transform(window_data, use_pca=True, n_components=10)
    processed_ref = pipeline.transform(window_data).astype(np.float64)
    dummy_feat = extract_features_from_window(processed_ref).reshape(1, -1)
    X_dummy = np.tile(dummy_feat, (10, 1))
    y_dummy = np.array([0, 1] * 5)

    # ── 2. Benchmark ──────────────────────────────────────────────────
    models = get_models()
    all_results = []
    comparison_data = []
    
    initial_mem = get_memory_usage()

    for name, model in models.items():
        print(f"\n🚀 Benchmarking: {name}")
        
        mem_before = get_memory_usage()
        # Fit
        model.fit(X_dummy, y_dummy)
        mem_after = get_memory_usage()
        
        # RAM usage estimated as the delta after loading/fitting
        ram_mb = max(0, mem_after - mem_before)
        
        # Warm-up
        for _ in range(N_WARMUP):
            proc = pipeline.transform(window_data).astype(np.float64)
            feat = extract_features_from_window(proc).reshape(1, -1)
            model.predict(feat)
            
        # Benchmark
        model_times = []
        for _ in range(N_BENCHMARK):
            t_start = time.perf_counter()
            proc = pipeline.transform(window_data).astype(np.float64)
            feat = extract_features_from_window(proc).reshape(1, -1)
            model.predict(feat)
            t_end = time.perf_counter()
            
            elapsed_ms = (t_end - t_start) * 1000
            model_times.append(elapsed_ms)
            
            all_results.append({'model': name, 'latency_ms': elapsed_ms})
            
        avg_ms = np.mean(model_times)
        p95_ms = np.percentile(model_times, 95)
        fps = 1000.0 / avg_ms
        
        comparison_data.append({
            'Model': name,
            'Mean Latency': f"{avg_ms:.2f} ms",
            'p95 Latency': f"{p95_ms:.2f} ms",
            'Throughput': f"{fps:.0f} inf/sec",
            'RAM Usage': f"{ram_mb:.2f} MB"
        })
        print(f"   Mean: {avg_ms:.2f}ms | {fps:.0f} inf/sec | RAM: {ram_mb:.2f}MB")

    # ── 3. Report & Save ──────────────────────────────────────────────
    df_comp = pd.DataFrame(comparison_data)
    # Sort by Mean Latency (numeric extraction)
    df_comp['sort_val'] = df_comp['Mean Latency'].str.replace(' ms', '').astype(float)
    df_comp = df_comp.sort_values('sort_val').drop(columns=['sort_val'])
    
    print("\n" + "=" * 80)
    print(f"{'FINAL COMPARISON (Sorted by Speed)':^80}")
    print("=" * 80)
    print(df_comp.to_string(index=False))
    print("=" * 80)

    if args.save:
        # Save CSV
        df_raw = pd.DataFrame(all_results)
        df_raw.to_csv(args.output_csv, index=False)
        print(f"💾 Raw timings exported to: {args.output_csv}")
        
        # Save Plot
        Path(args.output_plot).parent.mkdir(parents=True, exist_ok=True)
        plot_comparison(df_comp, args.output_plot)
    else:
        print("ℹ️ Results not saved (use --save to export CSV and Plot)")

if __name__ == '__main__':
    main()