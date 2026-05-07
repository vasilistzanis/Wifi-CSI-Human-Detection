from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def add_bool_argument(
    parser: argparse.ArgumentParser,
    *,
    dest: str,
    default: bool,
    help: str,
    positive_flags: list[str] | tuple[str, ...],
    negative_flags: list[str] | tuple[str, ...] | None = None,
) -> None:
    negative_flags = tuple(negative_flags or [])
    parser.set_defaults(**{dest: bool(default)})
    parser.add_argument(*positive_flags, dest=dest, action="store_true", help=help)
    if negative_flags:
        parser.add_argument(
            *negative_flags,
            dest=dest,
            action="store_false",
            help=argparse.SUPPRESS,
        )


# ------------------------------------------------------------------------------
# Shared hardware and acquisition defaults
# ------------------------------------------------------------------------------
SERIAL_PORT = "COM6" if os.name == "nt" else "/dev/ttyUSB0"
BAUD_RATE = 2_000_000
RX_BUFFER_SIZE = 2_000_000
SAMPLING_RATE = 100.0
MAX_SUBCARRIERS = 128


# ------------------------------------------------------------------------------
# Shared paths and model defaults
# ------------------------------------------------------------------------------
DATASETS_DIR = "datasets"
MODELS_DIR = "models"
PLOTS_DIR = "models/plots"
METRICS_JSON_PATH = "models/metrics.json"
LATENCY_OUTPUT_CSV = "models/multi_model_latency.csv"
LATENCY_OUTPUT_PLOT = "models/plots/Latency_Comparison.png"
DEFAULT_WALK_FILE = "datasets/walk/walk_01.txt"

FILTER_CUTOFF_HZ = 10.0
WINDOW_SIZE = 100
PIPELINE_STEP_SIZE = 50
PREDICTION_STEP_SIZE = 10
STEP_SIZE = PIPELINE_STEP_SIZE

N_PCA_COMPONENTS = 10
RANDOM_SEED = 42
MODELS_TO_TRAIN = "all"
MODEL_KEYS = ["svm", "rf", "et", "knn", "lr", "gb", "mlp", "nb"]
MODEL_CHOICES = MODEL_KEYS + ["all"]
AUGMENTATION_TECHNIQUES = ["noise", "shift", "scale", "time_warp"]
TEST_RATIO = 0.2
N_AUGMENTS = 4
CV_FOLDS = 5
XAI_N_REPEATS = 10
TOP_FEATURES = 15
START_FRAME = 500
FILTER_WARMUP = 50


# ------------------------------------------------------------------------------
# Training classes
# ------------------------------------------------------------------------------
TRAINING_CLASS_CONFIG = {
    "empty": {"enabled": True, "folder": "empty"},
    "idle": {"enabled": True, "folder": "idle"},
    "walk": {"enabled": True, "folder": "walk"},
    "sit": {"enabled": False, "folder": "sit"},
    "fall": {"enabled": False, "folder": "fall"},
    "stand": {"enabled": False, "folder": "stand"},
    "run": {"enabled": False, "folder": "run"},
}


def get_enabled_training_classes() -> list[str]:
    return [
        class_name
        for class_name, class_cfg in TRAINING_CLASS_CONFIG.items()
        if class_cfg.get("enabled", True)
    ]


def get_known_training_classes() -> list[str]:
    return list(TRAINING_CLASS_CONFIG.keys())


def get_training_class_folder(class_name: str) -> str:
    class_cfg = TRAINING_CLASS_CONFIG.get(class_name, {})
    return str(class_cfg.get("folder", class_name))


def resolve_training_classes(
    requested_classes: list[str] | tuple[str, ...] | None = None,
    data_dir: str | Path = DATASETS_DIR,
    *,
    require_existing: bool = True,
    print_fn=print,
) -> tuple[list[str], dict[str, Path]]:
    requested = list(requested_classes) if requested_classes is not None else get_enabled_training_classes()
    data_root = resolve_project_path(data_dir)

    resolved_classes: list[str] = []
    resolved_paths: dict[str, Path] = {}
    seen: set[str] = set()

    for class_name in requested:
        if class_name in seen:
            continue
        seen.add(class_name)

        class_cfg = TRAINING_CLASS_CONFIG.get(class_name)
        if class_cfg is None:
            print_fn(
                f"[WARNING] Training class '{class_name}' is not declared in "
                "config.TRAINING_CLASS_CONFIG - skipped."
            )
            continue

        if not class_cfg.get("enabled", True):
            print_fn(f"[INFO] Training class '{class_name}' is disabled in config - skipped.")
            continue

        class_dir = data_root / get_training_class_folder(class_name)
        if require_existing and not class_dir.is_dir():
            print_fn(
                f"[WARNING] Training class '{class_name}' is enabled but folder was not found: "
                f"{class_dir} - skipped."
            )
            continue

        resolved_classes.append(class_name)
        resolved_paths[class_name] = class_dir

    return resolved_classes, resolved_paths


TARGET_CLASSES = get_enabled_training_classes()


# ------------------------------------------------------------------------------
# live_dashboard.py defaults
# ------------------------------------------------------------------------------
DASHBOARD_WAVEFORM_LEN = 160
DASHBOARD_REFRESH_MS = 20
DASHBOARD_MOTION_THRESHOLD = 0.05
DASHBOARD_COLOR_SMOOTH = 0.15


# ------------------------------------------------------------------------------
# live_sensing_1.py defaults
# ------------------------------------------------------------------------------
LIVE_SENSING_WAVEFORM_LEN = 60
LIVE_SENSING_REFRESH_MS = 50
LIVE_SENSING_MOTION_THRESHOLD = 0.18
LIVE_SENSING_COLOR_SMOOTH = 0.12
LIVE_SENSING_MAX_SUBCARRIERS = 64
LIVE_SENSING_DEMO = False


# ------------------------------------------------------------------------------
# live_data_visualization.py defaults
# ------------------------------------------------------------------------------
LIVE_DATA_BUFFER_SIZE = 200
LIVE_DATA_REFRESH_MS = 50
LIVE_DATA_SUBCARRIERS = 128
LIVE_DATA_SERIAL_TIMEOUT = 0.25


# ------------------------------------------------------------------------------
# csi_logger.py defaults
# ------------------------------------------------------------------------------
LOGGER_OUTPUT_DIR = DATASETS_DIR
LOGGER_IDLE_SLEEP = 0.001
LOGGER_FLUSH_INTERVAL = 0.5
LOGGER_STATUS_INTERVAL = 0.25
LOGGER_MAX_FILE_SIZE_MB = 500
LOGGER_WAIT_SECONDS = 5
LOGGER_DURATION_SECONDS = 0


# ------------------------------------------------------------------------------
# live_predict.py defaults
# ------------------------------------------------------------------------------
LIVE_PREDICT_MODEL = "rf"
LIVE_PREDICT_HISTORY = 3
LIVE_PREDICT_VERBOSE = False
LIVE_PREDICT_COLOR = True
LIVE_PREDICT_FPS_WINDOW = 60


# ------------------------------------------------------------------------------
# benchmark_latency.py defaults
# ------------------------------------------------------------------------------
BENCHMARK_SIMULATE = False
BENCHMARK_SAVE = False
BENCHMARK_WARMUP_RUNS = 10
BENCHMARK_RUNS = 50


# ------------------------------------------------------------------------------
# plotting / visualization defaults
# ------------------------------------------------------------------------------
ALL_PLOT_FIGURES_SAVE = False
ALL_PLOT_FIGURES_FFT_MAX = 25.0
ALL_PLOT_FIGURES_ROLLING_WINDOW = 50

PLOT_DATA_AUGMENTATION_SIMULATE = False
PLOT_DATA_AUGMENTATION_SAVE = False
PLOT_DATA_AUGMENTATION_SHOW = True
PLOT_DATA_AUGMENTATION_MIN_FRAMES = 200
PLOT_DATA_AUGMENTATION_SUBCARRIER = 30
PLOT_DATA_AUGMENTATION_CLASS_LABEL = "walk"

PLOT_ADVANCED_METRICS_ROC = True
PLOT_ADVANCED_METRICS_SAVE = False
PLOT_ADVANCED_METRICS_SHOW = True
PLOT_ADVANCED_METRICS_MODELS = ["all"]

PLOT_ML_RESULTS_OUT_DIR = PLOTS_DIR

VISUALIZE_ML_PIPELINE_COMPARE = False
VISUALIZE_ML_PIPELINE_FEATURES = False
VISUALIZE_ML_PIPELINE_SAVE = False
VISUALIZE_ML_PIPELINE_SHOW = True

PREPROCESSING_PLOTS_SAVE = False
PREPROCESSING_USE_DIFF = True


# ------------------------------------------------------------------------------
# Script-specific CLI defaults
# ------------------------------------------------------------------------------
SCRIPT_DEFAULTS = {
    "all_plot_figures": {
        "file": None,
        "save": ALL_PLOT_FIGURES_SAVE,
        "compare": None,
        "fs": SAMPLING_RATE,
        "fft_max": ALL_PLOT_FIGURES_FFT_MAX,
        "rolling_window": ALL_PLOT_FIGURES_ROLLING_WINDOW,
        "out_dir": None,
    },
    "benchmark_latency": {
        "simulate": BENCHMARK_SIMULATE,
        "save": BENCHMARK_SAVE,
        "output_csv": LATENCY_OUTPUT_CSV,
        "output_plot": LATENCY_OUTPUT_PLOT,
        "models_dir": MODELS_DIR,
        "pca": N_PCA_COMPONENTS,
        "file": DEFAULT_WALK_FILE,
        "start_frame": START_FRAME,
        "window_size": WINDOW_SIZE,
        "n_warmup": BENCHMARK_WARMUP_RUNS,
        "n_benchmark": BENCHMARK_RUNS,
        "seed": RANDOM_SEED,
    },
    "csi_logger": {
        "port": SERIAL_PORT,
        "baud": BAUD_RATE,
        "label": None,
        "output_dir": LOGGER_OUTPUT_DIR,
        "idle_sleep": LOGGER_IDLE_SLEEP,
        "flush_interval": LOGGER_FLUSH_INTERVAL,
        "status_interval": LOGGER_STATUS_INTERVAL,
        "serial_buffer_size": RX_BUFFER_SIZE,
        "max_size_mb": LOGGER_MAX_FILE_SIZE_MB,
        "wait": LOGGER_WAIT_SECONDS,
        "duration": LOGGER_DURATION_SECONDS,
    },
    "csi_ml_pipeline": {
        "data_dir": DATASETS_DIR,
        "classes": list(TARGET_CLASSES),
        "window_size": WINDOW_SIZE,
        "step": PIPELINE_STEP_SIZE,
        "fs": SAMPLING_RATE,
        "augment": list(AUGMENTATION_TECHNIQUES),
        "use_augment": True,
        "n_augments": N_AUGMENTS,
        "pca": N_PCA_COMPONENTS,
        "test_ratio": TEST_RATIO,
        "use_diff": True,
        "simulate": False,
        "save_model": False,
        "tune": False,
        "model": MODELS_TO_TRAIN,
        "seed": RANDOM_SEED,
        "cv_folds": CV_FOLDS,
        "cutoff": FILTER_CUTOFF_HZ,
        "models_dir": MODELS_DIR,
    },
    "explain_model_characteristics": {
        "model": "rf",
        "models_dir": MODELS_DIR,
        "data_dir": DATASETS_DIR,
        "classes": list(TARGET_CLASSES),
        "top": TOP_FEATURES,
        "repeats": XAI_N_REPEATS,
        "save": False,
        "out_dir": None,
        "simulate": False,
        "window_size": WINDOW_SIZE,
        "step": PIPELINE_STEP_SIZE,
        "pca": N_PCA_COMPONENTS,
        "fs": SAMPLING_RATE,
        "seed": RANDOM_SEED,
        "cutoff": FILTER_CUTOFF_HZ,
    },
    "live_data_visualization": {
        "port": SERIAL_PORT,
        "baud": BAUD_RATE,
        "buffer_size": LIVE_DATA_BUFFER_SIZE,
        "subcarriers": LIVE_DATA_SUBCARRIERS,
        "refresh_ms": LIVE_DATA_REFRESH_MS,
        "serial_timeout": LIVE_DATA_SERIAL_TIMEOUT,
        "serial_buffer_size": RX_BUFFER_SIZE,
    },
    "live_predict": {
        "port": SERIAL_PORT,
        "baud": BAUD_RATE,
        "models_dir": MODELS_DIR,
        "model": LIVE_PREDICT_MODEL,
        "window": WINDOW_SIZE,
        "step": PREDICTION_STEP_SIZE,
        "history": LIVE_PREDICT_HISTORY,
        "verbose": LIVE_PREDICT_VERBOSE,
        "color": LIVE_PREDICT_COLOR,
        "fps_window": LIVE_PREDICT_FPS_WINDOW,
        "warmup": FILTER_WARMUP,
        "rx_buf": RX_BUFFER_SIZE,
        "cutoff": FILTER_CUTOFF_HZ,
    },
    "live_sensing_1": {
        "port": SERIAL_PORT,
        "baud": BAUD_RATE,
        "window": LIVE_SENSING_WAVEFORM_LEN,
        "refresh": LIVE_SENSING_REFRESH_MS,
        "threshold": LIVE_SENSING_MOTION_THRESHOLD,
        "smooth": LIVE_SENSING_COLOR_SMOOTH,
        "max_sc": LIVE_SENSING_MAX_SUBCARRIERS,
        "rx_buf": RX_BUFFER_SIZE,
        "demo": LIVE_SENSING_DEMO,
        "fs": SAMPLING_RATE,
    },
    "plot_advanced_metrics": {
        "json_path": METRICS_JSON_PATH,
        "models_dir": MODELS_DIR,
        "out_dir": None,
        "window_size": WINDOW_SIZE,
        "step": PIPELINE_STEP_SIZE,
        "model": list(PLOT_ADVANCED_METRICS_MODELS),
        "roc": PLOT_ADVANCED_METRICS_ROC,
        "save": PLOT_ADVANCED_METRICS_SAVE,
        "show": PLOT_ADVANCED_METRICS_SHOW,
    },
    "plot_data_augmentation": {
        "simulate": PLOT_DATA_AUGMENTATION_SIMULATE,
        "save": PLOT_DATA_AUGMENTATION_SAVE,
        "output_dir": PLOTS_DIR,
        "show": PLOT_DATA_AUGMENTATION_SHOW,
        "min_frames": PLOT_DATA_AUGMENTATION_MIN_FRAMES,
        "file": DEFAULT_WALK_FILE,
        "subcarrier": PLOT_DATA_AUGMENTATION_SUBCARRIER,
        "segment_len": WINDOW_SIZE,
        "class_label": PLOT_DATA_AUGMENTATION_CLASS_LABEL,
    },
    "plot_lines_data_preprocessing": {
        "file": None,
        "save": PREPROCESSING_PLOTS_SAVE,
        "n_subcarriers": MAX_SUBCARRIERS,
        "pca_components": N_PCA_COMPONENTS,
        "cutoff": FILTER_CUTOFF_HZ,
        "use_diff": PREPROCESSING_USE_DIFF,
        "fs": SAMPLING_RATE,
    },
    "plot_ml_results": {
        "json_path": METRICS_JSON_PATH,
        "out_dir": PLOT_ML_RESULTS_OUT_DIR,
    },
    "visualize_all_steps_heatmap_data_preprocessing": {
        "file": None,
        "save": PREPROCESSING_PLOTS_SAVE,
        "pca_components": N_PCA_COMPONENTS,
        "cutoff": FILTER_CUTOFF_HZ,
        "use_diff": PREPROCESSING_USE_DIFF,
    },
    "visualize_ml_pipeline_view": {
        "models_dir": MODELS_DIR,
        "file": None,
        "start_frame": START_FRAME,
        "window_size": WINDOW_SIZE,
        "step": PIPELINE_STEP_SIZE,
        "compare": VISUALIZE_ML_PIPELINE_COMPARE,
        "features": VISUALIZE_ML_PIPELINE_FEATURES,
        "save": VISUALIZE_ML_PIPELINE_SAVE,
        "out_dir": None,
        "show": VISUALIZE_ML_PIPELINE_SHOW,
    },
}


def get_script_defaults(script_name: str) -> dict:
    return deepcopy(SCRIPT_DEFAULTS[script_name])
