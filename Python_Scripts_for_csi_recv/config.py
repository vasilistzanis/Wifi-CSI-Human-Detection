import os


# ------------------------------------------------------------------------------
# Shared hardware and acquisition defaults
# ------------------------------------------------------------------------------
SERIAL_PORT = "COM6" if os.name == "nt" else "/dev/ttyUSB0"
BAUD_RATE = 2_000_000
RX_BUFFER_SIZE = 2_000_000
SAMPLING_RATE = 100.0
MAX_SUBCARRIERS = 128


# ------------------------------------------------------------------------------
# Shared ML defaults
# ------------------------------------------------------------------------------
TARGET_CLASSES = ["empty", "idle", "walk"]

WINDOW_SIZE = 50
PIPELINE_STEP_SIZE = 25
PREDICTION_STEP_SIZE = 10
STEP_SIZE = PIPELINE_STEP_SIZE

N_PCA_COMPONENTS = 10
MODELS_TO_TRAIN = "all"
RANDOM_SEED = 42
XAI_N_REPEATS = 10
# Extra frames kept beyond WINDOW_SIZE for Butterworth edge-transient absorption.
# live_predict.py and benchmark_latency.py must always use this same value.
FILTER_WARMUP = 50


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


# ------------------------------------------------------------------------------
# live_data_visualization.py defaults
# ------------------------------------------------------------------------------
LIVE_DATA_BUFFER_SIZE = 200
LIVE_DATA_REFRESH_MS = 50
LIVE_DATA_SUBCARRIERS = 128
