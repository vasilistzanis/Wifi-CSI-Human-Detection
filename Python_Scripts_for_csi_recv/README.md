# WiFi CSI Human Activity Recognition (HAR) Pipeline

This directory contains the complete Python ecosystem for receiving, processing, visualizing, and classifying WiFi Channel State Information (CSI) data from ESP32 microcontrollers. 

This project transforms raw RF signals into a robust, real-time Human Activity Recognition (HAR) system using Digital Signal Processing (DSP) and Machine Learning (ML).

---

##  System Architecture & Workflow

The system is designed with a strict pipeline architecture, ensuring zero data leakage and 100% reproducibility between offline training and real-time inference.

1. **Collect Data:** Use `csi_logger.py` to record activities.
2. **Train Models:** Use `csi_ml_pipeline.py` to build the `CSIPipeline` (DSP) and train classifiers.
3. **Live Inference:** Use `live_predict.py` to load the exact artifacts and predict live data.
4. **Explain & Validate:** Use `explain_model_characteristics.py` to validate ML logic (XAI).

---

## ⚙️ Configuration

- **`config.py`**: The Single Source of Truth (SSOT) for the entire project. All hardware parameters (Baud rate, COM port), DSP constants (Window size, PCA components, Sampling rate), and UI preferences are defined here.
  - *Note:* Any Command Line Interface (CLI) arguments passed to individual scripts will safely override `config.py` defaults.

---

## 📂 Project Structure

### 1. Data Acquisition & Parsing
- **`csi_logger.py`**: Reads raw serial output from the ESP32 and exports strictly formatted CSV files with headers. Handles packet chunking and sleep intervals.
- **`csi_parser.py`**: The core string-to-matrix parser. Contains the centralized `SeqStats` and `SeqTransition` logic for robust packet loss, gap, and reset tracking across the entire codebase.

### 2. Signal Processing & Machine Learning
- **`data_preprocessing.py`**: Contains the `CSIPipeline` class. Implements the entire DSP chain: Null-subcarrier removal, Hampel filtering (outliers), Butterworth Low-pass filtering, Temporal Difference, PCA, and Z-score scaling.
- **`csi_ml_pipeline.py`**: The training engine. Loads datasets, applies physical data augmentations (noise, shift, scale, time-warp), strictly fits the `CSIPipeline` on training data only (Zero Leakage), trains multiple ML classifiers (RF, SVM, KNN, etc.), and exports `.joblib` artifacts.

### 3. Real-Time Inference & Dashboards
- **`live_predict.py`**: The production inference script. Loads the saved model and pipeline artifacts, reads live serial data, and outputs human activity predictions in real-time.
- **`live_dashboard.py`**: A comprehensive, high-performance `PyQtGraph` dashboard showing live amplitude, subcarrier variance, and motion thresholds.
- **`live_sensing_1.py`**: A specialized visualization focusing on motion energy and dynamic color mapping.
- **`live_data_visualization.py`**: A basic waterfall/heatmap monitor for raw CSI streams.

### 4. Explainable AI (XAI) & Profiling
- **`explain_model_characteristics.py`**: Generates publication-ready XAI charts (Feature Group Importance, Permutation Importance) to explain exactly *why* the model makes its decisions. Evaluates FFT and Statistical feature domains.
- **`benchmark_latency.py`**: Profiles the end-to-end latency of the DSP and ML inference steps to guarantee real-time performance capabilities.

### 5. Thesis & Publication Plotting
- **`all_plot_figures.py`**: Generates 7 core scientific figures (Amplitude, Heatmap, SC Profile, Energy/Variance, Spectrogram, Phase, Synchronized Motion Analysis). Integrates the exact `CSIPipeline` to mirror the ML data state, and includes a `--compare` mode for side-by-side class analysis.
- **`plot_lines_data_preprocessing.py`**: Generates 7 separate 2D line windows visualizing the exact effect of every single DSP step on the signal.
- **`visualize_all_steps_heatmap_data_preprocessing.py`**: Generates 7 separate heatmap windows showing the spatial-temporal transformations of the pipeline.
- **`plot_data_augmentation.py`**: Visualizes the exact augmentation functions used during training (Noise, Shift, Scale, Time-Warp) on filtered CSI waveforms. Supports `--class-label` to show class-aware parameter constraints (walk_activity/no_activity).
- **`plot_ml_results.py`**: Plots the output metrics and confusion matrices of the ML models. *(Note: Requires running `csi_ml_pipeline.py --save_model` first)*
- **`plot_advanced_metrics.py`**: Generates ROC curves, per-class accuracy bar charts, and advanced evaluation figures from saved model artifacts.
- **`visualize_ml_pipeline_view.py`**: Loads the saved `csi_pipeline.joblib` and visualises each DSP step, PCA scatter plots, and the final feature vector — for thesis illustration.

---

##  Quick Start

**0. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**1. Test Hardware Connection:**
```bash
python live_dashboard.py
```

**2. Record Dataset:**
```bash
python csi_logger.py --port COM6 --label walk_activity
```

**3. Train & Save the Model:**
```bash
python csi_ml_pipeline.py --data_dir ./datasets --classes walk_activity no_activity --save_model
```

**4. Run Live Inference:**
```bash
python live_predict.py --port COM6
```

**5. Explain Model Decisions (XAI):**
```bash
python explain_model_characteristics.py --models_dir ./models --classes walk_activity no_activity
```

**6. Generate Thesis Plots:**
```bash
python plot_ml_results.py
python all_plot_figures.py --file datasets/walk_activity/walk_activity_01_vasilis_.txt
```

## Web application (cloud API + browser dashboard)

The `web_app/` directory adds a FastAPI server, a browser dashboard with
administrator login, and a small edge collector. The collector runs on the
computer attached to the ESP32, reads the existing serial `CSI_DATA` stream,
and forwards batches over HTTPS. The cloud API runs the existing trained
pipeline and publishes live predictions to authenticated browser sessions.
No firmware changes are required.

### Start the cloud API

Model files are tracked using Git LFS. Before running locally or building the
container, install Git LFS and fetch the actual binaries from the repository
root:

```bash
git lfs install
git lfs pull --include="Python_Scripts_for_csi_recv/models/har/**,Python_Scripts_for_csi_recv/models/motion/**"
```

The API deliberately rejects LFS pointer text instead of reporting a vague
model-unpickling error. The container image does not copy model artifacts;
Compose mounts `models/har/` and `models/motion/` read-only at runtime. Any
mode without downloaded trained artifacts is disabled in the desktop settings.

### Run locally without Docker

This web app uses Python 3.11 or newer. From this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r web_app/requirements-dev.txt
cp web_app/.env.example web_app/.env
```

In VS Code, select `Python_Scripts_for_csi_recv/.venv/bin/python` using
**Python: Select Interpreter**. Otherwise Pylance may analyze `server.py` with
the system interpreter and report the web dependencies as missing even though
they are installed in this virtual environment.

Edit `web_app/.env`: set the administrator username and password, set a
random `CSI_EDGE_TOKEN` (at least 32 characters), and set
`CSI_COOKIE_SECURE=false` for local HTTP. Keep the local allowed-host and
origin values. The `.env` file is ignored by Git and should remain private.
After the one-time setup, start the API from one terminal:

```bash
./start_api.sh
```

Open `http://127.0.0.1:8000`. To run the desktop `live_dashboard.py` and send
the same serial CSI stream to the API, open a second terminal and run:

```bash
./start_dashboard.sh
```

The scripts load the same local `.env` automatically. The desktop reads the
serial port once and forwards raw frames in the background. When the API is
configured, it is the only inference engine: model/mode selection and runtime
inference settings are applied there, and both the desktop and browser display
the API's prediction. This prevents the two interfaces from producing
different results. The desktop Settings page controls model, mode, and runtime
inference parameters; those settings are persisted by the API and broadcast to
the browser. Non-default dashboard command-line options can override runtime
parameters at startup. The browser is a read-only live mirror plus session history. Set
`CSI_SERIAL_PORT` in `.env` to use a different serial port.
Do not run the standalone collector at the same time because both programs
would compete for the serial device. If using only the browser dashboard, run
the standalone collector from this directory instead:

```bash
set -a
source web_app/.env
set +a
.venv/bin/python -m web_app.collector --port "${CSI_SERIAL_PORT:-/dev/ttyACM0}"
```

On Linux, the account needs permission to access the serial device, commonly
by being a member of the `dialout` group followed by logging out and back in.

### Start with Docker Compose

From this directory, create `web_app/.env` from `web_app/.env.example`. Set a
unique administrator password (at least 12 characters) and a random
`CSI_EDGE_TOKEN` (at least 32 characters). Choose an existing model with
`CSI_MODEL` (`rf`, `svm`, `et`, `knn`, `lr`, `gb`, `mlp`, or `nb`).

```bash
cp web_app/.env.example web_app/.env
# Edit web_app/.env and replace the example credentials/secrets.
# Generate a token with: openssl rand -hex 32
docker compose -f web_app/docker-compose.yml up --build -d
```

Open `http://127.0.0.1:8000` for local testing. Compose binds only to loopback,
uses a non-root process, read-only model mount, read-only container filesystem,
and a persistent volume for application data. The web dashboard automatically
groups incoming frames into activity sessions; a session ends 15 seconds after
CSI stops arriving. Session details retain exact prediction totals and average class
confidence/probability by activity and model, the model-change history, and a
sampled prediction timeline. Model changes from the desktop app are synchronized
to connected pages. For a public deployment, put the
service behind a TLS-terminating reverse proxy; retain loopback-only port
binding and keep `CSI_COOKIE_SECURE=true`. Set `CSI_ALLOWED_HOSTS` to include
the deployment hostname (without scheme) and `CSI_ALLOWED_ORIGINS` to the
exact HTTPS origin (for example `https://csi.example.com`). Keep the loopback
hostname in `CSI_ALLOWED_HOSTS` for the container health check.
An example Nginx TLS/WebSocket reverse-proxy configuration is provided in
`web_app/nginx.conf.example`; replace the example domain and configure valid
certificates before enabling it.

The SQLite volume and in-memory inference buffer are intended for one API
instance. The first startup creates the administrator account. There is no
public signup; changing the environment password does not overwrite an
existing account. To create a user or reset a password, run
`docker compose -f web_app/docker-compose.yml exec api python -m web_app.admin --username admin`
and enter the new password at the prompts. Existing sessions for that user
are revoked. Raw CSI frames are processed in memory; only sampled prediction
timeline rows are retained, while per-session activity statistics aggregate
every prediction.

### Start the ESP32 edge collector

On the computer connected to the ESP32, install the Python requirements and
run the collector. Set `CSI_SERVER_URL` to the HTTPS URL of the deployed API
and `CSI_EDGE_TOKEN` to the same secret configured on the API.

```bash
pip install -r requirements.txt
export CSI_SERVER_URL=https://your-app.example.com
export CSI_EDGE_TOKEN='the-same-random-secret-as-the-cloud-server'
python -m web_app.collector --port /dev/ttyUSB0
```

On Windows, pass the appropriate serial port, for example
`python -m web_app.collector --port COM6`. Run the command from this directory
so Python can import the shared project configuration. The collector batches frames and
retries temporary network failures; if its bounded queue fills, it stops with
an explicit error rather than silently discarding CSI data.

The API also exposes `/healthz`, `/api/status`, `/api/models`,
`/api/predictions`, `/api/auth/*`, and the authenticated live event channel at
`/api/events`. Dashboard/API routes require the administrator session; only
health checks and the token-protected frame-ingest route are public. `/readyz`
checks that the application and model are ready. API documentation is
disabled by default; enable it only in a trusted environment with
`CSI_ENABLE_API_DOCS=true`.

### Tests

Run the focused API/authentication/ingest tests with:

```bash
.venv/bin/python -m pytest -q web_app/tests
```

`web_app/requirements.lock.txt` pins the resolved runtime dependency versions
used by the container build. The Docker Compose port is intentionally local
only; cloud providers that run the Dockerfile directly must provide HTTPS,
persistent storage for `CSI_DB_PATH`, the model mount, environment secrets,
and the configured host/origin allowlists.

---

##  Requirements
- `numpy`, `scipy`, `pandas`
- `scikit-learn`, `joblib`
- `matplotlib`, `seaborn`
- `pyqtgraph`, `PyQt5`
- `pyserial`
- `pywavelets` *(optional — DWT features are currently disabled; install only if you plan to re-enable them)*

*(Install via `pip install -r requirements.txt`)*
For only the cloud API, install `web_app/requirements.txt` instead; it omits
the desktop plotting and Qt dependencies.
