# CSI Radar — FastAPI + React + WebSocket

Real-time WiFi CSI Human Activity Recognition dashboard.

```
ESP32-C6
   │ serial (2 Mbaud)
   ▼
backend/server.py   (FastAPI + WebSocket)
   │ ws://localhost:8000/ws
   ▼
frontend/           (React + Vite)
   │ http://localhost:5173
   ▼
Browser Dashboard
```

---

## Setup

### 1. Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Place your trained models next to server.py (or adjust --models-dir):
#   models/csi_pipeline.joblib
#   models/label_encoder.joblib
#   models/SVM_RBF.joblib   (or whichever model you want)

# Also place these scripts in the parent directory (or adjust sys.path in server.py):
#   ../csi_plotter_heatmap.py
#   ../data_preprocessing.py
#   ../csi_ml_pipeline.py

# Run (adjust --port and --model as needed)
python server.py --port COM6 --model svm

# Without hardware (simulation mode):
python server.py --port COM6 --model svm
# → auto-detects missing models and runs synthetic data
```

Available options:
```
--port       COM6 | /dev/ttyUSB0   Serial port
--baud       2000000               Baud rate
--model      svm|rf|et|knn|lr|gb|mlp|nb
--models-dir ./models              Path to joblib model files
--host       127.0.0.1             Bind address
--ws-port    8000                  WebSocket port
```

### 2. Frontend

```bash
cd frontend

npm install
npm run dev
# → http://localhost:5173
```

---

## Architecture

### Backend (`server.py`)

- **FastAPI** — HTTP + WebSocket server
- **CSIReaderThread** — reads serial, runs inference in background thread
- **SharedState** — thread-safe bridge between reader and WebSocket handler
- **`/ws`** — WebSocket endpoint; broadcasts predictions to all connected clients
- **`/api/status`** — REST snapshot of last prediction (for debugging)

**Inference pipeline** (no Hampel for speed):
```
Serial frame → Null removal → Butterworth LP → Temporal diff → PCA → SVM → Prediction
```

### Frontend (`src/`)

| File | Purpose |
|---|---|
| `App.jsx` | WebSocket connection, state management, auto-reconnect |
| `components/Nav.jsx` | Logo + connection status indicator |
| `components/index.jsx` | Hero, SignalCard, MetricsRow, PredictionCard, Pipeline, Footer |
| `App.css` | Global dark theme, CSS variables |

### WebSocket message format

```json
{
  "label":          "walk",
  "smoothed":       "walk",
  "confidence":     0.869,
  "probabilities":  { "walk": 0.869, "idle": 0.131 },
  "fps":            82.4,
  "latency_ms":     23,
  "packet_loss":    1.3,
  "frame_count":    4821,
  "waveform":       [0.1, 0.8, ...],
  "subcarrier_map": [0.3, 0.7, ...],
  "connected":      true,
  "error":          "",
  "timestamp":      1714567890.123
}
```

---

## Production Build

```bash
# Build React for production
cd frontend && npm run build

# Serve static files from FastAPI (add to server.py):
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="static")

# Then a single command serves everything:
python server.py --host 0.0.0.0 --ws-port 8000
# → http://<your-ip>:8000
```
