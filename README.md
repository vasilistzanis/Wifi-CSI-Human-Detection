# WiFi CSI Human Activity Recognition (HAR)

This repository contains a complete, end-to-end system for receiving, processing, visualizing, and classifying WiFi Channel State Information (CSI) data to perform Human Activity Recognition (HAR). 

It includes custom C firmware for the ESP32-C6 microcontrollers to capture the data, and Python scripts to analyze it using Digital Signal Processing (DSP) and Machine Learning (ML).

---

## 1. Hardware & Firmware (`ESP_Communication_espressif/`)

The system uses two independent ESP32-C6 microcontrollers acting as a **Sender** (Transmitter) and a **Receiver** (Sniffer/Logger). They utilize **ESP-NOW** for connectionless packet transmission and strictly force **Wi-Fi 4 (HT40)** on the 2.4 GHz band to capture 128 CSI subcarriers.

### The Sender (`csi_send`)
- **Deterministic Timing Loop:** Guarantees a stable transmission frequency (default: 100 Hz), compensating for execution drift.
- **Magic Header Payload:** Injects a strict 4-byte signature (`0xA1B2C3D4`) allowing the Receiver to filter out background Wi-Fi noise and only process target packets.
- **Self-Healing Wi-Fi:** Dynamically restarts the ESP-NOW interface if severe interference causes consecutive callback failures.
- **Fixed TX Power & Antenna:** Uses an external antenna configuration and locks the TX power to 18 dBm to prevent physical layer variations.

### The Receiver (`csi_recv`)
- **FreeRTOS Architecture:** A fast interrupt context pushes packets into a FreeRTOS Queue, while a separate Consumer task performs UART printing (`ets_printf`) without blocking the Wi-Fi stack.
- **AGC & FFT Gain Lock:** Records the Automatic Gain Control (AGC) scaling for the first 100 packets, locks a baseline, and stops dynamic adjustments to guarantee physical amplitude consistency.
- **Static Aligned Buffers:** Uses pre-allocated, word-aligned buffers for UART strings, completely avoiding `malloc`/`free` to prevent Heap fragmentation.
- **Atomic Drop Metrics:** Uses RISC-V atomic variables to track Queue Overflows lock-free.

---

## 2. Python Ecosystem & Workflow (`Python_Scripts_for_csi_recv/`)

The Python ecosystem processes the raw serial data into a robust, real-time ML pipeline with zero data leakage.

### Architecture
- **`config.py`**: The Single Source of Truth (SSOT). Contains all hardware parameters (Baud rate, COM port) and DSP constants (Window size, PCA components).
- **Data Acquisition:** `apps/csi_logger.py` records raw serial output to CSV. `csi_parser.py` safely parses strings to matrices and handles packet loss tracking. The root-level `csi_logger.py` keeps the legacy command working.
- **Signal Processing (DSP):** `data_preprocessing.py` implements the `CSIPipeline` (Null-subcarrier removal, Hampel filtering, Low-pass filtering, Temporal Difference, PCA, and Z-score scaling).
- **Machine Learning:** `csi_ml_pipeline.py` applies physics-based augmentations (noise, shift, scale, time-warp) and trains multiple classifiers.
- **Live Dashboards:** Implementations live under `Python_Scripts_for_csi_recv/apps/`; root-level launchers keep existing commands working.
- **Explainable AI (XAI):** `visualization/model_analysis/explain_model_characteristics.py` generates Feature Group and Permutation Importance charts to validate ML logic.

### Scientific Plotting & Visualization
Offline plotting tools are grouped under
`Python_Scripts_for_csi_recv/visualization/` by purpose:

- **`signal_processing/`**: thesis figures and views of each DSP stage.
- **`augmentation/`**: visualizations of training augmentations.
- **`model_analysis/`**: model metrics, pipeline views, and XAI charts.
- **`shared/`**: common plot styling and Qt/Matplotlib window helpers.

From `Python_Scripts_for_csi_recv/`, run them as Python modules so the project
root and shared modules resolve correctly:

```bash
python -m visualization.signal_processing.all_plot_figures --file datasets/walk_activity/walk_activity_01_vasilis_.txt
python -m visualization.augmentation.plot_data_augmentation --save --no-show
python -m visualization.model_analysis.plot_ml_results
python -m visualization.model_analysis.explain_model_characteristics --models_dir ./models --classes walk_activity no_activity
```

Runnable applications are organized under `Python_Scripts_for_csi_recv/apps/`,
and research/benchmark runners under `Python_Scripts_for_csi_recv/experiments/`.
Existing commands such as `python live_dashboard.py` remain available through
small compatibility launchers in the project root.

---

## Quick Start

### Hardware Setup & Flashing
Navigate to the firmware folders and use ESP-IDF to build and flash. Ensure the MAC addresses in `app_main.c` match your boards.
```bash
# Flash Receiver
cd ESP_Communication_espressif/csi_recv
idf.py set-target esp32c6
idf.py build
idf.py -p COM_PORT flash monitor
```

### Python Setup
```bash
cd Python_Scripts_for_csi_recv
pip install -r requirements.txt
```

### Pipeline Execution
1. **Test Connection:**
   ```bash
   python live_dashboard.py
   ```
2. **Record Dataset:**
   ```bash
   python csi_logger.py --port COM6 --label walk
   ```
3. **Train & Save the Model:**
   ```bash
   python csi_ml_pipeline.py --data_dir ./datasets --classes walk idle --save_model
   ```
4. **Run Live Inference:**
   ```bash
   python live_predict.py --port COM6
   ```

---

## Requirements
- **Hardware:** 2x ESP32-C6 Microcontrollers
- **Firmware:** ESP-IDF v5.0+
- **Python:** `numpy`, `scipy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`, `pyqtgraph`, `PyQt5`, `pyserial`, `pywavelets`

## Browser dashboard and cloud API

The Python project includes an optional web application in
[`Python_Scripts_for_csi_recv/web_app/`](Python_Scripts_for_csi_recv/web_app/):
a FastAPI API, an authenticated browser dashboard, and an edge collector for
the computer connected to the ESP32 serial port. The collector forwards CSI
frames to the cloud API over HTTPS, where the existing model performs live
inference. See the [web application setup guide](Python_Scripts_for_csi_recv/README.md#web-application-cloud-api--browser-dashboard)
for Docker deployment and collector setup.
