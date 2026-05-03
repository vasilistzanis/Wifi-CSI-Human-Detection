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
- **`explain_model_characteristics.py`**: Generates publication-ready XAI charts (Feature Group Importance, Permutation Importance) to explain exactly *why* the model makes its decisions. Evaluates FFT, Statistical, and DWT domains.
- **`benchmark_latency.py`**: Profiles the end-to-end latency of the DSP and ML inference steps to guarantee real-time performance capabilities.

### 5. Thesis & Publication Plotting
- **`all_plot_figures.py`**: Generates 7 core scientific figures (Amplitude, Heatmap, SC Profile, Energy/Variance, Spectrogram, Phase, Synchronized Motion Analysis). Integrates the exact `CSIPipeline` to mirror the ML data state, and includes a `--compare` mode for side-by-side class analysis.
- **`plot_lines_data_preprocessing.py`**: Generates 7 separate 2D line windows visualizing the exact effect of every single DSP step on the signal.
- **`visualize_all_steps_heatmap_data_preprocessing.py`**: Generates 7 separate heatmap windows showing the spatial-temporal transformations of the pipeline.
- **`plot_data_augmentation.py`**: Visualizes the effects of the physics-based augmentations (Noise, Shift, Scale, Time-Warp) on the filtered CSI waveforms. Supports `--realistic` flag to toggle between educational and ML-equivalent parameters.
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
python csi_logger.py --port COM6 --label walk
```

**3. Train & Save the Model:**
```bash
python csi_ml_pipeline.py --data_dir ./datasets --classes walk idle --save_model
```

**4. Run Live Inference:**
```bash
python live_predict.py --port COM6
```

**5. Explain Model Decisions (XAI):**
```bash
python explain_model_characteristics.py --models_dir ./models --classes walk idle
```

**6. Generate Thesis Plots:**
```bash
python plot_ml_results.py
python all_plot_figures.py --file datasets/walk/walk_01.txt
```

---

##  Requirements
- `numpy`, `scipy`, `pandas`
- `scikit-learn`, `joblib`
- `matplotlib`, `seaborn`
- `pyqtgraph`, `PyQt5`
- `pyserial`
- `pywavelets` *(optional — DWT features are currently disabled; install only if you plan to re-enable them)*

*(Install via `pip install -r requirements.txt`)*
