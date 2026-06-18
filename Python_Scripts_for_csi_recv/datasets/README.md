# WiFi CSI Datasets

This directory (`datasets/`) contains the **raw CSI (Channel State Information) data** collected from the WiFi router (Espressif ESP32) for the purposes of the thesis project.

In total, the directory includes **100 recordings**, perfectly balanced across classes:
- **`walk_activity/`**: 50 recordings (Walking activity in the area)
- **`no_activity/`**: 50 recordings (Absence of motion / Silence)

---

## 📁 Structure and Naming Convention
The data is organized into subdirectories based on their target class (label). The naming convention of each `.txt` file is designed to be self-explanatory and allows the Python code to automatically parse metadata (used for Cross-Environment & Cross-Subject analysis).

The basic naming structure is:
`[Class]_[ID]_[Environment]_[Subject]_[Timestamp].txt`

*Example: `walk_activity_36_room1_vasilis_1777976169.txt`*
* **Class:** `walk_activity` (Refers to the walking activity).
* **ID:** `36` (The sequential number of this specific recording).
* **Environment / Room:** `room1` (The physical space where the recording took place. Another environment found in the dataset is `livroom`).
* **Subject:** `vasilis` (The person performing the activity).
* **Timestamp:** `1777976169` (The Unix timestamp ensuring the file's uniqueness).

---

## 📊 Data Format
Each `.txt` file represents a **CSI Time Series**.
- The **lines** of the file represent the temporal samples (packets) received over time.
- Each line contains a sequence of numerical values corresponding to the **CSI Amplitudes** across the different subcarriers of the WiFi antenna.

---

## 🚀 Data Processing Pipeline
1. The script (`experiment_runner.py` / `csi_ml_pipeline.py`) simultaneously reads all `.txt` files in this directory.
2. It parses their filenames to extract the ground truth label (class) and the environment metadata.
3. The raw data passes through a **Butterworth Filter (lowpass)** to eliminate high-frequency noise.
4. The cleaned signal is segmented into "windows" (e.g., chunks of 100 packets).
5. From each window, the system extracts **Features** such as PCA components, Fourier Transforms (FFT), statistical variances, Zero-Crossing Rates, etc.
6. These engineered features are finally fed to the models (in the `models/` directory) for training!

It is extremely important to maintain this exact directory structure. In the event of future dataset expansion (e.g., adding a `sit_activity` class), the code will seamlessly read the new folder and train new multi-class models 100% automatically!
