# CSI HAR — Multi-Model Benchmark Report

_Generated: 2026-06-19T23:35:40_

## System

- **OS**: Windows 10 (64bit)
- **CPU**: AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD
- **Python**: 3.11.9
- **NumPy / Pandas / scikit-learn**: 2.4.4 / 3.0.2 / 1.8.0

## Benchmark configuration

- **Window size**: 100 frames (+ 50 warmup = 150 buffered)
- **Step size**: 50 frames @ 100.0 Hz ⇒ real-time budget per step = **500.0 ms**
- **PCA components**: 10
- **Stats per component**: 25
- **Feature dimension**: 250
- **Warm-up runs**: 20
- **Benchmark runs (N per model)**: 200
- **Data source**: `datasets\walk_activity\walk_activity_11_room1_vasilis_1777802888.txt`
- **Training samples used for fit timing**: 7964 (classes: no_activity, walk_activity)
- **Feature vector version**: 5

## Results

| Μοντέλο | Πηγή | Χρόνος εκπαίδευσης<br/>(Train, s) | Μέγεθος μοντέλου<br/>(Size, KB) | Μέσος χρόνος πρόβλεψης<br/>(Inf mean, ms) | Χειρότερος χρόνος πρόβλεψης<br/>(Inf p95, ms) | Συνολικός χρόνος ανά παράθυρο<br/>(E2E p95, ms) | Χρήση προθεσμίας real-time<br/>(% budget, p95) |
|---|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | saved | 106.908 | 263.76 | 0.40 | 0.57 | 13.96 | 2.79% |
| Logistic Reg | saved | 0.235 | 9.06 | 0.34 | 0.54 | 15.70 | 3.14% |
| SVM (RBF) | saved | 11.395 | 6286.86 | 1.13 | 1.66 | 16.09 | 3.22% |
| K-NN | saved | 0.030 | 39204.97 | 8.29 | 9.97 | 24.80 | 4.96% |
| Random Forest | saved | 0.786 | 3669.71 | 33.78 | 48.74 | 66.04 | 13.21% |
| Extra Trees | saved | 0.323 | 2632.20 | 38.58 | 51.18 | 72.87 | 14.57% |

### Notes

- **Feature extraction (shared by every model)**: mean = 12.70 ms, p95 = 19.02 ms (pooled over N = 1200 window evaluations). Includes `pipeline.transform` (filter + diff + PCA) and `extract_features_from_window` (250 statistical / FFT features).
- **End-to-end (E2E) latency** per window = feature extraction + model inference. The **% budget** column reports `E2E p95 / 500.0 ms × 100`, where the budget is the time available before the next window starts (`step_size / fs = 50 / 100.0 Hz`). Anything < 100 % means the model meets the real-time deadline at p95.
- **Inference (`Inf`)** isolates `model.predict(feat)` only, so different model families can be compared head-to-head independent of preprocessing.
- **Training time** is the median wall-clock of `sklearn.base.clone(model).fit(X_train, y_train)` on the real CSI training split (no augmentation).
- **Model size** is the `.joblib` file size on disk (KB).
- **Measurement**: `time.perf_counter`, single-threaded, Python GC disabled during the timed loop, 20 warm-up runs preceding each 200-run measurement loop.
