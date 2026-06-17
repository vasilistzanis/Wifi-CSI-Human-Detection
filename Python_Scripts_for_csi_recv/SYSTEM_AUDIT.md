# SYSTEM AUDIT — WiFi CSI HAR Thesis Project
**Project root:** `C:\Diplomatiki_2026\WIFI CSI PROJECT\Python_Scripts_for_csi_recv`
**Generated:** 2026-06-14
**Purpose:** Hand this document to the AI reviewer for independent cross-checking of experiments, code, model behavior, recording details, and pipeline correctness.

> READER NOTE — this is a forensic, line-level audit. Every constant, knob, window, step, model file, recording count, formula, and known inconsistency that exists in the repo is enumerated below. If something is not here, it does not exist in the project.

---

## 0. TL;DR (one-screen overview)

- **Hardware:** ESP32-C6 sending CSI over serial @ **COM6 / 2,000,000 baud**, **100 Hz**, **128 subcarriers** (114 active after null removal).
- **IQ convention:** ESP32 buffer = `[imag0, real0, imag1, real1, …]`, decoded as `complex = real + j·imag` (verified in `csi_parser.parse_csi_line`).
- **Two active classes only:** `no_activity`, `walk_activity` (5 other classes declared but **disabled** in `config.TRAINING_CLASS_CONFIG`).
- **DSP chain (CSIPipeline):** null subcarrier removal → vectorized Hampel (W=11, 3σ MAD) → Butterworth LP order-4 SOS @ 10 Hz (`sosfiltfilt`) → temporal first diff → PCA (10 comps) → StandardScaler.
- **Feature vector:** **FEATURE_VECTOR_VERSION = "5"** → **25 stats × 10 PCA = 250-dim** vector.
- **Training:** file-level (recording-level) train/test split 80/20, seeded shuffle, fit DSP on TRAIN recordings only, augment 4 techniques × 4 augments **on raw windows before PCA**, CV = **StratifiedGroupKFold (5)** on non-augmented data.
- **Models (HAR, window=100, step=50):** SVM, RF, ET, KNN, LR, GB, MLP, NB — all retrained @ v5; best on hold-out **GB 97.17 %, ET 97.12 %, RF 96.98 %, LR 96.85 %, SVM 96.80 %, MLP 96.53 %, KNN 93.56 %, NB 61.26 %**.
- **Models (MOTION, window=40, step=8):** RF, GB, ET, LR — best **GB 96.65 %, ET 96.37 %, RF 96.22 %, LR 96.25 %**.
- **Live stack:** `live_dashboard.py` uses a true 3-process design (GUI / Reader / Inference) backed by `multiprocessing.shared_memory` + `mp.Queue`. `live_predict.py` is the terminal twin.
- **Dataset on disk:** 50 walk + 50 no_activity recordings (~217 MB each side, total ~434 MB raw text).
- **STALE-MEMORY notes for the AI reviewer:** older project memory mentions a `FallEventDetector` class in `live_dashboard.py` — **it no longer exists**; older notes mention `220-dim` features (v4) — **current is 250-dim (v5)**. Both deltas confirmed against current source.

---

## 1. Repository inventory (root files, with byte sizes)

| File | Bytes | Role |
|---|---|---|
| `live_dashboard.py` | 121 594 | Multi-process Qt GUI dashboard (Monitor / Signal / Log / System / Settings pages). |
| `csi_ml_pipeline.py` | 67 805 | Training engine — augmentation, feature extraction, GroupKFold CV, GridSearchCV, save_models. |
| `explain_model_characteristics.py` | 27 905 | XAI charts — permutation importance, group importance. |
| `plot_advanced_metrics.py` | 27 132 | ROC + per-class accuracy bar charts from `metrics.json`. |
| `all_plot_figures.py` | 27 069 | Seven thesis DSP figures (amp, heatmap, SC profile, energy, spectrogram, phase, sync motion). |
| `visualize_ml_pipeline_view.py` | 21 939 | Visualises saved pipeline + PCA scatter + feature vector. |
| `live_predict.py` | 21 723 | Terminal-only live inference (ANSI bar). |
| `plot_lines_data_preprocessing.py` | 20 241 | 7 line-plot windows for each DSP step. |
| `data_preprocessing.py` | 19 168 | `CSIPipeline` + `load_csi_csv`. |
| `benchmark_latency.py` | 18 541 | Multi-model end-to-end latency benchmark. |
| `csi_logger.py` | 18 287 | Serial CSI recorder with activity presets and space-bar event markers. |
| `live_csi_dual_view.py` | 18 061 | Two-window time-domain / current-frame Qt viewer. |
| `config.py` | 18 027 | Single source of truth — all constants, CLI defaults, per-script presets. |
| `visualize_all_steps_heatmap_data_preprocessing.py` | 13 928 | 7 heatmap windows of DSP transformations. |
| `live_sensing_1.py` | 12 875 | "Oscilloscope" waveform with breathing color (calm→warn→peak). |
| `plot_data_augmentation.py` | 12 187 | Visualises 4 augmentation techniques on filtered CSI. |
| `csi_parser.py` | 10 088 | Shared line parser, SeqStats, IQ decoder. |
| `plot_ml_results.py` | 7 038 | Plots metrics + confusion matrices from `metrics.json`. |
| `README.md` | 5 905 | Quick start. |
| `plot_window_utils.py` | 4 295 | Qt window-centering helper. |
| `plot_styles.py` | 2 585 | Matplotlib style. |
| `requirements.txt` | 341 | Dependencies (no version pins). |

Subdirs:
- `datasets/no_activity/` — 50 .txt + 50 derived .csv + many .png plots (≈ **217 MB** raw text in .txt files; min 1.36 MB, max 5.80 MB, avg 4.35 MB).
- `datasets/walk_activity/` — 50 .txt + 50 .csv + plot dirs (≈ **217 MB**; min 0.79 MB, max 5.73 MB, avg 4.34 MB). One subdir `walk_activity_35_room1_vasilis_1777805631_thesis_plots/` of generated thesis plots.
- `datasets/no_activity/*.meta.json` — **0 sidecar files** (logger writes them only when `--mode` or event markers are present; the bulk of recordings were captured without that path → no sidecars).
- `datasets/walk_activity/*.meta.json` — **0 sidecar files**.
- `models/` — top-level (`har` snapshot duplicated here): `csi_pipeline.joblib`, `label_encoder.joblib`, 8 classifier `.joblib`, `metrics.json`, `experiment_config.json`, `plots/`.
- `models/har/` — production HAR set (8 classifiers + pipeline + LE + metrics + experiment_config).
- `models/motion/` — Super Motion mode set (rf, gb, et, lr only; pipeline + LE + metrics + experiment_config).
- `tests/__pycache__/` — only compiled `.pyc` of `conftest`, `test_csi_parser`, `test_data_preprocessing`, `test_extract_features`, `test_reader_process`. **No `.py` source files present** — the test sources have been removed (or never committed); only compiled bytecode remains. This is a real gap: tests cannot be re-run.

---

## 2. `config.py` — every constant (the single source of truth)

### Hardware / acquisition
| Constant | Value |
|---|---|
| `SERIAL_PORT` | `COM6` on Windows / `/dev/ttyUSB0` otherwise |
| `BAUD_RATE` | `2_000_000` |
| `RX_BUFFER_SIZE` | `2_000_000` bytes |
| `SAMPLING_RATE` | `100.0` Hz |
| `MAX_SUBCARRIERS` | `128` |

### Paths
`DATASETS_DIR = "datasets"`, `MODELS_DIR = "models"`, `MODELS_HAR_DIR = "models/har"`, `MODELS_MOTION_DIR = "models/motion"`, `PLOTS_DIR = "models/plots"`, `METRICS_JSON_PATH = "models/metrics.json"`, `LATENCY_OUTPUT_CSV = "models/multi_model_latency.csv"`, `LATENCY_OUTPUT_PLOT = "models/plots/Latency_Comparison.png"`, `DEFAULT_WALK_FILE = "datasets/walk_activity/walk_activity_01_vasilis_.txt"`.

### Core ML / DSP constants
| Constant | Value |
|---|---|
| `FILTER_CUTOFF_HZ` | `10.0` Hz |
| `WINDOW_SIZE` | `100` frames (1.00 s @ 100 Hz) |
| `PIPELINE_STEP_SIZE` | `50` frames (50 % overlap) |
| `PREDICTION_STEP_SIZE` | `10` frames (live `live_predict.py`) |
| `STEP_SIZE` | alias of `PIPELINE_STEP_SIZE` |
| `N_PCA_COMPONENTS` | `10` |
| `RANDOM_SEED` | `42` |
| `MODELS_TO_TRAIN` | `"all"` |
| `MODEL_KEYS` | `["svm","rf","et","knn","lr","gb","mlp","nb"]` |
| `MODEL_FILES` | `{key: "<key>.joblib"}` |
| `AUGMENTATION_TECHNIQUES` | `["noise","shift","scale","time_warp"]` |
| `TEST_RATIO` | `0.2` |
| `N_AUGMENTS` | `4` |
| `CV_FOLDS` | `5` |
| `XAI_N_REPEATS` | `10` |
| `TOP_FEATURES` | `15` |
| `START_FRAME` | `500` (benchmark trim) |
| `FILTER_WARMUP` | `50` frames (Butterworth edge transient absorption) |

### Training class configuration (`TRAINING_CLASS_CONFIG`)
| Class | Enabled? | Folder |
|---|---|---|
| `empty` | ❌ | `empty` |
| `no_activity` | ✅ | `no_activity` |
| `walk_activity` | ✅ | `walk_activity` |
| `sit` | ❌ | `sit` |
| `fall` | ❌ | `fall` |
| `stand` | ❌ | `stand` |
| `run` | ❌ | `run` |

`get_enabled_training_classes() == ["no_activity","walk_activity"]`. `TARGET_CLASSES` resolves to the same list at import time.

### Plot sizes
`FIGURE_SIZE = (10, 7)`, `FIGURE_DPI = 100`, `QT_WINDOW_W=800`, `QT_WINDOW_H=750`. All per-script size aliases (`ALL_PLOT_FIGURES_SIZE`, `BENCHMARK_FIGURE_SIZE`, etc.) currently equal `FIGURE_SIZE`.

### Live dashboard knobs
`DASHBOARD_WAVEFORM_LEN=200`, `DASHBOARD_REFRESH_MS=10` (100 Hz GUI timer — experimental), `DASHBOARD_STEP_SIZE=15`, `DASHBOARD_EMA_ALPHA=0.6`, `DASHBOARD_CONF_THRESH=70.0`, `DASHBOARD_MAX_LOG=60`, `DASHBOARD_DEMO=False`, `DASHBOARD_HYST_COUNT=2`.

### `live_sensing_1.py`
`LIVE_SENSING_WAVEFORM_LEN=60`, `LIVE_SENSING_REFRESH_MS=50`, `LIVE_SENSING_MOTION_THRESHOLD=0.18`, `LIVE_SENSING_COLOR_SMOOTH=0.12`, `LIVE_SENSING_MAX_SUBCARRIERS=64`.

### `live_csi_dual_view.py`
`LIVE_CDV_BUFFER_SIZE=200`, `LIVE_CDV_REFRESH_MS=33`, `LIVE_CDV_MAX_SUBCARRIERS=128`.

### `live_data_visualization.py` placeholders
`LIVE_DATA_BUFFER_SIZE=200`, `LIVE_DATA_REFRESH_MS=50`, `LIVE_DATA_SUBCARRIERS=128`, `LIVE_DATA_SERIAL_TIMEOUT=0.25`. **NOTE for the AI reviewer:** the file `live_data_visualization.py` is referenced in `config.SCRIPT_DEFAULTS` and in `README.md` but **does not exist in the repo** — dead config. Confirmed by `dir` listing.

### `csi_logger.py`
`LOGGER_OUTPUT_DIR=datasets`, `LOGGER_IDLE_SLEEP=0.001`, `LOGGER_FLUSH_INTERVAL=0.5`, `LOGGER_STATUS_INTERVAL=0.25`, `LOGGER_MAX_FILE_SIZE_MB=500`, `LOGGER_WAIT_SECONDS=5`, `LOGGER_DURATION_SECONDS=0` (continuous).

**`LOGGER_ACTIVITY_PRESETS` — exact values used during recording (the AI reviewer, cross-check against the corpus we have):**

| Preset | Output dir | Duration (s) | Wait (s) | Label prefix |
|---|---|---|---|---|
| `fall` | `datasets/fall` | 4 | 5 | `fall` |
| `sit` | `datasets/sit` | 4 | 5 | `sit` |
| `walk` | `datasets/walk_activity` | **60** | 5 | `walk_activity` |
| `idle` | `datasets/no_activity` | **120** | 5 | `no_activity` |

(So each `walk_activity_NN_*.txt` is ~60 s of recording → at 100 Hz, ~6 000 frames; each `no_activity_NN_*.txt` is ~120 s → ~12 000 frames. The actual recording length per file is logged inside the .meta.json sidecar **when present** — note that none of the current 100 dataset files have sidecars.)

### `live_predict.py` defaults
`LIVE_PREDICT_MODEL="rf"`, `LIVE_PREDICT_HISTORY=3` (majority-vote smoothing), `LIVE_PREDICT_VERBOSE=False`, `LIVE_PREDICT_COLOR=True`, `LIVE_PREDICT_FPS_WINDOW=60`.

### `benchmark_latency.py`
`BENCHMARK_SIMULATE=False`, `BENCHMARK_SAVE=False`, `BENCHMARK_WARMUP_RUNS=10`, `BENCHMARK_RUNS=50`.

### Plotting defaults
`ALL_PLOT_FIGURES_FFT_MAX=25.0`, `ALL_PLOT_FIGURES_ROLLING_WINDOW=50`. `PLOT_DATA_AUGMENTATION_MIN_FRAMES=200`, `PLOT_DATA_AUGMENTATION_SUBCARRIER=30`, `PLOT_DATA_AUGMENTATION_CLASS_LABEL="walk_activity"`. `PLOT_ADVANCED_METRICS_ROC=True`, `PLOT_ADVANCED_METRICS_MODELS=["all"]`.

### `SCRIPT_DEFAULTS` map
Every script pulls its defaults via `config.get_script_defaults(name)`. Names registered: `all_plot_figures`, `benchmark_latency`, `csi_logger`, `csi_ml_pipeline`, `explain_model_characteristics`, `live_dashboard`, `live_csi_dual_view`, `live_data_visualization` (script missing), `live_predict`, `live_sensing_1`, `plot_advanced_metrics`, `plot_data_augmentation`, `plot_lines_data_preprocessing`, `plot_ml_results`, `visualize_all_steps_heatmap_data_preprocessing`, `visualize_ml_pipeline_view`.

---

## 3. `csi_parser.py` — line format + sequence statistics

- `RECV_FIELD_COUNT = 15` (the recv-line schema).
- `split_recv_fields(line)` splits with maxsplit `RECV_FIELD_COUNT-1`; **rejects** lines whose integer fields (indices 1,3,4,5,6,7,8,9,10,11,12,13) do not parse as `int`. Line must start with `CSI_DATA` literal.
- `parse_csi_line(...)`:
  - extracts payload from `parts[14]` between `"[ ... ]"`,
  - parses with `np.fromstring(..., sep=",", dtype=float32)`, fallback to comprehension,
  - rejects when `values.size != token_count` (catches embedded garbage), or `values.size < 2`, or `values.size % 2 != 0`,
  - optional `expected_subcarriers` strict check,
  - if `parts[13] != 0` (first_word_invalid), zero the first 4 raw values (≈ 2 subcarriers) — DC scrub,
  - **IQ decode:** `imag = values[0::2]`, `real = values[1::2]`, returns `(real + 1j*imag).astype(complex64)`.
- `SeqTransition`:
  - `diff > 1` ⇒ gap (`missing_count = diff-1`).
  - `diff == 0` ⇒ duplicate (does not update `last_seq`).
  - `diff < 0` ⇒ reset (treated as reorder).
- `SeqStats` tracks `received_count`, `missing_count`, `gap_events`, `duplicate_count`, `reset_count`, `loss_percent = missing / expected_count`.
- `load_csi_matrix(path)` returns `(complex_matrix, dropped_frames, SeqStats)`; raises `FileNotFoundError`, `ValueError` on empty/malformed.

> **Cross-check for the AI reviewer:** sample line from `walk_activity_11_room1_vasilis_1777802888.txt` (line 1) has `first_word=11` (non-zero) AND the payload’s first 6 values are NOT zero in some frames. Verify whether the `first_word_invalid → zero first 4` scrub is doing what the data layout expects, given that frames 2 and 3 already start with 12 leading zeros (a different subcarrier-null pattern). This may indicate that the channel/bandwidth selected on the ESP32 produces a 12-value (6-subcarrier) null guard band that the parser does NOT need to scrub. Confirm consistency.

---

## 4. `csi_logger.py` — recording details

- CSV header injected during the `.csv` export step:
  `type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data\n`
- File naming during capture: `{label}_{unix_int}.txt`. Auto-numbered preset labels: `{prefix}_NNN` (3-digit, max 999) using `_next_auto_label`. Range 1000+ is excluded so that recordings tagged with raw unix timestamps in their filename do not confuse the counter.
- Sample timestamp parsing: `int(output_path.stem.split("_")[-1])` (used for `recording_unix` in sidecar). When the stem ends with a non-numeric token this falls back to `int(time.time())`. **Most current dataset filenames end with a unix timestamp.**
- Space / Enter while recording → push elapsed-time mark to `event_markers` list; saved to `.meta.json` sidecar only when `args.mode` or `event_markers` exist.
- Windows-only port validation via `pyserial.tools.list_ports`; sets RX buffer with `ser.set_buffer_size(rx_size=...)`.
- Performance: writes binary chunks `ser.read(ser.in_waiting)` into a 1 MiB-buffered file handle; flushes on `--flush-interval`; reports `KB`, `KB/s`, `% full`, optional countdown of remaining seconds.
- Multi-session loop with `-n N` (recorder pauses between captures and waits for user `Enter`).
- Sleep on empty serial: `args.idle_sleep = 0.001 s`. Status print every `0.25 s`.
- **NOTE for the AI reviewer:** when no preset is supplied, the logger prompts the operator via `input()` — this matters if running under non-interactive automation; would silently hang.

---

## 5. `data_preprocessing.py` — DSP pipeline

### `load_csi_csv(filepath)`
- Returns `(complex_matrix complex64, metadata_df)` with columns `seq, rssi, fft_gain, agc_gain, len, first_word`.
- Uses `csi_parser.parse_csi_line` (single source of truth for IQ decode).
- Tracks sequence anomalies via `SeqStats`; prints warning summary if missing / reset / duplicate.
- **Frame-length resolution:** if mixed subcarrier counts appear in one file, keeps only frames with the *most common* length and prints how many were kept. This is a graceful fallback rather than aborting on a malformed segment.

### `CSIPipeline`
Configurable with `fs=100`, `use_diff=True`, `cutoff=10`.
State after fitting:
- `active_mask` (boolean, shape `(MAX_SUBCARRIERS,)`),
- `pca` (sklearn PCA, `n_components_` typically 10),
- `scaler` (StandardScaler default; MinMaxScaler optionally),
- `is_fitted`,
- `_fitted_n_subcarriers` (the raw input width = 128 by design).

Steps (in order):
1. **`remove_null_subcarriers`** — `np.abs > 1e-3` over all frames to build `active_mask` (only during `fit=True`). Raises if 0 active. On `n_active=0` after null mask the pipeline aborts.
2. **`apply_hampel_filter` (vectorized)** — window_size=11, `n_sigmas=3.0`, reflect-pad, `np.lib.stride_tricks.sliding_window_view`, threshold `max(3·1.4826·MAD, 1e-6)`, replace outliers with rolling median. Comments correctly note ~25 MB peak RAM for 5 000-frame inputs.
3. **`apply_lowpass_filter`** — Butterworth order 4, SOS form, `sosfiltfilt`. Skips with warning if `cutoff >= nyquist` or `n_frames <= padlen = 3·(2·n_sections+1)`.
4. **`apply_temporal_diff`** — `np.diff(data, n=1, axis=0)`; output is `(N-1, n_active)`; skipped (with warning) if fewer than 2 frames.
5. **PCA fit / transform** — `PCA(n_components=min(req, N-1, n_features))`; prints explained variance %.
6. **Scaler** — `'standard'` (default) or `'minmax'`. `'minmax'` is supported but never used by training scripts; trees / SVM all use `standard`.

`fit_transform(...)` and `fit_from_recordings([cm, cm, …])`:
- The latter is the training-canonical path. Builds null mask from concatenated raw frames, then applies steps 2–4 **per recording** so Butterworth edge transients and the temporal-diff `(N-1)` shrinkage never cross recording boundaries. PCA + scaler are fit on `np.vstack(per_recording_blocks)`.

`transform(complex_matrix)`:
- Validates `n_subcarriers == self._fitted_n_subcarriers`; raises `ValueError` with explicit hardware/null-mask hint if mismatch.
- Runs all 6 steps in order, returns `(N-1, n_pca)` matrix.

> **Cross-check for the AI reviewer:** confirm PCA fits on the *concatenated post-diff* features (so `n_features_in_ = active_mask.sum()`). Sample verification: `models/har/csi_pipeline.joblib` was fit on **114 active subcarriers** (per project memory), and `transform()` shape check enforces 128 raw subcarriers.

---

## 6. `csi_ml_pipeline.py` — training engine

### Feature engineering — `extract_features_from_window`
Input `(WINDOW_SIZE=100, n_pca=10) → 250-dim output`. **FEATURE_VECTOR_VERSION = "5"**, `N_STATS = 25`. Per PCA component, in this exact extraction order:

1. `mean`
2. `std` (+ 1e-8)
3. `max`
4. `min`
5. `range` = max − min
6. `median`
7. `energy` = Σ col²
8. `skewness` = mean(((col-μ)/σ)³)
9. `excess_kurtosis` = mean(((col-μ)/σ)⁴) − 3 (note: "excess", not raw)
10. `fft_mean` over **active band only** (`freqs ≤ cutoff_hz = 10 Hz` ⇒ ~11 bins)
11. `fft_std` over active band
12. `zcr` over `col − mean` (sign changes / (N−1))
13. `fft_peak_idx` = `(argmax(no_dc)+1) / n_bins_active` — **normalised 0…1 within active band** (10 bins after dropping DC)
14. `spectral_entropy` = − Σ p·log₂(p+ε) over active band (no DC)
15. `autocorr_peak` = max of lag-1+ autocorrelation
16. `autocorr_dominant_lag` = `(argmax+1)/fs` (seconds)
17. `gait_band_ratio` = Σ FFT-power ∈ [0.5,3] Hz / total FFT-power
18. `spectral_centroid` weighted by `fft_vals_no_dc` (active band, no DC)
19. `peak_prominence` = `max(no_dc) − mean(no_dc)` (active band)
20. `signal_mobility` (Hjorth) = std(diff) / std(col)
21. `signal_complexity` (Hjorth) = mobility(diff) / mobility(col)
22. `waveform_length` = Σ |diff(col)|
23. `impulse_ratio` = max(|col|) / mean(|col|) — crest factor (added v5)
24. `burst_duration` = fraction of samples > 0.5·max(|col|) (FWHM proxy)
25. `rise_fall_ratio` = |rise_slope| / |fall_slope|, slopes around `argmax(|col|)`

DWT features are coded but disabled (`_DWT_STATS_PER_COMPONENT = 0`). Re-enable requires `WINDOW_SIZE ≥ 100` AND `cutoff ≥ 25 Hz` — comment says so explicitly.

Feature naming (used in importance plots): `PC{c+1}_{stat}` (e.g. `PC2_waveform_length`). Group classification in `_STAT_TO_GROUP`:
- **Statistical (10):** mean, std, max, min, range, median, energy, skewness, excess_kurtosis, zcr, impulse_ratio
- **FFT (7):** fft_mean, fft_std, fft_peak_idx, spectral_entropy, gait_band_ratio, spectral_centroid, peak_prominence
- **Temporal (7):** autocorr_peak, autocorr_dominant_lag, signal_mobility, signal_complexity, waveform_length, burst_duration, rise_fall_ratio

### Augmentation — physics-aware, class-aware
All augmentations are applied **on the raw post-(null/Hampel/Butterworth/diff) signal**, BEFORE PCA projection. Then each augmented window is independently projected through the *training-fit* PCA + scaler.

| Technique | Rule |
|---|---|
| `noise` | Gaussian noise ~ `U[0.003,0.01]·std(signal)`. `fall` → ×0.5; `sit` → ×0.7 (defensive — both classes disabled currently). |
| `shift` | Edge-padded forward/backward shift of `randint(1,4)` frames (never circular — avoids discontinuities). |
| `scale` | Magnitude scale: `walk/default` `U[0.9,1.1]`; `sit` `U[0.95,1.05]`; `fall` `U[0.97,1.03]`. |
| `time_warp` | Linear-interp warp with factor `walk_activity` `U[0.9,1.1]`, `sit` `U[0.95,1.05]`, else `U[0.98,1.02]`. Reflect-padded overflow. **Disabled for `fall`** (gravity invariance). |

`augment_window(window, n_augments=4, …)` picks 1 or 2 techniques per pass, repeats `n_augments` times → produces 4 augmented copies. If `class_label == 'fall'` filters out `time_warp`; if everything is filtered, falls back to `['noise']` with a `RuntimeWarning`.

### Dataset construction — `build_dataset(...)`
- Resolves enabled classes via `config.resolve_training_classes`.
- File-level test split: `random.Random(seed).shuffle(files)`, then last `max(1, len(files)*test_ratio)` go to test.
- The `experiment_config.json` saved alongside models stores **the exact `train_files` and `test_files` per class** as `posix paths relative to data_dir` — fully reproducible.
- Fits `CSIPipeline` via `fit_from_recordings` on **train recordings only** (no leakage).
- For TRAIN: per file → raw_pre_pca (null/Hampel/butter/diff) → sliding windows → project (PCA+scaler) → features `feat_orig` recorded with `train_groups_orig = recording_group_id`. Then augmentation: augment(raw) → project → features (no group assigned — augmented samples are NOT used in CV grouping).
- For TEST: `pipeline.transform(cm)` end-to-end (no augmentation). Sliding windows → features.
- Returns 10 arrays + `LabelEncoder` + `pipeline` + `dataset_info` dict.

### CV builder — `_make_group_cv`
- Prefers `StratifiedGroupKFold` (sklearn ≥ 1.3 with `shuffle=True, random_state=seed`).
- Falls back to `GroupKFold` if class group counts < 2.
- `n_splits = min(requested, min(class_group_counts), n_unique_groups)`.

### Hyperparameter tuning — `tune_hyperparameters`
GridSearchCV on **non-augmented** `X_train_orig` with the group-aware splitter. Grids:
- SVM (rbf, balanced, probability=True): `C∈{1,10,100}`, `gamma∈{scale,auto,0.01,0.001}`.
- RF (balanced, n_jobs=-1): `n_estimators∈{100,200,300}`, `max_depth∈{10,15,20,None}`, `min_samples_leaf∈{1,2,4}`.
- ET: same grid as RF.
- KNN: `n_neighbors∈{3,5,7,9}`, `weights∈{uniform,distance}`, `metric∈{euclidean,manhattan}`.
- LR (l2 lbfgs balanced): `C∈{0.1,1,10,100}`.
- GB: `n_estimators∈{100,200}`, `learning_rate∈{0.05,0.1,0.2}`, `max_depth∈{3,5}`.
- MLP: `hidden_layer_sizes∈{(100,),(100,50),(50,50)}`, `alpha∈{0.0001,0.001,0.01}`, `learning_rate∈{constant,adaptive}`.

NB has no grid (uses defaults).

### Model construction in `train_and_evaluate`
- Scalers are bundled INSIDE the sklearn `Pipeline` for `svm`, `knn`, `lr`, `mlp` only — because the CSIPipeline scaler is fitted on the **pre-feature-extraction** PCA outputs, but the per-window engineered features (energy, ZCR, …) are NOT scaled by CSIPipeline. Tree-based models (`rf`, `et`, `gb`, `nb`) are NOT wrapped (correct: they’re scale-invariant).
- Default constructor args (used when `--no-tune` or hyperparams not found):
  - SVM: `C=10, gamma=scale, kernel=rbf, balanced, probability=True`.
  - RF: `n_estimators=200, max_depth=15, min_samples_leaf=2, balanced, n_jobs=-1, seed=42`.
  - ET: `n_estimators=200, max_depth=None, min_samples_leaf=1, balanced`.
  - KNN: `n_neighbors=5, weights=distance, metric=euclidean`.
  - LR: `C=1.0, l2, lbfgs, max_iter=1000, balanced`.
  - GB: `n_estimators=100, learning_rate=0.1, max_depth=3`.
  - MLP: `hidden=(100,), alpha=0.0001, max_iter=500`.
  - NB: `GaussianNB()` (no params).
- **CV is run on `X_train_orig` (non-augmented)**, final fit on `X_train` (augmented). Train accuracy reported on `X_train_orig` (not the trivial near-100% augmented value).

### Persisted artifacts on `--save_model`
Inside `models_dir/`:
- `csi_pipeline.joblib`
- `label_encoder.joblib`
- `<model>.joblib` for each trained classifier
- `metrics.json` — per-model `cv_accuracy_mean`, `cv_accuracy_std`, `cv_scores` (5 floats), `train_accuracy`, `test_accuracy`, `test_f1_macro`, `confusion_matrix`, `classes`, `feature_importances` (top-10 only when model exposes them), `cv_splitter`, `feature_vector_version`.
- `experiment_config.json` — `data_dir`, `requested_classes`, `classes`, `pipeline_kwargs`, `window_size`, `step`, `test_recording_ratio`, `random_seed`, `n_pca`, `cutoff`, `simulation_mode`, **full `train_files` + `test_files`** per class (relative posix paths), `augment_techniques`, `n_augments`, `cv_folds`, `target_model`.

### CLI surface
`--data_dir`, `--classes`, `--window_size`, `--step`, `--fs`, `--augment` (list), `--use-augment / --no-augment`, `--n_augments`, `--pca`, `--test_ratio`, `--diff / --no-diff`, `--simulate / --no-simulate`, `--save_model / --no-save_model`, `--tune / --no-tune`, `--model` (list, includes `all`), `--seed`, `--cv_folds`, `--cutoff`, `--models_dir`, `--features` (22…25, defaults to 25).

---

## 7. `live_predict.py` — terminal-only inference

- Reads serial line by line (`pyserial.readline()`), parses with `parse_csi_line`, accumulates a deque of `maxlen = WINDOW_SIZE + FILTER_WARMUP = 100 + 50 = 150`.
- Inference gate: `frames_since_pred >= step` (default `PREDICTION_STEP_SIZE = 10`) — independent of dropped frames.
- Each inference:
  1. `cm = np.vstack(buffer)` → (≤150, 128).
  2. Pre-flight shape check vs `pipeline._fitted_n_subcarriers`; bail on mismatch with hint.
  3. `pipeline.transform(cm, cutoff)` → (≤149, n_pca) (temporal diff drops one).
  4. `processed[-window_size:]` → (100, n_pca).
  5. `extract_features_from_window(...)` → (1, 250). Auto-detects `n_stats` from `model.n_features_in_ // pca.n_components_`.
  6. Guard against non-finite features.
  7. `predict_proba` (or `predict` fallback) → top label + confidence.
- Smoothing: `Counter(pred_history).most_common(1)[0][0]` over last `args.history = 3` raw predictions (no probability EMA in this script — pure majority).
- FPS via `RollingFPS(maxlen=60)`; observed FPS vs `pipeline.fs` compared after warmup — warns if >20 % off ("FFT features will be at wrong frequencies — retrain").
- **Feature-vector version mismatch warning** at load time by comparing `FEATURE_VECTOR_VERSION` to all `feature_vector_version` values in `metrics.json`.
- CLI: `-p/--port`, `-b/--baud`, `--models_dir` (defaults to `models/`), `--model` (one of MODEL_KEYS), `--window`, `--step`, `--history`, `--verbose / --no-verbose`, `--color / --no-color`, `--fps-window`, `--warmup`, `--rx-buf`, `--cutoff`.

---

## 8. `live_dashboard.py` — production live GUI (3-process design)

### Architecture
| Process | Role | Communication |
|---|---|---|
| GUI (main) | Qt event loop, renders Monitor / Signal / Log / System / Settings pages at `--refresh ms` (default `10 ms` ≈ 100 Hz). | reads `shared_memory` via mp.Lock, polls output Queue. |
| Reader | Owns serial port (or demo), pushes raw frames into shared memory + assembles inference windows. | Writes structured numpy view of `shared_memory.SharedMemory(create=True)`; pushes `(cm, frame_count)` into `mp.Queue(maxsize=2)` — newest only (drops stale with `get_nowait` first). |
| Inference | Pulls cm from input queue, runs `pipeline.transform` + features + `model.predict_proba`. | Pushes `(raw_label, conf%, probs, latency_ms, frame_count)` into `mp.Queue(maxsize=8)`. |

Reader → Inference Queue is supplied by reader and **handed** to InferenceProcess so that the GUI thread never touches the data path. Confirmed in `ReaderProcess.infer_in_q` / `InferenceProcess.__init__`.

### Shared-state dtype (`_make_shared_dtype`)
Single struct view via `np.dtype([...])`:
- `n_active i4`, `frame_count i8`, `wave_ptr i4`, `variance f8`, `mean_amp f8`, `connected i4`, `fps_n i8`, `fps_times f8[60]`, `wave_buf f4[waveform_len]`, `last_amp f4[max_subcarriers]`.

EMA-style accumulators inside the reader:
- `state["variance"] = prev_var * 0.97 + (ma - prev_ma)² * 0.03`
- `state["mean_amp"] = prev_ma * 0.97 + ma * 0.03`

FPS is computed by the GUI snapshot using `min(fps_n, 60)` most-recent timestamps.

### Reader scheduling inside `_reader_process_fn`
Pushes window to inference every `step` frames and only when `len(deque) >= window_size`. **`step` default in dashboard = `DASHBOARD_STEP_SIZE = 15` frames (150 ms)**, distinct from training step (50) and `live_predict` step (10).

### OS-level tuning
- `sys.setswitchinterval(0.001)` — tightens GIL switch from 5 ms to 1 ms.
- Windows: `winmm.timeBeginPeriod(1)` — 1 ms timer resolution.
- Reader child also sets `winmm.timeBeginPeriod(1)` and `SetThreadPriority HIGHEST`.

### `_inference_worker_fn`
- Auto-detects `n_stats` from `model.n_features_in_ // pipeline.pca.n_components_`; fallback `_N_STATS_DEFAULT = 25`.
- Re-imports `extract_features_from_window` inside the child (Windows spawn).
- Subcarrier mismatch warned ONCE (`_mismatch_warned`).
- Feature-count mismatch with `model.n_features_in_` warns inline.
- Errors silently swallowed for the prediction; (raw_cand, conf_cand, probs_cand, latency, frame_count) is always emitted (None for "no prediction").

### GUI hooks / mode switch
Two MODES selectable in `SettingsPage`:
- **HAR Mode** — loads `config.MODELS_HAR_DIR = "models/har"`.
- **Super Motion** — loads `config.MODELS_MOTION_DIR = "models/motion"`.

Switching mode calls `InferenceProcess.restart(...)` with the new pipeline + model + LE, no GUI restart.

### CLI flags (`_parse_args`)
`-p/--port`, `--baud`, `--models_dir` (default `models/har`), `--model` (default `rf`), `--window` (default `100`), `--step` (default `15`), `--ema-alpha` (default `0.6`), `--conf-thresh` (default `70.0`), `--waveform-len` (default `200`), `--refresh` (default `10`), `--max-log` (default `60`), `--rx-buf` (default `2_000_000`), `--cutoff` (default `10.0`), `--warmup` (default `50`), `--hyst-count` (default `2`), `--demo / --no-demo` (default `False`).

### Pages
- `MonitorPage` — `ActivityBlock` (gradient + accent + label), `ConfidenceGauge` (custom QPainter arc), `StatusPill` (animated, live > 10 frames), `MotionIntensityBar`, live waveform plot.
- `SignalViewPage` — subcarrier power + waveform + signal metrics.
- `ActivityLogPage` — session summary, distribution, full log (capped at `--max-log`).
- `SystemInfoPage` — OS, CPU, Python, NumPy, scikit-learn versions.
- `SettingsPage` — pick mode (HAR / Motion), pick model from radio buttons, click *Deploy* to live-switch inference.

> **CRITICAL OBSERVATION FOR SYSTEM:**
> Project memory (~31 days old) claims `FallEventDetector` lives in `live_dashboard.py` after `InferenceProcess`. **It does not.** A grep over the entire project for `FallEvent|fall_detection|FALL_` returns 0 hits. No fall-detection alert banner exists in `MonitorPage`. The `fall` class is also still disabled in `TRAINING_CLASS_CONFIG`. Either the feature was removed/never landed in `main`, or this is stale memory. Cross-check vs git log: `git log --oneline -- live_dashboard.py`.

---

## 9. `live_csi_dual_view.py` and `live_sensing_1.py`

- `live_csi_dual_view.py` — two-pane Qt viewer; threaded reader (NOT mp.Process — uses `threading.Thread`); buffers amplitude & phase circularly (`buffer_size=200`, refresh 33 ms ≈ 30 Hz). Strides subcarriers to `_MAX_LINES = 32` lines per axis (selects every `n // 32`-th SC).
- `live_sensing_1.py` — single-pane waveform with cyan→amber→red gradient and 3-layer glow (44/18/7 px line widths). Energy = RMS of last 10 first-diff samples on the normalized waveform. Reader uses `deque(maxlen=60)` of mean amplitude across all non-zero subcarriers (DC/null ignored).

Both also support `--demo` for offline UI testing. Neither performs ML inference.

---

## 10. `benchmark_latency.py`

- Loads `models/<model>.joblib` if present, else builds a *fresh* untrained fallback (and warns).
- Fails over to `args.file` for the inference window:
  - With `--simulate` → synthetic `complex64` from RNG.
  - Else loads `pipeline_path = models_dir/csi_pipeline.joblib`; if missing, fits fresh on `args.file` data (fallback path).
  - `buf_size = window_size + FILTER_WARMUP = 150` — matches `live_predict.py`.
- For each model: 10 warmup runs + 50 timed runs; measures full path = `pipeline.transform → features → model.predict`. RAM delta measured around the first `predict`.
- Reports mean & p95 latency, throughput (inf/sec), RAM delta. Saves CSV + bar plot to `models/multi_model_latency.csv` and `models/plots/Latency_Comparison.png` if `--save`.
- Catches **stale models** by comparing `model.n_features_in_` to current feature vector length; **also** compares `feature_vector_version` in `metrics.json` vs the constant.

---

## 11. `models/har/` artifacts (the production HAR set)

### `experiment_config.json`
- `data_dir`: absolute path on the developer machine.
- `classes`: `["no_activity","walk_activity"]`.
- `pipeline_kwargs`: `{fs:100, use_diff:true}`.
- `window_size:100`, `step:50`, `test_recording_ratio:0.2`, `random_seed:42`, `n_pca:10`, `cutoff:10.0`, `simulation_mode:false`.
- `augment_techniques`: `["noise","shift","scale","time_warp"]`, `n_augments:4`, `cv_folds:5`, `target_model:"all"`.
- `train_files["no_activity"]` — **40 files**, including 10 `livroom_*` (older living-room recordings) + 30 `room1_vasilis_*`.
- `train_files["walk_activity"]` — **40 files**, parallel structure.
- `test_files["no_activity"]` — **10 files** (recordings 12, 16, 17, 18, 19, 25, 26, 28, 45 + 1 `livroom_01`).
- `test_files["walk_activity"]` — **10 files** (matching indices + 1 `livroom_01`).

Total training corpus: **80 recordings**. Total test: **20 recordings**. Class balance is perfect 50/50 at the recording level.

### `metrics.json` — model-by-model (Hold-out test on 2 189 windows = 1 098 no_activity + 1 091 walk_activity)

| Model | CV mean ± std | CV scores | Train acc | Test acc | Test F1-macro | Top feature (importance) |
|---|---|---|---|---|---|---|
| **gb** | 95.65 % ± 2.48 % | 97.42/93.84/97.33/98.00/91.66 | 99.02 % | **97.17 %** | 97.17 % | PC2_waveform_length (91.6 %) |
| **et** | 95.41 % ± 2.88 % | 97.80/91.52/97.52/97.94/92.28 | 98.17 % | **97.12 %** | 97.12 % | PC2_waveform_length (6.2 %) |
| **rf** | 95.79 % ± 2.46 % | 97.30/93.72/97.52/98.38/92.03 | 99.77 % | **96.98 %** | 96.98 % | PC1_fft_mean (8.0 %) |
| **lr** | 95.01 % ± 3.21 % | 97.61/91.39/97.59/97.69/90.79 | 98.05 % | **96.85 %** | 96.85 % | n/a (linear) |
| **svm** | 95.15 % ± 2.83 % | 96.48/91.65/97.65/98.12/91.85 | 98.08 % | **96.80 %** | 96.80 % | n/a |
| **mlp** | 94.52 % ± 2.91 % | 96.10/91.14/97.02/97.50/90.85 | 100.00 % | **96.53 %** | 96.53 % | n/a |
| **knn** | 92.96 % ± 4.42 % | 93.65/89.82/97.14/97.94/86.25 | 100.00 % | **93.56 %** | 93.56 % | n/a |
| **nb** | 62.31 % ± 7.42 % | 65.60/51.13/73.71/59.31/61.79 | 64.80 % | **61.26 %** | 55.27 % | n/a — feature distributions clearly non-Gaussian |

Confusion (test, no_activity vs walk_activity):
- GB: [[1059, 39], [23, 1068]] — FN_walk=23 (best recall on walk).
- ET: [[1060, 38], [25, 1066]] — close second.
- RF: [[1062, 36], [30, 1061]].
- SVM: [[1062, 36], [34, 1057]].
- KNN: [[1043, 55], [86, 1005]].
- NB: [[1071, 27], [821, 270]] — catastrophic walk recall (collapse).

`cv_splitter` = `"StratifiedGroupKFold"` for every model. `feature_vector_version` = `"5"` everywhere. All 8 models are aligned.

### `models/metrics.json` (top-level)
**Byte-for-byte identical to `models/har/metrics.json`.** Likely a leftover from when training wrote into `models/` directly; the HAR mode artifacts are now duplicated. Recommend audit-cleanup but it does not affect inference (dashboard reads from `models/har`).

### Note on KNN file size
`models/har/knn.joblib` = **40 MB**, vs SVM 6.4 MB, ET 2.7 MB, RF 3.8 MB. KNN stores the full training set (lazy method) — expected.

---

## 12. `models/motion/` artifacts (Super Motion mode)

### `experiment_config.json`
- Same data_dir, same 80/20 split, same seed, same classes — only difference:
- `window_size:40`, `step:8` (400 ms windows, 80 ms step → way denser overlap).
- `target_model:["rf","gb","et","lr"]` (only 4 trained).

### `metrics.json` — Hold-out test on 13 799 windows = 6 928 no_activity + 6 871 walk_activity

| Model | CV mean ± std | Train acc | Test acc | Test F1 | Top feature |
|---|---|---|---|---|---|
| **gb** | 95.24 % ± 3.15 % | 97.82 % | **96.65 %** | 96.65 % | PC2_waveform_length (77.3 %) |
| **et** | 95.24 % ± 3.27 % | 99.36 % | **96.37 %** | 96.37 % | PC4_min (3.6 %) |
| **lr** | 94.98 % ± 2.83 % | 97.28 % | **96.25 %** | 96.25 % | n/a |
| **rf** | 95.34 % ± 3.13 % | 99.97 % | **96.22 %** | 96.22 % | PC2_waveform_length (8.2 %) |

GB collapses to one feature (PC2_waveform_length 77 % importance) — flagged as a potential overfit-to-one-projection risk. ET and RF spread importance more evenly. **Worth the AI reviewer's attention:** GB's 77 % single-feature concentration in motion mode and 91.6 % in HAR mode means PC2_waveform_length is functionally a hand-coded gait detector. Confirm this is desirable in the thesis defense or rebalance via feature subsampling.

### File sizes
- `et.joblib` = **140 MB** (the giant — large number of windows × 200 trees with max_depth=None makes Extra Trees blow up).
- `rf.joblib` = 30 MB; GB = 142 KB; LR = 9 KB; pipeline = 7 KB.

> Motion mode does NOT include SVM/KNN/MLP/NB. Dashboard `SettingsPage` will still attempt to load `<model>.joblib` for any selected MODEL_KEYS — the AI reviewer should verify the GUI handles a missing-file gracefully (looking at `_load_models` → `raise FileNotFoundError`; the dashboard would crash on switch to a non-available model in motion mode).

---

## 13. Datasets — exact recording details

### Counts and naming
- `datasets/no_activity/` — **50 .txt recordings** (+ derived .csv with the header injected, + auto-generated .png plots).
- `datasets/walk_activity/` — **50 .txt recordings** (+ .csv, + .png, + 1 `*_thesis_plots/` subdirectory).
- **No `.meta.json` sidecars** in either folder — recordings were not captured via `--mode`, and no SPACE markers were pressed.

### Filename conventions observed
- `no_activity_NN_room1_vasilis_<unix>.txt` (where NN = 11…50; recording 11 = first in current corpus, recording 50 = last).
- `no_activity_livroom_NN_<unix>.txt` (older living-room sessions, NN = 01…10).
- `walk_activity_NN_room1_vasilis_<unix>.txt` (NN = 11…50).
- `walk_activity_livroom_NN_vasilis_<unix>.txt` (NN = 01…10; note variation: one file is `walk_activity_livroom_09_vasilis1776097226.txt` — missing `_` between `vasilis` and unix; another is `walk_activity_42_room1_vasilis_1777806213____.txt` — trailing underscores). These naming inconsistencies **do not affect** training because file selection is glob-based on `*.txt` and class membership is by parent folder.
- One file `walk_activity_livroom_01_vasilis_.txt` (TRAILING UNDERSCORE, no timestamp) — referenced explicitly by `config.DEFAULT_WALK_FILE`. **the AI reviewer:** check that this file exists; the benchmark falls back gracefully if not, but `live_predict.py`/training do not use it.

### Duration / frame count (expected vs measured)
- Preset says: walk = 60 s, idle = 120 s. Bytes: walk avg 4.34 MB, idle avg 4.35 MB.
- A single CSI line including 128-subcarrier payload is roughly 700–900 bytes → ~5 000 frames per ~4 MB file → consistent with 50–100 s @ 100 Hz. Some shorter files exist (min 0.79 MB ≈ ~10 s of walk; min 1.36 MB ≈ ~15 s of idle).
- **the AI reviewer check:** `csi_ml_pipeline.build_dataset` will silently skip recordings with `cm.size == 0` or `< WINDOW_SIZE` frames after diff (≤99 raw frames). Confirm short files are tolerated and not over-represented.

### Train/test class split (exact files)
- `experiment_config.json` lists every file by path. Held-out recordings (NEVER seen by training):
  - `no_activity_12, 16, 17, 18, 19, 25, 26, 28, 45 + livroom_01`
  - `walk_activity_12, 16, 17, 18, 19, 25, 26, 28, 45 + livroom_01`
- The other 80 recordings are used for training + augmentation. Random shuffle with `seed=42` is reproducible.

---

## 14. Tests (`tests/`)

- Contains **only** `__pycache__/` with compiled bytecode:
  - `conftest.cpython-311-pytest-9.0.3.pyc`
  - `test_csi_parser.cpython-311-pytest-9.0.3.pyc`
  - `test_data_preprocessing.cpython-311-pytest-9.0.3.pyc`
  - `test_extract_features.cpython-311-pytest-9.0.3.pyc`
  - `test_reader_process.cpython-311-pytest-9.0.3.pyc`
- **No `.py` source files** in `tests/`. The tests have been removed or never committed — they cannot be re-run. This is a real reproducibility gap for a thesis.

---

## 15. Cross-script invariants — what MUST stay aligned

| Constant | Where it lives | Cross-checked locations |
|---|---|---|
| `WINDOW_SIZE=100` | `config.py` | training (`csi_ml_pipeline`), `live_predict`, `benchmark_latency`, `live_dashboard` (default), `models/har/experiment_config.json` ✅ |
| `step=50` (training), `=10` (live_predict), `=15` (dashboard) | `config.py` | These intentionally differ — *every* doc must distinguish them. |
| `N_PCA_COMPONENTS=10` | `config.py` | pipeline fit + feature vector length × 25 = 250 ✅ |
| `FILTER_WARMUP=50` | `config.py` | buffer = window + warmup = 150 in live + benchmark ✅ |
| `FILTER_CUTOFF_HZ=10.0` | `config.py` | Butterworth + feature `active_band` mask ✅ |
| `FEATURE_VECTOR_VERSION="5"` | `csi_ml_pipeline.py` | stamped into every metrics.json record (verified for all 8 HAR + 4 motion models) ✅ |
| `n_features_in_` per model | inside `.joblib` | must equal 250 for every model in HAR mode. Auto-detected in live_predict / dashboard / benchmark via `n_features_in_ // pca.n_components_`. |
| `_fitted_n_subcarriers` of pipeline | inside `csi_pipeline.joblib` | must equal `cm.shape[1]` from serial = 128 ✅ |
| `RANDOM_SEED=42` | `config.py` | training shuffle, splitter, augmentations ✅ |

---

## 16. Known inconsistencies / stale state / open audit questions

These are the things the AI reviewer should specifically verify:

1. **`tests/` has no .py sources.** Only `.pyc` files. Either restore the source or document why they're gone.
2. **Stale memory: `FallEventDetector`** — referenced in older project memory but **does not exist** in current `live_dashboard.py`. Verify with `git log -- live_dashboard.py | head -30`.
3. **Stale memory: `220-dim` feature vector (v4)** — current is **250-dim (v5)**. `FEATURE_VECTOR_VERSION = "5"` confirmed in source and in every saved `metrics.json`.
4. **`live_data_visualization.py` is missing** but referenced in `config.SCRIPT_DEFAULTS` and `README.md`. Either restore or delete the orphan defaults.
5. **`models/metrics.json` (top-level)** duplicates `models/har/metrics.json`. Inference path uses `models/har`; the duplicate just clutters the tree.
6. **GB feature concentration**: GB is the best HAR test-acc model (97.17 %) but assigns **91.6 % importance to a single feature** (`PC2_waveform_length`). In Motion mode it's **77.3 %**. For a thesis defense, justify why this is desirable (or run GB with `max_features` < 1.0 / use shrinkage to spread importance).
7. **NB collapse**: NB gets only 61.26 % test accuracy — features clearly violate the Gaussian-independence assumption. It's documented but the AI reviewer should confirm whether keeping NB in `MODELS_TO_TRAIN="all"` is desirable.
8. **`PLOT_DATA_AUGMENTATION_SUBCARRIER = 30`** — the augmentation visualisation script picks subcarrier index 30, which is INSIDE the active band but should be sanity-checked against `active_mask` (some active-mask indices change file by file).
9. **`step (live_dashboard) = 15` vs `step (live_predict) = 10`** — two live paths predict at different rates without a shared rationale documented.
10. **`benchmark_latency.py` line 192** help text says "Number of frames per inference window (default: 100)" — the actual default IS 100 (`WINDOW_SIZE`), so this is now consistent. (Older audit noted "(default: 50)" — already fixed.)
11. **`csi_parser.first_word_invalid` scrub** zeroes the first 4 raw values when `parts[13] != 0`. Sample line 1 of `walk_activity_11_*.txt` has `first_word=11` AND a 12-element zero guard band AT THE END of the payload — verify the scrub direction matches the ESP32 bandwidth/channel configuration in use.
12. **Motion mode missing models**: `models/motion/` lacks `svm/knn/mlp/nb`. Dashboard `SettingsPage` will fail to switch to those in Super Motion mode. Either restrict the UI choices when in Motion mode, or train the remaining models.
13. **`requirements.txt` has no version pins.** Sklearn ≥ 1.3 is required for `StratifiedGroupKFold(shuffle, random_state)`. The fallback `GroupKFold` works on older sklearn but loses stratification.
14. **`get_known_training_classes`** in `config.py` is defined but never called anywhere in the repo (dead helper).
15. **`PREPROCESSING_USE_DIFF = True`** in `config.py` and `pipeline_kwargs.use_diff = True` in `experiment_config.json` are consistent. Always verify after any DSP change.

---

## 17. Reproducing the headline results

Exact commands that should regenerate `models/har/`:
```bash
python csi_ml_pipeline.py \
  --data_dir ./datasets \
  --classes walk_activity no_activity \
  --window_size 100 --step 50 --fs 100 \
  --augment noise shift scale time_warp --use-augment --n_augments 4 \
  --pca 10 --test_ratio 0.2 --diff \
  --tune --model all \
  --seed 42 --cv_folds 5 --cutoff 10 \
  --models_dir models/har --features 25 --save_model
```
Exact commands for `models/motion/`:
```bash
python csi_ml_pipeline.py \
  --data_dir ./datasets \
  --classes walk_activity no_activity \
  --window_size 40 --step 8 --fs 100 \
  --augment noise shift scale time_warp --use-augment --n_augments 4 \
  --pca 10 --test_ratio 0.2 --diff \
  --tune --model rf gb et lr \
  --seed 42 --cv_folds 5 --cutoff 10 \
  --models_dir models/motion --features 25 --save_model
```

Live deployment:
```bash
python live_dashboard.py --port COM6 --models_dir models/har --model gb
# or:
python live_predict.py -p COM6 --models_dir models/har --model gb -v
```

Latency benchmark (run on the same machine that will host inference):
```bash
python benchmark_latency.py --file datasets/walk_activity/walk_activity_11_room1_vasilis_1777802888.txt --save
```

---

## 18. Things to ask the AI reviewer to validate

Hand this section to the AI reviewer verbatim:

1. Walk through `data_preprocessing.CSIPipeline.fit_from_recordings` for the **HAR experiment config** (80 train recordings, mixed `room1` + `livroom`). Confirm:
   - `_fitted_n_subcarriers == 128`.
   - `active_mask.sum()` is plausible (project memory says ~114).
   - PCA `explained_variance_ratio_.sum()` is reasonable (>70 %).
2. Reproduce one inference pass on a held-out test file:
   - Load `walk_activity_16_room1_vasilis_1777803445.txt` → `pipeline.transform` → `extract_features_from_window(processed[-100:])` → predict with each model.
   - Verify the resulting (class, confidence) is consistent across models.
3. Reconstruct `metrics.json` numbers from `experiment_config.json` to confirm the train/test split was honored.
4. Audit `csi_ml_pipeline.train_and_evaluate` for **scaling consistency**:
   - SVM/KNN/LR/MLP wrap StandardScaler INSIDE the pipeline — correct.
   - Trees + NB unwrapped — correct.
   - Confirm CSIPipeline's StandardScaler is fit ONLY on training data (it is, via `fit_from_recordings`).
5. Verify the `experiment_config.json` `data_dir` is an **absolute** developer path — the AI reviewer should NOT rely on it for path resolution. Use the `posix relative` paths in `train_files`/`test_files`.
6. Audit `_inference_worker_fn` for race conditions when `restart(...)` is called from the GUI (`InferenceProcess.restart` → `stop` → `start` with new pipeline). Confirm no half-poisoned queue state.
7. Sanity-check that **the dashboard's `step=15`** matches what the operator actually feels at runtime — 6.67 Hz predictions vs `live_predict`'s 10 Hz step (10 Hz predictions). Document the rationale or unify.
8. Re-run `benchmark_latency.py --simulate` and compare to the saved `models/multi_model_latency.csv` (if any). Flag drift.
9. For the thesis, recompute hold-out class balance: 1 098 / 2 189 ≈ 50.16 % no_activity. Confirm there's no slight skew that would inflate accuracy. (It does not — close to balanced.)
10. The **HAR best model is GB (97.17 %)**, the **dashboard default is RF (96.98 %)**. RF was likely chosen for stability (less single-feature concentration) — verify with the user whether GB should be the deployed default.

---

## 19. Final attachment-list (everything in the box)

Code files (root): 22 `.py` + `README.md` + `requirements.txt`.
Models (HAR): `csi_pipeline.joblib`, `label_encoder.joblib`, `svm.joblib`, `rf.joblib`, `et.joblib`, `knn.joblib`, `lr.joblib`, `gb.joblib`, `mlp.joblib`, `nb.joblib`, `metrics.json`, `experiment_config.json`.
Models (Motion): same minus svm/knn/mlp/nb.
Datasets: 50 `no_activity/*.txt` + 50 `walk_activity/*.txt` (~217 MB each). No sidecars.
Plots: `models/plots/` — populated only when training/benchmark/figures scripts are run with `--save`.
Tests: empty source, compiled `.pyc` only — **not runnable** without restoring source.

— END OF AUDIT —
