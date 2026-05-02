#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESP32 CSI - Live HAR Prediction
=============================================
Reads CSI frames from serial, runs the trained pipeline + model in real-time,
and prints predictions to the terminal with a confidence bar.

Design goals:
  * Zero GUI dependencies (no Qt / Tk)
  * Single-threaded read loop - no locking overhead
  * Rolling deque buffer - O(1) frame push / O(1) pop
  * Reuses parse_csi_line from csi_parser  (no code duplication)
  * Reuses extract_features_from_window from csi_ml_pipeline

Usage:
  python live_predict.py -p COM6
  python live_predict.py -p /dev/ttyUSB0 --model svm --history 7
  python live_predict.py -p COM6 --window 50 --step 10 --models_dir ./models
  python live_predict.py -p COM6 --verbose          # show all class probabilities
  python live_predict.py -p COM6 --no-color         # plain text (for file logging)
"""

import argparse
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

from csi_parser import configure_console_output
configure_console_output()

import numpy as np
import serial

# --- Local imports ------------------------------------------------------------
try:
    from csi_parser import parse_csi_line
except ImportError:
    print("[ERROR]  csi_parser.py not found in the same directory.")
    sys.exit(1)

try:
    from data_preprocessing import CSIPipeline  # noqa: F401  (used via joblib load)
except ImportError:
    print("[ERROR]  data_preprocessing.py not found in the same directory.")
    sys.exit(1)

try:
    from csi_ml_pipeline import extract_features_from_window
except ImportError:
    print("[ERROR]  csi_ml_pipeline.py not found in the same directory.")
    sys.exit(1)

try:
    import joblib
except ImportError:
    print("[ERROR]  joblib not installed.  Run:  pip install joblib")
    sys.exit(1)


# --- Defaults -----------------------------------------------------------------
import config
_IS_WIN        = os.name == "nt"
DEFAULT_PORT   = config.SERIAL_PORT if _IS_WIN else "/dev/ttyUSB0"
DEFAULT_BAUD   = config.BAUD_RATE
WINDOW_SIZE    = config.WINDOW_SIZE      # frames per inference window (must match training)
STEP           = config.PREDICTION_STEP_SIZE      # predict every N new frames (lower = more frequent)
# Extra frames kept beyond window_size so the Butterworth filter has enough
# edge context.  padlen for 4th-order SOS ~ 3*(2*n_sections+1) ~ 27.
FILTER_WARMUP  = 50
SERIAL_BUF_MB  = config.RX_BUFFER_SIZE
# Rolling FPS window: measure rate over the last N frames
FPS_WINDOW     = 60


# --- ANSI helpers -------------------------------------------------------------
_ANSI = {
    "green":   "\033[92m",
    "yellow":  "\033[93m",
    "cyan":    "\033[96m",
    "magenta": "\033[95m",
    "blue":    "\033[94m",
    "red":     "\033[91m",
    "bold":    "\033[1m",
    "dim":     "\033[2m",
    "reset":   "\033[0m",
}
_CLASS_PALETTE = ["green", "yellow", "cyan", "magenta", "blue"]

# Will be set to False by --no-color or when stdout is not a TTY
_USE_COLOR: bool = True


def _enable_win_ansi() -> bool:
    """Enable VT100 escape codes on Windows 10+. Returns True on success."""
    if not _IS_WIN:
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        return True
    except Exception:
        return False


def _c(text: str, *styles: str) -> str:
    """Wrap text in ANSI escape codes (no-op when color is disabled)."""
    if not _USE_COLOR:
        return text
    codes = "".join(_ANSI[s] for s in styles if s in _ANSI)
    return f"{codes}{text}{_ANSI['reset']}"


def _bar(value: float, width: int = 20) -> str:
    """ASCII confidence bar  0..100 -> '########'"""
    filled = max(0, min(width, round(value / 100 * width)))
    return "#" * filled + "-" * (width - filled)


# --- Argument parsing ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ESP32 CSI - Live HAR Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-p", "--port",       default=DEFAULT_PORT,
                   help="Serial port (e.g. COM6). Optional for --demo.")
    p.add_argument("-b", "--baud",       type=int, default=DEFAULT_BAUD,
                   help="Baud rate")
    p.add_argument("--models_dir",       default="./models",
                   help="Directory containing joblib model files")
    p.add_argument("--model",            default="rf", 
                   choices=["rf", "svm", "et", "knn", "lr", "gb", "mlp", "nb"],
                   help="Which classifier to load (rf, svm, et, knn, lr, gb, mlp, nb)")
    p.add_argument("--window",           type=int, default=WINDOW_SIZE,
                   help="Frames per inference window (must match training)")
    p.add_argument("--step",             type=int, default=STEP,
                   help="Run inference every N new frames")
    p.add_argument("--history",          type=int, default=3,
                   help="Smoothing: majority vote over last N raw predictions")
    p.add_argument("--verbose", "-v",    action="store_true",
                   help="Show per-class probability breakdown each prediction")
    p.add_argument("--no-color",         action="store_true",
                   help="Disable ANSI colors (useful for logging to file)")
    p.add_argument("--fps-window",       type=int, default=FPS_WINDOW,
                   help="Rolling FPS window: measure rate over the last N frames")
    p.add_argument("--warmup",           type=int, default=FILTER_WARMUP,
                   help="Extra frames kept for filter warmup context")
    p.add_argument("--rx-buf",           type=int, default=SERIAL_BUF_MB,
                   help="Serial RX buffer size in bytes")
    p.add_argument("--cutoff",           type=float, default=10.0,
                   help="Butterworth filter cutoff frequency (Hz)")
    return p.parse_args()


# --- Model loader -------------------------------------------------------------
def load_models(models_dir: str, model_choice: str):
    """
    Load pipeline, label-encoder, and chosen classifier from disk.
    Exits with a clear message if any file is missing.
    """
    d = Path(models_dir)
    model_files = {
        "rf": "rf.joblib",
        "svm": "svm.joblib",
        "et": "et.joblib",
        "knn": "knn.joblib",
        "lr": "lr.joblib",
        "gb": "gb.joblib",
        "mlp": "mlp.joblib",
        "nb": "nb.joblib"
    }
    files = {
        "pipeline": d / "csi_pipeline.joblib",
        "le":       d / "label_encoder.joblib",
        "model":    d / model_files.get(model_choice, "rf.joblib"),
    }
    for key, path in files.items():
        if not path.exists():
            print(f"[ERROR]  Missing file: {path}")
            print("   Train first:  python csi_ml_pipeline.py --save_model")
            sys.exit(1)

    pipeline = joblib.load(files["pipeline"])
    le       = joblib.load(files["le"])
    model    = joblib.load(files["model"])
    return pipeline, le, model


# --- Core inference -----------------------------------------------------------
def run_inference(
    buffer: deque,
    pipeline,
    model,
    le,
    window_size: int,
    cutoff: float = 10.0,
) -> "tuple[str, float, np.ndarray] | None":
    """
    Transform the current rolling buffer and return
    (top_label, top_confidence_pct, all_probs_array).

    Returns None if preprocessing fails or buffer is too short.

    Steps
    -----
    1. Stack raw complex frames  -> (N, n_subcarriers)
    2. pipeline.transform()      -> (N-1, n_pca)   [temporal diff reduces by 1]
    3. Take last `window_size` rows as inference window
    4. extract_features_from_window -> (110,) feature vector
    5. model.predict_proba        -> argmax + per-class probabilities
    """
    # 1. Stack - frames may rarely have inconsistent lengths if the ESP32
    #    sent a truncated packet just before a reset; catch that explicitly.
    try:
        cm = np.vstack(buffer).astype(np.complex64)         # (N, n_sub)
    except ValueError as exc:
        print(f"\n[WARNING]   Buffer vstack failed ({exc}) - skipping",
              file=sys.stderr)
        return None

    # 2. Preprocess using the unified transform method
    #    This ensures live inference exactly matches the training pipeline.
    try:
        # P2 Fix: Explicit shape check to avoid IndexError before transform
        if cm.shape[1] != pipeline._fitted_n_subcarriers:
            print(f"\n[ERROR] Subcarrier mismatch: got {cm.shape[1]}, "
                  f"expected {pipeline._fitted_n_subcarriers}", file=sys.stderr)
            return None

        # Call the unified transform (handles nulls, hampel, lowpass, diff, pca, scaler)
        processed = pipeline.transform(cm, use_pca=True, cutoff=cutoff)

    except ValueError as exc:
        # Most likely: subcarrier count mismatch between training and live data
        print(f"\n[ERROR]  Pipeline transform error: {exc}", file=sys.stderr)
        print("   Check ESP32 config (bandwidth/channel) matches training.",
              file=sys.stderr)
        return None
    except Exception as exc:
        print(f"\n[WARNING]   Unexpected pipeline error: {exc}", file=sys.stderr)
        return None

    # 3. Need at least window_size processed frames
    if processed.shape[0] < window_size:
        return None

    window = processed[-window_size:]                       # (window_size, n_pca)

    # 4. Feature extraction -> flat vector
    features = extract_features_from_window(window).reshape(1, -1)

    # Guard against corrupted data propagating to model
    if not np.all(np.isfinite(features)):
        print("\n[WARNING] Non-finite features detected - skipping", file=sys.stderr)
        return None

    # 5. Predict
    if hasattr(model, "predict_proba"):
        all_probs  = model.predict_proba(features)[0]       # (n_classes,)
        idx        = int(np.argmax(all_probs))
        confidence = float(all_probs[idx]) * 100.0
    else:
        # Fallback for models without probability support (e.g. LinearSVC)
        idx        = int(model.predict(features)[0])
        n_classes  = len(le.classes_)
        all_probs  = np.zeros(n_classes, dtype=np.float32)
        all_probs[idx] = 1.0
        confidence = 100.0

    label = str(le.inverse_transform([idx])[0])
    return label, confidence, all_probs


# --- Display ------------------------------------------------------------------
def print_compact(
    raw_label: str,
    confidence: float,
    smoothed: str,
    class_colors: dict,
    frame_count: int,
    fps: float,
    latency_ms: float,
) -> None:
    """Single-line overwriting display (default mode)."""
    color     = class_colors.get(smoothed, "green")
    raw_color = class_colors.get(raw_label, "green")
    bar       = _bar(confidence)

    line = (
        f"\r  {_c(f'{smoothed:<12}', color, 'bold')}"
        f"  [{bar}] {confidence:5.1f}%"
        f"  raw: {_c(f'{raw_label}', raw_color)}"
        f"  {_c(f'{frame_count:>6} fr', 'dim')}"
        f"  {_c(f'fps~{fps:5.1f}', 'dim')}"
        f"  {_c(f'{latency_ms:4.0f}ms', 'dim')}"
        f"   "    # trailing spaces clear any leftover chars from a longer prev line
    )
    print(line, end="", flush=True)


def print_verbose(
    raw_label: str,
    confidence: float,
    smoothed: str,
    all_probs: np.ndarray,
    classes: list,
    class_colors: dict,
    frame_count: int,
    fps: float,
    latency_ms: float,
) -> None:
    """
    Multi-line display showing all per-class probabilities.
    Prints a full block each prediction (scrolls, does not overwrite).
    """
    sep = "-" * 46
    print(f"\n{sep}")
    print(
        f"  frames: {frame_count:>6}  "
        f"fps~{fps:5.1f}  "
        f"latency: {latency_ms:.0f} ms"
    )
    for cls, prob in zip(classes, all_probs):
        pct    = float(prob) * 100.0
        color  = class_colors.get(cls, "green")
        bar    = _bar(pct, width=16)
        marker = _c(">>", color, "bold") if cls == smoothed else "  "
        print(
            f"  {marker}{_c(f'{cls:<12}', color)}  "
            f"[{bar}] {pct:5.1f}%"
        )
    raw_color = class_colors.get(raw_label, "green")
    print(
        f"  {sep}\n"
        f"  Smoothed -> {_c(smoothed, class_colors.get(smoothed,'green'), 'bold')}"
        f"   raw -> {_c(raw_label, raw_color)}"
        f"   conf {confidence:.1f}%"
    )


# --- Rolling FPS helper -------------------------------------------------------
class RollingFPS:
    """Compute FPS as frames/second over a sliding window of recent timestamps."""

    def __init__(self, maxlen: int = FPS_WINDOW):
        self._times: deque = deque(maxlen=maxlen)

    def tick(self) -> None:
        self._times.append(time.monotonic())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# --- Main loop ----------------------------------------------------------------
def main() -> int:
    global _USE_COLOR

    args = parse_args()

    # -- Color setup ------------------------------------------------------
    if args.no_color or not sys.stdout.isatty():
        _USE_COLOR = False
    elif _IS_WIN:
        _USE_COLOR = _enable_win_ansi()

    # -- Header -----------------------------------------------------------
    print("\n" + "=" * 52)
    print("  ESP32 CSI - Live HAR Prediction")
    print("=" * 52)

    pipeline, le, model = load_models(args.models_dir, args.model)
    classes      = list(le.classes_)
    class_colors = {
        cls: _CLASS_PALETTE[i % len(_CLASS_PALETTE)]
        for i, cls in enumerate(classes)
    }

    model_names = {
        "rf": "Random Forest",
        "svm": "SVM (RBF)",
        "et": "Extra Trees",
        "knn": "K-NN (k=5)",
        "lr": "Logistic Regression",
        "gb": "Gradient Boosting",
        "mlp": "MLP (Neural Network)",
        "nb": "Naive Bayes"
    }
    model_name = model_names.get(args.model, "Unknown Model")
    print(f"  Model   : {model_name}")
    print(f"  Classes : {classes}")
    print(f"  Port    : {args.port} @ {args.baud} baud")
    print(f"  Window  : {args.window} frames  "
          f"| Step: {args.step}  "
          f"| History: {args.history}")
    print(f"  Verbose : {args.verbose}")
    print("=" * 52)
    print("  Press Ctrl+C to stop.\n")

    # -- Rolling buffer ---------------------------------------------------
    # maxlen = window + warmup; deque auto-drops oldest frames when full
    buf_size = args.window + args.warmup
    buffer: deque = deque(maxlen=buf_size)

    pred_history: deque = deque(maxlen=args.history)
    fps_tracker  = RollingFPS(maxlen=args.fps_window)

    frame_count       = 0   # valid frames pushed to buffer
    frames_since_pred = 0   # counts valid frames since last inference
                            # (separate from frame_count so dropped serial
                            #   frames never cause a missed prediction step)
    warmup_done       = False

    # -- Open serial ------------------------------------------------------
    ser = None      # initialized to None so the finally block is always safe
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as exc:
        print(f"[ERROR]  Cannot open {args.port}: {exc}")
        return 1

    if _IS_WIN and hasattr(ser, "set_buffer_size"):
        try:
            ser.set_buffer_size(rx_size=args.rx_buf)
        except Exception:
            pass

    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    print(f"  Warming up - need {buf_size} frames before first prediction...")

    # -- Read loop --------------------------------------------------------
    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("CSI_DATA"):
                continue

            frame = parse_csi_line(line)
            if frame is None:
                continue

            buffer.append(frame)
            frame_count       += 1
            frames_since_pred += 1
            fps_tracker.tick()

            # -- Warmup progress bar --------------------------------------
            if not warmup_done:
                pct = min(frame_count / buf_size * 100, 100)
                bar = _bar(pct, width=20)
                print(
                    f"\r  Warming up  [{bar}] {frame_count}/{buf_size}",
                    end="", flush=True,
                )
                # Use >= (not ==) so we don't miss the trigger if a frame
                # happened to be dropped during warmup
                if frame_count >= buf_size:
                    warmup_done = True
                    print(
                        f"\r  [OK]  Buffer ready - live predictions starting!"
                        f"{'':30}"
                    )
                continue

            # -- Inference gate -------------------------------------------
            # Use a dedicated counter, NOT frame_count % step, so that any
            # dropped/unparseable serial frames don't accidentally skip a step.
            if frames_since_pred < args.step:
                continue
            frames_since_pred = 0

            t0         = time.monotonic()
            result     = run_inference(buffer, pipeline, model, le, args.window, cutoff=args.cutoff)
            latency_ms = (time.monotonic() - t0) * 1000.0

            if result is None:
                continue

            raw_label, confidence, all_probs = result
            pred_history.append(raw_label)
            smoothed = Counter(pred_history).most_common(1)[0][0]
            fps      = fps_tracker.fps

            if args.verbose:
                print_verbose(
                    raw_label, confidence, smoothed,
                    all_probs, classes, class_colors,
                    frame_count, fps, latency_ms,
                )
            else:
                print_compact(
                    raw_label, confidence, smoothed,
                    class_colors, frame_count, fps, latency_ms,
                )

    except serial.SerialException as exc:
        print(f"\n[ERROR]  Serial error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\n\n[INFO]   Stopped by user.")
    finally:
        # ser is None if serial.Serial() raised - guard against NameError
        if ser is not None and ser.is_open:
            ser.close()

    print(f"  Total valid frames : {frame_count}")
    print(f"  Rolling fps (last {args.fps_window} frames) : {fps_tracker.fps:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
