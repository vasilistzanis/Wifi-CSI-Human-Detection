#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSI Real-Time Inference Script (Thesis Grade — Grand Finale)
============================================================
Reads live CSI data from ESP32 via Serial, applies the preprocessing
pipeline, and classifies human activity using the trained BiLSTM model.

Features:
  - 2 Mbps Serial ingestion (threaded, zero dropped frames)
  - Live preprocessing: null removal → Hampel → Butterworth → diff
  - Rolling window inference (stride-based, low latency)
  - Console UI with per-class probability bars

Usage:
  python real_time_inference.py --port COM6
  python real_time_inference.py --port COM6 --model-dir ./output
  python real_time_inference.py --port COM6 --stride 30 --confidence-thresh 0.7
"""

import os
import sys
import time
import json
import argparse
import threading
from collections import deque
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# ✅ FIX Bug 1: _parse_recv_row does NOT exist in data_preprocessing.
# We import only what actually exists there. Line parsing is done here
# using the same logic as csi_plotter_heatmap.py (which is what motion_detector uses).
from data_preprocessing import CSIPipeline, _build_complex_frame


def configure_console_output() -> None:
    """Avoid UnicodeEncodeError on legacy Windows console encodings."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass


configure_console_output()

# ════════════════════════════════════════════════════════════════════════
# ARGS
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="CSI Real-Time HAR Inference")
    p.add_argument("--port",               type=str,   default=None,
                   help="Serial port (e.g. COM6 or /dev/ttyUSB0). Auto-detects if omitted.")
    p.add_argument("--baud",               type=int,   default=2_000_000,
                   help="Baud rate (default: 2000000)")
    p.add_argument("--model-dir",          type=str,   default=".",
                   help="Directory containing best_model.keras + training_config.json")
    p.add_argument("--stride",             type=int,   default=50,
                   help="Predict every N new frames (default: 50 ≈ 0.5 s at 100 Hz)")
    p.add_argument("--confidence-thresh",  type=float, default=0.6,
                   help="Min probability to declare a class (default: 0.6)")
    p.add_argument("--calibration-s",      type=float, default=2.0,
                   help="Seconds of empty-room calibration before inference (default: 2)")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════
# CSI LINE PARSING
# (Same logic as csi_plotter_heatmap.py — kept local, no circular import)
# ════════════════════════════════════════════════════════════════════════

RECV_FIELD_COUNT = 15

def _parse_csi_line(line: str):
    """
    Parse one CSI_DATA line into a complex64 array.
    Returns None if the line is malformed.

    IQ convention (ESP32 buf): [imag0, real0, imag1, real1, ...]
    complex(i) = real[i] + j*imag[i]
    """
    if not line.startswith("CSI_DATA"):
        return None

    # Split into exactly 15 CSV fields
    parts = [p.strip() for p in line.strip().split(",", RECV_FIELD_COUNT - 1)]
    if len(parts) != RECV_FIELD_COUNT:
        return None

    # Validate numeric fields (seq, rssi, … len, first_word)
    for idx in (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13):
        try:
            int(parts[idx])
        except ValueError:
            return None

    # Extract data payload: field 14, strip quotes and brackets
    payload = parts[14].strip().strip('"')
    if not (payload.startswith("[") and payload.endswith("]")):
        return None
    payload = payload[1:-1].strip()
    if not payload:
        return None

    # Parse int array
    try:
        raw = [int(v) for v in payload.split(",")]
    except ValueError:
        return None

    first_word = int(parts[13])
    return _build_complex_frame(raw, bool(first_word))


# ════════════════════════════════════════════════════════════════════════
# CONFIG + MODEL LOADING
# ════════════════════════════════════════════════════════════════════════

def load_config_and_model(model_dir: str):
    d = Path(model_dir)
    config_file = d / "training_config.json"
    model_file  = d / "best_model.keras"

    for f, name in [(config_file, "training_config.json"), (model_file, "best_model.keras")]:
        if not f.exists():
            print(f"❌ Not found: {f}")
            sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    required = ["frames", "subcarriers", "classes"]
    for key in required:
        if key not in config:
            print(f"❌ training_config.json missing key: '{key}'")
            sys.exit(1)

    print(f"📂 Config  : {config_file.name}")
    print(f"🧠 Model   : {model_file.name}")
    model = tf.keras.models.load_model(str(model_file))
    model.trainable = False

    # Warm-up: allocate TF graph so first real prediction is not slow
    dummy = np.zeros((1, config["frames"], config["subcarriers"]), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print("   Warm-up done.")

    return config, model


# ════════════════════════════════════════════════════════════════════════
# SERIAL READER THREAD
# ════════════════════════════════════════════════════════════════════════

class CSISerialReader:
    """
    Background thread that reads CSI_DATA lines from serial and appends
    parsed complex64 frames to a shared deque (thread-safe via a Lock).
    """

    def __init__(self, port: str, baud: int, buffer: deque, lock: threading.Lock):
        self.port   = port
        self.baud   = baud
        self.buffer = buffer
        self.lock   = lock
        self._running = threading.Event()
        self._thread  = None
        self.frames_received = 0
        self.last_error: str = ""

    def start(self):
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.5)
            self._ser.reset_input_buffer()
        except serial.SerialException as e:
            print(f"❌ Cannot open {self.port}: {e}")
            sys.exit(1)

        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="CSIReader")
        self._thread.start()
        print(f"📡 Connected: {self.port} @ {self.baud} bps")

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
        if hasattr(self, "_ser") and self._ser.is_open:
            self._ser.close()

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def _loop(self):
        while self._running.is_set():
            try:
                raw = self._ser.readline()
            except serial.SerialException as e:
                self.last_error = str(e)
                self._running.clear()
                break

            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            frame = _parse_csi_line(line)
            if frame is None:
                continue

            # ✅ FIX Bug 7: Lock around deque mutation for thread-safe snapshot
            with self.lock:
                self.buffer.append(frame)
                self.frames_received += 1


def auto_detect_port() -> str | None:
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").upper()
        if any(kw in desc for kw in ("USB", "UART", "SERIAL", "CP210", "CH340")):
            return p.device
    return ports[0].device if ports else None


# ════════════════════════════════════════════════════════════════════════
# CONSOLE UI
# ════════════════════════════════════════════════════════════════════════

_CLASS_COLORS = {
    "idle": "\033[92m",   # green
    "walk": "\033[93m",   # yellow
    "sit":  "\033[96m",   # cyan
    "fall": "\033[91m",   # red
}
_RESET = "\033[0m"
_GRAY  = "\033[90m"


def _color(cls_name: str) -> str:
    return _CLASS_COLORS.get(cls_name.lower(), "\033[97m")


def display(classes, probs, threshold, frames_rx, latency_ms):
    """Overwrite current console line with live inference result."""
    best = int(np.argmax(probs))
    conf = float(probs[best])

    if conf >= threshold:
        label = classes[best].upper()
        col   = _color(classes[best])
    else:
        label = "???"
        col   = _GRAY

    bars = []
    for cls, p in zip(classes, probs):
        filled = int(p * 20)
        bar = "█" * filled + "░" * (20 - filled)
        bars.append(f"{cls:<6}[{bar}]{p*100:5.1f}%")

    line = (
        f"\r{col}▶ {label:<22}{_RESET}"
        f"| {' | '.join(bars)}"
        f"| Rx:{frames_rx:<6} | {latency_ms:.0f}ms"
    )
    sys.stdout.write("\033[K" + line)
    sys.stdout.flush()


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    port = args.port or auto_detect_port()
    if not port:
        print("❌ No serial port found. Specify --port COMx")
        sys.exit(1)

    config, model = load_config_and_model(args.model_dir)

    WINDOW  = config["frames"]        # e.g. 300
    N_FEAT  = config["subcarriers"]   # e.g. 114
    CLASSES = config["classes"]

    print(f"\n   Window : {WINDOW} frames ({WINDOW/100:.1f} s at 100 Hz)")
    print(f"   Features: {N_FEAT} subcarriers")
    print(f"   Classes : {CLASSES}")
    print(f"   Stride  : {args.stride} frames")
    print(f"   Threshold: {args.confidence_thresh:.0%}")
    print("-" * 60)

    # Buffer holds WINDOW+1 frames so diff produces exactly WINDOW frames
    lock       = threading.Lock()
    raw_buffer = deque(maxlen=WINDOW + 1)
    reader     = CSISerialReader(port, args.baud, raw_buffer, lock)
    reader.start()

    # ── Preprocessing pipeline (null removal, Hampel, Butterworth only)
    # background_frames=0 and use_diff=False because:
    #   - We do temporal diff manually after the rolling window
    #   - No background subtraction in inference (stateless, can't estimate bg live)
    pipeline      = CSIPipeline(fs=100.0, background_frames=0, use_diff=False)
    mask_fitted   = False
    subcarrier_ok = None   # True once shape verified against config

    # ── Buffers for stability ──
    # [3] Probability smoothing (rolling average of last 5 predictions)
    prob_history = deque(maxlen=5)

    # Variables for static calibration Z-score
    calib_mean = None
    calib_std  = None

    # ── Auto-Calibration (Future Work / Optional) ──
    # idle_streak_frames = 0

    # ── Calibration: collect enough frames before starting ───────────────
    calib_frames = max(int(args.calibration_s * 100), WINDOW + 1)
    print(f"\n🟡 Calibrating — hold still for {args.calibration_s:.0f}s "
          f"({calib_frames} frames needed)...")

    while reader.frames_received < calib_frames:
        if not reader.running:
            print(f"\n❌ Serial disconnected during calibration: {reader.last_error}")
            sys.exit(1)
        time.sleep(0.05)

    # ✅ [2] STATIC CALIBRATION: Calculate noise floor (Mean/Std) from static room
    # This prevents the Z-score from amplifying noise during Idle states.
    with lock:
        calib_snapshot = list(raw_buffer)
    
    # Process calibration frames
    calib_mat = np.array(calib_snapshot, dtype=np.complex64)
    # Remove nulls, Hampel, Low-pass
    c_amp = pipeline.remove_null_subcarriers(calib_mat, fit=True)
    mask_fitted = True
    c_amp = pipeline.apply_hampel_filter(c_amp)
    c_amp = pipeline.apply_lowpass_filter(c_amp, cutoff=12.0)
    # Temporal Diff
    c_diff = np.diff(c_amp, n=1, axis=0).astype(np.float32)
    
    # Store noise floor stats
    calib_mean = c_diff.mean(axis=0, keepdims=True)
    calib_std  = c_diff.std(axis=0, keepdims=True) + 1e-6

    print("🟢 Calibration done. Live inference started.\n")

    last_inference_at = reader.frames_received

    try:
        while reader.running:
            current = reader.frames_received

            # Wait until we have stride new frames
            if current - last_inference_at < args.stride:
                time.sleep(0.005)
                continue

            t0 = time.perf_counter()

            # ✅ Check buffer size BEFORE acquiring lock to avoid holding
            # the lock while sleeping (which would block the serial reader thread).
            if len(raw_buffer) < WINDOW + 1:
                time.sleep(0.005)
                continue

            with lock:
                snapshot = list(raw_buffer)   # consistent copy under lock

            last_inference_at = current

            # Convert snapshot to complex matrix
            complex_matrix = np.array(snapshot, dtype=np.complex64)

            # Step 1: Null subcarrier removal
            # fit only once (mask is fixed by hardware — same every window)
            if not mask_fitted:
                amp = pipeline.remove_null_subcarriers(complex_matrix, fit=True)
                mask_fitted = True
            else:
                amp = pipeline.remove_null_subcarriers(complex_matrix, fit=False)

            # Step 2: Hampel spike removal
            amp = pipeline.apply_hampel_filter(amp)

            # Step 3: Butterworth low-pass (12 Hz)
            amp = pipeline.apply_lowpass_filter(amp, cutoff=12.0)

            # Step 4: Temporal diff — WINDOW+1 → WINDOW frames
            amp_diff = np.diff(amp, n=1, axis=0).astype(np.float32)

            # ✅ [2] Use static calibration scaling instead of dynamic window-based.
            # This ensures that an empty room results in near-zero inputs.
            window_scaled = (amp_diff - calib_mean) / calib_std  # (WINDOW, N_active)

            # ✅ FIX Bug 8: Check shape mismatch ONCE and exit cleanly
            if subcarrier_ok is None:
                n_active = window_scaled.shape[1]
                if n_active != N_FEAT:
                    print(f"\n❌ Subcarrier count mismatch: "
                          f"live={n_active} vs config={N_FEAT}.\n"
                          f"   Re-train with the same ESP32 configuration "
                          f"(same bandwidth, channel, receiver).")
                    sys.exit(1)
                subcarrier_ok = True

            if window_scaled.shape != (WINDOW, N_FEAT):
                # Transient (buffer not yet full), skip silently
                continue

            # ✅ [1] ULTRA-FAST INFERENCE: Call model directly instead of .predict()
            # This eliminates ~50-100ms of overhead for single samples.
            X         = window_scaled[np.newaxis, ...]   # (1, WINDOW, N_FEAT)
            probs_raw = model(X, training=False).numpy()[0]

            # ✅ [3] PROBABILITY SMOOTHING: Rolling average to prevent flickering results
            prob_history.append(probs_raw)
            probs_smooth = np.mean(list(prob_history), axis=0)

            # ── Auto-Calibration (Optional / Commented Out) ──
            # if CLASSES[int(np.argmax(probs_smooth))] == "idle" and float(np.max(probs_smooth)) > 0.95:
            #     idle_streak_frames += args.stride
            # else:
            #     idle_streak_frames = 0
            # 
            # # If idle for ~10 seconds (1000 frames)
            # if idle_streak_frames > 1000:
            #     calib_mean = amp_diff.mean(axis=0, keepdims=True)
            #     calib_std  = amp_diff.std(axis=0, keepdims=True) + 1e-6
            #     idle_streak_frames = 0

            latency_ms = (time.perf_counter() - t0) * 1000
            display(CLASSES, probs_smooth, args.confidence_thresh, current, latency_ms)

    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user.")
    finally:
        reader.stop()
        if reader.last_error:
            print(f"⚠️  Last serial error: {reader.last_error}")


if __name__ == "__main__":
    main()