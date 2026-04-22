#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSI HAR  FastAPI WebSocket Server
====================================
Bridges live_predict.py inference with the React dashboard.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import threading
from collections import deque, Counter
from pathlib import Path

import numpy as np
import serial
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

#  Local imports 
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def parse_csi_line(line: str):
    """
    Built-in parser for ESP32 CSI strings.
    Handles format: CSI_DATA,...,"[re, im, re, im, ...]"
    """
    if "CSI_DATA" not in line: return None
    try:
        if "[" not in line or "]" not in line: return None
        parts = line.split("[")
        data_str = parts[-1].split("]")[0]
        # Clean quotes and whitespace
        data_str = data_str.replace('"', '').strip()
        vals = [int(v.strip()) for v in data_str.split(",") if v.strip()]
        if len(vals) < 2: return None
        # Convert pairs to complex
        complex_data = [complex(vals[i], vals[i+1]) for i in range(0, len(vals), 2)]
        return np.array(complex_data, dtype=np.complex64)
    except Exception:
        return None

try:
    from data_preprocessing import CSIPipeline       # noqa
    from csi_ml_pipeline import extract_features_from_window
    import joblib
    _IMPORTS_OK = True
except ImportError:
    _IMPORTS_OK = False

# 
# CONFIG
# 

DEFAULT_PORT      = "COM6" if os.name == "nt" else "/dev/ttyUSB0"
DEFAULT_BAUD      = 2000000
DEFAULT_MODELS    = "./models"
DEFAULT_MODEL     = "svm"
WINDOW_SIZE       = 50
FILTER_WARMUP     = 20
STEP              = 10
HISTORY           = 3
SERIAL_BUF_MB     = 2000000
FPS_WINDOW        = 60

MODEL_FILES = {
    "rf":  "Random_Forest.joblib",
    "svm": "SVM_RBF.joblib",
    "et":  "Extra_Trees.joblib",
    "knn": "K-NN_k=5.joblib",
    "lr":  "Logistic_Regression.joblib",
    "gb":  "Gradient_Boosting.joblib",
    "mlp": "MLP_Neural_Network.joblib",
    "nb":  "Naive_Bayes.joblib",
}

# 
# SHARED STATE
# 

class SharedState:
    def __init__(self):
        self.lock            = threading.Lock()
        self.frame_count     = 0
        self.connected       = False
        self.error           = ""
        self.last_prediction = {
            "label":         "idle",
            "smoothed":      "idle",
            "confidence":    0.0,
            "probabilities": {},
            "fps":           0.0,
            "latency_ms":    0,
            "packet_loss":   0.0,
            "frame_count":   0,
            "waveform":      [0.0] * 60,
            "subcarrier_map": [0.0] * 57,
            "connected":     False,
            "error":         "",
            "timestamp":     time.time(),
        }
        self.new_event = asyncio.Event()

    def update(self, payload: dict):
        with self.lock:
            self.last_prediction.update(payload)

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.last_prediction)

state = SharedState()

# 
# SERIAL READER + INFERENCE
# 

class CSIReaderThread(threading.Thread):
    def __init__(self, port: str, baud: int,
                 models_dir: str, model_key: str,
                 loop: asyncio.AbstractEventLoop):
        super().__init__(daemon=True, name="CSIReader")
        self.port       = port
        self.baud       = baud
        self.models_dir = Path(models_dir)
        self.model_key  = model_key
        self.loop       = loop
        self._stop      = threading.Event()

    def _load_models(self):
        if not _IMPORTS_OK:
            return None, None, None
        d = self.models_dir
        pipeline_path = d / "csi_pipeline.joblib"
        le_path       = d / "label_encoder.joblib"
        model_path    = d / MODEL_FILES.get(self.model_key, "SVM_RBF.joblib")
        for p in [pipeline_path, le_path, model_path]:
            if not p.exists():
                return None, None, None
        pipeline = joblib.load(pipeline_path)
        le       = joblib.load(le_path)
        model    = joblib.load(model_path)
        return pipeline, le, model

    def _infer(self, buffer, pipeline, model, le):
        try:
            cm = np.vstack(buffer).astype(np.complex64)
            data = pipeline.remove_null_subcarriers(cm, fit=False)
            data = pipeline.apply_lowpass_filter(data)
            if pipeline.use_diff:
                data = pipeline.apply_temporal_diff(data)
            if pipeline.pca is not None:
                data = pipeline.pca.transform(data)
            processed = pipeline.scaler.transform(data)
            if processed.shape[0] < WINDOW_SIZE:
                return None
            window   = processed[-WINDOW_SIZE:]
            features = extract_features_from_window(window).reshape(1, -1)
            if hasattr(model, "predict_proba"):
                probs      = model.predict_proba(features)[0]
                idx        = int(np.argmax(probs))
                confidence = float(probs[idx])
                prob_dict  = {str(le.classes_[i]): float(p) for i, p in enumerate(probs)}
            else:
                idx        = int(model.predict(features)[0])
                confidence = 1.0
                prob_dict  = {str(le.classes_[i]): 0.0 for i in range(len(le.classes_))}
                prob_dict[str(le.classes_[idx])] = 1.0
            label = str(le.inverse_transform([idx])[0])
            return label, confidence, prob_dict
        except Exception:
            return None

    def run(self):
        pipeline, le, model = self._load_models()
        ser = None
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.5)
            if os.name == "nt" and hasattr(ser, "set_buffer_size"):
                ser.set_buffer_size(rx_size=SERIAL_BUF_MB)
            ser.reset_input_buffer()
            print(f"Serial hardware detected: {self.port} @ {self.baud}")
            self.loop.call_soon_threadsafe(
                lambda: state.update({"connected": True, "error": ""})
            )
        except Exception as e:
            print(f"Hardware not detected on {self.port}: {e}")
            if pipeline is None:
                self._run_simulation()
                return
            else:
                err = f"Serial error: {e}"
                self.loop.call_soon_threadsafe(
                    lambda: state.update({"connected": False, "error": err})
                )
                return

        buf_size        = WINDOW_SIZE + FILTER_WARMUP
        buffer          = deque(maxlen=buf_size)
        pred_history    = deque(maxlen=HISTORY)
        fps_times       = deque(maxlen=FPS_WINDOW)
        frames_since    = 0
        frame_count     = 0

        try:
            while not self._stop.is_set():
                raw = ser.readline()
                if not raw: continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if "CSI_DATA" not in line: continue
                frame = parse_csi_line(line)
                if frame is None: continue

                buffer.append(frame)
                frame_count   += 1
                frames_since  += 1
                fps_times.append(time.monotonic())

                label, conf, prob_dict = "Hardware Live", 1.0, {}
                if pipeline and model and le:
                    res = self._infer(buffer, pipeline, model, le)
                    if res:
                        label, conf, prob_dict = res
                
                if frames_since < STEP and frame_count >= buf_size:
                    continue
                frames_since = 0

                pred_history.append(label)
                smoothed = Counter(pred_history).most_common(1)[0][0]
                fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0]) if len(fps_times) >= 2 else 0.0

                # Waveform snapshot (using active subcarriers mean)
                amp_arr   = np.abs(np.vstack(buffer))
                # Skip first 6 null subcarriers, take mean of active ones
                active_amp = amp_arr[:, 6:50] if amp_arr.shape[1] > 50 else amp_arr
                waveform   = np.mean(active_amp, axis=1).tolist()[-60:]
                # Heatmap: take a slice of active carriers
                sc_map     = amp_arr[-1, 6:63].tolist() if amp_arr.shape[1] >= 63 else amp_arr[-1].tolist()
                
                # Auto-scale for visualization
                if waveform:
                    mx = max(waveform) or 1.0
                    waveform = [v / mx for v in waveform]
                if sc_map:
                    mx = max(sc_map) or 1.0
                    sc_map = [v / mx for v in sc_map]

                payload = {
                    "label":          label,
                    "smoothed":       smoothed,
                    "confidence":     round(conf, 4),
                    "probabilities":  prob_dict,
                    "fps":            round(fps, 1),
                    "latency_ms":     0 if not pipeline else 10,
                    "packet_loss":    0.0,
                    "frame_count":    frame_count,
                    "waveform":       waveform,
                    "subcarrier_map": sc_map,
                    "connected":      True,
                    "error":          "",
                    "timestamp":      time.monotonic(),
                }
                self.loop.call_soon_threadsafe(lambda p=payload: self._post(p))
        except Exception as e:
            print(f"Hardware Loop Error: {e}")
        finally:
            if ser and ser.is_open:
                ser.close()

    def _post(self, payload: dict):
        state.update(payload)
        state.new_event.set()

    def _run_simulation(self):
        import math
        rng       = np.random.default_rng(42)
        walk_conf = 0.80
        frame_c   = 0
        t0        = time.monotonic()
        while not self._stop.is_set():
            time.sleep(1.8)
            walk_conf += float(rng.uniform(-0.12, 0.12))
            walk_conf  = float(np.clip(walk_conf, 0.52, 0.99))
            idle_conf  = 1.0 - walk_conf
            label      = "walk" if walk_conf > 0.5 else "idle"
            frame_c   += STEP
            elapsed   = max(time.monotonic() - t0, 1e-6)
            fps       = frame_c / elapsed
            wf  = [float(abs(math.sin(i * 0.3 + time.time()))) * rng.uniform(0.4, 1.0) for i in range(60)]
            sc  = [float(rng.uniform(0.1, 1.0)) for _ in range(57)]
            payload = {
                "label":          label,
                "smoothed":       label,
                "confidence":     round(max(walk_conf, idle_conf), 4),
                "probabilities":  {"walk": round(walk_conf, 4), "idle": round(idle_conf, 4)},
                "fps":            round(fps, 1),
                "latency_ms":     int(rng.integers(15, 35)),
                "packet_loss":    round(float(rng.uniform(0.5, 2.5)), 2),
                "frame_count":    frame_c,
                "waveform":       wf,
                "subcarrier_map": sc,
                "connected":      True,
                "error":          "",
                "timestamp":      time.monotonic(),
            }
            self.loop.call_soon_threadsafe(lambda p=payload: self._post(p))

    def stop(self):
        self._stop.set()

app = FastAPI(title="CSI HAR Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_connections = []
_conn_lock = asyncio.Lock()
_reader_thread = None

@app.on_event("startup")
async def startup():
    global _reader_thread
    loop = asyncio.get_event_loop()
    args = _parse_args()
    _reader_thread = CSIReaderThread(port=args.port, baud=args.baud, models_dir=args.models_dir, model_key=args.model, loop=loop)
    _reader_thread.start()
    print(f"CSI HAR Server ready on {args.ws_port}")

@app.on_event("shutdown")
async def shutdown():
    if _reader_thread:
        _reader_thread.stop()

@app.get("/api/status")
async def get_status():
    return state.snapshot()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    async with _conn_lock:
        _connections.append(ws)
    print("Client connected")
    try:
        await ws.send_text(json.dumps(state.snapshot()))
        while True:
            state.new_event.clear()
            try:
                await asyncio.wait_for(state.new_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"heartbeat": True, "timestamp": time.monotonic()}))
                continue
            await ws.send_text(json.dumps(state.snapshot()))
    except WebSocketDisconnect:
        pass
    finally:
        async with _conn_lock:
            if ws in _connections: _connections.remove(ws)
        print("Client disconnected")

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port",       default=DEFAULT_PORT)
    p.add_argument("--baud",       type=int, default=DEFAULT_BAUD)
    p.add_argument("--models-dir", default=DEFAULT_MODELS)
    p.add_argument("--model",      default=DEFAULT_MODEL, choices=list(MODEL_FILES.keys()))
    p.add_argument("--host",       default="127.0.0.1")
    p.add_argument("--ws-port",    type=int, default=8000)
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    uvicorn.run("server:app", host=args.host, port=args.ws_port, reload=False, workers=1)
