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
ROOT = Path(__file__).resolve().parent.parent.parent / "Python_Scripts_for_csi_recv"
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
DEFAULT_MODELS    = r"C:\Diplomatiki_2026\WIFI CSI PROJECT\Python_Scripts_for_csi_recv\models"
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
            "port":          "",
            "baud":          0,
            "model_name":    "",
            "pca_dims":      0,
            "window_size":   50,
            "start_time":    time.time(),
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
        self._reload    = threading.Event()
        self.new_model_key = model_key
        self.pipeline   = None
        self.le         = None
        self.model      = None

    def update_model(self, model_key: str):
        with state.lock:
            self.new_model_key = model_key
            self._reload.set()

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
        import serial.tools.list_ports
        target_port = self.port

        def try_connect(p):
            try:
                s = serial.Serial(p, self.baud, timeout=0.5)
                if os.name == "nt" and hasattr(s, "set_buffer_size"):
                    s.set_buffer_size(rx_size=SERIAL_BUF_MB)
                s.reset_input_buffer()
                return s
            except Exception: return None

        while not self._stop.is_set():
            ser = try_connect(target_port)
            if ser is None:
                ports = list(serial.tools.list_ports.comports())
                for p in ports:
                    if "Bluetooth" in p.description: continue
                    ser = try_connect(p.device)
                    if ser is not None:
                        target_port = p.device
                        break

            if ser is None:
                err = "Hardware not detected. Waiting for ESP32..."
                self.loop.call_soon_threadsafe(lambda: state.update({"connected": False, "error": err}))
                for _ in range(30):
                    if self._stop.is_set(): return
                    time.sleep(0.1)
                continue

            print(f"Serial hardware detected: {target_port} @ {self.baud}")
            
            # Inner loop for reloading model without losing serial connection
            while not self._stop.is_set() and not self._reload.is_set():
                self._reload.clear()
                pipeline, le, model = self._load_models()
                self.pipeline, self.le, self.model = pipeline, le, model
                
                m_name = MODEL_FILES.get(self.model_key, "Unknown").replace(".joblib", "").replace("_", " ")
                p_dims = pipeline.pca.n_components_ if pipeline and hasattr(pipeline, "pca") and pipeline.pca else 0
                self.loop.call_soon_threadsafe(
                    lambda: state.update({
                        "connected": True, "error": "", 
                        "port": target_port, "baud": self.baud,
                        "model_name": m_name, "pca_dims": p_dims,
                        "window_size": WINDOW_SIZE
                    })
                )

                buf_size, buffer = WINDOW_SIZE + FILTER_WARMUP, deque(maxlen=WINDOW_SIZE + FILTER_WARMUP)
                pred_history, fps_times = deque(maxlen=HISTORY), deque(maxlen=FPS_WINDOW)
                frame_count, frames_since = 0, 0

                try:
                    while not self._stop.is_set() and not self._reload.is_set():
                        raw = ser.readline()
                        if not raw: continue
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if "CSI_DATA" not in line: continue
                        frame = parse_csi_line(line)
                        if frame is None: continue

                        buffer.append(frame)
                        frame_count += 1
                        frames_since += 1
                        fps_times.append(time.monotonic())

                        label, conf, prob_dict = "Hardware Live", 1.0, {}
                        if self.pipeline and self.model and self.le:
                            res = self._infer(buffer, self.pipeline, self.model, self.le)
                            if res: label, conf, prob_dict = res
                        
                        if frames_since < STEP and frame_count >= buf_size: continue
                        frames_since = 0

                        pred_history.append(label)
                        smoothed = Counter(pred_history).most_common(1)[0][0]
                        fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0]) if len(fps_times) >= 2 else 0.0

                        # Waveform & SC Map logic
                        amp_arr = np.abs(np.vstack(buffer))
                        active_amp = amp_arr[:, 6:50] if amp_arr.shape[1] > 50 else amp_arr
                        waveform = np.mean(active_amp, axis=1).tolist()[-60:]
                        sc_map = amp_arr[-1, 6:63].tolist() if amp_arr.shape[1] >= 63 else amp_arr[-1].tolist()
                        
                        if waveform: waveform = [v / (max(waveform) or 1.0) for v in waveform]
                        if sc_map: sc_map = [v / (max(sc_map) or 1.0) for v in sc_map]

                        payload = {
                            "label":          label,
                            "smoothed":       smoothed,
                            "confidence":     round(conf, 4),
                            "probabilities":  prob_dict,
                            "fps":            round(fps, 1),
                            "latency":        0 if not self.pipeline else 10,
                            "loss":           0.0,
                            "frame_count":    frame_count,
                            "waveform":       waveform,
                            "subcarrier_map": sc_map,
                            "connected":      True,
                            "error":          "",
                            "timestamp":      time.monotonic(),
                        }
                        self.loop.call_soon_threadsafe(lambda p=payload: self._post(p))

                    if self._reload.is_set():
                        print(f"🔄 Reloading model to: {self.new_model_key}")
                        self.model_key = self.new_model_key
                        self._reload.clear()
                except Exception as e:
                    print(f"Hardware Loop Error: {e}")
                    break
            
            if ser and ser.is_open:
                ser.close()
            time.sleep(1)

    def _post(self, payload: dict):
        state.update(payload)
        state.new_event.set()


    def stop(self):
        self._stop.set()

app = FastAPI(title="CSI HAR Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_connections = []
_conn_lock = asyncio.Lock()
_reader_thread = None
_training_process = None # Global handle for the training process

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

@app.get("/api/models")
async def list_models():
    args = _parse_args()
    models_dir = Path(args.models_dir)
    trained = []
    for key, filename in MODEL_FILES.items():
        if (models_dir / filename).exists():
            trained.append(key)
    return {"trained_models": trained, "all_models": list(MODEL_FILES.keys())}

@app.post("/api/deploy")
async def deploy_model(data: dict):
    model_key = data.get("model")
    if model_key not in MODEL_FILES:
        return {"success": False, "error": "Invalid model key"}
    if _reader_thread:
        _reader_thread.update_model(model_key)
        async with _conn_lock:
            for ws in _connections:
                try: await ws.send_text(json.dumps({"event": "model_deployed", "model": model_key}))
                except: pass
        return {"success": True}
    return {"success": False, "error": "Inference thread not running"}

@app.post("/api/train")
async def train_model(params: dict):
    """Execute the ML training pipeline script with full parameters."""
    import subprocess
    import os
    
    # Ensure data_dir is absolute relative to ROOT if it's not already absolute
    data_dir = params.get("data_dir", "datasets")
    if not os.path.isabs(data_dir):
        data_dir = str(ROOT / data_dir)
    
    cmd = [sys.executable, str(ROOT / "csi_ml_pipeline.py")]
    cmd += ["--data_dir", data_dir]
    
    # Map other settings
    if params.get("classes"): cmd += ["--classes"] + params["classes"]
    if params.get("model"): cmd += ["--model", params["model"]]
    if params.get("window_size"): cmd += ["--window_size", str(params["window_size"])]
    if params.get("step"): cmd += ["--step", str(params["step"])]
    if params.get("fs"): cmd += ["--fs", str(params["fs"])]
    if params.get("pca"): cmd += ["--pca", str(params["pca"])]
    if params.get("test_ratio"): cmd += ["--test_ratio", str(params["test_ratio"])]
    if params.get("seed"): cmd += ["--seed", str(params["seed"])]
    
    if params.get("no_augment"):
        cmd += ["--no_augment"]
    elif params.get("augment"):
        cmd += ["--augment"] + params["augment"]
        if params.get("n_augments"): cmd += ["--n_augments", str(params["n_augments"])]
    
    if not params.get("use_diff"): cmd += ["--no_diff"]
    
    printable_cmd = ' '.join(['"' + c + '"' if ' ' in c else c for c in cmd])
    print(f"🚀 Starting Training: {printable_cmd}")
    
    try:
        # Clear existing log if any
        log_path = ROOT / "training.log"
        if log_path.exists():
            try: os.remove(log_path)
            except: pass
            
        # Redirect output to a fresh log file
        log_file = open(log_path, "w", encoding="utf-8")
        
        global _training_process
        # Use shell=True for Windows to handle spaces correctly in command line
        _training_process = subprocess.Popen(
            cmd, 
            stdout=log_file, 
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            bufsize=1, # Line buffered
            universal_newlines=True,
            shell=True
        )
        return {
            "success": True, 
            "message": f"Training started. Monitoring log...",
            "command": " ".join(cmd)
        }
    except Exception as e:
        print(f"❌ Training Failed to Start: {e}")
        return {"success": False, "error": f"Failed to start training script: {str(e)}"}

@app.get("/api/train/status")
async def train_status():
    """Check the status of the latest training run and return the log tail."""
    log_path = ROOT / "training.log"
    
    # If log doesn't exist yet, it's still starting
    if not log_path.exists():
        return {"running": True, "log": "Initializing training..."}
    
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            log_content = f.read()
            log_tail = log_content[-2000:] # Get last 2k chars
        
        # Detection logic
        has_error = "Error" in log_tail or "ValueError" in log_tail or "Traceback" in log_tail
        is_complete = "Execution Complete" in log_tail or "manually stopped" in log_tail
        
        # If the log is very short and doesn't have much content, it's definitely still running
        if len(log_content) < 100 and not has_error:
            return {"running": True, "log": log_tail}

        return {
            "running": not (has_error or is_complete),
            "log": log_tail
        }
    except Exception as e:
        return {"running": False, "log": f"Error reading log: {str(e)}"}

@app.post("/api/train/stop")
async def stop_training():
    """Terminate the currently running training process."""
    global _training_process
    print("🛑 Stop request received...")
    if _training_process:
        try:
            # On Windows with shell=True, we must kill the child processes tree
            if os.name == 'nt':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(_training_process.pid)], capture_output=True)
            
            try: _training_process.kill()
            except: pass
            
            _training_process = None
            # Append a note to the log
            log_path = ROOT / "training.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n\n Training manually stopped by user.\n")
            
            return {"success": True, "message": "Training killed."}
        except Exception as e:
            print(f"Error killing process: {e}")
            _training_process = None 
            return {"success": True, "message": "Attempted to kill process."}
    return {"success": False, "error": "No process found."}

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
