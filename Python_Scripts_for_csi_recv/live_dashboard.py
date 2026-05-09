#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSI Live HAR Dashboard — multi-page desktop app
================================================
Pages accessible from the left sidebar:
  Monitor      — live signal + activity inference + stats
  Signal View  — subcarrier power + waveform + signal metrics
  Activity Log — session summary + distribution + full log
  System Info  — hardware / software info cards

Custom painted widgets:
  ActivityBlock    — gradient background + colored accent bar
  ConfidenceGauge  — circular arc gauge (QPainter)
  StatusPill       — animated blinking status indicator

Usage
-----
  python live_dashboard.py
  python live_dashboard.py --port COM6 --model rf
  python live_dashboard.py --demo
"""

import argparse, math, multiprocessing as mp, os, sys, threading, time, warnings
from collections import Counter, deque
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from csi_parser import configure_console_output
configure_console_output()

import numpy as np
os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPainter, QPen,
)
from PyQt5.QtWidgets import (
    QApplication, QButtonGroup, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QPushButton, QProgressBar, QRadioButton,
    QStackedWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

import config
from plot_window_utils import center_qt_window

try:
    from csi_parser import parse_csi_line
    _PARSER_OK = True
except ImportError:
    _PARSER_OK = False

_INFERENCE_OK = False
_IMPORT_ERR   = ""
try:
    from data_preprocessing import CSIPipeline          # noqa: F401
    from csi_ml_pipeline import extract_features_from_window
    import joblib
    _INFERENCE_OK = True
except ImportError as _e:
    _IMPORT_ERR = str(_e)


# ============================================================================
# THEME
# ============================================================================
_BG     = "#0d1117"
_PANEL  = "#161b22"
_BORDER = "#30363d"
_DARK2  = "#21262d"
_TEXT   = "#c9d1d9"
_DIM    = "#8b949e"
_BLUE   = "#58a6ff"
_GREEN  = "#3fb950"
_YELLOW = "#d29922"
_RED    = "#f85149"
_ORANGE = "#f0883e"
_PURPLE = "#a371f7"

_CLASS_COLORS = {
    "empty": _BLUE,   "no_activity":  _GREEN,  "walk_activity":  _ORANGE,
    "idle":  _GREEN,  "walk":         _ORANGE,
    "sit":   _YELLOW, "fall":  _RED,    "stand": _PURPLE, "run": "#ff7b72",
}
_DISPLAY_NAMES = {
    "walk":          "walk/activity",
    "walk_activity": "walk/activity",
    "idle":          "no activity",
    "no_activity":   "no activity",
}
_MODEL_NAMES = {
    "rf": "Random Forest", "svm": "SVM (RBF)",    "et":  "Extra Trees",
    "knn": "K-NN",         "lr":  "Logistic Reg.", "gb":  "Gradient Boost",
    "mlp": "MLP Net",      "nb":  "Naive Bayes",
}
_MODEL_DESC = {
    "rf":  "Robust ensemble of decision trees. Good general accuracy.",
    "svm": "Support Vector Machine with RBF kernel. High CSI accuracy.",
    "et":  "Extra Trees — faster training, similar accuracy to RF.",
    "knn": "K-Nearest Neighbors. Simple, interpretable, low overhead.",
    "lr":  "Logistic Regression. Fastest inference, lower complexity.",
    "gb":  "Gradient Boosting. High accuracy, slower inference.",
    "mlp": "Neural network (MLP). Flexible non-linear feature learning.",
    "nb":  "Naive Bayes. Fastest, best for real-time constrained use.",
}

def _cc(label: str) -> str:
    return _CLASS_COLORS.get(label.lower(), _BLUE)

def _disp(label: str) -> str:
    return _DISPLAY_NAMES.get(label.lower(), label)

_QSS = f"""
QMainWindow, QWidget {{
    background:{_BG}; color:{_TEXT};
    font-family:'Segoe UI','Arial',sans-serif;
}}
QLabel  {{ color:{_TEXT}; background:transparent; }}
QWidget#sidebar {{ border-right:1px solid {_BORDER}; }}
QTableWidget {{
    background:{_PANEL}; border:1px solid {_BORDER}; border-radius:6px;
    gridline-color:{_BORDER}; color:{_TEXT}; font-size:11px;
}}
QTableWidget::item {{ padding:2px 4px; }}
QTableWidget::item:selected {{ background:{_DARK2}; }}
QHeaderView::section {{
    background:{_DARK2}; color:{_DIM}; border:none;
    border-bottom:1px solid {_BORDER}; padding:4px 8px;
    font-size:9px; font-weight:bold; letter-spacing:0.06em;
}}
QScrollBar:vertical {{ background:{_BG}; width:5px; border-radius:2px; }}
QScrollBar::handle:vertical {{ background:{_BORDER}; border-radius:2px; min-height:16px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0px; }}
QProgressBar {{ background:{_DARK2}; border:none; border-radius:3px; }}
QProgressBar::chunk {{ border-radius:3px; }}
"""
_NAV_ON  = (f"QPushButton{{background:{_DARK2}; color:{_BLUE}; font-weight:bold;"
            f" border-left:3px solid {_BLUE}; padding-left:13px;"
            f" border-radius:0; font-size:13px; text-align:left;}}")
_NAV_OFF = (f"QPushButton{{background:transparent; color:{_DIM};"
            f" border-left:3px solid transparent; padding-left:13px;"
            f" border-radius:0; font-size:13px; text-align:left;}}"
            f"QPushButton:hover{{background:{_DARK2}; color:{_TEXT};}}")


# ============================================================================
# CUSTOM PAINTED WIDGETS
# ============================================================================

class ActivityBlock(QWidget):
    """
    Custom-painted activity indicator.
    Gradient background + left colored accent bar + large label + sub-text.
    """
    def __init__(self):
        super().__init__()
        self._label = "—"
        self._sub   = ""
        self._color = _DIM
        self.setMinimumHeight(96)

    def set(self, label: str, sub: str, color: str):
        self._label = label.upper()
        self._sub   = sub
        self._color = color
        self.update()

    def paintEvent(self, event):
        p  = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r  = self.rect()
        c  = QColor(self._color)

        # Gradient background
        g = QLinearGradient(0.0, 0.0, float(r.width()), 0.0)
        c1 = QColor(c); c1.setAlphaF(0.18)
        c2 = QColor(c); c2.setAlphaF(0.04)
        g.setColorAt(0.0, c1)
        g.setColorAt(1.0, c2)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(g))
        p.drawRoundedRect(r, 8.0, 8.0)

        # Left accent bar
        p.setBrush(QColor(self._color))
        p.drawRoundedRect(0, 12, 4, r.height() - 24, 2.0, 2.0)

        # Large activity label
        p.setPen(QColor(self._color))
        f = QFont("Segoe UI", 26, QFont.Bold)
        p.setFont(f)
        lbl_r = QRectF(16, 6, r.width() - 20, r.height() - 30)
        p.drawText(lbl_r, Qt.AlignLeft | Qt.AlignVCenter, self._label)

        # Sub-text
        if self._sub:
            p.setPen(QColor(_DIM))
            sf = QFont("Segoe UI", 10)
            p.setFont(sf)
            sub_r = QRectF(16, r.height() - 26, r.width() - 20, 22)
            p.drawText(sub_r, Qt.AlignLeft | Qt.AlignVCenter, self._sub)

        p.end()


class ConfidenceGauge(QWidget):
    """
    Custom-painted circular arc confidence gauge.
    Track arc (dark) + value arc (colored) + center percentage text.
    """
    def __init__(self, size: int = 88):
        super().__init__()
        self._pct   = 0.0
        self._color = _DIM
        self.setFixedSize(size, size)

    def set(self, pct: float, color: str):
        self._pct   = max(0.0, min(100.0, pct))
        self._color = color
        self.update()

    def paintEvent(self, event):
        p   = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        m   = 10
        sz  = min(self.width(), self.height()) - 2 * m
        arc = QRectF(m, m, sz, sz)

        # Track
        pen = QPen(QColor(_DARK2), 7, Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(arc, 225 * 16, -270 * 16)

        # Value arc
        if self._pct > 0:
            pen.setColor(QColor(self._color))
            p.setPen(pen)
            p.drawArc(arc, 225 * 16, int(-270 * 16 * self._pct / 100))

        # Center text
        p.setPen(QColor(_TEXT))
        fs = max(8, sz // 5)
        f  = QFont("Segoe UI", fs, QFont.Bold)
        p.setFont(f)
        p.drawText(arc.toRect(), Qt.AlignCenter, f"{int(self._pct)}%")
        p.end()


class StatusPill(QWidget):
    """
    Animated top-right status indicator pill.
    Blinks yellow while connecting, solid green when live.
    """
    def __init__(self):
        super().__init__()
        self._live = False
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 12, 0)
        lay.setSpacing(6)
        self._dot  = QLabel("●")
        self._text = QLabel("Connecting…")
        self._dot.setStyleSheet(f"font-size:10px; color:{_YELLOW};")
        self._text.setStyleSheet(f"font-size:11px; color:{_YELLOW};")
        lay.addWidget(self._dot)
        lay.addWidget(self._text)
        self.setFixedHeight(28)
        self._set_style("connecting")

        self._blink = QtCore.QTimer(self)
        self._blink.timeout.connect(self._toggle_dot)
        self._blink.start(600)
        self._dot_on = True

    def _toggle_dot(self):
        if not self._live:
            self._dot_on = not self._dot_on
            self._dot.setVisible(self._dot_on)

    def _set_style(self, state: str):
        if state == "live":
            s = (f"background:{_GREEN}1a; border-radius:13px;"
                 f" border:1px solid {_GREEN}55;")
        else:
            s = (f"background:{_YELLOW}1a; border-radius:13px;"
                 f" border:1px solid {_YELLOW}44;")
        self.setStyleSheet(s)

    def update_status(self, frames: int, fps: float):
        self._live = frames > 10
        if self._live:
            self._blink.stop()
            self._dot.setVisible(True)
            self._dot.setStyleSheet(f"font-size:10px; color:{_GREEN};")
            self._text.setText(f"System Live · {fps:.0f} fr/s")
            self._text.setStyleSheet(f"font-size:11px; color:{_GREEN};")
            self._set_style("live")
        else:
            if not self._blink.isActive():
                self._blink.start(600)
            self._dot.setStyleSheet(f"font-size:10px; color:{_YELLOW};")
            self._text.setText("Connecting…")
            self._text.setStyleSheet(f"font-size:11px; color:{_YELLOW};")
            self._set_style("connecting")


# ============================================================================
# INFERENCE HELPERS
# ============================================================================

def _load_models(models_dir: str, model_key: str):
    d = Path(models_dir)
    paths = {
        "pipeline": d / "csi_pipeline.joblib",
        "le":       d / "label_encoder.joblib",
        "model":    d / f"{model_key}.joblib",
    }
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}\n  Train first: python csi_ml_pipeline.py --save_model"
            )
    return (joblib.load(paths["pipeline"]),
            joblib.load(paths["le"]),
            joblib.load(paths["model"]))


# ============================================================================
# READER THREAD
# ============================================================================

class ReaderThread(threading.Thread):
    def __init__(self, port, baud, demo, stop_event,
                 waveform_len: int, infer_buf_size: int, rx_buf: int):
        super().__init__(daemon=True)
        self.port = port;  self.baud = baud;  self.demo = demo
        self.stop_event    = stop_event
        self.waveform_len  = waveform_len
        self.rx_buf        = rx_buf
        self._lock         = threading.Lock()
        self._frame_count  = 0
        self._n_active     = 0
        self._fps_times    = deque(maxlen=60)
        self._start_time   = time.monotonic()
        self._wave_buf     = np.zeros(waveform_len, dtype=np.float32)
        self._wave_ptr     = 0
        self._last_amp     = np.zeros(config.MAX_SUBCARRIERS, dtype=np.float32)
        self._variance     = 0.0
        self._mean_amp     = 0.0
        self._infer_deque  = deque(maxlen=infer_buf_size)
        self.infer_buf_max = infer_buf_size
        self.connected     = False

    def snapshot(self) -> dict | None:
        with self._lock:
            if self._n_active == 0:
                return None
            ft  = list(self._fps_times)
            fps = (len(ft) - 1) / (ft[-1] - ft[0]) if len(ft) >= 2 else 0.0
            return {
                "wave":        np.roll(self._wave_buf, -self._wave_ptr),
                "last_amp":    self._last_amp[:self._n_active].copy(),
                "n":           self._n_active,
                "frame_count": self._frame_count,
                "fps":         fps,
                "uptime_s":    time.monotonic() - self._start_time,
                "variance":    self._variance,
                "mean_amp":    self._mean_amp,
                "infer_snap":  list(self._infer_deque),
                "infer_fill":  len(self._infer_deque) / max(1, self.infer_buf_max),
                "connected":   self.connected,
            }

    def _push(self, cf_frame: np.ndarray):
        n   = min(cf_frame.size, config.MAX_SUBCARRIERS)
        cf  = cf_frame[:n].astype(np.complex64)
        amp = np.abs(cf)
        ma  = float(amp.mean())
        with self._lock:
            p = self._wave_ptr
            self._wave_buf[p] = ma
            self._wave_ptr    = (p + 1) % self.waveform_len
            mx = amp.max()
            self._last_amp[:n] = amp / mx if mx > 0 else amp
            self._n_active     = max(self._n_active, n)
            self._frame_count += 1
            self._fps_times.append(time.monotonic())
            self._variance  = self._variance * 0.97 + (ma - self._mean_amp) ** 2 * 0.03
            self._mean_amp  = self._mean_amp * 0.97 + ma * 0.03
            self._infer_deque.append(cf_frame)

    def run(self):
        if self.demo:
            self.connected = True
            self._run_demo()
            return
        ser = None
        try:
            import serial
            ser = serial.Serial(self.port, self.baud, timeout=0.5)
            if os.name == "nt" and hasattr(ser, "set_buffer_size"):
                ser.set_buffer_size(rx_size=self.rx_buf)
            ser.reset_input_buffer()
            self.connected = True
            print(f"[OK]  Connected: {self.port} @ {self.baud}")
        except Exception as e:
            print(f"[ERROR] {e}  → demo mode")
            self.connected = True
            self._run_demo()
            return
        try:
            while not self.stop_event.is_set():
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("CSI_DATA"):
                    continue
                frame = parse_csi_line(line) if _PARSER_OK else None
                if frame is not None:
                    self._push(frame)
        finally:
            if ser and ser.is_open:
                ser.close()
            self.connected = False

    def _run_demo(self):
        rng = np.random.default_rng(0)
        t   = 0.0
        print("[INFO] Demo mode — synthetic CSI data")
        while not self.stop_event.is_set():
            time.sleep(0.01)
            t  += 0.05
            sc  = config.MAX_SUBCARRIERS
            idx = np.arange(sc, dtype=np.float32)
            p1  = np.exp(-0.5 * ((idx - (sc * .3 + 10 * math.sin(t * .7))) / 6) ** 2)
            p2  = np.exp(-0.5 * ((idx - (sc * .7 +  8 * math.cos(t * .5))) / 5) ** 2)
            amp = p1 * .7 + p2 * .5 + rng.uniform(0, .04, sc).astype(np.float32)
            ph  = t * .8 + idx / sc * 2 * np.pi
            self._push((amp * (np.cos(ph) + 1j * np.sin(ph))).astype(np.complex64))


# ============================================================================
# INFERENCE PROCESS  (runs preprocessing + ML in a separate OS process)
# ============================================================================

def _inference_worker_fn(in_q, out_q, pipeline, model, le, classes,
                         window_size, cutoff):
    """Target for the inference child process — fully GIL-free."""
    # Re-import in child process (Windows 'spawn' requirement)
    import numpy as _np
    import time as _time
    try:
        from csi_ml_pipeline import extract_features_from_window as _extract
    except ImportError:
        _extract = None

    while True:
        try:
            msg = in_q.get(timeout=1.0)
        except Exception:
            continue
        if msg is None:            # poison pill
            break

        cm, frame_count = msg
        raw_cand = conf_cand = probs_cand = None
        latency = 0.0

        if pipeline is not None and _extract is not None:
            if cm.shape[0] >= window_size + 1:
                try:
                    if cm.shape[1] != pipeline._fitted_n_subcarriers:
                        pass  # shape mismatch
                    else:
                        t0 = _time.monotonic()
                        processed = pipeline.transform(cm, use_pca=True, cutoff=cutoff)
                        if processed.shape[0] >= window_size:
                            features = _extract(processed[-window_size:]).reshape(1, -1)
                            if _np.all(_np.isfinite(features)):
                                ok = True
                                if hasattr(model, "n_features_in_"):
                                    ok = features.shape[1] == model.n_features_in_
                                if ok:
                                    if hasattr(model, "predict_proba"):
                                        probs = model.predict_proba(features)[0]
                                        idx = int(_np.argmax(probs))
                                        conf_cand = float(probs[idx]) * 100.0
                                    else:
                                        idx = int(model.predict(features)[0])
                                        probs = _np.zeros(len(le.classes_), dtype=_np.float32)
                                        probs[idx] = 1.0
                                        conf_cand = 100.0
                                    raw_cand = str(le.inverse_transform([idx])[0])
                                    probs_cand = probs
                        latency = (_time.monotonic() - t0) * 1000.0
                except Exception:
                    pass

        try:
            out_q.put_nowait((raw_cand, conf_cand, probs_cand, latency, frame_count))
        except Exception:
            pass  # output queue full — discard


class InferenceProcess:
    """Manages a child process for CSI inference (bypasses GIL)."""

    def __init__(self, pipeline, model, le, classes, window_size, cutoff):
        self._pipeline    = pipeline
        self._model       = model
        self._le          = le
        self._classes     = list(classes)
        self._window_size = window_size
        self._cutoff      = cutoff
        self._in_q  = None
        self._out_q = None
        self._proc  = None

    def start(self):
        self._in_q  = mp.Queue(maxsize=2)
        self._out_q = mp.Queue(maxsize=8)
        self._proc  = mp.Process(
            target=_inference_worker_fn,
            args=(self._in_q, self._out_q,
                  self._pipeline, self._model, self._le, self._classes,
                  self._window_size, self._cutoff),
            daemon=True,
        )
        self._proc.start()

    def submit(self, infer_snap, frame_count):
        """Pre-stack frames and send to worker (non-blocking)."""
        try:
            cm = np.vstack(infer_snap).astype(np.complex64)
        except (ValueError, TypeError):
            return
        # Discard stale pending request
        try:
            self._in_q.get_nowait()
        except Exception:
            pass
        try:
            self._in_q.put_nowait((cm, frame_count))
        except Exception:
            pass

    def get_result(self):
        """Drain queue and return only the latest result (non-blocking)."""
        result = None
        while True:
            try:
                result = self._out_q.get_nowait()
            except Exception:
                break
        return result

    def stop(self):
        if self._proc is None:
            return
        try:
            self._in_q.put(None, timeout=2.0)   # poison pill
        except Exception:
            pass
        self._proc.join(timeout=3.0)
        if self._proc.is_alive():
            self._proc.terminate()
        for q in (self._in_q, self._out_q):
            try:
                q.close()
            except Exception:
                pass

    def restart(self, pipeline, model, le, classes):
        """Stop old worker and start a new one with updated model."""
        self.stop()
        self._pipeline = pipeline
        self._model    = model
        self._le       = le
        self._classes  = list(classes)
        self.start()

    def join(self, timeout=2.0):
        self.stop()


# ============================================================================
# SHARED HELPERS
# ============================================================================

def _card() -> QWidget:
    w = QWidget()
    w.setStyleSheet(
        f"QWidget {{ background:{_PANEL}; border-radius:8px;"
        f" border:1px solid {_BORDER}; }}"
    )
    return w

def _hdr(text: str) -> QLabel:
    l = QLabel(text)
    l.setStyleSheet(
        f"font-size:8px; font-weight:bold; color:{_DIM};"
        f" letter-spacing:0.12em; border:none; background:transparent;"
    )
    return l

def _make_pw(title: str, left_lbl: str, bot_lbl: str) -> pg.PlotWidget:
    pw = pg.PlotWidget(background=_PANEL)
    pw.setMenuEnabled(False)
    pw.setMouseEnabled(x=False, y=False)
    pw.setTitle(title, color=_DIM, size="9pt")
    pw.setLabel("left",   left_lbl, color=_DIM, size="9pt")
    pw.setLabel("bottom", bot_lbl,  color=_DIM, size="9pt")
    pw.getAxis("left").setTextPen(pg.mkPen(_DIM))
    pw.getAxis("bottom").setTextPen(pg.mkPen(_DIM))
    return pw

def _stat_chip(label: str, value: str = "—") -> tuple[QWidget, QLabel]:
    w = QWidget()
    w.setStyleSheet(
        f"QWidget {{ background:{_DARK2}; border-radius:6px;"
        f" border:1px solid {_BORDER}; }}"
    )
    v = QVBoxLayout(w)
    v.setContentsMargins(10, 6, 10, 6)
    v.setSpacing(2)
    lbl = QLabel(label)
    lbl.setStyleSheet(
        f"font-size:8px; color:{_DIM}; letter-spacing:0.08em;"
        f" border:none; background:transparent;"
    )
    val = QLabel(value)
    val.setStyleSheet(
        f"font-size:14px; font-weight:bold; color:{_TEXT};"
        f" border:none; background:transparent;"
    )
    v.addWidget(lbl)
    v.addWidget(val)
    return w, val


# ============================================================================
# SIDEBAR
# ============================================================================

_NAV_ITEMS = [
    ("▣  Monitor",      0),
    ("∿  Signal View",  1),
    ("≡  Activity Log", 2),
    ("⊙  System Info",  3),
    ("⚙  Settings",     4),
]

class Sidebar(QWidget):
    page_changed = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        self.setFixedWidth(226)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Logo
        logo = QWidget()
        logo.setFixedHeight(72)
        logo.setStyleSheet(f"background:{_BG};")
        ll = QVBoxLayout(logo)
        ll.setContentsMargins(16, 16, 16, 12)
        ll.setSpacing(3)
        t1 = QLabel("WiFi CSI Analyzer")
        t1.setStyleSheet(
            f"font-size:14px; font-weight:bold; color:{_TEXT};"
            f" background:transparent; border:none;"
        )
        t2 = QLabel("LIVE MONITORING")
        t2.setStyleSheet(
            f"font-size:7px; color:{_DIM}; letter-spacing:0.14em;"
            f" background:transparent; border:none;"
        )
        ll.addWidget(t1)
        ll.addWidget(t2)
        root.addWidget(logo)
        root.addWidget(self._div())

        # Nav header
        nh = QLabel("  NAVIGATION")
        nh.setFixedHeight(30)
        nh.setStyleSheet(
            f"font-size:8px; color:{_DIM}; font-weight:bold;"
            f" letter-spacing:0.12em; padding:10px 0 0 16px;"
            f" background:transparent; border:none;"
        )
        root.addWidget(nh)

        self._btns: list[QPushButton] = []
        for label, idx in _NAV_ITEMS:
            btn = QPushButton(label)
            btn.setFixedHeight(40)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setStyleSheet(_NAV_OFF)
            btn.clicked.connect(lambda _, i=idx: self._click(i))
            root.addWidget(btn)
            self._btns.append(btn)

        root.addStretch()
        root.addWidget(self._div())

        # System Health
        sh = QWidget()
        sh.setStyleSheet(f"background:{_BG};")
        shv = QVBoxLayout(sh)
        shv.setContentsMargins(16, 10, 16, 14)
        shv.setSpacing(8)
        shv.addWidget(_hdr("SYSTEM HEALTH"))

        self._bar_refs: list[tuple[QProgressBar, QLabel]] = []
        for lbl_text in ("Buffer", "Load"):
            row = QHBoxLayout()
            row.setSpacing(8)
            l = QLabel(lbl_text)
            l.setFixedWidth(38)
            l.setStyleSheet(f"font-size:10px; color:{_DIM}; background:transparent; border:none;")
            bar = QProgressBar()
            bar.setRange(0, 100); bar.setValue(0)
            bar.setFixedHeight(4); bar.setTextVisible(False)
            bar.setStyleSheet(
                f"QProgressBar{{background:{_DARK2};border-radius:2px;border:none;}}"
                f"QProgressBar::chunk{{background:{_BLUE};border-radius:2px;}}"
            )
            pct = QLabel("0%")
            pct.setFixedWidth(28)
            pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pct.setStyleSheet(f"font-size:10px; color:{_DIM}; background:transparent; border:none;")
            row.addWidget(l); row.addWidget(bar, 1); row.addWidget(pct)
            shv.addLayout(row)
            self._bar_refs.append((bar, pct))

        root.addWidget(sh)
        self.set_active(0)

    def _div(self) -> QWidget:
        w = QWidget(); w.setFixedHeight(1)
        w.setStyleSheet(f"background:{_BORDER}; border:none;")
        return w

    def _click(self, idx: int):
        self.set_active(idx)
        self.page_changed.emit(idx)

    def set_active(self, idx: int):
        for i, btn in enumerate(self._btns):
            btn.setStyleSheet(_NAV_ON if i == idx else _NAV_OFF)

    def update_health(self, buf_pct: int, load_pct: int):
        for (bar, pct_lbl), val in zip(self._bar_refs, (buf_pct, load_pct)):
            bar.setValue(val)
            pct_lbl.setText(f"{val}%")


# ============================================================================
# PAGE 0 — MONITOR
# ============================================================================

class MonitorPage(QWidget):
    def __init__(self, classes: list, waveform_len: int,
                 model_key: str, port: str):
        super().__init__()
        self.classes      = classes
        self.waveform_len = waveform_len
        self._wave_scale  = 1.0
        self._build(model_key, port)

    def _build(self, model_key: str, port: str):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Alert banner
        self._banner = QWidget()
        self._banner.setFixedHeight(44)
        self._banner.setStyleSheet(
            f"background:#2d1418; border-left:4px solid {_RED}; border:none;"
        )
        bl = QHBoxLayout(self._banner)
        bl.setContentsMargins(16, 0, 16, 0)
        bl_lbl = QLabel("⚠   Hardware not detected. Waiting for ESP32…")
        bl_lbl.setStyleSheet(f"color:#ff9999; font-size:12px; background:transparent; border:none;")
        bl.addWidget(bl_lbl)
        self._banner.setVisible(False)
        root.addWidget(self._banner)

        # Body
        body = QHBoxLayout()
        body.setContentsMargins(12, 8, 12, 4)
        body.setSpacing(10)
        root.addLayout(body, stretch=1)

        # ── Left: plots ───────────────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(8)

        self._pw_wave = _make_pw("LIVE SIGNAL", "Amplitude", "Frame")
        self._pw_wave.showGrid(x=False, y=True, alpha=0.10)
        wl = self.waveform_len
        self._wave_x     = np.arange(wl, dtype=np.float32)
        self._curve_wave = self._pw_wave.plot(
            self._wave_x, np.zeros(wl),
            pen=pg.mkPen(color=_BLUE, width=2),
        )
        self._pw_wave.setXRange(0, wl, padding=0)
        left.addWidget(self._pw_wave, stretch=1)

        # Subcarrier bar (thin strip below waveform)
        self._pw_sc = _make_pw("Subcarrier Power", "", "Subcarrier")
        self._pw_sc.setFixedHeight(82)
        self._pw_sc.setLabel("left", "", color=_DIM, size="8pt")
        self._pw_sc.setYRange(0, 1.05, padding=0)
        self._sc_bars = pg.BarGraphItem(
            x=np.arange(1), height=np.zeros(1), width=0.8,
            brush=pg.mkBrush(_BLUE + "99"),
        )
        self._pw_sc.addItem(self._sc_bars)
        left.addWidget(self._pw_sc)
        body.addLayout(left, stretch=3)

        # ── Right: inference + stats ───────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(8)
        right.setContentsMargins(0, 0, 0, 0)

        # Activity Inference card
        inf_card = _card()
        iv = QVBoxLayout(inf_card)
        iv.setContentsMargins(14, 12, 14, 12)
        iv.setSpacing(8)
        iv.addWidget(_hdr("ACTIVITY INFERENCE"))

        self._activity_block = ActivityBlock()
        iv.addWidget(self._activity_block)

        # Gauge row
        g_row = QHBoxLayout()
        g_row.setSpacing(12)
        self._gauge = ConfidenceGauge(88)
        g_row.addWidget(self._gauge)
        info_v = QVBoxLayout()
        info_v.setSpacing(4)
        info_v.setAlignment(Qt.AlignVCenter)
        self._conf_lbl = QLabel("Confidence: —")
        self._conf_lbl.setStyleSheet(
            f"font-size:12px; color:{_TEXT}; background:transparent; border:none;"
        )
        self._raw_lbl = QLabel("Raw: —")
        self._raw_lbl.setStyleSheet(
            f"font-size:10px; color:{_DIM}; background:transparent; border:none;"
        )
        info_v.addWidget(self._conf_lbl)
        info_v.addWidget(self._raw_lbl)
        info_v.addStretch()
        g_row.addLayout(info_v, 1)
        iv.addLayout(g_row)
        right.addWidget(inf_card)

        # Link Quality + FPS/LOSS
        lq_row = QHBoxLayout()
        lq_row.setSpacing(8)
        lq_card = _card()
        lq_card.setFixedHeight(86)
        lqv = QVBoxLayout(lq_card)
        lqv.setContentsMargins(12, 10, 12, 10)
        lqv.setSpacing(4)
        lqv.addWidget(_hdr("LINK QUALITY"))
        self._lq_lbl = QLabel("No Signal")
        self._lq_lbl.setStyleSheet(
            f"font-size:13px; font-weight:bold; color:{_DIM};"
            f" background:transparent; border:none;"
        )
        self._lq_sub = QLabel("—")
        self._lq_sub.setStyleSheet(
            f"font-size:10px; color:{_DIM}; background:transparent; border:none;"
        )
        lqv.addWidget(self._lq_lbl)
        lqv.addWidget(self._lq_sub)
        lq_row.addWidget(lq_card, 2)

        chips_v = QVBoxLayout()
        chips_v.setSpacing(6)
        fps_c,  self._fps_val  = _stat_chip("⚡ FPS")
        loss_c, self._loss_val = _stat_chip("⬇ LOSS")
        fps_c.setFixedHeight(38); loss_c.setFixedHeight(38)
        chips_v.addWidget(fps_c); chips_v.addWidget(loss_c)
        lq_row.addLayout(chips_v, 1)
        right.addLayout(lq_row)

        # Recent Activity
        ra_card = _card()
        rav = QVBoxLayout(ra_card)
        rav.setContentsMargins(12, 10, 12, 10)
        rav.setSpacing(6)
        rav.addWidget(_hdr("RECENT ACTIVITY"))
        self._no_recent = QLabel("No activity yet")
        self._no_recent.setAlignment(Qt.AlignCenter)
        self._no_recent.setStyleSheet(
            f"color:{_DIM}; font-size:11px; background:transparent; border:none;"
        )
        self._recent = QTableWidget(0, 3)
        self._recent.setHorizontalHeaderLabels(["TIME", "ACTIVITY", "CONF"])
        hh = self._recent.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.Fixed)
        self._recent.setColumnWidth(0, 62); self._recent.setColumnWidth(2, 42)
        self._recent.verticalHeader().setVisible(False)
        self._recent.setEditTriggers(QTableWidget.NoEditTriggers)
        self._recent.setSelectionMode(QTableWidget.NoSelection)
        self._recent.setVisible(False)
        rav.addWidget(self._no_recent)
        rav.addWidget(self._recent, 1)
        right.addWidget(ra_card, 1)

        body.addLayout(right, stretch=1)

        # Bottom stat row
        bot = QHBoxLayout()
        bot.setContentsMargins(12, 0, 12, 8)
        bot.setSpacing(8)
        fc, self._sv_frames = _stat_chip("FRAMES")
        uc, self._sv_uptime = _stat_chip("UPTIME")
        mc, self._sv_model  = _stat_chip("MODEL",     _MODEL_NAMES.get(model_key, model_key))
        ic, self._sv_iface  = _stat_chip("INTERFACE", port)
        for w in (fc, uc, mc, ic):
            w.setFixedHeight(56)
            bot.addWidget(w, 1)
        root.addLayout(bot)

    def set_model_name(self, model_key: str):
        self._sv_model.setText(_MODEL_NAMES.get(model_key, model_key))

    def update(self, state: dict):
        fc = state.get("frame_count", 0)
        self._banner.setVisible(fc == 0)

        wave = state.get("wave")
        if wave is not None:
            mx = float(wave.max())
            self._wave_scale = max(mx * 1.05, self._wave_scale * 0.998)
            self._pw_wave.setYRange(0, max(self._wave_scale, 1e-6), padding=0.02)
            self._curve_wave.setData(self._wave_x, wave)

        n = state.get("n", 1); amp = state.get("last_amp")
        if amp is not None:
            self._sc_bars.setOpts(x=np.arange(n, dtype=np.float32), height=amp, width=0.8)
            self._pw_sc.setXRange(-0.5, n - 0.5, padding=0)

        label = state.get("label", "—")
        raw   = state.get("raw_label", "—")
        conf  = state.get("confidence", 0.0)
        color = _cc(label)
        self._activity_block.set(_disp(label), f"Raw: {_disp(raw)}", color)
        self._gauge.set(conf, color)
        self._conf_lbl.setText(f"Confidence: {conf:.1f}%")
        self._raw_lbl.setText(f"Raw: {_disp(raw)}")

        fps = state.get("fps", 0.0)
        if fps <= 0:
            self._lq_lbl.setText("No Signal")
            self._lq_lbl.setStyleSheet(
                f"font-size:13px; font-weight:bold; color:{_DIM};"
                f" background:transparent; border:none;"
            )
        elif fps >= 80:
            self._lq_lbl.setText("Excellent")
            self._lq_lbl.setStyleSheet(
                f"font-size:13px; font-weight:bold; color:{_GREEN};"
                f" background:transparent; border:none;"
            )
        elif fps >= 50:
            self._lq_lbl.setText("Good")
            self._lq_lbl.setStyleSheet(
                f"font-size:13px; font-weight:bold; color:{_YELLOW};"
                f" background:transparent; border:none;"
            )
        else:
            self._lq_lbl.setText("Poor")
            self._lq_lbl.setStyleSheet(
                f"font-size:13px; font-weight:bold; color:{_RED};"
                f" background:transparent; border:none;"
            )
        self._lq_sub.setText(f"{fps:.0f} fps")

        expected = config.SAMPLING_RATE
        loss_pct = max(0.0, (1.0 - fps / expected) * 100) if fps > 0 else 0.0
        self._fps_val.setText(f"{fps:.1f}")
        self._loss_val.setText(f"{loss_pct:.0f}%")

        uptime = state.get("uptime_s", 0.0)
        h = int(uptime // 3600); m = int((uptime % 3600) // 60); s = int(uptime % 60)
        self._sv_frames.setText(f"{fc:,}")
        self._sv_uptime.setText(f"{h:02d}:{m:02d}:{s:02d}")

        log = state.get("log_entries", [])
        show = len(log) > 0
        self._no_recent.setVisible(not show)
        self._recent.setVisible(show)
        if show:
            n_rows = min(6, len(log))
            self._recent.setRowCount(n_rows)
            for r, (ts, lbl, cf, _fr) in enumerate(log[:n_rows]):
                c = _cc(lbl)
                items = [QTableWidgetItem(ts), QTableWidgetItem(_disp(lbl).upper()),
                         QTableWidgetItem(f"{cf:.0f}%")]
                items[1].setForeground(pg.mkColor(c))
                for col, item in enumerate(items):
                    item.setTextAlignment(Qt.AlignCenter)
                    self._recent.setItem(r, col, item)
                self._recent.setRowHeight(r, 22)


# ============================================================================
# PAGE 1 — SIGNAL VIEW
# ============================================================================

class SignalViewPage(QWidget):
    def __init__(self, waveform_len: int):
        super().__init__()
        self.waveform_len = waveform_len
        self._wave_scale  = 1.0
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(8)

        # Top stat chips
        top = QHBoxLayout(); top.setSpacing(8)
        tw, self._sv_fps     = _stat_chip("⚡ THROUGHPUT")
        lw, self._sv_latency = _stat_chip("⏱ LATENCY")
        pw, self._sv_loss    = _stat_chip("⬇ PACKET LOSS")
        rw, self._sv_sig     = _stat_chip("📶 SIG POWER")
        vw, self._sv_var     = _stat_chip("〜 VARIANCE")
        for w in (tw, lw, pw, rw, vw):
            w.setFixedHeight(62); top.addWidget(w, 1)
        root.addLayout(top)

        # Subcarrier power distribution
        self._pw_sc = _make_pw(
            "SUBCARRIER POWER DISTRIBUTION", "Power (norm.)", "Subcarrier Index"
        )
        self._pw_sc.setYRange(0, 1.05, padding=0)
        self._pw_sc.showGrid(x=False, y=True, alpha=0.10)
        self._sc_bars = pg.BarGraphItem(
            x=np.arange(1), height=np.zeros(1), width=0.8,
            brush=pg.mkBrush(_BLUE + "99"),
        )
        self._pw_sc.addItem(self._sc_bars)
        root.addWidget(self._pw_sc, stretch=1)

        # Live signal waveform
        self._pw_wave = _make_pw(
            "LIVE SIGNAL WAVEFORM (AMPLITUDE)", "Amplitude", "Frame"
        )
        self._pw_wave.showGrid(x=False, y=True, alpha=0.10)
        wl = self.waveform_len
        self._wave_x     = np.arange(wl, dtype=np.float32)
        self._curve_wave = self._pw_wave.plot(
            self._wave_x, np.zeros(wl), pen=pg.mkPen(color=_BLUE, width=2)
        )
        self._pw_wave.setXRange(0, wl, padding=0)
        root.addWidget(self._pw_wave, stretch=2)

    def update(self, state: dict):
        fps      = state.get("fps", 0.0)
        lat      = state.get("latency_ms", 0.0)
        var      = state.get("variance", 0.0)
        mean_a   = state.get("mean_amp", 0.0)
        expected = config.SAMPLING_RATE
        loss_pct = max(0.0, (1.0 - fps / expected) * 100) if fps > 0 else 0.0
        sig_db   = 20 * math.log10(max(mean_a, 1e-9))

        self._sv_fps.setText(f"{fps:.1f} fps")
        self._sv_latency.setText(f"{lat:.1f} ms" if lat > 0 else "—")
        self._sv_loss.setText(f"{loss_pct:.0f}%")
        self._sv_sig.setText(f"{sig_db:.1f} dB")
        self._sv_var.setText(f"{var:.4f}")

        wave = state.get("wave")
        if wave is not None:
            mx = float(wave.max())
            self._wave_scale = max(mx * 1.05, self._wave_scale * 0.998)
            self._pw_wave.setYRange(0, max(self._wave_scale, 1e-6), padding=0.02)
            self._curve_wave.setData(self._wave_x, wave)

        n = state.get("n", 1); amp = state.get("last_amp")
        if amp is not None:
            self._sc_bars.setOpts(x=np.arange(n, dtype=np.float32), height=amp, width=0.8)
            self._pw_sc.setXRange(-0.5, n - 0.5, padding=0)


# ============================================================================
# PAGE 2 — ACTIVITY LOG
# ============================================================================

class ActivityLogPage(QWidget):
    def __init__(self, classes: list, on_clear):
        super().__init__()
        self.classes          = classes
        self._x_dist          = np.arange(len(classes), dtype=np.float32)
        self._session_start   = time.monotonic()
        self._last_table_upd  = 0.0
        self._build(on_clear)

    def _build(self, on_clear):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(8)

        # ── Top stat chips ────────────────────────────────────────────────────
        top = QHBoxLayout(); top.setSpacing(8)
        tc, self._v_total = _stat_chip("TOTAL PREDICTIONS")
        ac, self._v_avgc  = _stat_chip("AVG. CONFIDENCE")
        dc, self._v_dur   = _stat_chip("SESSION DURATION")
        oc, self._v_dom   = _stat_chip("DOMINANT ACTIVITY")
        for w in (tc, ac, dc, oc):
            w.setFixedHeight(62); top.addWidget(w, 1)
        root.addLayout(top)

        # ── Body ─────────────────────────────────────────────────────────────
        body = QHBoxLayout(); body.setSpacing(10)
        root.addLayout(body, stretch=1)

        # Left: distribution chart + clear button
        left = QVBoxLayout(); left.setSpacing(8)

        dist_card = _card(); dist_card.setFixedWidth(264)
        dv = QVBoxLayout(dist_card)
        dv.setContentsMargins(12, 10, 12, 10); dv.setSpacing(4)
        dv.addWidget(_hdr("ACTIVITY DISTRIBUTION"))

        self._pw_dist = pg.PlotWidget(background=_PANEL)
        self._pw_dist.setMenuEnabled(False)
        self._pw_dist.setMouseEnabled(x=False, y=False)
        self._pw_dist.getAxis("bottom").setTextPen(pg.mkPen(_DIM))
        self._pw_dist.getAxis("left").setTextPen(pg.mkPen(_DIM))
        self._pw_dist.getAxis("bottom").setStyle(
            tickFont=QFont("Segoe UI", 8), tickLength=-4
        )
        self._pw_dist.hideAxis("left")
        self._pw_dist.setYRange(0, 1, padding=0.22)
        ticks = [[(i, c[:5]) for i, c in enumerate(self.classes)]]
        self._pw_dist.getAxis("bottom").setTicks(ticks)
        brushes = [pg.mkBrush(_cc(c) + "cc") for c in self.classes]
        self._dist_bars = pg.BarGraphItem(
            x=self._x_dist, height=np.zeros(len(self.classes)), width=0.6,
            brushes=brushes,
        )
        self._pw_dist.addItem(self._dist_bars)
        self._pct_labels: list[pg.TextItem] = []
        for cls in self.classes:
            ti = pg.TextItem("", anchor=(0.5, 1.0), color=_DIM)
            ti.setFont(QFont("Segoe UI", 8))
            self._pw_dist.addItem(ti)
            self._pct_labels.append(ti)
        self._no_dist = QLabel("No data yet")
        self._no_dist.setAlignment(Qt.AlignCenter)
        self._no_dist.setStyleSheet(
            f"color:{_DIM}; font-size:11px; background:transparent; border:none;"
        )
        dv.addWidget(self._pw_dist, 1)
        dv.addWidget(self._no_dist)
        self._pw_dist.setVisible(False)
        left.addWidget(dist_card, 1)

        clr = QPushButton("🗑   Clear Log")
        clr.setFixedHeight(34); clr.setFixedWidth(264)
        clr.setStyleSheet(
            f"QPushButton{{background:{_DARK2}; color:{_DIM};"
            f" border:1px solid {_BORDER}; border-radius:6px;"
            f" font-size:12px; padding:0;}}"
            f"QPushButton:hover{{color:{_TEXT}; border-color:{_TEXT};}}"
        )
        clr.clicked.connect(on_clear)
        left.addWidget(clr)
        body.addLayout(left)

        # Right: full log table
        log_card = _card()
        tv = QVBoxLayout(log_card)
        tv.setContentsMargins(12, 10, 12, 10); tv.setSpacing(6)
        tv.addWidget(_hdr("ACTIVITY LOG"))
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["TIME", "ACTIVITY", "CONFIDENCE", "FRAME"])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.Fixed)
        hh.setSectionResizeMode(3, QHeaderView.Fixed)
        self._table.setColumnWidth(0, 72); self._table.setColumnWidth(2, 84)
        self._table.setColumnWidth(3, 60)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        self._no_log = QLabel("No activity recorded yet")
        self._no_log.setAlignment(Qt.AlignCenter)
        self._no_log.setStyleSheet(
            f"color:{_DIM}; font-size:13px; background:transparent; border:none;"
        )
        tv.addWidget(self._table, 1); tv.addWidget(self._no_log)
        self._table.setVisible(False)
        body.addWidget(log_card, 1)

    def update(self, state: dict):
        total  = state.get("total_predictions", 0)
        avg_c  = state.get("avg_confidence", 0.0)
        counts = state.get("class_counts", {})
        log    = state.get("log_entries", [])

        # Stat chips
        elapsed = time.monotonic() - self._session_start
        h = int(elapsed // 3600); m = int((elapsed % 3600) // 60); s = int(elapsed % 60)
        self._v_dur.setText(f"{h:02d}:{m:02d}:{s:02d}")
        self._v_total.setText(str(total))

        c_avg = _GREEN if avg_c >= 70 else (_YELLOW if avg_c >= 50 else _RED)
        self._v_avgc.setText(f"{avg_c:.0f}%")
        self._v_avgc.setStyleSheet(
            f"font-size:14px; font-weight:bold; color:{c_avg};"
            f" border:none; background:transparent;"
        )
        if total > 0 and counts:
            dom = max(counts, key=counts.get)
            self._v_dom.setText(_disp(dom).upper())
            self._v_dom.setStyleSheet(
                f"font-size:14px; font-weight:bold; color:{_cc(dom)};"
                f" border:none; background:transparent;"
            )

        # Distribution chart
        if total > 0:
            heights = np.array(
                [counts.get(c, 0) / total for c in self.classes], dtype=np.float32
            )
            self._dist_bars.setOpts(height=heights)
            mx = max(float(heights.max()), 0.05)
            self._pw_dist.setYRange(0, mx * 1.40, padding=0)
            for i, (hv, ti) in enumerate(zip(heights, self._pct_labels)):
                ti.setText(f"{hv*100:.0f}%")
                ti.setPos(i, hv)
            self._pw_dist.setVisible(True); self._no_dist.setVisible(False)
        else:
            self._pw_dist.setVisible(False); self._no_dist.setVisible(True)
            for ti in self._pct_labels:
                ti.setText("")

        # Log table — refresh at most once per second
        show = len(log) > 0
        self._table.setVisible(show); self._no_log.setVisible(not show)
        now = time.monotonic()
        if show and (now - self._last_table_upd) >= 1.0:
            self._last_table_upd = now
            self._table.setRowCount(len(log))
            for r, (ts, lbl, cf, fr) in enumerate(log):
                c = _cc(lbl)
                items = [QTableWidgetItem(ts), QTableWidgetItem(_disp(lbl).upper()),
                         QTableWidgetItem(f"{cf:.1f}%"), QTableWidgetItem(str(fr))]
                items[1].setForeground(pg.mkColor(c))
                for col, item in enumerate(items):
                    item.setTextAlignment(Qt.AlignCenter)
                    self._table.setItem(r, col, item)
                self._table.setRowHeight(r, 22)


# ============================================================================
# PAGE 3 — SYSTEM INFO
# ============================================================================

def _info_field(label: str, value: str) -> tuple[QVBoxLayout, QLabel]:
    fld = QVBoxLayout(); fld.setSpacing(2)
    lbl = QLabel(label)
    lbl.setStyleSheet(
        f"font-size:8px; color:{_DIM}; letter-spacing:0.08em;"
        f" background:transparent; border:none;"
    )
    val = QLabel(value)
    val.setStyleSheet(
        f"font-size:13px; font-weight:bold; color:{_TEXT};"
        f" background:transparent; border:none;"
    )
    fld.addWidget(lbl); fld.addWidget(val)
    return fld, val

def _info_card(title: str, fields: list[tuple[str, str]]) -> tuple[QWidget, dict]:
    card = _card()
    cv = QVBoxLayout(card)
    cv.setContentsMargins(20, 14, 20, 16); cv.setSpacing(10)
    t = QLabel(title)
    t.setStyleSheet(
        f"font-size:15px; font-weight:bold; color:{_TEXT};"
        f" background:transparent; border:none;"
    )
    cv.addWidget(t)
    sep = QWidget(); sep.setFixedHeight(1)
    sep.setStyleSheet(f"background:{_BORDER}; border:none;")
    cv.addWidget(sep)
    g = QHBoxLayout(); g.setSpacing(24)
    col1 = QVBoxLayout(); col1.setSpacing(10)
    col2 = QVBoxLayout(); col2.setSpacing(10)
    refs: dict[str, QLabel] = {}
    for i, (fl, fv) in enumerate(fields):
        fld, val = _info_field(fl, fv)
        (col1 if i % 2 == 0 else col2).addLayout(fld)
        refs[fl] = val
    col1.addStretch(); col2.addStretch()
    g.addLayout(col1); g.addLayout(col2)
    cv.addLayout(g)
    return card, refs


class SystemInfoPage(QWidget):
    def __init__(self, model_key: str, window_size: int, port: str, baud: int):
        super().__init__()
        self._model_key   = model_key
        self._window_size = window_size
        self._port        = port
        self._baud        = baud
        self._conn_status_lbl   = None
        self._conn_latency_lbl  = None
        self._infer_latency_lbl = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8); root.setSpacing(10)

        # ── Status banner ─────────────────────────────────────────────────────
        hc = _card(); hc.setFixedHeight(64)
        self._hc = hc  # kept for dynamic border-color update
        hv = QHBoxLayout(hc); hv.setContentsMargins(20, 0, 24, 0); hv.setSpacing(0)

        dot_col = QVBoxLayout(); dot_col.setSpacing(2)
        dot_col.addWidget(_hdr("SYSTEM STATUS"))
        sr = QHBoxLayout(); sr.setSpacing(8)
        self._sys_dot = QLabel("●")
        self._sys_dot.setStyleSheet(
            f"font-size:14px; color:{_RED}; background:transparent; border:none;"
        )
        self._sys_lbl = QLabel("OFFLINE")
        self._sys_lbl.setStyleSheet(
            f"font-size:16px; font-weight:bold; color:{_RED};"
            f" letter-spacing:0.06em; background:transparent; border:none;"
        )
        sr.addWidget(self._sys_dot); sr.addWidget(self._sys_lbl)
        dot_col.addLayout(sr)
        hv.addLayout(dot_col); hv.addStretch()

        for label, attr in [("LIVE UPTIME",  "_si_uptime"),
                             ("THROUGHPUT",   "_si_fps"),
                             ("BUFFER FILL",  "_si_buf"),
                             ("INF. LATENCY", "_si_lat"),
                             ("FRAMES",       "_si_frames")]:
            col = QVBoxLayout(); col.setSpacing(2); col.setContentsMargins(16, 0, 0, 0)
            lbl_w = QLabel(label)
            lbl_w.setStyleSheet(
                f"font-size:8px; color:{_DIM}; letter-spacing:0.08em;"
                f" background:transparent; border:none;"
            )
            val_w = QLabel("—")
            val_w.setStyleSheet(
                f"font-size:13px; font-weight:bold; color:{_TEXT};"
                f" background:transparent; border:none;"
            )
            col.addWidget(lbl_w); col.addWidget(val_w)
            hv.addLayout(col)
            setattr(self, attr, val_w)
        root.addWidget(hc)

        # ── Row 1: WiFi Radar + Inference ─────────────────────────────────────
        r1 = QHBoxLayout(); r1.setSpacing(10)
        radar_card, radar_refs = _info_card("WiFi Radar", [
            ("CHIPSET",    "ESP32-C6"),
            ("BAND",       "2.4 GHz (HT40)"),
            ("ANTENNAS",   "1×1 SISO"),
            ("CSI FORMAT", "Complex64"),
        ])
        infer_card, infer_refs = _info_card("Inference Engine", [
            ("MODEL",   _MODEL_NAMES.get(self._model_key, self._model_key)),
            ("WINDOW",  f"{self._window_size} frames"),
            ("LATENCY", "—"),
            ("CLASSES", "—"),
        ])
        # Add buffer-fill progress bar to the Inference card
        infer_cv = infer_card.layout()
        buf_section = QVBoxLayout(); buf_section.setSpacing(4)
        buf_hdr = QLabel("BUFFER FILL")
        buf_hdr.setStyleSheet(
            f"font-size:8px; color:{_DIM}; letter-spacing:0.08em;"
            f" background:transparent; border:none;"
        )
        buf_row = QHBoxLayout(); buf_row.setSpacing(8)
        self._buf_bar = QProgressBar()
        self._buf_bar.setRange(0, 100); self._buf_bar.setValue(0)
        self._buf_bar.setFixedHeight(5); self._buf_bar.setTextVisible(False)
        self._buf_bar.setStyleSheet(
            f"QProgressBar{{background:{_DARK2}; border-radius:2px; border:none;}}"
            f"QProgressBar::chunk{{background:{_BLUE}; border-radius:2px;}}"
        )
        self._buf_pct_lbl = QLabel("0%")
        self._buf_pct_lbl.setStyleSheet(
            f"font-size:10px; color:{_DIM}; background:transparent; border:none;"
        )
        buf_row.addWidget(self._buf_bar, 1); buf_row.addWidget(self._buf_pct_lbl)
        buf_section.addWidget(buf_hdr); buf_section.addLayout(buf_row)
        infer_cv.addLayout(buf_section)

        r1.addWidget(radar_card, 1); r1.addWidget(infer_card, 1)
        root.addLayout(r1, stretch=1)
        self._infer_latency_lbl = infer_refs.get("LATENCY")
        self._infer_classes_lbl = infer_refs.get("CLASSES")
        self._infer_model_lbl   = infer_refs.get("MODEL")

        # ── Row 2: Software + Connection ──────────────────────────────────────
        r2 = QHBoxLayout(); r2.setSpacing(10)
        sw_card, _ = _info_card("Software Stack", [
            ("RUNTIME",  "Python 3"),
            ("FRONTEND", "PyQt5 + pyqtgraph"),
            ("PROTOCOL", "UART / Serial"),
            ("VERSION",  "v1.0.0"),
        ])
        conn_card, conn_refs = _info_card("Connection", [
            ("PORT",    self._port),
            ("BAUD",    f"{self._baud:,}"),
            ("LATENCY", "—"),
            ("STATUS",  "LOST"),
        ])
        r2.addWidget(sw_card, 1); r2.addWidget(conn_card, 1)
        root.addLayout(r2, stretch=1)
        self._conn_status_lbl  = conn_refs.get("STATUS")
        self._conn_latency_lbl = conn_refs.get("LATENCY")

    def set_classes(self, classes: list):
        if self._infer_classes_lbl:
            self._infer_classes_lbl.setText(str(len(classes)))

    def set_model_name(self, model_key: str):
        if self._infer_model_lbl:
            self._infer_model_lbl.setText(_MODEL_NAMES.get(model_key, model_key))

    @staticmethod
    def _metric_style(color: str) -> str:
        return (f"font-size:13px; font-weight:bold; color:{color};"
                f" background:transparent; border:none;")

    def update(self, state: dict):
        connected = state.get("connected", False)
        dot_color = _GREEN if connected else _RED
        sys_text  = "LIVE" if connected else "OFFLINE"

        # Header card: tinted left border depending on status
        self._hc.setStyleSheet(
            f"QWidget{{background:{_PANEL}; border-radius:8px;"
            f" border-left:3px solid {dot_color}; border-top:1px solid {_BORDER};"
            f" border-right:1px solid {_BORDER}; border-bottom:1px solid {_BORDER};}}"
        )
        self._sys_dot.setStyleSheet(
            f"font-size:14px; color:{dot_color}; background:transparent; border:none;"
        )
        self._sys_lbl.setText(sys_text)
        self._sys_lbl.setStyleSheet(
            f"font-size:16px; font-weight:bold; color:{dot_color};"
            f" letter-spacing:0.06em; background:transparent; border:none;"
        )

        if self._conn_status_lbl:
            self._conn_status_lbl.setText("LIVE" if connected else "LOST")
            self._conn_status_lbl.setStyleSheet(
                self._metric_style(dot_color)
            )
        if self._conn_latency_lbl:
            self._conn_latency_lbl.setText(
                f"{state.get('latency_ms', 0.0):.1f} ms"
                if state.get("latency_ms", 0.0) > 0 else "—"
            )

        # Uptime + frames (always white)
        uptime = state.get("uptime_s", 0.0)
        h = int(uptime // 3600); m = int((uptime % 3600) // 60); s = int(uptime % 60)
        self._si_uptime.setText(f"{h:02d}:{m:02d}:{s:02d}")
        self._si_frames.setText(f"{state.get('frame_count', 0):,}")

        # FPS — color-coded
        fps = state.get("fps", 0.0)
        fps_color = _GREEN if fps >= 80 else (_YELLOW if fps >= 40 else _RED)
        self._si_fps.setText(f"{fps:.1f} fps" if fps > 0 else "—")
        self._si_fps.setStyleSheet(
            self._metric_style(fps_color if fps > 0 else _DIM)
        )

        # Buffer fill — color-coded label + progress bar in Inference card
        buf = int(state.get("infer_fill", 0) * 100)
        buf_color = _GREEN if buf >= 80 else (_YELLOW if buf >= 40 else _BLUE)
        self._si_buf.setText(f"{buf}%")
        self._si_buf.setStyleSheet(self._metric_style(buf_color))
        self._buf_bar.setValue(buf)
        self._buf_bar.setStyleSheet(
            f"QProgressBar{{background:{_DARK2}; border-radius:2px; border:none;}}"
            f"QProgressBar::chunk{{background:{buf_color}; border-radius:2px;}}"
        )
        self._buf_pct_lbl.setText(f"{buf}%")
        self._buf_pct_lbl.setStyleSheet(
            f"font-size:10px; color:{buf_color}; background:transparent; border:none;"
        )

        # Inference latency — color-coded
        lat = state.get("latency_ms", 0.0)
        lat_s = f"{lat:.1f} ms" if lat > 0 else "—"
        lat_color = _GREEN if lat <= 5 else (_YELLOW if lat <= 30 else _RED)
        self._si_lat.setText(lat_s)
        self._si_lat.setStyleSheet(
            self._metric_style(lat_color if lat > 0 else _DIM)
        )
        if self._infer_latency_lbl:
            self._infer_latency_lbl.setText(lat_s)


# ============================================================================
# PAGE 4 — SETTINGS
# ============================================================================

class SettingsPage(QWidget):
    def __init__(self, models_dir: str, current_model: str, on_deploy):
        super().__init__()
        self._models_dir    = models_dir
        self._current       = current_model
        self._selected      = current_model
        self._on_deploy     = on_deploy
        self._radios: dict[str, QRadioButton] = {}
        self._status_pills: dict[str, QLabel] = {}
        self._build()

    def _scan(self) -> list[tuple[str, bool]]:
        d = Path(self._models_dir)
        pipeline_ok = (d / "csi_pipeline.joblib").exists() and (d / "label_encoder.joblib").exists()
        return [(k, pipeline_ok and (d / f"{k}.joblib").exists()) for k in config.MODEL_KEYS]

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16); root.setSpacing(16)

        h1 = QLabel("Settings")
        h1.setStyleSheet(
            f"font-size:22px; font-weight:bold; color:{_TEXT};"
            f" background:transparent; border:none;"
        )
        root.addWidget(h1)

        # Model selection card
        mc = _card()
        mv = QVBoxLayout(mc); mv.setContentsMargins(20, 16, 20, 16); mv.setSpacing(10)
        mv.addWidget(_hdr("MODEL SELECTION"))

        dir_lbl = QLabel(f"Models directory: {self._models_dir}")
        dir_lbl.setStyleSheet(
            f"font-size:10px; color:{_DIM}; background:transparent; border:none;"
        )
        dir_lbl.setWordWrap(True)
        mv.addWidget(dir_lbl)

        sep = QWidget(); sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{_BORDER}; border:none;")
        mv.addWidget(sep)

        group = QButtonGroup(self)
        models = self._scan()
        for key, available in models:
            row = QWidget()
            row_bg = _DARK2 if available else f"{_DARK2}44"
            row_border = _BORDER if available else f"{_BORDER}44"
            row.setStyleSheet(
                f"QWidget{{background:{row_bg}; border-radius:6px;"
                f" border:1px solid {row_border};}}"
            )
            rl = QHBoxLayout(row); rl.setContentsMargins(14, 10, 14, 10); rl.setSpacing(12)

            rb = QRadioButton()
            rb.setChecked(key == self._current)
            rb.setEnabled(available)
            rb.setStyleSheet("QRadioButton{background:transparent; border:none;}")
            group.addButton(rb)
            self._radios[key] = rb
            rb.toggled.connect(lambda checked, k=key: self._on_radio(k, checked))
            rl.addWidget(rb)

            info_v = QVBoxLayout(); info_v.setSpacing(2)
            name_lbl = QLabel(_MODEL_NAMES.get(key, key))
            name_lbl.setStyleSheet(
                f"font-size:13px; font-weight:bold;"
                f" color:{_TEXT if available else _DIM};"
                f" background:transparent; border:none;"
            )
            desc_lbl = QLabel(_MODEL_DESC.get(key, ""))
            desc_lbl.setStyleSheet(
                f"font-size:10px; color:{_DIM}; background:transparent; border:none;"
            )
            info_v.addWidget(name_lbl); info_v.addWidget(desc_lbl)
            rl.addLayout(info_v, 1)

            file_lbl = QLabel(f"{key}.joblib")
            file_lbl.setStyleSheet(
                f"font-size:9px; color:{_DIM}; background:transparent; border:none;"
            )
            rl.addWidget(file_lbl)

            if key == self._current:
                pill_text, pill_color = "ACTIVE", _GREEN
            elif available:
                pill_text, pill_color = "READY", _BLUE
            else:
                pill_text, pill_color = "NOT TRAINED", _DIM

            pill = QLabel(pill_text)
            pill.setFixedWidth(80)
            pill.setAlignment(Qt.AlignCenter)
            pill.setStyleSheet(
                f"font-size:9px; font-weight:bold; color:{pill_color};"
                f" background:{pill_color}22; border:1px solid {pill_color}44;"
                f" border-radius:8px; padding:2px 6px;"
            )
            self._status_pills[key] = pill
            rl.addWidget(pill)
            mv.addWidget(row)

        mv.addSpacing(6)

        # Deploy row
        deploy_row = QHBoxLayout(); deploy_row.setSpacing(12)
        self._deploy_btn = QPushButton("⚡   Deploy Model")
        self._deploy_btn.setFixedHeight(38)
        self._deploy_btn.setStyleSheet(
            f"QPushButton{{background:{_BLUE}; color:#ffffff; border:none;"
            f" border-radius:6px; font-size:13px; font-weight:bold; padding:0 20px;}}"
            f"QPushButton:hover{{background:{_BLUE}dd;}}"
            f"QPushButton:disabled{{background:{_DARK2}; color:{_DIM};"
            f" border:1px solid {_BORDER};}}"
        )
        self._deploy_btn.setEnabled(False)
        self._deploy_btn.clicked.connect(self._deploy)

        self._status_lbl = QLabel(
            f"Active: {_MODEL_NAMES.get(self._current, self._current)}"
        )
        self._status_lbl.setStyleSheet(
            f"font-size:11px; color:{_DIM}; background:transparent; border:none;"
        )
        deploy_row.addWidget(self._deploy_btn)
        deploy_row.addWidget(self._status_lbl, 1)
        mv.addLayout(deploy_row)

        root.addWidget(mc)
        root.addStretch()

    def _on_radio(self, key: str, checked: bool):
        if checked:
            self._selected = key
            self._deploy_btn.setEnabled(key != self._current)

    def _deploy(self):
        key = self._selected
        self._deploy_btn.setEnabled(False)
        self._deploy_btn.setText("⏳   Loading…")
        QApplication.processEvents()
        ok, msg = self._on_deploy(key)
        self._deploy_btn.setText("⚡   Deploy Model")
        if ok:
            # Update pill states
            for k, pill in self._status_pills.items():
                if k == key:
                    pill.setText("ACTIVE")
                    pill.setStyleSheet(
                        f"font-size:9px; font-weight:bold; color:{_GREEN};"
                        f" background:{_GREEN}22; border:1px solid {_GREEN}44;"
                        f" border-radius:8px; padding:2px 6px;"
                    )
                elif pill.text() == "ACTIVE":
                    pill.setText("READY")
                    pill.setStyleSheet(
                        f"font-size:9px; font-weight:bold; color:{_BLUE};"
                        f" background:{_BLUE}22; border:1px solid {_BLUE}44;"
                        f" border-radius:8px; padding:2px 6px;"
                    )
            self._current = key
            self._status_lbl.setText(f"✓  {msg}")
            self._status_lbl.setStyleSheet(
                f"font-size:11px; color:{_GREEN}; background:transparent; border:none;"
            )
        else:
            self._status_lbl.setText(f"✗  {msg}")
            self._status_lbl.setStyleSheet(
                f"font-size:11px; color:{_RED}; background:transparent; border:none;"
            )
        self._deploy_btn.setEnabled(self._selected != self._current)

    def update(self, state: dict):
        pass


# ============================================================================
# DASHBOARD WINDOW
# ============================================================================

class DashboardWindow(QMainWindow):
    def __init__(self, *, reader: ReaderThread,
                 pipeline, le, model, classes: list, model_key: str,
                 window_size: int, step: int, ema_alpha: float,
                 conf_thresh: float, cutoff: float, refresh_ms: int,
                 max_log: int, port: str, baud: int, demo: bool,
                 models_dir: str, hyst_count: int = 2):
        super().__init__()
        self.reader      = reader
        self.pipeline    = pipeline; self.le = le; self.model = model
        self.classes     = classes; self.model_key = model_key
        self.models_dir  = models_dir
        self.window_size = window_size; self.step = step; self.cutoff = cutoff
        self.ema_alpha   = ema_alpha; self.conf_thresh = conf_thresh
        self.max_log     = max_log; self.demo = demo
        self._hyst_min   = hyst_count

        self._ema_probs       = np.zeros(len(classes), dtype=np.float32)
        self._frames_since    = 0; self._last_seen = 0
        self._latency_ms      = 0.0; self._demo_tick = 0
        self._pred_hist       = deque(maxlen=10)
        self._last_record_t   = 0.0
        self._last_record_lbl = ""
        self._hyst_pending    = ""
        self._hyst_count      = 0
        self._log_entries: list      = []
        self._class_counts: dict     = {c: 0 for c in classes}
        self._conf_sum               = 0.0; self._total_preds = 0
        self._state: dict            = {
            "label": "—", "raw_label": "—", "confidence": 0.0,
            "all_probs": np.zeros(len(classes)), "latency_ms": 0.0,
            "log_entries": [], "total_predictions": 0,
            "avg_confidence": 0.0, "class_counts": {},
        }

        # Background inference worker (bypasses GIL via multiprocessing)
        self._infer_worker = InferenceProcess(
            pipeline=pipeline, model=model, le=le, classes=classes,
            window_size=window_size, cutoff=cutoff,
        )
        self._infer_worker.start()

        self.setWindowTitle("WiFi CSI Analyzer — Live Dashboard")
        self.setStyleSheet(_QSS)
        pg.setConfigOptions(antialias=False)
        self._build_ui(port, baud, window_size)

    def _build_ui(self, port, baud, window_size):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        self._sidebar = Sidebar()
        self._sidebar.page_changed.connect(self._switch_page)
        root.addWidget(self._sidebar)

        # Right side: thin top bar + stack
        right = QVBoxLayout(); right.setContentsMargins(0, 0, 0, 0); right.setSpacing(0)

        # Top bar with status pill
        top_bar = QWidget(); top_bar.setFixedHeight(36)
        top_bar.setStyleSheet(f"background:{_BG}; border-bottom:1px solid {_BORDER};")
        tb = QHBoxLayout(top_bar); tb.setContentsMargins(12, 0, 12, 0)
        tb.addStretch()
        self._pill = StatusPill()
        tb.addWidget(self._pill)
        right.addWidget(top_bar)

        self._stack = QStackedWidget()
        wl = self.reader.waveform_len
        self._monitor_page  = MonitorPage(self.classes, wl, self.model_key, port)
        self._sysinfo_page  = SystemInfoPage(self.model_key, window_size, port, baud)
        self._sysinfo_page.set_classes(self.classes)
        self._settings_page = SettingsPage(
            self.models_dir, self.model_key, self.switch_model
        )
        self._pages = [
            self._monitor_page,
            SignalViewPage(wl),
            ActivityLogPage(self.classes, self._clear_log),
            self._sysinfo_page,
            self._settings_page,
        ]
        for page in self._pages:
            self._stack.addWidget(page)

        right.addWidget(self._stack, 1)
        root.addLayout(right, 1)

    def switch_model(self, model_key: str) -> tuple[bool, str]:
        try:
            pipeline, le, model = _load_models(self.models_dir, model_key)
            self.pipeline  = pipeline
            self.le        = le
            self.model     = model
            self.model_key = model_key
            self.classes   = list(le.classes_)
            self._ema_probs       = np.zeros(len(self.classes), dtype=np.float32)
            self._last_record_lbl = ""
            self._last_record_t   = 0.0
            self._hyst_pending    = ""
            self._hyst_count      = 0
            # Restart inference process with new model
            self._infer_worker.restart(pipeline, model, le, self.classes)
            # Reset displayed prediction so stale label doesn't persist
            self._state.update({
                "label": "—", "raw_label": "—",
                "confidence": 0.0,
                "all_probs": np.zeros(len(self.classes)),
            })
            self._sysinfo_page.set_classes(self.classes)
            self._monitor_page.set_model_name(model_key)
            self._sysinfo_page.set_model_name(model_key)
            return True, f"'{_MODEL_NAMES.get(model_key, model_key)}' is now active"
        except Exception as e:
            return False, str(e)

    def _switch_page(self, idx: int):
        self._stack.setCurrentIndex(idx)

    def _clear_log(self):
        self._log_entries.clear()
        self._class_counts = {c: 0 for c in self.classes}
        self._conf_sum = 0.0; self._total_preds = 0

    def start_timer(self, refresh_ms: int):
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(refresh_ms)

    def _refresh(self):
        snap = self.reader.snapshot()
        if snap is None:
            return

        new_frames = snap["frame_count"] - self._last_seen
        self._last_seen    = snap["frame_count"]
        self._frames_since += new_frames

        # ── Submit work to background thread (non-blocking) ──────────────
        if self._frames_since >= self.step:
            self._frames_since = 0
            if self.demo and self.pipeline is None:
                self._demo_predict(snap["frame_count"])
            else:
                self._infer_worker.submit(
                    snap["infer_snap"],
                    snap["frame_count"],
                )

        # ── Poll for completed inference result (non-blocking) ───────────
        result = self._infer_worker.get_result()
        if result is not None:
            raw_cand, conf_cand, probs_cand, latency, fc = result
            self._latency_ms = latency

            if raw_cand is not None and probs_cand is not None:
                # Adaptive EMA: scale update speed by model confidence
                max_prob = float(np.max(probs_cand))
                dynamic_alpha = self.ema_alpha * max_prob

                self._ema_probs = (dynamic_alpha * probs_cand +
                                   (1.0 - dynamic_alpha) * self._ema_probs)

                best_idx = int(np.argmax(self._ema_probs))
                smoothed_prob = float(self._ema_probs[best_idx]) * 100.0
                smoothed_cand = self.classes[best_idx]

                if smoothed_prob < self.conf_thresh:
                    if self._last_record_lbl:
                        smoothed_cand = self._last_record_lbl
                    else:
                        smoothed_cand = "—"

                # ── Hysteresis (State Transition Delay) ──────────────────────
                if smoothed_cand == self._hyst_pending:
                    self._hyst_count += 1
                else:
                    self._hyst_pending = smoothed_cand
                    self._hyst_count   = 1

                if self._hyst_count >= self._hyst_min:
                    now = time.monotonic()
                    if (smoothed_cand != self._last_record_lbl or
                            (now - self._last_record_t) >= 4.0):
                        self._last_record_lbl = smoothed_cand
                        self._last_record_t   = now
                        self._record(smoothed_cand, smoothed_prob, self._ema_probs, fc)

        state = {**snap, **self._state, "latency_ms": self._latency_ms}

        self._sidebar.update_health(int(snap["infer_fill"] * 100), 0)
        self._pill.update_status(snap["frame_count"], snap["fps"])
        for page in self._pages:
            page.update(state)

    def _record(self, label, conf, probs, frame):
        self._total_preds += 1
        self._conf_sum    += conf
        self._class_counts[label] = self._class_counts.get(label, 0) + 1
        self._log_entries.insert(0, (time.strftime("%H:%M:%S"), label, conf, frame))
        if len(self._log_entries) > self.max_log:
            self._log_entries.pop()
        self._state.update({
            "label":             label,     "raw_label":      label,
            "confidence":        conf,      "all_probs":      probs,
            "log_entries":       self._log_entries,
            "total_predictions": self._total_preds,
            "avg_confidence":    self._conf_sum / self._total_preds,
            "class_counts":      dict(self._class_counts),
        })

    def _demo_predict(self, frame_count):
        self._demo_tick += 1
        if self._demo_tick % 20 != 0 or not self.classes:
            return
        rng   = np.random.default_rng(self._demo_tick)
        probs = rng.dirichlet(np.ones(len(self.classes))).astype(np.float32)
        idx   = int(np.argmax(probs))
        label = self.classes[idx]
        self._pred_hist.append(label)
        smoothed = Counter(self._pred_hist).most_common(1)[0][0]
        self._record(smoothed, float(probs[idx]) * 100.0, probs, frame_count)


# ============================================================================
# ARGUMENT PARSING + MAIN
# ============================================================================

def _parse_args():
    d = config.get_script_defaults("live_dashboard")
    p = argparse.ArgumentParser(description="CSI Live HAR Dashboard")
    p.add_argument("-p", "--port",      default=d["port"])
    p.add_argument("--baud",    type=int,   default=d["baud"])
    p.add_argument("--models_dir",          default=d["models_dir"])
    p.add_argument("--model",               default=d["model"], choices=config.MODEL_KEYS)
    p.add_argument("--window",  type=int,   default=d["window"])
    p.add_argument("--step",    type=int,   default=d["step"])
    p.add_argument("--ema-alpha", type=float, default=d["ema_alpha"],
                   help="EMA factor for probability smoothing (1.0 = no smoothing)")
    p.add_argument("--conf-thresh", type=float, default=d["conf_thresh"],
                   help="Minimum confidence %% to switch state")
    p.add_argument("--waveform-len", type=int, default=d["waveform_len"])
    p.add_argument("--refresh", type=int,   default=d["refresh_ms"])
    p.add_argument("--max-log", type=int,   default=d["max_log"])
    p.add_argument("--rx-buf",  type=int,   default=d["rx_buf"])
    p.add_argument("--cutoff",      type=float, default=d["cutoff"])
    p.add_argument("--warmup",      type=int,   default=d["warmup"])
    p.add_argument("--hyst-count",  type=int,   default=d["hyst_count"],
                   help="Consecutive confirmations before label switches (hysteresis)")
    config.add_bool_argument(
        p, dest="demo", default=d["demo"],
        help="Synthetic demo data (no hardware/model needed)",
        positive_flags=["--demo"], negative_flags=["--no-demo"],
    )
    return p.parse_args()


def main():
    args = _parse_args()
    app  = QApplication(sys.argv)

    pipeline = le = model = None
    classes  = list(config.get_enabled_training_classes())

    if not args.demo:
        if not _INFERENCE_OK:
            print(f"[ERROR] Missing inference deps: {_IMPORT_ERR}")
            print("        Run with --demo to preview without a model.")
            sys.exit(1)
        try:
            pipeline, le, model = _load_models(args.models_dir, args.model)
            classes = list(le.classes_)
            print(f"[OK]  Model: {args.model}  classes={classes}")
        except FileNotFoundError as e:
            from PyQt5.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setWindowTitle("Models Not Found")
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No trained models found.")
            msg.setInformativeText(
                "Train your models first, then relaunch the dashboard.\n\n"
                "Run:  python csi_ml_pipeline.py --save_model\n\n"
                "Or launch in demo mode:  python live_dashboard.py --demo"
            )
            msg.setDetailedText(str(e))
            msg.exec_()
            sys.exit(1)
    else:
        print("[INFO] Demo mode — no model loaded")

    infer_buf_size = args.window + args.warmup
    stop           = threading.Event()
    reader         = ReaderThread(
        port=args.port, baud=args.baud, demo=args.demo,
        stop_event=stop, waveform_len=args.waveform_len,
        infer_buf_size=infer_buf_size, rx_buf=args.rx_buf,
    )
    reader.start()

    win = DashboardWindow(
        reader=reader,
        pipeline=pipeline, le=le, model=model,
        classes=classes, model_key=args.model,
        window_size=args.window, step=args.step,
        ema_alpha=args.ema_alpha, conf_thresh=args.conf_thresh,
        cutoff=args.cutoff, refresh_ms=args.refresh,
        max_log=args.max_log, port=args.port, baud=args.baud,
        demo=args.demo, models_dir=args.models_dir,
        hyst_count=args.hyst_count,
    )
    center_qt_window(win, w=1440, h=840)
    win.start_timer(args.refresh)
    win.show()

    code = app.exec_()
    stop.set()
    win._infer_worker.join(timeout=2.0)
    reader.join(timeout=2.0)
    sys.exit(code)


if __name__ == "__main__":
    main()
