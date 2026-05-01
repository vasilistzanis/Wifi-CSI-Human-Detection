#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSI Live Waveform Monitor  v4
==============================
Layout:
  +-------------------------------------+------------------+
  |  Waveform  (Y-axis numbers, glow)   |  Subcarrier      |
  |                                     |  Power Dist.     |
  |                                     |  (color bars)    |
  |------|------|------|------|---------|------------------|
  | RSSI | LOSS |  VAR |  LAT |  FREQ                      |
  +------+------+------+------+----------------------------+

Usage:
  python live_waveform.py --port COM6
  python live_waveform.py --demo
"""

import argparse
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

def configure_console_output() -> None:
    """Avoid UnicodeEncodeError on legacy Windows console encodings."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass

configure_console_output()
import numpy as np
import pyqtgraph as pg
import serial
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QSizePolicy,
)

# --- Local imports ------------------------------------------------------------
try:
    from csi_parser import analyze_seq_transition, parse_csi_line
    _PARSER_OK = True
except ImportError:
    class _SeqTransitionFallback:
        def __init__(self, missing_count=0, gap_event=False, duplicate=False, reset=False):
            self.missing_count = missing_count
            self.gap_event = gap_event
            self.duplicate = duplicate
            self.reset = reset

    def analyze_seq_transition(previous_seq, current_seq):
        if previous_seq is None:
            return _SeqTransitionFallback()
        diff = current_seq - previous_seq
        if diff > 1:
            return _SeqTransitionFallback(missing_count=diff - 1, gap_event=True)
        if diff == 0:
            return _SeqTransitionFallback(duplicate=True)
        return _SeqTransitionFallback(reset=True)

    _PARSER_OK = False

# --- Constants & Aesthetics ---------------------------------------------------

# Colors (HSL-tailored for a premium dark feel)
BG       = "#0d1117"  # GitHub dark
SURFACE  = "#161b22"
SURFACE2 = "#21262d"
BORDER   = "#30363d"
ACCENT   = "#58a6ff"  # Soft blue
TEXT_HI  = "#f0f6fc"
TEXT_MID = "#8b949e"
TEXT_DIM = "#484f58"
GRID_CLR = "#30363d66"

# Dimensions
import config
WAVEFORM_LEN = config.DASHBOARD_WAVEFORM_LEN
MAX_SC       = config.MAX_SUBCARRIERS
REFRESH_MS   = config.DASHBOARD_REFRESH_MS
BAUD         = config.BAUD_RATE

# Logic
MOTION_THRESHOLD = config.DASHBOARD_MOTION_THRESHOLD
COLOR_SMOOTH     = config.DASHBOARD_COLOR_SMOOTH

# ========================================================================
# HELPERS
# ========================================================================

def sc_color(mag: float) -> tuple:
    """Map normalized magnitude [0..1] to a cyan-blue gradient."""
    # mag 0 -> (10, 20, 40) dark
    # mag 1 -> (88, 166, 255) accent
    r = int(10 + mag * 78)
    g = int(20 + mag * 146)
    b = int(40 + mag * 215)
    return (r, g, b)

# ========================================================================
# UI COMPONENTS
# ========================================================================

def create_stat_panel(key: str, unit: str):
    """Premium stat box with a key, value, and unit."""
    frame = QFrame()
    frame.setObjectName("stat_strip")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(15, 10, 15, 10)
    layout.setSpacing(2)

    k_lbl = QLabel(key.upper())
    k_lbl.setObjectName("stat_key")
    
    v_lbl = QLabel("0")
    v_lbl.setObjectName("stat_val")
    v_lbl.setAlignment(Qt.AlignCenter)
    
    u_lbl = QLabel(unit)
    u_lbl.setObjectName("stat_unit")
    u_lbl.setAlignment(Qt.AlignRight)

    layout.addWidget(k_lbl)
    layout.addWidget(v_lbl)
    layout.addWidget(u_lbl)
    
    return frame, v_lbl

def create_waveform_plot(window_size):
    """Main waveform plot with subtle glow effect."""
    pw = pg.PlotWidget(background=BG)
    pw.setMenuEnabled(False)
    pw.setMouseEnabled(x=False, y=False)
    
    # Grid
    pw.showGrid(x=True, y=True, alpha=0.1)
    
    # X axis - hidden but defined
    ax_b = pw.getAxis("bottom")
    ax_b.setTicks([])
    ax_b.setPen(pg.mkPen(BORDER))
    
    # Y axis
    ax_l = pw.getAxis("left")
    ax_l.setPen(pg.mkPen(BORDER))
    ax_l.setTextPen(pg.mkPen(TEXT_DIM))
    ax_l.setTickFont(QtGui.QFont("Courier New", 7))
    
    pw.setXRange(0, max(0, window_size - 1), padding=0.01)
    pw.setYRange(0, 3.5, padding=0)
    return pw

def create_sc_plot(max_sc):
    """PlotWidget for the subcarrier power bar chart."""
    pw = pg.PlotWidget(background=SURFACE)
    pw.setMenuEnabled(False)
    pw.setMouseEnabled(x=False, y=False)

    # Y axis
    ax_l = pw.getAxis("left")
    ax_l.setStyle(tickLength=-4, tickTextOffset=3)
    ax_l.setTextPen(pg.mkPen(TEXT_DIM))
    ax_l.setPen(pg.mkPen(BORDER))
    ax_l.setTickFont(QtGui.QFont("Courier New", 7))
    ax_l.setTicks([[(v, f"{v:.1f}") for v in [0, 0.5, 1.0]]])

    # X axis - subcarrier index
    ax_b = pw.getAxis("bottom")
    ax_b.setStyle(tickLength=-4, tickTextOffset=3)
    ax_b.setTextPen(pg.mkPen(TEXT_DIM))
    ax_b.setPen(pg.mkPen(BORDER))
    ax_b.setTickFont(QtGui.QFont("Courier New", 7))
    tick_step = max(1, max_sc // 4)
    tick_positions = list(range(0, max_sc, tick_step))
    if (max_sc - 1) not in tick_positions:
        tick_positions.append(max_sc - 1)
    ticks = [(i, str(i)) for i in sorted(set(tick_positions))]
    ax_b.setTicks([ticks])

    pw.setYRange(0.0, 1.05, padding=0)
    pw.setXRange(-0.5, max_sc - 0.5, padding=0.01)

    # Horizontal reference
    pw.addItem(pg.InfiniteLine(pos=0.5, angle=0,
                               pen=pg.mkPen(GRID_CLR, width=1,
                                            style=Qt.DashLine)))
    return pw

# ========================================================================
# QSS
# ========================================================================

QSS = f"""
QWidget   {{ background: {BG};      color: {TEXT_HI}; }}
QFrame#panel {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QFrame#stat_strip {{
    background: {SURFACE2};
    border-top: 1px solid {BORDER};
}}
QLabel {{ font-family: "Courier New", monospace; }}
QLabel#stat_key {{
    font-size: 9px;
    color: {TEXT_DIM};
    letter-spacing: 2px;
    text-transform: uppercase;
}}
QLabel#stat_val {{
    font-size: 22px;
    font-weight: bold;
    color: {TEXT_HI};
    letter-spacing: 1px;
}}
QLabel#stat_unit {{
    font-size: 9px;
    color: {TEXT_MID};
}}
"""

# ========================================================================
# READER THREAD
# ========================================================================

class ReaderThread(threading.Thread):

    def __init__(self, port, baud, demo, stop_event, window_size=WAVEFORM_LEN, max_sc=MAX_SC, rx_buffer_size=2_000_000, fs=100.0):
        super().__init__(daemon=True)
        self.port        = port
        self.baud        = baud
        self.demo        = demo
        self.stop_event  = stop_event
        self.window_size = window_size
        self.max_sc      = max_sc
        self.rx_buffer_size = rx_buffer_size
        self.fs          = fs
        self._lock       = threading.Lock()

        # Rolling waveform (mean amplitude per frame)
        self._raw    = deque([0.0] * self.window_size, maxlen=self.window_size)
        # Latest full subcarrier magnitude array
        self._sc_mags = np.zeros(self.max_sc, dtype=np.float32)

        # Meta
        self._frames   = 0
        self._rssi     = -100
        self._pkt_loss = 0
        self._last_seq = -1
        self._t_last   = time.perf_counter()
        self._latencies = deque(maxlen=20)

    def get_snapshot(self):
        """Atomically copy state for UI thread."""
        with self._lock:
            raw      = list(self._raw)
            sc_mags  = self._sc_mags.copy()
            fc       = self._frames
            rssi     = self._rssi
            pkt_loss = self._pkt_loss
            lat_ms   = float(np.mean(self._latencies)) * 1000 if self._latencies else 0.0

        wf = np.array(raw, dtype=float)

        # -- Short-window normalization (last 20 frames only) -----------
        NORM_WIN   = 20
        recent_max = wf[-NORM_WIN:].max() if len(wf) >= NORM_WIN else wf.max()
        mx         = max(float(recent_max), 1e-6)
        wf_norm    = np.clip(wf / mx, 0.0, 1.0)

        # Energy: RMS of short-term diff (last 10 samples)
        diff   = np.diff(wf_norm, prepend=wf_norm[0])
        energy = float(np.sqrt(np.mean(diff[-10:] ** 2)))

        # Instantaneous dominant frequency via FFT of waveform
        fps      = 1.0 / max(lat_ms / 1000, 0.001) if lat_ms > 0 else self.fs
        fps      = min(fps, 1000.0)
        freqs    = np.fft.rfftfreq(len(wf_norm), d=1.0 / fps)
        fft_mag  = np.abs(np.fft.rfft(wf_norm - wf_norm.mean()))
        if len(fft_mag) > 1:
            peak_idx  = np.argmax(fft_mag[1:]) + 1
            inst_freq = float(freqs[peak_idx])
        else:
            inst_freq = 0.0

        # Variance of recent window
        variance = float(np.var(wf_norm[-20:]))

        return {
            "wf":       wf_norm,
            "sc_mags":  sc_mags,
            "energy":   energy,
            "frames":   fc,
            "rssi":     rssi,
            "pkt_loss": pkt_loss,
            "lat_ms":   lat_ms,
            "variance": variance,
            "freq_hz":  inst_freq,
        }

    def _push(self, frame, rssi=None, seq=None):
        now = time.perf_counter()
        amp    = np.abs(frame)
        active = amp[amp > 0.0]
        val    = float(active.mean()) if active.size > 0 else 0.0

        n = min(len(amp), self.max_sc)
        sc = np.zeros(self.max_sc, dtype=np.float32)
        sc[:n] = amp[:n]
        mx = sc.max() or 1.0
        sc_norm = sc / mx

        with self._lock:
            self._raw.append(val)
            self._sc_mags[:] = sc_norm
            self._frames += 1
            if rssi is not None: self._rssi = float(rssi)
            if seq is not None and self._last_seq >= 0:
                transition = analyze_seq_transition(self._last_seq, seq)
                if transition.gap_event:
                    self._pkt_loss += transition.missing_count
            if seq is not None: self._last_seq = seq
            self._latencies.append(now - self._t_last)
            self._t_last = now

    def run(self):
        if self.demo:
            self._run_demo(); return
        
        ser = None  # Initialize to None for safe finally block
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.5)
            if os.name == "nt" and hasattr(ser, "set_buffer_size"):
                ser.set_buffer_size(rx_size=self.rx_buffer_size)
            ser.reset_input_buffer()
            print(f"[OK]  {self.port} @ {self.baud}")
        except Exception as e:
            print(f"[ERROR]  {e}  ->  demo mode")
            self._run_demo(); return
        try:
            while not self.stop_event.is_set():
                raw = ser.readline()
                if not raw: continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("CSI_DATA"):
                    continue
                rssi, seq = self._extract_meta(line)
                frame = (parse_csi_line(line) if _PARSER_OK
                         else self._fallback(line))
                if frame is not None:
                    self._push(frame, rssi=rssi, seq=seq)
        finally:
            if ser is not None and ser.is_open:
                ser.close()

    @staticmethod
    def _extract_meta(line):
        try:
            parts = line.split(",")
            seq  = int(parts[1].strip())  if len(parts) > 1 else None
            rssi = float(parts[3].strip()) if len(parts) > 3 else None
            return rssi, seq
        except (ValueError, IndexError):
            return None, None

    @staticmethod
    def _fallback(line):
        """Minimal fallback parser if csi_parser.py is missing. Handles hardware fix."""
        try:
            parts = line.split(",")
            ds = line.split("[")[-1].split("]")[0].replace('"', '').strip()
            vs = np.array([float(v.strip()) for v in ds.split(",") if v.strip()],
                          dtype=np.float32)

            if len(parts) > 13 and int(parts[13]) != 0 and vs.size >= 4:
                vs = vs.copy(); vs[:4] = 0.0

            if vs.size < 2: return None
            imag, real = vs[0::2], vs[1::2]
            return (real + 1j * imag).astype(np.complex64)
        except Exception:
            return None

    def _run_demo(self):
        rng, ph, seq = np.random.default_rng(42), 0.0, 0
        print("[DEMO]  Demo mode")
        while not self.stop_event.is_set():
            time.sleep(0.01)
            ph    += 0.18
            burst  = 1.0 + 2.8 * max(0.0, math.sin(ph * 0.07) ** 8)
            val    = (abs(math.sin(ph) * 0.65 + math.sin(ph * 0.37) * 0.35)
                      * float(rng.uniform(0.88, 1.0)) * burst)
            fake_sc = np.exp(-np.linspace(0, 3, self.max_sc)) * rng.uniform(0.9, 1.1, self.max_sc)
            self._push(fake_sc * (val + 1), rssi=-50 + val*2, seq=seq)
            seq += 1

# ========================================================================
# MAIN WINDOW
# ========================================================================

class WaveformMonitor(QWidget):

    def __init__(self, reader: ReaderThread, port: str, refresh_ms=REFRESH_MS, threshold=MOTION_THRESHOLD, color_smooth=COLOR_SMOOTH):
        super().__init__()
        self.reader    = reader
        self.refresh_ms = refresh_ms
        self.threshold  = threshold
        self.color_smooth = color_smooth
        self._t_color  = 0.0
        self.setWindowTitle(f"CSI Monitor  -  {port}")
        self.resize(1400, 680)
        self.setStyleSheet(QSS)
        pg.setConfigOptions(antialias=True)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(20)

        # Upper row: Waveform + SC Bars
        top_row = QHBoxLayout()
        top_row.setSpacing(20)
        
        # 1. Waveform Panel
        wf_panel = QFrame()
        wf_panel.setObjectName("panel")
        wf_lay = QVBoxLayout(wf_panel)
        wf_lay.setContentsMargins(1, 1, 1, 1)
        self.plot = create_waveform_plot(reader.window_size)
        wf_lay.addWidget(self.plot)
        
        self._x    = np.arange(reader.window_size)
        self._line = self.plot.plot(self._x, np.zeros(reader.window_size), pen=pg.mkPen(ACCENT, width=3))
        
        # Glow / Fill
        self._fill = pg.FillBetweenItem(
            pg.PlotDataItem(self._x, np.zeros(reader.window_size)),
            pg.PlotDataItem(self._x, np.zeros(reader.window_size)),
            brush=pg.mkBrush(88, 166, 255, 45)
        )
        self.plot.addItem(self._fill)
        
        top_row.addWidget(wf_panel, stretch=3)

        # 2. Subcarrier Panel
        sc_panel = QFrame()
        sc_panel.setObjectName("panel")
        sc_lay = QVBoxLayout(sc_panel)
        sc_lay.setContentsMargins(10, 10, 10, 10)
        sc_lay.addWidget(QLabel("SUBCARRIER POWER DISTRIBUTION"))
        self.sc_plot = create_sc_plot(reader.max_sc)
        sc_lay.addWidget(self.sc_plot)
        
        self._bars = pg.BarGraphItem(x=np.arange(reader.max_sc), height=np.zeros(reader.max_sc), width=0.7, brush=ACCENT)
        self.sc_plot.addItem(self._bars)
        
        top_row.addWidget(sc_panel, stretch=1)
        root.addLayout(top_row)

        # Bottom row: Stats
        stat_row = QHBoxLayout()
        stat_row.setSpacing(15)
        
        self.panels = {}
        for key, unit in [("RSSI", "dBm"), ("Packet Loss", "pkts"), ("Variance", "Var"), ("Latency", "ms"), ("Dom. Freq", "Hz"), ("Frames", "count")]:
            p, v = create_stat_panel(key, unit)
            stat_row.addWidget(p)
            self.panels[key] = v

        self._v_rssi, self._v_loss, self._v_var, self._v_lat, self._v_freq, self._v_frames = \
            self.panels["RSSI"], self.panels["Packet Loss"], self.panels["Variance"], self.panels["Latency"], self.panels["Dom. Freq"], self.panels["Frames"]

        root.addLayout(stat_row)

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(self.refresh_ms)

    def update_ui(self):
        data = self.reader.get_snapshot()
        wf, sc_mags, energy, fc, rssi, loss, lat_ms, var, freq_hz = \
            data["wf"], data["sc_mags"], data["energy"], data["frames"], data["rssi"], data["pkt_loss"], data["lat_ms"], data["variance"], data["freq_hz"]

        # Motion color logic
        target = 1.0 if var > self.threshold else 0.0
        self._t_color += (target - self._t_color) * self.color_smooth
        t = self._t_color
        
        # Color interpolation (Dark -> Cyan -> Bright Cyan)
        r = int(88 * t + 40 * (1-t))
        g = int(166 * t + 90 * (1-t))
        b = int(255 * t + 160 * (1-t))
        
        self._fill.setBrush(pg.mkBrush(r, g, b, int(45 + t * 85)))
        self._line.setData(self._x, wf)
        self._line.setPen(pg.mkPen((r, g, b, 255), width=3.0))

        # Subcarrier bars
        brushes = [pg.mkBrush(*sc_color(float(v)), 200) for v in sc_mags]
        self._bars.setOpts(height=sc_mags.astype(float), brushes=brushes)

        # Stats
        rssi_color = (ACCENT if rssi > -65 else "#ff8c14" if rssi > -80 else "#f85149")
        self._color_stat(self._v_rssi, f"{rssi:.0f}", rssi_color)
        loss_color = (ACCENT if loss == 0 else "#ff8c14" if loss < 50 else "#f85149")
        self._color_stat(self._v_loss, str(loss), loss_color)
        self._color_stat(self._v_var, f"{var:.4f}", TEXT_HI if var < self.threshold else "#aff5b4")
        self._color_stat(self._v_lat, f"{lat_ms:.1f}", TEXT_HI if lat_ms < 20 else "#ff8c14")
        self._color_stat(self._v_freq, f"{freq_hz:.2f}", TEXT_HI)
        self._color_stat(self._v_frames, str(fc), TEXT_MID)

    def _color_stat(self, lbl, txt, color):
        lbl.setText(txt)
        lbl.setStyleSheet(f"color: {color};")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--port",      default=config.SERIAL_PORT, help="Serial port (e.g. COM6). Optional for --demo.")
    p.add_argument("--baud",      type=int,   default=BAUD)
    p.add_argument("--window",    type=int,   default=WAVEFORM_LEN)
    p.add_argument("--refresh",   type=int,   default=REFRESH_MS)
    p.add_argument("--threshold", type=float, default=MOTION_THRESHOLD)
    p.add_argument("--smooth",    type=float, default=COLOR_SMOOTH)
    p.add_argument("--max-sc",    type=int,   default=MAX_SC)
    p.add_argument("--rx-buf",    type=int,   default=config.RX_BUFFER_SIZE, help="Windows RX buffer size")
    p.add_argument("--demo",      action="store_true", help="Run with synthetic data")
    p.add_argument("--fs",        type=float, default=config.SAMPLING_RATE, help="Expected sampling frequency (Hz)")
    return p.parse_args()

def main():
    args = parse_args()
    if args.max_sc < 1:
        print("[ERROR] --max-sc must be at least 1")
        return 1
    app  = QApplication(sys.argv)
    stop   = threading.Event()
    reader = ReaderThread(port=args.port, baud=args.baud, demo=args.demo, stop_event=stop, 
                          window_size=args.window, max_sc=args.max_sc, rx_buffer_size=args.rx_buf, fs=args.fs)
    reader.start()
    win = WaveformMonitor(reader=reader, port=args.port, refresh_ms=args.refresh, 
                          threshold=args.threshold, color_smooth=args.smooth)
    screen_rect = app.primaryScreen().availableGeometry()
    win.move((screen_rect.width() - win.width()) // 2, (screen_rect.height() - win.height()) // 2)
    win.show()
    res = app.exec_()
    stop.set()
    reader.join(timeout=1.0)
    return res

if __name__ == "__main__":
    sys.exit(main())
