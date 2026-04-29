#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
CSI Live Waveform Monitor
=========================
Oscilloscope-style window with:
  - Filled waveform that "breathes" - cyan=calm -> amber/red=motion
  - Glow layers (3 levels) that intensify based on signal energy
  - Motion badge + dynamic color intensity
  - No diff panel - motion is visible from the raw signal shape itself

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
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


try:
    from csi_parser import parse_csi_line
    _PARSER_OK = True
except ImportError:
    _PARSER_OK = False


# ========================================================================
# CONFIG
# ========================================================================


BAUD             = 2_000_000
WAVEFORM_LEN     = 60
REFRESH_MS       = 50
MOTION_THRESHOLD = 0.18   # normalized motion energy threshold


# Color response speed (smoothing factor 0-1)
# 1.0 = instant, 0.05 = slow / inertia
COLOR_SMOOTH     = 0.12
MAX_SC           = 64


# ========================================================================
# THEME
# ========================================================================


BG       = "#04080f"
SURFACE  = "#060b15"
GRID_CLR = "#0b1520"
TEXT_DIM = "#1e3348"
TEXT_MID = "#4a7080"


#                    R    G    B
CLR_CALM   = (  32, 200, 255)   # cyan-aqua   - Calm
CLR_WARN   = ( 255, 170,  20)   # amber       - Motion
CLR_PEAK   = ( 255,  45,  70)   # red         - Intense Motion


QSS = f"""
QWidget {{ background: {BG}; }}
"""


def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def tri_lerp(t):
    """
    3-stop color ramp:
      0.0 -> CLR_CALM  (cyan)
      0.5 -> CLR_WARN  (amber)
      1.0 -> CLR_PEAK  (red)
    """
    if t <= 0.5:
        return lerp_color(CLR_CALM, CLR_WARN, t * 2)
    return lerp_color(CLR_WARN, CLR_PEAK, (t - 0.5) * 2)


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
        self._raw        = deque([0.0] * self.window_size, maxlen=self.window_size)
        self._frames     = 0


    def snapshot(self):
        with self._lock:
            raw = list(self._raw)
            fc  = self._frames


        wf = np.array(raw, dtype=float)
        mx = wf.max() or 1.0
        wf_norm = wf / mx


        # Energy: rms of short-term differences (last 10 samples)
        diff   = np.diff(wf_norm, prepend=wf_norm[0])
        energy = float(np.sqrt(np.mean(diff[-10:] ** 2)))
        return wf_norm, energy, fc


    def _push(self, frame):
        amp    = np.abs(frame)
        # Use all subcarriers - exclude only DC/null (magnitude == 0)
        active = amp[amp > 0.0]
        val    = float(active.mean()) if active.size > 0 else 0.0
        with self._lock:
            self._raw.append(val)
            self._frames += 1


    def run(self):
        if self.demo:
            self._run_demo(); return
        ser = None
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
                frame = (parse_csi_line(line) if _PARSER_OK
                         else self._fallback(line))
                if frame is not None:
                    self._push(frame)
        finally:
            if ser is not None and ser.is_open:
                ser.close()


    @staticmethod
    def _fallback(line):
        try:
            ds = line.split("[")[-1].split("]")[0].replace('"', '').strip()
            vs = [int(v.strip()) for v in ds.split(",") if v.strip()]
            if len(vs) < 2: return None
            return np.array([complex(vs[i+1], vs[i])
                             for i in range(0, len(vs)-1, 2)], dtype=np.complex64)
        except Exception:
            return None


    def _run_demo(self):
        rng, ph = np.random.default_rng(0), 0.0
        print("[INFO]  Demo mode")
        while not self.stop_event.is_set():
            time.sleep(0.01)
            ph += 0.18
            burst = 1.0 + 2.8 * max(0.0, math.sin(ph * 0.07) ** 8)
            val   = (abs(math.sin(ph) * 0.65 + math.sin(ph * 0.37) * 0.35)
                     * float(rng.uniform(0.88, 1.0)) * burst)
            with self._lock:
                self._raw.append(val)
                self._frames += 1


# ========================================================================
# WAVEFORM WINDOW
# ========================================================================


class WaveformMonitor(QWidget):


    def __init__(self, reader, port, refresh_ms=REFRESH_MS, threshold=MOTION_THRESHOLD, color_smooth=COLOR_SMOOTH):
        super().__init__()
        self.reader    = reader
        self.refresh_ms = refresh_ms
        self.threshold  = threshold
        self.color_smooth = color_smooth
        self._t_color  = 0.0   # smoothed color parameter [0-1]
        self.setWindowTitle(f"CSI Monitor - {port}")
        self.resize(1200, 500)
        self.setStyleSheet(QSS)
        pg.setConfigOptions(antialias=True)


        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_plot())


        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(self.refresh_ms)


    # -- Plot construction ---------------------------------------------


    def _build_plot(self):
        pw = pg.PlotWidget(background=BG)
        pw.setMenuEnabled(False)
        pw.setMouseEnabled(x=False, y=False)
        pw.hideAxis("bottom")
        pw.hideAxis("left")
        pw.setYRange(0.0, 1.12, padding=0)
        pw.setXRange(0, self.reader.window_size - 1, padding=0.01)


        # Subtle grid
        for yv in np.linspace(0, 1.1, 7):
            pw.addItem(pg.InfiniteLine(pos=yv, angle=0,
                                       pen=pg.mkPen(GRID_CLR, width=1)))
        for xv in np.linspace(0, self.reader.window_size - 1, 9):
            pw.addItem(pg.InfiniteLine(pos=xv, angle=90,
                                       pen=pg.mkPen(GRID_CLR, width=1)))


        x  = np.arange(self.reader.window_size, dtype=float)
        y0 = np.zeros(self.reader.window_size)


        # Layer 1 - wide outer glow
        self._g3 = pw.plot(x, y0, pen=pg.mkPen((*CLR_CALM, 10), width=44))
        # Layer 2 - mid glow
        self._g2 = pw.plot(x, y0, pen=pg.mkPen((*CLR_CALM, 30), width=18))
        # Layer 3 - inner glow
        self._g1 = pw.plot(x, y0, pen=pg.mkPen((*CLR_CALM, 65), width=7))


        # Filled body
        _z = pg.PlotDataItem(x, y0, pen=None)
        _w = pg.PlotDataItem(x, y0, pen=None)
        self._fill = pg.FillBetweenItem(_z, _w, brush=pg.mkBrush(*CLR_CALM, 35))
        pw.addItem(self._fill)


        # Sharp top line
        self._line = pw.plot(x, y0, pen=pg.mkPen((*CLR_CALM, 255), width=2.0))


        self._x  = x
        self._y0 = y0
        return pw


    # -- Refresh -------------------------------------------------------


    def _refresh(self):
        wf, energy, _ = self.reader.snapshot()


        # Smooth color parameter toward target energy
        t_target       = min(1.0, energy / (self.threshold * 1.5))
        self._t_color += self.color_smooth * (t_target - self._t_color)
        t = self._t_color


        r, g, b = tri_lerp(t)


        self._g3.setData(self._x, wf)
        self._g3.setPen(pg.mkPen((r, g, b, int(6  + t * 24)), width=44))


        self._g2.setData(self._x, wf)
        self._g2.setPen(pg.mkPen((r, g, b, int(22 + t * 58)), width=18))


        self._g1.setData(self._x, wf)
        self._g1.setPen(pg.mkPen((r, g, b, int(55 + t * 95)), width=7))


        self._fill.setCurves(
            pg.PlotDataItem(self._x, self._y0, pen=None),
            pg.PlotDataItem(self._x, wf,       pen=None),
        )
        self._fill.setBrush(pg.mkBrush(r, g, b, int(30 + t * 85)))


        self._line.setData(self._x, wf)
        self._line.setPen(pg.mkPen((r, g, b, 255), width=2.0))




# ========================================================================
# MAIN
# ========================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--port",      default="COM6", help="Serial port (e.g. COM6).")
    p.add_argument("--baud",      type=int,   default=BAUD)
    p.add_argument("--window",    type=int,   default=WAVEFORM_LEN)
    p.add_argument("--refresh",   type=int,   default=REFRESH_MS)
    p.add_argument("--threshold", type=float, default=MOTION_THRESHOLD)
    p.add_argument("--smooth",    type=float, default=COLOR_SMOOTH)
    p.add_argument("--max-sc",    type=int,   default=MAX_SC)
    p.add_argument("--rx-buf",    type=int,   default=2_000_000, help="Windows RX buffer size")
    p.add_argument("--demo",      action="store_true", help="Run with synthetic data")
    p.add_argument("--fs",        type=float, default=100.0, help="Expected sampling frequency (Hz)")
    return p.parse_args()


def main():
    args = parse_args()
    app  = QApplication(sys.argv)


    stop   = threading.Event()
    reader = ReaderThread(port=args.port, baud=args.baud, demo=args.demo, stop_event=stop, 
                          window_size=args.window, max_sc=args.max_sc, rx_buffer_size=args.rx_buf, fs=args.fs)
    reader.start()


    win = WaveformMonitor(reader=reader, port=args.port, refresh_ms=args.refresh, 
                          threshold=args.threshold, color_smooth=args.smooth)
    win.show()


    code = app.exec_()
    stop.set()
    reader.join(timeout=2.0)
    sys.exit(code)


if __name__ == "__main__":
    main()
