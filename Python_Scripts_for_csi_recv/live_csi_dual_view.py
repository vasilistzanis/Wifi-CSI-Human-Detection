#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSI Live Dual View
==================
Two side-by-side windows fed from a single reader thread:

  Left  — Time domain    : Amplitude vs Time  (top)   — one line per subcarrier
                           Phase vs Time       (bottom) — one line per subcarrier

  Right — Current frame  : Amplitude vs Subcarrier Index  (live line + fill)
                           Phase vs Subcarrier Index       (live line, radians)

Reader uses a circular buffer (zero-copy) and pre-computes amp / phase so the
UI thread calls only setData() without any heavy numpy on every tick.

Usage
-----
  python live_csi_dual_view.py
  python live_csi_dual_view.py --port COM6
  python live_csi_dual_view.py --demo
"""

import argparse
import math
import os
import sys
import threading
import time

from csi_parser import configure_console_output
configure_console_output()

import numpy as np

os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"
import pyqtgraph as pg
import serial
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget

import config
from plot_window_utils import center_qt_window

try:
    from csi_parser import parse_csi_line
    _PARSER_OK = True
except ImportError:
    _PARSER_OK = False


# ============================================================================
# THEME
# ============================================================================

_BG       = "#04080f"
_TEXT_MID = "#4a7080"
_TEXT_BRT = "#7aacbc"
_QSS      = f"QWidget {{ background: {_BG}; }}"

_AMP_PEN    = pg.mkPen(color=(0,   200, 255),      width=2)
_PHASE_PEN  = pg.mkPen(color=(255, 100, 200),      width=2)
_AMP_BRUSH  = pg.mkBrush(0, 180, 255, 50)



# ============================================================================
# READER THREAD
# ============================================================================

class ReaderThread(threading.Thread):
    """
    Reads serial frames and maintains circular buffers.

    Pre-computes per-frame amplitude (norm. 0–1) and phase (norm. 0–1 for
    heatmap; raw radians for the subcarrier line plot) so the UI thread does
    zero heavy computation on every tick.

    snapshot() → dict with all fields needed by both windows, under lock.
    """

    def __init__(self, port, baud, demo, stop_event, buffer_size, max_sc, rx_buf):
        super().__init__(daemon=True)
        self.port        = port
        self.baud        = baud
        self.demo        = demo
        self.stop_event  = stop_event
        self.buffer_size = buffer_size
        self.max_sc      = max_sc
        self.rx_buf      = rx_buf

        self._lock      = threading.Lock()
        self._ptr       = 0
        self._n_active  = 0

        # Circular buffers — written once per frame (O(n)), read per UI tick
        self._amp_buf   = np.zeros((buffer_size, max_sc), dtype=np.float32)
        self._phase_buf = np.zeros((buffer_size, max_sc), dtype=np.float32)

        # Latest frame — subcarrier window reads these
        self._last_amp       = np.zeros(max_sc, dtype=np.float32)
        self._last_phase_rad = np.zeros(max_sc, dtype=np.float32)

    # ---- public API --------------------------------------------------------

    def snapshot(self):
        """
        Return a consistent snapshot of all buffers.
        Returns None until the first frame arrives.
        All arrays are copies — safe to use outside the lock.
        """
        with self._lock:
            n = self._n_active
            if n == 0:
                return None
            return {
                "amp":       self._amp_buf[:, :n].copy(),
                "phase":     self._phase_buf[:, :n].copy(),
                "ptr":       self._ptr,
                "last_amp":  self._last_amp[:n].copy(),
                "last_phase_rad": self._last_phase_rad[:n].copy(),
                "n":         n,
            }

    # ---- internal ----------------------------------------------------------

    def _push(self, complex_frame: np.ndarray) -> None:
        n = min(complex_frame.size, self.max_sc)
        cf = complex_frame[:n].astype(np.complex64)

        # Amplitude: per-frame max-normalised [0, 1]
        amp = np.abs(cf)
        mx  = amp.max()
        if mx > 0:
            amp = amp / mx

        # Phase: raw radians for line plot; normalised [0,1] for heatmap
        phase_rad  = np.angle(cf)                           # [-π, π]
        phase_norm = (phase_rad + np.pi) / (2 * np.pi)     # [0, 1]

        with self._lock:
            p = self._ptr
            self._amp_buf[p, :n]   = amp
            self._amp_buf[p, n:]   = 0.0
            self._phase_buf[p, :n] = phase_norm
            self._phase_buf[p, n:] = 0.5
            self._last_amp[:n]       = amp
            self._last_phase_rad[:n] = phase_rad
            self._n_active = max(self._n_active, n)
            self._ptr = (p + 1) % self.buffer_size

    def run(self):
        if self.demo:
            self._run_demo()
            return
        ser = None
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.5)
            if os.name == "nt" and hasattr(ser, "set_buffer_size"):
                ser.set_buffer_size(rx_size=self.rx_buf)
            ser.reset_input_buffer()
            print(f"[OK]  Connected: {self.port} @ {self.baud}")
        except Exception as e:
            print(f"[ERROR] {e}  → demo mode")
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
            if ser is not None and ser.is_open:
                ser.close()

    def _run_demo(self):
        """Synthetic: two drifting Gaussian peaks with time-varying phase."""
        rng = np.random.default_rng(42)
        t   = 0.0
        print("[INFO] Demo mode active")
        while not self.stop_event.is_set():
            time.sleep(0.01)
            t += 0.05
            sc  = np.arange(self.max_sc, dtype=np.float32)
            p1  = np.exp(-0.5 * ((sc - (self.max_sc * 0.3 + 10 * math.sin(t * 0.7))) / 6) ** 2)
            p2  = np.exp(-0.5 * ((sc - (self.max_sc * 0.7 +  8 * math.cos(t * 0.5))) / 5) ** 2)
            amp = (p1 * 0.7 + p2 * 0.5) * (0.6 + 0.4 * math.sin(t * 1.3))
            amp = amp + rng.uniform(0, 0.03, self.max_sc).astype(np.float32)
            # Varying phase across subcarriers
            phase = t * 0.8 + sc / self.max_sc * 2 * np.pi
            fake  = (amp * (np.cos(phase) + 1j * np.sin(phase))).astype(np.complex64)
            self._push(fake)


# ============================================================================
# WINDOW 1 — TIME DOMAIN (line plots)
# ============================================================================

class TimeWindow(QWidget):
    """
    Amplitude vs Time (top) + Phase vs Time (bottom).
    One line per subcarrier (strided to _MAX_LINES when n > _MAX_LINES).
    Circular buffer is unrolled into chronological order on each tick.
    """

    _MAX_LINES = 32   # max simultaneous subcarrier lines

    def __init__(self, reader: ReaderThread, refresh_ms: int):
        super().__init__()
        self.reader = reader
        self.setWindowTitle("CSI — Time Domain")
        self.setStyleSheet(_QSS)
        pg.setConfigOptions(antialias=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._x = np.arange(reader.buffer_size, dtype=np.float32)

        # Build per-subcarrier pen colours from colormaps
        try:
            lut_a = pg.colormap.get("viridis").getLookupTable(0.0, 1.0, self._MAX_LINES)
            lut_p = pg.colormap.get("plasma").getLookupTable(0.0, 1.0, self._MAX_LINES)
        except Exception:
            lut_a = [(0, 200, 255)] * self._MAX_LINES
            lut_p = [(255, 100, 200)] * self._MAX_LINES

        # ---- Amplitude lines ----
        self._pw_amp = self._make_pw(
            "Amplitude vs Time  (each line = one subcarrier)",
            "Amplitude (0–1)", "Frame",
        )
        self._pw_amp.setYRange(0.0, 1.05, padding=0)
        self._pw_amp.setXRange(0, reader.buffer_size, padding=0)
        self._pw_amp.showGrid(x=False, y=True, alpha=0.15)
        self._curves_amp = [
            self._pw_amp.plot(pen=pg.mkPen(color=tuple(lut_a[i]), width=1))
            for i in range(self._MAX_LINES)
        ]

        # ---- Phase lines ----
        self._pw_phase = self._make_pw(
            "Phase vs Time  (each line = one subcarrier, radians)",
            "Phase (rad)", "Frame",
        )
        self._pw_phase.setYRange(-np.pi - 0.2, np.pi + 0.2, padding=0)
        self._pw_phase.setXRange(0, reader.buffer_size, padding=0)
        self._pw_phase.showGrid(x=False, y=True, alpha=0.15)
        self._pw_phase.addItem(
            pg.InfiniteLine(pos=0, angle=0,
                            pen=pg.mkPen(color=(255, 255, 255, 40), width=1))
        )
        self._curves_phase = [
            self._pw_phase.plot(pen=pg.mkPen(color=tuple(lut_p[i]), width=1))
            for i in range(self._MAX_LINES)
        ]

        layout.addWidget(self._pw_amp)
        layout.addWidget(self._pw_phase)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(refresh_ms)

    def _make_pw(self, title, left_label, bottom_label):
        pw = pg.PlotWidget(background=_BG)
        pw.setMenuEnabled(False)
        pw.setMouseEnabled(x=False, y=False)
        pw.setTitle(title, color=_TEXT_BRT, size="9pt")
        pw.setLabel("left",   left_label,   color=_TEXT_MID, size="9pt")
        pw.setLabel("bottom", bottom_label, color=_TEXT_MID, size="9pt")
        pw.getAxis("left").setTextPen(pg.mkPen(_TEXT_MID))
        pw.getAxis("bottom").setTextPen(pg.mkPen(_TEXT_MID))
        return pw

    def _refresh(self):
        snap = self.reader.snapshot()
        if snap is None:
            return

        n   = snap["n"]
        ptr = snap["ptr"]

        # Unroll circular buffer → chronological order (oldest frame first)
        amp   = np.roll(snap["amp"],   -ptr, axis=0)                        # (buf, n)
        phase = np.roll(snap["phase"], -ptr, axis=0) * (2 * np.pi) - np.pi # (buf, n) → radians

        stride  = max(1, n // self._MAX_LINES)
        indices = list(range(0, n, stride))[:self._MAX_LINES]
        x       = self._x  # length = buffer_size

        for k, sc_idx in enumerate(indices):
            self._curves_amp[k].setData(x, amp[:, sc_idx])
            self._curves_phase[k].setData(x, phase[:, sc_idx])

        # Clear any unused curves
        for k in range(len(indices), self._MAX_LINES):
            self._curves_amp[k].setData([], [])
            self._curves_phase[k].setData([], [])


# ============================================================================
# WINDOW 2 — SUBCARRIER DOMAIN (current frame)
# ============================================================================

class SubcarrierWindow(QWidget):
    """
    Amplitude vs Subcarrier Index (top) + Phase vs Subcarrier Index (bottom).
    Both show only the latest received frame — updates every tick.
    """

    def __init__(self, reader: ReaderThread, refresh_ms: int):
        super().__init__()
        self.reader  = reader
        self._n      = reader.max_sc
        self._x      = np.arange(reader.max_sc, dtype=float)
        self._zeros  = np.zeros(reader.max_sc)
        self.setWindowTitle("CSI — Current Frame  (Subcarrier Domain)")
        self.setStyleSheet(_QSS)
        pg.setConfigOptions(antialias=True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- Amplitude line + fill ----
        self._pw_amp = self._make_pw(
            "Amplitude vs Subcarrier Index  (current frame)",
            "Amplitude (norm. 0–1)", "Subcarrier Index",
        )
        self._pw_amp.setYRange(0.0, 1.05, padding=0)
        self._pw_amp.showGrid(x=False, y=True, alpha=0.18)

        _fill_lo = pg.PlotDataItem(self._x, self._zeros, pen=None)
        _fill_hi = pg.PlotDataItem(self._x, self._zeros, pen=None)
        self._fill     = pg.FillBetweenItem(_fill_lo, _fill_hi, brush=_AMP_BRUSH)
        self._fill_lo  = _fill_lo
        self._fill_hi  = _fill_hi
        self._pw_amp.addItem(self._fill)
        self._curve_amp = self._pw_amp.plot(self._x, self._zeros, pen=_AMP_PEN)

        # ---- Phase line ----
        self._pw_phase = self._make_pw(
            "Phase vs Subcarrier Index  (current frame, radians)",
            "Phase (rad)", "Subcarrier Index",
        )
        self._pw_phase.setYRange(-np.pi, np.pi, padding=0.05)
        self._pw_phase.showGrid(x=False, y=True, alpha=0.18)
        # Zero-line reference
        self._pw_phase.addItem(
            pg.InfiniteLine(pos=0, angle=0,
                            pen=pg.mkPen(color=(255, 255, 255, 40), width=1))
        )
        self._curve_phase = self._pw_phase.plot(self._x, self._zeros, pen=_PHASE_PEN)

        layout.addWidget(self._pw_amp)
        layout.addWidget(self._pw_phase)

        self._x_set = False
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(refresh_ms)

    def _make_pw(self, title, left_label, bottom_label):
        pw = pg.PlotWidget(background=_BG)
        pw.setMenuEnabled(False)
        pw.setMouseEnabled(x=False, y=False)
        pw.setTitle(title, color=_TEXT_BRT, size="9pt")
        pw.setLabel("left",   left_label,   color=_TEXT_MID, size="9pt")
        pw.setLabel("bottom", bottom_label, color=_TEXT_MID, size="9pt")
        pw.getAxis("left").setTextPen(pg.mkPen(_TEXT_MID))
        pw.getAxis("bottom").setTextPen(pg.mkPen(_TEXT_MID))
        return pw

    def _refresh(self):
        snap = self.reader.snapshot()
        if snap is None:
            return

        n         = snap["n"]
        last_amp  = snap["last_amp"]
        last_phase = snap["last_phase_rad"]

        if not self._x_set:
            self._pw_amp.setXRange(-0.5, n - 0.5, padding=0)
            self._pw_phase.setXRange(-0.5, n - 0.5, padding=0)
            self._x_set = True

        x = self._x[:n]

        # Amplitude: line + filled area beneath
        self._curve_amp.setData(x, last_amp)
        self._fill_hi.setData(x, last_amp)
        self._fill_lo.setData(x, self._zeros[:n])
        self._fill.setCurves(self._fill_lo, self._fill_hi)

        # Phase: unwrapped so no jumps at ±π boundary
        self._curve_phase.setData(x, np.unwrap(last_phase))


# ============================================================================
# WINDOW PLACEMENT
# ============================================================================

def _place_side_by_side(app, win_left, win_right):
    """Place two windows side by side on the primary screen."""
    screen = app.primaryScreen().availableGeometry()
    half_w = screen.width() // 2
    h      = min(screen.height(), 800)
    y      = screen.y() + max(0, (screen.height() - h) // 2)
    win_left.setGeometry(screen.x(),           y, half_w, h)
    win_right.setGeometry(screen.x() + half_w, y, half_w, h)


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def _parse_args():
    defaults = config.get_script_defaults("live_csi_dual_view")
    p = argparse.ArgumentParser(description="CSI Live Dual View")
    p.add_argument("-p", "--port",    default=defaults["port"])
    p.add_argument("--baud",    type=int, default=defaults["baud"])
    p.add_argument("--buffer",  type=int, default=defaults["buffer_size"],
                   help="History frames in the time-domain window")
    p.add_argument("--refresh", type=int, default=defaults["refresh_ms"],
                   help="UI refresh interval (ms)")
    p.add_argument("--max-sc",  type=int, default=defaults["max_sc"],
                   help="Max subcarriers to display")
    p.add_argument("--rx-buf",  type=int, default=defaults["rx_buf"])
    p.add_argument(
        "--window", choices=["both", "time", "subcarrier"], default="both",
        help="Which window(s) to open: both (default), time, or subcarrier",
    )
    config.add_bool_argument(
        p, dest="demo", default=defaults["demo"],
        help="Run with synthetic demo data",
        positive_flags=["--demo"], negative_flags=["--no-demo"],
    )
    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = _parse_args()
    app  = QApplication(sys.argv)

    stop   = threading.Event()
    reader = ReaderThread(
        port=args.port, baud=args.baud, demo=args.demo,
        stop_event=stop, buffer_size=args.buffer,
        max_sc=args.max_sc, rx_buf=args.rx_buf,
    )
    reader.start()

    show_time = args.window in ("both", "time")
    show_sc   = args.window in ("both", "subcarrier")

    win_time = TimeWindow(reader=reader, refresh_ms=args.refresh) if show_time else None
    win_sc   = SubcarrierWindow(reader=reader, refresh_ms=args.refresh) if show_sc else None

    if win_time and win_sc:
        _place_side_by_side(app, win_time, win_sc)
    elif win_time:
        center_qt_window(win_time)
    elif win_sc:
        center_qt_window(win_sc)

    if win_time:
        win_time.show()
    if win_sc:
        win_sc.show()

    code = app.exec_()
    stop.set()
    reader.join(timeout=2.0)
    sys.exit(code)


if __name__ == "__main__":
    main()
