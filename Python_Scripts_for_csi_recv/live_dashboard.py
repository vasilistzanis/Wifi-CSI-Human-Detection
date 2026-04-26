#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSI Live Waveform Monitor  v4
==============================
Layout:
  ┌─────────────────────────────────────┬──────────────────┐
  │  Waveform  (Y-axis numbers, glow)   │  Subcarrier      │
  │                                     │  Power Dist.     │
  │                                     │  (color bars)    │
  ├──────┬──────┬──────┬──────┬─────────┴──────────────────┤
  │ RSSI │ LOSS │  VAR │  LAT │  FREQ                      │
  └──────┴──────┴──────┴──────┴────────────────────────────┘

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

import numpy as np
import pyqtgraph as pg
import serial
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QSizePolicy,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from csi_parser import parse_csi_line
    _PARSER_OK = True
except ImportError:
    _PARSER_OK = False

# ════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════

BAUD             = 2_000_000
WAVEFORM_LEN     = 60          # rolling window — shorter = more responsive
REFRESH_MS       = 50          # UI timer (20 Hz)
MOTION_THRESHOLD = 0.15        # lower = more sensitive to subtle motion
COLOR_SMOOTH     = 0.15        # color inertia (higher = faster response)
MAX_SC           = 128         # bar chart width (pad/trim subcarrier array)

# ════════════════════════════════════════════════════════════════════════
# PALETTE  — lighter navy (easier to read on screen)
# ════════════════════════════════════════════════════════════════════════

BG        = "#0d1520"
SURFACE   = "#111d2b"
SURFACE2  = "#162030"
BORDER    = "#1e3045"
GRID_CLR  = "#192840"
TEXT_HI   = "#c8ddf0"
TEXT_MID  = "#6a90aa"
TEXT_DIM  = "#2e4a60"
ACCENT    = "#20c8ff"

CLR_CALM  = ( 32, 200, 255)   # cyan
CLR_WARN  = (255, 170,  20)   # amber
CLR_PEAK  = (255,  45,  70)   # red

# Subcarrier power bar colormap stops  (magnitude 0→1)
SC_COLORS = [
    (0.00, ( 20, 120, 200)),   # low   — blue
    (0.35, ( 32, 200, 255)),   # mid   — cyan
    (0.65, (255, 170,  20)),   # high  — amber
    (1.00, (255,  45,  70)),   # peak  — red
]


def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def tri_lerp(t):
    """3-stop ramp: calm → warn → peak."""
    if t <= 0.5:
        return lerp_color(CLR_CALM, CLR_WARN, t * 2)
    return lerp_color(CLR_WARN, CLR_PEAK, (t - 0.5) * 2)


def sc_color(v):
    """Map subcarrier magnitude [0,1] → (R,G,B) using SC_COLORS ramp."""
    v = max(0.0, min(1.0, v))
    for i in range(len(SC_COLORS) - 1):
        t0, c0 = SC_COLORS[i]
        t1, c1 = SC_COLORS[i + 1]
        if v <= t1:
            t = (v - t0) / (t1 - t0 + 1e-9)
            return lerp_color(c0, c1, t)
    return SC_COLORS[-1][1]


# ════════════════════════════════════════════════════════════════════════
# QSS
# ════════════════════════════════════════════════════════════════════════

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




# ════════════════════════════════════════════════════════════════════════
# READER THREAD
# ════════════════════════════════════════════════════════════════════════

class ReaderThread(threading.Thread):

    def __init__(self, port, demo, stop_event):
        super().__init__(daemon=True)
        self.port       = port
        self.demo       = demo
        self.stop_event = stop_event
        self._lock      = threading.Lock()

        # Rolling waveform (mean amplitude per frame)
        self._raw    = deque([0.0] * WAVEFORM_LEN, maxlen=WAVEFORM_LEN)
        # Latest full subcarrier magnitude array
        self._sc_mags = np.zeros(MAX_SC, dtype=np.float32)

        # Stats
        self._frames      = 0
        self._rssi        = -100.0
        self._last_seq    = -1
        self._pkt_loss    = 0        # cumulative lost packets
        self._t_last      = time.perf_counter()
        self._latencies   = deque([0.0] * 20, maxlen=20)

    # ── Public snapshot ──────────────────────────────────────────────

    def snapshot(self):
        """Return everything the UI needs in one atomic copy."""
        with self._lock:
            raw      = list(self._raw)
            sc_mags  = self._sc_mags.copy()
            fc       = self._frames
            rssi     = self._rssi
            pkt_loss = self._pkt_loss
            lat_ms   = float(np.mean(self._latencies)) * 1000

        wf = np.array(raw, dtype=float)

        # ── Short-window normalization (last 20 frames only) ───────────
        # Full-buffer max keeps historic peaks for 80 frames and squashes
        # subsequent quiet signal → waveform looks flat / unresponsive.
        # Using only the recent window the normalization resets quickly.
        NORM_WIN   = 20
        recent_max = wf[-NORM_WIN:].max() if len(wf) >= NORM_WIN else wf.max()
        mx         = max(float(recent_max), 1e-6)
        wf_norm    = np.clip(wf / mx, 0.0, 1.0)

        # Energy: RMS of short-term diff (last 10 samples)
        diff   = np.diff(wf_norm, prepend=wf_norm[0])
        energy = float(np.sqrt(np.mean(diff[-10:] ** 2)))

        # Instantaneous dominant frequency via FFT of waveform
        # Sample rate ≈ 1 frame / mean_latency (capped at 1000 Hz)
        fps      = 1.0 / max(lat_ms / 1000, 0.001) if lat_ms > 0 else 100.0
        fps      = min(fps, 1000.0)
        freqs    = np.fft.rfftfreq(len(wf_norm), d=1.0 / fps)
        fft_mag  = np.abs(np.fft.rfft(wf_norm - wf_norm.mean()))
        # Exclude DC (index 0)
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

    # ── Internal push ────────────────────────────────────────────────

    def _push(self, frame, rssi=None, seq=None):
        now = time.perf_counter()
        amp    = np.abs(frame)
        active = amp[amp > 0.0]
        val    = float(active.mean()) if active.size > 0 else 0.0

        # Normalised full subcarrier array (padded / trimmed to MAX_SC)
        n = min(len(amp), MAX_SC)
        sc = np.zeros(MAX_SC, dtype=np.float32)
        sc[:n] = amp[:n]
        mx = sc.max() or 1.0
        sc_norm = sc / mx

        with self._lock:
            self._raw.append(val)
            self._sc_mags[:] = sc_norm
            self._frames += 1

            if rssi is not None:
                self._rssi = float(rssi)

            # Packet-loss detection via sequence number gaps
            if seq is not None and self._last_seq >= 0:
                gap = (seq - self._last_seq - 1) % 4096
                if 0 < gap < 200:
                    self._pkt_loss += gap
            if seq is not None:
                self._last_seq = seq

            # Inter-frame latency
            dt = now - self._t_last
            self._t_last = now
            self._latencies.append(dt)

    # ── Run ──────────────────────────────────────────────────────────

    def run(self):
        if self.demo:
            self._run_demo(); return
        try:
            ser = serial.Serial(self.port, BAUD, timeout=0.5)
            if os.name == "nt" and hasattr(ser, "set_buffer_size"):
                ser.set_buffer_size(rx_size=2_000_000)
            ser.reset_input_buffer()
            print(f"✅  {self.port} @ {BAUD}")
        except Exception as e:
            print(f"❌  {e}  →  demo mode")
            self._run_demo(); return
        try:
            while not self.stop_event.is_set():
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if "CSI_DATA" not in line:
                    continue
                rssi, seq = self._extract_meta(line)
                frame = (parse_csi_line(line) if _PARSER_OK
                         else self._fallback(line))
                if frame is not None:
                    self._push(frame, rssi=rssi, seq=seq)
        finally:
            if ser.is_open:
                ser.close()

    @staticmethod
    def _extract_meta(line):
        """Extract RSSI and sequence number from Magic Header CSV line."""
        try:
            parts = line.split(",")
            # CSI_DATA,seq,mac,rssi,...
            seq  = int(parts[1].strip())  if len(parts) > 1 else None
            rssi = float(parts[3].strip()) if len(parts) > 3 else None
            return rssi, seq
        except (ValueError, IndexError):
            return None, None

    @staticmethod
    def _fallback(line):
        try:
            ds = line.split("[")[-1].split("]")[0].replace('"', '').strip()
            vs = [int(v.strip()) for v in ds.split(",") if v.strip()]
            if len(vs) < 2:
                return None
            return np.array([complex(vs[i + 1], vs[i])
                             for i in range(0, len(vs) - 1, 2)],
                            dtype=np.complex64)
        except Exception:
            return None

    def _run_demo(self):
        """Synthetic data: quiet baseline with periodic motion bursts."""
        rng   = np.random.default_rng(42)
        ph    = 0.0
        seq   = 0
        print("🎮  Demo mode")
        while not self.stop_event.is_set():
            time.sleep(0.01)
            ph    += 0.18
            burst  = 1.0 + 2.8 * max(0.0, math.sin(ph * 0.07) ** 8)
            val    = (abs(math.sin(ph) * 0.65 + math.sin(ph * 0.37) * 0.35)
                      * float(rng.uniform(0.88, 1.0)) * burst)

            # Fake full subcarrier array (64 subcarriers, varying shape)
            sc_arr = (np.abs(np.sin(np.linspace(0, math.pi, 64) + ph * 0.3))
                      * val * rng.uniform(0.7, 1.3, 64)).astype(np.float32)
            frame  = sc_arr.astype(np.complex64)  # treat as magnitudes

            rssi = -55.0 + 10.0 * math.sin(ph * 0.05) + rng.normal(0, 1.5)
            self._push(frame, rssi=rssi, seq=seq % 4096)
            seq += 1


# ════════════════════════════════════════════════════════════════════════
# HELPERS — pyqtgraph plot factories
# ════════════════════════════════════════════════════════════════════════

def _waveform_plot():
    """PlotWidget for the main waveform with labelled Y axis."""
    pw = pg.PlotWidget(background=SURFACE)
    pw.setMenuEnabled(False)
    pw.setMouseEnabled(x=False, y=False)

    # Show left (Y) axis with numeric labels
    ax_l = pw.getAxis("left")
    ax_l.setStyle(tickLength=-6, tickTextOffset=4)
    ax_l.setTextPen(pg.mkPen(TEXT_MID))
    ax_l.setPen(pg.mkPen(BORDER))
    ax_l.setTicks([[(v, f"{v:.1f}") for v in np.linspace(0, 1.5, 4)]])

    # Bottom axis — frame index
    ax_b = pw.getAxis("bottom")
    ax_b.setStyle(tickLength=-4, tickTextOffset=3)
    ax_b.setTextPen(pg.mkPen(TEXT_DIM))
    ax_b.setPen(pg.mkPen(BORDER))
    ax_b.setTickFont(QtGui.QFont("Courier New", 8))

    # Setting Y-range to 1.5 puts 1.0 comfortably in the upper-middle
    pw.setYRange(0.0, 1.5, padding=0)
    pw.setXRange(0, WAVEFORM_LEN - 1, padding=0.02)

    # Grid
    for yv in np.linspace(0, 1.5, 4):
        pw.addItem(pg.InfiniteLine(pos=yv, angle=0,
                                   pen=pg.mkPen(GRID_CLR, width=1)))
    for xv in np.linspace(0, WAVEFORM_LEN - 1, 9):
        pw.addItem(pg.InfiniteLine(pos=xv, angle=90,
                                   pen=pg.mkPen(GRID_CLR, width=1)))
    return pw


def _sc_plot():
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

    # X axis — subcarrier index
    ax_b = pw.getAxis("bottom")
    ax_b.setStyle(tickLength=-4, tickTextOffset=3)
    ax_b.setTextPen(pg.mkPen(TEXT_DIM))
    ax_b.setPen(pg.mkPen(BORDER))
    ax_b.setTickFont(QtGui.QFont("Courier New", 7))
    ticks = [(i, str(i)) for i in range(0, MAX_SC + 1, MAX_SC // 4)]
    ax_b.setTicks([ticks])

    pw.setYRange(0.0, 1.05, padding=0)
    pw.setXRange(-0.5, MAX_SC - 0.5, padding=0.01)

    # Horizontal reference
    pw.addItem(pg.InfiniteLine(pos=0.5, angle=0,
                               pen=pg.mkPen(GRID_CLR, width=1,
                                            style=Qt.DashLine)))
    return pw


# ════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ════════════════════════════════════════════════════════════════════════

class WaveformMonitor(QWidget):

    def __init__(self, reader: ReaderThread, port: str):
        super().__init__()
        self.reader   = reader
        self._t_color = 0.0
        self.setWindowTitle(f"CSI Monitor  ·  {port}")
        self.resize(1400, 680)
        self.setStyleSheet(QSS)
        pg.setConfigOptions(antialias=True)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ═══════════════════════════════════════════════════════════════
        # TOP ROW: waveform (left)  +  stat cards column (right)
        # ═══════════════════════════════════════════════════════════════
        main_row = QHBoxLayout()
        main_row.setSpacing(8)

        # ── Waveform ──────────────────────────────────────────────────
        self._pw_wf = _waveform_plot()
        main_row.addWidget(self._pw_wf, stretch=3)

        # ── Stats cards — stacked vertically on the right ─────────────
        cards_col = QVBoxLayout()
        cards_col.setSpacing(6)

        self._v_rssi   = self._make_card(cards_col, "RSSI",      "dBm")
        self._v_loss   = self._make_card(cards_col, "PKT  LOSS", "frames")
        self._v_var    = self._make_card(cards_col, "VARIANCE",  "")
        self._v_lat    = self._make_card(cards_col, "LATENCY",   "ms")
        self._v_freq   = self._make_card(cards_col, "INST FREQ", "Hz")
        self._v_frames = self._make_card(cards_col, "FRAMES",    "")

        main_row.addLayout(cards_col, stretch=1)
        root.addLayout(main_row, stretch=3)

        # ═══════════════════════════════════════════════════════════════
        # BOTTOM: Subcarrier Power Distribution — full width
        # ═══════════════════════════════════════════════════════════════
        sc_wrap = QFrame(); sc_wrap.setObjectName("panel")
        sc_outer = QVBoxLayout(sc_wrap)
        sc_outer.setContentsMargins(8, 6, 8, 6)
        sc_outer.setSpacing(4)

        sc_title = QLabel("SUBCARRIER  POWER  DISTRIBUTION  ·  all active SC")
        sc_title.setObjectName("stat_key")
        sc_outer.addWidget(sc_title)

        self._pw_sc = _sc_plot()
        sc_outer.addWidget(self._pw_sc)

        root.addWidget(sc_wrap, stretch=1)

        # ── Waveform plot items ───────────────────────────────────────
        x  = np.arange(WAVEFORM_LEN, dtype=float)
        y0 = np.zeros(WAVEFORM_LEN)

        self._g3   = self._pw_wf.plot(x, y0, pen=pg.mkPen((*CLR_CALM, 10), width=44))
        self._g2   = self._pw_wf.plot(x, y0, pen=pg.mkPen((*CLR_CALM, 28), width=18))
        self._g1   = self._pw_wf.plot(x, y0, pen=pg.mkPen((*CLR_CALM, 60), width=7))

        _z = pg.PlotDataItem(x, y0, pen=None)
        _w = pg.PlotDataItem(x, y0, pen=None)
        self._fill = pg.FillBetweenItem(_z, _w, brush=pg.mkBrush(*CLR_CALM, 35))
        self._pw_wf.addItem(self._fill)

        self._line = self._pw_wf.plot(x, y0,
                                      pen=pg.mkPen((*CLR_CALM, 255), width=2.0))
        self._x  = x
        self._y0 = y0

        # ── Subcarrier bar chart items ────────────────────────────────
        sc_x    = np.arange(MAX_SC, dtype=float)
        brushes = [pg.mkBrush(*CLR_CALM, 180)] * MAX_SC
        self._bars = pg.BarGraphItem(
            x=sc_x, height=np.zeros(MAX_SC, dtype=float),
            width=0.85, brushes=brushes, pen=pg.mkPen(None),
        )
        self._pw_sc.addItem(self._bars)

        # ── Timer ─────────────────────────────────────────────────────
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(REFRESH_MS)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _make_card(col_layout: QVBoxLayout, key: str, unit: str) -> QLabel:
        """
        Add a stat card to a vertical column layout.
        Returns the value QLabel for later updates.
        """
        frame = QFrame(); frame.setObjectName("panel")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        inner = QVBoxLayout(frame)
        inner.setContentsMargins(12, 8, 12, 8)
        inner.setSpacing(2)
        inner.setAlignment(Qt.AlignCenter)

        k_lbl = QLabel(key)
        k_lbl.setObjectName("stat_key")
        k_lbl.setAlignment(Qt.AlignCenter)

        v_lbl = QLabel("—")
        v_lbl.setObjectName("stat_val")
        v_lbl.setAlignment(Qt.AlignCenter)

        u_lbl = QLabel(unit)
        u_lbl.setObjectName("stat_unit")
        u_lbl.setAlignment(Qt.AlignCenter)

        inner.addWidget(k_lbl)
        inner.addWidget(v_lbl)
        inner.addWidget(u_lbl)
        col_layout.addWidget(frame)
        return v_lbl

    @staticmethod
    def _color_stat(label: QLabel, text: str, color: str = TEXT_HI):
        label.setText(text)
        label.setStyleSheet(
            f"font-family:'Courier New'; font-size:18px; "
            f"font-weight:bold; color:{color};"
        )

    # ── Refresh ───────────────────────────────────────────────────────

    def _refresh(self):
        d = self.reader.snapshot()
        wf      = d["wf"]
        sc_mags = d["sc_mags"]
        energy  = d["energy"]
        rssi    = d["rssi"]
        loss    = d["pkt_loss"]
        lat_ms  = d["lat_ms"]
        var     = d["variance"]
        freq_hz = d["freq_hz"]
        fc      = d["frames"]

        # ── Color parameter (smoothed) ────────────────────────────────
        t_target       = min(1.0, energy / (MOTION_THRESHOLD * 1.5))
        self._t_color += COLOR_SMOOTH * (t_target - self._t_color)
        t = self._t_color
        r, g, b = tri_lerp(t)

        # ── Waveform glow layers ──────────────────────────────────────
        self._g3.setData(self._x, wf)
        self._g3.setPen(pg.mkPen((r, g, b, int( 6 + t * 22)), width=44))
        self._g2.setData(self._x, wf)
        self._g2.setPen(pg.mkPen((r, g, b, int(22 + t * 55)), width=18))
        self._g1.setData(self._x, wf)
        self._g1.setPen(pg.mkPen((r, g, b, int(55 + t * 90)), width=7))

        self._fill.setCurves(
            pg.PlotDataItem(self._x, self._y0, pen=None),
            pg.PlotDataItem(self._x, wf,       pen=None),
        )
        self._fill.setBrush(pg.mkBrush(r, g, b, int(30 + t * 85)))

        self._line.setData(self._x, wf)
        self._line.setPen(pg.mkPen((r, g, b, 255), width=2.0))

        # ── Subcarrier power bars (per-bar color) ─────────────────────
        brushes = [pg.mkBrush(*sc_color(float(v)), 200) for v in sc_mags]
        self._bars.setOpts(height=sc_mags.astype(float), brushes=brushes)

        # ── Stats ─────────────────────────────────────────────────────
        # RSSI
        rssi_color = (ACCENT if rssi > -65
                      else "#ff8c14" if rssi > -80
                      else "#f85149")
        self._color_stat(self._v_rssi, f"{rssi:.0f}", rssi_color)

        # Packet loss
        loss_color = (ACCENT if loss == 0 else
                      "#ff8c14" if loss < 50 else "#f85149")
        self._color_stat(self._v_loss, str(loss), loss_color)

        # Variance
        var_color = (ACCENT if var < 0.01 else
                     "#ff8c14" if var < 0.05 else "#f85149")
        self._color_stat(self._v_var, f"{var:.4f}", var_color)

        # Latency
        lat_color = (ACCENT if lat_ms < 15 else
                     "#ff8c14" if lat_ms < 40 else "#f85149")
        self._color_stat(self._v_lat, f"{lat_ms:.1f}", lat_color)

        # Instantaneous frequency
        self._color_stat(self._v_freq, f"{freq_hz:.2f}", TEXT_HI)

        # Frames
        self._color_stat(self._v_frames, str(fc), TEXT_MID)


# ════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM6" if os.name == "nt" else "/dev/ttyUSB0")
    p.add_argument("--demo", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    app  = QApplication(sys.argv)

    stop   = threading.Event()
    reader = ReaderThread(port=args.port, demo=args.demo, stop_event=stop)
    reader.start()

    win = WaveformMonitor(reader=reader, port=args.port)
    
    # Center the window on the computer screen
    screen_rect = app.primaryScreen().availableGeometry()
    x = (screen_rect.width() - win.width()) // 2
    y = (screen_rect.height() - win.height()) // 2
    win.move(x, y)
    
    win.show()

    code = app.exec_()
    stop.set()
    reader.join(timeout=2.0)
    sys.exit(code)


if __name__ == "__main__":
    main()