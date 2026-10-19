"""Signal, activity-log, and system-information dashboard pages."""

import math
import time

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QPushButton, QProgressBar,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import config

from .dashboard_theme import (
    _BLUE, _BORDER, _DARK2, _DIM, _GREEN, _MODEL_NAMES, _PANEL, _RED,
    _TEXT, _YELLOW, _cc, _disp,
)
from .dashboard_widgets import _card, _hdr, _make_pw, _stat_chip


class SignalViewPage(QWidget):
    def __init__(self, waveform_len: int):
        super().__init__()
        self.waveform_len = waveform_len
        self._wave_scale  = 1.0
        self._wave_scale_applied = 0.0
        self._sc_x        = None   # cached subcarrier x-axis (like MonitorPage)
        # Cached chip texts — skip setText round-trips when value unchanged.
        self._last_fps_txt = self._last_lat_txt = ""
        self._last_loss_txt = self._last_sig_txt = self._last_var_txt = ""
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

    def update_curves(self, snap: dict):
        """Light per-tick update — only the live plots (see MonitorPage)."""
        wave = snap.get("wave")
        if wave is not None:
            mx = float(wave.max())
            self._wave_scale = max(mx * 1.05, self._wave_scale * 0.998)
            new_scale = max(self._wave_scale, 1e-6)
            if abs(new_scale - self._wave_scale_applied) / max(self._wave_scale_applied, 1e-6) >= 0.015:
                self._pw_wave.setYRange(0, new_scale, padding=0.02)
                self._wave_scale_applied = new_scale
            self._curve_wave.setData(self._wave_x, wave)

        n = snap.get("n", 1); amp = snap.get("last_amp")
        if amp is not None:
            if self._sc_x is None or len(self._sc_x) != n:
                self._sc_x = np.arange(n, dtype=np.float32)
                self._pw_sc.setXRange(-0.5, n - 0.5, padding=0)
                self._sc_bars.setOpts(x=self._sc_x, height=amp, width=0.8)
            else:
                self._sc_bars.setOpts(height=amp)

    def update(self, state: dict):
        fps      = state.get("fps", 0.0)
        lat      = state.get("latency_ms", 0.0)
        var      = state.get("variance", 0.0)
        mean_a   = state.get("mean_amp", 0.0)
        expected = config.SAMPLING_RATE
        loss_pct = max(0.0, (1.0 - fps / expected) * 100) if fps > 0 else 0.0
        sig_db   = 20 * math.log10(max(mean_a, 1e-9))

        fps_txt  = f"{fps:.1f} fps"
        lat_txt  = f"{lat:.1f} ms" if lat > 0 else "—"
        loss_txt = f"{loss_pct:.0f}%"
        sig_txt  = f"{sig_db:.1f} dB"
        var_txt  = f"{var:.4f}"
        if fps_txt  != self._last_fps_txt:  self._last_fps_txt  = fps_txt;  self._sv_fps.setText(fps_txt)
        if lat_txt  != self._last_lat_txt:  self._last_lat_txt  = lat_txt;  self._sv_latency.setText(lat_txt)
        if loss_txt != self._last_loss_txt: self._last_loss_txt = loss_txt; self._sv_loss.setText(loss_txt)
        if sig_txt  != self._last_sig_txt:  self._last_sig_txt  = sig_txt;  self._sv_sig.setText(sig_txt)
        if var_txt  != self._last_var_txt:  self._last_var_txt  = var_txt;  self._sv_var.setText(var_txt)

class ActivityLogPage(QWidget):
    def __init__(self, classes: list, on_clear):
        super().__init__()
        self.classes          = classes
        self._x_dist          = np.arange(len(classes), dtype=np.float32)
        self._session_start   = time.monotonic()
        self._last_table_upd  = 0.0
        self._last_table_len  = -1   # guard: skip rebuild if log didn't change
        self._last_avgc_color = None
        self._last_dom_label  = None
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
        # Same perf flags as _make_pw — appearance unchanged.
        _dist_pi = self._pw_dist.getPlotItem()
        _dist_pi.setDownsampling(auto=True, mode="peak")
        _dist_pi.setClipToView(True)
        _dist_pi.getViewBox().setDefaultPadding(0.0)
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

    def set_mode(self, new_classes: list):
        """Switch the distribution chart to a new set of classes (e.g. motion levels)."""
        self.classes = new_classes
        n = len(new_classes)
        self._x_dist = np.arange(n, dtype=np.float32)
        self._pw_dist.removeItem(self._dist_bars)
        for ti in self._pct_labels:
            self._pw_dist.removeItem(ti)
        ticks = [[(i, c[:6]) for i, c in enumerate(new_classes)]]
        self._pw_dist.getAxis("bottom").setTicks(ticks)
        brushes = [pg.mkBrush(_cc(c) + "cc") for c in new_classes]
        self._dist_bars = pg.BarGraphItem(
            x=self._x_dist, height=np.zeros(n), width=0.6, brushes=brushes,
        )
        self._pw_dist.addItem(self._dist_bars)
        self._pct_labels = []
        for cls in new_classes:
            ti = pg.TextItem("", anchor=(0.5, 1.0), color=_DIM)
            ti.setFont(QFont("Segoe UI", 8))
            self._pw_dist.addItem(ti)
            self._pct_labels.append(ti)
        self._pw_dist.setYRange(0, 1, padding=0.22)
        self._pw_dist.setVisible(False)
        self._no_dist.setVisible(True)

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
        if c_avg != self._last_avgc_color:
            self._last_avgc_color = c_avg
            self._v_avgc.setStyleSheet(
                f"font-size:14px; font-weight:bold; color:{c_avg};"
                f" border:none; background:transparent;"
            )
        if total > 0 and counts:
            dom = max(counts, key=counts.get)
            self._v_dom.setText(_disp(dom).upper())
            if dom != self._last_dom_label:
                self._last_dom_label = dom
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

        # Log table — refresh at most once per second, only if log changed
        n_log = len(log)
        show  = n_log > 0
        self._table.setVisible(show); self._no_log.setVisible(not show)
        now = time.monotonic()
        if show and (now - self._last_table_upd) >= 1.0 and n_log != self._last_table_len:
            self._last_table_upd = now
            self._last_table_len = n_log
            self._table.setRowCount(n_log)
            for r, (ts, lbl, cf, fr) in enumerate(log):
                c = _cc(lbl)
                items = [QTableWidgetItem(ts), QTableWidgetItem(_disp(lbl).upper()),
                         QTableWidgetItem(f"{cf:.1f}%"), QTableWidgetItem(str(fr))]
                items[1].setForeground(pg.mkColor(c))
                for col, item in enumerate(items):
                    item.setTextAlignment(Qt.AlignCenter)
                    self._table.setItem(r, col, item)
                    self._table.setRowHeight(r, 22)

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
        # Guards: only call setStyleSheet when the color category changes
        self._last_conn_color = None
        self._last_fps_color  = None
        self._last_buf_color  = None
        self._last_lat_color  = None
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

        # Connection status — only update stylesheet when color changes
        if dot_color != self._last_conn_color:
            self._last_conn_color = dot_color
            self._hc.setStyleSheet(
                f"QWidget{{background:{_PANEL}; border-radius:8px;"
                f" border-left:3px solid {dot_color}; border-top:1px solid {_BORDER};"
                f" border-right:1px solid {_BORDER}; border-bottom:1px solid {_BORDER};}}"
            )
            self._sys_dot.setStyleSheet(
                f"font-size:14px; color:{dot_color}; background:transparent; border:none;"
            )
            self._sys_lbl.setStyleSheet(
                f"font-size:16px; font-weight:bold; color:{dot_color};"
                f" letter-spacing:0.06em; background:transparent; border:none;"
            )
            if self._conn_status_lbl:
                self._conn_status_lbl.setStyleSheet(self._metric_style(dot_color))
        self._sys_lbl.setText("LIVE" if connected else "OFFLINE")
        if self._conn_status_lbl:
            self._conn_status_lbl.setText("LIVE" if connected else "LOST")
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

        # FPS — only setStyleSheet when color category changes
        fps = state.get("fps", 0.0)
        fps_color = _GREEN if fps >= 80 else (_YELLOW if fps >= 40 else _RED)
        eff_fps_color = fps_color if fps > 0 else _DIM
        self._si_fps.setText(f"{fps:.1f} fps" if fps > 0 else "—")
        if eff_fps_color != self._last_fps_color:
            self._last_fps_color = eff_fps_color
            self._si_fps.setStyleSheet(self._metric_style(eff_fps_color))

        # Buffer fill — only setStyleSheet when color category changes
        buf = int(state.get("infer_fill", 0) * 100)
        buf_color = _GREEN if buf >= 80 else (_YELLOW if buf >= 40 else _BLUE)
        self._si_buf.setText(f"{buf}%")
        if buf_color != self._last_buf_color:
            self._last_buf_color = buf_color
            self._si_buf.setStyleSheet(self._metric_style(buf_color))
            self._buf_bar.setStyleSheet(
                f"QProgressBar{{background:{_DARK2}; border-radius:2px; border:none;}}"
                f"QProgressBar::chunk{{background:{buf_color}; border-radius:2px;}}"
            )
            self._buf_pct_lbl.setStyleSheet(
                f"font-size:10px; color:{buf_color}; background:transparent; border:none;"
            )
        self._buf_bar.setValue(buf)
        self._buf_pct_lbl.setText(f"{buf}%")

        # Inference latency — only setStyleSheet when color category changes
        lat = state.get("latency_ms", 0.0)
        lat_s = f"{lat:.1f} ms" if lat > 0 else "—"
        lat_color = _GREEN if lat <= 5 else (_YELLOW if lat <= 30 else _RED)
        eff_lat_color = lat_color if lat > 0 else _DIM
        self._si_lat.setText(lat_s)
        if eff_lat_color != self._last_lat_color:
            self._last_lat_color = eff_lat_color
            self._si_lat.setStyleSheet(self._metric_style(eff_lat_color))
        if self._infer_latency_lbl:
            self._infer_latency_lbl.setText(lat_s)
