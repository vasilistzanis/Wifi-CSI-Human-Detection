"""Live signal and inference monitor page."""

import time

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

import config

from .dashboard_theme import (
    _BLUE, _DIM, _GREEN, _MODEL_NAMES, _ORANGE, _RED, _TEXT, _YELLOW,
    _cc, _disp,
)
from .dashboard_widgets import (
    ActivityBlock, ConfidenceGauge, MotionIntensityBar, _card, _hdr,
    _make_pw, _stat_chip,
)


class MonitorPage(QWidget):
    def __init__(self, classes: list, waveform_len: int,
                 model_key: str, port: str):
        super().__init__()
        self.classes          = classes
        self.waveform_len     = waveform_len
        self._wave_scale      = 1.0
        self._wave_scale_applied = 0.0  # last value actually sent to setYRange
        self._current_mode    = "har"
        self._last_fps_level  = None   # guard: only setStyleSheet when fps category changes
        self._last_recent_upd = 0.0    # throttle: recent activity table max 1 Hz
        self._last_recent_head = None  # (frame, n_log) of newest entry — rebuild only on change
        self._sc_x            = None   # cached x-axis array for subcarrier bars
        # Cached label text — skip setText round-trips when value unchanged.
        self._last_conf_txt   = ""
        self._last_raw_txt    = ""
        self._last_fps_txt    = ""
        self._last_lq_sub_txt = ""
        self._last_loss_txt   = ""
        self._last_frames_txt = ""
        self._last_uptime_int = -1     # uptime only changes every 1s
        self._last_act_args   = (None, None, None)
        self._last_gauge_args = (None, None)
        self._build(model_key, port)

    def set_mode(self, mode: str):
        self._current_mode = mode

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

        # Motion intensity bar
        self._motion_bar = MotionIntensityBar()
        iv.addWidget(self._motion_bar)

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

    def update_curves(self, snap: dict):
        """Light per-tick update — only the live plots. Called every timer tick
        for smooth visual refresh; the heavy text/gauges/recent table go through
        update() at a lower cadence."""
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
        fc = state.get("frame_count", 0)
        # Banner visibility only flips once at startup — guard the Qt call.
        want_banner = (fc == 0)
        if want_banner != getattr(self, "_banner_visible", None):
            self._banner_visible = want_banner
            self._banner.setVisible(want_banner)

        label = state.get("label", "—")
        raw   = state.get("raw_label", "—")
        conf  = state.get("confidence", 0.0)
        intensity = state.get("motion_intensity", 0.0)

        if self._current_mode == "motion":
            # Model prediction OR signal intensity triggers MOTION display
            is_motion    = (label == "walk_activity") or (intensity > 0.10)
            display_conf = conf if label == "walk_activity" else intensity * 100.0
            if is_motion:
                act_args = ("MOTION", round(display_conf, 1), _ORANGE)
                if act_args != self._last_act_args:
                    self._last_act_args = act_args
                    self._activity_block.set("MOTION", f"Sensitivity: {display_conf:.1f}%", _ORANGE)
                gauge_args = (round(display_conf, 1), _ORANGE)
                if gauge_args != self._last_gauge_args:
                    self._last_gauge_args = gauge_args
                    self._gauge.set(display_conf, _ORANGE)
                conf_txt = f"Sensitivity: {display_conf:.1f}%"
                if conf_txt != self._last_conf_txt:
                    self._last_conf_txt = conf_txt
                    self._conf_lbl.setText(conf_txt)
                if self._last_raw_txt != "Movement detected":
                    self._last_raw_txt = "Movement detected"
                    self._raw_lbl.setText("Movement detected")
                self._motion_bar.set(intensity, "Motion")
            else:
                act_args = ("—", "", _DIM)
                if act_args != self._last_act_args:
                    self._last_act_args = act_args
                    self._activity_block.set("—", "", _DIM)
                gauge_args = (0.0, _DIM)
                if gauge_args != self._last_gauge_args:
                    self._last_gauge_args = gauge_args
                    self._gauge.set(0.0, _DIM)
                if self._last_conf_txt != "No motion":
                    self._last_conf_txt = "No motion"
                    self._conf_lbl.setText("No motion")
                if self._last_raw_txt != "Idle":
                    self._last_raw_txt = "Idle"
                    self._raw_lbl.setText("Idle")
                self._motion_bar.set(intensity)
        else:
            color = _cc(label)
            disp_label = _disp(label)
            disp_raw   = _disp(raw)
            act_args = (disp_label, disp_raw, color)
            if act_args != self._last_act_args:
                self._last_act_args = act_args
                self._activity_block.set(disp_label, f"Raw: {disp_raw}", color)
            gauge_args = (round(conf, 1), color)
            if gauge_args != self._last_gauge_args:
                self._last_gauge_args = gauge_args
                self._gauge.set(conf, color)
            conf_txt = f"Confidence: {conf:.1f}%"
            if conf_txt != self._last_conf_txt:
                self._last_conf_txt = conf_txt
                self._conf_lbl.setText(conf_txt)
            raw_txt = f"Raw: {disp_raw}"
            if raw_txt != self._last_raw_txt:
                self._last_raw_txt = raw_txt
                self._raw_lbl.setText(raw_txt)
            self._motion_bar.set(intensity)

        fps = state.get("fps", 0.0)
        fps_level = 0 if fps <= 0 else (3 if fps >= 80 else (2 if fps >= 50 else 1))
        if fps_level != self._last_fps_level:
            self._last_fps_level = fps_level
            if fps_level == 0:
                self._lq_lbl.setText("No Signal")
                self._lq_lbl.setStyleSheet(
                    f"font-size:13px; font-weight:bold; color:{_DIM};"
                    f" background:transparent; border:none;"
                )
            elif fps_level == 3:
                self._lq_lbl.setText("Excellent")
                self._lq_lbl.setStyleSheet(
                    f"font-size:13px; font-weight:bold; color:{_GREEN};"
                    f" background:transparent; border:none;"
                )
            elif fps_level == 2:
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
        lq_sub_txt = f"{fps:.0f} fps"
        if lq_sub_txt != self._last_lq_sub_txt:
            self._last_lq_sub_txt = lq_sub_txt
            self._lq_sub.setText(lq_sub_txt)

        expected = config.SAMPLING_RATE
        loss_pct = max(0.0, (1.0 - fps / expected) * 100) if fps > 0 else 0.0
        fps_txt  = f"{fps:.1f}"
        loss_txt = f"{loss_pct:.0f}%"
        if fps_txt != self._last_fps_txt:
            self._last_fps_txt = fps_txt
            self._fps_val.setText(fps_txt)
        if loss_txt != self._last_loss_txt:
            self._last_loss_txt = loss_txt
            self._loss_val.setText(loss_txt)

        # Uptime: only changes once per second — skip f-string entirely on
        # the other ~24 plot ticks per second.
        uptime_i = int(state.get("uptime_s", 0.0))
        if uptime_i != self._last_uptime_int:
            self._last_uptime_int = uptime_i
            h, rem = divmod(uptime_i, 3600); m, s = divmod(rem, 60)
            self._sv_uptime.setText(f"{h:02d}:{m:02d}:{s:02d}")
        frames_txt = f"{fc:,}"
        if frames_txt != self._last_frames_txt:
            self._last_frames_txt = frames_txt
            self._sv_frames.setText(frames_txt)

        log   = state.get("log_entries", [])
        n_log = len(log)
        show  = n_log > 0
        # Recent-table visibility flips at most twice per session — guard.
        if show != getattr(self, "_recent_visible", None):
            self._recent_visible = show
            self._no_recent.setVisible(not show)
            self._recent.setVisible(show)
        now = time.monotonic()
        if show and (now - self._last_recent_upd) >= 1.0:
            # Track newest entry by its frame number: catches new predictions
            # even after log reaches max_log capacity (where len stays constant).
            head = (log[0][3], n_log) if log else (None, 0)
            if head != self._last_recent_head:
                self._last_recent_upd  = now
                self._last_recent_head = head
                n_rows = min(6, n_log)
                self._recent.setRowCount(n_rows)
                for r, (ts, lbl, cf, _fr) in enumerate(log[:n_rows]):
                    c = _cc(lbl)
                    items = (QTableWidgetItem(ts),
                             QTableWidgetItem(_disp(lbl).upper()),
                             QTableWidgetItem(f"{cf:.0f}%"))
                    items[1].setForeground(pg.mkColor(c))
                    for col, item in enumerate(items):
                        item.setTextAlignment(Qt.AlignCenter)
                        self._recent.setItem(r, col, item)
                    self._recent.setRowHeight(r, 22)
