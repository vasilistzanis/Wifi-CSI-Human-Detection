"""Reusable custom widgets and shared dashboard UI helpers."""

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

import pyqtgraph as pg

from .dashboard_theme import (
    _BG, _BORDER, _BLUE, _DARK2, _DIM, _GREEN, _NAV_OFF, _NAV_ON,
    _ORANGE, _PANEL, _RED, _TEXT, _YELLOW,
)


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
        # Cached fonts — created once, reused every paintEvent
        self._font_lbl = QFont("Segoe UI", 26, QFont.Bold)
        self._font_sub = QFont("Segoe UI", 10)
        self._col_dim  = QColor(_DIM)

    def set(self, label: str, sub: str, color: str):
        lbl = label.upper()
        if lbl == self._label and sub == self._sub and color == self._color:
            return  # nothing changed — skip repaint entirely
        self._label = lbl
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
        p.setBrush(c)
        p.drawRoundedRect(0, 12, 4, r.height() - 24, 2.0, 2.0)

        # Large activity label
        p.setPen(c)
        p.setFont(self._font_lbl)
        lbl_r = QRectF(16, 6, r.width() - 20, r.height() - 30)
        p.drawText(lbl_r, Qt.AlignLeft | Qt.AlignVCenter, self._label)

        # Sub-text
        if self._sub:
            p.setPen(self._col_dim)
            p.setFont(self._font_sub)
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
        # Cached objects — avoid allocating on every paintEvent
        sz = size - 20
        self._font_gauge = QFont("Segoe UI", max(8, sz // 5), QFont.Bold)
        self._pen_track  = QPen(QColor(_DARK2), 7, Qt.SolidLine, Qt.RoundCap)
        self._pen_value  = QPen(QColor(_DIM),   7, Qt.SolidLine, Qt.RoundCap)
        self._col_text   = QColor(_TEXT)

    def set(self, pct: float, color: str):
        new_pct = max(0.0, min(100.0, pct))
        # Only trigger repaint when integer % or color changes
        if int(new_pct) == int(self._pct) and color == self._color:
            return
        self._pct   = new_pct
        if color != self._color:
            self._color    = color
            self._pen_value = QPen(QColor(color), 7, Qt.SolidLine, Qt.RoundCap)
        self.update()

    def paintEvent(self, event):
        p   = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        m   = 10
        sz  = min(self.width(), self.height()) - 2 * m
        arc = QRectF(m, m, sz, sz)

        # Track
        p.setPen(self._pen_track)
        p.drawArc(arc, 225 * 16, -270 * 16)

        # Value arc — use pre-built pen (rebuilt only when color changes)
        if self._pct > 0:
            p.setPen(self._pen_value)
            p.drawArc(arc, 225 * 16, int(-270 * 16 * self._pct / 100))

        # Center text
        p.setPen(self._col_text)
        p.setFont(self._font_gauge)
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
        self._last_fps_int: object = -1   # int when live, "conn" sentinel otherwise
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
        live = frames > 10
        # Only run the heavy setStyleSheet path when live state actually flips.
        if live != self._live:
            self._live = live
            if live:
                self._blink.stop()
                self._dot.setVisible(True)
                self._dot.setStyleSheet(f"font-size:10px; color:{_GREEN};")
                self._text.setStyleSheet(f"font-size:11px; color:{_GREEN};")
                self._set_style("live")
            else:
                if not self._blink.isActive():
                    self._blink.start(600)
                self._dot.setStyleSheet(f"font-size:10px; color:{_YELLOW};")
                self._text.setStyleSheet(f"font-size:11px; color:{_YELLOW};")
                self._set_style("connecting")
        # Text only on integer-fps change — the f-string + setText was running
        # 50×/sec for sub-1-fps fluctuations the user can't read anyway.
        if live:
            new_fps = int(fps)
            if new_fps != self._last_fps_int:
                self._last_fps_int = new_fps
                self._text.setText(f"System Live · {new_fps} fr/s")
        elif self._last_fps_int != "conn":
            self._last_fps_int = "conn"
            self._text.setText("Connecting…")

class MotionIntensityBar(QWidget):
    """Horizontal gradient bar showing real-time motion intensity (0–100%).
    In Super Motion mode an extra label row (Calm/Low/Medium/High) appears below the bar."""
    def __init__(self):
        super().__init__()
        self._intensity = 0.0
        self._overlay   = ""
        self.setFixedHeight(48)
        # Cache fonts — avoid allocating new QFont objects on every paintEvent
        self._font_lbl = QFont("Segoe UI", 8);  self._font_lbl.setBold(True)
        self._font_val = QFont("Segoe UI", 10); self._font_val.setBold(True)
        self._font_ovl = QFont("Segoe UI", 9);  self._font_ovl.setBold(True)
        # Cache gradient colors
        self._col_green  = QColor(_GREEN)
        self._col_yellow = QColor(_YELLOW)
        self._col_orange = QColor(_ORANGE)
        self._col_red    = QColor(_RED)
        self._col_dark2  = QColor(_DARK2)
        self._col_dim    = QColor(_DIM)
        self._col_text   = QColor(_TEXT)

    def set(self, intensity: float, overlay: str = ""):
        new_i = max(0.0, min(1.0, intensity))
        # Only repaint when integer percent or overlay text changes —
        # sub-percent fluctuations are invisible on a 200px-wide bar.
        if int(new_i * 100) == int(self._intensity * 100) and overlay == self._overlay:
            self._intensity = new_i
            return
        self._intensity = new_i
        self._overlay   = overlay
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect()

        # Label + percentage
        p.setPen(self._col_dim)
        p.setFont(self._font_lbl)
        p.drawText(QRectF(0, 0, r.width() * 0.7, 16),
                   Qt.AlignLeft | Qt.AlignVCenter, "MOTION INTENSITY")
        p.setPen(self._col_text)
        p.setFont(self._font_val)
        p.drawText(QRectF(r.width() * 0.7, 0, r.width() * 0.3, 16),
                   Qt.AlignRight | Qt.AlignVCenter, f"{int(self._intensity * 100)}%")

        # Track
        p.setPen(Qt.NoPen)
        p.setBrush(self._col_dark2)
        p.drawRoundedRect(QRectF(0, 20, r.width(), 10), 5.0, 5.0)

        # Filled gradient bar
        if self._intensity > 0.005:
            fill_w = max(10.0, self._intensity * r.width())
            g = QLinearGradient(0.0, 0.0, float(r.width()), 0.0)
            g.setColorAt(0.00, self._col_green)
            g.setColorAt(0.45, self._col_yellow)
            g.setColorAt(0.75, self._col_orange)
            g.setColorAt(1.00, self._col_red)
            p.setBrush(QBrush(g))
            p.drawRoundedRect(QRectF(0, 20, fill_w, 10), 5.0, 5.0)

        # Super Motion label below bar (Calm / Low / Medium / High)
        if self._overlay:
            if self._intensity < 0.25:   oc = self._col_green
            elif self._intensity < 0.50: oc = self._col_yellow
            elif self._intensity < 0.75: oc = self._col_orange
            else:                        oc = self._col_red
            p.setPen(oc)
            p.setFont(self._font_ovl)
            p.drawText(QRectF(0, 33, r.width(), 15),
                       Qt.AlignRight | Qt.AlignVCenter, self._overlay)

        p.end()

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
    # ─── Performance tweaks (no visual change) ───
    # auto-downsample to the number of visible pixels using peak-preserving
    # algorithm — visually identical for our densities, much cheaper to draw.
    pi = pw.getPlotItem()
    pi.setDownsampling(auto=True, mode="peak")
    pi.setClipToView(True)
    pi.getViewBox().setDefaultPadding(0.0)
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
        # Cache last values — at 50Hz the typical case is "no change",
        # in which case we skip both the f-string alloc and the setText round-trip.
        last = getattr(self, "_last_health", (-1, -1))
        if last == (buf_pct, load_pct):
            return
        self._last_health = (buf_pct, load_pct)
        for (bar, pct_lbl), val, prev in zip(
            self._bar_refs, (buf_pct, load_pct), last
        ):
            if val != prev:
                bar.setValue(val)
                pct_lbl.setText(f"{val}%")
