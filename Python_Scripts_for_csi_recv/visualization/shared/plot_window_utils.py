#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single point of responsibility for ALL window management.

This module:
  - Selects the matplotlib backend (Qt5Agg → TkAgg fallback)
  - Disables interactive mode (plt.ioff)
  - Centers and shows every matplotlib figure
  - Centers every live Qt window

Scripts must NOT call plt.show(), plt.ioff(), or matplotlib.use() directly.
Scripts must import from here BEFORE importing matplotlib.pyplot.
"""

import matplotlib
try:
    matplotlib.use("Qt5Agg")
except Exception:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

import matplotlib.pyplot as plt
plt.ioff()

import config

__all__ = ["setup_matplotlib", "show_figure", "show_all", "center_qt_window"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _screen_rect():
    """Return (x, y, width, height) of the primary screen's available area."""
    try:
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        r = app.primaryScreen().availableGeometry()
        return r.x(), r.y(), r.width(), r.height()
    except Exception:
        return 0, 0, 1920, 1080


def _center_window(window, w_px: int, h_px: int) -> None:
    """Resize and move any Qt window to the screen center."""
    sx, sy, sw, sh = _screen_rect()
    # Never exceed the available screen area
    w_px = min(w_px, sw)
    h_px = min(h_px, sh)
    x = sx + max(0, (sw - w_px) // 2)
    y = sy + max(0, (sh - h_px) // 2)
    def _apply():
        try:
            window.setGeometry(x, y, w_px, h_px)
        except Exception:
            pass

    try:
        _apply()  # pre-show hint
        from PyQt5.QtCore import QTimer
        # Windows WM sends WM_WINDOWPOSCHANGED asynchronously after show().
        # 0ms fires before that message arrives; 150ms fires after — overrides cascade.
        QTimer.singleShot(150, _apply)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_matplotlib() -> None:
    """Apply config figsize/DPI. Call once at the top of main()."""
    plt.rcParams["figure.figsize"] = list(config.FIGURE_SIZE)
    plt.rcParams["figure.dpi"] = config.FIGURE_DPI


def show_figure(fig) -> None:
    """Center and show a single matplotlib figure, blocking until closed."""
    show_all(figs=[fig])


def show_all(figs=None) -> None:
    """
    Center and show matplotlib figures, blocking until all are closed.

    Geometry is set BEFORE plt.show() so windows appear centered on first
    paint — no visible jump. The QTimer fallback in _center_window handles
    any residual WM cascade override after show().
    """
    if figs is None:
        nums = plt.get_fignums()
        if not nums:
            return
        figs = [plt.figure(n) for n in nums]
    if not figs:
        return

    try:
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
    except Exception:
        app = None

    # Center BEFORE show() — Qt window objects exist but are not yet visible,
    # so the WM receives our position as the initial placement hint.
    dpi = config.FIGURE_DPI
    for fig in figs:
        try:
            w_in, h_in = fig.get_size_inches()
            _center_window(fig.canvas.manager.window,
                           int(w_in * dpi), int(h_in * dpi))
        except Exception:
            pass

    plt.show(block=False)

    if app is not None:
        app.processEvents()
        app.exec_()
    else:
        plt.show()  # non-Qt backend fallback


def center_qt_window(widget, w: int = None, h: int = None) -> None:
    """
    Center and resize a live Qt window.

    Call BEFORE widget.show() so the window appears centered on first paint,
    bypassing the WM cascade placement algorithm.
    """
    _center_window(widget,
                   w if w is not None else config.QT_WINDOW_W,
                   h if h is not None else config.QT_WINDOW_H)
