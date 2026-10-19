#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI launcher and runtime entry point for the live CSI dashboard."""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from csi_parser import configure_console_output
configure_console_output()
os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"
from PyQt5.QtWidgets import QApplication
import config
from visualization.shared.plot_window_utils import center_qt_window
from .desktop.dashboard_window import DashboardWindow
from .desktop.dashboard_monitor import MonitorPage
from .desktop.dashboard_pages import ActivityLogPage, SignalViewPage, SystemInfoPage
from .desktop.dashboard_settings import SettingsPage
from .desktop.dashboard_widgets import (
    ActivityBlock, ConfidenceGauge, MotionIntensityBar, Sidebar, StatusPill,
    _NAV_ITEMS, _card, _hdr, _make_pw, _stat_chip,
)
from .desktop.desktop_models import load_models as _load_models
from .desktop.desktop_reader import ReaderProcess, _make_shared_dtype, _reader_process_fn

__all__ = [
    "ActivityBlock",
    "ActivityLogPage",
    "ConfidenceGauge",
    "DashboardWindow",
    "MonitorPage",
    "MotionIntensityBar",
    "ReaderProcess",
    "SettingsPage",
    "SignalViewPage",
    "Sidebar",
    "StatusPill",
    "SystemInfoPage",
    "_NAV_ITEMS",
    "_card",
    "_hdr",
    "_load_models",
    "_make_pw",
    "_make_shared_dtype",
    "_parse_args",
    "_reader_process_fn",
    "_stat_chip",
    "main",
]

_INFERENCE_OK = False
_IMPORT_ERR   = ""
try:
    from data_preprocessing import CSIPipeline          # noqa: F401
    from csi_ml_pipeline import extract_features_from_window
    import joblib
    _INFERENCE_OK = True
except ImportError as _e:
    _IMPORT_ERR = str(_e)

# ARGUMENT PARSING + MAIN

def _parse_args():
    d = config.get_script_defaults("live_dashboard")
    p = argparse.ArgumentParser(
        description=(
            "CSI Live HAR Dashboard. Set CSI_SERVER_URL and CSI_EDGE_TOKEN "
            "to forward serial frames to the web API."
        )
    )
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
    # Tighten the GIL switch interval so the serial reader thread gets the GIL
    # within ~1ms instead of the default 5ms — critical at 100 Hz (10ms/frame).
    import sys as _sys
    _sys.setswitchinterval(0.001)

    # Set Windows multimedia timer resolution to 1ms.
    # The default is 15.625ms, which causes QTimer(20ms) to fire every ~30ms
    # and time.sleep(0.01) to sleep ~15ms — both directly cause frame drops.
    _winmm = None
    if os.name == "nt":
        try:
            import ctypes
            _winmm = ctypes.windll.winmm
            _winmm.timeBeginPeriod(1)
        except Exception:
            _winmm = None

    args = _parse_args()
    app  = QApplication(sys.argv)

    server_url = os.environ.get("CSI_SERVER_URL", "").strip()
    edge_token = os.environ.get("CSI_EDGE_TOKEN", "")
    api_enabled = bool(server_url and len(edge_token) >= 32 and not args.demo)
    if bool(server_url) != bool(edge_token):
        print(
            "[ERROR] API mirror disabled: set CSI_SERVER_URL and "
            "CSI_EDGE_TOKEN together."
        )
    elif server_url and len(edge_token) < 32:
        print("[ERROR] API mirror disabled: CSI_EDGE_TOKEN must be 32+ characters.")
    elif api_enabled:
        try:
            from csi_forwarding import validate_server_url
            validate_server_url(server_url)
        except (ImportError, ValueError) as exc:
            print(f"[ERROR] API mirror disabled: {exc}")
            api_enabled = False

    pipeline = le = model = None
    classes  = list(config.get_enabled_training_classes())

    if not args.demo and not api_enabled:
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

    reader = ReaderProcess(
        port=args.port, baud=args.baud, demo=args.demo,
        waveform_len=args.waveform_len, rx_buf=args.rx_buf,
        window_size=args.window, warmup=args.warmup, step=args.step,
        infer_enabled=not api_enabled,
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
        api_enabled=api_enabled,
    )
    if api_enabled:
        defaults = config.get_script_defaults("live_dashboard")
        launch_settings = {}
        for argument, setting in (
            ("model", "model"),
            ("window", "window_size"),
            ("step", "step_size"),
            ("warmup", "warmup_frames"),
            ("cutoff", "cutoff"),
            ("ema_alpha", "ema_alpha"),
            ("conf_thresh", "conf_thresh"),
            ("hyst_count", "hyst_count"),
        ):
            if getattr(args, argument) != defaults[argument]:
                launch_settings[setting] = getattr(args, argument)
        if launch_settings:
            win._sync_settings_with_api(launch_settings, "launch configuration")
    center_qt_window(win, w=1440, h=840)
    win.start_timer(args.refresh)
    win.show()

    code = app.exec_()
    # Stop reader first so it stops pushing into the inference queue, then
    # poison-pill the inference worker.
    reader.stop()
    win.stop_background_tasks()
    if win._infer_worker is not None:
        win._infer_worker.stop()
    if _winmm is not None:
        try:
            _winmm.timeEndPeriod(1)
        except Exception:
            pass
    sys.exit(code)

if __name__ == "__main__":
    from logging_setup import setup_run_log
    setup_run_log(__file__)
    main()
