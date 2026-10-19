"""Main window coordinating dashboard pages and runtime state."""

import math
import os
import queue
import threading
import time
from collections import Counter, deque

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtWidgets import QHBoxLayout, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

import config

from .dashboard_monitor import MonitorPage
from .dashboard_pages import ActivityLogPage, SignalViewPage, SystemInfoPage
from .dashboard_settings import SettingsPage
from .dashboard_theme import _BG, _BORDER, _MODEL_NAMES, _PANEL, _QSS, _TEXT
from .dashboard_widgets import Sidebar, StatusPill
from .desktop_inference import InferenceProcess
from .desktop_models import load_models
from .desktop_reader import ReaderProcess


class DashboardWindow(QMainWindow):
    def __init__(self, *, reader: "ReaderProcess",
                 pipeline, le, model, classes: list, model_key: str,
                 window_size: int, step: int, ema_alpha: float,
                 conf_thresh: float, cutoff: float, refresh_ms: int,
                 max_log: int, port: str, baud: int, demo: bool,
                 models_dir: str, hyst_count: int = config.DASHBOARD_HYST_COUNT,
                 api_enabled: bool = False):
        super().__init__()
        self.reader      = reader
        self.pipeline    = pipeline; self.le = le; self.model = model
        self.classes     = classes; self.model_key = model_key
        self.models_dir  = models_dir
        self.window_size = window_size; self.step = step; self.cutoff = cutoff
        self.ema_alpha   = ema_alpha; self.conf_thresh = conf_thresh
        self.max_log     = max_log; self.demo = demo
        self.api_enabled = api_enabled
        self._hyst_min   = hyst_count
        self.motion_threshold = 0.10

        self._current_mode    = "har"
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
        self._api_status_queue: queue.Queue = queue.Queue(maxsize=1)
        self._api_error_queue: queue.Queue = queue.Queue(maxsize=1)
        self._api_stop_event = threading.Event()
        self._api_thread: threading.Thread | None = None
        self._last_api_prediction = ""

        self._infer_worker = None
        if not self.api_enabled:
            self._infer_worker = InferenceProcess(
                in_q=reader.infer_in_q,
                pipeline=pipeline, model=model, le=le, classes=classes,
                window_size=window_size, cutoff=cutoff,
            )
            self._infer_worker.start()

        self.setWindowTitle("WiFi CSI Analyzer — Live Dashboard")
        self.setStyleSheet(_QSS)
        # Performance-tuned pyqtgraph globals (appearance unchanged).
        pg.setConfigOptions(
            antialias=False,
            enableExperimental=True,
            useOpenGL=False,
            useNumba=False,
            foreground=_TEXT,
            background=_PANEL,
        )
        # Opt-in OpenGL only if PyOpenGL is actually installed; otherwise
        # pyqtgraph would spam warnings at first paint.
        try:
            import OpenGL  # noqa: F401
            pg.setConfigOptions(useOpenGL=True)
        except ImportError:
            pass
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
            self.models_dir, self.model_key, self.switch_model,
            current_mode=self._current_mode,
            on_mode_change=self.switch_mode_dir,
            runtime_settings={
                "window_size": self.window_size,
                "step_size": self.step,
                "warmup_frames": config.FILTER_WARMUP,
                "cutoff": self.cutoff,
                "ema_alpha": self.ema_alpha,
                "conf_thresh": self.conf_thresh,
                "hyst_count": self._hyst_min,
                "motion_threshold": 0.10,
            },
            on_runtime_settings=self._apply_runtime_settings,
            api_enabled=self.api_enabled,
        )
        self._log_page = ActivityLogPage(self.classes, self._clear_log)
        self._pages = [
            self._monitor_page,
            SignalViewPage(wl),
            self._log_page,
            self._sysinfo_page,
            self._settings_page,
        ]
        for page in self._pages:
            self._stack.addWidget(page)

        right.addWidget(self._stack, 1)
        root.addLayout(right, 1)

    @staticmethod
    def _queue_latest(target: queue.Queue, value):
        try:
            target.put_nowait(value)
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            target.put_nowait(value)

    def _start_api_poller(self):
        if not self.api_enabled or self._api_thread is not None:
            return

        server_url = os.environ.get("CSI_SERVER_URL", "").strip()
        token = os.environ.get("CSI_EDGE_TOKEN", "")

        def _poll():
            from csi_forwarding import get_edge_status

            last_error = 0.0
            while not self._api_stop_event.is_set():
                try:
                    self._queue_latest(
                        self._api_status_queue,
                        get_edge_status(server_url, token),
                    )
                    last_error = 0.0
                except (
                    RuntimeError,
                    ValueError,
                    OSError,
                    TimeoutError,
                    ConnectionError,
                ) as exc:
                    now = time.monotonic()
                    if now - last_error >= 10.0:
                        message = f"status unavailable: {exc}"
                        print(f"[ERROR] API {message}")
                        self._queue_latest(self._api_error_queue, message)
                        last_error = now
                self._api_stop_event.wait(0.5)

        self._api_thread = threading.Thread(
            target=_poll,
            name="csi-api-status-poll",
            daemon=True,
        )
        self._api_thread.start()

    def stop_background_tasks(self):
        self._api_stop_event.set()
        if self._api_thread is not None:
            self._api_thread.join(timeout=2)

    def _apply_api_status(self, status: dict):
        self._api_status = status
        classes = [str(label) for label in status.get("classes", [])]
        if classes and classes != self.classes:
            self.classes = classes
            self._class_counts = {label: 0 for label in classes}
            self._ema_probs = np.zeros(len(classes), dtype=np.float32)
            self._sysinfo_page.set_classes(classes)
            self._log_page.set_mode(classes)

        model_key = str(status.get("model", self.model_key))
        mode = str(status.get("mode", self._current_mode))
        if model_key in config.MODEL_KEYS:
            self.model_key = model_key
        if mode in {"har", "motion"}:
            self._current_mode = mode
            models_dir = (
                config.MODELS_HAR_DIR if mode == "har" else config.MODELS_MOTION_DIR
            )
            if models_dir != self.models_dir:
                self.models_dir = models_dir
                self._settings_page.refresh_models_dir(models_dir)
            self._monitor_page.set_mode(mode)
        settings = status.get("settings", {})
        self._settings_page.apply_remote_state(
            self.model_key,
            self._current_mode,
            settings if isinstance(settings, dict) else None,
            status.get("available_models_by_mode"),
        )
        self._monitor_page.set_model_name(self.model_key)
        self._sysinfo_page.set_model_name(self.model_key)

        if isinstance(settings, dict):
            self.window_size = int(settings.get("window_size", self.window_size))
            self.step = int(settings.get("step_size", self.step))
            self.cutoff = float(settings.get("cutoff", self.cutoff))
            self.ema_alpha = float(settings.get("ema_alpha", self.ema_alpha))
            self.conf_thresh = float(settings.get("conf_thresh", self.conf_thresh))
            self._hyst_min = int(settings.get("hyst_count", self._hyst_min))
            self.motion_threshold = float(
                settings.get("motion_threshold", self.motion_threshold)
            )

        prediction = status.get("latest_prediction")
        if not isinstance(prediction, dict):
            self._last_api_prediction = ""
            self._state.update(
                {
                    "label": "—",
                    "raw_label": "—",
                    "confidence": 0.0,
                    "all_probs": np.zeros(len(self.classes), dtype=np.float32),
                }
            )
            return
        created_at = str(prediction.get("created_at", ""))
        if not created_at or created_at == self._last_api_prediction:
            return
        self._last_api_prediction = created_at
        label = str(prediction.get("label", "—"))
        confidence = float(prediction.get("confidence", 0.0))
        probability_map = prediction.get("probabilities", {})
        probabilities = np.asarray(
            [float(probability_map.get(name, 0.0)) / 100.0 for name in self.classes],
            dtype=np.float32,
        )
        frame_count = int(prediction.get("frame_count", status.get("frames_received", 0)))
        self._state.update(
            {
                "label": label,
                "raw_label": label,
                "confidence": confidence,
                "all_probs": probabilities,
            }
        )
        now = time.monotonic()
        if (
            label != self._last_record_lbl
            or now - self._last_record_t >= 4.0
        ):
            self._last_record_lbl = label
            self._last_record_t = now
            self._record(label, confidence, probabilities, frame_count)

    def _consume_api_state(self):
        status = None
        while True:
            try:
                status = self._api_status_queue.get_nowait()
            except queue.Empty:
                break
        if status is not None:
            self._apply_api_status(status)
        try:
            error = self._api_error_queue.get_nowait()
        except queue.Empty:
            return
        self._settings_page.show_api_error(error)

    def switch_model(self, model_key: str) -> tuple[bool, str]:
        if self.api_enabled:
            self.model_key = model_key
            self._sync_settings_with_api({"model": model_key}, "model")
            return True, f"Requesting API model '{model_key}'"
        try:
            pipeline, le, model = load_models(self.models_dir, model_key)
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
            if self._infer_worker is not None:
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

    def _sync_settings_with_api(self, settings: dict, setting_name: str) -> None:
        server_url = os.environ.get("CSI_SERVER_URL", "").strip()
        token = os.environ.get("CSI_EDGE_TOKEN", "")
        if not self.api_enabled:
            return

        def _send_settings_update():
            try:
                from csi_forwarding import (
                    send_settings_with_retry,
                    validate_server_url,
                )
                validate_server_url(server_url)
                status = send_settings_with_retry(server_url, token, settings)
                self._queue_latest(self._api_status_queue, status)
                print(f"[OK]  API inference {setting_name} synchronized.")
            except (
                ImportError,
                RuntimeError,
                ValueError,
                OSError,
                TimeoutError,
                ConnectionError,
            ) as exc:
                message = f"{setting_name} update failed: {exc}"
                print(f"[ERROR] API {message}")
                self._queue_latest(self._api_error_queue, message)

        threading.Thread(
            target=_send_settings_update,
            name="csi-api-settings-sync",
            daemon=True,
        ).start()

    def _apply_runtime_settings(self, settings: dict):
        self._sync_settings_with_api(settings, "inference settings")

    def switch_mode_dir(self, mode: str):
        """Switch models directory between HAR (models/har) and Super Motion (models/motion)."""
        if mode not in {"har", "motion"}:
            return
        new_dir = config.MODELS_HAR_DIR if mode == "har" else config.MODELS_MOTION_DIR
        if self.api_enabled:
            selected_model = self.model_key
            model_path = config.resolve_project_path(new_dir) / f"{selected_model}.joblib"
            if not model_path.is_file():
                selected_model = "rf"
            self._current_mode = mode
            self.model_key = selected_model
            self._monitor_page.set_mode(mode)
            if new_dir != self.models_dir:
                self.models_dir = new_dir
                self._settings_page.refresh_models_dir(new_dir)
            self._sync_settings_with_api(
                {"mode": mode, "model": selected_model},
                "inference mode",
            )
            self._log_page.set_mode(list(self.classes))
            return
        self._current_mode = mode
        self._monitor_page.set_mode(mode)
        # Load new model first so self.classes reflects the target mode's label encoder
        if new_dir != self.models_dir:
            self.models_dir = new_dir
            self._settings_page.refresh_models_dir(new_dir)
            ok, msg = self.switch_model(self.model_key)
            if not ok:
                print(f"[WARN] Mode switch to '{mode}' failed: {msg}")
        # Reset log using the now-correct class list from the loaded model
        new_classes = list(self.classes)
        self._log_entries.clear()
        self._class_counts = {c: 0 for c in new_classes}
        self._conf_sum = 0.0; self._total_preds = 0
        self._last_record_lbl = ""; self._hyst_pending = ""; self._hyst_count = 0
        self._state.update({
            "label": "—", "raw_label": "—", "confidence": 0.0,
            "log_entries": [], "total_predictions": 0,
            "avg_confidence": 0.0, "class_counts": {},
        })
        self._log_page.set_mode(new_classes)

    def _switch_page(self, idx: int):
        self._stack.setCurrentIndex(idx)
        # Immediately refresh the newly-visible page so it's never stale
        snap = self.reader.snapshot()
        if snap is not None and 0 <= idx < len(self._pages):
            motion_intensity = 0.0
            ma = snap.get("mean_amp", 0.0)
            if ma > 1e-6:
                motion_intensity = min(1.0, math.sqrt(snap.get("variance", 0.0)) / ma * 4.0)
            state = {**snap, **self._state, "latency_ms": self._latency_ms,
                     "motion_intensity": motion_intensity, "mode": self._current_mode}
            page = self._pages[idx]
            if hasattr(page, "update_curves"):
                page.update_curves(snap)
            page.update(state)

    def _clear_log(self):
        self._log_entries.clear()
        self._class_counts = {c: 0 for c in self.classes}
        self._conf_sum = 0.0; self._total_preds = 0

    def start_timer(self, refresh_ms: int):
        self._start_api_poller()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(refresh_ms)
        # Frame skip counter: heavy page.update(state) runs every N-th tick.
        # At refresh_ms=20 (50Hz) with skip=2 → plots refresh at 25Hz, but the
        # reader thread gets 2× more GIL time → recovers dropped CSI frames.
        self._plot_skip_n   = 2
        self._plot_skip_ctr = 0

    def _refresh(self):
        # Full snapshot every tick — we now feed update_curves() on every tick
        # for smooth wave/bar visuals at the full timer rate. The wave_buf copy
        # is ~200 floats under the lock; the lock-hold delta vs snapshot_light()
        # is microseconds and doesn't hurt the 100 Hz reader.
        snap = self.reader.snapshot()
        if snap is None:
            return

        new_frames = snap["frame_count"] - self._last_seen
        self._last_seen    = snap["frame_count"]
        self._frames_since += new_frames

        variance = float(snap.get("variance", 0.0))
        mean_amp = float(snap.get("mean_amp", 0.0))
        if mean_amp > 1e-6:
            motion_intensity = min(1.0, math.sqrt(variance) / mean_amp * 4.0)
        else:
            motion_intensity = 0.0

        # ── HAR mode: ML inference ───────────────────────────────────────
        # Real inference is scheduled inside the reader process. The GUI only
        # drives the synthetic demo predictor (random labels) when no model
        # is loaded.
        if self.api_enabled:
            self._consume_api_state()
        elif self._current_mode == "har":
            if self.demo and self.pipeline is None and self._frames_since >= self.step:
                self._frames_since = 0
                self._demo_predict(snap["frame_count"])

            result = self._infer_worker.get_result()
            if result is not None:
                raw_cand, conf_cand, probs_cand, latency, fc = result
                self._latency_ms = latency

                if raw_cand is not None and probs_cand is not None:
                    max_prob      = float(np.max(probs_cand))
                    dynamic_alpha = self.ema_alpha * max_prob
                    self._ema_probs = (dynamic_alpha * probs_cand +
                                       (1.0 - dynamic_alpha) * self._ema_probs)
                    best_idx      = int(np.argmax(self._ema_probs))
                    smoothed_prob = float(self._ema_probs[best_idx]) * 100.0
                    smoothed_cand = self.classes[best_idx]
                    # No uncertain "—" state in HAR — always show the best prediction

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

        # ── Super Motion mode: hyper-sensitive ML inference ─────────────
        elif self._current_mode == "motion":
            if self.demo and self.pipeline is None and self._frames_since >= self.step:
                self._frames_since = 0
                self._demo_predict(snap["frame_count"])

            result = self._infer_worker.get_result()
            if result is not None:
                raw_cand, conf_cand, probs_cand, latency, fc = result
                self._latency_ms = latency

                if raw_cand is not None and probs_cand is not None:
                    best_idx        = int(np.argmax(probs_cand))
                    best_prob       = float(probs_cand[best_idx]) * 100.0
                    model_says_walk = (self.classes[best_idx] == "walk_activity")

                    # Hyper-sensitive: model OR signal intensity above low threshold
                    is_motion   = model_says_walk or motion_intensity > 0.10
                    motion_cand = "walk_activity" if is_motion else "no_activity"
                    report_conf = best_prob if model_says_walk else motion_intensity * 100.0

                    now = time.monotonic()
                    if (motion_cand != self._last_record_lbl or
                            (now - self._last_record_t) >= 4.0):
                        self._last_record_lbl = motion_cand
                        self._last_record_t   = now
                        self._record(motion_cand, report_conf, probs_cand, fc)

        self._sidebar.update_health(int(snap["infer_fill"] * 100), 0)
        self._pill.update_status(snap["frame_count"], snap["fps"])
        # Smooth-tick path: visible page's live plots refresh every tick
        # (cheap setData/setOpts). The heavy text/gauges/recent-table update
        # is throttled to every N-th tick to keep GIL pressure low so the
        # reader thread can sustain 100 Hz CSI frames.
        visible_page = self._pages[self._stack.currentIndex()]
        if hasattr(visible_page, "update_curves"):
            visible_page.update_curves(snap)

        self._plot_skip_ctr += 1
        if self._plot_skip_ctr >= self._plot_skip_n:
            self._plot_skip_ctr = 0
            state = {**snap, **self._state, "latency_ms": self._latency_ms,
                     "motion_intensity": motion_intensity,
                     "mode": self._current_mode}
            visible_page.update(state)

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
