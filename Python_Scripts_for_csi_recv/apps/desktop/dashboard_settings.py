"""Model, inference-mode, and runtime settings page."""

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QButtonGroup, QDoubleSpinBox, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QPushButton, QRadioButton, QScrollArea,
    QSpinBox, QVBoxLayout, QWidget,
)

import config

from .dashboard_theme import (
    _BLUE, _BORDER, _DARK2, _DIM, _GREEN, _MODEL_DESC, _MODEL_NAMES,
    _RED, _TEXT, _YELLOW,
)
from .dashboard_widgets import _card, _hdr


class SettingsPage(QWidget):
    def __init__(self, models_dir: str, current_model: str, on_deploy,
                 current_mode: str = "har", on_mode_change=None,
                 runtime_settings: dict | None = None,
                 on_runtime_settings=None, api_enabled: bool = False):
        super().__init__()
        self._models_dir      = models_dir
        self._current         = current_model
        self._selected        = current_model
        self._on_deploy       = on_deploy
        self._current_mode    = current_mode
        self._selected_mode   = current_mode
        self._on_mode_change  = on_mode_change
        self._runtime_settings = runtime_settings or {}
        self._on_runtime_settings = on_runtime_settings
        self._api_enabled = api_enabled
        self._runtime_dirty = False
        self._runtime_request_pending = False
        self._runtime_controls: dict[str, QWidget] = {}
        self._radios: dict[str, QRadioButton] = {}
        self._status_pills: dict[str, QLabel] = {}
        self._build()

    def _scan(self) -> list[tuple[str, bool]]:
        d = Path(self._models_dir)
        pipeline_ok = (d / "csi_pipeline.joblib").exists() and (d / "label_encoder.joblib").exists()
        return [(k, pipeline_ok and (d / f"{k}.joblib").exists()) for k in config.MODEL_KEYS]

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(24, 16, 24, 16); root.setSpacing(16)

        h1 = QLabel("Settings")
        h1.setStyleSheet(
            f"font-size:22px; font-weight:bold; color:{_TEXT};"
            f" background:transparent; border:none;"
        )
        root.addWidget(h1)

        # ── Inference Mode card ───────────────────────────────────────────────
        mode_card = _card()
        modelay = QVBoxLayout(mode_card)
        modelay.setContentsMargins(20, 16, 20, 16); modelay.setSpacing(10)
        modelay.addWidget(_hdr("INFERENCE MODE"))

        _MODE_INFO = {
            "har":    ("HAR Mode",      "Human Activity Recognition — classify activity from motion.", config.MODELS_HAR_DIR),
            "motion": ("Super Motion",  "Motion intensity detection — different training parameters.", config.MODELS_MOTION_DIR),
        }
        mode_group = QButtonGroup(self)
        self._mode_radios: dict[str, QRadioButton] = {}
        for mkey, (mname, mdesc, mdir) in _MODE_INFO.items():
            mrow = QWidget()
            mrow.setStyleSheet(
                f"QWidget{{background:{_DARK2}; border-radius:6px;"
                f" border:1px solid {_BORDER};}}"
            )
            mrl = QHBoxLayout(mrow); mrl.setContentsMargins(14, 10, 14, 10); mrl.setSpacing(12)
            mrb = QRadioButton()
            mrb.setChecked(mkey == self._current_mode)
            mrb.setStyleSheet("QRadioButton{background:transparent; border:none;}")
            mode_group.addButton(mrb)
            self._mode_radios[mkey] = mrb
            mrb.toggled.connect(lambda checked, k=mkey: self._on_mode_radio(k, checked))
            mrl.addWidget(mrb)
            minfo = QVBoxLayout(); minfo.setSpacing(2)
            mn_lbl = QLabel(mname)
            mn_lbl.setStyleSheet(
                f"font-size:13px; font-weight:bold; color:{_TEXT};"
                f" background:transparent; border:none;"
            )
            md_lbl = QLabel(mdesc)
            md_lbl.setStyleSheet(
                f"font-size:10px; color:{_DIM}; background:transparent; border:none;"
            )
            minfo.addWidget(mn_lbl); minfo.addWidget(md_lbl)
            mrl.addLayout(minfo, 1)
            mdir_lbl = QLabel(mdir)
            mdir_lbl.setStyleSheet(
                f"font-size:9px; color:{_DIM}; background:transparent; border:none;"
            )
            mrl.addWidget(mdir_lbl)
            if mkey == self._current_mode:
                mpill_text, mpill_color = "ACTIVE", _GREEN
            else:
                mpill_text, mpill_color = "READY", _DIM
            mpill = QLabel(mpill_text)
            mpill.setFixedWidth(64)
            mpill.setAlignment(Qt.AlignCenter)
            mpill.setStyleSheet(
                f"font-size:9px; font-weight:bold; color:{mpill_color};"
                f" background:{mpill_color}22; border:1px solid {mpill_color}44;"
                f" border-radius:8px; padding:2px 6px;"
            )
            self._mode_pills = getattr(self, "_mode_pills", {})
            self._mode_pills[mkey] = mpill
            mrl.addWidget(mpill)
            modelay.addWidget(mrow)

        modelay.addSpacing(4)
        mode_act_row = QHBoxLayout(); mode_act_row.setSpacing(12)
        self._mode_btn = QPushButton("⚡   Switch Mode")
        self._mode_btn.setFixedHeight(38)
        self._mode_btn.setStyleSheet(
            f"QPushButton{{background:{_BLUE}; color:#ffffff; border:none;"
            f" border-radius:6px; font-size:13px; font-weight:bold; padding:0 20px;}}"
            f"QPushButton:hover{{background:{_BLUE}dd;}}"
            f"QPushButton:disabled{{background:{_DARK2}; color:{_DIM};"
            f" border:1px solid {_BORDER};}}"
        )
        self._mode_btn.setEnabled(False)
        self._mode_btn.clicked.connect(self._switch_mode)
        self._mode_status_lbl = QLabel(
            f"Active: {_MODE_INFO.get(self._current_mode, ('—',))[0]}"
        )
        self._mode_status_lbl.setStyleSheet(
            f"font-size:11px; color:{_DIM}; background:transparent; border:none;"
        )
        mode_act_row.addWidget(self._mode_btn)
        mode_act_row.addWidget(self._mode_status_lbl, 1)
        modelay.addLayout(mode_act_row)
        root.addWidget(mode_card)

        # Model selection card
        mc = _card()
        mv = QVBoxLayout(mc); mv.setContentsMargins(20, 16, 20, 16); mv.setSpacing(10)
        mv.addWidget(_hdr("MODEL SELECTION"))

        self._dir_lbl = QLabel(f"Models directory: {self._models_dir}")
        self._dir_lbl.setStyleSheet(
            f"font-size:10px; color:{_DIM}; background:transparent; border:none;"
        )
        self._dir_lbl.setWordWrap(True)
        mv.addWidget(self._dir_lbl)

        sep = QWidget(); sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{_BORDER}; border:none;")
        mv.addWidget(sep)

        self._model_rows:      dict[str, QWidget] = {}
        self._model_name_lbls: dict[str, QLabel]  = {}
        group = QButtonGroup(self)
        models = self._scan()
        for key, available in models:
            row = QWidget()
            row_bg = _DARK2 if available else f"{_DARK2}44"
            row_border = _BORDER if available else f"{_BORDER}44"
            row.setStyleSheet(
                f"QWidget{{background:{row_bg}; border-radius:6px;"
                f" border:1px solid {row_border};}}"
            )
            self._model_rows[key] = row
            rl = QHBoxLayout(row); rl.setContentsMargins(14, 10, 14, 10); rl.setSpacing(12)

            rb = QRadioButton()
            rb.setChecked(key == self._current)
            rb.setEnabled(available)
            rb.setStyleSheet("QRadioButton{background:transparent; border:none;}")
            group.addButton(rb)
            self._radios[key] = rb
            rb.toggled.connect(lambda checked, k=key: self._on_radio(k, checked))
            rl.addWidget(rb)

            info_v = QVBoxLayout(); info_v.setSpacing(2)
            name_lbl = QLabel(_MODEL_NAMES.get(key, key))
            name_lbl.setStyleSheet(
                f"font-size:13px; font-weight:bold;"
                f" color:{_TEXT if available else _DIM};"
                f" background:transparent; border:none;"
            )
            self._model_name_lbls[key] = name_lbl
            desc_lbl = QLabel(_MODEL_DESC.get(key, ""))
            desc_lbl.setStyleSheet(
                f"font-size:10px; color:{_DIM}; background:transparent; border:none;"
            )
            info_v.addWidget(name_lbl); info_v.addWidget(desc_lbl)
            rl.addLayout(info_v, 1)

            file_lbl = QLabel(f"{key}.joblib")
            file_lbl.setStyleSheet(
                f"font-size:9px; color:{_DIM}; background:transparent; border:none;"
            )
            rl.addWidget(file_lbl)

            if key == self._current:
                pill_text, pill_color = "ACTIVE", _GREEN
            elif available:
                pill_text, pill_color = "READY", _BLUE
            else:
                pill_text, pill_color = "NOT TRAINED", _DIM

            pill = QLabel(pill_text)
            pill.setFixedWidth(80)
            pill.setAlignment(Qt.AlignCenter)
            pill.setStyleSheet(
                f"font-size:9px; font-weight:bold; color:{pill_color};"
                f" background:{pill_color}22; border:1px solid {pill_color}44;"
                f" border-radius:8px; padding:2px 6px;"
            )
            self._status_pills[key] = pill
            rl.addWidget(pill)
            mv.addWidget(row)

        mv.addSpacing(6)

        # Deploy row
        deploy_row = QHBoxLayout(); deploy_row.setSpacing(12)
        self._deploy_btn = QPushButton("⚡   Deploy Model")
        self._deploy_btn.setFixedHeight(38)
        self._deploy_btn.setStyleSheet(
            f"QPushButton{{background:{_BLUE}; color:#ffffff; border:none;"
            f" border-radius:6px; font-size:13px; font-weight:bold; padding:0 20px;}}"
            f"QPushButton:hover{{background:{_BLUE}dd;}}"
            f"QPushButton:disabled{{background:{_DARK2}; color:{_DIM};"
            f" border:1px solid {_BORDER};}}"
        )
        self._deploy_btn.setEnabled(False)
        self._deploy_btn.clicked.connect(self._deploy)

        self._status_lbl = QLabel(
            f"Active: {_MODEL_NAMES.get(self._current, self._current)}"
        )
        self._status_lbl.setStyleSheet(
            f"font-size:11px; color:{_DIM}; background:transparent; border:none;"
        )
        deploy_row.addWidget(self._deploy_btn)
        deploy_row.addWidget(self._status_lbl, 1)
        mv.addLayout(deploy_row)

        root.addWidget(mc)

        runtime_card = _card()
        runtime_layout = QVBoxLayout(runtime_card)
        runtime_layout.setContentsMargins(20, 16, 20, 16)
        runtime_layout.setSpacing(10)
        runtime_layout.addWidget(_hdr("RUNTIME INFERENCE SETTINGS"))
        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(8)
        defaults = {
            "window_size": config.WINDOW_SIZE,
            "step_size": config.PREDICTION_STEP_SIZE,
            "warmup_frames": config.FILTER_WARMUP,
            "cutoff": config.FILTER_CUTOFF_HZ,
            "ema_alpha": config.DASHBOARD_EMA_ALPHA,
            "conf_thresh": config.DASHBOARD_CONF_THRESH,
            "hyst_count": config.DASHBOARD_HYST_COUNT,
            "motion_threshold": 0.10,
        }
        defaults.update(self._runtime_settings)
        controls = (
            ("window_size", "Window size", 10, 1000, 1),
            ("step_size", "Prediction step", 1, 1000, 1),
            ("warmup_frames", "Filter warm-up frames", 0, 1000, 1),
            ("cutoff", "Filter cutoff (Hz)", 0.1, 50.0, 0.1),
            ("ema_alpha", "EMA smoothing (HAR)", 0.01, 1.0, 0.05),
            ("conf_thresh", "Confidence threshold (HAR, %)", 0.0, 100.0, 1.0),
            ("hyst_count", "Hysteresis confirmations", 1, 100, 1),
            ("motion_threshold", "Motion trigger threshold", 0.01, 1.0, 0.01),
        )
        for key, label, minimum, maximum, step in controls:
            if key in {"cutoff", "ema_alpha", "conf_thresh", "motion_threshold"}:
                control = QDoubleSpinBox()
                control.setDecimals(2)
            else:
                control = QSpinBox()
            control.setRange(minimum, maximum)
            control.setSingleStep(step)
            control.setValue(defaults[key])
            control.setEnabled(self._api_enabled)
            control.setStyleSheet(
                f"background:{_DARK2}; color:{_TEXT}; border:1px solid {_BORDER};"
                "border-radius:4px; padding:4px;"
            )
            control.valueChanged.connect(self._mark_runtime_dirty)
            self._runtime_controls[key] = control
            form.addRow(label, control)
        runtime_layout.addLayout(form)
        self._runtime_apply_btn = QPushButton("Apply inference settings")
        self._runtime_apply_btn.setEnabled(self._api_enabled)
        self._runtime_apply_btn.clicked.connect(self._apply_runtime_settings)
        runtime_layout.addWidget(self._runtime_apply_btn, alignment=Qt.AlignRight)
        self._runtime_status_lbl = QLabel(
            "Changes are saved by the API and shared with the web dashboard."
            if self._api_enabled
            else "Connect the API to edit authoritative inference settings."
        )
        self._runtime_status_lbl.setStyleSheet(
            f"font-size:10px; color:{_DIM}; background:transparent; border:none;"
        )
        runtime_layout.addWidget(self._runtime_status_lbl)
        root.addWidget(runtime_card)
        root.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _on_mode_radio(self, key: str, checked: bool):
        if checked:
            self._selected_mode = key
            self._mode_btn.setEnabled(key != self._current_mode)

    def _switch_mode(self):
        new_mode = self._selected_mode
        self._mode_btn.setEnabled(False)
        self._mode_btn.setText("⏳   Switching…")
        QApplication.processEvents()
        if self._on_mode_change:
            self._on_mode_change(new_mode)
        self._mode_btn.setText("⚡   Switch Mode")
        _MODE_NAMES = {"har": "HAR Mode", "motion": "Super Motion"}
        for k, pill in self._mode_pills.items():
            if k == new_mode:
                pill.setText("ACTIVE")
                pill.setStyleSheet(
                    f"font-size:9px; font-weight:bold; color:{_GREEN};"
                    f" background:{_GREEN}22; border:1px solid {_GREEN}44;"
                    f" border-radius:8px; padding:2px 6px;"
                )
            else:
                pill.setText("READY")
                pill.setStyleSheet(
                    f"font-size:9px; font-weight:bold; color:{_DIM};"
                    f" background:{_DIM}22; border:1px solid {_DIM}44;"
                    f" border-radius:8px; padding:2px 6px;"
                )
        self._current_mode = new_mode
        self._mode_status_lbl.setText(f"✓  Switched to {_MODE_NAMES.get(new_mode, new_mode)}")
        self._mode_status_lbl.setStyleSheet(
            f"font-size:11px; color:{_GREEN}; background:transparent; border:none;"
        )

    def _on_radio(self, key: str, checked: bool):
        if checked:
            self._selected = key
            self._deploy_btn.setEnabled(key != self._current)

    def _deploy(self):
        key = self._selected
        self._deploy_btn.setEnabled(False)
        self._deploy_btn.setText("⏳   Loading…")
        QApplication.processEvents()
        ok, msg = self._on_deploy(key)
        self._deploy_btn.setText("⚡   Deploy Model")
        if ok:
            # Update pill states
            for k, pill in self._status_pills.items():
                if k == key:
                    pill.setText("ACTIVE")
                    pill.setStyleSheet(
                        f"font-size:9px; font-weight:bold; color:{_GREEN};"
                        f" background:{_GREEN}22; border:1px solid {_GREEN}44;"
                        f" border-radius:8px; padding:2px 6px;"
                    )
                elif pill.text() == "ACTIVE":
                    pill.setText("READY")
                    pill.setStyleSheet(
                        f"font-size:9px; font-weight:bold; color:{_BLUE};"
                        f" background:{_BLUE}22; border:1px solid {_BLUE}44;"
                        f" border-radius:8px; padding:2px 6px;"
                    )
            self._current = key
            self._status_lbl.setText(f"✓  {msg}")
            self._status_lbl.setStyleSheet(
                f"font-size:11px; color:{_GREEN}; background:transparent; border:none;"
            )
        else:
            self._status_lbl.setText(f"✗  {msg}")
            self._status_lbl.setStyleSheet(
                f"font-size:11px; color:{_RED}; background:transparent; border:none;"
            )
        self._deploy_btn.setEnabled(self._selected != self._current)

    def _mark_runtime_dirty(self, *_args):
        self._runtime_dirty = True
        self._runtime_apply_btn.setEnabled(self._api_enabled)
        self._runtime_status_lbl.setText("Unsaved inference settings.")
        self._runtime_status_lbl.setStyleSheet(
            f"font-size:10px; color:{_YELLOW}; background:transparent; border:none;"
        )

    def _apply_runtime_settings(self):
        if not self._api_enabled or self._on_runtime_settings is None:
            return
        values = {
            key: control.value()
            for key, control in self._runtime_controls.items()
        }
        self._runtime_request_pending = True
        self._runtime_apply_btn.setEnabled(False)
        self._runtime_status_lbl.setText("Applying settings to the API…")
        self._on_runtime_settings(values)

    def refresh_models_dir(self, new_dir: str):
        """Re-scan a new models directory and update all model row states."""
        self._models_dir = new_dir
        self._dir_lbl.setText(f"Models directory: {new_dir}")
        models = self._scan()
        for key, available in models:
            # Radio button
            rb = self._radios[key]
            rb.setEnabled(available)
            if not available and rb.isChecked():
                rb.setChecked(False)
            # Row background
            row_bg     = _DARK2 if available else f"{_DARK2}44"
            row_border = _BORDER if available else f"{_BORDER}44"
            self._model_rows[key].setStyleSheet(
                f"QWidget{{background:{row_bg}; border-radius:6px;"
                f" border:1px solid {row_border};}}"
            )

            # Name label color
            self._model_name_lbls[key].setStyleSheet(
                f"font-size:13px; font-weight:bold;"
                f" color:{_TEXT if available else _DIM};"
                f" background:transparent; border:none;"
            )
            # Status pill
            if key == self._current and available:
                pill_text, pill_color = "ACTIVE", _GREEN
            elif available:
                pill_text, pill_color = "READY", _BLUE
            else:
                pill_text, pill_color = "NOT TRAINED", _DIM
            pill = self._status_pills[key]
            pill.setText(pill_text)
            pill.setStyleSheet(
                f"font-size:9px; font-weight:bold; color:{pill_color};"
                f" background:{pill_color}22; border:1px solid {pill_color}44;"
                f" border-radius:8px; padding:2px 6px;"
            )
        # Reset deploy row
        self._deploy_btn.setEnabled(False)
        self._status_lbl.setText(
            f"Active: {_MODEL_NAMES.get(self._current, self._current)}"
        )
        self._status_lbl.setStyleSheet(
            f"font-size:11px; color:{_DIM}; background:transparent; border:none;"
        )

    def apply_remote_state(
        self,
        model_key: str,
        mode: str,
        settings: dict | None = None,
        models_by_mode: dict | None = None,
    ):
        self._current = self._selected = model_key
        self._current_mode = self._selected_mode = mode
        available_models = set((models_by_mode or {}).get(mode, ()))
        available_modes = set((models_by_mode or {}).keys())
        for key, radio in self._radios.items():
            if available_modes:
                radio.setEnabled(key in available_models)
            radio.setChecked(key == model_key)
        for key, radio in self._mode_radios.items():
            if available_modes:
                radio.setEnabled(bool((models_by_mode or {}).get(key)))
            radio.setChecked(key == mode)
        for key, pill in self._status_pills.items():
            pill.setText(
                "ACTIVE" if key == model_key
                else "READY" if key in available_models
                else "NOT TRAINED" if available_modes
                else "READY"
            )
        for key, pill in self._mode_pills.items():
            pill.setText("ACTIVE" if key == mode else "READY")
        self._deploy_btn.setEnabled(False)
        self._mode_btn.setEnabled(False)
        mode_name = "HAR Mode" if mode == "har" else "Super Motion"
        self._mode_status_lbl.setText(f"Active: {mode_name}")
        if not isinstance(settings, dict):
            return
        keys = tuple(self._runtime_controls)
        matches = all(
            abs(float(self._runtime_controls[key].value()) - float(settings[key])) < 0.005
            for key in keys
            if key in settings
        ) and all(key in settings for key in keys)
        if matches:
            self._runtime_dirty = False
            self._runtime_request_pending = False
            self._runtime_apply_btn.setEnabled(False)
            self._runtime_status_lbl.setText("Saved and active on the API.")
        elif not self._runtime_dirty:
            for key, control in self._runtime_controls.items():
                if key not in settings:
                    continue
                control.blockSignals(True)
                control.setValue(settings[key])
                control.blockSignals(False)
            self._runtime_apply_btn.setEnabled(False)
            self._runtime_status_lbl.setText("Current settings from the API.")
        elif not self._runtime_request_pending:
            self._runtime_apply_btn.setEnabled(self._api_enabled)

    def show_api_error(self, message: str):
        self._status_lbl.setText(f"✗  API: {message}")
        self._status_lbl.setStyleSheet(
            f"font-size:11px; color:{_RED}; background:transparent; border:none;"
        )
        self._runtime_status_lbl.setText(f"API error: {message}")
        self._runtime_request_pending = False
        self._runtime_apply_btn.setEnabled(self._api_enabled)
        self._runtime_status_lbl.setStyleSheet(
            f"font-size:10px; color:{_RED}; background:transparent; border:none;"
        )

    def update(self, state: dict):
        pass
