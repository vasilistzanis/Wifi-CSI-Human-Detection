from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from fastapi import WebSocket
import joblib
import numpy as np
from starlette.websockets import WebSocketDisconnect

import config
from csi_ml_pipeline import FEATURE_VECTOR_VERSION
from csi_parser import parse_csi_line
from live_predict import run_inference
from web_app.persistence import (
    _close_activity_session,
    _create_activity_session,
    _get_runtime_settings,
    _save_activity_stat,
    _save_prediction,
    _save_runtime_setting,
    _save_setting_change,
    _update_activity_session,
)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get(
    "CSI_MODEL_DIR", config.resolve_project_path(config.MODELS_HAR_DIR)
))
MODEL_KEY = os.environ.get("CSI_MODEL", "rf").lower()
MODE_KEY = os.environ.get("CSI_MODE", "har").lower()
MAX_FRAME_LENGTH = 16_384
PREDICTION_SAVE_INTERVAL = 5.0
ACTIVITY_SESSION_IDLE_SECONDS = 15

if MODEL_KEY not in config.MODEL_KEYS:
    raise RuntimeError(f"CSI_MODEL must be one of: {', '.join(config.MODEL_KEYS)}")
if MODE_KEY not in {"har", "motion"}:
    raise RuntimeError("CSI_MODE must be either 'har' or 'motion'.")


def _model_dir_for_mode(mode_key: str) -> Path:
    if mode_key not in {"har", "motion"}:
        raise ValueError("Mode must be either 'har' or 'motion'.")
    if mode_key == "har":
        return MODEL_DIR
    if MODEL_DIR.name.lower() in {"har", "motion"}:
        return MODEL_DIR.parent / "motion"
    return config.resolve_project_path(config.MODELS_MOTION_DIR)


def _load_model_artifacts(
    model_key: str,
    mode_key: str = "har",
) -> tuple[object, object, object, list[str]]:
    if model_key not in config.MODEL_KEYS:
        raise ValueError(f"Model must be one of: {', '.join(config.MODEL_KEYS)}.")
    model_dir = _model_dir_for_mode(mode_key)
    paths = (
        model_dir / "csi_pipeline.joblib",
        model_dir / "label_encoder.joblib",
        model_dir / config.MODEL_FILES[model_key],
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required model artifact not found: {path}")
        with path.open("rb") as artifact:
            if artifact.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise RuntimeError(
                    f"{path} is a Git LFS pointer, not a downloaded model. "
                    "Install Git LFS and run 'git lfs pull' before starting."
                )
    pipeline, label_encoder, model = (joblib.load(path) for path in paths)
    metrics_path = model_dir / "metrics.json"
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        selected_metrics = metrics.get(model_key, {})
        saved_version = (
            selected_metrics.get("feature_vector_version")
            if isinstance(selected_metrics, dict)
            else None
        )
        if saved_version and saved_version != FEATURE_VECTOR_VERSION:
            raise RuntimeError(
                "Model feature-vector version does not match the current "
                "inference code; retrain the model before starting the API."
            )
    classes = [str(label) for label in label_encoder.classes_]
    if not classes:
        raise RuntimeError("The loaded label encoder contains no classes.")
    return pipeline, label_encoder, model, classes


def _model_artifacts_available(model_key: str, mode_key: str = "har") -> bool:
    model_dir = _model_dir_for_mode(mode_key)
    paths = (
        model_dir / "csi_pipeline.joblib",
        model_dir / "label_encoder.joblib",
        model_dir / config.MODEL_FILES[model_key],
    )
    for path in paths:
        if not path.is_file():
            return False
        with path.open("rb") as artifact:
            if artifact.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                return False
    return True


class InferenceEngine:
    def __init__(self) -> None:
        saved_settings = _get_runtime_settings()
        self.model_key = saved_settings.get("model", MODEL_KEY)
        self.mode_key = saved_settings.get("mode", MODE_KEY)
        self.window_size = int(saved_settings.get("window_size", config.WINDOW_SIZE))
        self.step_size = int(
            saved_settings.get("step_size", config.PREDICTION_STEP_SIZE)
        )
        self.warmup_frames = int(
            saved_settings.get("warmup_frames", config.FILTER_WARMUP)
        )
        self.cutoff = float(saved_settings.get("cutoff", config.FILTER_CUTOFF_HZ))
        self.ema_alpha = float(
            saved_settings.get("ema_alpha", config.DASHBOARD_EMA_ALPHA)
        )
        self.conf_thresh = float(
            saved_settings.get("conf_thresh", config.DASHBOARD_CONF_THRESH)
        )
        self.hyst_min = int(
            saved_settings.get("hyst_count", config.DASHBOARD_HYST_COUNT)
        )
        self.motion_threshold = float(
            saved_settings.get("motion_threshold", "0.10")
        )
        if self.model_key not in config.MODEL_KEYS or self.mode_key not in {"har", "motion"}:
            raise RuntimeError("Saved inference model or mode is invalid.")
        (
            self.pipeline,
            self.label_encoder,
            self.model,
            self.classes,
        ) = _load_model_artifacts(self.model_key, self.mode_key)
        self.buffer: deque[np.ndarray] = deque(
            maxlen=self.window_size + self.warmup_frames
        )
        self.frames_received = 0
        self.invalid_frames = 0
        self.frames_since_prediction = 0
        self.frame_times: deque[float] = deque(maxlen=1000)
        self.latest_prediction: dict | None = None
        self.last_persisted_at = 0.0
        self.activity_session_id: int | None = None
        self.last_frame_monotonic: float | None = None
        self.last_frame_at: str | None = None
        self.ema_probs = np.zeros(len(self.classes), dtype=np.float32)
        self.pending_label = ""
        self.pending_count = 0
        self.active_label = ""
        self.last_recorded_label = ""
        self.last_recorded_at = 0.0
        self.mean_amp = 0.0
        self.amplitude_variance = 0.0
        self.lock = asyncio.Lock()
        self.settings_lock = asyncio.Lock()
        self.clients: set[WebSocket] = set()

    def models(self) -> dict:
        return {
            "selected": self.model_key,
            "available": [
                key for key in config.MODEL_FILES
                if _model_artifacts_available(key, self.mode_key)
            ],
        }

    async def select_model(self, model_key: str) -> dict:
        return await self.update_settings({"model": model_key})

    async def update_settings(self, requested: dict[str, object]) -> dict:
        async with self.settings_lock:
            return await self._update_settings(requested)

    async def _update_settings(self, requested: dict[str, object]) -> dict:
        candidate = {
            "model": self.model_key,
            "mode": self.mode_key,
            "window_size": self.window_size,
            "step_size": self.step_size,
            "warmup_frames": self.warmup_frames,
            "cutoff": self.cutoff,
            "ema_alpha": self.ema_alpha,
            "conf_thresh": self.conf_thresh,
            "hyst_count": self.hyst_min,
            "motion_threshold": self.motion_threshold,
        }
        candidate.update(requested)
        model_key = str(candidate["model"]).strip().lower()
        mode_key = str(candidate["mode"]).strip().lower()
        if model_key not in config.MODEL_KEYS:
            raise ValueError(f"Model must be one of: {', '.join(config.MODEL_KEYS)}.")
        if mode_key not in {"har", "motion"}:
            raise ValueError("Mode must be either 'har' or 'motion'.")
        candidate["model"] = model_key
        candidate["mode"] = mode_key
        changed = {
            name: (getattr(self, attribute), candidate[name])
            for name, attribute in (
                ("model", "model_key"),
                ("mode", "mode_key"),
                ("window_size", "window_size"),
                ("step_size", "step_size"),
                ("warmup_frames", "warmup_frames"),
                ("cutoff", "cutoff"),
                ("ema_alpha", "ema_alpha"),
                ("conf_thresh", "conf_thresh"),
                ("hyst_count", "hyst_min"),
                ("motion_threshold", "motion_threshold"),
            )
            if getattr(self, attribute) != candidate[name]
        }
        if not changed:
            return self.status()

        loaded = None
        if (
            model_key != self.model_key
            or mode_key != self.mode_key
        ):
            loaded = await asyncio.to_thread(
                _load_model_artifacts, model_key, mode_key
            )
        async with self.lock:
            changed_at = datetime.now(UTC).isoformat()
            session_id = self.activity_session_id
            for setting, (old_value, new_value) in changed.items():
                _save_setting_change(
                    session_id,
                    setting,
                    str(old_value),
                    str(new_value),
                    changed_at,
                )
                await asyncio.to_thread(
                    _save_runtime_setting, setting, str(new_value)
                )
            if loaded is not None:
                (
                    self.pipeline,
                    self.label_encoder,
                    self.model,
                    self.classes,
                ) = loaded
                self.model_key = model_key
                self.mode_key = mode_key
            self.window_size = int(candidate["window_size"])
            self.step_size = int(candidate["step_size"])
            self.warmup_frames = int(candidate["warmup_frames"])
            self.cutoff = float(candidate["cutoff"])
            self.ema_alpha = float(candidate["ema_alpha"])
            self.conf_thresh = float(candidate["conf_thresh"])
            self.hyst_min = int(candidate["hyst_count"])
            self.motion_threshold = float(candidate["motion_threshold"])
            self.buffer = deque(
                self.buffer,
                maxlen=self.window_size + self.warmup_frames,
            )
            self._reset_prediction_state()
            status = self.status()
        change = {
            "settings": {
                name: {"old_value": str(old), "new_value": str(new)}
                for name, (old, new) in changed.items()
            },
            "changed_at": changed_at,
            "session_id": session_id,
        }
        await self.broadcast(
            {"type": "settings_changed", "status": status, "change": change}
        )
        return status

    def _reset_prediction_state(self) -> None:
        self.frames_since_prediction = 0
        self.last_persisted_at = 0.0
        self.latest_prediction = None
        self.ema_probs = np.zeros(len(self.classes), dtype=np.float32)
        self.pending_label = ""
        self.pending_count = 0
        self.active_label = ""
        self.last_recorded_label = ""
        self.last_recorded_at = 0.0
        self.buffer.clear()

    def _ensure_activity_session(self, timestamp: str, monotonic_now: float) -> int:
        if (
            self.activity_session_id is not None
            and self.last_frame_monotonic is not None
            and monotonic_now - self.last_frame_monotonic
            <= ACTIVITY_SESSION_IDLE_SECONDS
        ):
            return self.activity_session_id
        if self.activity_session_id is not None and self.last_frame_at is not None:
            _close_activity_session(self.activity_session_id, self.last_frame_at)
        session_id = _create_activity_session(timestamp)
        self.activity_session_id = session_id
        self.last_persisted_at = 0.0
        return session_id

    async def close_idle_session(self) -> bool:
        async with self.lock:
            if (
                self.activity_session_id is None
                or self.last_frame_monotonic is None
                or self.last_frame_at is None
                or time.monotonic() - self.last_frame_monotonic
                < ACTIVITY_SESSION_IDLE_SECONDS
            ):
                return False
            session_id = self.activity_session_id
            _close_activity_session(session_id, self.last_frame_at)
            self.activity_session_id = None
            self.last_frame_monotonic = None
            self.last_frame_at = None
            status = self.status()
        await self.broadcast(
            {"type": "session_closed", "status": status, "session_id": session_id}
        )
        return True

    async def process_batch(self, lines: list[str]) -> dict:
        updates: list[dict] = []
        accepted = 0
        session_id: int | None = None
        async with self.lock:
            for line in lines:
                if len(line) > MAX_FRAME_LENGTH:
                    self.invalid_frames += 1
                    continue
                frame = parse_csi_line(line)
                if frame is None:
                    self.invalid_frames += 1
                    continue
                expected = getattr(self.pipeline, "_fitted_n_subcarriers", None)
                if expected is not None and frame.shape[0] != expected:
                    self.invalid_frames += 1
                    continue

                now_monotonic = time.monotonic()
                now_timestamp = datetime.now(UTC).isoformat()
                if session_id is None:
                    session_id = self._ensure_activity_session(
                        now_timestamp, now_monotonic
                    )
                self.last_frame_monotonic = now_monotonic
                self.last_frame_at = now_timestamp
                self.buffer.append(frame)
                amplitude_mean = float(np.abs(frame).mean())
                self.amplitude_variance = (
                    self.amplitude_variance * 0.97
                    + (amplitude_mean - self.mean_amp) ** 2 * 0.03
                )
                self.mean_amp = self.mean_amp * 0.97 + amplitude_mean * 0.03
                accepted += 1
                self.frames_received += 1
                self.frames_since_prediction += 1
                self.frame_times.append(time.monotonic())

                if (
                    len(self.buffer) >= self.buffer.maxlen
                    and self.frames_since_prediction >= self.step_size
                ):
                    prediction = await asyncio.to_thread(
                        run_inference,
                        self.buffer,
                        self.pipeline,
                        self.model,
                        self.label_encoder,
                        self.window_size,
                        self.cutoff,
                    )
                    self.frames_since_prediction = 0
                    if prediction is not None:
                        _, _, probabilities = prediction
                        probabilities = np.asarray(probabilities, dtype=np.float32)
                        motion_intensity = (
                            min(
                                1.0,
                                np.sqrt(self.amplitude_variance)
                                / self.mean_amp
                                * 4.0,
                            )
                            if self.mean_amp > 1e-6
                            else 0.0
                        )
                        if self.mode_key == "motion":
                            best_index = int(np.argmax(probabilities))
                            model_says_motion = (
                                self.classes[best_index] == "walk_activity"
                            )
                            is_motion = (
                                model_says_motion
                                or motion_intensity > self.motion_threshold
                            )
                            label = "walk_activity" if is_motion else "no_activity"
                            confidence = (
                                float(probabilities[best_index]) * 100.0
                                if model_says_motion
                                else motion_intensity * 100.0
                            )
                            probabilities_to_publish = probabilities
                        else:
                            if self.ema_probs.shape != probabilities.shape:
                                self.ema_probs = np.zeros_like(probabilities)
                            max_probability = float(np.max(probabilities))
                            alpha = self.ema_alpha * max_probability
                            self.ema_probs = (
                                alpha * probabilities
                                + (1.0 - alpha) * self.ema_probs
                            )
                            best_index = int(np.argmax(self.ema_probs))
                            candidate_label = self.classes[best_index]
                            if candidate_label == self.pending_label:
                                self.pending_count += 1
                            else:
                                self.pending_label = candidate_label
                                self.pending_count = 1
                            if (
                                self.pending_count >= self.hyst_min
                                and float(self.ema_probs[best_index]) * 100
                                >= self.conf_thresh
                            ):
                                self.active_label = candidate_label
                            if not self.active_label:
                                self.active_label = candidate_label
                            label = self.active_label
                            confidence = float(
                                self.ema_probs[self.classes.index(label)] * 100
                            )
                            probabilities_to_publish = self.ema_probs
                        result = {
                            "label": label,
                            "confidence": round(float(confidence), 2),
                            "probabilities": {
                                name: round(float(value) * 100, 2)
                                for name, value in zip(
                                    self.classes, probabilities_to_publish
                                )
                            },
                            "motion_intensity": round(float(motion_intensity), 4),
                            "frame_count": self.frames_received,
                            "session_id": session_id,
                            "model": self.model_key,
                            "mode": self.mode_key,
                            "created_at": datetime.now(UTC).isoformat(),
                        }
                        self.latest_prediction = result
                        updates.append(result)
                        now = time.monotonic()
                        if (
                            label != self.last_recorded_label
                            or now - self.last_recorded_at >= 4.0
                        ):
                            await asyncio.to_thread(_save_activity_stat, result)
                            self.last_recorded_label = label
                            self.last_recorded_at = now
                        if now - self.last_persisted_at >= PREDICTION_SAVE_INTERVAL:
                            await asyncio.to_thread(_save_prediction, result)
                            self.last_persisted_at = now

            if accepted and session_id is not None and self.last_frame_at is not None:
                await asyncio.to_thread(
                    _update_activity_session,
                    session_id,
                    accepted,
                    self.last_frame_at,
                )
            payload = self.status()

        event = {"type": "update", "status": payload, "predictions": updates}
        await self.broadcast(event)
        return {"accepted": accepted, **payload}

    def status(self) -> dict:
        elapsed = 0.0
        if len(self.frame_times) > 1:
            elapsed = self.frame_times[-1] - self.frame_times[0]
        connected = bool(
            self.frame_times and time.monotonic() - self.frame_times[-1] < 5
        )
        fps = (
            (len(self.frame_times) - 1) / elapsed
            if connected and elapsed > 0
            else 0.0
        )
        available_models_by_mode = {
            mode: [
                key for key in config.MODEL_FILES
                if _model_artifacts_available(key, mode)
            ]
            for mode in ("har", "motion")
        }
        return {
            "device_connected": connected,
            "frames_received": self.frames_received,
            "invalid_frames": self.invalid_frames,
            "buffered_frames": len(self.buffer),
            "buffer_capacity": self.buffer.maxlen,
            "fps": round(fps, 1),
            "model": self.model_key,
            "mode": self.mode_key,
            "available_models": available_models_by_mode[self.mode_key],
            "available_models_by_mode": available_models_by_mode,
            "classes": self.classes,
            "settings": {
                "window_size": self.window_size,
                "step_size": self.step_size,
                "warmup_frames": self.warmup_frames,
                "cutoff": self.cutoff,
                "ema_alpha": self.ema_alpha,
                "conf_thresh": self.conf_thresh,
                "hyst_count": self.hyst_min,
                "motion_threshold": self.motion_threshold,
            },
            "window_size": self.window_size,
            "session_id": self.activity_session_id,
            "latest_prediction": self.latest_prediction,
        }

    async def broadcast(self, event: dict) -> None:
        stale: list[WebSocket] = []
        for client in tuple(self.clients):
            try:
                await client.send_json(event)
            except (RuntimeError, OSError, WebSocketDisconnect):
                stale.append(client)
        for client in stale:
            self.clients.discard(client)
