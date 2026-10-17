from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    WebSocket,
)
from fastapi.responses import FileResponse
import joblib
import numpy as np
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

import config
from csi_ml_pipeline import FEATURE_VECTOR_VERSION
from csi_parser import parse_csi_line
from live_predict import run_inference

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
DB_PATH = Path(os.environ.get("CSI_DB_PATH", APP_DIR / "data" / "app.sqlite3"))
MODEL_DIR = Path(os.environ.get(
    "CSI_MODEL_DIR", config.resolve_project_path(config.MODELS_HAR_DIR)
))
MODEL_KEY = os.environ.get("CSI_MODEL", "rf").lower()
EDGE_TOKEN = os.environ.get("CSI_EDGE_TOKEN", "")
ADMIN_USERNAME = os.environ.get("CSI_ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.environ.get("CSI_ADMIN_PASSWORD", "")
COOKIE_SECURE = os.environ.get("CSI_COOKIE_SECURE", "true").lower() not in {
    "0", "false", "no",
}
SESSION_TTL_SECONDS = 12 * 60 * 60
PASSWORD_ITERATIONS = 600_000
MAX_FRAME_LENGTH = 16_384
MAX_BATCH_SIZE = 100
PREDICTION_SAVE_INTERVAL = 5.0
MAX_SAVED_PREDICTIONS = 50_000
ACTIVITY_SESSION_IDLE_SECONDS = 15
_UNKNOWN_USER_SALT = b"csi-web-invalid-user-salt"
ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get(
        "CSI_ALLOWED_HOSTS", "localhost,127.0.0.1"
    ).split(",")
    if host.strip()
]
ALLOWED_ORIGINS = {
    origin.strip().rstrip("/")
    for origin in os.environ.get(
        "CSI_ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
}
ENABLE_API_DOCS = os.environ.get("CSI_ENABLE_API_DOCS", "false").lower() in {
    "1", "true", "yes",
}

if MODEL_KEY not in config.MODEL_KEYS:
    raise RuntimeError(f"CSI_MODEL must be one of: {', '.join(config.MODEL_KEYS)}")


def _connect_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _initialize_database() -> None:
    with _connect_db() as connection:
        connection.executescript(
            """
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
                expires_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS sessions_expiry_idx ON sessions(expires_at);
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities TEXT NOT NULL,
                frame_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                session_id INTEGER REFERENCES activity_sessions(id),
                model_key TEXT NOT NULL DEFAULT 'rf'
            );
            CREATE TABLE IF NOT EXISTS activity_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                last_frame_at TEXT NOT NULL,
                ended_at TEXT,
                total_frames INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS activity_sessions_started_idx
                ON activity_sessions(started_at DESC);
            CREATE TABLE IF NOT EXISTS activity_stats (
                session_id INTEGER NOT NULL REFERENCES activity_sessions(id) ON DELETE CASCADE,
                label TEXT NOT NULL,
                model_key TEXT NOT NULL,
                prediction_count INTEGER NOT NULL,
                confidence_sum REAL NOT NULL,
                probability_sum REAL NOT NULL,
                first_prediction_at TEXT NOT NULL,
                last_prediction_at TEXT NOT NULL,
                PRIMARY KEY (session_id, label, model_key)
            );
            CREATE TABLE IF NOT EXISTS activity_settings_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER REFERENCES activity_sessions(id) ON DELETE SET NULL,
                setting TEXT NOT NULL,
                old_value TEXT NOT NULL,
                new_value TEXT NOT NULL,
                changed_at TEXT NOT NULL
            );
            """
        )
        prediction_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(predictions)")
        }
        if "session_id" not in prediction_columns:
            connection.execute(
                "ALTER TABLE predictions ADD COLUMN session_id INTEGER "
                "REFERENCES activity_sessions(id)"
            )
        if "model_key" not in prediction_columns:
            connection.execute(
                "ALTER TABLE predictions ADD COLUMN model_key TEXT NOT NULL DEFAULT 'rf'"
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS predictions_session_idx "
            "ON predictions(session_id, id DESC)"
        )
        connection.execute(
            "UPDATE activity_sessions SET ended_at = last_frame_at WHERE ended_at IS NULL"
        )
        connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (int(time.time()),))


def _password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )


def _create_user(username: str, password: str) -> None:
    salt = secrets.token_bytes(16)
    with _connect_db() as connection:
        connection.execute(
            "INSERT OR IGNORE INTO users (username, salt, password_hash, created_at) "
            "VALUES (?, ?, ?, ?)",
            (username, salt, _password_hash(password, salt), int(time.time())),
        )


def set_user_password(username: str, password: str) -> None:
    salt = secrets.token_bytes(16)
    with _connect_db() as connection:
        connection.execute(
            "INSERT INTO users (username, salt, password_hash, created_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(username) DO UPDATE SET "
            "salt = excluded.salt, password_hash = excluded.password_hash",
            (username, salt, _password_hash(password, salt), int(time.time())),
        )
        connection.execute("DELETE FROM sessions WHERE username = ?", (username,))


def _authenticate_user(username: str, password: str) -> bool:
    with _connect_db() as connection:
        row = connection.execute(
            "SELECT salt, password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
    if row is None:
        _password_hash(password, _UNKNOWN_USER_SALT)
        return False
    return hmac.compare_digest(_password_hash(password, row["salt"]), row["password_hash"])


def _new_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    with _connect_db() as connection:
        connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (int(time.time()),))
        connection.execute(
            "INSERT INTO sessions (token_hash, username, expires_at) VALUES (?, ?, ?)",
            (
                hashlib.sha256(token.encode("utf-8")).hexdigest(),
                username,
                int(time.time()) + SESSION_TTL_SECONDS,
            ),
        )
    return token


def _session_username(token: str | None) -> str | None:
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with _connect_db() as connection:
        row = connection.execute(
            "SELECT username FROM sessions WHERE token_hash = ? AND expires_at > ?",
            (token_hash, int(time.time())),
        ).fetchone()
    return str(row["username"]) if row else None


def _delete_session(token: str | None) -> None:
    if not token:
        return
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with _connect_db() as connection:
        connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))


def _bootstrap_admin() -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.@-]{3,64}", ADMIN_USERNAME):
        raise RuntimeError(
            "Set CSI_ADMIN_USERNAME to a 3-64 character username using letters, "
            "numbers, '.', '_', '@' or '-'."
        )
    if len(ADMIN_PASSWORD) < 12:
        raise RuntimeError("CSI_ADMIN_PASSWORD must contain at least 12 characters.")
    _create_user(ADMIN_USERNAME, ADMIN_PASSWORD)


class LoginPayload(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class IngestPayload(BaseModel):
    frames: list[str] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class ModelSelectionPayload(BaseModel):
    model: str = Field(min_length=1, max_length=32)


def _load_model_artifacts(model_key: str) -> tuple[object, object, object, list[str]]:
    if model_key not in config.MODEL_KEYS:
        raise ValueError(f"Model must be one of: {', '.join(config.MODEL_KEYS)}.")
    paths = (
        MODEL_DIR / "csi_pipeline.joblib",
        MODEL_DIR / "label_encoder.joblib",
        MODEL_DIR / config.MODEL_FILES[model_key],
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
    metrics_path = MODEL_DIR / "metrics.json"
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


def _model_artifacts_available(model_key: str) -> bool:
    paths = (
        MODEL_DIR / "csi_pipeline.joblib",
        MODEL_DIR / "label_encoder.joblib",
        MODEL_DIR / config.MODEL_FILES[model_key],
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
        self.model_key = MODEL_KEY
        (
            self.pipeline,
            self.label_encoder,
            self.model,
            self.classes,
        ) = _load_model_artifacts(self.model_key)
        self.window_size = config.WINDOW_SIZE
        self.step_size = config.PREDICTION_STEP_SIZE
        self.buffer: deque[np.ndarray] = deque(
            maxlen=self.window_size + config.FILTER_WARMUP
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
        self.lock = asyncio.Lock()
        self.clients: set[WebSocket] = set()

    def models(self) -> dict:
        return {
            "selected": self.model_key,
            "available": [
                key for key in config.MODEL_FILES
                if _model_artifacts_available(key)
            ],
        }

    async def select_model(self, model_key: str) -> dict:
        if model_key not in config.MODEL_KEYS:
            raise ValueError(f"Model must be one of: {', '.join(config.MODEL_KEYS)}.")
        if model_key == self.model_key:
            return self.status()
        loaded = await asyncio.to_thread(_load_model_artifacts, model_key)
        async with self.lock:
            old_model = self.model_key
            changed_at = datetime.now(UTC).isoformat()
            session_id = self.activity_session_id
            _save_setting_change(
                session_id,
                "model",
                old_model,
                model_key,
                changed_at,
            )
            (
                self.pipeline,
                self.label_encoder,
                self.model,
                self.classes,
            ) = loaded
            self.model_key = model_key
            self.buffer.clear()
            self.frames_since_prediction = 0
            self.last_persisted_at = 0.0
            self.latest_prediction = None
            status = self.status()
        change = {
            "setting": "model",
            "old_value": old_model,
            "new_value": model_key,
            "changed_at": changed_at,
            "session_id": session_id,
        }
        await self.broadcast(
            {"type": "settings_changed", "status": status, "change": change}
        )
        return status

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
        with _connect_db() as connection:
            cursor = connection.execute(
                "INSERT INTO activity_sessions (started_at, last_frame_at) VALUES (?, ?)",
                (timestamp, timestamp),
            )
            session_id = int(cursor.lastrowid)
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
                    )
                    self.frames_since_prediction = 0
                    if prediction is not None:
                        label, confidence, probabilities = prediction
                        result = {
                            "label": label,
                            "confidence": round(float(confidence), 2),
                            "probabilities": {
                                name: round(float(value) * 100, 2)
                                for name, value in zip(self.classes, probabilities)
                            },
                            "frame_count": self.frames_received,
                            "session_id": session_id,
                            "model": self.model_key,
                            "created_at": datetime.now(UTC).isoformat(),
                        }
                        self.latest_prediction = result
                        updates.append(result)
                        await asyncio.to_thread(_save_activity_stat, result)
                        now = time.monotonic()
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
        return {
            "device_connected": connected,
            "frames_received": self.frames_received,
            "invalid_frames": self.invalid_frames,
            "buffered_frames": len(self.buffer),
            "buffer_capacity": self.buffer.maxlen,
            "fps": round(fps, 1),
            "model": self.model_key,
            "available_models": self.models()["available"],
            "classes": self.classes,
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


def _save_prediction(prediction: dict) -> None:
    with _connect_db() as connection:
        connection.execute(
            "INSERT INTO predictions "
            "(label, confidence, probabilities, frame_count, created_at, session_id, model_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                prediction["label"],
                prediction["confidence"],
                json.dumps(prediction["probabilities"]),
                prediction["frame_count"],
                prediction["created_at"],
                prediction["session_id"],
                prediction["model"],
            ),
        )
        connection.execute(
            "DELETE FROM predictions WHERE id <= "
            "(SELECT id FROM predictions ORDER BY id DESC LIMIT 1 OFFSET ?)",
            (MAX_SAVED_PREDICTIONS,),
        )


def _save_activity_stat(prediction: dict) -> None:
    label = str(prediction["label"])
    probability = float(prediction["probabilities"].get(label, 0.0))
    with _connect_db() as connection:
        connection.execute(
            """
            INSERT INTO activity_stats (
                session_id, label, model_key, prediction_count,
                confidence_sum, probability_sum,
                first_prediction_at, last_prediction_at
            ) VALUES (?, ?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT(session_id, label, model_key) DO UPDATE SET
                prediction_count = prediction_count + 1,
                confidence_sum = confidence_sum + excluded.confidence_sum,
                probability_sum = probability_sum + excluded.probability_sum,
                last_prediction_at = excluded.last_prediction_at
            """,
            (
                prediction["session_id"],
                label,
                prediction["model"],
                prediction["confidence"],
                probability,
                prediction["created_at"],
                prediction["created_at"],
            ),
        )


def _update_activity_session(
    session_id: int,
    accepted_frames: int,
    last_frame_at: str,
) -> None:
    with _connect_db() as connection:
        connection.execute(
            """
            UPDATE activity_sessions
            SET total_frames = total_frames + ?, last_frame_at = ?
            WHERE id = ? AND ended_at IS NULL
            """,
            (accepted_frames, last_frame_at, session_id),
        )


def _close_activity_session(session_id: int, ended_at: str) -> None:
    with _connect_db() as connection:
        connection.execute(
            "UPDATE activity_sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
            (ended_at, session_id),
        )


def _save_setting_change(
    session_id: int | None,
    setting: str,
    old_value: str,
    new_value: str,
    changed_at: str,
) -> None:
    with _connect_db() as connection:
        connection.execute(
            """
            INSERT INTO activity_settings_changes
                (session_id, setting, old_value, new_value, changed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, setting, old_value, new_value, changed_at),
        )


def _get_predictions(limit: int) -> list[dict]:
    with _connect_db() as connection:
        rows = connection.execute(
            "SELECT label, confidence, probabilities, frame_count, created_at, "
            "session_id, model_key "
            "FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "label": row["label"],
            "confidence": row["confidence"],
            "probabilities": json.loads(row["probabilities"]),
            "frame_count": row["frame_count"],
            "created_at": row["created_at"],
            "session_id": row["session_id"],
            "model": row["model_key"],
        }
        for row in rows
    ]


def _get_activity_sessions(limit: int) -> list[dict]:
    with _connect_db() as connection:
        rows = connection.execute(
            """
            SELECT s.id, s.started_at, s.last_frame_at, s.ended_at, s.total_frames,
                   COALESCE(SUM(st.prediction_count), 0) AS prediction_count
            FROM activity_sessions AS s
            LEFT JOIN activity_stats AS st ON st.session_id = s.id
            GROUP BY s.id
            ORDER BY s.started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    result = []
    for row in rows:
        started_at = str(row["started_at"])
        last_frame_at = str(row["last_frame_at"])
        elapsed = max(
            0.0,
            (
                datetime.fromisoformat(last_frame_at)
                - datetime.fromisoformat(started_at)
            ).total_seconds(),
        )
        result.append(
            {
                "id": int(row["id"]),
                "started_at": started_at,
                "last_frame_at": last_frame_at,
                "ended_at": row["ended_at"],
                "active": row["ended_at"] is None,
                "duration_seconds": round(elapsed, 1),
                "total_frames": int(row["total_frames"]),
                "prediction_count": int(row["prediction_count"]),
            }
        )
    return result


def _get_activity_session(session_id: int) -> dict | None:
    with _connect_db() as connection:
        session = connection.execute(
            """
            SELECT id, started_at, last_frame_at, ended_at, total_frames
            FROM activity_sessions WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if session is None:
            return None
        stats = connection.execute(
            """
            SELECT label, model_key, prediction_count, confidence_sum,
                   probability_sum, first_prediction_at, last_prediction_at
            FROM activity_stats
            WHERE session_id = ?
            ORDER BY label, model_key
            """,
            (session_id,),
        ).fetchall()
        predictions = connection.execute(
            """
            SELECT label, confidence, probabilities, frame_count,
                   created_at, model_key
            FROM predictions
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT 1000
            """,
            (session_id,),
        ).fetchall()
        changes = connection.execute(
            """
            SELECT setting, old_value, new_value, changed_at
            FROM activity_settings_changes
            WHERE session_id = ?
            ORDER BY id
            """,
            (session_id,),
        ).fetchall()

    started_at = str(session["started_at"])
    last_frame_at = str(session["last_frame_at"])
    duration_seconds = max(
        0.0,
        (
            datetime.fromisoformat(last_frame_at)
            - datetime.fromisoformat(started_at)
        ).total_seconds(),
    )
    return {
        "id": int(session["id"]),
        "started_at": started_at,
        "last_frame_at": last_frame_at,
        "ended_at": session["ended_at"],
        "active": session["ended_at"] is None,
        "duration_seconds": round(duration_seconds, 1),
        "total_frames": int(session["total_frames"]),
        "statistics": [
            {
                "label": str(row["label"]),
                "model": str(row["model_key"]),
                "prediction_count": int(row["prediction_count"]),
                "average_confidence": round(
                    float(row["confidence_sum"]) / int(row["prediction_count"]), 2
                ),
                "average_probability": round(
                    float(row["probability_sum"]) / int(row["prediction_count"]), 2
                ),
                "first_prediction_at": str(row["first_prediction_at"]),
                "last_prediction_at": str(row["last_prediction_at"]),
            }
            for row in stats
        ],
        "predictions": [
            {
                "label": str(row["label"]),
                "confidence": float(row["confidence"]),
                "probabilities": json.loads(row["probabilities"]),
                "frame_count": int(row["frame_count"]),
                "created_at": str(row["created_at"]),
                "model": str(row["model_key"]),
            }
            for row in predictions
        ],
        "settings_changes": [
            {
                "setting": str(row["setting"]),
                "old_value": str(row["old_value"]),
                "new_value": str(row["new_value"]),
                "changed_at": str(row["changed_at"]),
            }
            for row in changes
        ],
    }


def _require_user(request: Request) -> str:
    username = _session_username(request.cookies.get("csi_session"))
    if username is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return username


login_attempts: dict[str, deque[float]] = defaultdict(deque)
login_attempts_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if len(EDGE_TOKEN) < 32:
        raise RuntimeError("CSI_EDGE_TOKEN must be at least 32 characters.")
    _initialize_database()
    _bootstrap_admin()
    engine = InferenceEngine()
    app.state.inference_engine = engine
    session_watchdog = asyncio.create_task(_watch_activity_sessions(engine))
    try:
        yield
    finally:
        session_watchdog.cancel()
        try:
            await session_watchdog
        except asyncio.CancelledError:
            pass


async def _watch_activity_sessions(engine: InferenceEngine) -> None:
    while True:
        await asyncio.sleep(1)
        await engine.close_idle_session()


app = FastAPI(
    title="WiFi CSI Human Detection API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if ENABLE_API_DOCS else None,
    redoc_url=None,
    openapi_url="/openapi.json" if ENABLE_API_DOCS else None,
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def add_security_headers(request: Request, call_next) -> Response:
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "connect-src 'self' ws: wss:; img-src 'self' data:; "
        "base-uri 'self'; form-action 'self'; frame-ancestors 'none'; object-src 'none'",
    )
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    elif request.url.path.startswith("/static/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


if not ALLOWED_HOSTS:
    raise RuntimeError("CSI_ALLOWED_HOSTS must include at least one allowed hostname.")


def _get_engine(request: Request) -> InferenceEngine:
    return request.app.state.inference_engine


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readyz(engine: Annotated[InferenceEngine, Depends(_get_engine)]) -> dict:
    return {"status": "ready", "model": engine.status()["model"]}


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/auth/login")
async def login(
    payload: LoginPayload,
    request: Request,
    response: Response,
) -> dict:
    remote = request.client.host if request.client else "unknown"
    now = time.monotonic()
    async with login_attempts_lock:
        if remote not in login_attempts and len(login_attempts) >= 10_000:
            for address, attempts in tuple(login_attempts.items()):
                while attempts and now - attempts[0] > 60:
                    attempts.popleft()
                if not attempts:
                    login_attempts.pop(address, None)
            if len(login_attempts) >= 10_000:
                raise HTTPException(status_code=503, detail="Login service is busy.")
        attempts = login_attempts[remote]
        while attempts and now - attempts[0] > 60:
            attempts.popleft()
        if len(attempts) >= 5:
            raise HTTPException(status_code=429, detail="Too many login attempts.")

    if not await asyncio.to_thread(_authenticate_user, payload.username, payload.password):
        async with login_attempts_lock:
            login_attempts[remote].append(now)
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    async with login_attempts_lock:
        login_attempts.pop(remote, None)
    token = await asyncio.to_thread(_new_session, payload.username)
    response.set_cookie(
        "csi_session",
        token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="strict",
        path="/",
    )
    return {"username": payload.username}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response, _: Annotated[str, Depends(_require_user)]) -> dict:
    _delete_session(request.cookies.get("csi_session"))
    response.delete_cookie(
        "csi_session", path="/", secure=COOKIE_SECURE, httponly=True, samesite="strict"
    )
    return {"status": "ok"}


@app.get("/api/auth/me")
def current_user(username: Annotated[str, Depends(_require_user)]) -> dict:
    return {"username": username}


@app.get("/api/status")
def get_status(
    _: Annotated[str, Depends(_require_user)],
    engine: Annotated[InferenceEngine, Depends(_get_engine)],
) -> dict:
    return engine.status()


@app.get("/api/models")
def get_models(
    _: Annotated[str, Depends(_require_user)],
    engine: Annotated[InferenceEngine, Depends(_get_engine)],
) -> dict:
    return engine.models()


@app.post("/api/settings/model")
async def set_model(
    payload: ModelSelectionPayload,
    _: Annotated[str, Depends(_require_user)],
    engine: Annotated[InferenceEngine, Depends(_get_engine)],
) -> dict:
    try:
        return await engine.select_model(payload.model.strip().lower())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/edge/model")
async def set_edge_model(
    payload: ModelSelectionPayload,
    engine: Annotated[InferenceEngine, Depends(_get_engine)],
    authorization: Annotated[str | None, Header()] = None,
) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required.")
    supplied_token = authorization.removeprefix("Bearer ")
    if not hmac.compare_digest(supplied_token, EDGE_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid edge token.")
    try:
        status = await engine.select_model(payload.model.strip().lower())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"model": status["model"]}


@app.get("/api/activity-sessions")
def get_activity_sessions(
    limit: int = 100,
    _: str = Depends(_require_user),
) -> list[dict]:
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500.")
    return _get_activity_sessions(limit)


@app.get("/api/activity-sessions/{session_id}")
def get_activity_session(
    session_id: int,
    _: str = Depends(_require_user),
) -> dict:
    if session_id < 1:
        raise HTTPException(status_code=422, detail="session_id must be positive.")
    session = _get_activity_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Activity session not found.")
    return session


@app.get("/api/predictions")
def get_prediction_history(
    limit: int = 50,
    _: str = Depends(_require_user),
) -> list[dict]:
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200.")
    return _get_predictions(limit)


@app.post("/api/ingest")
async def ingest(
    payload: IngestPayload,
    engine: Annotated[InferenceEngine, Depends(_get_engine)],
    authorization: Annotated[str | None, Header()] = None,
) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required.")
    supplied_token = authorization.removeprefix("Bearer ")
    if not hmac.compare_digest(supplied_token, EDGE_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid edge token.")
    if any(len(line) > MAX_FRAME_LENGTH for line in payload.frames):
        raise HTTPException(status_code=413, detail="CSI frame exceeds the size limit.")
    return await engine.process_batch(payload.frames)


@app.websocket("/api/events")
async def events(websocket: WebSocket) -> None:
    origin = websocket.headers.get("origin")
    if not origin or origin.rstrip("/") not in ALLOWED_ORIGINS:
        await websocket.close(code=4403, reason="Origin is not allowed.")
        return
    token = websocket.cookies.get("csi_session")
    username = await asyncio.to_thread(_session_username, token)
    if username is None:
        await websocket.close(code=4401, reason="Authentication required.")
        return
    await websocket.accept()
    engine: InferenceEngine = websocket.app.state.inference_engine
    engine.clients.add(websocket)
    try:
        await websocket.send_json({"type": "update", "status": engine.status(), "predictions": []})
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=60)
            except TimeoutError:
                if await asyncio.to_thread(_session_username, token) is None:
                    await websocket.close(code=4401, reason="Session expired.")
                    return
                await websocket.send_json(
                    {"type": "update", "status": engine.status(), "predictions": []}
                )
    except WebSocketDisconnect:
        pass
    finally:
        engine.clients.discard(websocket)
