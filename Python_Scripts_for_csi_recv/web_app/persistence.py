from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import config

APP_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("CSI_DB_PATH", APP_DIR / "data" / "app.sqlite3"))
MODEL_KEY = os.environ.get("CSI_MODEL", "rf").lower()
MODE_KEY = os.environ.get("CSI_MODE", "har").lower()
ADMIN_USERNAME = os.environ.get("CSI_ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.environ.get("CSI_ADMIN_PASSWORD", "")
SESSION_TTL_SECONDS = 12 * 60 * 60
PASSWORD_ITERATIONS = 600_000
MAX_SAVED_PREDICTIONS = 50_000
_UNKNOWN_USER_SALT = b"csi-web-invalid-user-salt"

__all__ = [
    "DB_PATH",
    "SESSION_TTL_SECONDS",
    "_authenticate_user",
    "_bootstrap_admin",
    "_close_activity_session",
    "_connect_db",
    "_create_activity_session",
    "_delete_session",
    "_get_activity_session",
    "_get_activity_sessions",
    "_get_predictions",
    "_get_runtime_settings",
    "_initialize_database",
    "_new_session",
    "_save_activity_stat",
    "_save_prediction",
    "_save_runtime_setting",
    "_save_setting_change",
    "_session_username",
    "_update_activity_session",
    "set_user_password",
]


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
            CREATE TABLE IF NOT EXISTS runtime_settings (
                setting TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT OR IGNORE INTO runtime_settings (setting, value) VALUES (?, ?)",
            [
                ("model", MODEL_KEY),
                ("mode", MODE_KEY),
                ("window_size", str(config.WINDOW_SIZE)),
                ("step_size", str(config.PREDICTION_STEP_SIZE)),
                ("warmup_frames", str(config.FILTER_WARMUP)),
                ("cutoff", str(config.FILTER_CUTOFF_HZ)),
                ("ema_alpha", str(config.DASHBOARD_EMA_ALPHA)),
                ("conf_thresh", str(config.DASHBOARD_CONF_THRESH)),
                ("hyst_count", str(config.DASHBOARD_HYST_COUNT)),
                ("motion_threshold", "0.10"),
            ],
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
        connection.execute(
            "DELETE FROM sessions WHERE expires_at <= ?", (int(time.time()),)
        )


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
    return hmac.compare_digest(
        _password_hash(password, row["salt"]), row["password_hash"]
    )


def _new_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    with _connect_db() as connection:
        connection.execute(
            "DELETE FROM sessions WHERE expires_at <= ?", (int(time.time()),)
        )
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
    with _connect_db() as connection:
        existing_user = connection.execute(
            "SELECT 1 FROM users WHERE username = ?", (ADMIN_USERNAME,)
        ).fetchone()
    if existing_user is not None:
        return
    if len(ADMIN_PASSWORD) < 12:
        raise RuntimeError("CSI_ADMIN_PASSWORD must contain at least 12 characters.")
    _create_user(ADMIN_USERNAME, ADMIN_PASSWORD)


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


def _create_activity_session(timestamp: str) -> int:
    with _connect_db() as connection:
        cursor = connection.execute(
            "INSERT INTO activity_sessions (started_at, last_frame_at) VALUES (?, ?)",
            (timestamp, timestamp),
        )
        return int(cursor.lastrowid)


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


def _get_runtime_settings() -> dict[str, str]:
    with _connect_db() as connection:
        rows = connection.execute(
            "SELECT setting, value FROM runtime_settings"
        ).fetchall()
    return {str(row["setting"]): str(row["value"]) for row in rows}


def _save_runtime_setting(setting: str, value: str) -> None:
    with _connect_db() as connection:
        connection.execute(
            "INSERT INTO runtime_settings (setting, value) VALUES (?, ?) "
            "ON CONFLICT(setting) DO UPDATE SET value = excluded.value",
            (setting, value),
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
