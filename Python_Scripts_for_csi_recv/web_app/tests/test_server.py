from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

os.environ.setdefault("CSI_ADMIN_USERNAME", "test-admin")
os.environ.setdefault("CSI_ADMIN_PASSWORD", "test-password-is-long-enough")
os.environ.setdefault("CSI_EDGE_TOKEN", "test-only-edge-token-with-at-least-32-characters")
os.environ.setdefault("CSI_COOKIE_SECURE", "false")
os.environ.setdefault("CSI_ALLOWED_HOSTS", "testserver")
os.environ.setdefault("CSI_ALLOWED_ORIGINS", "http://testserver")

from web_app import server


class FakeInferenceEngine:
    def __init__(self) -> None:
        self.clients = set()
        self.model_key = "rf"
        self.activity_session_id = None
        self.events = []

    def status(self) -> dict:
        return {
            "device_connected": False,
            "frames_received": 0,
            "invalid_frames": 0,
            "buffered_frames": 0,
            "buffer_capacity": 150,
            "fps": 0.0,
            "model": self.model_key,
            "available_models": ["rf", "svm"],
            "classes": ["no_activity", "walk_activity"],
            "window_size": 100,
            "session_id": self.activity_session_id,
            "latest_prediction": None,
        }

    def models(self) -> dict:
        return {"selected": self.model_key, "available": ["rf", "svm"]}

    async def select_model(self, model: str) -> dict:
        if model not in {"rf", "svm"}:
            raise ValueError(f"Unsupported model: {model}")
        old_model = self.model_key
        self.model_key = model
        status = self.status()
        await self.broadcast(
            {
                "type": "settings_changed",
                "status": status,
                "change": {
                    "setting": "model",
                    "old_value": old_model,
                    "new_value": model,
                },
            }
        )
        return status

    async def process_batch(self, frames: list[str]) -> dict:
        return {"accepted": len(frames), **self.status()}

    async def close_idle_session(self) -> bool:
        return False

    async def broadcast(self, event: dict) -> None:
        self.events.append(event)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server, "DB_PATH", tmp_path / "test.sqlite3")
    monkeypatch.setattr(server, "InferenceEngine", FakeInferenceEngine)
    with TestClient(server.app) as test_client:
        yield test_client


def test_health_and_static_dashboard(client: TestClient) -> None:
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "/static/app.js" in dashboard.text
    assert "Activity sessions" in dashboard.text
    assert 'id="model"' in dashboard.text
    assert "model-select" not in dashboard.text
    assert dashboard.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in dashboard.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200


def test_authentication_protects_api_and_logout(client: TestClient) -> None:
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/predictions").status_code == 401

    invalid = client.post(
        "/api/auth/login",
        json={"username": "test-admin", "password": "wrong-password"},
    )
    assert invalid.status_code == 401

    login = client.post(
        "/api/auth/login",
        json={"username": "test-admin", "password": "test-password-is-long-enough"},
    )
    assert login.status_code == 200
    assert client.get("/api/auth/me").json() == {"username": "test-admin"}
    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.headers["cache-control"] == "no-store"
    assert status.json()["model"] == "rf"

    logout = client.post("/api/auth/logout")
    assert logout.status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_ingest_requires_bearer_token(client: TestClient) -> None:
    payload = {"frames": ["a CSI frame"]}
    assert client.post("/api/ingest", json=payload).status_code == 401
    assert client.post(
        "/api/ingest",
        json=payload,
        headers={"Authorization": "Bearer incorrect-token"},
    ).status_code == 401

    response = client.post(
        "/api/ingest",
        json=payload,
        headers={"Authorization": f"Bearer {server.EDGE_TOKEN}"},
    )
    assert response.status_code == 200
    assert response.json()["accepted"] == 1


def test_authenticated_websocket_checks_origin_and_session(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/api/events", headers={"origin": "http://attacker.invalid"}
        ):
            pass

    client.post(
        "/api/auth/login",
        json={"username": "test-admin", "password": "test-password-is-long-enough"},
    )
    with client.websocket_connect(
        "/api/events", headers={"origin": "http://testserver"}
    ) as websocket:
        event = websocket.receive_json()
        assert event["type"] == "update"
        assert event["status"]["model"] == "rf"


def test_password_reset_revokes_existing_sessions(
    client: TestClient,
) -> None:
    login = client.post(
        "/api/auth/login",
        json={"username": "test-admin", "password": "test-password-is-long-enough"},
    )
    assert login.status_code == 200
    assert client.get("/api/auth/me").status_code == 200

    server.set_user_password("test-admin", "another-password-is-long-enough")
    assert client.get("/api/auth/me").status_code == 401
    assert client.post(
        "/api/auth/login",
        json={"username": "test-admin", "password": "test-password-is-long-enough"},
    ).status_code == 401
    assert client.post(
        "/api/auth/login",
        json={"username": "test-admin", "password": "another-password-is-long-enough"},
    ).status_code == 200


def test_model_selection_is_authenticated_and_updates_status(client: TestClient) -> None:
    assert client.get("/api/models").status_code == 401
    assert client.post("/api/settings/model", json={"model": "svm"}).status_code == 401

    client.post(
        "/api/auth/login",
        json={"username": "test-admin", "password": "test-password-is-long-enough"},
    )
    models = client.get("/api/models")
    assert models.json() == {"selected": "rf", "available": ["rf", "svm"]}

    changed = client.post("/api/settings/model", json={"model": "svm"})
    assert changed.status_code == 200
    assert changed.json()["model"] == "svm"
    assert client.get("/api/models").json()["selected"] == "svm"
    assert client.app.state.inference_engine.events[-1]["type"] == "settings_changed"
    assert client.post("/api/settings/model", json={"model": "unknown"}).status_code == 422


def test_edge_model_sync_requires_edge_token_and_broadcasts_change(
    client: TestClient,
) -> None:
    assert client.post(
        "/api/edge/model",
        json={"model": "svm"},
        headers={"Authorization": "Bearer wrong-token"},
    ).status_code == 401

    response = client.post(
        "/api/edge/model",
        json={"model": "svm"},
        headers={"Authorization": f"Bearer {server.EDGE_TOKEN}"},
    )
    assert response.status_code == 200
    assert response.json() == {"model": "svm"}
    assert client.app.state.inference_engine.events[-1]["type"] == "settings_changed"
    assert client.app.state.inference_engine.events[-1]["status"]["model"] == "svm"
    assert client.post(
        "/api/edge/model",
        json={"model": "unknown"},
        headers={"Authorization": f"Bearer {server.EDGE_TOKEN}"},
    ).status_code == 422


def test_activity_session_closes_after_fifteen_idle_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = object.__new__(server.InferenceEngine)
    engine.lock = asyncio.Lock()
    engine.activity_session_id = 7
    engine.last_frame_monotonic = time.monotonic() - 14
    last_frame_at = datetime.now(UTC).isoformat()
    engine.last_frame_at = last_frame_at
    engine.clients = set()
    engine.status = lambda: {"session_id": engine.activity_session_id}
    closed_sessions: list[tuple[int, str]] = []
    monkeypatch.setattr(
        server,
        "_close_activity_session",
        lambda session_id, ended_at: closed_sessions.append((session_id, ended_at)),
    )

    assert not asyncio.run(engine.close_idle_session())
    assert closed_sessions == []

    engine.last_frame_monotonic = time.monotonic() - 16
    assert asyncio.run(engine.close_idle_session())
    assert closed_sessions == [(7, last_frame_at)]


def test_activity_session_detail_contains_per_class_and_model_statistics(
    client: TestClient,
) -> None:
    now = datetime.now(UTC)
    started_at = now.isoformat()
    later = (now + timedelta(seconds=12)).isoformat()
    with server._connect_db() as connection:
        cursor = connection.execute(
            "INSERT INTO activity_sessions "
            "(started_at, last_frame_at, total_frames) VALUES (?, ?, ?)",
            (started_at, later, 1200),
        )
        session_id = int(cursor.lastrowid)
    server._save_activity_stat(
        {
            "session_id": session_id,
            "label": "walk_activity",
            "model": "rf",
            "confidence": 88.0,
            "probabilities": {"walk_activity": 88.0, "no_activity": 12.0},
            "created_at": later,
        }
    )
    server._save_activity_stat(
        {
            "session_id": session_id,
            "label": "no_activity",
            "model": "svm",
            "confidence": 91.0,
            "probabilities": {"walk_activity": 9.0, "no_activity": 91.0},
            "created_at": later,
        }
    )
    server._save_prediction(
        {
            "session_id": session_id,
            "label": "walk_activity",
            "model": "rf",
            "confidence": 88.0,
            "probabilities": {"walk_activity": 88.0, "no_activity": 12.0},
            "frame_count": 1200,
            "created_at": later,
        }
    )
    server._save_setting_change(session_id, "model", "rf", "svm", later)

    assert client.get("/api/activity-sessions").status_code == 401
    client.post(
        "/api/auth/login",
        json={"username": "test-admin", "password": "test-password-is-long-enough"},
    )
    sessions = client.get("/api/activity-sessions")
    assert sessions.status_code == 200
    assert sessions.json()[0]["id"] == session_id
    assert sessions.json()[0]["total_frames"] == 1200

    detail = client.get(f"/api/activity-sessions/{session_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert {item["label"] for item in payload["statistics"]} == {
        "walk_activity",
        "no_activity",
    }
    assert {item["model"] for item in payload["statistics"]} == {"rf", "svm"}
    assert payload["predictions"][0]["model"] == "rf"
    assert payload["settings_changes"][0]["new_value"] == "svm"
    assert client.get("/api/activity-sessions/9999").status_code == 404
