from __future__ import annotations

import json
import urllib.request

import pytest

import csi_forwarding


def test_validate_server_url_requires_https_for_remote_hosts() -> None:
    csi_forwarding.validate_server_url("http://127.0.0.1:8000")
    csi_forwarding.validate_server_url("https://csi.example.com")

    with pytest.raises(ValueError, match="Use HTTPS"):
        csi_forwarding.validate_server_url("http://csi.example.com")


def test_send_batch_posts_frames_with_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def read() -> bytes:
            return b'{"accepted": 2}'

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> Response:
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    accepted = csi_forwarding._send_batch(
        "http://127.0.0.1:8000/",
        "test-edge-token",
        ["CSI_DATA,one", "CSI_DATA,two"],
    )

    assert accepted == 2
    assert captured == {
        "url": "http://127.0.0.1:8000/api/ingest",
        "authorization": "Bearer test-edge-token",
        "payload": {"frames": ["CSI_DATA,one", "CSI_DATA,two"]},
        "timeout": 15,
    }


def test_send_model_posts_model_with_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def read() -> bytes:
            return b'{"model": "svm"}'

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> Response:
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    model = csi_forwarding._send_model(
        "http://127.0.0.1:8000/",
        "test-edge-token",
        "svm",
    )

    assert model == "svm"
    assert captured["url"] == "http://127.0.0.1:8000/api/edge/model"
    assert captured["authorization"] == "Bearer test-edge-token"
    assert captured["payload"] == {"model": "svm"}
    assert captured["timeout"] == 15
