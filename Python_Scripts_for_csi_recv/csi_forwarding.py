from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from urllib.parse import urlsplit


def validate_server_url(server_url: str) -> None:
    parsed = urlsplit(server_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Server URL must start with http:// or https://.")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Server URL must not contain credentials or a URL fragment.")
    if parsed.scheme != "https" and parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError("Use HTTPS for non-local API servers.")


def _send_batch(server_url: str, token: str, frames: Sequence[str]) -> int:
    result = _request_json(
        server_url,
        token,
        "/api/ingest",
        payload={"frames": list(frames)},
        timeout=15,
    )
    return int(result.get("accepted", 0))


def _send_model(server_url: str, token: str, model: str) -> str:
    result = _request_json(
        server_url,
        token,
        "/api/edge/model",
        payload={"model": model},
        timeout=15,
    )
    return str(result["model"])


def _request_json(
    server_url: str,
    token: str,
    path: str,
    *,
    payload: dict | None = None,
    timeout: float = 5.0,
) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{server_url.rstrip('/')}{path}",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"CSI API returned HTTP {response.status}.")
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise RuntimeError("CSI API returned an invalid JSON object.")
    return result


def _send_settings(server_url: str, token: str, settings: dict) -> dict:
    return _request_json(
        server_url,
        token,
        "/api/edge/settings",
        payload=settings,
        timeout=15,
    )


def get_edge_status(server_url: str, token: str) -> dict:
    return _request_json(
        server_url,
        token,
        "/api/edge/status",
        timeout=5,
    )


def send_with_retry(
    server_url: str,
    token: str,
    frames: Sequence[str],
) -> int:
    delay = 1.0
    while True:
        try:
            return _send_batch(server_url, token, frames)
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 401, 403, 413, 422):
                raise RuntimeError(
                    f"CSI API rejected the batch (HTTP {exc.code})."
                ) from exc
            print(f"CSI API returned HTTP {exc.code}; retrying in {delay:.0f}s.")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print(f"CSI API unavailable ({exc}); retrying in {delay:.0f}s.")
        time.sleep(delay)
        delay = min(delay * 2, 10.0)


def send_model_with_retry(server_url: str, token: str, model: str) -> str:
    delay = 1.0
    while True:
        try:
            return _send_model(server_url, token, model)
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 401, 403, 404, 413, 422):
                raise RuntimeError(
                    f"CSI API rejected the model update (HTTP {exc.code})."
                ) from exc
            print(f"CSI API returned HTTP {exc.code}; retrying in {delay:.0f}s.")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print(f"CSI API unavailable ({exc}); retrying in {delay:.0f}s.")
        time.sleep(delay)
        delay = min(delay * 2, 10.0)


def send_settings_with_retry(
    server_url: str,
    token: str,
    settings: dict,
) -> dict:
    delay = 1.0
    for attempt in range(5):
        try:
            return _send_settings(server_url, token, settings)
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 401, 403, 404, 413, 422):
                raise RuntimeError(
                    f"CSI API rejected the settings update (HTTP {exc.code})."
                ) from exc
            if attempt == 4:
                raise RuntimeError(
                    "CSI API settings update failed after 5 attempts."
                ) from exc
            print(f"CSI API returned HTTP {exc.code}; retrying in {delay:.0f}s.")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == 4:
                raise RuntimeError(
                    "CSI API settings update failed after 5 attempts."
                ) from exc
            print(f"CSI API unavailable ({exc}); retrying in {delay:.0f}s.")
        time.sleep(delay)
        delay = min(delay * 2, 10.0)
    raise RuntimeError("CSI API settings update failed after 5 attempts.")
