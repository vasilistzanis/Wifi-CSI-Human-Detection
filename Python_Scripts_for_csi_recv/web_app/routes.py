from __future__ import annotations

import asyncio
import hmac
import os
import time
from collections import defaultdict, deque
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
    Response,
    WebSocket,
)
from starlette.websockets import WebSocketDisconnect

from web_app.inference import MAX_FRAME_LENGTH, InferenceEngine
from web_app.persistence import (
    SESSION_TTL_SECONDS,
    _authenticate_user,
    _delete_session,
    _get_activity_session,
    _get_activity_sessions,
    _get_predictions,
    _new_session,
    _session_username,
)
from web_app.schemas import (
    IngestPayload,
    LoginPayload,
    ModelSelectionPayload,
    RuntimeSettingsPayload,
)

EDGE_TOKEN = os.environ.get("CSI_EDGE_TOKEN", "")
COOKIE_SECURE = os.environ.get("CSI_COOKIE_SECURE", "true").lower() not in {
    "0",
    "false",
    "no",
}
ALLOWED_ORIGINS = {
    origin.strip().rstrip("/")
    for origin in os.environ.get(
        "CSI_ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
}

router = APIRouter()


def _get_engine(request: Request) -> InferenceEngine:
    return request.app.state.inference_engine


def _require_user(request: Request) -> str:
    username = _session_username(request.cookies.get("csi_session"))
    if username is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return username


def _require_edge_token(authorization: str | None) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required.")
    supplied_token = authorization.removeprefix("Bearer ")
    if not hmac.compare_digest(supplied_token, EDGE_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid edge token.")


login_attempts: dict[str, deque[float]] = defaultdict(deque)
login_attempts_lock = asyncio.Lock()


@router.post("/api/auth/login")
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


@router.post("/api/auth/logout")
def logout(request: Request, response: Response, _: Annotated[str, Depends(_require_user)]) -> dict:
    _delete_session(request.cookies.get("csi_session"))
    response.delete_cookie(
        "csi_session", path="/", secure=COOKIE_SECURE, httponly=True, samesite="strict"
    )
    return {"status": "ok"}


@router.get("/api/auth/me")
def current_user(username: Annotated[str, Depends(_require_user)]) -> dict:
    return {"username": username}


@router.get("/api/status")
def get_status(
    _: Annotated[str, Depends(_require_user)],
    engine: Annotated[InferenceEngine, Depends(_get_engine)],
) -> dict:
    return engine.status()


@router.get("/api/models")
def get_models(
    _: Annotated[str, Depends(_require_user)],
    engine: Annotated[InferenceEngine, Depends(_get_engine)],
) -> dict:
    return engine.models()


@router.post("/api/settings/model")
async def set_model(
    payload: ModelSelectionPayload,
    _: Annotated[str, Depends(_require_user)],
    engine: Annotated[InferenceEngine, Depends(_get_engine)],
) -> dict:
    try:
        return await engine.select_model(payload.model.strip().lower())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/api/edge/model")
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


@router.get("/api/edge/status")
def get_edge_status(
    engine: Annotated[InferenceEngine, Depends(_get_engine)],
    authorization: Annotated[str | None, Header()] = None,
) -> dict:
    _require_edge_token(authorization)
    return engine.status()


@router.post("/api/edge/settings")
async def update_edge_settings(
    payload: RuntimeSettingsPayload,
    engine: Annotated[InferenceEngine, Depends(_get_engine)],
    authorization: Annotated[str | None, Header()] = None,
) -> dict:
    _require_edge_token(authorization)
    try:
        return await engine.update_settings(payload.model_dump(exclude_none=True))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/api/activity-sessions")
def get_activity_sessions(
    limit: int = 100,
    _: str = Depends(_require_user),
) -> list[dict]:
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500.")
    return _get_activity_sessions(limit)


@router.get("/api/activity-sessions/{session_id}")
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


@router.get("/api/predictions")
def get_prediction_history(
    limit: int = 50,
    _: str = Depends(_require_user),
) -> list[dict]:
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200.")
    return _get_predictions(limit)


@router.post("/api/ingest")
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


@router.websocket("/api/events")
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
