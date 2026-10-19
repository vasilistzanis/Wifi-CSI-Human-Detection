from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    Request,
    Response,
)
from fastapi.responses import FileResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles

from web_app.inference import InferenceEngine
from web_app.persistence import (
    _bootstrap_admin,
    _initialize_database,
)
from web_app.routes import EDGE_TOKEN, router as api_router

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get(
        "CSI_ALLOWED_HOSTS", "localhost,127.0.0.1"
    ).split(",")
    if host.strip()
]
ENABLE_API_DOCS = os.environ.get("CSI_ENABLE_API_DOCS", "false").lower() in {
    "1", "true", "yes",
}


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
app.include_router(api_router)


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
