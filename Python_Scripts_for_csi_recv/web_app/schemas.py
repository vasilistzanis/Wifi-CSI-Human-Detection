from __future__ import annotations

from pydantic import BaseModel, Field

MAX_BATCH_SIZE = 100


class LoginPayload(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class IngestPayload(BaseModel):
    frames: list[str] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class ModelSelectionPayload(BaseModel):
    model: str = Field(min_length=1, max_length=32)


class RuntimeSettingsPayload(BaseModel):
    model: str | None = Field(default=None, min_length=1, max_length=32)
    mode: str | None = Field(default=None, min_length=1, max_length=16)
    window_size: int | None = Field(default=None, ge=10, le=1000)
    step_size: int | None = Field(default=None, ge=1, le=1000)
    warmup_frames: int | None = Field(default=None, ge=0, le=1000)
    cutoff: float | None = Field(default=None, gt=0, le=50)
    ema_alpha: float | None = Field(default=None, gt=0, le=1)
    conf_thresh: float | None = Field(default=None, ge=0, le=100)
    hyst_count: int | None = Field(default=None, ge=1, le=100)
    motion_threshold: float | None = Field(default=None, gt=0, le=1)
