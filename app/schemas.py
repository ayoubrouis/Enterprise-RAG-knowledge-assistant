"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)


class Source(BaseModel):
    source: str
    page: int | None = None
    similarity: float
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    grounded: bool = True  # False when the grounding guardrail replaced the answer


class StatsResponse(BaseModel):
    tenant_id: str
    documents: int
    chunks: int


class IngestResponse(BaseModel):
    tenant_id: str
    documents: int
    chunks: int
    saved_to: str


class JobStatus(BaseModel):
    tenant_id: str
    status: str  # idle | queued | running | done | failed
    documents: int = 0
    chunks: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    saved_to: str | None = None


class DocumentInfo(BaseModel):
    filename: str
    size: int


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=8, max_length=256)


class ChangePasswordResponse(BaseModel):
    token: str
    token_type: str = "bearer"
    username: str
    role: str
    tenant_id: str


class ReadinessResponse(BaseModel):
    status: str  # ready | degraded
    checks: dict[str, bool]


class SetupStatus(BaseModel):
    needed: bool


class SetupRequest(BaseModel):
    tenant_name: str = Field(min_length=1, max_length=128)
    username: str = Field(min_length=3, max_length=128, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=256)


class LoginResponse(BaseModel):
    token: str
    token_type: str
    username: str
    role: str
    tenant_id: str
    tenant_name: str


class MeResponse(BaseModel):
    username: str
    role: str
    tenant_id: str
    via: str


class TenantCreate(BaseModel):
    tenant_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    name: str = Field(min_length=1, max_length=128)


class TenantUpdate(BaseModel):
    is_active: bool


class TenantOut(BaseModel):
    tenant_id: str
    name: str
    created_at: float
    is_active: bool
    users: int = 0
    documents: int = 0
    api_keys: int = 0


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    # Empty password => a strong random one is generated and returned once.
    password: str = Field(default="", max_length=256)
    role: str = Field(default="user", pattern=r"^(superadmin|admin|user)$")


class UserUpdate(BaseModel):
    is_active: bool


class UserOut(BaseModel):
    id: int
    tenant_id: str
    username: str
    role: str
    is_active: bool
    created_at: float


class UserCreated(UserOut):
    # Populated only when the password was auto-generated (shown once).
    password: str | None = None


class ApiKeyCreate(BaseModel):
    label: str = Field(default="", max_length=128)


class ApiKeyUpdate(BaseModel):
    is_active: bool


class ApiKeyOut(BaseModel):
    key_hash: str
    label: str
    created_at: float
    is_active: bool


class ApiKeyCreated(BaseModel):
    tenant_id: str
    label: str
    key: str
    note: str = "Copy this key now; it is shown only once."


class AuditLogOut(BaseModel):
    id: int
    created_at: float
    actor: str
    actor_role: str | None
    tenant_id: str | None
    action: str
    detail: str | None


class QueryLogOut(BaseModel):
    id: int
    tenant_id: str
    username: str | None
    question: str
    answer: str | None
    latency_ms: float | None
    created_at: float
