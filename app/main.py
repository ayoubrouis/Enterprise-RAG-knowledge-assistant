"""FastAPI backend for the Enterprise RAG Knowledge Assistant.

Multi-tenant: every request is scoped to the tenant resolved from its
credentials (Bearer token or API key). Tenants are fully isolated on disk
(``data/tenants/<tenant_id>/``).

Run with:
    uvicorn app.main:app --reload

Interactive API docs are available at http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import secrets
import shutil
import threading
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import db
from app.auth import (
    AuthContext,
    ensure_tenant_access,
    get_auth_context,
    require_admin,
    require_superadmin,
)
from app.config import settings
from app.rag.ingestion import ingest_documents
from app.rag.pipeline import RAGPipeline
from app.rag.vectorstore import build_vectorstore, save_vectorstore, store_exists
from app.schemas import (
    ApiKeyCreated,
    ApiKeyCreate,
    ApiKeyOut,
    DocumentInfo,
    IngestResponse,
    LoginRequest,
    LoginResponse,
    MeResponse,
    QueryLogOut,
    QueryRequest,
    QueryResponse,
    SetupRequest,
    SetupStatus,
    StatsResponse,
    TenantCreate,
    TenantOut,
    TenantUpdate,
    UserCreate,
    UserCreated,
    UserOut,
    UserUpdate,
)
from app.security import sign_token, verify_password

@asynccontextmanager
async def lifespan(_: FastAPI):
    db.seed_defaults()
    yield


app = FastAPI(
    title="Enterprise RAG Knowledge Assistant",
    description="Local-first, multi-tenant retrieval-augmented generation over "
    "your documents.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pipeline registry (bounded LRU cache - one pipeline per tenant, evicted)
# ---------------------------------------------------------------------------

class _PipelineCache:
    """Thread-safe LRU cache for per-tenant RAGPipeline objects.

    Each pipeline holds that tenant's FAISS index in memory, so the cache is
    bounded (settings.PIPELINE_CACHE_SIZE): the least recently used tenant's
    index is dropped and reloaded from disk on next use. Construction is
    serialized per tenant under that tenant's lock.
    """

    def __init__(self, max_size: int = 8, factory=None) -> None:
        self.max_size = max(1, max_size)
        self._factory = factory or (lambda tid: RAGPipeline(tenant_id=tid))
        self._data: OrderedDict[str, RAGPipeline] = OrderedDict()
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def lock_for(self, tenant_id: str) -> threading.Lock:
        """The per-tenant construction lock (shared with re-index operations so
        a re-index and a lazy load of the old index cannot race)."""
        with self._guard:
            return self._locks.setdefault(tenant_id, threading.Lock())

    def get_or_create(self, tenant_id: str) -> RAGPipeline:
        lock = self.lock_for(tenant_id)
        with self._guard:
            cached = self._data.get(tenant_id)
            if cached is not None:
                self._data.move_to_end(tenant_id)  # mark most-recently-used
                return cached
        with lock:
            # Double-checked: a concurrent thread may have built it meanwhile.
            with self._guard:
                cached = self._data.get(tenant_id)
                if cached is not None:
                    self._data.move_to_end(tenant_id)
                    return cached
                pipeline = self._factory(tenant_id)
                self._data[tenant_id] = pipeline
                self._evict_locked()
                return pipeline

    def invalidate(self, tenant_id: str) -> None:
        """Drop a tenant's cached pipeline (e.g. after a re-index) so the next
        request reloads it from disk."""
        with self._guard:
            self._data.pop(tenant_id, None)

    def _evict_locked(self) -> None:
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)  # evict least-recently-used

    def __len__(self) -> int:
        with self._guard:
            return len(self._data)


_pipeline_cache = _PipelineCache(max_size=settings.PIPELINE_CACHE_SIZE)


# ---------------------------------------------------------------------------
# Query rate limiter (sliding window, per authenticated caller)
# ---------------------------------------------------------------------------

class _SlidingWindowRateLimiter:
    """Thread-safe sliding-window limiter keyed by caller identity.

    Each key keeps a deque of hit timestamps; hits older than the window are
    dropped on access, so memory is bounded by the number of active callers.
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max(1, max_requests)
        self.window = max(1.0, float(window_seconds))
        self._hits: dict[str, deque[float]] = {}
        self._guard = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        """Record a hit; return False (without recording) when over the limit."""
        now = time.monotonic() if now is None else now
        cutoff = now - self.window
        with self._guard:
            q = self._hits.setdefault(key, deque())
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= self.max_requests:
                return False
            q.append(now)
            return True

    def retry_after(self, key: str, now: float | None = None) -> float:
        """Seconds until the oldest hit expires (0.0 when not limited)."""
        now = time.monotonic() if now is None else now
        with self._guard:
            q = self._hits.get(key)
            if not q:
                return 0.0
            return max(0.0, q[0] + self.window - now)

    def clear(self, key: str) -> None:
        with self._guard:
            self._hits.pop(key, None)

    def __len__(self) -> int:
        with self._guard:
            return len(self._hits)


_query_rate_limiter = _SlidingWindowRateLimiter(
    max_requests=settings.QUERY_RATE_LIMIT_MAX,
    window_seconds=settings.QUERY_RATE_LIMIT_WINDOW_SECONDS,
)


def enforce_query_rate_limit(auth: AuthContext) -> None:
    """429 when the caller has exceeded their sliding-window /query budget."""
    key = f"{auth.tenant_id}:{auth.subject}"
    if _query_rate_limiter.allow(key):
        return
    retry = int(_query_rate_limiter.retry_after(key)) + 1
    raise HTTPException(
        status_code=429,
        detail=f"Rate limit exceeded. Retry in {retry}s.",
        headers={"Retry-After": str(retry)},
    )


def get_pipeline(tenant_id: str) -> RAGPipeline:
    """Return the tenant's pipeline, constructing it lazily under a lock."""
    return _pipeline_cache.get_or_create(tenant_id)


def get_pipeline_for_auth(
    auth: AuthContext = Depends(get_auth_context),
) -> RAGPipeline | None:
    """The tenant's pipeline, or None when the tenant has no index yet."""
    try:
        return get_pipeline(auth.tenant_id)
    except RuntimeError:
        return None


def _reindex(tenant_id: str) -> IngestResponse:
    """Rebuild a tenant's index from its docs folder and refresh the cache."""
    lock = _pipeline_cache.lock_for(tenant_id)
    with lock:
        documents, chunks = ingest_documents(settings.tenant_docs_dir(tenant_id))
        save_vectorstore(build_vectorstore(chunks), tenant_id)
        _pipeline_cache.invalidate(tenant_id)  # next request reloads from disk
    return IngestResponse(
        tenant_id=tenant_id,
        documents=len(documents),
        chunks=len(chunks),
        saved_to=str(settings.tenant_vectorstore_dir(tenant_id)),
    )


def _sanitize_filename(name: str) -> str:
    """Strip any directory components from an uploaded file name."""
    return Path(name).name


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict:
    return {
        "service": "Enterprise RAG Knowledge Assistant",
        "status": "running",
        "docs": "/docs",
        "version": app.version,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/auth/setup", response_model=SetupStatus)
def setup_status() -> SetupStatus:
    return SetupStatus(needed=not db.is_bootstrapped())


@app.post("/auth/setup", response_model=LoginResponse)
def setup(request: SetupRequest) -> LoginResponse:
    try:
        user = db.run_setup(request.tenant_name, request.username, request.password)
    except ValueError:
        raise HTTPException(409, "Setup already completed")
    tenant = db.get_tenant(user["tenant_id"])
    token = sign_token({"uid": user["id"], "tid": user["tenant_id"], "role": user["role"]})
    return LoginResponse(
        token=token,
        token_type="bearer",
        username=user["username"],
        role=user["role"],
        tenant_id=user["tenant_id"],
        tenant_name=tenant["name"],
    )


@app.post("/auth/login", response_model=LoginResponse)
def login(request: LoginRequest, req: Request) -> LoginResponse:
    ip = req.client.host if req.client else "unknown"

    # Brute-force protection: enforce limits *before* verifying, so an attacker
    # locked out of an account also stops burning PBKDF2 cycles.
    window_start = time.time() - settings.LOGIN_FAILURE_WINDOW_SECONDS
    if (
        db.count_login_failures(request.username, window_start)
        >= settings.LOGIN_MAX_FAILURES
    ):
        raise HTTPException(429, "Too many failed attempts. Try again later.")
    if (
        db.count_login_failures_by_ip(ip, window_start)
        >= settings.LOGIN_MAX_FAILURES_PER_IP
    ):
        raise HTTPException(429, "Too many failed attempts. Try again later.")

    user = db.get_user_by_username(request.username)
    if user is None or not verify_password(request.password, user["password_hash"]):
        db.record_login_failure(request.username, ip)
        raise HTTPException(401, "Invalid username or password")
    if not user["is_active"]:
        raise HTTPException(403, "Account disabled")
    tenant = db.get_tenant(user["tenant_id"])
    if tenant is None or not tenant["is_active"]:
        raise HTTPException(403, "Tenant disabled")
    db.clear_login_failures(request.username)
    token = sign_token({"uid": user["id"], "tid": user["tenant_id"], "role": user["role"]})
    return LoginResponse(
        token=token,
        token_type="bearer",
        username=user["username"],
        role=user["role"],
        tenant_id=user["tenant_id"],
        tenant_name=tenant["name"],
    )


@app.get("/auth/me", response_model=MeResponse)
def me(auth: AuthContext = Depends(get_auth_context)) -> MeResponse:
    return MeResponse(
        username=auth.username, role=auth.role, tenant_id=auth.tenant_id, via=auth.via
    )


# ---------------------------------------------------------------------------
# Tenant-scoped data endpoints (require auth)
# ---------------------------------------------------------------------------

@app.get("/stats", response_model=StatsResponse)
def stats(
    auth: AuthContext = Depends(get_auth_context),
    pipeline: RAGPipeline | None = Depends(get_pipeline_for_auth),
) -> StatsResponse:
    if pipeline is None:
        data = {"documents": 0, "chunks": 0}
    else:
        data = pipeline.stats()
    return StatsResponse(tenant_id=auth.tenant_id, **data)


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    auth: AuthContext = Depends(get_auth_context),
    pipeline: RAGPipeline | None = Depends(get_pipeline_for_auth),
) -> QueryResponse:
    enforce_query_rate_limit(auth)
    if pipeline is None:
        raise HTTPException(400, "No documents indexed for this tenant yet.")
    start = time.perf_counter()
    result = pipeline.answer(request.question, top_k=request.top_k)
    latency_ms = (time.perf_counter() - start) * 1000.0
    db.log_query(
        auth.tenant_id, auth.username, request.question, result["answer"], latency_ms
    )
    return QueryResponse(**result)


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents(auth: AuthContext = Depends(get_auth_context)) -> list[DocumentInfo]:
    docs_dir = settings.tenant_docs_dir(auth.tenant_id)
    if not docs_dir.exists():
        return []
    return [
        DocumentInfo(filename=p.name, size=p.stat().st_size)
        for p in sorted(docs_dir.iterdir())
        if p.is_file()
    ]


@app.post("/documents", response_model=IngestResponse)
def upload_document(
    file: UploadFile = File(...), auth: AuthContext = Depends(get_auth_context)
) -> IngestResponse:
    """Save an uploaded document to the tenant's docs folder and re-index it."""
    filename = _sanitize_filename(file.filename or "upload")
    ext = Path(filename).suffix.lower()
    if ext not in settings.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            415,
            f"Unsupported file type '{ext}'. Allowed: "
            + ", ".join(settings.SUPPORTED_EXTENSIONS),
        )

    docs_dir = settings.tenant_docs_dir(auth.tenant_id)
    settings.ensure_tenant_dirs(auth.tenant_id)
    dest = docs_dir / filename
    counter = 1
    while dest.exists():
        dest = docs_dir / f"{dest.stem}_{counter}{dest.suffix}"
        counter += 1

    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        return _reindex(auth.tenant_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.delete("/documents/{filename}", response_model=IngestResponse)
def delete_document(
    filename: str, auth: AuthContext = Depends(get_auth_context)
) -> IngestResponse:
    dest = settings.tenant_docs_dir(auth.tenant_id) / _sanitize_filename(filename)
    if not dest.exists() or not dest.is_file():
        raise HTTPException(404, "Document not found")
    dest.unlink()
    if store_exists(auth.tenant_id):
        try:
            return _reindex(auth.tenant_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    return IngestResponse(
        tenant_id=auth.tenant_id,
        documents=0,
        chunks=0,
        saved_to=str(settings.tenant_vectorstore_dir(auth.tenant_id)),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(auth: AuthContext = Depends(get_auth_context)) -> IngestResponse:
    """Rebuild this tenant's index from everything in its docs folder."""
    try:
        return _reindex(auth.tenant_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# ---------------------------------------------------------------------------
# Admin endpoints (role=admin)
# ---------------------------------------------------------------------------

def _tenant_out(t: dict) -> TenantOut:
    return TenantOut(
        tenant_id=t["tenant_id"],
        name=t["name"],
        created_at=t["created_at"],
        is_active=bool(t["is_active"]),
        users=db.count_users(t["tenant_id"]),
        documents=db.count_documents(t["tenant_id"]),
        api_keys=db.count_api_keys(t["tenant_id"]),
    )


@app.post("/admin/tenants", response_model=TenantOut)
def admin_create_tenant(
    body: TenantCreate, _: AuthContext = Depends(require_superadmin)
) -> TenantOut:
    if db.get_tenant(body.tenant_id):
        raise HTTPException(409, "Tenant already exists")
    try:
        record = db.create_tenant(body.tenant_id, body.name)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return _tenant_out(record)


@app.get("/admin/tenants", response_model=list[TenantOut])
def admin_list_tenants(_: AuthContext = Depends(require_superadmin)) -> list[TenantOut]:
    return [_tenant_out(t) for t in db.list_tenants()]


@app.patch("/admin/tenants/{tenant_id}", response_model=TenantOut)
def admin_update_tenant(
    tenant_id: str, body: TenantUpdate, _: AuthContext = Depends(require_superadmin)
) -> TenantOut:
    if not db.get_tenant(tenant_id):
        raise HTTPException(404, "Tenant not found")
    db.set_tenant_active(tenant_id, body.is_active)
    return _tenant_out(db.get_tenant(tenant_id))


@app.post("/admin/tenants/{tenant_id}/users", response_model=UserCreated)
def admin_create_user(
    tenant_id: str,
    body: UserCreate,
    auth: AuthContext = Depends(require_admin),
) -> UserCreated:
    ensure_tenant_access(auth, tenant_id)
    if not db.get_tenant(tenant_id):
        raise HTTPException(404, "Tenant not found")
    if body.role == "superadmin" and auth.role != "superadmin":
        raise HTTPException(403, "Only a platform admin can create platform admins")
    generated = ""
    if body.password:
        if len(body.password) < 8:
            raise HTTPException(422, "Password must be at least 8 characters")
        password = body.password
    else:
        password = secrets.token_urlsafe(12)  # ~96 bits, shown once below
        generated = password
    try:
        user = db.create_user(tenant_id, body.username, password, body.role)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    created = UserCreated(**user)
    created.password = generated or None
    return created


@app.get("/admin/tenants/{tenant_id}/users", response_model=list[UserOut])
def admin_list_users(
    tenant_id: str, auth: AuthContext = Depends(require_admin)
) -> list[UserOut]:
    ensure_tenant_access(auth, tenant_id)
    if not db.get_tenant(tenant_id):
        raise HTTPException(404, "Tenant not found")
    return [UserOut(**u) for u in db.list_users(tenant_id)]


@app.patch("/admin/tenants/{tenant_id}/users/{username}", response_model=UserOut)
def admin_update_user(
    tenant_id: str,
    username: str,
    body: UserUpdate,
    auth: AuthContext = Depends(require_admin),
) -> UserOut:
    ensure_tenant_access(auth, tenant_id)
    user = db.get_user_by_username(username)
    if user is None or user["tenant_id"] != tenant_id:
        raise HTTPException(404, "User not found")
    db.set_user_active(user["id"], body.is_active)
    return UserOut(**db.get_user_by_id(user["id"]))


@app.post("/admin/tenants/{tenant_id}/api-keys", response_model=ApiKeyCreated)
def admin_create_api_key(
    tenant_id: str,
    body: ApiKeyCreate,
    auth: AuthContext = Depends(require_admin),
) -> ApiKeyCreated:
    ensure_tenant_access(auth, tenant_id)
    if not db.get_tenant(tenant_id):
        raise HTTPException(404, "Tenant not found")
    plain, record = db.create_api_key(tenant_id, body.label)
    return ApiKeyCreated(
        tenant_id=tenant_id, label=record["label"], key=plain
    )


@app.get("/admin/tenants/{tenant_id}/api-keys", response_model=list[ApiKeyOut])
def admin_list_api_keys(
    tenant_id: str, auth: AuthContext = Depends(require_admin)
) -> list[ApiKeyOut]:
    ensure_tenant_access(auth, tenant_id)
    if not db.get_tenant(tenant_id):
        raise HTTPException(404, "Tenant not found")
    return [ApiKeyOut(**k) for k in db.list_api_keys(tenant_id)]


@app.get("/admin/logs", response_model=list[QueryLogOut])
def admin_logs(
    auth: AuthContext = Depends(require_admin),
    tenant_id: str | None = None,
    limit: int = 50,
) -> list[QueryLogOut]:
    # Enterprise admins only ever see their own tenant's log.
    if auth.role == "admin":
        tenant_id = auth.tenant_id
    return [
        QueryLogOut(**r) for r in db.list_logs(tenant_id=tenant_id, limit=min(limit, 500))
    ]
