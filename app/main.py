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
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app import db
from app.auth import (
    AuthContext,
    ensure_tenant_access,
    get_auth_context,
    require_admin,
    require_superadmin,
)
from app.config import settings
from app.logging_utils import log, request_id_var, setup_logging
from app.metrics import metrics
from app.rag.ingestion import ingest_documents
from app.rag.pipeline import RAGPipeline
from app.rag.vectorstore import (
    build_vectorstore,
    delete_vectorstore,
    save_vectorstore,
)
from app.schemas import (
    ApiKeyCreated,
    ApiKeyCreate,
    ApiKeyOut,
    ApiKeyUpdate,
    AuditLogOut,
    ChangePasswordRequest,
    ChangePasswordResponse,
    DocumentInfo,
    IngestResponse,
    JobStatus,
    LoginRequest,
    LoginResponse,
    MeResponse,
    QueryLogOut,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
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
from app.security import make_login_token, needs_rehash, verify_password_with_hash

@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    db.seed_defaults()
    _register_metrics()
    # Prune expired revocation records on startup so the table stays bounded.
    db.prune_revoked_tokens()
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
    allow_origins=settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.middleware("http")
async def _global_rate_limit(request: Request, call_next):
    """Per-IP sliding-window rate limit on all endpoints."""
    # Skip rate limiting for health checks and metrics.
    if request.url.path in ("/health", "/health/ready", "/metrics"):
        return await call_next(request)
    ip = request.client.host if request.client else "unknown"
    if not _global_rate_limiter.allow(ip):
        retry = int(_global_rate_limiter.retry_after(ip)) + 1
        return Response(
            content='{"detail":"Rate limit exceeded. Try again later."}',
            status_code=429,
            media_type="application/json",
            headers={
                "Retry-After": str(retry),
                "X-RateLimit-Limit": str(_global_rate_limiter.max_requests),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(_global_rate_limiter.reset_after(ip)),
                "X-Request-ID": request_id_var.get(""),
            },
        )
    response = await call_next(request)
    remaining = _global_rate_limiter.remaining(ip)
    response.headers["X-RateLimit-Limit"] = str(_global_rate_limiter.max_requests)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(_global_rate_limiter.reset_after(ip))
    return response


# ---------------------------------------------------------------------------
# Metrics + structured request logging (request id correlation)
# ---------------------------------------------------------------------------

def _register_metrics() -> None:
    metrics.register("rag_http_requests_total", "HTTP requests by method/path/status")
    metrics.register(
        "rag_http_request_duration_seconds", "HTTP request latency histogram"
    )
    metrics.register("rag_query_requests_total", "Grounded /query calls")
    metrics.register("rag_query_latency_seconds", "/query latency histogram")
    metrics.register("rag_index_documents", "Indexed documents per tenant")
    metrics.register("rag_index_chunks", "Indexed chunks per tenant")
    metrics.register("rag_active_ingest_jobs", "Queued + running ingest jobs")
    metrics.register("rag_pipeline_cache_size", "Tenant pipelines held in memory")


def _metric_path(path: str) -> str:
    """Coarse-grained path for metric labels (low cardinality).

    Identifier-like segments (ids, hashes, filenames) are replaced with
    ``:id`` so a misbehaving URL cannot explode label cardinality.
    """
    parts: list[str] = []
    for seg in path.split("/"):
        if not seg:
            continue
        if (
            seg.isdigit()
            or (len(seg) >= 16 and all(c in "0123456789abcdef" for c in seg))
            or ("." in seg)
        ):
            parts.append(":id")
        else:
            parts.append(seg)
    return "/" + "/".join(parts)


@app.middleware("http")
async def _request_context(request: Request, call_next):
    raw_id = request.headers.get("X-Request-ID", "")
    # Sanitize: only allow alphanumeric up to 64 chars, else generate a new one.
    request_id = (
        raw_id if len(raw_id) <= 64 and raw_id.isalnum()
        else uuid.uuid4().hex
    )
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Request-ID"] = request_id
        path = _metric_path(request.url.path)
        metrics.inc(
            "rag_http_requests_total",
            {"method": request.method, "path": path, "status": str(response.status_code)},
        )
        metrics.observe(
            "rag_http_request_duration_seconds",
            duration_ms / 1000.0,
            {"method": request.method, "path": path},
        )
        log.info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response
    finally:
        request_id_var.reset(token)


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
        # RLock so gauge updates (which take the same guard) can run while we
        # already hold it inside get_or_create/invalidate.
        self._guard = threading.RLock()

    def _set_gauge(self) -> None:
        with self._guard:
            metrics.set_gauge("rag_pipeline_cache_size", len(self._data))

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
                self._set_gauge()
                return pipeline

    def invalidate(self, tenant_id: str) -> None:
        """Drop a tenant's cached pipeline (e.g. after a re-index) so the next
        request reloads it from disk."""
        with self._guard:
            self._data.pop(tenant_id, None)
            self._set_gauge()

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
            # Remove empty deques so old keys don't leak memory.
            if not q:
                del self._hits[key]
                q = self._hits.setdefault(key, deque())
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

    def remaining(self, key: str) -> int:
        """How many more requests the caller can make right now."""
        now = time.monotonic()
        cutoff = now - self.window
        with self._guard:
            q = self._hits.get(key)
            if not q:
                return self.max_requests
            active = sum(1 for t in q if t > cutoff)
            return max(0, self.max_requests - active)

    def reset_after(self, key: str) -> int:
        """Seconds until the oldest hit in the window expires (rate limit resets)."""
        retry = self.retry_after(key)
        return int(retry) + (1 if retry > int(retry) else 0)

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


# Global per-IP rate limiter (protects all endpoints, including unauthenticated).
_global_rate_limiter = _SlidingWindowRateLimiter(
    max_requests=settings.GLOBAL_RATE_LIMIT_MAX,
    window_seconds=settings.GLOBAL_RATE_LIMIT_WINDOW_SECONDS,
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
    """Rebuild a tenant's index from its docs folder and refresh the cache.

    A tenant with no (supported) documents gets a clean empty index rather than
    an error, so deleting the last document degrades gracefully.
    """
    lock = _pipeline_cache.lock_for(tenant_id)
    with lock:
        try:
            documents, chunks = ingest_documents(settings.tenant_docs_dir(tenant_id))
        except ValueError:
            documents, chunks = [], []
            delete_vectorstore(tenant_id)
        if chunks:
            save_vectorstore(build_vectorstore(chunks), tenant_id)
        _pipeline_cache.invalidate(tenant_id)  # next request reloads from disk
    metrics.set_gauge("rag_index_documents", len(documents), {"tenant": tenant_id})
    metrics.set_gauge("rag_index_chunks", len(chunks), {"tenant": tenant_id})
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
# Background ingest jobs (uploads/re-indexes never block the HTTP request)
# ---------------------------------------------------------------------------

class _IngestJobManager:
    """Tracks the re-index state per tenant.

    A re-index is queued as a FastAPI background task and runs under the
    tenant's pipeline lock (so it cannot race a lazy index load or another
    re-index). Consecutive requests for the same tenant share one job.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def _blank(self) -> dict:
        return {
            "status": "idle",
            "version": 0,
            "documents": 0,
            "chunks": 0,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "saved_to": None,
        }

    def enqueue(self, tenant_id: str, background: BackgroundTasks) -> JobStatus:
        """Queue a re-index for the tenant. Reuses an in-flight job."""
        with self._guard:
            job = self._jobs.get(tenant_id)
            if job and job["status"] in ("queued", "running"):
                return self._status_of(tenant_id, job)
            job = self._blank()
            job["status"] = "queued"
            job["version"] += 1
            self._jobs[tenant_id] = job
            self._gauge()
        background.add_task(self._run, tenant_id, job["version"])
        return self._status_of(tenant_id, job)

    def status(self, tenant_id: str) -> JobStatus:
        with self._guard:
            job = self._jobs.get(tenant_id)
            return self._status_of(tenant_id, job or self._blank())

    def _status_of(self, tenant_id: str, job: dict) -> JobStatus:
        return JobStatus(
            tenant_id=tenant_id,
            status=job["status"],
            documents=job["documents"],
            chunks=job["chunks"],
            started_at=job["started_at"],
            finished_at=job["finished_at"],
            error=job["error"],
            saved_to=job["saved_to"],
        )

    def _gauge(self) -> None:
        metrics.set_gauge(
            "rag_active_ingest_jobs",
            sum(
                1
                for j in self._jobs.values()
                if j["status"] in ("queued", "running")
            ),
        )

    def any_active(self) -> bool:
        with self._guard:
            return any(
                j["status"] in ("queued", "running") for j in self._jobs.values()
            )

    def _run(self, tenant_id: str, version: int) -> None:
        with self._guard:
            job = self._jobs.get(tenant_id)
            if (
                job is None
                or job["version"] != version
                or job["status"] != "queued"
            ):
                return  # superseded or already handled
            job["status"] = "running"
            job["started_at"] = time.time()
            self._gauge()
        try:
            result = _reindex(tenant_id)
        except Exception as exc:  # noqa: BLE001 - report job failure to the UI
            log.exception("ingest_job_failed", extra={"tenant_id": tenant_id})
            with self._guard:
                job["status"] = "failed"
                job["error"] = str(exc)
                job["finished_at"] = time.time()
                self._gauge()
            return
        with self._guard:
            job["status"] = "done"
            job["documents"] = result.documents
            job["chunks"] = result.chunks
            job["saved_to"] = result.saved_to
            job["finished_at"] = time.time()
            self._gauge()


_ingest_jobs = _IngestJobManager()


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


@app.get("/health/ready", response_model=ReadinessResponse)
def readiness() -> ReadinessResponse:
    """Readiness probe for orchestrators (Docker/K8s): the process is ready
    when the database responds and no ingest is mid-flight."""
    checks = {
        "database": _db_reachable(),
        "indexing_idle": not _ingest_jobs.any_active(),
    }
    return ReadinessResponse(
        status="ready" if all(checks.values()) else "degraded",
        checks=checks,
    )


@app.get("/metrics", include_in_schema=False)
def metrics_endpoint() -> Response:
    return Response(
        content=metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


def _db_reachable() -> bool:
    try:
        conn = db.get_conn()
        conn.execute("SELECT 1").fetchone()
        return True
    except Exception:  # noqa: BLE001 - readiness should never raise
        return False


@app.get("/auth/setup", response_model=SetupStatus)
def setup_status() -> SetupStatus:
    return SetupStatus(needed=not db.is_bootstrapped())


@app.post("/auth/setup", response_model=LoginResponse)
def setup(request: SetupRequest) -> LoginResponse:
    if len(request.password) < 8:
        raise HTTPException(422, "Password must be at least 8 characters")
    try:
        user = db.run_setup(request.tenant_name, request.username, request.password)
    except ValueError:
        raise HTTPException(409, "Setup already completed")
    tenant = db.get_tenant(user["tenant_id"])
    token = make_login_token(user)
    db.log_audit(
        user["tenant_id"],
        user["username"],
        user["role"],
        "setup",
        f"enterprise '{request.tenant_name}' created",
    )
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
        db.log_audit(
            None, request.username, None, "login_locked", f"ip={ip}"
        )
        raise HTTPException(429, "Too many failed attempts. Try again later.")
    if (
        db.count_login_failures_by_ip(ip, window_start)
        >= settings.LOGIN_MAX_FAILURES_PER_IP
    ):
        db.log_audit(None, request.username, None, "login_throttled", f"ip={ip}")
        raise HTTPException(429, "Too many failed attempts. Try again later.")

    user = db.get_user_by_username(request.username)
    ok, parsed = (
        verify_password_with_hash(request.password, user["password_hash"])
        if user is not None
        else (False, None)
    )
    if not ok:
        db.record_login_failure(request.username, ip)
        db.log_audit(None, request.username, None, "login_failed", f"ip={ip}")
        raise HTTPException(401, "Invalid username or password")
    if not user["is_active"]:
        db.log_audit(None, request.username, None, "login_disabled_account", f"ip={ip}")
        raise HTTPException(401, "Invalid username or password")
    tenant = db.get_tenant(user["tenant_id"])
    if tenant is None or not tenant["is_active"]:
        db.log_audit(None, request.username, None, "login_disabled_tenant", f"ip={ip}")
        raise HTTPException(401, "Invalid username or password")
    db.clear_login_failures(request.username)
    if needs_rehash(parsed):
        # Stored hash predates the current PBKDF2 cost: upgrade it in place.
        # No token-version bump here, so the token we are about to issue works.
        db.update_password_hash(user["id"], request.password)
    db.log_audit(user["tenant_id"], user["username"], user["role"], "login", f"ip={ip}")
    token = make_login_token(user)
    return LoginResponse(
        token=token,
        token_type="bearer",
        username=user["username"],
        role=user["role"],
        tenant_id=user["tenant_id"],
        tenant_name=tenant["name"],
    )


@app.post("/auth/change-password", response_model=ChangePasswordResponse)
def change_password(
    request: ChangePasswordRequest, auth: AuthContext = Depends(get_auth_context)
) -> ChangePasswordResponse:
    """Self-service password rotation. Bumps the user's token version so every
    other session signed before the change is revoked immediately; the response
    carries a fresh token for this session."""
    if auth.user_id is None:
        raise HTTPException(
            400, "Password change requires token-based authentication, not an API key"
        )
    user = db.get_user_by_id(auth.user_id)
    ok, _ = verify_password_with_hash(request.old_password, user["password_hash"])
    if not ok:
        db.log_audit(
            auth.tenant_id, auth.username, auth.role, "password_change_failed"
        )
        raise HTTPException(400, "Current password is incorrect")
    if request.new_password == request.old_password:
        raise HTTPException(
            422, "New password must be different from the current password"
        )
    if len(request.new_password) < 8:
        raise HTTPException(422, "New password must be at least 8 characters")
    db.set_user_password(user["id"], request.new_password)
    # Revoke the current session's token so the old jti cannot be reused.
    if auth.jti:
        db.revoke_token(auth.jti, auth.user_id)
    db.clear_login_failures(auth.username)
    db.log_audit(auth.tenant_id, auth.username, auth.role, "password_change")
    fresh = db.get_user_by_id(user["id"])
    return ChangePasswordResponse(
        token=make_login_token(fresh),
        username=fresh["username"],
        role=fresh["role"],
        tenant_id=fresh["tenant_id"],
    )


@app.get("/auth/me", response_model=MeResponse)
def me(auth: AuthContext = Depends(get_auth_context)) -> MeResponse:
    return MeResponse(
        username=auth.username, role=auth.role, tenant_id=auth.tenant_id, via=auth.via
    )


@app.post("/auth/logout")
def logout(auth: AuthContext = Depends(get_auth_context)) -> dict:
    """Revoke the current session's token. The token can no longer be used
    to authenticate until the user logs in again. Other sessions are unaffected.
    API key callers should disable the key via admin endpoints instead."""
    if auth.via != "token" or not auth.jti:
        raise HTTPException(400, "API key sessions cannot be logged out this way")
    db.revoke_token(auth.jti, auth.user_id)
    db.log_audit(auth.tenant_id, auth.username, auth.role, "logout")
    return {"status": "logged_out"}


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
    metrics.inc("rag_query_requests_total", {"tenant": auth.tenant_id})
    metrics.observe(
        "rag_query_latency_seconds", latency_ms / 1000.0, {"tenant": auth.tenant_id}
    )
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


@app.post("/documents", response_model=JobStatus, status_code=202)
def upload_document(
    file: UploadFile = File(...),
    auth: AuthContext = Depends(get_auth_context),
    background: BackgroundTasks = None,  # injected fresh per request by FastAPI
) -> JobStatus:
    """Save an uploaded document and queue a background re-index."""
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

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    try:
        with dest.open("wb") as out:
            written = _copy_upload_limited(file.file, out, max_bytes)
    except HTTPException:
        dest.unlink(missing_ok=True)  # never leave a partial file behind
        raise
    db.log_audit(
        auth.tenant_id,
        auth.username,
        auth.role,
        "document.upload",
        f"{filename} ({written} bytes)",
    )
    return _ingest_jobs.enqueue(auth.tenant_id, background)


@app.delete("/documents/{filename}", response_model=JobStatus, status_code=202)
def delete_document(
    filename: str,
    auth: AuthContext = Depends(get_auth_context),
    background: BackgroundTasks = None,  # injected fresh per request by FastAPI
) -> JobStatus:
    dest = settings.tenant_docs_dir(auth.tenant_id) / _sanitize_filename(filename)
    if not dest.exists() or not dest.is_file():
        raise HTTPException(404, "Document not found")
    dest.unlink()
    db.log_audit(
        auth.tenant_id, auth.username, auth.role, "document.delete", filename
    )
    return _ingest_jobs.enqueue(auth.tenant_id, background)


@app.post("/ingest", response_model=JobStatus, status_code=202)
def ingest(
    auth: AuthContext = Depends(get_auth_context),
    background: BackgroundTasks = None,  # injected fresh per request by FastAPI
) -> JobStatus:
    """Queue a full re-index of this tenant's docs folder."""
    db.log_audit(auth.tenant_id, auth.username, auth.role, "ingest.start")
    return _ingest_jobs.enqueue(auth.tenant_id, background)


@app.get("/ingest/status", response_model=JobStatus)
def ingest_status(auth: AuthContext = Depends(get_auth_context)) -> JobStatus:
    """Current re-index state for this tenant (poll after upload/delete)."""
    return _ingest_jobs.status(auth.tenant_id)


def _copy_upload_limited(src, dst, max_bytes: int) -> int:
    """Copy an upload to disk, hard-failing with 413 past the size cap.

    The cap is enforced on the bytes actually read, not on any client-supplied
    header, so a lying Content-Length cannot bypass it."""
    written = 0
    while True:
        chunk = src.read(1024 * 1024)
        if not chunk:
            break
        written += len(chunk)
        if written > max_bytes:
            raise HTTPException(
                413,
                f"File exceeds the {settings.MAX_UPLOAD_MB} MiB upload limit.",
            )
        dst.write(chunk)
    return written


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
    body: TenantCreate, auth: AuthContext = Depends(require_superadmin)
) -> TenantOut:
    if db.get_tenant(body.tenant_id):
        raise HTTPException(409, "Tenant already exists")
    try:
        record = db.create_tenant(body.tenant_id, body.name)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    db.log_audit(
        body.tenant_id,
        auth.username,
        auth.role,
        "tenant.create",
        f"name='{body.name}'",
    )
    return _tenant_out(record)


@app.get("/admin/tenants", response_model=list[TenantOut])
def admin_list_tenants(_: AuthContext = Depends(require_superadmin)) -> list[TenantOut]:
    return [_tenant_out(t) for t in db.list_tenants()]


@app.patch("/admin/tenants/{tenant_id}", response_model=TenantOut)
def admin_update_tenant(
    tenant_id: str,
    body: TenantUpdate,
    auth: AuthContext = Depends(require_superadmin),
) -> TenantOut:
    if not db.get_tenant(tenant_id):
        raise HTTPException(404, "Tenant not found")
    db.set_tenant_active(tenant_id, body.is_active)
    db.log_audit(
        tenant_id,
        auth.username,
        auth.role,
        "tenant.update",
        f"is_active={body.is_active}",
    )
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
    db.log_audit(
        tenant_id,
        auth.username,
        auth.role,
        "user.create",
        f"username='{body.username}' role={body.role}",
    )
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
    # Prevent admins from disabling their own account (self-lockout).
    if not body.is_active and auth.username == username:
        raise HTTPException(400, "Cannot disable your own account")
    db.set_user_active(user["id"], body.is_active)
    db.log_audit(
        tenant_id,
        auth.username,
        auth.role,
        "user.update",
        f"username='{username}' is_active={body.is_active}",
    )
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
    db.log_audit(
        tenant_id,
        auth.username,
        auth.role,
        "api_key.create",
        f"label='{body.label}' key={record['key_hash'][:12]}...",
    )
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


@app.patch("/admin/tenants/{tenant_id}/api-keys/{key_hash}", response_model=ApiKeyOut)
def admin_update_api_key(
    tenant_id: str,
    key_hash: str,
    body: ApiKeyUpdate,
    auth: AuthContext = Depends(require_admin),
) -> ApiKeyOut:
    """Enable/disable an API key (rotate = disable the old one, create a new)."""
    ensure_tenant_access(auth, tenant_id)
    if not db.set_api_key_active(tenant_id, key_hash, body.is_active):
        raise HTTPException(404, "API key not found")
    db.log_audit(
        tenant_id,
        auth.username,
        auth.role,
        "api_key.update",
        f"key={key_hash[:12]}... is_active={body.is_active}",
    )
    record = db.get_api_key_by_hash(key_hash)
    return ApiKeyOut(**record)


@app.delete("/admin/tenants/{tenant_id}/api-keys/{key_hash}")
def admin_delete_api_key(
    tenant_id: str,
    key_hash: str,
    auth: AuthContext = Depends(require_admin),
) -> dict:
    """Permanently revoke an API key (irreversible)."""
    ensure_tenant_access(auth, tenant_id)
    if not db.delete_api_key(tenant_id, key_hash):
        raise HTTPException(404, "API key not found")
    db.log_audit(
        tenant_id,
        auth.username,
        auth.role,
        "api_key.revoke",
        f"key={key_hash[:12]}...",
    )
    return {"status": "revoked", "key_hash": key_hash[:12] + "..."}


@app.get("/admin/logs", response_model=list[QueryLogOut])
def admin_logs(
    auth: AuthContext = Depends(require_admin),
    tenant_id: str | None = None,
    limit: int = 50,
) -> list[QueryLogOut]:
    # Enterprise admins only ever see their own tenant's log.
    if auth.role == "admin":
        tenant_id = auth.tenant_id
    limit = max(1, min(limit, 500))
    return [
        QueryLogOut(**r) for r in db.list_logs(tenant_id=tenant_id, limit=limit)
    ]


@app.get("/admin/audit-logs", response_model=list[AuditLogOut])
def admin_audit_logs(
    auth: AuthContext = Depends(require_admin),
    tenant_id: str | None = None,
    limit: int = 100,
) -> list[AuditLogOut]:
    """Append-only record of admin/auth actions. Enterprise admins are scoped
    to their own tenant; the platform admin sees everything."""
    if auth.role == "admin":
        tenant_id = auth.tenant_id
    limit = max(1, min(limit, 1000))
    return [
        AuditLogOut(**r)
        for r in db.list_audit_logs(tenant_id=tenant_id, limit=limit)
    ]
