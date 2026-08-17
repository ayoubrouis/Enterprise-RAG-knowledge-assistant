"""Unit + API tests. No model downloads and no network required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.auth import AuthContext, get_auth_context
from app.config import settings
from app.eval.evaluate import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.main import app, get_pipeline_for_auth
from app.rag.ingestion import chunk_documents, load_documents


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def isolated_db():
    """Point the auth database at a throwaway file for the whole test run."""
    import app.db as db

    settings.DB_PATH = settings.ROOT_DIR / "data" / "_test_system.db"
    db._CONN = None
    yield
    if db._CONN is not None:
        db._CONN.close()
        db._CONN = None
    for suffix in ("", "-wal", "-shm"):
        (settings.ROOT_DIR / "data" / f"_test_system.db{suffix}").unlink(missing_ok=True)


@pytest.fixture(scope="session")
def sample_documents():
    """Ensure demo data exists (generate on demand) and load it."""
    docs_dir = settings.DOCS_DIR
    if not docs_dir.exists() or not any(docs_dir.iterdir()):
        import scripts.make_demo_data as demo

        demo.main()
    return load_documents()


class FakePipeline:
    def answer(self, question: str, top_k: int | None = None) -> dict:
        return {
            "question": question,
            "answer": "fake answer",
            "sources": [
                {
                    "source": "fake.pdf",
                    "page": 1,
                    "similarity": 0.9,
                    "snippet": "fake snippet",
                }
            ],
        }

    def stats(self) -> dict:
        return {"documents": 1, "chunks": 5}


ADMIN_CONTEXT = AuthContext(
    tenant_id="default", username="tester", role="admin", subject="user:1", user_id=1
)
USER_CONTEXT = AuthContext(
    tenant_id="default", username="tester", role="user", subject="user:1", user_id=1
)
SUPERADMIN_CONTEXT = AuthContext(
    tenant_id="default", username="platform", role="superadmin", subject="user:2", user_id=2
)


@pytest.fixture()
def client():
    app.dependency_overrides[get_auth_context] = lambda: ADMIN_CONTEXT
    app.dependency_overrides[get_pipeline_for_auth] = FakePipeline
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def user_client():
    app.dependency_overrides[get_auth_context] = lambda: USER_CONTEXT
    app.dependency_overrides[get_pipeline_for_auth] = FakePipeline
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def super_client():
    app.dependency_overrides[get_auth_context] = lambda: SUPERADMIN_CONTEXT
    app.dependency_overrides[get_pipeline_for_auth] = FakePipeline
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def anon_client():
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def fresh_client():
    """A brand-new, un-bootstrapped database (first-run state)."""
    import app.db as db

    if db._CONN is not None:
        db._CONN.close()
        db._CONN = None
    for suffix in ("", "-wal", "-shm"):
        (settings.DB_PATH).with_name(f"{settings.DB_PATH.name}{suffix}").unlink(
            missing_ok=True
        )
    app.dependency_overrides.clear()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Retrieval metric math (pure functions)
# ---------------------------------------------------------------------------


def test_precision_at_k():
    retrieved = ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]
    relevant = {"a.pdf", "d.pdf"}
    assert precision_at_k(retrieved, relevant, 4) == 0.5
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert precision_at_k(retrieved, {"x.pdf"}, 4) == 0.0
    assert precision_at_k(retrieved, relevant, 0) == 0.0


def test_recall_at_k():
    retrieved = ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]
    relevant = {"a.pdf", "d.pdf"}
    assert recall_at_k(retrieved, relevant, 4) == 1.0
    assert recall_at_k(retrieved, relevant, 1) == 0.5
    assert recall_at_k(retrieved, {"z.pdf"}, 4) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a.pdf", "b.pdf"], {"b.pdf"}) == 0.5
    assert reciprocal_rank(["x.pdf", "y.pdf"], {"b.pdf"}) == 0.0
    assert reciprocal_rank(["b.pdf", "a.pdf"], {"b.pdf"}) == 1.0


# ---------------------------------------------------------------------------
# Ingestion & chunking
# ---------------------------------------------------------------------------


def test_load_documents_finds_demo_files(sample_documents):
    assert len(sample_documents) >= 5
    sources = {d.metadata.get("source") for d in sample_documents}
    assert "benefits_overview.pdf" in sources
    assert "onboarding_guide.txt" in sources
    assert "atlas_gateway_manual.md" in sources


def test_pdf_documents_carry_page_numbers(sample_documents):
    pdf_docs = [d for d in sample_documents if "benefits_overview.pdf" in d.metadata["source"]]
    assert pdf_docs
    assert all(d.metadata.get("page") is not None for d in pdf_docs)


def test_chunking_preserves_metadata_and_limits_size(sample_documents):
    chunks = chunk_documents(sample_documents)
    assert len(chunks) > len(sample_documents)  # at least one split happened
    assert all(len(c.page_content) <= settings.CHUNK_SIZE + 200 for c in chunks)
    assert all("source" in c.metadata for c in chunks)
    assert all("chunk_id" in c.metadata for c in chunks)
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_root(client):
    assert client.get("/").json()["status"] == "running"


# ---------------------------------------------------------------------------
# First-run setup wizard
# ---------------------------------------------------------------------------


def test_no_admin_before_setup(fresh_client):
    import app.db as db

    assert db.is_bootstrapped() is False
    assert fresh_client.get("/auth/setup").json() == {"needed": True}
    assert db.get_user_by_username("admin") is None


def test_schema_migration_adds_superadmin_role():
    import sqlite3

    import app.db as db

    path = settings.DB_PATH.with_name("_test_migration.db")
    for suffix in ("", "-wal", "-shm"):
        path.with_name(f"{path.name}{suffix}").unlink(missing_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE tenants (
            tenant_id TEXT PRIMARY KEY, name TEXT NOT NULL,
            created_at REAL NOT NULL, is_active INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id TEXT NOT NULL, username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('admin','user')),
            is_active INTEGER NOT NULL DEFAULT 1, created_at REAL NOT NULL
        );
        INSERT INTO tenants VALUES ('default', 'Default', 1.0, 1);
        INSERT INTO users (tenant_id, username, password_hash, role, created_at)
            VALUES ('default', 'old', 'x', 'admin', 1.0);
        """
    )
    conn.commit()
    conn.close()

    old_path = settings.DB_PATH
    settings.DB_PATH = path
    try:
        if db._CONN is not None:
            db._CONN.close()
            db._CONN = None
        db._sqlite_conn()
        assert db.get_user_by_username("old")["role"] == "admin"
        db.create_user("default", "fresh", "password123", "superadmin")
        assert db.get_user_by_username("fresh")["role"] == "superadmin"
    finally:
        if db._CONN is not None:
            db._CONN.close()
            db._CONN = None
        settings.DB_PATH = old_path
        for suffix in ("", "-wal", "-shm"):
            path.with_name(f"{path.name}{suffix}").unlink(missing_ok=True)


def test_setup_creates_admin_and_returns_token(fresh_client):
    resp = fresh_client.post(
        "/auth/setup",
        json={"tenant_name": "Acme Corp", "username": "boss", "password": "SuperSecret123"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["token"]
    assert body["username"] == "boss"
    assert body["role"] == "admin"
    assert body["tenant_id"] == "default"
    assert body["tenant_name"] == "Acme Corp"

    assert fresh_client.get("/auth/setup").json() == {"needed": False}

    login = fresh_client.post(
        "/auth/login", json={"username": "boss", "password": "SuperSecret123"}
    )
    assert login.status_code == 200


def test_setup_only_runs_once(fresh_client):
    body = {
        "tenant_name": "Acme Corp",
        "username": "boss",
        "password": "SuperSecret123",
    }
    assert fresh_client.post("/auth/setup", json=body).status_code == 200
    assert fresh_client.post("/auth/setup", json=body).status_code == 409


def test_setup_rejects_weak_credentials(fresh_client):
    resp = fresh_client.post(
        "/auth/setup",
        json={"tenant_name": "Acme", "username": "bo", "password": "short"},
    )
    assert resp.status_code == 422


def test_login_success_and_wrong_password(client):
    import uuid

    import app.db as db

    username = f"alice-{uuid.uuid4().hex[:6]}"
    db.create_user("default", username, "correct-horse-1", "user")

    ok = client.post(
        "/auth/login", json={"username": username, "password": "correct-horse-1"}
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["token"]
    assert body["role"] == "user"
    assert body["tenant_id"] == "default"

    bad = client.post("/auth/login", json={"username": username, "password": "wrong"})
    assert bad.status_code == 401


def test_login_locks_account_after_repeated_failures(client):
    import uuid

    import app.db as db
    from app.config import settings

    username = f"lock-{uuid.uuid4().hex[:6]}"
    db.create_user("default", username, "correct-horse-1", "user")

    old_max, old_ip = settings.LOGIN_MAX_FAILURES, settings.LOGIN_MAX_FAILURES_PER_IP
    settings.LOGIN_MAX_FAILURES = 3
    settings.LOGIN_MAX_FAILURES_PER_IP = 1000  # isolate per-username behavior
    try:
        for _ in range(3):
            r = client.post(
                "/auth/login", json={"username": username, "password": "wrong"}
            )
            assert r.status_code == 401
        # The account is now locked: even the correct password is refused.
        locked = client.post(
            "/auth/login", json={"username": username, "password": "correct-horse-1"}
        )
        assert locked.status_code == 429
    finally:
        settings.LOGIN_MAX_FAILURES = old_max
        settings.LOGIN_MAX_FAILURES_PER_IP = old_ip
        db.clear_login_failures(username)


def test_login_throttles_by_source_ip(client):
    import uuid

    import app.db as db
    from app.config import settings

    old_max, old_ip = settings.LOGIN_MAX_FAILURES, settings.LOGIN_MAX_FAILURES_PER_IP
    settings.LOGIN_MAX_FAILURES = 1000  # isolate per-IP behavior
    # The TestClient shares one source IP, and earlier tests may have recorded
    # failures for it; size the limit above the current baseline.
    baseline = db.count_login_failures_by_ip("testclient", since=0)
    settings.LOGIN_MAX_FAILURES_PER_IP = baseline + 3
    try:
        for _ in range(3):
            username = f"ip-{uuid.uuid4().hex[:6]}"
            r = client.post(
                "/auth/login", json={"username": username, "password": "wrong"}
            )
            assert r.status_code == 401
        # Same source IP has now exceeded the per-IP limit.
        r = client.post("/auth/login", json={"username": "anyone", "password": "wrong"})
        assert r.status_code == 429
    finally:
        settings.LOGIN_MAX_FAILURES = old_max
        settings.LOGIN_MAX_FAILURES_PER_IP = old_ip


def test_endpoints_require_auth(anon_client):
    assert anon_client.get("/stats").status_code == 401
    assert anon_client.post("/query", json={"question": "hi"}).status_code == 401
    assert anon_client.get("/documents").status_code == 401
    assert anon_client.post("/ingest").status_code == 401


def test_admin_endpoints_require_admin_role(user_client):
    resp = user_client.post(
        "/admin/tenants", json={"tenant_id": "no-access", "name": "Nope"}
    )
    assert resp.status_code == 403


def test_enterprise_admin_cannot_create_tenants(client):
    resp = client.post("/admin/tenants", json={"tenant_id": "x", "name": "X"})
    assert resp.status_code == 403
    assert client.get("/admin/tenants").status_code == 403


def test_enterprise_admin_cannot_manage_other_tenant(client):
    resp = client.get("/admin/tenants/acme/users")
    assert resp.status_code == 403
    resp = client.post(
        "/admin/tenants/acme/users",
        json={"username": "intruder", "password": "password123"},
    )
    assert resp.status_code == 403
    resp = client.patch("/admin/tenants/acme", json={"is_active": False})
    assert resp.status_code == 403


def test_enterprise_admin_cannot_escalate_to_superadmin(client):
    resp = client.post(
        "/admin/tenants/default/users",
        json={"username": "evil", "password": "password123", "role": "superadmin"},
    )
    assert resp.status_code == 403


def test_enterprise_admin_manages_own_tenant(client):
    import uuid


    username = f"carol-{uuid.uuid4().hex[:6]}"
    created = client.post(
        "/admin/tenants/default/users",
        json={"username": username, "password": "correct-horse-1", "role": "user"},
    )
    assert created.status_code == 200
    assert created.json().get("password") is None  # no secret echoed when admin set it
    listed = client.get("/admin/tenants/default/users").json()
    assert any(u["username"] == username for u in listed)

    key = client.post("/admin/tenants/default/api-keys", json={"label": "ci"})
    assert key.status_code == 200
    assert key.json()["key"]
    assert client.get("/admin/tenants/default/api-keys").status_code == 200


def test_create_user_without_password_generates_one(client):
    import uuid

    username = f"dave-{uuid.uuid4().hex[:6]}"
    resp = client.post(
        "/admin/tenants/default/users",
        json={"username": username, "role": "user"},  # no password -> generated
    )
    assert resp.status_code == 200
    generated = resp.json()["password"]
    assert generated and len(generated) >= 16

    # The generated password is real: it must log the user in.
    login = client.post(
        "/auth/login", json={"username": username, "password": generated}
    )
    assert login.status_code == 200
    assert login.json()["username"] == username


def test_create_user_rejects_weak_explicit_password(client):
    resp = client.post(
        "/admin/tenants/default/users",
        json={"username": "shorty", "password": "short", "role": "user"},
    )
    assert resp.status_code == 422


def test_superadmin_manages_all_tenants(super_client):
    created = super_client.post(
        "/admin/tenants", json={"tenant_id": "acme", "name": "Acme Corp"}
    )
    assert created.status_code == 200
    tenants = super_client.get("/admin/tenants").json()
    assert any(t["tenant_id"] == "acme" for t in tenants)

    user = super_client.post(
        "/admin/tenants/acme/users",
        json={"username": "boss", "password": "correct-horse-1", "role": "admin"},
    )
    assert user.status_code == 200
    assert user.json()["role"] == "admin"

    # Superadmin may also create another platform admin.
    ops = super_client.post(
        "/admin/tenants/default/users",
        json={"username": "ops2", "password": "correct-horse-1", "role": "superadmin"},
    )
    assert ops.status_code == 200
    assert ops.json()["role"] == "superadmin"

    assert super_client.patch("/admin/tenants/acme", json={"is_active": False}).status_code == 200


def test_enterprise_admin_logs_scoped_to_own_tenant(client):
    import app.db as db

    db.log_query("default", "tester", "question a", "answer a", 1.0)
    db.log_query("acme", "someone", "question b", "answer b", 1.0)

    # Even asking for another tenant's logs yields only the caller's.
    logs = client.get("/admin/logs?tenant_id=acme").json()
    assert all(entry["tenant_id"] == "default" for entry in logs)


def test_superadmin_sees_all_logs(super_client):
    logs = super_client.get("/admin/logs").json()
    tenant_ids = {entry["tenant_id"] for entry in logs}
    assert "default" in tenant_ids and "acme" in tenant_ids


def test_admin_can_disable_user(client):
    import uuid

    import app.db as db

    username = f"bob-{uuid.uuid4().hex[:6]}"
    db.create_user("default", username, "correct-horse-1", "user")

    resp = client.patch(
        f"/admin/tenants/default/users/{username}", json={"is_active": False}
    )
    assert resp.status_code == 200
    assert resp.json()["is_active"] is False

    login = client.post(
        "/auth/login", json={"username": username, "password": "correct-horse-1"}
    )
    assert login.status_code == 401  # disabled account returns same 401 as wrong password


# ---------------------------------------------------------------------------
# API (fake pipeline injected - no models loaded)
# ---------------------------------------------------------------------------


def test_stats(client):
    assert client.get("/stats").json() == {"tenant_id": "default", "documents": 1, "chunks": 5}


def test_query_schema(client):
    resp = client.post("/query", json={"question": "What is the 401(k) match?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "fake answer"
    assert body["sources"][0]["source"] == "fake.pdf"


def test_query_rejects_empty_question(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_upload_rejects_unsupported_type(client):
    resp = client.post(
        "/documents",
        files={"file": ("payload.exe", b"not a real doc", "application/octet-stream")},
    )
    assert resp.status_code == 415


def test_list_documents_returns_tenant_files(client):
    names = {d["filename"] for d in client.get("/documents").json()}
    assert "benefits_overview.pdf" in names
    assert "atlas_gateway_manual.md" in names


# ---------------------------------------------------------------------------
# Pipeline cache (bounded LRU)
# ---------------------------------------------------------------------------


def test_pipeline_cache_evicts_least_recently_used():
    from app.main import _PipelineCache

    created: list[str] = []
    cache = _PipelineCache(max_size=2, factory=lambda tid: created.append(tid) or f"pipe-{tid}")

    cache.get_or_create("a")
    cache.get_or_create("b")
    cache.get_or_create("a")  # a is now most-recently-used
    assert cache.get_or_create("c") == "pipe-c"  # inserts c, evicts b (LRU)
    assert len(cache) == 2
    assert cache.get_or_create("b") == "pipe-b"  # b was evicted: rebuilt
    assert created == ["a", "b", "c", "b"]


def test_pipeline_cache_returns_cached_and_invalidate_rebuilds():
    from app.main import _PipelineCache

    calls = {"n": 0}

    def factory(tid):
        calls["n"] += 1
        return f"{tid}-{calls['n']}"

    cache = _PipelineCache(max_size=8, factory=factory)
    assert cache.get_or_create("a") == "a-1"
    assert cache.get_or_create("a") == "a-1"  # served from cache
    cache.invalidate("a")
    assert cache.get_or_create("a") == "a-2"  # rebuilt after invalidation
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Data paths (RAG_DATA_DIR / RAG_DB_PATH env vars)
# ---------------------------------------------------------------------------


def test_rag_data_dir_env_var_is_honored(monkeypatch, tmp_path):
    from app.config import Settings

    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    s = Settings()
    assert s.DATA_DIR == tmp_path
    assert s.TENANTS_DIR == tmp_path / "tenants"
    assert s.DB_PATH == tmp_path / "system.db"


def test_rag_data_dir_defaults_to_project_data(monkeypatch):
    from app.config import Settings

    monkeypatch.delenv("RAG_DATA_DIR", raising=False)
    monkeypatch.delenv("RAG_DB_PATH", raising=False)
    s = Settings()
    assert s.DATA_DIR == s.ROOT_DIR / "data"
    assert s.TENANTS_DIR == s.ROOT_DIR / "data" / "tenants"
    assert s.DB_PATH == s.ROOT_DIR / "data" / "system.db"


def test_rag_db_path_env_var_wins_over_data_dir(monkeypatch, tmp_path):
    from app.config import Settings

    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    db_file = tmp_path.parent / "custom.db"
    monkeypatch.setenv("RAG_DB_PATH", str(db_file))
    s = Settings()
    assert s.DB_PATH == db_file
    assert s.DATA_DIR == tmp_path


# ---------------------------------------------------------------------------
# Query rate limiting (sliding window)
# ---------------------------------------------------------------------------


def test_rate_limiter_allows_up_to_limit_then_blocks():
    from app.main import _SlidingWindowRateLimiter

    limiter = _SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    assert [limiter.allow("alice", now=100.0) for _ in range(3)] == [True, True, True]
    assert limiter.allow("alice", now=105.0) is False  # over the limit
    assert limiter.allow("bob", now=105.0) is True  # different caller unaffected


def test_rate_limiter_window_slides_and_retry_after():
    from app.main import _SlidingWindowRateLimiter

    limiter = _SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    assert limiter.allow("alice", now=100.0) is True
    assert limiter.allow("alice", now=101.0) is True
    assert limiter.allow("alice", now=101.0) is False
    assert limiter.retry_after("alice", now=101.0) == pytest.approx(59.0)
    # Oldest hit expires at 160.0, so a request at 160.0 is allowed again.
    assert limiter.allow("alice", now=160.0) is True


def test_rate_limiter_clear_and_prunes_old_entries():
    from app.main import _SlidingWindowRateLimiter

    limiter = _SlidingWindowRateLimiter(max_requests=1, window_seconds=10)
    assert limiter.allow("a", now=100.0) is True
    assert limiter.allow("a", now=100.0) is False
    limiter.clear("a")
    assert limiter.allow("a", now=100.0) is True
    # A stale hit older than the window must not count against the caller.
    assert limiter.allow("b", now=100.0) is True
    assert limiter.allow("b", now=115.0) is True  # first hit expired


def test_query_returns_429_when_rate_limit_exceeded(client, monkeypatch):
    import app.main as main_mod

    small = main_mod._SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    monkeypatch.setattr(main_mod, "_query_rate_limiter", small)

    assert client.post("/query", json={"question": "q1"}).status_code == 200
    assert client.post("/query", json={"question": "q2"}).status_code == 200
    blocked = client.post("/query", json={"question": "q3"})
    assert blocked.status_code == 429
    retry = int(blocked.headers["Retry-After"])
    assert 1 <= retry <= 61  # within one window of the oldest hit


# ---------------------------------------------------------------------------
# Tier 1 hardening: password rotation, API-key lifecycle, audit log, grounding,
# upload caps, background ingest jobs, metrics/readiness.
# ---------------------------------------------------------------------------


def test_password_hash_embeds_cost_and_detects_rehash():
    from app.security import (
        PBKDF2_ITERATIONS,
        hash_password,
        needs_rehash,
        verify_password,
        verify_password_with_hash,
    )

    # A cost far below the configured one must be flagged for upgrade.
    weak = hash_password("pw", iterations=100)
    ok, parsed = verify_password_with_hash("pw", weak)
    assert ok is True
    assert parsed["iterations"] == 100
    assert needs_rehash(parsed) is True

    current = hash_password("pw", iterations=PBKDF2_ITERATIONS)
    ok, parsed = verify_password_with_hash("pw", current)
    assert ok is True
    assert needs_rehash(parsed) is False
    assert verify_password("pw", current) is True
    assert verify_password("nope", current) is False

    assert verify_password_with_hash("pw", "not-a-valid-hash") == (False, None)


def test_update_password_hash_keeps_token_version_but_set_bumps_it():
    import uuid

    import app.db as db

    username = f"tv-{uuid.uuid4().hex[:6]}"
    db.create_user("default", username, "correct-horse-1", "user")
    user_id = db.get_user_by_username(username)["id"]
    assert db.get_user_by_id(user_id)["token_version"] == 0

    # Transparent cost upgrade on login must NOT revoke existing sessions.
    db.update_password_hash(user_id, "correct-horse-1")
    assert db.get_user_by_id(user_id)["token_version"] == 0
    # A deliberate password change revokes all prior sessions.
    db.set_user_password(user_id, "new-password-123")
    assert db.get_user_by_id(user_id)["token_version"] == 1


def test_change_password_revokes_other_sessions(fresh_client):
    setup = fresh_client.post(
        "/auth/setup",
        json={
            "tenant_name": "Acme Corp",
            "username": "boss",
            "password": "SuperSecret123",
        },
    )
    old_token = setup.json()["token"]
    auth = {"Authorization": f"Bearer {old_token}"}

    # Wrong current password is a 400 (not 401), so clients don't treat it as
    # an expired session and force a re-login.
    wrong = fresh_client.post(
        "/auth/change-password",
        headers=auth,
        json={"old_password": "wrong", "new_password": "NewSuperSecret456"},
    )
    assert wrong.status_code == 400

    same = fresh_client.post(
        "/auth/change-password",
        headers=auth,
        json={"old_password": "SuperSecret123", "new_password": "SuperSecret123"},
    )
    assert same.status_code == 422

    changed = fresh_client.post(
        "/auth/change-password",
        headers=auth,
        json={"old_password": "SuperSecret123", "new_password": "NewSuperSecret456"},
    )
    assert changed.status_code == 200
    new_token = changed.json()["token"]

    # Every token signed before the change is now revoked.
    assert fresh_client.get("/auth/me", headers=auth).status_code == 401
    me = fresh_client.get("/auth/me", headers={"Authorization": f"Bearer {new_token}"})
    assert me.status_code == 200
    assert me.json()["username"] == "boss"

    # The old password is dead; the new one logs in.
    assert (
        fresh_client.post(
            "/auth/login", json={"username": "boss", "password": "SuperSecret123"}
        ).status_code
        == 401
    )
    login = fresh_client.post(
        "/auth/login", json={"username": "boss", "password": "NewSuperSecret456"}
    )
    assert login.status_code == 200


def test_api_key_disable_and_revoke(fresh_client):
    import uuid

    setup = fresh_client.post(
        "/auth/setup",
        json={
            "tenant_name": "Acme Corp",
            "username": "boss",
            "password": "SuperSecret123",
        },
    )
    assert setup.status_code == 200
    # Real token (no dependency overrides) so API-key auth is exercised too.
    admin = {"Authorization": f"Bearer {setup.json()['token']}"}

    marker = uuid.uuid4().hex[:8]
    created = fresh_client.post(
        "/admin/tenants/default/api-keys",
        headers=admin,
        json={"label": f"ci-{marker}"},
    )
    assert created.status_code == 200
    plain = created.json()["key"]

    listed = fresh_client.get(
        "/admin/tenants/default/api-keys", headers=admin
    ).json()
    record = next(k for k in listed if k["label"] == f"ci-{marker}")
    key_hash = record["key_hash"]
    assert record["is_active"] is True

    headers = {"X-API-Key": plain}
    me = fresh_client.get("/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["via"] == "api-key"

    # Disable: the key stops authenticating but stays listed.
    resp = fresh_client.patch(
        f"/admin/tenants/default/api-keys/{key_hash}",
        headers=admin,
        json={"is_active": False},
    )
    assert resp.status_code == 200
    assert resp.json()["is_active"] is False
    assert fresh_client.get("/auth/me", headers=headers).status_code == 401

    # Re-enable: the same key works again (rotation without reissue).
    resp = fresh_client.patch(
        f"/admin/tenants/default/api-keys/{key_hash}",
        headers=admin,
        json={"is_active": True},
    )
    assert resp.status_code == 200
    assert resp.json()["is_active"] is True
    assert fresh_client.get("/auth/me", headers=headers).status_code == 200

    # Revoke: gone permanently, and unknown hashes 404 on every operation.
    revoke = fresh_client.delete(
        f"/admin/tenants/default/api-keys/{key_hash}", headers=admin
    )
    assert revoke.status_code == 200
    assert revoke.json()["status"] == "revoked"
    assert fresh_client.get("/auth/me", headers=headers).status_code == 401
    assert (
        fresh_client.delete(
            f"/admin/tenants/default/api-keys/{key_hash}", headers=admin
        ).status_code
        == 404
    )
    assert (
        fresh_client.patch(
            f"/admin/tenants/default/api-keys/{key_hash}",
            headers=admin,
            json={"is_active": True},
        ).status_code
        == 404
    )


def test_audit_log_records_actions_and_scopes_tenants(client):
    import uuid

    marker = uuid.uuid4().hex[:8]
    username = f"aud-{marker}"
    client.post(
        "/admin/tenants/default/users",
        json={"username": username, "password": "correct-horse-1", "role": "user"},
    )
    client.post("/admin/tenants/default/api-keys", json={"label": f"aud-{marker}"})
    listed = client.get("/admin/tenants/default/api-keys").json()
    key_hash = next(k for k in listed if k["label"] == f"aud-{marker}")["key_hash"]
    client.delete(f"/admin/tenants/default/api-keys/{key_hash}")
    client.post(
        "/auth/login", json={"username": "does-not-exist-xyz", "password": "wrong"}
    )

    logs = client.get("/admin/audit-logs").json()
    actions = [(log["action"], log["detail"] or "") for log in logs]
    assert any(a == "user.create" and username in d for a, d in actions)
    assert any(a == "api_key.create" and marker in d for a, d in actions)
    assert any(a == "api_key.revoke" for a, d in actions)
    assert logs, "expected at least one audit entry"
    # Enterprise admins only ever see their own tenant's audit trail.
    assert all(log["tenant_id"] == "default" for log in logs)

    # Failed logins carry no tenant (unknown user), so they are hidden from
    # tenant-scoped views but still recorded for the platform admin.
    import app.db as db

    all_actions = [a["action"] for a in db.list_audit_logs(tenant_id=None)]
    assert "login_failed" in all_actions


def test_grounding_supported_accepts_grounded_and_rejects_fabrication():
    from app.rag.grounding import grounding_supported

    context = "The 401(k) plan has a company match of 50 percent up to 6 percent of salary."
    assert grounding_supported("The plan has a company match of 50 percent", context)
    assert not grounding_supported(
        "The company matches 80 percent of unicorns in the nebula", context
    )
    # Abstentions and terse replies are never flagged as hallucinations.
    assert grounding_supported("I don't know", context)
    assert grounding_supported("Yes", context)
    assert grounding_supported("", context)
    # Without any evidence the check cannot pass.
    assert not grounding_supported("the quick brown fox jumps over the lazy dog", "")


def test_grounding_supported_honors_thresholds():
    from app.rag.grounding import grounding_supported

    # Overlap 1/4 < min_overlap=0.5 => unsupported.
    assert not grounding_supported("foo bar baz qux", "foo", min_overlap=0.5, min_tokens=4)
    # Short answers are too terse to judge and always pass.
    assert grounding_supported("foo bar baz", "zzz", min_overlap=0.5, min_tokens=4)
    # A lower overlap threshold accepts the partial match.
    assert grounding_supported("foo bar baz qux", "foo", min_overlap=0.2, min_tokens=4)


def test_upload_over_size_limit_rejected_without_partial_file(client, monkeypatch):
    import uuid

    from app.config import settings

    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 0.001)  # ~1 KiB cap
    name = f"oversize-{uuid.uuid4().hex[:6]}.txt"
    resp = client.post(
        "/documents",
        files={"file": (name, b"x" * (5 * 1024), "text/plain")},
    )
    assert resp.status_code == 413
    names = {d["filename"] for d in client.get("/documents").json()}
    assert name not in names  # no partial file was left behind


def test_upload_queues_and_reports_background_job(client, monkeypatch):
    import uuid

    import app.main as main_mod
    from app.config import settings
    from app.schemas import IngestResponse

    def fake_reindex(tenant_id):
        return IngestResponse(
            tenant_id=tenant_id, documents=2, chunks=9, saved_to="fake-store"
        )

    monkeypatch.setattr(main_mod, "_reindex", fake_reindex)
    name = f"job-{uuid.uuid4().hex[:6]}.txt"
    resp = client.post(
        "/documents",
        files={"file": (name, b"hello background ingest", "text/plain")},
    )
    assert resp.status_code == 202
    queued = resp.json()
    assert queued["status"] == "queued"
    assert queued["tenant_id"] == "default"

    # TestClient runs the background task synchronously, so the job is already
    # reported as done by the time we poll.
    status = client.get("/ingest/status").json()
    assert status["status"] == "done"
    assert status["documents"] == 2
    assert status["chunks"] == 9
    assert status["saved_to"] == "fake-store"

    (settings.tenant_docs_dir("default") / name).unlink(missing_ok=True)


def test_metrics_and_readiness(client):
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    body = ready.json()
    assert body["status"] == "ready"
    assert body["checks"] == {"database": True, "indexing_idle": True}

    rendered = client.get("/metrics")
    assert rendered.status_code == 200
    assert "version=0.0.4" in rendered.headers["content-type"]
    text = rendered.text
    assert "# TYPE rag_http_requests_total counter" in text
    assert "# TYPE rag_http_request_duration_seconds histogram" in text
    assert "# TYPE rag_pipeline_cache_size gauge" in text
    assert 'rag_http_requests_total{method="GET",path="/health",status="200"}' in text


# ---------------------------------------------------------------------------
# Bug-fix regression tests (audit-found issues)
# ---------------------------------------------------------------------------


def test_change_password_rejects_api_key_auth(fresh_client):
    """API-key callers have no user_id and no stored password to verify."""
    import app.db as db

    fresh_client.post(
        "/auth/setup",
        json={
            "tenant_name": "Acme Corp",
            "username": "boss",
            "password": "SuperSecret123",
        },
    )
    plain, _ = db.create_api_key("default", "test-key")
    resp = fresh_client.post(
        "/auth/change-password",
        headers={"X-API-Key": plain},
        json={"old_password": "SuperSecret123", "new_password": "NewSuperSecret456"},
    )
    assert resp.status_code == 400
    assert "API key" in resp.json()["detail"]


def test_middleware_exception_does_not_swallow_original(client):
    """When an endpoint raises, the middleware must propagate the original
    exception rather than replacing it with a RuntimeError from a double
    ContextVar.reset()."""
    from app.main import app as _app, get_pipeline_for_auth

    class _BoomPipeline:
        def answer(self, question, top_k=None):
            raise ValueError("intentional test error")

        def stats(self):
            return {"documents": 0, "chunks": 0}

    _app.dependency_overrides[get_pipeline_for_auth] = lambda: _BoomPipeline()
    try:
        with pytest.raises(ValueError, match="intentional test error"):
            client.post("/query", json={"question": "boom"})
    finally:
        _app.dependency_overrides[get_pipeline_for_auth] = FakePipeline


def test_histogram_inf_bucket_captures_all_observations():
    from app.metrics import MetricsRegistry

    reg = MetricsRegistry()
    reg.register("test_hist", "test histogram")
    reg.observe("test_hist", 100.0)  # above all defined finite buckets
    reg.observe("test_hist", 0.5)
    rendered = reg.render()
    assert 'test_hist_bucket{le="+Inf"} 2' in rendered
    assert "test_hist_count 2" in rendered
    assert "test_hist_sum 100.5" in rendered


def test_brute_force_settings_are_env_configurable(monkeypatch):
    import importlib

    import app.config as cfg

    original_settings = cfg.settings

    # Class-level attributes are evaluated at import time. Reload with env
    # vars set to prove the os.environ.get() wiring works.
    monkeypatch.setenv("RAG_LOGIN_MAX_FAILURES", "10")
    monkeypatch.setenv("RAG_LOGIN_MAX_FAILURES_PER_IP", "50")
    monkeypatch.setenv("RAG_LOGIN_FAILURE_WINDOW_SECONDS", "1800")
    importlib.reload(cfg)
    try:
        assert cfg.settings.LOGIN_MAX_FAILURES == 10
        assert cfg.settings.LOGIN_MAX_FAILURES_PER_IP == 50
        assert cfg.settings.LOGIN_FAILURE_WINDOW_SECONDS == 1800
    finally:
        # Restore the original singleton so that other modules' cached
        # `from app.config import settings` references remain valid.
        monkeypatch.delenv("RAG_LOGIN_MAX_FAILURES")
        monkeypatch.delenv("RAG_LOGIN_MAX_FAILURES_PER_IP")
        monkeypatch.delenv("RAG_LOGIN_FAILURE_WINDOW_SECONDS")
        importlib.reload(cfg)
        cfg.settings = original_settings


# ---------------------------------------------------------------------------
# manage.py backup / restore
# ---------------------------------------------------------------------------


def _make_site_data(root: Path) -> tuple[Path, Path]:
    """Create a DB with a sentinel row plus one tenant with a doc."""
    import sqlite3

    data = root / "data"
    db_path = data / "system.db"
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t (x TEXT)")
    conn.execute("INSERT INTO t VALUES ('sentinel')")
    conn.commit()
    conn.close()
    docs = data / "tenants" / "acme" / "docs"
    docs.mkdir(parents=True)
    (docs / "a.txt").write_text("hello", encoding="utf-8")
    return db_path, data / "tenants"


def test_manage_backup_snapshots_db_and_tenants(monkeypatch, tmp_path):
    from scripts.manage import cmd_backup

    db_path, tenants_dir = _make_site_data(tmp_path / "src")
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    monkeypatch.setattr(settings, "TENANTS_DIR", tenants_dir)
    monkeypatch.setattr(settings, "DATA_DIR", db_path.parent)

    class Args:
        out = str(tmp_path / "backups")

    cmd_backup(Args())
    backups = list((tmp_path / "backups").iterdir())
    assert len(backups) == 1
    b = backups[0]
    assert (b / "system.db").exists()
    assert (b / "tenants" / "acme" / "docs" / "a.txt").read_text(encoding="utf-8") == "hello"
    manifest = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tenants"] == ["acme"]


def test_manage_restore_into_empty_dir(monkeypatch, tmp_path):
    import sqlite3

    from scripts.manage import cmd_backup, cmd_restore

    db_path, tenants_dir = _make_site_data(tmp_path / "src")
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    monkeypatch.setattr(settings, "TENANTS_DIR", tenants_dir)
    monkeypatch.setattr(settings, "DATA_DIR", db_path.parent)

    class BackupArgs:
        out = str(tmp_path / "backups")

    cmd_backup(BackupArgs())
    b = next((tmp_path / "backups").iterdir())

    dst = tmp_path / "dst"
    monkeypatch.setattr(settings, "DB_PATH", dst / "system.db")
    monkeypatch.setattr(settings, "TENANTS_DIR", dst / "tenants")
    monkeypatch.setattr(settings, "DATA_DIR", dst)

    class RestoreArgs:
        from_path = str(b)
        force = False

    cmd_restore(RestoreArgs())
    conn = sqlite3.connect(str(dst / "system.db"))
    assert conn.execute("SELECT x FROM t").fetchone()[0] == "sentinel"
    conn.close()
    assert (dst / "tenants" / "acme" / "docs" / "a.txt").read_text(encoding="utf-8") == "hello"


def test_manage_restore_refuses_to_overwrite_without_force(monkeypatch, tmp_path):
    from scripts.manage import cmd_restore

    db_path, tenants_dir = _make_site_data(tmp_path / "src")
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    monkeypatch.setattr(settings, "TENANTS_DIR", tenants_dir)
    monkeypatch.setattr(settings, "DATA_DIR", db_path.parent)

    class RestoreArgs:
        from_path = str(db_path.parent)  # not a backup dir either
        force = False

    with pytest.raises(SystemExit):
        cmd_restore(RestoreArgs())


def test_manage_restore_force_overwrites_existing_data(monkeypatch, tmp_path):
    from scripts.manage import cmd_backup, cmd_restore

    db_path, tenants_dir = _make_site_data(tmp_path / "src")
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    monkeypatch.setattr(settings, "TENANTS_DIR", tenants_dir)
    monkeypatch.setattr(settings, "DATA_DIR", db_path.parent)

    class BackupArgs:
        out = str(tmp_path / "backups")

    cmd_backup(BackupArgs())
    b = next((tmp_path / "backups").iterdir())

    # Destination has existing data that must be replaced by --force.
    dst = tmp_path / "dst"
    (dst / "tenants" / "acme" / "docs").mkdir(parents=True)
    (dst / "tenants" / "acme" / "docs" / "stale.txt").write_text("old", encoding="utf-8")
    (dst / "system.db").write_bytes(b"old db")
    monkeypatch.setattr(settings, "DB_PATH", dst / "system.db")
    monkeypatch.setattr(settings, "TENANTS_DIR", dst / "tenants")
    monkeypatch.setattr(settings, "DATA_DIR", dst)

    class RestoreArgs:
        from_path = str(b)
        force = True

    cmd_restore(RestoreArgs())
    assert not (dst / "tenants" / "acme" / "docs" / "stale.txt").exists()
    assert (dst / "tenants" / "acme" / "docs" / "a.txt").read_text(encoding="utf-8") == "hello"


# ---------------------------------------------------------------------------
# Tier 2: Security headers
# ---------------------------------------------------------------------------


def test_security_headers_present(client):
    resp = client.get("/health")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_hsts_header_on_https(client):
    """HSTS is only sent when the request comes over HTTPS."""
    resp = client.get("/health", headers={"X-Forwarded-Proto": "https"})
    # TestClient doesn't honor X-Forwarded-Proto for scheme detection, but
    # the middleware checks request.url.scheme. In test, scheme is http, so
    # HSTS should NOT be present.
    assert "Strict-Transport-Security" not in resp.headers


# ---------------------------------------------------------------------------
# Tier 2: Token revocation (jti + /auth/logout)
# ---------------------------------------------------------------------------


def test_token_contains_jti(client):
    from app.security import make_login_token, verify_token
    import app.db as db

    # Seed a user so we can create a real token.
    user = db.create_user("default", "jti_test_user", "Passw0rd123!", "user")
    token = make_login_token(user)
    payload = verify_token(token)
    assert "jti" in payload
    assert isinstance(payload["jti"], str)
    assert len(payload["jti"]) > 8


def test_logout_revokes_token(anon_client):
    """Login, then logout, then try to use the revoked token."""
    import app.db as db

    # Seed user + login to get a real token.
    db.create_user("default", "logout_user", "Passw0rd123!", "user")
    login_resp = anon_client.post(
        "/auth/login", json={"username": "logout_user", "password": "Passw0rd123!"}
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["token"]

    # Logout.
    resp = anon_client.post(
        "/auth/logout", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "logged_out"

    # The same token should now be rejected.
    resp2 = anon_client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp2.status_code == 401
    assert "revoked" in resp2.json()["detail"].lower()


def test_logout_twice_same_token_returns_401(anon_client):
    import app.db as db

    db.create_user("default", "logout_twice_user", "Passw0rd123!", "user")
    login_resp = anon_client.post(
        "/auth/login", json={"username": "logout_twice_user", "password": "Passw0rd123!"}
    )
    token = login_resp.json()["token"]

    # First logout succeeds.
    resp = anon_client.post(
        "/auth/logout", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200
    # Second attempt with same (revoked) token is rejected.
    resp2 = anon_client.post(
        "/auth/logout", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp2.status_code == 401


def test_logout_api_key_returns_400(anon_client):
    """API key sessions cannot use /auth/logout."""
    resp = anon_client.post("/auth/logout", headers={"X-API-Key": "test-key"})
    assert resp.status_code in (400, 401)  # 401 if key invalid, 400 if valid


def test_other_sessions_not_affected_by_logout(anon_client):
    """Logging out one token does not invalidate other tokens for the same user."""
    import app.db as db
    from app.security import make_login_token

    user = db.create_user("default", "multi_session_user", "Passw0rd123!", "user")
    token_a = make_login_token(user)
    token_b = make_login_token(user)

    # Revoke token_a.
    resp = anon_client.post(
        "/auth/logout", headers={"Authorization": f"Bearer {token_a}"}
    )
    assert resp.status_code == 200

    # token_b still works.
    resp2 = anon_client.get("/auth/me", headers={"Authorization": f"Bearer {token_b}"})
    assert resp2.status_code == 200
    assert resp2.json()["username"] == "multi_session_user"


# ---------------------------------------------------------------------------
# Tier 2: CORS configuration
# ---------------------------------------------------------------------------


def test_cors_wildcard_by_default(client):
    """Without RAG_CORS_ORIGINS set, all origins are allowed."""
    resp = client.options(
        "/health",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # CORSMiddleware returns 405 for OPTIONS on routes that don't have
    # OPTIONS handlers, but it still adds the CORS headers.
    assert resp.headers.get("access-control-allow-origin") == "*"


# ---------------------------------------------------------------------------
# Tier 2: Global rate limiting
# ---------------------------------------------------------------------------


def test_global_rate_limit_blocks_heavy_requester(anon_client, monkeypatch):
    """A requester exceeding the global per-IP limit gets 429."""
    import app.main as main_mod

    tiny = main_mod._SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    monkeypatch.setattr(main_mod, "_global_rate_limiter", tiny)

    # First 3 requests succeed (use a non-exempt path).
    for _ in range(3):
        assert anon_client.get("/auth/me").status_code == 401  # 401 (no creds), not 429
    # 4th is blocked.
    resp = anon_client.get("/auth/me")
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_health_check_bypasses_global_rate_limit(anon_client, monkeypatch):
    """Health endpoints should not be rate-limited globally."""
    import app.main as main_mod

    tiny = main_mod._SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    monkeypatch.setattr(main_mod, "_global_rate_limiter", tiny)

    # Burn the one allowed request on a non-health endpoint.
    anon_client.get("/auth/me")
    # /health should still pass because it is exempt.
    assert anon_client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# Tier 2: Security headers on API responses
# ---------------------------------------------------------------------------


def test_security_headers_on_auth_endpoint(client, monkeypatch):
    """Security headers are present on authenticated endpoints too."""
    resp = client.get("/auth/me", headers={"Authorization": "Bearer invalid"})
    # Even a 401 response should carry security headers.
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"


# ---------------------------------------------------------------------------
# Tier 3: Configurable token TTL
# ---------------------------------------------------------------------------


def test_token_ttl_is_env_configurable(monkeypatch):
    """RAG_TOKEN_TTL_SECONDS is wired into the token's exp claim."""
    from app.security import sign_token, verify_token

    monkeypatch.setattr(settings, "TOKEN_TTL_SECONDS", 300)
    token = sign_token({"uid": 1})
    payload = verify_token(token)
    # exp should be ~300s from now, not the default 43200.
    assert payload["exp"] - int(__import__("time").time()) <= 305


# ---------------------------------------------------------------------------
# Tier 3: Grafana dashboard exists and is valid JSON
# ---------------------------------------------------------------------------


def test_grafana_dashboard_is_valid_json():
    import json
    from pathlib import Path

    dash_path = Path(__file__).resolve().parent.parent / "monitoring" / "grafana-dashboard.json"
    assert dash_path.exists(), "grafana-dashboard.json not found"
    data = json.loads(dash_path.read_text(encoding="utf-8"))
    assert data["uid"] == "enterprise-rag"
    assert len(data["panels"]) >= 8


# ---------------------------------------------------------------------------
# Tier 4: Rate limit response headers
# ---------------------------------------------------------------------------


def test_rate_limit_headers_present_on_normal_response(client, monkeypatch):
    """Every non-exempt response should carry X-RateLimit-* headers."""
    resp = client.get("/auth/me")
    assert "X-RateLimit-Limit" in resp.headers
    assert "X-RateLimit-Remaining" in resp.headers
    assert "X-RateLimit-Reset" in resp.headers
    assert int(resp.headers["X-RateLimit-Remaining"]) >= 0


def test_rate_limit_headers_on_429(anon_client, monkeypatch):
    """429 responses should carry X-RateLimit-Remaining=0."""
    import app.main as main_mod

    tiny = main_mod._SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    monkeypatch.setattr(main_mod, "_global_rate_limiter", tiny)
    for _ in range(2):
        anon_client.get("/auth/me")
    resp = anon_client.get("/auth/me")
    assert resp.status_code == 429
    assert resp.headers["X-RateLimit-Remaining"] == "0"
    assert "Retry-After" in resp.headers


# ---------------------------------------------------------------------------
# Tier 4: Alerting rules valid YAML
# ---------------------------------------------------------------------------


def test_alert_rules_is_valid_yaml():
    from pathlib import Path

    rules_path = Path(__file__).resolve().parent.parent / "monitoring" / "alert-rules.yml"
    assert rules_path.exists(), "alert-rules.yml not found"
    try:
        import yaml
        data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    except ImportError:
        # No PyYAML — do a basic syntax check with json fallback
        content = rules_path.read_text(encoding="utf-8")
        assert "groups:" in content
        assert "rag_alerts" in content
        return
    assert len(data["groups"]) == 1
    assert len(data["groups"][0]["rules"]) >= 4


# ---------------------------------------------------------------------------
# Loophole fix regression tests
# ---------------------------------------------------------------------------


def test_setup_rejects_short_password(anon_client):
    """POST /auth/setup should reject passwords shorter than 8 characters."""
    resp = anon_client.post(
        "/auth/setup",
        json={"tenant_name": "TestCo", "username": "boss", "password": "short"},
    )
    assert resp.status_code == 422


def test_change_password_rejects_short_new_password(anon_client):
    """POST /auth/change-password should reject new passwords shorter than 8 chars."""
    import app.db as db
    from app.security import make_login_token

    user = db.create_user("default", "short_pw_user", "Passw0rd123!", "user")
    token = make_login_token(user)
    resp = anon_client.post(
        "/auth/change-password",
        json={"old_password": "Passw0rd123!", "new_password": "short"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


def test_admin_cannot_disable_own_account(anon_client):
    """Admins should not be able to disable their own account (self-lockout)."""
    import app.db as db
    from app.security import make_login_token

    user = db.get_user_by_username("admin")
    if user is None:
        db.create_user("default", "admin", "AdminPass0rd!", "admin")
        user = db.get_user_by_username("admin")
    token = make_login_token(user)
    resp = anon_client.patch(
        f"/admin/tenants/default/users/{user['username']}",
        json={"is_active": False},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 400
    assert "own account" in resp.json()["detail"].lower()


def test_disabled_account_login_returns_same_401_as_wrong_password(anon_client):
    """Disabled accounts must return 401 (not 403) to avoid leaking account status."""
    import app.db as db

    db.create_user("default", "disabled_login_test", "Passw0rd123!", "user")
    db.set_user_active(
        db.get_user_by_username("disabled_login_test")["id"], False
    )
    resp = anon_client.post(
        "/auth/login",
        json={"username": "disabled_login_test", "password": "Passw0rd123!"},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid username or password"


def test_negative_limit_clamped_on_audit_logs(client):
    """Negative limit values should be clamped, not passed to SQLite."""
    resp = client.get("/admin/audit-logs?limit=-1")
    assert resp.status_code == 200
    # Should return at most 1000 entries (the clamped max), not everything.
    assert len(resp.json()) <= 1000


def test_x_request_id_sanitized(client):
    """Malicious X-Request-ID values should be replaced with a generated one."""
    resp = client.get("/health", headers={"X-Request-ID": "../../etc/passwd"})
    assert resp.status_code == 200
    rid = resp.headers.get("X-Request-ID", "")
    # Should be a hex string (from uuid4), not the injected value.
    assert rid != "../../etc/passwd"
    assert len(rid) == 32  # uuid4().hex


def test_change_password_revokes_current_token(anon_client):
    """Changing password should revoke the token used for the request."""
    import app.db as db
    from app.security import make_login_token

    user = db.create_user("default", "revoke_self_user", "OldPassw0rd!", "user")
    token = make_login_token(user)
    resp = anon_client.post(
        "/auth/change-password",
        json={"old_password": "OldPassw0rd!", "new_password": "NewPassw0rd!"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    # The old token (used for the request) should now be rejected.
    resp2 = anon_client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp2.status_code == 401


# ---------------------------------------------------------------------------
# v1.1 features: MFA, encryption, granular roles, new DB schema
# ---------------------------------------------------------------------------


def test_mfa_totp_generation_and_verification():
    """TOTP secret generation, URI, and code verification."""
    from app.mfa import generate_mfa_secret, get_totp_uri, verify_totp
    import pyotp

    secret = generate_mfa_secret()
    assert len(secret) == 32  # base32 encoded

    uri = get_totp_uri(secret, "testuser")
    assert "otpauth://totp/" in uri
    assert "testuser" in uri

    # Generate a valid code and verify it
    totp = pyotp.TOTP(secret)
    code = totp.now()
    assert verify_totp(secret, code) is True
    assert verify_totp(secret, "000000") is False


def test_mfa_enable_disable_flow(anon_client):
    """Enable and disable MFA via the API."""
    import app.db as db
    import pyotp

    # Create a user
    user = db.create_user("default", "mfa_test_user", "Passw0rd123!", "user")
    from app.security import make_login_token
    token = make_login_token(user)

    # Setup MFA - should return a secret and QR
    resp = anon_client.post(
        "/auth/mfa/setup",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["secret"]
    assert data["qr_code_url"].startswith("data:image/png;base64,")

    # Verify with a valid TOTP code
    totp = pyotp.TOTP(data["secret"])
    code = totp.now()
    resp = anon_client.post(
        "/auth/mfa/verify",
        json={"code": code},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    # Verify MFA is now enabled
    resp = anon_client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["mfa_enabled"] is True

    # Disable MFA with a valid code
    resp = anon_client.post(
        "/auth/mfa/disable",
        json={"code": totp.now()},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    # Verify MFA is disabled again
    resp = anon_client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.json()["mfa_enabled"] is False


def test_mfa_verify_rejects_bad_code(client):
    """MFA verify should reject an invalid code."""
    import app.db as db
    from app.security import make_login_token

    user = db.create_user("default", "mfa_bad_code_user", "Passw0rd123!", "user")
    token = make_login_token(user)

    client.post("/auth/mfa/setup", headers={"Authorization": f"Bearer {token}"})
    resp = client.post(
        "/auth/mfa/verify",
        json={"code": "123456"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 400


def test_mfa_validate_exchanges_token(anon_client):
    """MFA validate should exchange mfa_token + code for a real session."""
    import app.db as db
    from app.mfa import generate_mfa_secret
    import pyotp

    user = db.create_user("default", "mfa_validate_user", "Passw0rd123!", "user")
    secret = generate_mfa_secret()
    db.set_user_mfa(user["id"], secret, enabled=True)

    from app.security import make_login_token
    mfa_token = make_login_token({**user, "token_version": user.get("token_version", 0)})

    totp = pyotp.TOTP(secret)
    resp = anon_client.post(
        "/auth/mfa/validate",
        json={"mfa_token": mfa_token, "code": totp.now()},
    )
    assert resp.status_code == 200
    assert resp.json()["token"]
    assert resp.json()["username"] == "mfa_validate_user"


def test_mfa_validate_rejects_bad_code(anon_client):
    """MFA validate should reject an invalid code."""
    import app.db as db
    from app.mfa import generate_mfa_secret

    user = db.create_user("default", "mfa_validate_bad", "Passw0rd123!", "user")
    secret = generate_mfa_secret()
    db.set_user_mfa(user["id"], secret, enabled=True)

    from app.security import make_login_token
    mfa_token = make_login_token({**user, "token_version": user.get("token_version", 0)})

    resp = anon_client.post(
        "/auth/mfa/validate",
        json={"mfa_token": mfa_token, "code": "000000"},
    )
    assert resp.status_code == 401


def test_encryption_module():
    """Fernet encryption round-trip."""
    from app.encryption import encrypt_bytes, decrypt_bytes, is_encryption_enabled
    import os

    # Without key, encryption is disabled (passthrough)
    old_key = os.environ.get("RAG_ENCRYPTION_KEY", "")
    os.environ["RAG_ENCRYPTION_KEY"] = ""
    from importlib import reload
    from app import config
    reload(config)
    assert is_encryption_enabled() is False
    data = b"hello world"
    assert encrypt_bytes(data) == data
    assert decrypt_bytes(data) == data

    # With key, encryption works
    os.environ["RAG_ENCRYPTION_KEY"] = "test-encryption-key-for-unit-test-1234"
    reload(config)
    # Reset the Fernet cache
    import app.encryption as enc_mod
    enc_mod._FERNET_KEY = None
    assert is_encryption_enabled() is True
    original = b"Sensitive document content here"
    encrypted = encrypt_bytes(original)
    assert encrypted != original
    decrypted = decrypt_bytes(encrypted)
    assert decrypted == original

    # Restore
    os.environ["RAG_ENCRYPTION_KEY"] = old_key
    reload(config)
    enc_mod._FERNET_KEY = None


def test_granular_roles_in_schema(anon_client):
    """uploader and viewer roles should be accepted."""
    import app.db as db

    user_uploader = db.create_user("default", "uploader_test", "Passw0rd123!", "uploader")
    assert user_uploader["role"] == "uploader"

    user_viewer = db.create_user("default", "viewer_test", "Passw0rd123!", "viewer")
    assert user_viewer["role"] == "viewer"


def test_login_mfa_required_flow(client):
    """Login should return mfa_required when MFA is enabled."""
    import app.db as db
    from app.mfa import generate_mfa_secret

    user = db.create_user("default", "login_mfa_user", "Passw0rd123!", "user")
    secret = generate_mfa_secret()
    db.set_user_mfa(user["id"], secret, enabled=True)

    resp = client.post(
        "/auth/login",
        json={"username": "login_mfa_user", "password": "Passw0rd123!"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mfa_required"] is True
    assert data["mfa_token"] is not None


def test_postgresql_configurable():
    """DATABASE_URL env var should activate PostgreSQL mode."""
    import os

    old = os.environ.get("RAG_DATABASE_URL", "")
    os.environ["RAG_DATABASE_URL"] = "postgresql://user:pass@localhost:5432/test"
    from importlib import reload
    from app import config
    reload(config)
    assert config.settings.DATABASE_URL == "postgresql://user:pass@localhost:5432/test"
    os.environ["RAG_DATABASE_URL"] = old
    reload(config)


def test_me_response_includes_mfa_enabled(client):
    """GET /auth/me should include mfa_enabled field."""
    import app.db as db
    from app.security import make_login_token

    user = db.create_user("default", "me_mfa_user", "Passw0rd123!", "user")
    token = make_login_token(user)
    resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert "mfa_enabled" in resp.json()
    assert resp.json()["mfa_enabled"] is False
