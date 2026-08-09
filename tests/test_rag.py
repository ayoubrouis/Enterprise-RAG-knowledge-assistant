"""Unit + API tests. No model downloads and no network required."""

from __future__ import annotations

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


ADMIN_CONTEXT = AuthContext(tenant_id="default", username="tester", role="admin", user_id=1)
USER_CONTEXT = AuthContext(tenant_id="default", username="tester", role="user", user_id=1)
SUPERADMIN_CONTEXT = AuthContext(
    tenant_id="default", username="platform", role="superadmin", user_id=2
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
        db.get_conn()
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

    import app.db as db

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
    assert all(l["tenant_id"] == "default" for l in logs)


def test_superadmin_sees_all_logs(super_client):
    logs = super_client.get("/admin/logs").json()
    tenant_ids = {l["tenant_id"] for l in logs}
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
    assert login.status_code == 403  # account disabled


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
