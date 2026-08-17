"""SQLite persistence for tenants, users, API keys, and query logs.

Uses only the standard library so the project keeps zero paid/commercial deps.
SQLite gives transactional safety and safe concurrent access without running a
separate database server - perfect for an on-prem single-node deployment.

All writes are serialized through a module-level lock (a single connection in
WAL mode); this is more than fast enough for admin/user management, while
per-tenant FAISS indexes and the models themselves live on the filesystem.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager

from app.config import settings
from app.security import hash_password, new_api_key

_DB_LOCK = threading.Lock()
_CONN: sqlite3.Connection | None = None


def _connect() -> sqlite3.Connection:
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        _CONN = _connect()
        _init_schema()
    return _CONN


@contextmanager
def tx():
    conn = get_conn()
    with _DB_LOCK:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def _init_schema() -> None:
    with tx() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tenants (
                tenant_id   TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                created_at  REAL NOT NULL,
                is_active   INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id     TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
                username      TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL CHECK (role IN ('superadmin','admin','user')),
                is_active     INTEGER NOT NULL DEFAULT 1,
                token_version INTEGER NOT NULL DEFAULT 0,
                created_at    REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS api_keys (
                key_hash   TEXT PRIMARY KEY,
                tenant_id  TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
                label      TEXT NOT NULL,
                created_at REAL NOT NULL,
                is_active  INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS query_logs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id  TEXT NOT NULL,
                username   TEXT,
                question   TEXT NOT NULL,
                answer     TEXT,
                latency_ms REAL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS login_failures (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT NOT NULL,
                ip         TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_logs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                actor      TEXT NOT NULL,
                actor_role TEXT,
                tenant_id  TEXT,
                action     TEXT NOT NULL,
                detail     TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_login_failures_user_time
                ON login_failures(username, created_at);
            CREATE INDEX IF NOT EXISTS idx_login_failures_ip_time
                ON login_failures(ip, created_at);
            CREATE INDEX IF NOT EXISTS idx_audit_tenant_time
                ON audit_logs(tenant_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_audit_actor_time
                ON audit_logs(actor, created_at);
            """
        )
        _migrate_role_column(conn)
        _migrate_users_schema(conn)


def _migrate_role_column(conn: sqlite3.Connection) -> None:
    """Rebuild the users table when it still has the pre-superadmin role CHECK."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='users'"
    ).fetchone()
    if row is None or "superadmin" in row["sql"]:
        return
    conn.executescript(
        """
        CREATE TABLE users_new (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id     TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            username      TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL CHECK (role IN ('superadmin','admin','user')),
            is_active     INTEGER NOT NULL DEFAULT 1,
            token_version INTEGER NOT NULL DEFAULT 0,
            created_at    REAL NOT NULL
        );
        INSERT INTO users_new (id, tenant_id, username, password_hash, role, is_active, created_at)
            SELECT id, tenant_id, username, password_hash, role, is_active, created_at FROM users;
        DROP TABLE users;
        ALTER TABLE users_new RENAME TO users;
        """
    )


def _migrate_users_schema(conn: sqlite3.Connection) -> None:
    """Add the token_version column to users tables created before it existed."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "token_version" not in cols:
        conn.execute(
            "ALTER TABLE users ADD COLUMN token_version INTEGER NOT NULL DEFAULT 0"
        )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def seed_defaults() -> None:
    """Ensure the default tenant exists.

    Only auto-creates a platform admin (superadmin) when RAG_ADMIN_PASSWORD is
    set (unattended / Docker deploys). Otherwise the system stays un-bootstrapped
    and the first-run setup wizard (/auth/setup) creates the first enterprise
    admin interactively.
    """
    settings.ensure_tenant_dirs(settings.DEFAULT_TENANT)
    with tx() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO tenants (tenant_id, name, created_at) VALUES (?, ?, ?)",
            (settings.DEFAULT_TENANT, "Default", time.time()),
        )
        if not settings.ADMIN_PASSWORD:
            return  # first-run wizard will create the first admin user
        admin = conn.execute(
            "SELECT COUNT(*) AS c FROM users WHERE role='superadmin'"
        ).fetchone()["c"]
        if admin:
            return
        conn.execute(
            "INSERT INTO users (tenant_id, username, password_hash, role, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                settings.DEFAULT_TENANT,
                settings.ADMIN_USERNAME,
                hash_password(settings.ADMIN_PASSWORD),
                "superadmin",
                time.time(),
            ),
        )


def is_bootstrapped() -> bool:
    """True once at least one user exists (i.e. setup has been completed)."""
    with tx() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
    return row["c"] > 0


def run_setup(tenant_name: str, username: str, password: str, role: str = "admin") -> dict:
    """Create the default tenant and its first user (an enterprise admin by
    default; pass ``role="superadmin"`` to seed a platform admin instead).

    Allowed only when no user exists yet; raising ``ValueError`` otherwise
    (and rolling back so a concurrent setup cannot create two admins).
    """
    if is_bootstrapped():
        raise ValueError("Setup already completed")
    settings.ensure_tenant_dirs(settings.DEFAULT_TENANT)
    with tx() as conn:
        conn.execute(
            "INSERT INTO tenants (tenant_id, name, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(tenant_id) DO UPDATE SET name = excluded.name",
            (settings.DEFAULT_TENANT, tenant_name, time.time()),
        )
        # Re-check inside the transaction to close the TOCTOU gap.
        existing = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
        if existing:
            raise ValueError("Setup already completed")
        cur = conn.execute(
            "INSERT INTO users (tenant_id, username, password_hash, role, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (settings.DEFAULT_TENANT, username, hash_password(password), role, time.time()),
        )
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return dict(row)


# ---------------------------------------------------------------------------
# Tenants
# ---------------------------------------------------------------------------

def create_tenant(tenant_id: str, name: str) -> dict:
    settings.ensure_tenant_dirs(tenant_id)
    with tx() as conn:
        try:
            conn.execute(
                "INSERT INTO tenants (tenant_id, name, created_at) VALUES (?, ?, ?)",
                (tenant_id, name, time.time()),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"Tenant '{tenant_id}' already exists")
        row = conn.execute(
            "SELECT * FROM tenants WHERE tenant_id = ?", (tenant_id,)
        ).fetchone()
    return dict(row)


def get_tenant(tenant_id: str) -> dict | None:
    with tx() as conn:
        row = conn.execute(
            "SELECT * FROM tenants WHERE tenant_id = ?", (tenant_id,)
        ).fetchone()
    return dict(row) if row else None


def list_tenants() -> list[dict]:
    with tx() as conn:
        rows = conn.execute("SELECT * FROM tenants ORDER BY created_at").fetchall()
    return [dict(r) for r in rows]


def set_tenant_active(tenant_id: str, active: bool) -> None:
    with tx() as conn:
        conn.execute(
            "UPDATE tenants SET is_active = ? WHERE tenant_id = ?",
            (1 if active else 0, tenant_id),
        )


def count_users(tenant_id: str) -> int:
    with tx() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM users WHERE tenant_id = ?", (tenant_id,)
        ).fetchone()
    return row["c"]


def count_api_keys(tenant_id: str) -> int:
    with tx() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM api_keys WHERE tenant_id = ?", (tenant_id,)
        ).fetchone()
    return row["c"]


def count_documents(tenant_id: str) -> int:
    docs_dir = settings.tenant_docs_dir(tenant_id)
    if not docs_dir.exists():
        return 0
    return sum(1 for p in docs_dir.iterdir() if p.is_file())


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def create_user(tenant_id: str, username: str, password: str, role: str = "user") -> dict:
    with tx() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (tenant_id, username, password_hash, role, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (tenant_id, username, hash_password(password), role, time.time()),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"Username '{username}' already exists")
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return dict(row)


def list_users(tenant_id: str) -> list[dict]:
    with tx() as conn:
        rows = conn.execute(
            "SELECT * FROM users WHERE tenant_id = ? ORDER BY id", (tenant_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_user_by_username(username: str) -> dict | None:
    with tx() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with tx() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def set_user_active(user_id: int, active: bool) -> None:
    with tx() as conn:
        conn.execute(
            "UPDATE users SET is_active = ? WHERE id = ?",
            (1 if active else 0, user_id),
        )


def set_user_password(user_id: int, password: str) -> None:
    """Reset a user's password and bump their token version so every login
    token signed before this change stops working (sessions are revoked)."""
    with tx() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ?, token_version = token_version + 1 "
            "WHERE id = ?",
            (hash_password(password), user_id),
        )


def update_password_hash(user_id: int, password: str) -> None:
    """Rehash an already-verified password at the current cost without bumping
    the token version (used on login when a stored hash is older/weaker)."""
    with tx() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(password), user_id),
        )


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

def create_api_key(tenant_id: str, label: str = "") -> tuple[str, dict]:
    """Create an API key. Returns (plain_key, record); plain shown only once."""
    plain, key_hash = new_api_key()
    with tx() as conn:
        conn.execute(
            "INSERT INTO api_keys (key_hash, tenant_id, label, created_at) VALUES (?, ?, ?, ?)",
            (key_hash, tenant_id, label, time.time()),
        )
    return plain, get_api_key_by_hash(key_hash)


def get_api_key_by_hash(key_hash: str) -> dict | None:
    with tx() as conn:
        row = conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)
        ).fetchone()
    return dict(row) if row else None


def list_api_keys(tenant_id: str) -> list[dict]:
    with tx() as conn:
        rows = conn.execute(
            "SELECT * FROM api_keys WHERE tenant_id = ? ORDER BY created_at DESC",
            (tenant_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def set_api_key_active(tenant_id: str, key_hash: str, active: bool) -> bool:
    """Enable/disable an API key. Returns False when no such key exists in the
    tenant (the caller should 404)."""
    with tx() as conn:
        cur = conn.execute(
            "UPDATE api_keys SET is_active = ? WHERE key_hash = ? AND tenant_id = ?",
            (1 if active else 0, key_hash, tenant_id),
        )
    return cur.rowcount > 0


def delete_api_key(tenant_id: str, key_hash: str) -> bool:
    """Permanently remove an API key. Returns False when it did not exist in
    the tenant (the caller should 404)."""
    with tx() as conn:
        cur = conn.execute(
            "DELETE FROM api_keys WHERE key_hash = ? AND tenant_id = ?",
            (key_hash, tenant_id),
        )
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Query logs (lightweight observability)
# ---------------------------------------------------------------------------

def log_query(
    tenant_id: str, username: str | None, question: str, answer: str, latency_ms: float
) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT INTO query_logs (tenant_id, username, question, answer, latency_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (tenant_id, username, question, answer, latency_ms, time.time()),
        )


def list_logs(tenant_id: str | None = None, limit: int = 50) -> list[dict]:
    with tx() as conn:
        if tenant_id:
            rows = conn.execute(
                "SELECT * FROM query_logs WHERE tenant_id = ? ORDER BY id DESC LIMIT ?",
                (tenant_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM query_logs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Login brute-force protection
# ---------------------------------------------------------------------------

def record_login_failure(username: str, ip: str) -> None:
    """Record a failed sign-in. Old rows are pruned opportunistically so the
    table stays bounded to ~2 windows of data."""
    cutoff = time.time() - settings.LOGIN_FAILURE_WINDOW_SECONDS * 2
    with tx() as conn:
        conn.execute(
            "INSERT INTO login_failures (username, ip, created_at) VALUES (?, ?, ?)",
            (username, ip, time.time()),
        )
        conn.execute("DELETE FROM login_failures WHERE created_at < ?", (cutoff,))


def count_login_failures(username: str, since: float) -> int:
    with tx() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM login_failures WHERE username = ? AND created_at >= ?",
            (username, since),
        ).fetchone()
    return row["c"]


def count_login_failures_by_ip(ip: str, since: float) -> int:
    with tx() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM login_failures WHERE ip = ? AND created_at >= ?",
            (ip, since),
        ).fetchone()
    return row["c"]


def clear_login_failures(username: str) -> None:
    """Wipe failure history for a user who just signed in successfully."""
    with tx() as conn:
        conn.execute("DELETE FROM login_failures WHERE username = ?", (username,))


# ---------------------------------------------------------------------------
# Audit log (append-only record of admin/auth actions)
# ---------------------------------------------------------------------------

def log_audit(
    tenant_id: str | None,
    actor: str,
    actor_role: str | None,
    action: str,
    detail: str | None = None,
) -> None:
    """Append an immutable audit entry. Old rows are pruned opportunistically
    (on insert) beyond the configured retention window so the table stays
    bounded while remaining append-only in normal operation."""
    cutoff = time.time() - settings.AUDIT_LOG_RETENTION_DAYS * 86400
    with tx() as conn:
        conn.execute(
            "INSERT INTO audit_logs (created_at, actor, actor_role, tenant_id, action, detail) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), actor, actor_role, tenant_id, action, detail),
        )
        conn.execute("DELETE FROM audit_logs WHERE created_at < ?", (cutoff,))


def list_audit_logs(tenant_id: str | None = None, limit: int = 100) -> list[dict]:
    with tx() as conn:
        if tenant_id:
            rows = conn.execute(
                "SELECT * FROM audit_logs WHERE tenant_id = ? ORDER BY id DESC LIMIT ?",
                (tenant_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]
