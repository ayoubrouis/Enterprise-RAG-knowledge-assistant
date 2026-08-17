"""Database persistence for tenants, users, API keys, query logs, and MFA secrets.

Supports two backends:
  - **PostgreSQL** (production): connection-pooled via ``psycopg2``; real ACID,
    row-level locking, and multi-process safety. Set ``RAG_DATABASE_URL`` to
    enable (e.g. ``postgresql://user:pass@localhost:5432/rag``).
  - **SQLite** (development / tests): the original stdlib backend, zero setup.

The public API is backend-agnostic — every function works identically regardless
of which database is configured. Switching between backends requires only an
env var change and (for PostgreSQL) a running server.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any

from app.config import settings
from app.security import hash_password, new_api_key

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_USE_PG = bool(settings.DATABASE_URL)

# ---------------------------------------------------------------------------
# PostgreSQL pool (lazy singleton)
# ---------------------------------------------------------------------------

_pg_pool = None
_pg_lock = threading.Lock()


def _get_pg_pool():
    global _pg_pool
    if _pg_pool is None:
        import psycopg2.pool

        _pg_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=20,
            dsn=settings.DATABASE_URL,
            options="-c statement_timeout=30000",
        )
        _init_pg_schema()
    return _pg_pool


@contextmanager
def _pg_conn():
    pool = _get_pg_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def _pg_execute(sql: str, params: tuple = (), *, fetch: bool = False, fetchone: bool = False, executemany: bool = False) -> Any:
    with _pg_conn() as conn:
        cur = conn.cursor()
        if executemany:
            cur.executany(sql, params)
            conn.commit()
            return cur
        cur.execute(sql, params)
        if fetchone:
            row = cur.fetchone()
            return dict(row) if row else None
        if fetch:
            return [dict(r) for r in cur.fetchall()]
        return cur


def _pg_execute_returning(sql: str, params: tuple = ()) -> dict | None:
    """Execute a RETURNING query and return the first row as a dict."""
    with _pg_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None


def _init_pg_schema() -> None:
    """Create tables if they don't exist (idempotent)."""
    with _pg_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tenants (
                tenant_id   TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                created_at  DOUBLE PRECISION NOT NULL,
                is_active   BOOLEAN NOT NULL DEFAULT TRUE
            );
            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL PRIMARY KEY,
                tenant_id     TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
                username      TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL CHECK (role IN ('superadmin','admin','user','uploader','viewer')),
                is_active     BOOLEAN NOT NULL DEFAULT TRUE,
                token_version INTEGER NOT NULL DEFAULT 0,
                mfa_secret    TEXT,
                mfa_enabled   BOOLEAN NOT NULL DEFAULT FALSE,
                created_at    DOUBLE PRECISION NOT NULL
            );
            CREATE TABLE IF NOT EXISTS api_keys (
                key_hash   TEXT PRIMARY KEY,
                tenant_id  TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
                label      TEXT NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                is_active  BOOLEAN NOT NULL DEFAULT TRUE
            );
            CREATE TABLE IF NOT EXISTS query_logs (
                id         SERIAL PRIMARY KEY,
                tenant_id  TEXT NOT NULL,
                username   TEXT,
                question   TEXT NOT NULL,
                answer     TEXT,
                latency_ms DOUBLE PRECISION,
                created_at DOUBLE PRECISION NOT NULL
            );
            CREATE TABLE IF NOT EXISTS login_failures (
                id         SERIAL PRIMARY KEY,
                username   TEXT NOT NULL,
                ip         TEXT NOT NULL,
                created_at DOUBLE PRECISION NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_logs (
                id         SERIAL PRIMARY KEY,
                created_at DOUBLE PRECISION NOT NULL,
                actor      TEXT NOT NULL,
                actor_role TEXT,
                tenant_id  TEXT,
                action     TEXT NOT NULL,
                detail     TEXT
            );
            CREATE TABLE IF NOT EXISTS revoked_tokens (
                jti        TEXT PRIMARY KEY,
                user_id    INTEGER,
                revoked_at DOUBLE PRECISION NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_login_failures_user_time
                ON login_failures(username, created_at);
            CREATE INDEX IF NOT EXISTS idx_login_failures_ip_time
                ON login_failures(ip, created_at);
            CREATE INDEX IF NOT EXISTS idx_audit_tenant_time
                ON audit_logs(tenant_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_audit_actor_time
                ON audit_logs(actor, created_at);
            CREATE INDEX IF NOT EXISTS idx_revoked_tokens_user
                ON revoked_tokens(user_id, revoked_at);
        """)


# ---------------------------------------------------------------------------
# SQLite backend (original, unchanged API)
# ---------------------------------------------------------------------------

_DB_LOCK = threading.Lock()
_INIT_LOCK = threading.Lock()
_CONN: sqlite3.Connection | None = None


def _connect() -> sqlite3.Connection:
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _sqlite_conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        with _INIT_LOCK:
            if _CONN is None:
                _CONN = _connect()
                _init_sqlite_schema()
    return _CONN


@contextmanager
def _sqlite_tx():
    conn = _sqlite_conn()
    with _DB_LOCK:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def _init_sqlite_schema() -> None:
    with _sqlite_tx() as conn:
        conn.executescript("""
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
                role          TEXT NOT NULL CHECK (role IN ('superadmin','admin','user','uploader','viewer')),
                is_active     INTEGER NOT NULL DEFAULT 1,
                token_version INTEGER NOT NULL DEFAULT 0,
                mfa_secret    TEXT,
                mfa_enabled   INTEGER NOT NULL DEFAULT 0,
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
            CREATE TABLE IF NOT EXISTS revoked_tokens (
                jti        TEXT PRIMARY KEY,
                user_id    INTEGER,
                revoked_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_revoked_tokens_user
                ON revoked_tokens(user_id, revoked_at);
        """)
        _migrate_sqlite(conn)


def _migrate_sqlite(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='users'"
    ).fetchone()
    if row and "superadmin" not in row["sql"]:
        old_cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
        new_cols = [
            "id", "tenant_id", "username", "password_hash", "role",
            "is_active", "token_version", "mfa_secret", "mfa_enabled", "created_at",
        ]
        select_parts = []
        for c in new_cols:
            if c in old_cols:
                select_parts.append(c)
            elif c == "token_version":
                select_parts.append("0 AS token_version")
            elif c == "mfa_secret":
                select_parts.append("NULL AS mfa_secret")
            elif c == "mfa_enabled":
                select_parts.append("0 AS mfa_enabled")
        conn.executescript(f"""
            CREATE TABLE users_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('superadmin','admin','user','uploader','viewer')),
                is_active INTEGER NOT NULL DEFAULT 1,
                token_version INTEGER NOT NULL DEFAULT 0,
                mfa_secret TEXT,
                mfa_enabled INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            );
            INSERT INTO users_new ({', '.join(new_cols)})
                SELECT {', '.join(select_parts)} FROM users;
            DROP TABLE users;
            ALTER TABLE users_new RENAME TO users;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username);
        """)
    else:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "token_version" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN token_version INTEGER NOT NULL DEFAULT 0")
        if "mfa_secret" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN mfa_secret TEXT")
        if "mfa_enabled" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN mfa_enabled INTEGER NOT NULL DEFAULT 0")


# ---------------------------------------------------------------------------
# Unified query helpers (auto-detect backend)
# ---------------------------------------------------------------------------

def _execute(sql: str, params: tuple = (), *, fetch: bool = False, fetchone: bool = False, executemany: bool = False) -> Any:
    if _USE_PG:
        return _pg_execute(sql, params, fetch=fetch, fetchone=fetchone, executemany=executemany)
    conn = _sqlite_conn()
    with _DB_LOCK:
        cur = conn.execute(sql, params)
        if executemany:
            conn.commit()
            return cur
        if fetchone:
            row = cur.fetchone()
            return dict(row) if row else None
        if fetch:
            return [dict(r) for r in cur.fetchall()]
        conn.commit()
        return cur


def _execute_insert(sql: str, params: tuple = ()) -> dict | None:
    """Execute an INSERT and return the inserted row."""
    if _USE_PG:
        return _pg_execute_returning(sql + " RETURNING *", params)
    conn = _sqlite_conn()
    with _DB_LOCK:
        cur = conn.execute(sql, params)
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row) if row else None


def _tx_execute(sql: str, params: tuple = (), *, fetch: bool = False, fetchone: bool = False) -> Any:
    """Execute within a transaction (for multi-statement operations)."""
    if _USE_PG:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            if fetchone:
                row = cur.fetchone()
                return dict(row) if row else None
            if fetch:
                return [dict(r) for r in cur.fetchall()]
            return cur
    conn = _sqlite_conn()
    with _DB_LOCK:
        cur = conn.execute(sql, params)
        if fetchone:
            row = cur.fetchone()
            return dict(row) if row else None
        if fetch:
            return [dict(r) for r in cur.fetchall()]
        return cur


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def seed_defaults() -> None:
    settings.ensure_tenant_dirs(settings.DEFAULT_TENANT)
    now = time.time()
    if _USE_PG:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO tenants (tenant_id, name, created_at) VALUES (%s, %s, %s) "
                "ON CONFLICT (tenant_id) DO UPDATE SET name = EXCLUDED.name",
                (settings.DEFAULT_TENANT, "Default", now),
            )
            if not settings.ADMIN_PASSWORD:
                return
            cur.execute("SELECT COUNT(*) AS c FROM users WHERE role='superadmin'")
            if cur.fetchone()["c"]:
                return
            cur.execute(
                "INSERT INTO users (tenant_id, username, password_hash, role, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (settings.DEFAULT_TENANT, settings.ADMIN_USERNAME, hash_password(settings.ADMIN_PASSWORD), "superadmin", now),
            )
    else:
        with _sqlite_tx() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO tenants (tenant_id, name, created_at) VALUES (?, ?, ?)",
                (settings.DEFAULT_TENANT, "Default", now),
            )
            if not settings.ADMIN_PASSWORD:
                return
            admin = conn.execute("SELECT COUNT(*) AS c FROM users WHERE role='superadmin'").fetchone()["c"]
            if admin:
                return
            conn.execute(
                "INSERT INTO users (tenant_id, username, password_hash, role, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (settings.DEFAULT_TENANT, settings.ADMIN_USERNAME, hash_password(settings.ADMIN_PASSWORD), "superadmin", now),
            )


def is_bootstrapped() -> bool:
    row = _execute("SELECT COUNT(*) AS c FROM users", fetchone=True)
    return row["c"] > 0


def run_setup(tenant_name: str, username: str, password: str, role: str = "admin") -> dict:
    if is_bootstrapped():
        raise ValueError("Setup already completed")
    settings.ensure_tenant_dirs(settings.DEFAULT_TENANT)
    now = time.time()
    if _USE_PG:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO tenants (tenant_id, name, created_at) VALUES (%s, %s, %s) "
                "ON CONFLICT (tenant_id) DO UPDATE SET name = EXCLUDED.name",
                (settings.DEFAULT_TENANT, tenant_name, now),
            )
            cur.execute("SELECT COUNT(*) AS c FROM users")
            if cur.fetchone()["c"]:
                raise ValueError("Setup already completed")
            cur.execute(
                "INSERT INTO users (tenant_id, username, password_hash, role, created_at) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING *",
                (settings.DEFAULT_TENANT, username, hash_password(password), role, now),
            )
            return dict(cur.fetchone())
    else:
        with _sqlite_tx() as conn:
            conn.execute(
                "INSERT INTO tenants (tenant_id, name, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT(tenant_id) DO UPDATE SET name = excluded.name",
                (settings.DEFAULT_TENANT, tenant_name, now),
            )
            existing = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
            if existing:
                raise ValueError("Setup already completed")
            cur = conn.execute(
                "INSERT INTO users (tenant_id, username, password_hash, role, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (settings.DEFAULT_TENANT, username, hash_password(password), role, now),
            )
            row = conn.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
            return dict(row)


# ---------------------------------------------------------------------------
# Tenants
# ---------------------------------------------------------------------------

def create_tenant(tenant_id: str, name: str) -> dict:
    settings.ensure_tenant_dirs(tenant_id)
    now = time.time()
    if _USE_PG:
        try:
            return _pg_execute_returning(
                "INSERT INTO tenants (tenant_id, name, created_at) VALUES (%s, %s, %s) RETURNING *",
                (tenant_id, name, now),
            )
        except Exception:
            raise ValueError(f"Tenant '{tenant_id}' already exists")
    with _sqlite_tx() as conn:
        try:
            conn.execute("INSERT INTO tenants (tenant_id, name, created_at) VALUES (?, ?, ?)", (tenant_id, name, now))
        except sqlite3.IntegrityError:
            raise ValueError(f"Tenant '{tenant_id}' already exists")
        row = conn.execute("SELECT * FROM tenants WHERE tenant_id = ?", (tenant_id,)).fetchone()
        return dict(row)


def get_tenant(tenant_id: str) -> dict | None:
    return _execute("SELECT * FROM tenants WHERE tenant_id = %s" if _USE_PG else "SELECT * FROM tenants WHERE tenant_id = ?", (tenant_id,), fetchone=True)


def list_tenants() -> list[dict]:
    return _execute("SELECT * FROM tenants ORDER BY created_at", fetch=True)


def set_tenant_active(tenant_id: str, active: bool) -> None:
    if _USE_PG:
        _execute("UPDATE tenants SET is_active = %s WHERE tenant_id = %s", (active, tenant_id))
    else:
        _execute("UPDATE tenants SET is_active = ? WHERE tenant_id = ?", (1 if active else 0, tenant_id))


def count_users(tenant_id: str) -> int:
    col = "c"
    row = _execute(
        "SELECT COUNT(*) AS c FROM users WHERE tenant_id = %s" if _USE_PG else "SELECT COUNT(*) AS c FROM users WHERE tenant_id = ?",
        (tenant_id,), fetchone=True,
    )
    return row["c"]


def count_api_keys(tenant_id: str) -> int:
    row = _execute(
        "SELECT COUNT(*) AS c FROM api_keys WHERE tenant_id = %s" if _USE_PG else "SELECT COUNT(*) AS c FROM api_keys WHERE tenant_id = ?",
        (tenant_id,), fetchone=True,
    )
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
    if get_tenant(tenant_id) is None:
        raise ValueError(f"Tenant '{tenant_id}' does not exist")
    now = time.time()
    if _USE_PG:
        try:
            return _pg_execute_returning(
                "INSERT INTO users (tenant_id, username, password_hash, role, created_at) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING *",
                (tenant_id, username, hash_password(password), role, now),
            )
        except Exception:
            raise ValueError(f"Username '{username}' already exists")
    with _sqlite_tx() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (tenant_id, username, password_hash, role, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (tenant_id, username, hash_password(password), role, now),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"Username '{username}' already exists")
        row = conn.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)


def list_users(tenant_id: str) -> list[dict]:
    return _execute(
        "SELECT * FROM users WHERE tenant_id = %s ORDER BY id" if _USE_PG else "SELECT * FROM users WHERE tenant_id = ? ORDER BY id",
        (tenant_id,), fetch=True,
    )


def get_user_by_username(username: str) -> dict | None:
    return _execute(
        "SELECT * FROM users WHERE username = %s" if _USE_PG else "SELECT * FROM users WHERE username = ?",
        (username,), fetchone=True,
    )


def get_user_by_id(user_id: int) -> dict | None:
    return _execute(
        "SELECT * FROM users WHERE id = %s" if _USE_PG else "SELECT * FROM users WHERE id = ?",
        (user_id,), fetchone=True,
    )


def set_user_active(user_id: int, active: bool) -> None:
    if _USE_PG:
        _execute("UPDATE users SET is_active = %s WHERE id = %s", (active, user_id))
    else:
        _execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if active else 0, user_id))


def set_user_password(user_id: int, password: str) -> None:
    _execute(
        "UPDATE users SET password_hash = %s, token_version = token_version + 1 WHERE id = %s" if _USE_PG
        else "UPDATE users SET password_hash = ?, token_version = token_version + 1 WHERE id = ?",
        (hash_password(password), user_id),
    )


def update_password_hash(user_id: int, password: str) -> None:
    _execute(
        "UPDATE users SET password_hash = %s WHERE id = %s" if _USE_PG
        else "UPDATE users SET password_hash = ? WHERE id = ?",
        (hash_password(password), user_id),
    )


# ---------------------------------------------------------------------------
# MFA (TOTP)
# ---------------------------------------------------------------------------

def set_user_mfa(user_id: int, secret: str, enabled: bool) -> None:
    if _USE_PG:
        _execute("UPDATE users SET mfa_secret = %s, mfa_enabled = %s WHERE id = %s", (secret, enabled, user_id))
    else:
        _execute("UPDATE users SET mfa_secret = ?, mfa_enabled = ? WHERE id = ?", (secret, 1 if enabled else 0, user_id))


def get_user_mfa_secret(user_id: int) -> str | None:
    row = _execute(
        "SELECT mfa_secret FROM users WHERE id = %s" if _USE_PG else "SELECT mfa_secret FROM users WHERE id = ?",
        (user_id,), fetchone=True,
    )
    return row["mfa_secret"] if row else None


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

def create_api_key(tenant_id: str, label: str = "") -> tuple[str, dict]:
    plain, key_hash = new_api_key()
    now = time.time()
    if _USE_PG:
        _pg_execute_returning(
            "INSERT INTO api_keys (key_hash, tenant_id, label, created_at) VALUES (%s, %s, %s, %s) RETURNING *",
            (key_hash, tenant_id, label, now),
        )
    else:
        with _sqlite_tx() as conn:
            conn.execute("INSERT INTO api_keys (key_hash, tenant_id, label, created_at) VALUES (?, ?, ?, ?)", (key_hash, tenant_id, label, now))
    return plain, get_api_key_by_hash(key_hash)


def get_api_key_by_hash(key_hash: str) -> dict | None:
    return _execute(
        "SELECT * FROM api_keys WHERE key_hash = %s" if _USE_PG else "SELECT * FROM api_keys WHERE key_hash = ?",
        (key_hash,), fetchone=True,
    )


def list_api_keys(tenant_id: str) -> list[dict]:
    return _execute(
        "SELECT * FROM api_keys WHERE tenant_id = %s ORDER BY created_at DESC" if _USE_PG
        else "SELECT * FROM api_keys WHERE tenant_id = ? ORDER BY created_at DESC",
        (tenant_id,), fetch=True,
    )


def set_api_key_active(tenant_id: str, key_hash: str, active: bool) -> bool:
    if _USE_PG:
        cur = _pg_execute("UPDATE api_keys SET is_active = %s WHERE key_hash = %s AND tenant_id = %s", (active, key_hash, tenant_id))
        return cur.rowcount > 0
    cur = _execute("UPDATE api_keys SET is_active = ? WHERE key_hash = ? AND tenant_id = ?", (1 if active else 0, key_hash, tenant_id))
    return cur.rowcount > 0


def delete_api_key(tenant_id: str, key_hash: str) -> bool:
    if _USE_PG:
        cur = _pg_execute("DELETE FROM api_keys WHERE key_hash = %s AND tenant_id = %s", (key_hash, tenant_id))
        return cur.rowcount > 0
    cur = _execute("DELETE FROM api_keys WHERE key_hash = ? AND tenant_id = ?", (key_hash, tenant_id))
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Query logs
# ---------------------------------------------------------------------------

def log_query(tenant_id: str, username: str | None, question: str, answer: str, latency_ms: float) -> None:
    _execute(
        "INSERT INTO query_logs (tenant_id, username, question, answer, latency_ms, created_at) VALUES (%s, %s, %s, %s, %s, %s)" if _USE_PG
        else "INSERT INTO query_logs (tenant_id, username, question, answer, latency_ms, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (tenant_id, username, question, answer, latency_ms, time.time()),
    )


def list_logs(tenant_id: str | None = None, limit: int = 50) -> list[dict]:
    if tenant_id:
        return _execute(
            "SELECT * FROM query_logs WHERE tenant_id = %s ORDER BY id DESC LIMIT %s" if _USE_PG
            else "SELECT * FROM query_logs WHERE tenant_id = ? ORDER BY id DESC LIMIT ?",
            (tenant_id, limit), fetch=True,
        )
    return _execute(
        "SELECT * FROM query_logs ORDER BY id DESC LIMIT %s" if _USE_PG
        else "SELECT * FROM query_logs ORDER BY id DESC LIMIT ?",
        (limit,), fetch=True,
    )


# ---------------------------------------------------------------------------
# Login brute-force protection
# ---------------------------------------------------------------------------

def record_login_failure(username: str, ip: str) -> None:
    cutoff = time.time() - settings.LOGIN_FAILURE_WINDOW_SECONDS * 2
    if _USE_PG:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO login_failures (username, ip, created_at) VALUES (%s, %s, %s)", (username, ip, time.time()))
            cur.execute("DELETE FROM login_failures WHERE created_at < %s", (cutoff,))
    else:
        with _sqlite_tx() as conn:
            conn.execute("INSERT INTO login_failures (username, ip, created_at) VALUES (?, ?, ?)", (username, ip, time.time()))
            conn.execute("DELETE FROM login_failures WHERE created_at < ?", (cutoff,))


def count_login_failures(username: str, since: float) -> int:
    row = _execute(
        "SELECT COUNT(*) AS c FROM login_failures WHERE username = %s AND created_at >= %s" if _USE_PG
        else "SELECT COUNT(*) AS c FROM login_failures WHERE username = ? AND created_at >= ?",
        (username, since), fetchone=True,
    )
    return row["c"]


def count_login_failures_by_ip(ip: str, since: float) -> int:
    row = _execute(
        "SELECT COUNT(*) AS c FROM login_failures WHERE ip = %s AND created_at >= %s" if _USE_PG
        else "SELECT COUNT(*) AS c FROM login_failures WHERE ip = ? AND created_at >= ?",
        (ip, since), fetchone=True,
    )
    return row["c"]


def clear_login_failures(username: str) -> None:
    _execute(
        "DELETE FROM login_failures WHERE username = %s" if _USE_PG
        else "DELETE FROM login_failures WHERE username = ?",
        (username,),
    )


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def log_audit(tenant_id: str | None, actor: str, actor_role: str | None, action: str, detail: str | None = None) -> None:
    cutoff = time.time() - settings.AUDIT_LOG_RETENTION_DAYS * 86400
    if _USE_PG:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO audit_logs (created_at, actor, actor_role, tenant_id, action, detail) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (time.time(), actor, actor_role, tenant_id, action, detail),
            )
            cur.execute("DELETE FROM audit_logs WHERE created_at < %s", (cutoff,))
    else:
        with _sqlite_tx() as conn:
            conn.execute(
                "INSERT INTO audit_logs (created_at, actor, actor_role, tenant_id, action, detail) VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), actor, actor_role, tenant_id, action, detail),
            )
            conn.execute("DELETE FROM audit_logs WHERE created_at < ?", (cutoff,))


def list_audit_logs(tenant_id: str | None = None, limit: int = 100) -> list[dict]:
    if tenant_id:
        return _execute(
            "SELECT * FROM audit_logs WHERE tenant_id = %s ORDER BY id DESC LIMIT %s" if _USE_PG
            else "SELECT * FROM audit_logs WHERE tenant_id = ? ORDER BY id DESC LIMIT ?",
            (tenant_id, limit), fetch=True,
        )
    return _execute(
        "SELECT * FROM audit_logs ORDER BY id DESC LIMIT %s" if _USE_PG
        else "SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?",
        (limit,), fetch=True,
    )


# ---------------------------------------------------------------------------
# Token revocation
# ---------------------------------------------------------------------------

def revoke_token(jti: str, user_id: int) -> None:
    if _USE_PG:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO revoked_tokens (jti, user_id, revoked_at) VALUES (%s, %s, %s) ON CONFLICT (jti) DO NOTHING",
                (jti, user_id, time.time()),
            )
    else:
        with _sqlite_tx() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO revoked_tokens (jti, user_id, revoked_at) VALUES (?, ?, ?)",
                (jti, user_id, time.time()),
            )


def is_token_revoked(jti: str) -> bool:
    row = _execute(
        "SELECT 1 FROM revoked_tokens WHERE jti = %s" if _USE_PG
        else "SELECT 1 FROM revoked_tokens WHERE jti = ?",
        (jti,), fetchone=True,
    )
    return row is not None


def prune_revoked_tokens() -> None:
    cutoff = time.time() - settings.TOKEN_TTL_SECONDS
    _execute(
        "DELETE FROM revoked_tokens WHERE revoked_at < %s" if _USE_PG
        else "DELETE FROM revoked_tokens WHERE revoked_at < ?",
        (cutoff,),
    )


# ---------------------------------------------------------------------------
# DB health check (for readiness probe)
# ---------------------------------------------------------------------------

def is_db_reachable() -> bool:
    try:
        if _USE_PG:
            with _pg_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1")
        else:
            conn = _sqlite_conn()
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def close() -> None:
    """Release all DB resources (for graceful shutdown)."""
    global _pg_pool, _CONN
    if _pg_pool:
        _pg_pool.closeall()
        _pg_pool = None
    if _CONN:
        _CONN.close()
        _CONN = None
