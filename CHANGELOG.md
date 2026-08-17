# Changelog

## v2.1.0 — Major Upgrade (v1.1 branch)

Adds PostgreSQL, SSO/OIDC/LDAP, TOTP-based MFA, at-rest encryption, CI/CD,
and expanded role definitions. **93 tests**, all passing.

### PostgreSQL support

- `app/db.py` rewritten with dual-backend: PostgreSQL (`psycopg2.pool.ThreadedConnectionPool`, 2–20 conns) + SQLite fallback.
- Set `RAG_DATABASE_URL=postgresql://user:pass@localhost:5432/rag` to enable PostgreSQL.
- Thread-safe pool with checkout/return, `statement_timeout=30s`, idempotent schema creation.
- All CRUD functions work identically on both backends; switching requires only an env var change.

### SSO / OIDC / LDAP

- `app/sso.py` — OAuth2/OIDC (discovery doc, authorization URL, code exchange via `httpx`) + LDAP (`ldap3` bind/search/authenticate).
- `sso_authenticate()` unified entry point; `provision_sso_user()` auto-creates local users from SSO claims.
- Endpoints: `GET /auth/sso/login`, `GET /auth/sso/callback`.

### Multi-Factor Authentication (TOTP)

- `app/mfa.py` — TOTP generation/verification via `pyotp`, QR codes via `qrcode[pil]`.
- `enable_mfa()` / `disable_mfa()` / `verify_totp()` with ±1 window for clock skew.
- Endpoints: `POST /auth/mfa/setup`, `POST /auth/mfa/verify`, `POST /auth/mfa/disable`, `POST /auth/mfa/validate`.
- Login flow supports `mfa_required` response; `mfa_token` exchanges with TOTP code.

### At-rest encryption

- `app/encryption.py` — Fernet (AES-128-CBC + HMAC-SHA256) via `cryptography`.
- PBKDF2 key derivation from `RAG_ENCRYPTION_KEY`; transparent encrypt-on-upload, decrypt-on-read.
- Set `RAG_ENCRYPTION_KEY` to enable; empty (default) = plaintext passthrough.
- Documents show encryption status in API responses (`encrypted` field).

### Granular roles

- Added `uploader` and `viewer` roles (5 total: `superadmin`, `admin`, `user`, `uploader`, `viewer`).
- Schema CHECK constraint updated; SQLite migration rebuilds users table when outdated.

### CI/CD

- `.github/workflows/ci.yml` — ruff lint + pytest + Docker build on push to `main`/`v1.1` and PRs.
- Python 3.11/3.12 matrix.

### Docker Compose

- Added `postgres` service (16-alpine, healthcheck, pgdata volume).
- API depends on postgres with healthcheck; all new env vars passed through.

### Dependencies added

`psycopg2-binary`, `pyotp`, `qrcode[pil]`, `cryptography`, `httpx`.

### Bug fixes

- `test_mfa_enable_disable_flow` — fixed stale auth dependency override (now uses `anon_client`).
- `test_encryption_module` — `encryption.py` reads `ENCRYPTION_KEY` from env directly (avoids stale singleton after `reload(config)`).
- `_migrate_sqlite()` properly rebuilds users table when role CHECK constraint is outdated.
- `conftest.py` removes `RAG_DATABASE_URL` to ensure tests use SQLite.

### New tests (12)

MFA flow, MFA bad code, MFA validate, MFA validate bad code, encryption round-trip, granular roles, login MFA flow, PostgreSQL config, `me` mfa_enabled.

---

## v2.0.0 — Enterprise Hardening

Major release transforming the single-tenant prototype into a production-ready,
sellable on-prem multi-tenant RAG system. **83 tests**, all passing.

### Tier 1 — Core Hardening

**Authentication & authorization:**
- Two-tier RBAC: platform admin (`superadmin`) + enterprise admin (`admin`) + employee (`user`)
- PBKDF2-HMAC-SHA256 at 600k iterations (stdlib-only, no paid deps)
- Transparent rehash: old weak hashes are upgraded on next successful login
- Brute-force protection: per-username lockout (5 failures / 15 min) + per-IP throttle (20 / 15 min)
- `RAG_SECRET_KEY` required (>= 16 chars); app refuses to start without it
- API-key authentication (`X-API-Key`) with per-tenant scoping

**Security:**
- Per-caller rate limiting on `/query` (configurable window + max)
- Configurable CORS (`RAG_CORS_ORIGINS`)
- Upload size cap (`RAG_MAX_UPLOAD_MB`, default 50 MiB)
- Tenant-scoped document paths (physical isolation on disk)

**Observability:**
- JSON structured logs with `X-Request-ID` correlation
- Prometheus metrics (`/metrics`) with HTTP, query, ingest, and cache counters/histograms
- `/health/ready` readiness probe (database + indexing status)

**Pipeline:**
- Grounding guardrail: lexical overlap check rejects unsupported LLM answers
- Background ingest jobs (202 + pollable `/ingest/status`)

### Tier 2 — Security Hardening

**Token revocation:**
- Login tokens now carry a `jti` (unique token ID)
- `revoked_tokens` SQLite table for permanent revocation
- `POST /auth/logout` revokes the current session without affecting others
- Password change revokes all other sessions (token-version bump + jti list)

**Hardened networking:**
- Global per-IP rate limiter (120 req/min sliding window, configurable via `RAG_GLOBAL_RATE_LIMIT_MAX`)
- CORS allowlist (env-configurable)
- Security headers on every response: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, `Strict-Transport-Security` (HTTPS only)

**Docker hardening:**
- Non-root container (`read_only`, `cap_drop: ALL`, `no-new-privileges`)
- Only `/tmp` writable (tmpfs)

**CLI & operations:**
- `scripts/backup.sh` and `scripts/restore.sh` for WAL-safe backups
- `scripts/manage.py`: token-version bump on password reset, API-key revoke command

**Tests:** 56 → 71

### Tier 3 — Monitoring & Operations

**Configurable token TTL:**
- `RAG_TOKEN_TTL_SECONDS` env var (default 43200 = 12 h)
- Tokens expire automatically; change-password or logout for early revocation

**Automated backups:**
- `scripts/cron-entrypoint.sh` — cron sidecar entrypoint
- Docker Compose `backup` profile: `docker compose --profile backup up -d backup`
- `BACKUP_CRON` env var (default `0 2 * * *` = daily at 2 AM)

**Grafana dashboard:**
- `monitoring/grafana-dashboard.json` — 8 panels (HTTP RPS, latency, query latency, ingest jobs, cache, token TTL)

**Tests:** 71 → 73

### Tier 4 — Enterprise Polish

**Rate-limit headers:**
- `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` on all non-exempt responses

**Prometheus alerting rules:**
- `monitoring/alert-rules.yml` — 5 alerts (high error rate, slow queries, low cache hit rate, high ingest failure rate, token nearing expiry)

**TLS reverse proxy:**
- `Caddyfile` — Caddy-based TLS termination
- `docker-compose.tls.yml` — Compose overlay for Caddy
- Docs: certificate placement, production TLS setup

**Tests:** 73 → 76

### Audit Fixes (post-Tier 4)

**Critical:**
- Login returns 401 (not 403) for disabled accounts — prevents information leak that distinguishes disabled from non-existent accounts

**Auth:**
- `/auth/setup` enforces password minimum length (>= 8 chars) at both schema and handler level
- `/auth/change-password` validates new password length; revokes the current session's `jti` (prevents stale token reuse after password change)
- Admin self-disable guard: `PATCH /admin/tenants/{id}/users/{username}` rejects `is_active: false` on the caller's own account (prevents admin lockout)
- `create_user()` raises `ValueError` if tenant doesn't exist (clear error vs misleading UNIQUE constraint)

**Thread safety:**
- `get_conn()` uses a separate `_INIT_LOCK` to avoid deadlock with `_DB_LOCK` during concurrent first-access

**Data hygiene:**
- Rate limiter prunes empty deques from `_hits` dict (prevents unbounded memory growth)
- `prune_revoked_tokens()` called on startup (expired tokens cleaned immediately, not on next insert)
- X-Request-ID sanitized: non-alphanumeric characters stripped, max 64 chars (prevents log injection / header abuse)
- Admin audit-log `limit` clamped to `[1, 1000]` (prevents negative-value bypass that dumps entire log)

**Streamlit UI:**
- Sign-out calls `POST /auth/logout` server-side before clearing session (server-side revocation, not just client-side cookie drop)
- Admin panel handles non-200 API responses (no more `TypeError` crash on list users / list API keys)
- Document delete shows error message on failure instead of silently swallowing exceptions

**Tests:** 76 → 83
