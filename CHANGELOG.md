# Changelog

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
