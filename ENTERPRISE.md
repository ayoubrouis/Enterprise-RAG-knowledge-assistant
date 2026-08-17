# Enterprise Deployment Guide

How to install, run, update, back up, and monitor the Enterprise RAG Knowledge Assistant
**on-prem inside a customer's network**. Everything is free and open source: the software,
the models, and the inference engines. The customer's documents never leave their premises.

---

## Architecture

One Docker Compose stack per enterprise, deployed in the customer's own network (their
server room, or a VM inside their own cloud account):

```
Customer network
┌───────────────────────────────────────────────────────────────┐
│  docker compose (one stack)                                   │
│                                                               │
│  Browser ──► Streamlit UI ──► FastAPI API ──► LLM backend     │
│   :8501        (login/upload)    :8000     transformers /     │
│                                          Ollama / vLLM        │
│                                              │                │
│                                    data/ + models/ volumes    │
│                                    (tenant docs + FAISS +     │
│                                     auth db + model weights)  │
└───────────────────────────────────────────────────────────────┘
```

- **You (vendor)**: pay nothing per customer. You ship the image(s); updates are
  `docker compose pull` / a build + restart.
- **Customer**: runs everything on their own hardware. Data never leaves their network.

---

## Hardware sizing

| Deployment         | Minimum                          | Recommended                       |
|--------------------|----------------------------------|-----------------------------------|
| transformers (CPU) | 4 vCPU / 8 GB RAM                | 8 vCPU / 16 GB RAM (flan-t5-base) |
| Ollama (CPU)       | 8 vCPU / 16 GB RAM               | 8+ vCPU / 32 GB RAM (7B Q4)       |
| vLLM (GPU)         | NVIDIA T4 / 16 GB VRAM (7B AWQ)  | L4 / A10, 16-24 GB VRAM           |

Disk: index + documents are small (a FAISS index for 10k chunks is a few MB). Budget
10 GB+ for models (flan-t5-base ~1 GB, Qwen-7B AWQ ~5 GB, embedding model ~90 MB).

Supported OS: Linux (recommended) or Windows with Docker Desktop. vLLM requires Linux +
NVIDIA GPU + NVIDIA Container Toolkit.

---

## Install

### 1. Prepare

```bash
# Customer server (or VM). Needs Docker Engine + Compose.
git clone <your-repo> /opt/rag-assistant
cd /opt/rag-assistant
```

### 2. Configure

```bash
cp .env.example .env
```

Set at minimum:

| Variable               | Purpose                                                        |
|------------------------|----------------------------------------------------------------|
| `RAG_SECRET_KEY`       | Long random string signing login tokens. **Required** (>= 16 chars); the API refuses to start without it. |
| `RAG_ADMIN_PASSWORD`   | Leave empty for the first-run setup wizard; set it only to skip it (auto-seeds the platform admin / superadmin). |
| `RAG_LLM_BACKEND`      | `transformers` (default) / `openai`.                           |
| `RAG_LLM_BASE_URL`     | Ollama/vLLM host when using `openai`.                          |
| `RAG_LLM_MODEL`        | Model id / served model name.                                  |
| `RAG_PIPELINE_CACHE_SIZE` | Optional: max tenants whose FAISS indexes stay loaded in RAM (LRU, default 8). Raise on big-RAM boxes with many active tenants. |
| `RAG_DATA_DIR`            | Optional: relocate the whole data folder (tenants/ + SQLite DB) to e.g. a mounted volume. Default: `<project>/data`. |
| `RAG_DB_PATH`             | Optional: override just the SQLite auth database file (defaults to `$RAG_DATA_DIR/system.db`). |
| `RAG_QUERY_RATE_LIMIT_MAX` | Optional: max `/query` calls per caller per sliding window (default 30). Guards the LLM+FAISS endpoint against abuse / runaway cost. |
| `RAG_QUERY_RATE_LIMIT_WINDOW_SECONDS` | Optional: sliding window length for the above (default 60). |
| `RAG_MAX_UPLOAD_MB` | Optional: hard cap on a single uploaded document in MiB (default 50). Oversized uploads get a 413 and leave no partial file. |
| `RAG_PBKDF2_ITERATIONS` | Optional: PBKDF2-HMAC-SHA256 cost for **new** password hashes (default 600000). Old hashes are rehashed on next login — raising this never breaks accounts. |
| `RAG_GROUNDING_CHECK` | Optional: enable the answer-grounding guardrail (default `1`). Set `0` only to disable. |
| `RAG_GROUNDING_MIN_OVERLAP` | Optional: min fraction of answer content words that must appear in the retrieved context (default 0.5). |
| `RAG_GROUNDING_MIN_TOKENS` | Optional: minimum answer length before the grounding check kicks in (default 4). |
| `RAG_AUDIT_LOG_RETENTION_DAYS` | Optional: how long audit rows are kept before pruning on insert (default 365). |
| `RAG_LOGIN_MAX_FAILURES` | Optional: per-username lockout threshold within the rolling window (default 5). |
| `RAG_LOGIN_MAX_FAILURES_PER_IP` | Optional: per-source-IP throttle threshold within the rolling window (default 20). |
| `RAG_LOGIN_FAILURE_WINDOW_SECONDS` | Optional: brute-force rolling window length in seconds (default 900 = 15 min). |

Generate a secret key:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

> The app fails closed: if `RAG_SECRET_KEY` is unset or shorter than 16
> characters, the API process exits with a clear error at startup instead of
> signing tokens with a known default. (Local development only: set
> `RAG_ALLOW_INSECURE_DEV_KEY=1` to use a fixed dev key — never in production.)

### 3. Start

```bash
docker compose up -d --build
```

### 4. First-run setup (register your enterprise)

With `RAG_ADMIN_PASSWORD` left empty (default), the system starts un-bootstrapped. Open
the UI (`http://host:8501`) — the first visitor sees a one-time setup page: company name
+ admin username + password (>= 8 chars). This creates the tenant and its **enterprise
admin** (who manages only that enterprise); afterwards it is sign-in only.

> To skip the wizard in unattended installs, set `RAG_ADMIN_PASSWORD` in `.env` — then a
> **platform admin** (`superadmin`) is auto-created on first boot; use it to onboard your
> enterprise tenant and its enterprise admin.

If the `transformers` backend is used and `models/` is empty, the API downloads the model
on first query (needs internet at install time). For offline installs, pre-download the
weights on a connected machine and copy them into `models/flan-t5-base/`.

### 5. Verify

- UI: `http://<server-ip>:8501` — complete the one-time wizard (creates your enterprise
  + enterprise admin), upload a document, ask a question.
- API: `http://<server-ip>:8000/docs` — try `/auth/setup`, `/auth/login`, `/query`.

---

## Choosing the LLM backend

### transformers (default) — works on any CPU, zero extra services

```yaml
RAG_LLM_BACKEND=transformers
RAG_LLM_MODEL=google/flan-t5-base
```

Fast to set up, ~5-10 s/query on CPU, fine for a small team. Weaker accuracy than a 7B.

### Ollama — CPU or GPU, easy

```bash
docker compose -f docker-compose.yml -f docker-compose.ollama.yml up -d
docker compose exec ollama ollama pull qwen2.5:7b-instruct-q4_K_M
docker compose up -d
```

`RAG_LLM_MODEL` should be the Ollama tag (e.g. `qwen2.5:7b-instruct-q4_K_M`). Ollama
batches requests, so concurrent users don't serialize like the transformers backend.

### vLLM — NVIDIA GPU, best throughput

```bash
docker compose -f docker-compose.yml -f docker-compose.vllm.yml up -d
```

vLLM's continuous batching handles many concurrent users on one GPU (~100-200 tok/s on a
T4 for Qwen-7B Q4). If the AWQ repo pull is gated, set `HF_TOKEN` in `.env`. Model name as
seen by the app is `qwen2.5-7b` (the served name).

---

## Managing tenants, users, and API keys

Two admin tiers, cleanly separated:

- **Platform admin** (`superadmin`) — the app's operator: creates/disable tenants, sees
  all logs, and may create other platform admins. Seeded via `RAG_ADMIN_PASSWORD` or
  `scripts/manage.py ensure --superadmin`.
- **Enterprise admin** (`admin`) — created by the setup wizard (the first account) or by
  any admin; manages **only its own tenant** (users + API keys + documents). It cannot
  see or touch any other tenant.

All of it is available in the admin UI, the admin API (`/admin/...`), or the CLI:

```bash
# Inside the api container (or locally with the venv)
docker compose exec api python scripts/manage.py ensure --superadmin --username ops --password 'OpsPass123'
docker compose exec api python scripts/manage.py create-tenant --id acme --name "Acme Corp"
docker compose exec api python scripts/manage.py create-admin --tenant acme --username boss
docker compose exec api python scripts/manage.py create-user --tenant acme --username alice --role user
docker compose exec api python scripts/manage.py create-api-key --tenant acme --label "prod"
docker compose exec api python scripts/manage.py revoke-api-key --tenant acme --key <key-hash>
```

**Passwords are optional everywhere.** Omit `--password` (or leave the password field
blank in the admin UI / API) and a strong random password is generated and **shown once** —
only its PBKDF2 hash is stored, and it can never be retrieved again (use
`reset-password` if it is lost).

Typical patterns:

- **One department per tenant** on a shared box → separate docs, indexes, and users; the
  platform admin onboard's each department's admin.
- **One tenant per box** (most common) → the wizard's tenant is all you need; upload via UI.

API keys authenticate integrations: `curl -H "X-API-Key: <key>" ...`.

---

## Backups

The entire system state lives in two folders. **Back up these and you can restore the
whole installation** on any machine:

```bash
data/     # tenant docs + FAISS indexes + system.db (auth + query logs)
models/   # model weights (re-downloadable, optional to back up)
```

### Recommended: `manage.py backup / restore`

The CLI creates a WAL-safe SQLite snapshot (via the sqlite3 online-backup API) plus a
copy of every tenant's data into a timestamped directory:

```bash
# Snapshot -> data/backups/backup_<date>_<time>/ (or any --out path)
python scripts/manage.py backup
python scripts/manage.py backup --out /mnt/backups

# List your snapshots
ls data/backups/
```

Restore into a fresh location, or over the current one with `--force` (stop the API
first so it reopens the DB and reloads indexes):

```bash
python scripts/manage.py restore --from data/backups/backup_20260809_191030
python scripts/manage.py restore --from /mnt/backups/backup_20260809_191030 --force
```

Restore refuses to overwrite existing data unless `--force` is given. Both commands
honor `RAG_DATA_DIR` / `RAG_DB_PATH`, so you can restore into a relocated data folder.

### Manual alternative

```bash
tar -czf backup-$(date +%F).tgz data models
# or, to exclude the (re-downloadable) models:
tar -czf backup-$(date +%F).tgz data
```

Restore = stop stack, replace folders, `docker compose up -d`.

---

## Updating

You ship a new version; the customer (or you, over SSH) runs:

```bash
./scripts/update.sh        # or: .\scripts\update.ps1 on Windows
```

This rebuilds/pulls images, restarts the stack, and prunes old images. `data/` and
`models/` are volumes, so documents, indexes, users, and keys are untouched.

For a registry-based rollout (recommended at scale):

1. Tag images as `your-registry.example.com/rag/api:VERSION` in `docker-compose.yml`.
2. Push after CI build.
3. Customers run `docker compose pull && docker compose up -d`.

---

## Monitoring & troubleshooting

- **Audit log**: `GET /admin/audit-logs` (auth: admin) — an append-only trail of
  admin/auth actions (setup, logins and failures, password changes, user/API-key/tenant
  management, uploads). Enterprise admins see only their own tenant; the platform admin
  sees everything (including tenant-less events like failed logins). Retention:
  `RAG_AUDIT_LOG_RETENTION_DAYS` (default 365).
- **Query log**: `GET /admin/logs?tenant_id=&limit=` (auth: admin) — who asked what, the
  answer, and latency.
- **Health**: `GET /health` (liveness) and `GET /health/ready` (readiness: database
  reachable + no ingest job in flight — used by the compose healthcheck).
- **Metrics**: `GET /metrics` exposes Prometheus counters/gauges/histograms
  (`rag_http_requests_total`, `rag_query_latency_seconds`, `rag_index_documents`,
  `rag_active_ingest_jobs`, `rag_pipeline_cache_size`, ...) — point a Prometheus/Grafana
  scrape at it for dashboards and alerting.
- **Structured logs**: JSON-lines output with a `request_id` per request (echoed back in
  the `X-Request-ID` response header) so UI/API requests can be correlated across the
  stack: `docker compose logs -f api | jq -r '.request_id, .msg'`.
- **Container logs**: `docker compose logs -f api` / `docker compose logs -f ui`.
- **Statistics**: `GET /stats` per tenant.
- **Ingest jobs**: `GET /ingest/status` shows whether a re-index is `idle`, `queued`,
  `running`, `done`, or `failed` for the current tenant.

Common issues:

| Symptom | Fix |
|---|---|
| Sign-in page asks for credentials but you never set them | Setup wasn't run. Complete the one-time wizard at `http://host:8501` (open it fresh), or set `RAG_ADMIN_PASSWORD` in `.env` and re-create the stack. |
| Query returns "No documents indexed" | Upload documents first (UI sidebar or `POST /documents`). |
| Upload returns 413 | The file exceeds `RAG_MAX_UPLOAD_MB` (default 50). Raise it in `.env` if the cap is too low. |
| Every session logs out after a password change | Expected: changing a password revokes all other sessions (tokens signed before the change stop working). |
| Transformers model downloads at runtime | Pre-download into `models/flan-t5-base/` for offline installs. |
| Slow answers | Switch to Ollama/vLLM with a 7B model, or use `flan-t5-large`. |

---

## Security checklist

- Set a strong `RAG_SECRET_KEY` (tokens are signed with it; the API refuses to start
  without one).
- Login is brute-force protected out of the box: after `LOGIN_MAX_FAILURES` (default 5)
  failed attempts in a 15-minute rolling window an account is locked, and a source IP
  exceeding `LOGIN_MAX_FAILURES_PER_IP` (default 20) is throttled. A successful login
  clears that account's history; an admin can lift a lockout with
  `scripts/manage.py reset-password --username <user>` (password reset clears failures).
- Changing a password **revokes every other session instantly** (each login token is bound
  to the user's token version). Users rotate their own password in the UI sidebar or via
  `POST /auth/change-password`; the response carries a fresh token for the current session.
- API keys can be **disabled** (temporarily, `PATCH`) or **revoked** (permanently,
  `DELETE`) from the admin UI/API/CLI. Rotate credentials by disabling the old key and
  issuing a new one — the old key stops authenticating immediately.
- Every admin/auth action lands in the **audit log** (`GET /admin/audit-logs`): setup,
  successful and failed logins, password changes, user/API-key/tenant management, and
  document uploads/deletes. It is append-only and pruned only by retention age.
- Uploads are capped (`RAG_MAX_UPLOAD_MB`, default 50 MiB) — a runaway upload cannot fill
  the disk, and oversized files are rejected with 413 leaving no partial file.
- A grounding guardrail validates each LLM answer against the retrieved context before it
  is shown; unsupported answers are replaced with "I don't know." rather than surfaced.
- Complete the first-run wizard with a strong admin password; change passwords on demand.
- Keep the platform admin (`superadmin`) account separate from enterprise admins — its
  credentials unlock every tenant.
- Enterprise admins are hard-scoped: `GET /admin/tenants/{id}/users` and friends return
  `403 Not your tenant` when an admin targets a tenant that isn't theirs.
- Put the stack behind a reverse proxy with TLS if exposed beyond the LAN (Caddy/Nginx,
  free). Bind the UI/API to the LAN or VPN, not the public internet.
- The auth DB stores only PBKDF2 password hashes and SHA-256 API key hashes — no
  plaintext credentials.
- Tenants are isolated on disk; enabling multi-tenancy on one box keeps each tenant's
  documents and index separate.

---

## Data residency

All processing is local: documents, embeddings, indexes, and inference run on the
customer's own hardware. No data is sent to any third-party service. The only network
access is optional one-time model downloads at install time.

---

## Upgrade from v1 (single-tenant)

Run the migration once (existing `data/docs/` and `data/vectorstore/` are moved into the
default tenant):

```bash
docker compose exec api python scripts/migrate_legacy_data.py
```

Then sign in with the same admin flow (v1 had no auth; the wizard creates your first
admin, or set `RAG_ADMIN_PASSWORD` / run `scripts/manage.py ensure`).
