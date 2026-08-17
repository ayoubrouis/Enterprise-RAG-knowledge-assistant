# Enterprise RAG Knowledge Assistant

A production-style, **100% local**, **multi-tenant** Retrieval-Augmented Generation system
that ingests internal documents (PDF, TXT, MD, DOCX), stores embeddings in a FAISS vector
index, and answers natural-language questions with **source-cited responses**.

No paid APIs. No subscriptions. No cloud services. Everything runs on your hardware with
free, open-source models. Designed to be sold/installed **on-prem per enterprise** (see
[ENTERPRISE.md](ENTERPRISE.md)).

---

## Features

- **Multi-tenant** — every tenant is fully isolated on disk
  (`data/tenants/<tenant_id>/docs/` + `vectorstore/`); retrieval can never cross tenants.
- **Authentication** — JWT-style login tokens + per-tenant API keys. Passwords hashed
  with PBKDF2-HMAC-SHA256 at 600k iterations (stdlib only, no paid deps); old weak hashes
  are transparently rehashed on the next successful login. Login is brute-force protected:
  accounts are locked after repeated failures and source IPs are throttled (rolling window).
- **Password rotation & session revocation** — users can change their own password
  (UI or `POST /auth/change-password`); the change instantly revokes every other session.
  Individual sessions can be revoked via `POST /auth/logout` without affecting others.
  API keys can be *disabled* (temporarily) or *revoked* (permanently) by an admin.
- **Two admin tiers, no conflicts** — a *platform admin* (superadmin) runs the whole app
  (creates tenants, sees all logs); *enterprise admins* manage **only their own tenant**
  (users + API keys). Employees are tenant-scoped. No one can touch another enterprise.
- **Document upload** — upload documents through the UI or API; the tenant index rebuilds
  asynchronously (a background job, pollable via `/ingest/status`). Uploads are capped
  (`RAG_MAX_UPLOAD_MB`, default 50 MiB) and oversized files return 413 with no partial
  file left behind.
- **Ingestion pipeline** — loads `.pdf` / `.txt` / `.md` / `.docx`, chunks them with
  overlap, embeds them, and persists a FAISS index per tenant.
- **Grounded generation + guardrail** — answers use *only* retrieved context and say
  "I don't know" when the documents don't contain the answer. A lexical grounding check
  validates each answer against its context and refuses to surface fabrications.
- **Source citations** — every answer returns the source file, page, similarity, and snippet.
- **Audit log** — append-only record of admin/auth actions (setup, logins, password
  changes, user/API-key/tenant management, uploads), tenant-scoped for enterprise admins
  and fully visible to the platform admin (`GET /admin/audit-logs`).
- **Observability** — JSON structured logs with `X-Request-ID` correlation across
  UI/API requests, Prometheus metrics at `/metrics`, and a `/health/ready` readiness probe
  used by the Docker healthcheck.
- **Security hardening** — configurable CORS, security headers (HSTS, nosniff, DENY frame),
  non-root Docker container with read-only filesystem, global per-IP rate limiting, and
  per-caller rate limiting on the expensive `/query` endpoint.
- **Three LLM backends** — in-process transformers (default, any CPU), Ollama (CPU/GPU),
  or vLLM (NVIDIA GPU). Swap via one environment variable.
- **REST API** — FastAPI with interactive Swagger docs at `/docs`.
- **Chat UI** — Streamlit interface with login, upload, and expandable sources.
- **Admin UI** — role-aware browser panel: the platform admin sees all tenants and can
  create them; each enterprise admin sees only their own tenant's users and API keys
  (plus the same via REST API or `scripts/manage.py`).
- **Evaluation** — retrieval `precision@k`, `recall@k`, and `MRR` on a labeled set.
- **Docker** — CPU default + optional Ollama/vLLM profiles; `data/` and `models/` are
  volumes, so updates never touch customer data.

---

## Project layout

```
├── app/
│   ├── config.py            # every tunable setting in one place
│   ├── main.py              # FastAPI backend (multi-tenant, auth, admin)
│   ├── security.py          # PBKDF2 hashing, signed tokens, API keys (stdlib)
│   ├── db.py                # SQLite: tenants, users, keys, query + audit logs (stdlib)
│   ├── auth.py              # FastAPI auth dependency -> tenant context
│   ├── metrics.py           # Prometheus counters/gauges/histograms (stdlib)
│   ├── logging_utils.py     # JSON structured logs + request-id correlation
│   ├── schemas.py           # API request/response models
│   ├── rag/
│   │   ├── embeddings.py    # local embedding model (cached singleton)
│   │   ├── llm.py           # LLM backends: transformers / OpenAI-compatible
│   │   ├── ingestion.py     # load + chunk documents
│   │   ├── vectorstore.py   # per-tenant FAISS build / save / load
│   │   ├── grounding.py     # lexical answer-grounding guardrail
│   │   └── pipeline.py      # tenant-scoped retrieve -> ground -> generate -> cite
│   ├── ui/streamlit_app.py  # chat UI (login + upload + admin)
│   └── eval/evaluate.py     # retrieval precision/recall/MRR metrics
├── scripts/
│   ├── make_demo_data.py    # generates sample enterprise documents
│   ├── ingest.py            # builds a tenant's vector index (CLI)
│   ├── run_eval.py          # runs retrieval evaluation
│   ├── manage.py            # admin CLI (tenants, users, API keys)
│   ├── backup.sh            # snapshot DB + tenant data
│   ├── restore.sh           # restore from a snapshot
│   ├── cron-entrypoint.sh   # backup cron sidecar entrypoint
│   ├── migrate_legacy_data.py  # moves pre-v2 data into the tenant layout
│   ├── update.sh/.ps1       # on-prem update scripts
│   └── download_model.py    # resumable model downloader
├── tests/                   # no-network unit + API tests
├── monitoring/
│   └── grafana-dashboard.json  # ready-to-import Grafana dashboard
├── data/
│   ├── tenants/<id>/docs/   # each tenant's source documents
│   ├── tenants/<id>/vectorstore/  # each tenant's FAISS index
│   └── system.db            # auth + query logs (SQLite, auto-created)
└── models/                  # downloaded LLM weights (regenerate)
```

---

## Setup (local development)

Requires **Python 3.11** (also tested-compatible with 3.12).

```powershell
# 1. Create a virtual environment
python -m venv .venv

# 2. Install CPU-only PyTorch first (small download, no CUDA bloat on Windows)
.\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install everything else
.\.venv\Scripts\pip install -r requirements.txt
```

> On Linux/macOS use `.venv/bin/pip` instead of `.venv\Scripts\pip`.

### Downloading the local LLM (one-time, only for the `transformers` backend)

```powershell
.\.venv\Scripts\python scripts\download_model.py
```

If your connection to Hugging Face is slow, point it at a mirror first:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
.\.venv\Scripts\python scripts\download_model.py
```

The embedding model (~90 MB) downloads automatically on the first `ingest.py` run.

---

## Quickstart

```powershell
# 1. (Optional) Generate sample enterprise documents
.\.venv\Scripts\python scripts\make_demo_data.py

# 2. Build the default tenant's index (downloads the free embedding model once)
.\.venv\Scripts\python scripts\ingest.py

# 3. Set a signing secret (the API refuses to start without one, >= 16 chars):
$env:RAG_SECRET_KEY = "change-me-to-a-long-random-string"
#    ...or for quick local dev only:  $env:RAG_ALLOW_INSECURE_DEV_KEY = "1"

# 4a. Launch the REST API  -> http://127.0.0.1:8000/docs
.\.venv\Scripts\uvicorn app.main:app --reload

# 4b. Launch the chat UI   -> http://localhost:8501
.\.venv\Scripts\streamlit run app\ui\streamlit_app.py
```

**First run (setup wizard).** On the very first boot the system has no accounts, and
the UI shows a one-time setup page: enter your company name and an admin username +
password (at least 8 characters). The first admin account is created there — this is
the moment your enterprise "registers" itself. Afterwards it is sign-in only.

> **Unattended installs:** to skip the wizard (e.g. Docker with an admin password in
> `.env`), set `RAG_ADMIN_PASSWORD` — then a **platform admin** (`admin`, role
> `superadmin`) is auto-created on first boot and you sign in directly; use it to onboard
> your enterprise (a tenant) and its enterprise admin.

### Using your own documents

From the UI sidebar: **Upload a document** (`.pdf`/`.txt`/`.md`/`.docx`) — the tenant
index rebuilds automatically. Or drop files into `data/tenants/default/docs/` and run
`python scripts/ingest.py` (or press **Rebuild index** in the sidebar).

### Admin CLI (no web API needed)

```powershell
.\.venv\Scripts\python scripts\manage.py ensure
.\.venv\Scripts\python scripts\manage.py create-tenant --id acme --name "Acme Corp"
.\.venv\Scripts\python scripts\manage.py create-user --tenant acme --username alice --role user
.\.venv\Scripts\python scripts\manage.py create-api-key --tenant acme --label "prod"
.\.venv\Scripts\python scripts\manage.py revoke-api-key --tenant acme --key <key-hash>
.\.venv\Scripts\python scripts\manage.py list-tenants
```

---

## Roles

| Role | Created by | Scope | Can |
|---|---|---|---|
| **Platform admin** (`superadmin`) | `RAG_ADMIN_PASSWORD` seed or CLI (`manage.py ensure --superadmin`) | whole app | create/disable tenants, manage any tenant's users/keys, see all logs, create other platform admins |
| **Enterprise admin** (`admin`) | setup wizard (first account) or any admin | its own tenant only | add/disable its employees, create its API keys, manage its documents, see its own query log |
| **Employee** (`user`) | any admin | its own tenant only | ask questions, upload/manage its tenant's documents |
| **API integration** (`via=api-key`) | any admin (per tenant) | its own tenant only | query + upload for that tenant |

Enterprise admins can never see, modify, or disable **another** enterprise. Only the
platform admin (or the CLI operator) touches tenants across the app.

---

## API reference

Authenticate either with a login token (`Authorization: Bearer <token>`, from
`/auth/login`) or a tenant API key (`X-API-Key: <key>`). The tenant is always resolved
from the credentials, never from the request body.

| Method   | Path                                     | Auth  | Description                                   |
|----------|------------------------------------------|-------|-----------------------------------------------|
| GET      | `/health`                                | none  | Liveness check                                |
| GET      | `/health/ready`                          | none  | Readiness probe (`database`, `indexing_idle`) |
| GET      | `/metrics`                               | none  | Prometheus metrics (text exposition format)   |
| GET      | `/`                                      | none  | Service info                                  |
| GET      | `/auth/setup`                            | none  | `{"needed": bool}` first-run setup status     |
| POST     | `/auth/setup`                            | none  | One-time first-run wizard (`tenant_name`,`username`,`password`) -> token; 409 if already set up |
| POST     | `/auth/login`                            | none  | `{"username","password"}` -> token            |
| POST     | `/auth/change-password`                  | user  | `{"old_password","new_password"}` -> new token; revokes all other sessions |
| POST     | `/auth/logout`                           | user  | Revoke current session's token (other sessions unaffected) |
| GET      | `/auth/me`                               | user  | Current user + tenant                         |
| GET      | `/stats`                                 | user  | Tenant docs/chunks indexed                    |
| POST     | `/query`                                 | user  | `{"question","top_k"}` -> answer + sources (+ `grounded`) |
| GET      | `/documents`                             | user  | Files stored for this tenant                  |
| POST     | `/documents`                             | user  | Multipart upload -> **202** `JobStatus`, re-index runs in background (413 over the size cap) |
| DELETE   | `/documents/{filename}`                  | user  | Remove a file -> **202** `JobStatus`, re-index runs in background |
| POST     | `/ingest`                                | user  | Rebuild this tenant's index -> **202** `JobStatus` |
| GET      | `/ingest/status`                         | user  | Current index job: `idle\|queued\|running\|done\|failed` |
| POST     | `/admin/tenants`                         | superadmin | Create a tenant                               |
| GET      | `/admin/tenants`                         | superadmin | List tenants (with counts)                    |
| PATCH    | `/admin/tenants/{id}`                    | superadmin | Enable/disable a tenant                       |
| POST     | `/admin/tenants/{id}/users`              | tenant admin (own tenant) | Create a user (`password` optional -> auto-generated, returned once) |
| GET      | `/admin/tenants/{id}/users`              | tenant admin (own tenant) | List users                    |
| PATCH    | `/admin/tenants/{id}/users/{username}`   | tenant admin (own tenant) | Enable/disable a user          |
| POST     | `/admin/tenants/{id}/api-keys`           | tenant admin (own tenant) | Create an API key (shown once) |
| GET      | `/admin/tenants/{id}/api-keys`           | tenant admin (own tenant) | List API keys (hashes only)    |
| PATCH    | `/admin/tenants/{id}/api-keys/{hash}`    | tenant admin (own tenant) | Enable/disable an API key       |
| DELETE   | `/admin/tenants/{id}/api-keys/{hash}`    | tenant admin (own tenant) | Permanently revoke an API key   |
| GET      | `/admin/logs?tenant_id=&limit=`          | tenant admin (own tenant) / superadmin (any) | Recent query log |
| GET      | `/admin/audit-logs?tenant_id=&limit=`    | tenant admin (own tenant) / superadmin (any) | Append-only audit trail |

Example:

```powershell
$body = '{"username":"admin","password":"<your-password>"}'
$token = (Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/auth/login -ContentType "application/json" -Body $body).token
$q = '{"question": "What is the 401(k) matching policy?"}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/query -Headers @{ Authorization = "Bearer $token" } -ContentType "application/json" -Body $q
```

---

## LLM backends

Set `RAG_LLM_BACKEND` (default `transformers`):

| Backend     | Where                      | `RAG_LLM_BASE_URL`        | `RAG_LLM_MODEL` example                       | Hardware   |
|-------------|----------------------------|----------------------------|-----------------------------------------------|------------|
| transformers| in-process (default)       | —                          | `google/flan-t5-base` (in `models/`)          | any CPU    |
| openai      | Ollama                     | `http://localhost:11434`   | `qwen2.5:7b-instruct-q4_K_M`                  | CPU / GPU  |
| openai      | vLLM                       | `http://localhost:8000`    | `Qwen/Qwen2.5-7B-Instruct-AWQ`                | NVIDIA GPU |

Everything is free & open source: T5, Qwen2.5, MiniLM (Apache-2.0), Ollama & vLLM (MIT).

---

## Docker deployment

```powershell
# 1. Configure (required: set RAG_SECRET_KEY; optional: RAG_ADMIN_PASSWORD)
Copy-Item .env.example .env

# 2. Build and start the API + UI
docker compose up -d --build
```

> The `api` container **refuses to start without a strong `RAG_SECRET_KEY`**
> (missing `.env` = loud failure, not an insecure silent boot).

- API: http://localhost:8000/docs
- UI: http://localhost:8501 — the first visitor runs the one-time setup wizard
  (company name + admin account). To skip it, set `RAG_ADMIN_PASSWORD` in `.env`
  and the `admin` user is auto-created on first boot.

**Ollama (CPU/GPU):**

```powershell
docker compose -f docker-compose.yml -f docker-compose.ollama.yml up -d
docker compose exec ollama ollama pull qwen2.5:7b-instruct-q4_K_M
docker compose up -d
```

**vLLM (NVIDIA GPU):**

```powershell
docker compose -f docker-compose.yml -f docker-compose.vllm.yml up -d
```

**Updates** (data-safe; `data/` and `models/` are volumes):

```powershell
.\scripts\update.ps1
```

See [ENTERPRISE.md](ENTERPRISE.md) for the full on-prem install/upgrade/monitoring guide.

---

## How it works

1. **Tenancy** — the auth dependency resolves every request to a `tenant_id`; documents
   and indexes live under `data/tenants/<tenant_id>/`, so isolation is physical.
2. **Chunking** — documents are split into ~1000-char chunks with 150-char overlap.
3. **Embedding** — `all-MiniLM-L6-v2` maps each chunk to a 384-dim normalized vector.
4. **Indexing** — FAISS stores vectors + metadata on disk, per tenant.
5. **Retrieval** — the question is embedded with the same model; top-k nearest chunks are
   returned with similarity scores.
6. **Generation** — chunks go into a strict grounding prompt; the LLM (transformers,
   Ollama, or vLLM) answers from them only, with citations returned alongside.

### Tuning

All knobs live in `app/config.py`. Chunking/retrieval internals (`CHUNK_SIZE`,
`CHUNK_OVERLAP`, `TOP_K`, `EMBEDDING_MODEL`, `LLM_MODEL`, `MAX_NEW_TOKENS`) are set in
code; deployment tunables are env vars — see `.env.example` for upload caps
(`RAG_MAX_UPLOAD_MB`), PBKDF2 cost (`RAG_PBKDF2_ITERATIONS`), the grounding guardrail
(`RAG_GROUNDING_*`), audit-log retention (`RAG_AUDIT_LOG_RETENTION_DAYS`), and query
rate limits. After changing `LLM_MODEL` for the `transformers` backend, re-run
`scripts/download_model.py`.

---

## Evaluation

`python scripts/run_eval.py` scores retrieval on a labeled set of 9 questions:

- **Mean Precision@k** — how many of the top-k chunks were relevant.
- **Mean Recall@k** — what fraction of the relevant document was retrieved.
- **Mean MRR** — how high the first correct source ranked.

On the demo corpus expect **Recall@4 = 1.0 and MRR = 1.0**. Precision@4 is ~0.25 by
construction here (one relevant document per question at k=4) — with perfect recall/MRR
this just means the retriever adds filler context, which is normal in RAG.

### Known limitations

- The default `transformers` backend (`flan-t5-base`, 250M params) stays fast on CPU but
  can slip when several numeric facts share one paragraph. For higher accuracy use the
  Ollama/vLLM backend with a 7B Qwen2.5 model, or `google/flan-t5-large`.
- Answer-quality (faithfulness) evaluation is a natural next step beyond retrieval metrics.

---

## Testing

```powershell
.\.venv\Scripts\python -m pytest tests -q
```

Tests need no network and no model downloads (the API tests inject a fake pipeline and
use a throwaway SQLite file). The suite covers auth, tenant/admin scoping, token
revocation on password change, API-key disable/revoke, the audit log, grounding checks,
upload caps, background ingest jobs, and metrics/readiness.

---

## Dependencies (all free & open source)

| Purpose        | Libraries |
|----------------|-----------|
| ML / NLP       | torch (CPU), transformers, sentence-transformers |
| Orchestration  | langchain, langchain-community, langchain-huggingface, langchain-text-splitters |
| Vector store   | faiss-cpu |
| Document I/O   | pypdf, docx2txt, reportlab (demo PDFs) |
| Backend / UI   | fastapi, uvicorn, streamlit, pandas |
| Auth / storage | Python standard library only (hashlib, hmac, secrets, sqlite3) |
| Testing        | pytest, httpx |
