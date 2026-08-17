"""Central configuration for the Enterprise RAG Knowledge Assistant.

Every tunable lives here so the rest of the code stays free of magic numbers.
All models are free & open-source Hugging Face models that run 100% locally
(after a one-time download, they are cached on disk and used offline).

The system is fully multi-tenant: every tenant gets its own docs folder and
its own FAISS index under ``data/tenants/<tenant_id>/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


class Settings:
    """Runtime settings. Override any value via environment variables
    (e.g. ``RAG_EMBEDDING_MODEL=...``) for deployment flexibility."""

    def __init__(self) -> None:
        # Fail closed: refuse to boot unless a strong signing secret is set.
        # The dev escape hatch below is opt-in and only for local development.
        known_insecure = {"dev-secret-change-me", "change-me-to-a-long-random-string"}
        if not self.SECRET_KEY and self.ALLOW_INSECURE_DEV_KEY:
            self.SECRET_KEY = "dev-secret-change-me"
        if len(self.SECRET_KEY) < 16:
            raise RuntimeError(
                "RAG_SECRET_KEY is required and must be at least 16 characters. "
                "Generate one with:\n"
                '  python -c "import secrets; print(secrets.token_urlsafe(48))"\n'
                "For local development ONLY you may instead set "
                "RAG_ALLOW_INSECURE_DEV_KEY=1 (never in production)."
            )
        if self.SECRET_KEY in known_insecure and not self.ALLOW_INSECURE_DEV_KEY:
            raise RuntimeError(
                "RAG_SECRET_KEY is set to a known example value, so it is not "
                "secret. Generate a unique one with:\n"
                '  python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )

        # Paths are resolved at construction time so RAG_DATA_DIR and
        # RAG_DB_PATH are honored (each defaults to the project data folder).
        self.DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", str(self.ROOT_DIR / "data")))
        self.TENANTS_DIR = self.DATA_DIR / "tenants"
        self.DB_PATH = Path(
            os.environ.get("RAG_DB_PATH", str(self.DATA_DIR / "system.db"))
        )

    # ---- Paths (relative to the project root) ---------------------------
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent

    # ---- Tenancy ----------------------------------------------------------
    # The tenant used for local development and by scripts that predate the
    # multi-tenant layout. API users always resolve their tenant from auth.
    DEFAULT_TENANT: str = os.environ.get("RAG_DEFAULT_TENANT_ID", "default")

    def tenant_docs_dir(self, tenant_id: str) -> Path:
        return self.TENANTS_DIR / tenant_id / "docs"

    def tenant_vectorstore_dir(self, tenant_id: str) -> Path:
        return self.TENANTS_DIR / tenant_id / "vectorstore"

    def ensure_tenant_dirs(self, tenant_id: str) -> None:
        self.tenant_docs_dir(tenant_id).mkdir(parents=True, exist_ok=True)
        self.tenant_vectorstore_dir(tenant_id).mkdir(parents=True, exist_ok=True)

    # Backwards-compatible helpers pointing at the default tenant.
    @property
    def DOCS_DIR(self) -> Path:
        return self.tenant_docs_dir(self.DEFAULT_TENANT)

    @property
    def VECTORSTORE_DIR(self) -> Path:
        return self.tenant_vectorstore_dir(self.DEFAULT_TENANT)

    # ---- Models (free, open source, local) ------------------------------
    # Small, fast embedding model - excellent quality/size trade-off.
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Instruction-tuned encoder-decoder LLM that extracts grounded answers
    # well and runs acceptably on CPU. Swap to "google/flan-t5-large" for a
    # higher-quality but slower model.
    LLM_MODEL: str = os.environ.get("RAG_LLM_MODEL", "google/flan-t5-base")

    # ---- LLM backend ------------------------------------------------------
    # "transformers"  -> in-process Hugging Face model (default; works on any
    #                    CPU, zero extra services). Model: LLM_MODEL above.
    # "openai"        -> any OpenAI-compatible inference server, e.g.:
    #                      * Ollama (CPU or GPU) - set RAG_LLM_BASE_URL to
    #                        http://host:11434 and RAG_LLM_MODEL to an Ollama
    #                        tag such as "qwen2.5:7b-instruct-q4_K_M"
    #                      * vLLM (NVIDIA GPU) - set RAG_LLM_BASE_URL to
    #                        http://host:8000 and RAG_LLM_MODEL to the served
    #                        model name (e.g. "Qwen/Qwen2.5-7B-Instruct-AWQ")
    LLM_BACKEND: str = os.environ.get("RAG_LLM_BACKEND", "transformers")
    LLM_BASE_URL: str = os.environ.get("RAG_LLM_BASE_URL", "http://127.0.0.1:11434")
    LLM_API_KEY: str = os.environ.get("RAG_LLM_API_KEY", "")

    # ---- Auth -------------------------------------------------------------
    # Signing secret for login tokens. REQUIRED: the app refuses to start
    # without one (>= 16 characters) so a deployment can never silently run
    # with a known default key. Generate one with:
    #   python -c "import secrets; print(secrets.token_urlsafe(48))"
    # For quick local development ONLY you may set RAG_ALLOW_INSECURE_DEV_KEY=1
    # to fall back to a fixed dev key - never do this in production.
    SECRET_KEY: str = os.environ.get("RAG_SECRET_KEY", "").strip()
    ALLOW_INSECURE_DEV_KEY: bool = os.environ.get("RAG_ALLOW_INSECURE_DEV_KEY", "") == "1"
    TOKEN_TTL_SECONDS: int = 12 * 3600
    ADMIN_USERNAME: str = os.environ.get("RAG_ADMIN_USERNAME", "admin")
    # Leave empty for the first-run setup wizard (interactive installs): the UI
    # shows a one-time "create your enterprise + admin" page, and afterwards it
    # is sign-in only. Set it for unattended deploys (e.g. Docker CI) so the
    # admin is auto-seeded at first boot with this password.
    ADMIN_PASSWORD: str = os.environ.get("RAG_ADMIN_PASSWORD", "")

    # ---- Password hashing -------------------------------------------------
    # OWASP-recommended PBKDF2-HMAC-SHA256 cost factor. Existing hashes store
    # their own iteration count, so a running system rehashes on the next
    # successful login whenever this value changes (see app/security.py).
    PBKDF2_ITERATIONS: int = int(os.environ.get("RAG_PBKDF2_ITERATIONS", "600000"))

    # ---- Upload limits ----------------------------------------------------
    # Hard cap on a single uploaded document (MiB). Prevents a runaway upload
    # from filling the disk; rejects with 413 when exceeded.
    MAX_UPLOAD_MB: int = int(os.environ.get("RAG_MAX_UPLOAD_MB", "50"))

    # ---- Answer grounding (hallucination guardrail) -----------------------
    # After the LLM generates an answer, the pipeline checks that a meaningful
    # fraction of the answer's content words actually appear in the retrieved
    # context. When the answer is unsupported (or the check cannot be judged),
    # the pipeline replies "I don't know." instead of surfacing a fabrication.
    GROUNDING_CHECK: bool = os.environ.get("RAG_GROUNDING_CHECK", "1") == "1"
    GROUNDING_MIN_OVERLAP: float = float(
        os.environ.get("RAG_GROUNDING_MIN_OVERLAP", "0.5")
    )
    GROUNDING_MIN_TOKENS: int = int(os.environ.get("RAG_GROUNDING_MIN_TOKENS", "4"))

    # ---- Audit log --------------------------------------------------------
    # Admin/auth actions are written to an append-only audit table. Old rows
    # are pruned opportunistically (on insert) beyond this retention window.
    AUDIT_LOG_RETENTION_DAYS: int = int(
        os.environ.get("RAG_AUDIT_LOG_RETENTION_DAYS", "365")
    )

    # ---- UI ----------------------------------------------------------------
    # Where the Streamlit UI (a separate process) reaches the FastAPI backend.
    API_BASE_URL: str = os.environ.get("RAG_API_BASE_URL", "http://127.0.0.1:8000")

    # ---- Brute-force protection ---------------------------------------
    # Login failures are recorded per (username, ip) and enforced inside a
    # rolling window. Exceeding the per-username limit locks that account;
    # exceeding the per-IP limit throttles the whole source address.
    LOGIN_MAX_FAILURES: int = int(os.environ.get("RAG_LOGIN_MAX_FAILURES", "5"))
    LOGIN_MAX_FAILURES_PER_IP: int = int(os.environ.get("RAG_LOGIN_MAX_FAILURES_PER_IP", "20"))
    LOGIN_FAILURE_WINDOW_SECONDS: int = int(os.environ.get("RAG_LOGIN_FAILURE_WINDOW_SECONDS", "900"))

    # ---- Query rate limiting --------------------------------------------
    # Sliding-window cap on /query (the expensive LLM + FAISS endpoint),
    # enforced per authenticated caller (user or API key).
    QUERY_RATE_LIMIT_MAX: int = int(os.environ.get("RAG_QUERY_RATE_LIMIT_MAX", "30"))
    QUERY_RATE_LIMIT_WINDOW_SECONDS: int = int(
        os.environ.get("RAG_QUERY_RATE_LIMIT_WINDOW_SECONDS", "60")
    )

    # ---- Document chunking strategy -------------------------------------
    CHUNK_SIZE: int = 1000          # target characters per chunk
    CHUNK_OVERLAP: int = 150        # overlap keeps context across chunk edges

    # ---- Per-tenant pipeline cache -------------------------------------
    # A tenant's RAGPipeline holds its FAISS index in memory. The cache is an
    # LRU: at most this many tenants' indexes stay loaded at once, so a box
    # with many tenants does not accumulate unbounded RAM.
    PIPELINE_CACHE_SIZE: int = int(os.environ.get("RAG_PIPELINE_CACHE_SIZE", "8"))

    # ---- Retrieval ------------------------------------------------------
    TOP_K: int = 4                  # how many chunks to feed the LLM

    # ---- Generation -----------------------------------------------------
    MAX_NEW_TOKENS: int = 256

    # ---- Device ---------------------------------------------------------
    @staticmethod
    def _detect_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    DEVICE: str = _detect_device()

    # ---- Supported source formats ---------------------------------------
    SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt", ".md", ".docx")


settings = Settings()
