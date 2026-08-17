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

    # ---- Database -------------------------------------------------------
    # Set RAG_DATABASE_URL to use PostgreSQL in production.
    # Example: postgresql://user:password@localhost:5432/rag
    # When empty (default), SQLite is used (zero setup, perfect for dev/tests).
    DATABASE_URL: str = os.environ.get("RAG_DATABASE_URL", "")

    # ---- Tenancy ----------------------------------------------------------
    DEFAULT_TENANT: str = os.environ.get("RAG_DEFAULT_TENANT_ID", "default")

    def tenant_docs_dir(self, tenant_id: str) -> Path:
        return self.TENANTS_DIR / tenant_id / "docs"

    def tenant_vectorstore_dir(self, tenant_id: str) -> Path:
        return self.TENANTS_DIR / tenant_id / "vectorstore"

    def ensure_tenant_dirs(self, tenant_id: str) -> None:
        self.tenant_docs_dir(tenant_id).mkdir(parents=True, exist_ok=True)
        self.tenant_vectorstore_dir(tenant_id).mkdir(parents=True, exist_ok=True)

    @property
    def DOCS_DIR(self) -> Path:
        return self.tenant_docs_dir(self.DEFAULT_TENANT)

    @property
    def VECTORSTORE_DIR(self) -> Path:
        return self.tenant_vectorstore_dir(self.DEFAULT_TENANT)

    # ---- Models (free, open source, local) ------------------------------
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = os.environ.get("RAG_LLM_MODEL", "google/flan-t5-base")

    # ---- LLM backend ------------------------------------------------------
    LLM_BACKEND: str = os.environ.get("RAG_LLM_BACKEND", "transformers")
    LLM_BASE_URL: str = os.environ.get("RAG_LLM_BASE_URL", "http://127.0.0.1:11434")
    LLM_API_KEY: str = os.environ.get("RAG_LLM_API_KEY", "")

    # ---- Auth -------------------------------------------------------------
    SECRET_KEY: str = os.environ.get("RAG_SECRET_KEY", "").strip()
    ALLOW_INSECURE_DEV_KEY: bool = os.environ.get("RAG_ALLOW_INSECURE_DEV_KEY", "") == "1"
    TOKEN_TTL_SECONDS: int = int(os.environ.get("RAG_TOKEN_TTL_SECONDS", str(12 * 3600)))
    ADMIN_USERNAME: str = os.environ.get("RAG_ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: str = os.environ.get("RAG_ADMIN_PASSWORD", "")

    # ---- Password hashing -------------------------------------------------
    PBKDF2_ITERATIONS: int = int(os.environ.get("RAG_PBKDF2_ITERATIONS", "600000"))

    # ---- Upload limits ----------------------------------------------------
    MAX_UPLOAD_MB: int = int(os.environ.get("RAG_MAX_UPLOAD_MB", "50"))

    # ---- Answer grounding (hallucination guardrail) -----------------------
    GROUNDING_CHECK: bool = os.environ.get("RAG_GROUNDING_CHECK", "1") == "1"
    GROUNDING_MIN_OVERLAP: float = float(os.environ.get("RAG_GROUNDING_MIN_OVERLAP", "0.5"))
    GROUNDING_MIN_TOKENS: int = int(os.environ.get("RAG_GROUNDING_MIN_TOKENS", "4"))

    # ---- Audit log --------------------------------------------------------
    AUDIT_LOG_RETENTION_DAYS: int = int(os.environ.get("RAG_AUDIT_LOG_RETENTION_DAYS", "365"))

    # ---- UI ----------------------------------------------------------------
    API_BASE_URL: str = os.environ.get("RAG_API_BASE_URL", "http://127.0.0.1:8000")

    # ---- CORS -----------------------------------------------------------
    CORS_ORIGINS: str = os.environ.get("RAG_CORS_ORIGINS", "")

    # ---- Brute-force protection ---------------------------------------
    LOGIN_MAX_FAILURES: int = int(os.environ.get("RAG_LOGIN_MAX_FAILURES", "5"))
    LOGIN_MAX_FAILURES_PER_IP: int = int(os.environ.get("RAG_LOGIN_MAX_FAILURES_PER_IP", "20"))
    LOGIN_FAILURE_WINDOW_SECONDS: int = int(os.environ.get("RAG_LOGIN_FAILURE_WINDOW_SECONDS", "900"))

    # ---- Query rate limiting --------------------------------------------
    QUERY_RATE_LIMIT_MAX: int = int(os.environ.get("RAG_QUERY_RATE_LIMIT_MAX", "30"))
    QUERY_RATE_LIMIT_WINDOW_SECONDS: int = int(os.environ.get("RAG_QUERY_RATE_LIMIT_WINDOW_SECONDS", "60"))

    # ---- Global rate limiting -------------------------------------------
    GLOBAL_RATE_LIMIT_MAX: int = int(os.environ.get("RAG_GLOBAL_RATE_LIMIT_MAX", "120"))
    GLOBAL_RATE_LIMIT_WINDOW_SECONDS: int = int(os.environ.get("RAG_GLOBAL_RATE_LIMIT_WINDOW_SECONDS", "60"))

    # ---- Document chunking strategy -------------------------------------
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # ---- Per-tenant pipeline cache -------------------------------------
    PIPELINE_CACHE_SIZE: int = int(os.environ.get("RAG_PIPELINE_CACHE_SIZE", "8"))

    # ---- Retrieval ------------------------------------------------------
    TOP_K: int = 4

    # ---- Generation -----------------------------------------------------
    MAX_NEW_TOKENS: int = 256

    # ---- Device ---------------------------------------------------------
    @staticmethod
    def _detect_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    DEVICE: str = _detect_device()

    # ---- Supported source formats ---------------------------------------
    SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt", ".md", ".docx")

    # ---- SSO / OIDC ----------------------------------------------------
    SSO_ENABLED: bool = os.environ.get("RAG_SSO_ENABLED", "") == "1"
    SSO_PROVIDER: str = os.environ.get("RAG_SSO_PROVIDER", "oidc")  # oidc | ldap
    SSO_DEFAULT_TENANT: str = os.environ.get("RAG_SSO_DEFAULT_TENANT", "default")

    # OIDC settings
    OIDC_ISSUER_URL: str = os.environ.get("RAG_OIDC_ISSUER_URL", "")
    OIDC_CLIENT_ID: str = os.environ.get("RAG_OIDC_CLIENT_ID", "")
    OIDC_CLIENT_SECRET: str = os.environ.get("RAG_OIDC_CLIENT_SECRET", "")
    OIDC_REDIRECT_URI: str = os.environ.get("RAG_OIDC_REDIRECT_URI", "http://localhost:8000/auth/sso/callback")

    # LDAP settings
    LDAP_SERVER_URL: str = os.environ.get("RAG_LDAP_SERVER_URL", "ldap://localhost:389")
    LDAP_BIND_DN: str = os.environ.get("RAG_LDAP_BIND_DN", "")
    LDAP_BIND_PASSWORD: str = os.environ.get("RAG_LDAP_BIND_PASSWORD", "")
    LDAP_USER_SEARCH_BASE: str = os.environ.get("RAG_LDAP_USER_SEARCH_BASE", "")
    LDAP_USER_SEARCH_FILTER: str = os.environ.get("RAG_LDAP_USER_SEARCH_FILTER", "(uid={username})")

    # ---- At-rest encryption ---------------------------------------------
    # Set RAG_ENCRYPTION_KEY to enable Fernet encryption for uploaded documents.
    # When empty (default), documents are stored plaintext.
    ENCRYPTION_KEY: str = os.environ.get("RAG_ENCRYPTION_KEY", "")


settings = Settings()
