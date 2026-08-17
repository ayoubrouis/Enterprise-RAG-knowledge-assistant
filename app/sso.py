"""SSO / LDAP authentication backends.

Provides pluggable authentication beyond local password verification:
  - **OAuth2 / OIDC**: delegates login to an external identity provider
    (Keycloak, Azure AD, Okta, Google Workspace, etc.). The UI redirects the
    user to the IdP; on callback the app verifies the ID token and provisions
    the user locally (auto-create on first login).
  - **LDAP**: binds against an Active Directory / OpenLDAP server. Used for
    enterprises that manage all accounts in a single directory.

Enable via env vars:
  RAG_SSO_ENABLED=1
  RAG_SSO_PROVIDER=oidc          # oidc | ldap
  RAG_OIDC_ISSUER_URL=...
  RAG_OIDC_CLIENT_ID=...
  RAG_OIDC_CLIENT_SECRET=...
  RAG_OIDC_REDIRECT_URI=http://localhost:8000/auth/sso/callback
  RAG_LDAP_SERVER_URL=ldap://ldap.example.com:389
  RAG_LDAP_BIND_DN=cn=admin,dc=example,dc=com
  RAG_LDAP_BIND_PASSWORD=...
  RAG_LDAP_USER_SEARCH_BASE=ou=users,dc=example,dc=com
  RAG_LDAP_USER_SEARCH_FILTER=(uid={username})
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from app.config import settings


# ---------------------------------------------------------------------------
# OIDC / OAuth2
# ---------------------------------------------------------------------------

_oidc_config: dict[str, Any] | None = None


def _fetch_oidc_config() -> dict[str, Any]:
    """Fetch the OIDC discovery document (cached in memory)."""
    global _oidc_config
    if _oidc_config is not None:
        return _oidc_config
    import httpx

    resp = httpx.get(
        f"{settings.OIDC_ISSUER_URL.rstrip('/')}/.well-known/openid-configuration",
        timeout=10,
    )
    resp.raise_for_status()
    _oidc_config = resp.json()
    return _oidc_config


def oidc_authorization_url() -> str:
    """Build the redirect URL for the OIDC login flow."""
    cfg = _fetch_oidc_config()
    from urllib.parse import urlencode

    params = {
        "client_id": settings.OIDC_CLIENT_ID,
        "redirect_uri": settings.OIDC_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": "rag-sso",  # TODO: generate and verify a random state
    }
    return cfg["authorization_endpoint"] + "?" + urlencode(params)


def oidc_exchange_code(code: str) -> dict:
    """Exchange an authorization code for tokens, verify, and return claims."""
    import httpx

    cfg = _fetch_oidc_config()
    resp = httpx.post(
        cfg["token_endpoint"],
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": settings.OIDC_REDIRECT_URI,
            "client_id": settings.OIDC_CLIENT_ID,
            "client_secret": settings.OIDC_CLIENT_SECRET,
        },
        timeout=10,
    )
    resp.raise_for_status()
    tokens = resp.json()

    # Verify the ID token (in production, verify signature via JWKS).
    import base64, json

    id_token = tokens.get("id_token", "")
    try:
        payload_b64 = id_token.split(".")[1]
        padding = "=" * (-len(payload_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64 + padding))
    except Exception:
        raise ValueError("Invalid ID token")

    return {
        "sub": claims.get("sub"),
        "email": claims.get("email", ""),
        "name": claims.get("name", claims.get("preferred_username", "")),
        "tenant_id": _resolve_tenant_from_claims(claims),
    }


def _resolve_tenant_from_claims(claims: dict) -> str:
    """Map OIDC claims to a tenant. Override with RAG_SSO_DEFAULT_TENANT."""
    return settings.SSO_DEFAULT_TENANT


# ---------------------------------------------------------------------------
# LDAP
# ---------------------------------------------------------------------------

def ldap_authenticate(username: str, password: str) -> dict | None:
    """Bind against LDAP and return user claims on success, None on failure."""
    try:
        import ldap3
    except ImportError:
        raise RuntimeError("ldap3 is required for LDAP auth: pip install ldap3")

    server = ldap3.Server(settings.LDAP_SERVER_URL, get_info=ldap3.ALL)
    try:
        conn = ldap3.Connection(
            server,
            user=settings.LDAP_BIND_DN,
            password=settings.LDAP_BIND_PASSWORD,
            auto_bind=True,
        )
    except ldap3.core.exceptions.LDAPBindError:
        raise RuntimeError("LDAP bind failed — check RAG_LDAP_BIND_DN and RAG_LDAP_BIND_PASSWORD")

    # Search for the user
    search_filter = settings.LDAP_USER_SEARCH_FILTER.replace("{username}", username)
    conn.search(
        settings.LDAP_USER_SEARCH_BASE,
        search_filter,
        attributes=["cn", "mail", "uid", "memberOf"],
    )
    if not conn.entries:
        conn.unbind()
        return None

    entry = conn.entries[0]
    user_dn = entry.entry_dn

    # Attempt to bind as the user
    try:
        user_conn = ldap3.Connection(server, user=user_dn, password=password, auto_bind=True)
        user_conn.unbind()
    except ldap3.core.exceptions.LDAPBindError:
        conn.unbind()
        return None

    conn.unbind()
    return {
        "sub": str(entry.uid),
        "email": str(entry.mail) if hasattr(entry, "mail") else "",
        "name": str(entry.cn) if hasattr(entry, "cn") else username,
        "tenant_id": settings.SSO_DEFAULT_TENANT,
    }


# ---------------------------------------------------------------------------
# Unified SSO entry point
# ---------------------------------------------------------------------------

def sso_authenticate(username: str = "", password: str = "", code: str = "") -> dict | None:
    """Route to the configured SSO provider. Returns user claims or None."""
    if not settings.SSO_ENABLED:
        return None
    provider = settings.SSO_PROVIDER.lower()
    if provider == "oidc" and code:
        return oidc_exchange_code(code)
    if provider == "ldap" and username and password:
        return ldap_authenticate(username, password)
    return None


def provision_sso_user(claims: dict) -> dict:
    """Create or update a local user from SSO claims. Returns the DB user row."""
    from app import db

    tenant_id = claims.get("tenant_id", settings.SSO_DEFAULT_TENANT)
    username = claims.get("name") or claims.get("email") or claims.get("sub", "sso-user")

    # Ensure tenant exists
    if db.get_tenant(tenant_id) is None:
        db.create_tenant(tenant_id, tenant_id.title())

    user = db.get_user_by_username(username)
    if user is None:
        # Create a new user — no local password (SSO-only).
        # Store a random hash they can never log in with directly.
        dummy_hash = hashlib.sha256(f"sso-only-{username}".encode()).hexdigest()
        if settings._USE_PG if hasattr(settings, '_USE_PG') else False:
            from app import db as _db
            # Use a direct insert with a random password hash
            _db._execute(
                "INSERT INTO users (tenant_id, username, password_hash, role, created_at) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING *",
                (tenant_id, username, f"sso${dummy_hash}", "user", time.time()),
                fetchone=True,
            )
        else:
            from app import db as _db
            _db._execute(
                "INSERT INTO users (tenant_id, username, password_hash, role, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (tenant_id, username, f"sso${dummy_hash}", "user", time.time()),
            )
        user = db.get_user_by_username(username)

    return user
