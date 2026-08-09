"""FastAPI auth dependencies.

A request is authenticated either with a login token
(``Authorization: Bearer <token>``) or a tenant API key (``X-API-Key``).
The resolved :class:`AuthContext` carries the tenant_id, so every data access
in the app is scoped to the caller's tenant - never taken from client input.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app import db
from app.security import hash_api_key, verify_token

_bearer = HTTPBearer(auto_error=False)


@dataclass
class AuthContext:
    tenant_id: str
    username: str
    role: str
    subject: str            # stable unique principal id (for rate limiting etc.)
    user_id: int | None = None
    via: str = "token"


def get_auth_context(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    x_api_key: str | None = Header(default=None),
) -> AuthContext:
    if x_api_key:
        record = db.get_api_key_by_hash(hash_api_key(x_api_key))
        if record is None or not record["is_active"]:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")
        tenant = db.get_tenant(record["tenant_id"])
        if tenant is None or not tenant["is_active"]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Tenant disabled")
        return AuthContext(
            tenant_id=record["tenant_id"],
            username="api-key",
            role="user",
            subject=f"apikey:{record['key_hash']}",
            via="api-key",
        )

    if creds is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Missing credentials: use 'Authorization: Bearer <token>' or 'X-API-Key'",
        )

    try:
        payload = verify_token(creds.credentials)
    except ValueError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")

    user = db.get_user_by_id(payload.get("uid"))
    if user is None or not user["is_active"]:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Account disabled")
    tenant = db.get_tenant(user["tenant_id"])
    if tenant is None or not tenant["is_active"]:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Tenant disabled")
    return AuthContext(
        tenant_id=user["tenant_id"],
        username=user["username"],
        role=user["role"],
        subject=f"user:{user['id']}",
        user_id=user["id"],
        via="token",
    )


def require_admin(auth: AuthContext = Depends(get_auth_context)) -> AuthContext:
    """A platform admin (superadmin) or an enterprise admin."""
    if auth.role not in ("admin", "superadmin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin privileges required")
    return auth


def require_superadmin(auth: AuthContext = Depends(get_auth_context)) -> AuthContext:
    """A platform admin - the only actor with whole-app powers."""
    if auth.role != "superadmin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Platform admin privileges required")
    return auth


def ensure_tenant_access(auth: AuthContext, tenant_id: str) -> None:
    """Allow if the caller is a platform admin or an admin of this tenant.

    An enterprise admin managing a *different* tenant gets 403 - tenants can
    never administer each other.
    """
    if auth.role == "superadmin":
        return
    if auth.tenant_id != tenant_id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "Not your tenant"
        )
