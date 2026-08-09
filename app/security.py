"""Auth primitives: password hashing, signed tokens, API keys.

Standard-library only (hashlib / hmac / secrets) so the project keeps zero
paid or heavy dependencies and remains fully auditable.

Token format (JWT-like, hand-rolled for transparency):
    base64url(payload).base64url(hmac_sha256(payload, secret))
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time

from app.config import settings

PBKDF2_ITERATIONS = 120_000


# ---------------------------------------------------------------------------
# Passwords (PBKDF2-HMAC-SHA256, salted)
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt_hex, hash_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(digest.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Signed tokens
# ---------------------------------------------------------------------------

def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def sign_token(payload: dict, ttl_seconds: int | None = None) -> str:
    body = dict(payload)
    body["exp"] = int(time.time()) + (ttl_seconds or settings.TOKEN_TTL_SECONDS)
    encoded = _b64encode(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(
        settings.SECRET_KEY.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{encoded}.{_b64encode(signature)}"


def verify_token(token: str) -> dict:
    """Return the token payload, raising ValueError if invalid or expired."""
    try:
        encoded, signature = token.split(".", 1)
    except ValueError:
        raise ValueError("Malformed token")
    expected = hmac.new(
        settings.SECRET_KEY.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
    ).digest()
    if not hmac.compare_digest(_b64decode(signature), expected):
        raise ValueError("Bad signature")
    body = json.loads(_b64decode(encoded))
    if int(body.get("exp", 0)) < int(time.time()):
        raise ValueError("Token expired")
    return body


# ---------------------------------------------------------------------------
# API keys (random 32-byte secrets; only SHA-256 hashes are stored)
# ---------------------------------------------------------------------------

def new_api_key() -> tuple[str, str]:
    """Return (plain_key, sha256_hex). The plain key is shown only once."""
    key = secrets.token_urlsafe(32)
    return key, hash_api_key(key)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
