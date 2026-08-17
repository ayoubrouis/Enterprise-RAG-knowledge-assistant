"""TOTP-based Multi-Factor Authentication.

Generates TOTP secrets, creates QR codes for authenticator apps (Google
Authenticator, Authy, etc.), and verifies one-time codes.

Requires ``pyotp`` and ``qrcode`` (both free & open source).

Enable per-user via the API or admin UI:
  POST /auth/mfa/enable   -> returns a TOTP secret + QR code data URL
  POST /auth/mfa/verify   -> confirms the code is correct, enables MFA
  POST /auth/mfa/disable  -> requires current TOTP code to confirm

Once enabled, the login flow adds a second step:
  POST /auth/login        -> returns {"mfa_required": true, "mfa_token": "..."}
  POST /auth/mfa/validate -> exchanges mfa_token + TOTP code for a real token
"""

from __future__ import annotations

import base64
import io
from typing import Any


def generate_mfa_secret() -> str:
    """Generate a new random TOTP secret (base32-encoded)."""
    import pyotp
    return pyotp.random_base32()


def get_totp_uri(secret: str, username: str, issuer: str = "Enterprise RAG") -> str:
    """Build the otpauth:// URI for QR code generation."""
    import pyotp
    totp = pyotp.TOTP(secret)
    return totp.provisioning_uri(name=username, issuer_name=issuer)


def verify_totp(secret: str, code: str) -> bool:
    """Verify a TOTP code against the secret. Accepts codes within a
    ±1 window (30s each side) to handle clock skew."""
    import pyotp
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=1)


def generate_qr_data_url(uri: str) -> str:
    """Generate a base64-encoded PNG QR code data URL from an otpauth URI."""
    try:
        import qrcode
    except ImportError:
        raise RuntimeError("qrcode is required for QR generation: pip install qrcode")
    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def enable_mfa(user_id: int, totp_code: str) -> tuple[bool, str]:
    """Enable MFA for a user. Returns (success, error_or_qr_data_url).

    The user must provide a valid TOTP code from their authenticator app
    to prove they've set it up correctly before MFA is activated.
    """
    from app import db

    secret = db.get_user_mfa_secret(user_id)
    if secret is None:
        # First time — generate and store (not yet enabled)
        secret = generate_mfa_secret()
        db.set_user_mfa(user_id, secret, enabled=False)

    if not verify_totp(secret, totp_code):
        return False, "Invalid TOTP code"

    db.set_user_mfa(user_id, secret, enabled=True)
    return True, ""


def disable_mfa(user_id: int, totp_code: str) -> tuple[bool, str]:
    """Disable MFA. Requires a valid TOTP code to confirm."""
    from app import db

    secret = db.get_user_mfa_secret(user_id)
    if secret is None:
        return False, "MFA is not enabled"

    if not verify_totp(secret, totp_code):
        return False, "Invalid TOTP code"

    db.set_user_mfa(user_id, None, enabled=False)
    return True, ""


def is_mfa_enabled(user_id: int) -> bool:
    """Check if MFA is enabled for a user."""
    from app import db

    row = db._execute(
        "SELECT mfa_enabled FROM users WHERE id = %s" if db._USE_PG
        else "SELECT mfa_enabled FROM users WHERE id = ?",
        (user_id,), fetchone=True,
    )
    if row is None:
        return False
    val = row["mfa_enabled"]
    return bool(val) if isinstance(val, bool) else val == 1


def get_mfa_qr(user_id: int) -> str | None:
    """Return a QR code data URL for the user's MFA secret, or None."""
    from app import db

    secret = db.get_user_mfa_secret(user_id)
    if secret is None:
        return None
    user = db.get_user_by_id(user_id)
    if user is None:
        return None
    uri = get_totp_uri(secret, user["username"])
    return generate_qr_data_url(uri)
