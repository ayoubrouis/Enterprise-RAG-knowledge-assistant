"""At-rest encryption for uploaded documents.

Uses Fernet symmetric encryption (AES-128-CBC with HMAC-SHA256) from the
``cryptography`` library. Each document is encrypted with a per-file random
nonce prepended to the ciphertext; the master key is derived from
``RAG_ENCRYPTION_KEY`` via PBKDF2.

When ``RAG_ENCRYPTION_KEY`` is empty (the default), documents are stored
plaintext — zero overhead, zero config. Set the key to enable encryption:

    RAG_ENCRYPTION_KEY=<a long random string>

The key is stretched via PBKDF2-HMAC-SHA256 (100k iterations) to a Fernet-
compatible 32-byte key. The encryption is transparent: reading decrypts, and
the rest of the pipeline never sees plaintext vs ciphertext.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

_FERNET_KEY: str | None = None


def _encryption_key() -> str:
    """Read the encryption key from the environment (avoids stale singleton)."""
    return os.environ.get("RAG_ENCRYPTION_KEY", "")


def _get_fernet():
    """Lazy-init the Fernet cipher from the configured key."""
    global _FERNET_KEY
    enc_key = _encryption_key()
    if not enc_key:
        return None
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        raise RuntimeError(
            "cryptography is required for at-rest encryption: "
            "pip install cryptography"
        )
    if _FERNET_KEY is None:
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            enc_key.encode("utf-8"),
            b"rag-doc-encryption-v1",
            100_000,
        )
        _FERNET_KEY = Fernet(
            __import__("base64").urlsafe_b64encode(derived[:32])
        )
    return _FERNET_KEY


def is_encryption_enabled() -> bool:
    return bool(_encryption_key())


def encrypt_file(src_path: Path, dst_path: Path) -> None:
    """Encrypt a file from src_path, writing the ciphertext to dst_path."""
    fernet = _get_fernet()
    if fernet is None:
        # No encryption — just copy.
        import shutil
        shutil.copy2(src_path, dst_path)
        return
    data = src_path.read_bytes()
    encrypted = fernet.encrypt(data)
    dst_path.write_bytes(encrypted)


def decrypt_file(src_path: Path) -> bytes:
    """Read and decrypt a file. Returns plaintext bytes."""
    fernet = _get_fernet()
    if fernet is None:
        return src_path.read_bytes()
    encrypted = src_path.read_bytes()
    return fernet.decrypt(encrypted)


def encrypt_bytes(data: bytes) -> bytes:
    """Encrypt raw bytes (for in-memory use)."""
    fernet = _get_fernet()
    if fernet is None:
        return data
    return fernet.encrypt(data)


def decrypt_bytes(data: bytes) -> bytes:
    """Decrypt raw bytes."""
    fernet = _get_fernet()
    if fernet is None:
        return data
    return fernet.decrypt(data)
