"""Argon2id password hashing with one-login legacy SHA-256 upgrades."""

from __future__ import annotations

import hashlib
import hmac
import re

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

_hasher = PasswordHasher()
_legacy_sha256 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("Password cannot be empty.")
    return _hasher.hash(password)


def verify_password(stored_hash: str, password: str) -> tuple[bool, str | None]:
    """Return ``(valid, upgraded_hash)`` for Argon2id or legacy SHA-256."""
    if not stored_hash or not password:
        return False, None
    if stored_hash.startswith("$argon2"):
        try:
            valid = _hasher.verify(stored_hash, password)
        except (VerifyMismatchError, InvalidHashError):
            return False, None
        return bool(valid), hash_password(password) if _hasher.check_needs_rehash(stored_hash) else None
    if _legacy_sha256.fullmatch(stored_hash):
        candidate = hashlib.sha256(password.encode("utf-8")).hexdigest()
        if hmac.compare_digest(stored_hash.lower(), candidate):
            return True, hash_password(password)
    return False, None
