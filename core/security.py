# core/security.py
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Optional

from fastapi import HTTPException, Request, Response

from .config import (
    ADMIN_PASSWORD,
    COOKIE_SAMESITE,
    COOKIE_SECURE,
    SESSION_SECRET,
    SESSION_TTL_SECONDS,
    TEACHER_PASSWORD,
)

COOKIE_NAME = "mathgrade_session"
PASSWORD_HASH_SCHEME = "pbkdf2_sha256"
PASSWORD_HASH_ITERATIONS = 260_000

# If SESSION_SECRET is not configured, fall back to an ephemeral per-process
# secret so the app still works in dev — but sessions won't survive restarts
# and a deployer running multiple workers will see broken auth. The /routes/login
# endpoint surfaces a warning when this happens.
_RUNTIME_SECRET = SESSION_SECRET or secrets.token_urlsafe(48)
SESSION_SECRET_CONFIGURED = bool(SESSION_SECRET)


def _b64u_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _b64u_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _sign(payload_b64: str) -> str:
    digest = hmac.new(
        _RUNTIME_SECRET.encode("utf-8"),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return _b64u_encode(digest)


def encode_session(role: str, sub: str, ttl_seconds: Optional[int] = None) -> str:
    ttl = ttl_seconds if ttl_seconds is not None else SESSION_TTL_SECONDS
    payload = {
        "role": role,
        "sub": sub,
        "exp": int(time.time()) + int(ttl),
        "iat": int(time.time()),
    }
    payload_b64 = _b64u_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    return f"{payload_b64}.{_sign(payload_b64)}"


def decode_session(token: Optional[str]) -> Optional[dict]:
    if not token or token.count(".") != 1:
        return None
    payload_b64, sig = token.split(".", 1)
    expected = _sign(payload_b64)
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        data = json.loads(_b64u_decode(payload_b64).decode("utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if int(data.get("exp", 0)) < int(time.time()):
        return None
    if data.get("role") not in ("admin", "teacher", "student"):
        return None
    return data


def set_session_cookie(response: Response, role: str, sub: str) -> None:
    token = encode_session(role, sub)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        path="/",
    )




def set_profile_session_cookie(response: Response, *, role: str, sub: str, teacher_id: str = "") -> None:
    """Set a signed session cookie with optional teacher profile metadata."""
    ttl = SESSION_TTL_SECONDS
    payload = {
        "role": role,
        "sub": sub,
        "exp": int(time.time()) + int(ttl),
        "iat": int(time.time()),
    }
    if teacher_id:
        payload["teacher_id"] = teacher_id

    payload_b64 = _b64u_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    token = f"{payload_b64}.{_sign(payload_b64)}"
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/")


def get_session(request: Request) -> Optional[dict]:
    return decode_session(request.cookies.get(COOKIE_NAME))


def require_session(request: Request) -> dict:
    s = get_session(request)
    if not s:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return s


def require_admin(request: Request) -> dict:
    s = require_session(request)
    if s.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return s


def require_teacher(request: Request) -> dict:
    s = require_session(request)
    if s.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="Teacher access required")
    return s


def require_student_or_teacher(request: Request) -> dict:
    # Same as require_session today, but explicit for call sites.
    return require_session(request)


def hash_password(password: str) -> str:
    """Return a salted PBKDF2 password hash safe to store on disk."""
    password = password or ""
    salt = secrets.token_urlsafe(24)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PASSWORD_HASH_ITERATIONS,
    )
    return f"{PASSWORD_HASH_SCHEME}${PASSWORD_HASH_ITERATIONS}${salt}${_b64u_encode(digest)}"


def verify_password_hash(password: Optional[str], stored_hash: Optional[str]) -> bool:
    """Constant-time PBKDF2 password-hash verification."""
    if not password or not stored_hash:
        return False
    try:
        scheme, iterations_s, salt, digest_b64 = stored_hash.split("$", 3)
        if scheme != PASSWORD_HASH_SCHEME:
            return False
        iterations = int(iterations_s)
        expected = _b64u_decode(digest_b64)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations,
        )
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def verify_admin_password(pw: Optional[str]) -> bool:
    """Constant-time check of the app admin password against env."""
    if not ADMIN_PASSWORD or not pw:
        return False
    return hmac.compare_digest(pw.encode("utf-8"), ADMIN_PASSWORD.encode("utf-8"))


def verify_teacher_password(pw: Optional[str]) -> bool:
    """Backward-compatible legacy teacher-code check against env."""
    if not TEACHER_PASSWORD or not pw:
        return False
    return hmac.compare_digest(pw.encode("utf-8"), TEACHER_PASSWORD.encode("utf-8"))
