# core/security.py
from __future__ import annotations

from fastapi import HTTPException
from .config import TEACHER_PASSWORD

def require_teacher_password(pw: str | None) -> None:
    """Raise 401 if the teacher password is missing/incorrect.

    Password is stored in env var TEACHER_PASSWORD.
    """
    if not TEACHER_PASSWORD:
        raise HTTPException(
            status_code=500,
            detail="TEACHER_PASSWORD is not set in .env / environment.",
        )
    if not pw or pw != TEACHER_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized (bad teacher password).")