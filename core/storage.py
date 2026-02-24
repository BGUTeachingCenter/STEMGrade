# core/storage.py
from __future__ import annotations

import re
from pathlib import Path
from fastapi import HTTPException
from .config import BANK_ROOT

_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,80}$")

def require_safe_exam_id(exam_id: str) -> str:
    """Validate bank exam_id to avoid path traversal and weird names."""
    exam_id = (exam_id or "").strip()
    if not _SAFE_NAME_RE.match(exam_id):
        raise HTTPException(status_code=400, detail="Invalid exam_id. Use letters/numbers/_/- only.")
    return exam_id

def require_safe_filename(name: str) -> str:
    """Validate filename for safe access within uploads folder."""
    name = (name or "").strip()
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not name.lower().endswith(".tex"):
        raise HTTPException(status_code=400, detail="Filename must end with .tex")
    return name

def exam_dir(exam_id: str) -> Path:
    return BANK_ROOT / exam_id

def uploads_dir(exam_id: str) -> Path:
    d = exam_dir(exam_id) / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d