# api/web_pages.py
from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from core.config import PROJECT_ROOT
from core.security import require_teacher_password

router = APIRouter(tags=["web"])

@router.get("/", response_class=HTMLResponse)
def serve_index():
    p = PROJECT_ROOT / "web" / "index.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return p.read_text(encoding="utf-8", errors="replace")

@router.get("/teacher", response_class=HTMLResponse)
def serve_teacher(p: str | None = None):
    require_teacher_password(p)
    pth = PROJECT_ROOT / "web" / "teacher.html"
    if not pth.exists():
        raise HTTPException(status_code=404, detail="web/teacher.html not found")
    return pth.read_text(encoding="utf-8", errors="replace")


@router.get("/teacher-login")
def teacher_login():
    p = PROJECT_ROOT / "web" / "teacher_login.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="web/teacher_login.html not found")
    return FileResponse(str(p))