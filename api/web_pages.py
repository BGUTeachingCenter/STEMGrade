# api/web_pages.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from core.config import PROJECT_ROOT
from core.security import get_session

router = APIRouter(tags=["web"])

@router.get("/", response_class=HTMLResponse)
def serve_index():
    p = PROJECT_ROOT / "web" / "index.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return p.read_text(encoding="utf-8", errors="replace")

@router.get("/teacher")
def serve_teacher(request: Request):
    # Page-level gate: redirect unauthenticated visitors to the login page
    # instead of returning a JSON 401 (better UX for a navigable URL).
    s = get_session(request)
    if not s or s.get("role") != "teacher":
        return RedirectResponse(url="/teacher-login", status_code=303)
    pth = PROJECT_ROOT / "web" / "teacher.html"
    if not pth.exists():
        raise HTTPException(status_code=404, detail="web/teacher.html not found")
    return HTMLResponse(pth.read_text(encoding="utf-8", errors="replace"))


@router.get("/teacher-login")
def teacher_login():
    p = PROJECT_ROOT / "web" / "teacher_login.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="web/teacher_login.html not found")
    return FileResponse(str(p))