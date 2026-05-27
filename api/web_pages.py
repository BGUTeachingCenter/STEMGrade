# api/web_pages.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from core.config import PROJECT_ROOT
from core.security import get_session

router = APIRouter(tags=["web"])
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "web"))


def _render(request: Request, name: str, context: dict | None = None, status_code: int = 200):
    return templates.TemplateResponse(
        request=request,
        name=name,
        context=context or {},
        status_code=status_code,
    )


@router.get("/")
def serve_index(request: Request):
    p = PROJECT_ROOT / "web" / "index.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return _render(request, "index.html")


@router.get("/feedback")
def serve_feedback(request: Request):
    s = get_session(request)
    if not s:
        return RedirectResponse(url="/student-login", status_code=303)
    if s.get("role") == "teacher":
        return RedirectResponse(url="/teacher", status_code=303)

    p = PROJECT_ROOT / "web" / "feedback.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="web/feedback.html not found")
    return _render(request, "feedback.html")


@router.get("/teacher")
def serve_teacher(request: Request):
    s = get_session(request)
    if not s or s.get("role") != "teacher":
        return RedirectResponse(url="/teacher-login", status_code=303)
    pth = PROJECT_ROOT / "web" / "teacher.html"
    if not pth.exists():
        raise HTTPException(status_code=404, detail="web/teacher.html not found")
    return _render(request, "teacher.html")


@router.get("/teacher-login")
def teacher_login(request: Request):
    p = PROJECT_ROOT / "web" / "teacher_login.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="web/teacher_login.html not found")
    return _render(request, "teacher_login.html")


@router.get("/student-login")
def student_login(request: Request):
    s = get_session(request)
    if s and s.get("role") == "student":
        return RedirectResponse(url="/feedback", status_code=303)
    if s and s.get("role") == "teacher":
        return RedirectResponse(url="/teacher", status_code=303)

    p = PROJECT_ROOT / "web" / "student_login.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="web/student_login.html not found")
    return _render(request, "student_login.html")
