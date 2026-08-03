# routes/simple_ui.py
"""
Simple student UI (additive only).

Serves GET /simple/feedback: a minimal, RTL, handwriting-first student screen
that lives alongside the existing /feedback page. Nothing in this module edits
existing behaviour — new router, new template, new stylesheet.

The two-step flow the page drives:
    1. POST /routes/ocr_handwritten   -> student_tex + stored_work.work_id
    2. POST /routes/grade_tex_chatgpt -> graded PDF (or .tex fallback)
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from core.config import APP_NAME, PROJECT_ROOT
from core.security import get_session

router = APIRouter(prefix="/simple", tags=["simple-ui"])

# Own Jinja2 instance on purpose: routes/template_pages.py is a file the main
# developer edits, so we do not import from it and do not touch it.
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

TEMPLATE_NAME = "simple/student_page.html"


@router.get("/feedback")
def serve_simple_feedback(request: Request):
    """Render the simplified student screen for any signed-in session."""
    session = get_session(request)
    if not session:
        # Same behaviour as the existing /feedback page.
        return RedirectResponse(url="/student-login", status_code=303)

    template_path = PROJECT_ROOT / "templates" / TEMPLATE_NAME
    if not template_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"templates/{TEMPLATE_NAME} not found",
        )

    return templates.TemplateResponse(
        request=request,
        name=TEMPLATE_NAME,
        context={
            "app_name": APP_NAME,
            "role": session.get("role") or "student",
        },
    )
