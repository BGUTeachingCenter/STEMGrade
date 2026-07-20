# routes/template_pages.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from core.config import (
    PROJECT_ROOT,
    APP_NAME,
    APP_LOGO_MARK,
    APP_DOMAIN_LABEL,
    APP_HOME_SUBTITLE,
    APP_SUBTITLE,
    REFERENCE_BANK_LABEL,
    ASSESSMENT_LABEL,
)
from core.security import get_session
from services.teacher_profiles import get_teacher

router = APIRouter(tags=["templates"])
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))


def _brand_context() -> dict:
    return {
        "app_name": APP_NAME,
        "app_logo_mark": APP_LOGO_MARK,
        "app_domain_label": APP_DOMAIN_LABEL,
        "app_home_subtitle": APP_HOME_SUBTITLE,
        "app_subtitle": APP_SUBTITLE,
        "reference_bank_label": REFERENCE_BANK_LABEL,
        "assessment_label": ASSESSMENT_LABEL,
    }


def _render(request: Request, name: str, context: dict | None = None, status_code: int = 200):
    merged_context = _brand_context()
    merged_context.update(context or {})
    return templates.TemplateResponse(
        request=request,
        name=name,
        context=merged_context,
        status_code=status_code,
    )


@router.get("/")
def serve_index(request: Request):
    p = PROJECT_ROOT / "templates" / "index.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="templates/index.html not found")
    return _render(request, "index.html")


@router.get("/feedback")
def serve_feedback(request: Request):
    s = get_session(request)
    if not s:
        return RedirectResponse(url="/student-login", status_code=303)

    # Students and teachers may both use the feedback machine.
    # Teacher usage is test mode and should not be counted as student submissions.
    p = PROJECT_ROOT / "templates" / "student_page.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="templates/student_page.html not found")
    return _render(
        request,
        "student_page.html",
        context={
            "session_role": s.get("role"),
            "is_teacher_test": s.get("role") == "teacher",
        },
    )


@router.get("/admin-login")
def admin_login_page(request: Request):
    p = PROJECT_ROOT / "templates" / "admin_login.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="templates/admin_login.html not found")
    return _render(request, "admin_login.html")


@router.get("/admin")
def serve_admin(request: Request):
    s = get_session(request)
    if not s or s.get("role") != "admin":
        return RedirectResponse(url="/admin-login", status_code=303)
    p = PROJECT_ROOT / "templates" / "admin_page.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="templates/admin_page.html not found")
    return _render(request, "admin_page.html")


@router.get("/teacher-register")
def teacher_register_page(request: Request):
    session = get_session(request)

    if session and session.get("role") == "teacher":
        return RedirectResponse(
            url="/teacher-profile",
            status_code=303,
        )

    p = PROJECT_ROOT / "templates" / "teacher_register.html"

    if not p.exists():
        raise HTTPException(
            status_code=404,
            detail="templates/teacher_register.html not found",
        )

    return _render(request, "teacher_register.html")


@router.get("/teacher-profile")
def serve_teacher_profile(request: Request):
    session = get_session(request)

    if not session or session.get("role") != "teacher":
        return RedirectResponse(
            url="/teacher-login",
            status_code=303,
        )

    path = PROJECT_ROOT / "templates" / "teacher_profile.html"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="templates/teacher_profile.html not found",
        )

    return _render(request, "teacher_profile.html")


@router.get("/course-workspace")
def serve_course_workspace(request: Request):
    session = get_session(request)

    if not session or session.get("role") != "teacher":
        return RedirectResponse(
            url="/teacher-login",
            status_code=303,
        )

    teacher_id = str(
        session.get("teacher_id")
        or session.get("sub")
        or ""
    ).strip()

    profile = get_teacher(teacher_id)

    if not profile:
        return RedirectResponse(
            url="/teacher-login",
            status_code=303,
        )

    courses = profile.get("courses") or []
    active_course = profile.get("active_course") or {}

    # Accounts without courses belong on the profile page, where they can
    # redeem their first voucher.
    if not courses or not active_course.get("course_id"):
        return RedirectResponse(
            url="/teacher-profile",
            status_code=303,
        )

    path = PROJECT_ROOT / "templates" / "course_workspace.html"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="templates/course_workspace.html not found",
        )

    return _render(
        request,
        "course_workspace.html",
    )


@router.get("/teacher")
def legacy_teacher_page_redirect(request: Request):
    """
    Temporary compatibility redirect for old bookmarks and links.
    """
    session = get_session(request)

    if not session or session.get("role") != "teacher":
        return RedirectResponse(
            url="/teacher-login",
            status_code=303,
        )

    return RedirectResponse(
        url="/teacher-profile",
        status_code=303,
    )


@router.get("/teacher-login")
def teacher_login(request: Request):
    session = get_session(request)

    if session and session.get("role") == "teacher":
        return RedirectResponse(
            url="/teacher-profile",
            status_code=303,
        )

    path = PROJECT_ROOT / "templates" / "teacher_login.html"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="templates/teacher_login.html not found",
        )

    return _render(request, "teacher_login.html")


@router.get("/admin-login")
def admin_login(request: Request):
    s = get_session(request)
    if s and s.get("role") == "admin":
        return RedirectResponse(url="/admin", status_code=303)

    p = PROJECT_ROOT / "templates" / "admin_login.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="templates/admin_login.html not found")
    return _render(request, "admin_login.html")


@router.get("/admin")
def serve_admin(request: Request):
    s = get_session(request)
    if not s or s.get("role") != "admin":
        return RedirectResponse(url="/admin-login", status_code=303)

    p = PROJECT_ROOT / "templates" / "admin_page.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="templates/admin_page.html not found")
    return _render(request, "admin_page.html")


@router.get("/student-login")
def student_login(request: Request):
    s = get_session(request)

    if s and s.get("role") in {"student", "teacher"}:
        return RedirectResponse(url="/feedback", status_code=303)

    p = PROJECT_ROOT / "templates" / "student_login.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="templates/student_login.html not found")
    return _render(request, "student_login.html")
