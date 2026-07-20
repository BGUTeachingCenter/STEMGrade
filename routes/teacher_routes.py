from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Deque, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from core.security import (
    require_teacher,
    set_profile_session_cookie,
)
from services.student_access import create_student_codes
from services.teacher_profiles import (
    authenticate_teacher,
    change_teacher_password,
    create_teacher_account,
    get_teacher,
    list_teacher_courses,
    redeem_voucher_for_course,
    set_active_teacher_course,
    subject_options,
    update_teacher_profile,
)


router = APIRouter(
    prefix="/routes/teacher",
    tags=["teacher"],
)


# ---------------------------------------------------------------------
# Teacher login rate limiting
# ---------------------------------------------------------------------

_RATE_LIMIT_WINDOW_SECONDS = 60
_RATE_LIMIT_MAX_ATTEMPTS = 10
_RATE_LIMIT_LOCKOUT_SECONDS = 300

_rate_limit_lock = Lock()
_rate_limit_attempts: Dict[str, Deque[float]] = {}
_rate_limit_lockouts: Dict[str, float] = {}


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _check_rate_limit(ip: str) -> None:
    now = time.time()

    with _rate_limit_lock:
        locked_until = _rate_limit_lockouts.get(ip, 0)

        if locked_until > now:
            remaining = max(1, int(locked_until - now))
            raise HTTPException(
                status_code=429,
                detail=f"Too many attempts. Try again in {remaining}s.",
            )

        if locked_until:
            _rate_limit_lockouts.pop(ip, None)


def _record_failed_attempt(ip: str) -> None:
    now = time.time()
    cutoff = now - _RATE_LIMIT_WINDOW_SECONDS

    with _rate_limit_lock:
        attempts = _rate_limit_attempts.setdefault(ip, deque())

        while attempts and attempts[0] < cutoff:
            attempts.popleft()

        attempts.append(now)

        if len(attempts) >= _RATE_LIMIT_MAX_ATTEMPTS:
            _rate_limit_lockouts[ip] = (
                now + _RATE_LIMIT_LOCKOUT_SECONDS
            )
            attempts.clear()


def _record_successful_attempt(ip: str) -> None:
    with _rate_limit_lock:
        _rate_limit_attempts.pop(ip, None)
        _rate_limit_lockouts.pop(ip, None)


# ---------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------

class TeacherRegisterRequest(BaseModel):
    username: str
    password: str
    password_confirm: str


class TeacherLoginRequest(BaseModel):
    # "identifier" is kept temporarily so the current teacher_login.html
    # continues to work until its Phase 3 update.
    username: str = ""
    identifier: str = ""
    password: str


class UpdateTeacherProfileRequest(BaseModel):
    username: str | None = None

    # Temporary compatibility with the current course workspace.
    # The old page still sends "name".
    name: str | None = None

    subject: str | None = None
    grading_prompt_extra: str | None = None
    course_label: str | None = None


class ChangeTeacherPasswordRequest(BaseModel):
    old_password: str
    new_password: str
    new_password_confirm: str


class CreateTeacherCourseRequest(BaseModel):
    voucher_code: str
    course_label: str = ""
    subject: str = ""
    grading_prompt_extra: str = ""
    initial_student_count: int = 0


class SetActiveCourseRequest(BaseModel):
    course_id: str


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _teacher_id_from_session(session: dict) -> str:
    teacher_id = str(
        session.get("teacher_id")
        or session.get("sub")
        or ""
    ).strip()

    if not teacher_id:
        raise HTTPException(
            status_code=401,
            detail="Teacher session does not contain a teacher profile.",
        )

    return teacher_id


def _teacher_profile_from_session(session: dict) -> dict:
    teacher_id = _teacher_id_from_session(session)
    profile = get_teacher(teacher_id)

    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Teacher profile not found.",
        )

    return profile


# ---------------------------------------------------------------------
# Public teacher endpoints
# ---------------------------------------------------------------------

@router.get("/subjects")
def teacher_subject_options():
    return {
        "ok": True,
        "subjects": subject_options(),
    }


@router.post("/register")
def register_teacher_account(
    req: TeacherRegisterRequest,
    request: Request,
    response: Response,
):
    """
    Create a teacher profile only.

    This does not accept or redeem a course voucher.
    """
    ip = _client_ip(request)
    _check_rate_limit(ip)

    try:
        profile = create_teacher_account(
            username=req.username,
            password=req.password,
            password_confirm=req.password_confirm,
        )
    except HTTPException:
        _record_failed_attempt(ip)
        raise

    set_profile_session_cookie(
        response,
        role="teacher",
        sub=profile["teacher_id"],
        teacher_id=profile["teacher_id"],
    )

    _record_successful_attempt(ip)

    return {
        "ok": True,
        "role": "teacher",
        "teacher": profile,
        "courses": [],
        "active_course": {},
    }


@router.post("/login")
def login_teacher(
    req: TeacherLoginRequest,
    request: Request,
    response: Response,
):
    ip = _client_ip(request)
    _check_rate_limit(ip)

    username = (req.username or req.identifier or "").strip()

    if not username or not req.password:
        _record_failed_attempt(ip)
        raise HTTPException(
            status_code=400,
            detail="Enter both username and password.",
        )

    profile = authenticate_teacher(
        username,
        req.password,
    )

    if not profile:
        _record_failed_attempt(ip)
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    set_profile_session_cookie(
        response,
        role="teacher",
        sub=profile["teacher_id"],
        teacher_id=profile["teacher_id"],
    )

    _record_successful_attempt(ip)

    return {
        "ok": True,
        "role": "teacher",
        "teacher": profile,
        "courses": profile.get("courses") or [],
        "active_course": profile.get("active_course") or {},
    }


# ---------------------------------------------------------------------
# Authenticated teacher profile endpoints
# ---------------------------------------------------------------------

@router.get("/profile")
def get_teacher_profile(
    session: dict = Depends(require_teacher),
):
    teacher_id = _teacher_id_from_session(session)
    profile = _teacher_profile_from_session(session)

    return {
        "ok": True,
        "teacher": profile,
        "courses": list_teacher_courses(teacher_id),
        "active_course": profile.get("active_course") or {},
        "subjects": subject_options(),
    }


@router.post("/profile")
def save_teacher_profile(
    req: UpdateTeacherProfileRequest,
    session: dict = Depends(require_teacher),
):
    teacher_id = _teacher_id_from_session(session)

    # "name" remains a temporary alias for the current workspace.
    username = (
        req.username
        if req.username is not None
        else req.name
    )

    profile = update_teacher_profile(
        teacher_id=teacher_id,
        username=username,
        subject=req.subject,
        grading_prompt_extra=req.grading_prompt_extra,
        course_label=req.course_label,
    )

    return {
        "ok": True,
        "teacher": profile,
        "courses": profile.get("courses") or [],
        "active_course": profile.get("active_course") or {},
    }


@router.post("/change_password")
def change_password(
    req: ChangeTeacherPasswordRequest,
    session: dict = Depends(require_teacher),
):
    teacher_id = _teacher_id_from_session(session)

    change_teacher_password(
        teacher_id=teacher_id,
        old_password=req.old_password,
        new_password=req.new_password,
        new_password_confirm=req.new_password_confirm,
    )

    return {"ok": True}


# ---------------------------------------------------------------------
# Authenticated teacher course endpoints
# ---------------------------------------------------------------------

@router.get("/courses")
def get_teacher_courses(
    session: dict = Depends(require_teacher),
):
    teacher_id = _teacher_id_from_session(session)
    profile = _teacher_profile_from_session(session)

    return {
        "ok": True,
        "courses": list_teacher_courses(teacher_id),
        "active_course": profile.get("active_course") or {},
        "active_course_id": profile.get("active_course_id") or "",
        "subjects": subject_options(),
    }


@router.post("/courses")
def create_teacher_course(
    req: CreateTeacherCourseRequest,
    session: dict = Depends(require_teacher),
):
    teacher_id = _teacher_id_from_session(session)

    profile = redeem_voucher_for_course(
        teacher_id=teacher_id,
        voucher_code=req.voucher_code,
        course_label=req.course_label,
        subject=req.subject,
        grading_prompt_extra=req.grading_prompt_extra,
    )

    initial_count = max(
        0,
        min(int(req.initial_student_count or 0), 200),
    )

    initial_codes: list[dict] = []

    if initial_count:
        initial_codes = create_student_codes(
            teacher_id=teacher_id,
            count=initial_count,
            course_label=req.course_label,
        )

    return {
        "ok": True,
        "teacher": profile,
        "course": profile.get("active_course") or {},
        "courses": profile.get("courses") or [],
        "active_course": profile.get("active_course") or {},
        "initial_codes": initial_codes,
    }


@router.post("/active_course")
def select_active_course(
    req: SetActiveCourseRequest,
    session: dict = Depends(require_teacher),
):
    teacher_id = _teacher_id_from_session(session)

    profile = set_active_teacher_course(
        teacher_id=teacher_id,
        course_id=req.course_id,
    )

    return {
        "ok": True,
        "teacher": profile,
        "courses": profile.get("courses") or [],
        "active_course": profile.get("active_course") or {},
    }