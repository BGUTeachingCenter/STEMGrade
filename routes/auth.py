# routes/auth.py
from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Deque, Dict, Set

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from core.config import RUNS_ROOT
from core.security import (
    clear_session_cookie,
    get_session,
    set_session_cookie,
    set_profile_session_cookie,
    require_teacher,
)
from services.teacher_profiles import (
    authenticate_teacher,
    change_teacher_password,
    get_teacher,
    register_teacher_from_voucher,
    subject_options,
    update_teacher_profile,
)

from openpyxl import Workbook, load_workbook

router = APIRouter(prefix="/routes", tags=["auth"])

ALLOWLIST_XLSX = RUNS_ROOT / "student_codes_allowlist.xlsx"
ALLOWLIST_SHEET = "allowlist"

# --- Rate limiting -----------------------------------------------------------
# Simple per-IP sliding window for the login endpoint. Not a substitute for a
# real rate limiter, but enough to slow online brute force of student codes
_RL_WINDOW_SECONDS = 60
_RL_MAX_ATTEMPTS = 10
_RL_LOCKOUT_SECONDS = 300
_rl_lock = Lock()
_rl_attempts: Dict[str, Deque[float]] = {}
_rl_lockouts: Dict[str, float] = {}


def _client_ip(request: Request) -> str:
    # If you front this with a reverse proxy, populate X-Forwarded-For carefully
    # — accepting it blindly lets clients spoof the rate-limit key.
    return request.client.host if request.client else "unknown"


def _check_rate_limit(ip: str) -> None:
    now = time.time()
    with _rl_lock:
        locked_until = _rl_lockouts.get(ip, 0)
        if locked_until > now:
            raise HTTPException(
                status_code=429,
                detail=f"Too many attempts. Try again in {int(locked_until - now)}s.",
            )
        if locked_until and locked_until <= now:
            _rl_lockouts.pop(ip, None)


def _record_failed_attempt(ip: str) -> None:
    now = time.time()
    cutoff = now - _RL_WINDOW_SECONDS
    with _rl_lock:
        dq = _rl_attempts.setdefault(ip, deque())
        while dq and dq[0] < cutoff:
            dq.popleft()
        dq.append(now)
        if len(dq) >= _RL_MAX_ATTEMPTS:
            _rl_lockouts[ip] = now + _RL_LOCKOUT_SECONDS
            dq.clear()


def _record_successful_attempt(ip: str) -> None:
    with _rl_lock:
        _rl_attempts.pop(ip, None)
        _rl_lockouts.pop(ip, None)


# --- Allowlist ---------------------------------------------------------------

class LoginRequest(BaseModel):
    code: str


class TeacherLoginRequest(BaseModel):
    identifier: str
    password: str


class TeacherRegisterRequest(BaseModel):
    voucher_code: str
    teacher_name: str
    teacher_email: str
    subject: str = "math"
    password: str
    password_confirm: str
    grading_prompt_extra: str = ""


class UpdateTeacherProfileRequest(BaseModel):
    subject: str | None = None
    grading_prompt_extra: str | None = None
    name: str | None = None


class ChangeTeacherPasswordRequest(BaseModel):
    old_password: str
    new_password: str
    new_password_confirm: str


def _normalize_code(code: str) -> str:
    return (code or "").strip()


def _ensure_allowlist_file_exists() -> None:
    ALLOWLIST_XLSX.parent.mkdir(parents=True, exist_ok=True)
    if ALLOWLIST_XLSX.exists():
        return
    wb = Workbook()
    ws = wb.active
    ws.title = ALLOWLIST_SHEET
    ws.append(["code"])
    wb.save(ALLOWLIST_XLSX)


def _read_allowlist_codes() -> Set[str]:
    _ensure_allowlist_file_exists()
    wb = load_workbook(ALLOWLIST_XLSX)
    ws = wb[ALLOWLIST_SHEET] if ALLOWLIST_SHEET in wb.sheetnames else wb.active

    codes: Set[str] = set()
    for row in ws.iter_rows(min_row=2, max_col=1, values_only=True):
        v = row[0]
        if v is None:
            continue
        s = _normalize_code(str(v))
        if s:
            codes.add(s)
    return codes


def is_allowed_student_code(code: str) -> bool:
    code = _normalize_code(code)
    if not code:
        return False
    return code in _read_allowlist_codes()


@router.post("/login")
def login(req: LoginRequest, request: Request, response: Response):
    ip = _client_ip(request)
    _check_rate_limit(ip)

    code = _normalize_code(req.code)
    if not code:
        _record_failed_attempt(ip)
        raise HTTPException(status_code=400, detail="Missing code")

    if not is_allowed_student_code(code):
        _record_failed_attempt(ip)
        raise HTTPException(status_code=401, detail="Invalid code")

    set_session_cookie(response, role="student", sub=code)
    _record_successful_attempt(ip)
    return {"ok": True, "role": "student"}


@router.get("/teacher/subjects")
def teacher_subject_options():
    return {"ok": True, "subjects": subject_options()}


@router.post("/teacher/register")
def teacher_register(req: TeacherRegisterRequest, request: Request, response: Response):
    ip = _client_ip(request)
    _check_rate_limit(ip)
    profile = register_teacher_from_voucher(
        voucher_code=req.voucher_code,
        teacher_name=req.teacher_name,
        teacher_email=req.teacher_email,
        subject=req.subject,
        password=req.password,
        password_confirm=req.password_confirm,
        grading_prompt_extra=req.grading_prompt_extra,
    )
    set_profile_session_cookie(
        response,
        role="teacher",
        sub=profile["teacher_id"],
        teacher_id=profile["teacher_id"],
    )
    _record_successful_attempt(ip)
    return {"ok": True, "role": "teacher", "teacher": profile}


@router.post("/teacher/login")
def teacher_profile_login(req: TeacherLoginRequest, request: Request, response: Response):
    ip = _client_ip(request)
    _check_rate_limit(ip)
    profile = authenticate_teacher(req.identifier, req.password)
    if not profile:
        _record_failed_attempt(ip)
        raise HTTPException(status_code=401, detail="Invalid teacher email/ID or password")
    set_profile_session_cookie(
        response,
        role="teacher",
        sub=profile["teacher_id"],
        teacher_id=profile["teacher_id"],
    )
    _record_successful_attempt(ip)
    return {"ok": True, "role": "teacher", "teacher": profile}


@router.get("/teacher/profile")
def teacher_profile(_session: dict = Depends(require_teacher)):
    teacher_id = _session.get("teacher_id") or _session.get("sub")
    profile = get_teacher(teacher_id)
    if not profile:
        # Legacy env-code teacher. Keep the portal usable, but make it explicit.
        return {
            "ok": True,
            "legacy": True,
            "teacher": {
                "teacher_id": "legacy_teacher",
                "name": "Legacy teacher",
                "email": "",
                "subject": "math",
                "subject_label": "Math",
                "grading_prompt_extra": "",
            },
            "subjects": subject_options(),
        }
    return {"ok": True, "legacy": False, "teacher": profile, "subjects": subject_options()}


@router.post("/teacher/profile")
def teacher_profile_update(req: UpdateTeacherProfileRequest, _session: dict = Depends(require_teacher)):
    teacher_id = _session.get("teacher_id") or _session.get("sub")
    if not teacher_id or teacher_id == "teacher":
        raise HTTPException(status_code=400, detail="Legacy teacher sessions cannot edit a saved profile.")
    profile = update_teacher_profile(
        teacher_id=teacher_id,
        subject=req.subject,
        grading_prompt_extra=req.grading_prompt_extra,
        name=req.name,
    )
    return {"ok": True, "teacher": profile}


@router.post("/teacher/change_password")
def teacher_change_password(req: ChangeTeacherPasswordRequest, _session: dict = Depends(require_teacher)):
    teacher_id = _session.get("teacher_id") or _session.get("sub")
    if not teacher_id or teacher_id == "teacher":
        raise HTTPException(status_code=400, detail="Legacy teacher sessions do not have a saved profile password.")
    change_teacher_password(
        teacher_id=teacher_id,
        old_password=req.old_password,
        new_password=req.new_password,
        new_password_confirm=req.new_password_confirm,
    )
    return {"ok": True}


@router.post("/logout")
def logout(response: Response):
    clear_session_cookie(response)
    return {"ok": True}


@router.get("/me")
def me(request: Request):
    s = get_session(request)
    if not s:
        return {"ok": False, "role": None}
    # Never leak the student code back to the client; the cookie itself is
    # the source of truth.
    teacher_id = s.get("teacher_id") or (s.get("sub") if s.get("role") == "teacher" else "")
    return {"ok": True, "role": s.get("role"), "teacher_id": teacher_id}
