from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Deque, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from core.security import require_admin, set_session_cookie, verify_admin_password
from services.admin_usage_stats import attach_usage_to_teachers, attach_usage_to_vouchers
from services.teacher_profiles import (
    create_voucher,
    list_teachers,
    list_vouchers,
    subject_options,
)

router = APIRouter(prefix="/routes/admin", tags=["admin"])

# Separate admin rate limiter.
# This keeps admin login protected without mixing admin routes into routes/auth.py.
_RL_WINDOW_SECONDS = 60
_RL_MAX_ATTEMPTS = 10
_RL_LOCKOUT_SECONDS = 300
_rl_lock = Lock()
_rl_attempts: Dict[str, Deque[float]] = {}
_rl_lockouts: Dict[str, float] = {}


class AdminLoginRequest(BaseModel):
    password: str


class CreateVoucherRequest(BaseModel):
    subject: str = "math"
    note: str = ""


def _client_ip(request: Request) -> str:
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


@router.post("/login")
def admin_login(req: AdminLoginRequest, request: Request, response: Response):
    ip = _client_ip(request)
    _check_rate_limit(ip)

    if not verify_admin_password(req.password):
        _record_failed_attempt(ip)
        raise HTTPException(status_code=401, detail="Invalid admin password")

    set_session_cookie(response, role="admin", sub="admin")
    _record_successful_attempt(ip)
    return {"ok": True, "role": "admin"}


@router.get("/vouchers")
def admin_list_vouchers(_session: dict = Depends(require_admin)):
    vouchers = attach_usage_to_vouchers(list_vouchers())
    return {
        "ok": True,
        "vouchers": vouchers,
        "subjects": subject_options(),
    }


@router.post("/vouchers")
def admin_create_voucher(req: CreateVoucherRequest, _session: dict = Depends(require_admin)):
    voucher = create_voucher(
        created_by=_session.get("sub", "admin"),
        subject=req.subject,
        note=req.note,
    )
    return {"ok": True, "voucher": voucher}


@router.get("/teachers")
def admin_list_teachers(_session: dict = Depends(require_admin)):
    teachers = attach_usage_to_teachers(list_teachers())
    return {
        "ok": True,
        "teachers": teachers,
        "subjects": subject_options(),
    }
