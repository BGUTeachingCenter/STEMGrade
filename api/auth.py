# api/auth.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Set

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.config import RUNS_ROOT, TEACHER_PASSWORD

from openpyxl import Workbook, load_workbook

router = APIRouter(prefix="/api", tags=["auth"])

ALLOWLIST_XLSX = RUNS_ROOT / "student_codes_allowlist.xlsx"
ALLOWLIST_SHEET = "allowlist"


class LoginRequest(BaseModel):
    code: str


def _normalize_code(code: str) -> str:
    return (code or "").strip()


def _ensure_allowlist_file_exists() -> None:
    """Create an empty allowlist template if it doesn't exist yet."""
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

    # Expect header in row 1, codes from row 2 onwards in column A
    codes: Set[str] = set()
    for row in ws.iter_rows(min_row=2, max_col=1, values_only=True):
        v = row[0]
        if v is None:
            continue
        s = _normalize_code(str(v))
        if s:
            codes.add(s)
    return codes


def is_teacher_code(code: str) -> bool:
    code = _normalize_code(code)
    return bool(TEACHER_PASSWORD) and code == TEACHER_PASSWORD


def is_allowed_student_code(code: str) -> bool:
    code = _normalize_code(code)
    if not code:
        return False
    if is_teacher_code(code):
        return False
    allow = _read_allowlist_codes()
    return code in allow


@router.post("/login")
def login(req: LoginRequest, request: Request):
    code = _normalize_code(req.code)
    if not code:
        raise HTTPException(status_code=400, detail="Missing code")

    # Teacher/admin code
    if is_teacher_code(code):
        return {"ok": True, "role": "teacher"}

    # Student code must be in allowlist
    if not is_allowed_student_code(code):
        raise HTTPException(status_code=401, detail="Invalid student code")

    return {"ok": True, "role": "student"}