# routes/simple_assignments.py
"""
Assignments for the simple UI (additive only).

An "assignment" is a thin layer of metadata on top of an existing solution-bank
exam folder. It is stored as one small file next to the uploads:

    <bank_root>/<exam_id>/assignment.json

    {
      "schema_version": "simple_assignment_v1",
      "exam_id": "hw-20260804-4f2a",
      "title": "תרגיל 1 — גבולות ורציפות",   # Hebrew display name
      "due_at": "2026-03-12",                # "" means no deadline
      "published": true,                     # the teacher's checkbox
      "max_attempts": 2,
      "created_at": "...",
      "updated_at": "..."
    }

Why a separate file and a separate router:
  * routes/solution_bank_routes.py is a large file the main developer owns.
    Nothing here edits it — the teacher screen posts the file to the existing
    POST /routes/bank/upload unchanged, then posts metadata here.
  * exam_id must stay ASCII (core.storage._SAFE_NAME_RE) because it becomes a
    folder name, so the Hebrew title the teacher types lives in this file
    instead of in the folder name.

The deadline is display-only for now: nothing here blocks a late submission.
"""
from __future__ import annotations

import json
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.security import require_session, require_teacher
from core.storage import (
    bind_bank_root,
    exam_dir,
    require_safe_exam_id,
    teacher_bank_root,
)
from services.student_access import get_student_code_record
from services.teacher_profiles import get_teacher

router = APIRouter(prefix="/simple/api", tags=["simple-assignments"])

ASSIGNMENT_FILENAME = "assignment.json"
SCHEMA_VERSION = "simple_assignment_v1"

DEFAULT_MAX_ATTEMPTS = 2
MAX_ATTEMPTS_CEILING = 10

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Resolving which bank to read
# ---------------------------------------------------------------------------

def _voucher_id_for_teacher(teacher_id: str) -> str:
    """The active course id for this teacher, mirroring the bank routes."""
    teacher = get_teacher(teacher_id) or {}
    active_course = (
        teacher.get("active_course")
        if isinstance(teacher.get("active_course"), dict)
        else {}
    )
    return str(
        active_course.get("voucher_id")
        or active_course.get("course_id")
        or teacher.get("voucher_id")
        or teacher.get("voucher_hash")
        or teacher.get("voucher_code")
        or teacher.get("active_course_id")
        or ""
    ).strip()


def _bind_teacher_bank(session: dict) -> Path:
    teacher_id = str(
        session.get("teacher_id") or session.get("sub") or ""
    ).strip()

    if not teacher_id or teacher_id == "teacher":
        raise HTTPException(
            status_code=400,
            detail="Teacher profile is required for assignments.",
        )

    root = teacher_bank_root(
        teacher_id,
        voucher_id=_voucher_id_for_teacher(teacher_id),
    )
    bind_bank_root(root)
    return root


def _bind_student_bank(session: dict) -> Path:
    """Find the bank of the teacher who issued this student's code."""
    code = str(session.get("sub") or "").strip()
    if not code:
        raise HTTPException(
            status_code=403,
            detail="Missing student code in session.",
        )

    record = get_student_code_record(code) or {}
    teacher_id = str(record.get("teacher_id") or "").strip()

    if not teacher_id:
        raise HTTPException(
            status_code=403,
            detail="This student code is not attached to a course.",
        )

    voucher_id = str(
        record.get("voucher_id")
        or record.get("course_id")
        or record.get("voucher_hash")
        or record.get("voucher_code")
        or ""
    ).strip()

    root = teacher_bank_root(teacher_id, voucher_id=voucher_id)
    bind_bank_root(root)
    return root


def _bind_bank_for_session(session: dict) -> Path:
    role = str(session.get("role") or "").strip()

    if role == "teacher":
        return _bind_teacher_bank(session)
    if role == "student":
        return _bind_student_bank(session)

    raise HTTPException(
        status_code=403,
        detail="This session cannot read assignments.",
    )


# ---------------------------------------------------------------------------
# Reading and writing assignment.json
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _assignment_path(exam_id: str) -> Path:
    return exam_dir(exam_id) / ASSIGNMENT_FILENAME


def _clean_title(value: str) -> str:
    """One line, no control characters, capped."""
    title = re.sub(r"\s+", " ", str(value or "")).strip()
    return title[:120]


def _clean_due_at(value: str) -> str:
    """Accept an ISO date or an empty string. Anything else is rejected."""
    due = str(value or "").strip()
    if not due:
        return ""

    if not _DATE_RE.match(due):
        raise HTTPException(
            status_code=400,
            detail="due_at must be an ISO date (YYYY-MM-DD) or empty.",
        )

    try:
        datetime.strptime(due, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="due_at is not a real date.",
        ) from exc

    return due


def _clean_max_attempts(value: Any) -> int:
    try:
        attempts = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_ATTEMPTS
    return max(1, min(attempts, MAX_ATTEMPTS_CEILING))


def read_assignment(exam_id: str) -> dict[str, Any] | None:
    """The stored metadata for one exam, or None when it was never set."""
    path = _assignment_path(exam_id)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    return data if isinstance(data, dict) else None


def write_assignment(exam_id: str, data: dict[str, Any]) -> dict[str, Any]:
    path = _assignment_path(exam_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
    return data


def _new_exam_id(bank_root: Path, title: str) -> str:
    """
    Build an ASCII, collision-free folder name.

    The Hebrew title cannot be used: core.storage rejects non-ASCII exam ids.
    A Latin title still gives a readable folder; otherwise we fall back to a
    dated 'hw' prefix.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower()[:40]
    base = slug or ("hw-" + datetime.now().strftime("%Y%m%d"))

    if not re.match(r"^[a-zA-Z0-9]", base):
        base = "hw-" + base

    for _ in range(50):
        candidate = base + "-" + secrets.token_hex(2)
        if not (bank_root / candidate).exists():
            return require_safe_exam_id(candidate)

    raise HTTPException(
        status_code=500,
        detail="Could not allocate a folder name for this assignment.",
    )


# ---------------------------------------------------------------------------
# Reading the state of an exam folder
# ---------------------------------------------------------------------------

def _exam_state(exam_id: str) -> dict[str, Any]:
    """
    What the bank holds for this exam.

    gradeable is the flag that matters to a student: without
    full_solution_bundle.json the grader has nothing to compare against, and a
    submission would fail with "no reference found".
    """
    uploads = exam_dir(exam_id) / "uploads"

    has_questions = (uploads / "questions_only_bundle.json").exists()
    has_answers = (uploads / "answers_only_bundle.json").exists()
    gradeable = (uploads / "full_solution_bundle.json").exists()

    source_files: list[str] = []
    if uploads.exists():
        source_files = sorted(
            p.name
            for p in uploads.iterdir()
            if p.is_file() and p.suffix.lower() in {".tex", ".txt"}
        )

    return {
        "gradeable": gradeable,
        "has_questions": has_questions,
        "has_answers": has_answers,
        "has_any_file": bool(source_files) or has_questions or has_answers,
        "source_files": source_files,
    }


def _list_exam_ids(bank_root: Path) -> list[str]:
    if not bank_root.exists():
        return []
    return sorted(p.name for p in bank_root.iterdir() if p.is_dir())


def _assignment_payload(exam_id: str, *, for_teacher: bool) -> dict[str, Any]:
    stored = read_assignment(exam_id) or {}
    state = _exam_state(exam_id)

    payload = {
        "exam_id": exam_id,
        "title": _clean_title(stored.get("title") or "") or exam_id,
        "due_at": str(stored.get("due_at") or ""),
        "published": bool(stored.get("published")),
        "max_attempts": _clean_max_attempts(
            stored.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
        ),
        "gradeable": state["gradeable"],
        "has_metadata": bool(stored),
    }

    if for_teacher:
        payload.update(
            {
                "has_questions": state["has_questions"],
                "has_answers": state["has_answers"],
                "has_any_file": state["has_any_file"],
                "source_files": state["source_files"],
                "created_at": str(stored.get("created_at") or ""),
                "updated_at": str(stored.get("updated_at") or ""),
            }
        )

    return payload


def _sort_key(item: dict[str, Any]) -> tuple[int, str, str]:
    """Dated assignments first, in date order; undated last, by title."""
    due = item.get("due_at") or ""
    return (0, due, item["title"]) if due else (1, "", item["title"])


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class NewAssignmentReq(BaseModel):
    title: str = ""
    due_at: str = ""
    published: bool = False
    max_attempts: int = DEFAULT_MAX_ATTEMPTS


class UpdateAssignmentReq(BaseModel):
    exam_id: str
    title: str = ""
    due_at: str = ""
    published: bool = False
    max_attempts: int = DEFAULT_MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# Student endpoint
# ---------------------------------------------------------------------------

@router.get("/assignments")
def list_assignments(_session: dict = Depends(require_session)):
    """
    The rows a student sees.

    Only published assignments are returned; an unpublished exam in the bank
    stays invisible, so a teacher can keep past exams as grading references
    without them showing up as homework.

    A teacher hitting this endpoint gets their own published list, which makes
    "preview what my students see" a one-line call.
    """
    bank_root = _bind_bank_for_session(_session)

    items = [
        _assignment_payload(exam_id, for_teacher=False)
        for exam_id in _list_exam_ids(bank_root)
    ]
    published = [item for item in items if item["published"]]
    published.sort(key=_sort_key)

    return {"ok": True, "assignments": published}


# ---------------------------------------------------------------------------
# Teacher endpoints
# ---------------------------------------------------------------------------

@router.get("/teacher/assignments")
def teacher_list_assignments(_session: dict = Depends(require_teacher)):
    """Every exam in the active course's bank, published or not."""
    bank_root = _bind_teacher_bank(_session)

    items = [
        _assignment_payload(exam_id, for_teacher=True)
        for exam_id in _list_exam_ids(bank_root)
    ]
    items.sort(key=_sort_key)

    return {"ok": True, "assignments": items}


@router.post("/teacher/assignments/new")
def teacher_create_assignment(
    req: NewAssignmentReq,
    _session: dict = Depends(require_teacher),
):
    """
    Allocate a folder and store the metadata, before any file is uploaded.

    The teacher screen calls this first, then posts the file to
    POST /routes/bank/upload using the exam_id returned here.
    """
    bank_root = _bind_teacher_bank(_session)

    title = _clean_title(req.title)
    if not title:
        raise HTTPException(
            status_code=400,
            detail="An assignment needs a name.",
        )

    exam_id = _new_exam_id(bank_root, title)
    now = _now()

    record = {
        "schema_version": SCHEMA_VERSION,
        "exam_id": exam_id,
        "title": title,
        "due_at": _clean_due_at(req.due_at),
        "published": bool(req.published),
        "max_attempts": _clean_max_attempts(req.max_attempts),
        "created_at": now,
        "updated_at": now,
    }
    write_assignment(exam_id, record)

    return {"ok": True, "assignment": _assignment_payload(exam_id, for_teacher=True)}


@router.post("/teacher/assignments/update")
def teacher_update_assignment(
    req: UpdateAssignmentReq,
    _session: dict = Depends(require_teacher),
):
    """Edit the name, the deadline, the attempt limit, or the publish flag."""
    _bind_teacher_bank(_session)

    exam_id = require_safe_exam_id(req.exam_id)

    if not exam_dir(exam_id).exists():
        raise HTTPException(status_code=404, detail="Unknown assignment.")

    title = _clean_title(req.title)
    if not title:
        raise HTTPException(
            status_code=400,
            detail="An assignment needs a name.",
        )

    existing = read_assignment(exam_id) or {}

    record = {
        "schema_version": SCHEMA_VERSION,
        "exam_id": exam_id,
        "title": title,
        "due_at": _clean_due_at(req.due_at),
        "published": bool(req.published),
        "max_attempts": _clean_max_attempts(req.max_attempts),
        "created_at": str(existing.get("created_at") or _now()),
        "updated_at": _now(),
    }
    write_assignment(exam_id, record)

    return {"ok": True, "assignment": _assignment_payload(exam_id, for_teacher=True)}
