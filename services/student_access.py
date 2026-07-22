from __future__ import annotations

import hashlib
import json
import re
import secrets
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import HTTPException

from core.config import TEACHER_DATA_ROOT
from services.teacher_profiles import get_teacher

_CODES_PATH = TEACHER_DATA_ROOT / "student_codes.json"
_LOCK = Lock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_student_code(code: str | None) -> str:
    return (code or "").strip().upper().replace(" ", "")


def _hash_code(code: str) -> str:
    return hashlib.sha256(normalize_student_code(code).encode("utf-8")).hexdigest()


def _read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load() -> dict[str, Any]:
    data = _read_json(_CODES_PATH, {"schema_version": "student_codes_v1", "codes": {}})
    if not isinstance(data, dict):
        return {"schema_version": "student_codes_v1", "codes": {}}
    if not isinstance(data.get("codes"), dict):
        data["codes"] = {}
    return data


def _save(data: dict[str, Any]) -> None:
    _write_json(_CODES_PATH, data)


def _new_code(existing_hashes: set[str]) -> str:
    for _ in range(100):
        code = "STU-" + secrets.token_hex(3).upper() + "-" + secrets.token_hex(3).upper()
        if _hash_code(code) not in existing_hashes:
            return code
    raise RuntimeError("Could not generate a unique student code.")


def list_student_codes_for_teacher(
    teacher_id: str,
    *,
    voucher_id: str = "",
) -> list[dict[str, Any]]:
    teacher_id = (teacher_id or "").strip()
    voucher_id = (voucher_id or "").strip()

    if not teacher_id:
        return []

    with _LOCK:
        data = _load()
        out = []
        for rec in data.get("codes", {}).values():
            if not isinstance(rec, dict):
                continue
            if rec.get("teacher_id") != teacher_id:
                continue

            if voucher_id:
                rec_voucher = str(
                    rec.get("voucher_id")
                    or rec.get("voucher_code")
                    or rec.get("voucher_hash")
                    or ""
                ).strip()
                if rec_voucher != voucher_id:
                    continue

            item = dict(rec)
            item.pop("code_hash", None)
            out.append(item)

    out.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return out


def active_student_code_set_for_teacher(teacher_id: str) -> set[str]:
    return {
        str(x.get("code") or "").strip()
        for x in list_student_codes_for_teacher(teacher_id)
        if x.get("status") == "active" and x.get("code")
    }


def create_student_codes(
    *,
    teacher_id: str,
    count: int = 1,
    course_label: str = "",
    note: str = "",
    student_name: str = "",
    student_email: str = "",
) -> list[dict[str, Any]]:
    teacher_id = (teacher_id or "").strip()
    if not teacher_id:
        raise HTTPException(status_code=400, detail="Missing teacher profile.")

    teacher = get_teacher(teacher_id)
    if not teacher:
        raise HTTPException(status_code=404, detail="Teacher profile not found.")

    count = max(1, min(int(count or 1), 200))
    course_label = (course_label or "").strip() or str(teacher.get("course_label") or "").strip()
    active_course = teacher.get("active_course") if isinstance(teacher.get("active_course"), dict) else {}
    note = (note or "").strip()
    student_name = (student_name or "").strip()
    student_email = (student_email or "").strip().lower()

    with _LOCK:
        data = _load()
        existing_hashes = set(data.get("codes", {}).keys())
        created: list[dict[str, Any]] = []

        for _ in range(count):
            code = _new_code(existing_hashes)
            digest = _hash_code(code)
            existing_hashes.add(digest)

            rec = {
                "code": code,
                "code_hash": digest,
                "teacher_id": teacher_id,
                "teacher_name": teacher.get("name", ""),
                "teacher_email": teacher.get("email", ""),
                "voucher_id": active_course.get("voucher_id") or teacher.get("voucher_id", ""),
                "voucher_hash": active_course.get("voucher_hash") or teacher.get("voucher_hash", ""),
                "voucher_code": active_course.get("voucher_code") or teacher.get("voucher_code", ""),
                "course_id": active_course.get("course_id") or teacher.get("active_course_id", ""),
                "subject": active_course.get("subject") or teacher.get("subject", "math"),
                "subject_label": active_course.get("subject_label") or teacher.get("subject_label", teacher.get("subject", "math")),
                "course_label": course_label,
                "student_name": student_name,
                "student_email": student_email,
                "note": note,
                "status": "active",
                "created_at": _now(),
                "last_used_at": "",
                "uses": 0,
            }
            data["codes"][digest] = rec

            public = dict(rec)
            public.pop("code_hash", None)
            created.append(public)

        _save(data)

    return created


# A teacher-supplied login code must be a plausible credential: 4–64 chars,
# letters/digits plus dash or underscore. Codes are normalized to uppercase
# (spaces removed) before this check, matching how they are stored and how
# students type them at login.
_CODE_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9_-]{3,63}$")


def _validate_new_student_code(normalized: str) -> str:
    """Return an error message for an invalid new code, or "" if it is valid."""
    if len(normalized) < 4:
        return "code must be at least 4 characters."
    if len(normalized) > 64:
        return "code must be at most 64 characters."
    if not _CODE_PATTERN.match(normalized):
        return "code may only contain letters, digits, '-' and '_'."
    return ""


def update_student_codes_for_teacher(
    teacher_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Apply an Excel import of student codes for this teacher.

    Each row is matched by its access code:

    * an existing code owned by this teacher has its editable fields updated
      (blank cells leave the previous value unchanged); the code itself is
      never rewritten, since its hash is the storage key;
    * a code that does not exist yet creates a new student for this teacher,
      using the code exactly as typed (validated, and rejected if it collides
      with any existing code);
    * a code owned by another teacher, a blank code, or an invalid code is
      skipped with a warning.
    """
    teacher_id = (teacher_id or "").strip()
    if not teacher_id:
        raise HTTPException(status_code=400, detail="Missing teacher profile.")

    teacher = get_teacher(teacher_id) or {}
    active_course = teacher.get("active_course") if isinstance(teacher.get("active_course"), dict) else {}
    default_course_label = str(teacher.get("course_label") or "").strip()

    allowed_statuses = {"active", "disabled", "inactive"}
    editable_fields = {"student_name", "student_email", "course_label", "note"}

    updated = 0
    created = 0
    skipped = 0
    errors: list[str] = []
    created_codes: list[dict[str, Any]] = []

    def _row_status(row: dict[str, Any], row_index: int) -> tuple[str, str]:
        """Return (status, error). status is "" when not supplied."""
        if "status" not in row:
            return "", ""
        status = str(row.get("status") or "").strip().lower()
        if status and status not in allowed_statuses:
            return "", f"Row {row_index}: status must be active, disabled, or inactive."
        return status, ""

    with _LOCK:
        data = _load()
        codes = data.get("codes", {})

        for row_index, row in enumerate(rows, start=2):
            if not isinstance(row, dict):
                skipped += 1
                continue

            raw_code = str(row.get("code") or "").strip()
            if not raw_code:
                skipped += 1
                continue

            normalized = normalize_student_code(raw_code)
            digest = _hash_code(normalized)
            rec = codes.get(digest)

            # --- Existing code: update editable fields ---
            if isinstance(rec, dict):
                if rec.get("teacher_id") != teacher_id:
                    skipped += 1
                    errors.append(f"Row {row_index}: code already in use.")
                    continue

                changed = False
                for field in editable_fields:
                    if field not in row:
                        continue
                    value = str(row.get(field) or "").strip()
                    if field == "student_email":
                        value = value.lower()
                    if value != "" and value != str(rec.get(field) or ""):
                        rec[field] = value
                        changed = True

                status, status_err = _row_status(row, row_index)
                if status_err:
                    skipped += 1
                    errors.append(status_err)
                    continue
                if status and status != str(rec.get("status") or ""):
                    rec["status"] = status
                    changed = True

                if changed:
                    rec["updated_at"] = _now()
                    updated += 1
                continue

            # --- New code: create a student using the typed code ---
            code_err = _validate_new_student_code(normalized)
            if code_err:
                skipped += 1
                errors.append(f"Row {row_index}: {code_err}")
                continue

            status, status_err = _row_status(row, row_index)
            if status_err:
                skipped += 1
                errors.append(status_err)
                continue

            course_label = str(row.get("course_label") or "").strip() or default_course_label

            new_rec = {
                "code": normalized,
                "code_hash": digest,
                "teacher_id": teacher_id,
                "teacher_name": teacher.get("name", ""),
                "teacher_email": teacher.get("email", ""),
                "voucher_id": active_course.get("voucher_id") or teacher.get("voucher_id", ""),
                "voucher_hash": active_course.get("voucher_hash") or teacher.get("voucher_hash", ""),
                "voucher_code": active_course.get("voucher_code") or teacher.get("voucher_code", ""),
                "course_id": active_course.get("course_id") or teacher.get("active_course_id", ""),
                "subject": active_course.get("subject") or teacher.get("subject", "math"),
                "subject_label": active_course.get("subject_label") or teacher.get("subject_label", teacher.get("subject", "math")),
                "course_label": course_label,
                "student_name": str(row.get("student_name") or "").strip(),
                "student_email": str(row.get("student_email") or "").strip().lower(),
                "note": str(row.get("note") or "").strip(),
                "status": status or "active",
                "created_at": _now(),
                "last_used_at": "",
                "uses": 0,
            }
            codes[digest] = new_rec
            created += 1

            public = dict(new_rec)
            public.pop("code_hash", None)
            created_codes.append(public)

        _save(data)

    return {
        "updated": updated,
        "created": created,
        "skipped": skipped,
        "errors": errors[:20],
        "created_codes": created_codes,
    }




def get_student_code_record(code: str) -> dict[str, Any] | None:
    """
    Read a student-code record without incrementing usage counters.

    Used by the student dashboard after login, so refreshing the page does not
    count as a new student-code use.
    """
    normalized = normalize_student_code(code)
    if not normalized:
        return None

    digest = _hash_code(normalized)

    with _LOCK:
        data = _load()
        rec = data.get("codes", {}).get(digest)

        if not isinstance(rec, dict):
            return None

        public = dict(rec)
        public.pop("code_hash", None)
        return public


def authenticate_student_code(code: str) -> dict[str, Any] | None:
    normalized = normalize_student_code(code)
    if not normalized:
        return None

    digest = _hash_code(normalized)

    with _LOCK:
        data = _load()
        rec = data.get("codes", {}).get(digest)

        if not isinstance(rec, dict):
            return None
        if rec.get("status") != "active":
            return None

        rec["last_used_at"] = _now()
        rec["uses"] = int(rec.get("uses") or 0) + 1
        _save(data)

        public = dict(rec)
        public.pop("code_hash", None)
        return public