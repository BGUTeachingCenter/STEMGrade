from __future__ import annotations

import hashlib
import json
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


def list_student_codes_for_teacher(teacher_id: str) -> list[dict[str, Any]]:
    teacher_id = (teacher_id or "").strip()
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
) -> list[dict[str, Any]]:
    teacher_id = (teacher_id or "").strip()
    if not teacher_id:
        raise HTTPException(status_code=400, detail="Missing teacher profile.")

    teacher = get_teacher(teacher_id)
    if not teacher:
        raise HTTPException(status_code=404, detail="Teacher profile not found.")

    count = max(1, min(int(count or 1), 200))
    course_label = (course_label or "").strip()
    note = (note or "").strip()

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
                "subject": teacher.get("subject", "math"),
                "subject_label": teacher.get("subject_label", teacher.get("subject", "math")),
                "course_label": course_label,
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