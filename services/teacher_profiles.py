from __future__ import annotations

import json
import re
import secrets
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import HTTPException

from core.config import DEFAULT_SUBJECT, SUBJECT_LABELS, SUBJECT_OPTIONS, TEACHER_DATA_ROOT
from core.security import hash_password, verify_password_hash

_DATA_LOCK = Lock()
TEACHERS_PATH = TEACHER_DATA_ROOT / "teachers.json"
VOUCHERS_PATH = TEACHER_DATA_ROOT / "vouchers.json"

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

DEFAULT_SUBJECT_PROMPTS: dict[str, str] = {
    "math": "Grade mathematical reasoning, notation, algebra/calculus steps, final answer, and justification. Do not give credit for an unsupported final answer when the method is required.",
    "physics": "Grade physical reasoning, formula selection, unit consistency, substitution, signs/directions, significant figures, and interpretation of the result.",
    "chemistry": "Grade chemical notation, balanced equations, units, stoichiometry, charges, states/phases when relevant, and conceptual explanation.",
    "biology": "Grade biological terminology, process accuracy, evidence, diagrams/labels when relevant, and whether the explanation answers the requested mechanism or comparison.",
    "cs": "Grade algorithmic reasoning, code correctness, edge cases, complexity, outputs, syntax/indentation when relevant, and whether the answer matches the required programming task.",
    "engineering": "Grade modeling assumptions, equations, units, constraints, calculations, safety factors, diagrams, and interpretation of engineering results.",
    "general_stem": "Grade the STEM reasoning, correct use of domain concepts, units/notation where relevant, evidence, and clarity of explanation.",
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_subject(subject: str | None) -> str:
    s = (subject or DEFAULT_SUBJECT or "math").strip().lower()
    aliases = {
        "computer_science": "cs",
        "computerscience": "cs",
        "coding": "cs",
        "programming": "cs",
        "stem": "general_stem",
        "general": "general_stem",
    }
    s = aliases.get(s, s)
    return s if s in SUBJECT_OPTIONS else (DEFAULT_SUBJECT if DEFAULT_SUBJECT in SUBJECT_OPTIONS else "math")


def subject_options() -> list[dict[str, str]]:
    return [{"value": s, "label": SUBJECT_LABELS.get(s, s)} for s in SUBJECT_OPTIONS]


def default_subject_prompt(subject: str | None) -> str:
    return DEFAULT_SUBJECT_PROMPTS.get(normalize_subject(subject), DEFAULT_SUBJECT_PROMPTS["math"])


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


def _load_teachers() -> dict[str, Any]:
    data = _read_json(TEACHERS_PATH, {"schema_version": "teacher_profiles_v1", "teachers": {}})
    if not isinstance(data, dict):
        return {"schema_version": "teacher_profiles_v1", "teachers": {}}
    teachers = data.get("teachers")
    if not isinstance(teachers, dict):
        data["teachers"] = {}
    return data


def _save_teachers(data: dict[str, Any]) -> None:
    _write_json(TEACHERS_PATH, data)


def _load_vouchers() -> dict[str, Any]:
    data = _read_json(VOUCHERS_PATH, {"schema_version": "teacher_vouchers_v1", "vouchers": {}})
    if not isinstance(data, dict):
        return {"schema_version": "teacher_vouchers_v1", "vouchers": {}}
    vouchers = data.get("vouchers")
    if not isinstance(vouchers, dict):
        data["vouchers"] = {}
    return data


def _save_vouchers(data: dict[str, Any]) -> None:
    _write_json(VOUCHERS_PATH, data)


def _public_teacher(profile: dict[str, Any]) -> dict[str, Any]:
    out = dict(profile)
    out.pop("password_hash", None)
    return out


def _voucher_hash(code: str) -> str:
    import hashlib

    return hashlib.sha256((code or "").strip().encode("utf-8")).hexdigest()


def _slug_seed(value: str) -> str:
    value = (value or "teacher").strip().lower()
    if "@" in value:
        value = value.split("@", 1)[0]
    value = _SAFE_ID_RE.sub("_", value).strip("_")
    return value[:36] or "teacher"


def _new_teacher_id(email: str, name: str, existing: dict[str, Any]) -> str:
    base = _slug_seed(email or name)
    candidate = base
    while candidate in existing:
        candidate = f"{base}_{secrets.token_hex(3)}"
    return candidate


def _validate_password(password: str, password_confirm: str | None = None) -> None:
    password = password or ""
    if password_confirm is not None and password != password_confirm:
        raise HTTPException(status_code=400, detail="Passwords do not match.")
    if len(password) < 10:
        raise HTTPException(status_code=400, detail="Password must be at least 10 characters.")
    if password.lower() == password or password.upper() == password:
        raise HTTPException(status_code=400, detail="Password must include both uppercase and lowercase letters.")
    if not any(ch.isdigit() for ch in password):
        raise HTTPException(status_code=400, detail="Password must include at least one digit.")


def create_voucher(*, created_by: str, subject: str = "math", note: str = "") -> dict[str, Any]:
    subject = normalize_subject(subject)
    raw_code = f"STEM-{secrets.token_urlsafe(16).replace('-', '').replace('_', '')[:18].upper()}"
    digest = _voucher_hash(raw_code)

    with _DATA_LOCK:
        data = _load_vouchers()
        voucher_id = secrets.token_hex(8)

        data["vouchers"][digest] = {
            "voucher_id": voucher_id,
            "voucher_hash": digest,
            "voucher_hash_short": digest[:12] + "…",
            "voucher_code": raw_code,
            "status": "unused",
            "subject": subject,
            "subject_label": SUBJECT_LABELS.get(subject, subject),
            "note": (note or "").strip(),
            "created_at": _now(),
            "created_by": created_by or "admin",
            "used_at": "",
            "used_by_teacher_id": "",
            "used_by_email": "",
        }
        _save_vouchers(data)

    return {
        "voucher_id": voucher_id,
        "voucher_code": raw_code,
        "voucher_hash": digest,
        "voucher_hash_short": digest[:12] + "…",
        "status": "unused",
        "subject": subject,
        "subject_label": SUBJECT_LABELS.get(subject, subject),
        "note": (note or "").strip(),
    }


def list_vouchers() -> list[dict[str, Any]]:
    with _DATA_LOCK:
        data = _load_vouchers()
        vouchers = []
        for digest, v in data.get("vouchers", {}).items():
            if not isinstance(v, dict):
                continue
            item = dict(v)
            item.setdefault("voucher_id", digest[:12])
            item.setdefault("voucher_hash", digest)
            item["voucher_hash_short"] = item.get("voucher_hash_short") or (digest[:12] + "…")
            item.setdefault("voucher_code", "")
            item["subject_label"] = SUBJECT_LABELS.get(item.get("subject"), item.get("subject", ""))
            item["subject_label"] = SUBJECT_LABELS.get(item.get("subject"), item.get("subject", ""))
            vouchers.append(item)
    vouchers.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return vouchers


def list_teachers() -> list[dict[str, Any]]:
    with _DATA_LOCK:
        data = _load_teachers()
        teachers = [_public_teacher(p) for p in data.get("teachers", {}).values() if isinstance(p, dict)]
    teachers.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return teachers


def get_teacher(teacher_id: str | None) -> dict[str, Any] | None:
    teacher_id = (teacher_id or "").strip()
    if not teacher_id:
        return None
    with _DATA_LOCK:
        data = _load_teachers()
        profile = data.get("teachers", {}).get(teacher_id)
        return _public_teacher(profile) if isinstance(profile, dict) else None


def register_teacher_from_voucher(
    *,
    voucher_code: str,
    teacher_name: str,
    teacher_email: str,
    subject: str,
    password: str,
    password_confirm: str,
    grading_prompt_extra: str = "",
) -> dict[str, Any]:
    voucher_code = (voucher_code or "").strip()
    teacher_name = (teacher_name or "").strip()
    teacher_email = (teacher_email or "").strip().lower()

    if not voucher_code:
        raise HTTPException(status_code=400, detail="Missing voucher code.")
    if not teacher_name:
        raise HTTPException(status_code=400, detail="Missing teacher name.")
    if not _EMAIL_RE.match(teacher_email):
        raise HTTPException(status_code=400, detail="Enter a valid teacher email.")
    _validate_password(password, password_confirm)

    digest = _voucher_hash(voucher_code)

    with _DATA_LOCK:
        vouchers = _load_vouchers()
        voucher = vouchers.get("vouchers", {}).get(digest)
        if not isinstance(voucher, dict):
            raise HTTPException(status_code=401, detail="Invalid voucher code.")
        if voucher.get("status") != "unused":
            raise HTTPException(status_code=409, detail="This voucher was already used.")

        teachers = _load_teachers()
        for existing in teachers.get("teachers", {}).values():
            if isinstance(existing, dict) and existing.get("email") == teacher_email:
                raise HTTPException(status_code=409, detail="A teacher profile with this email already exists.")

        selected_subject = normalize_subject(subject or voucher.get("subject"))
        teacher_id = _new_teacher_id(teacher_email, teacher_name, teachers["teachers"])
        prompt_extra = (grading_prompt_extra or "").strip() or default_subject_prompt(selected_subject)

        profile = {
            "teacher_id": teacher_id,
            "name": teacher_name,
            "email": teacher_email,
            "subject": selected_subject,
            "subject_label": SUBJECT_LABELS.get(selected_subject, selected_subject),
            "voucher_id": voucher.get("voucher_id", ""),
            "voucher_hash": digest,
            "voucher_code": voucher.get("voucher_code", ""),
            "grading_prompt_extra": prompt_extra,
            "password_hash": hash_password(password),
            "created_at": _now(),
            "updated_at": _now(),
            "status": "active",
        }
        teachers["teachers"][teacher_id] = profile

        voucher["status"] = "used"
        voucher["used_at"] = _now()
        voucher["used_by_teacher_id"] = teacher_id
        voucher["used_by_email"] = teacher_email

        _save_teachers(teachers)
        _save_vouchers(vouchers)

    return _public_teacher(profile)


def authenticate_teacher(identifier: str, password: str) -> dict[str, Any] | None:
    ident = (identifier or "").strip().lower()
    if not ident or not password:
        return None

    with _DATA_LOCK:
        data = _load_teachers()
        candidates = []
        direct = data.get("teachers", {}).get(ident)
        if isinstance(direct, dict):
            candidates.append(direct)
        for profile in data.get("teachers", {}).values():
            if isinstance(profile, dict) and profile.get("email", "").lower() == ident:
                candidates.append(profile)

        for profile in candidates:
            if profile.get("status") != "active":
                continue
            if verify_password_hash(password, profile.get("password_hash")):
                return _public_teacher(profile)
    return None


def update_teacher_profile(
    *,
    teacher_id: str,
    subject: str | None = None,
    grading_prompt_extra: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    teacher_id = (teacher_id or "").strip()
    with _DATA_LOCK:
        data = _load_teachers()
        profile = data.get("teachers", {}).get(teacher_id)
        if not isinstance(profile, dict):
            raise HTTPException(status_code=404, detail="Teacher profile not found.")
        if subject is not None:
            profile["subject"] = normalize_subject(subject)
            profile["subject_label"] = SUBJECT_LABELS.get(profile["subject"], profile["subject"])
        if grading_prompt_extra is not None:
            profile["grading_prompt_extra"] = (grading_prompt_extra or "").strip()
        if name is not None and name.strip():
            profile["name"] = name.strip()
        profile["updated_at"] = _now()
        _save_teachers(data)
        return _public_teacher(profile)


def change_teacher_password(*, teacher_id: str, old_password: str, new_password: str, new_password_confirm: str) -> None:
    _validate_password(new_password, new_password_confirm)
    teacher_id = (teacher_id or "").strip()
    with _DATA_LOCK:
        data = _load_teachers()
        profile = data.get("teachers", {}).get(teacher_id)
        if not isinstance(profile, dict):
            raise HTTPException(status_code=404, detail="Teacher profile not found.")
        if not verify_password_hash(old_password, profile.get("password_hash")):
            raise HTTPException(status_code=401, detail="Current password is incorrect.")
        profile["password_hash"] = hash_password(new_password)
        profile["updated_at"] = _now()
        _save_teachers(data)
