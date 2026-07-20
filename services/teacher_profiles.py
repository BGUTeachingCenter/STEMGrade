from __future__ import annotations

import json
import os
import re
import secrets
import stat
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

# Kept temporarily so the old development registration flow continues to run
# until it is removed in Phase 2.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Usernames may contain Unicode letters/numbers, spaces, dots, underscores,
# and hyphens. The internal teacher_id is generated separately and is always
# safe to use as a folder name.
_USERNAME_EXTRA_CHARACTERS = {" ", ".", "_", "-"}
_USERNAME_MIN_LENGTH = 3
_USERNAME_MAX_LENGTH = 40

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


# Owner read/write only. These files hold usernames and salted password hashes,
# so keep them unreadable to other OS users on the host.
_SECRET_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR  # 0o600
_SECRET_DIR_MODE = stat.S_IRWXU  # 0o700


def _restrict(path: Path, mode: int) -> None:
    """Best-effort permission tightening; no-op where the OS doesn't support it (e.g. Windows)."""
    try:
        os.chmod(path, mode)
    except (OSError, NotImplementedError):
        pass


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _restrict(path.parent, _SECRET_DIR_MODE)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    # Tighten before the rename so the file is never briefly world-readable.
    _restrict(tmp, _SECRET_FILE_MODE)
    tmp.replace(path)
    _restrict(path, _SECRET_FILE_MODE)


def _load_teachers() -> dict[str, Any]:
    fallback = {
        "schema_version": "teacher_profiles_v2",
        "teachers": {},
    }

    data = _read_json(TEACHERS_PATH, fallback)

    if not isinstance(data, dict):
        return dict(fallback)

    teachers = data.get("teachers")
    if not isinstance(teachers, dict):
        data["teachers"] = {}

    data["schema_version"] = "teacher_profiles_v2"
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


def _course_id_from_voucher(voucher: dict[str, Any], digest: str) -> str:
    return str(
        voucher.get("voucher_id")
        or voucher.get("voucher_code")
        or digest[:12]
        or "course_default"
    ).strip() or "course_default"


def _course_from_voucher(
    *,
    voucher: dict[str, Any],
    digest: str,
    subject: str,
    course_label: str,
    grading_prompt_extra: str,
) -> dict[str, Any]:
    selected_subject = normalize_subject(subject or voucher.get("subject"))
    now = _now()
    return {
        "course_id": _course_id_from_voucher(voucher, digest),
        "voucher_id": voucher.get("voucher_id", ""),
        "voucher_hash": digest,
        "voucher_code": voucher.get("voucher_code", ""),
        "course_label": (course_label or "").strip() or (voucher.get("note") or "").strip() or "Course",
        "subject": selected_subject,
        "subject_label": SUBJECT_LABELS.get(selected_subject, selected_subject),
        "grading_prompt_extra": (grading_prompt_extra or "").strip() or default_subject_prompt(selected_subject),
        "created_at": now,
        "updated_at": now,
        "status": "active",
    }


def _courses_list(profile: dict[str, Any]) -> list[dict[str, Any]]:
    courses = profile.get("courses")
    if isinstance(courses, list):
        return [c for c in courses if isinstance(c, dict)]

    # Backward compatibility for teacher profiles created before courses existed.
    legacy_course_id = str(
        profile.get("voucher_id")
        or profile.get("voucher_code")
        or profile.get("voucher_hash")
        or "course_default"
    ).strip() or "course_default"

    legacy_course = {
        "course_id": legacy_course_id,
        "voucher_id": profile.get("voucher_id", ""),
        "voucher_hash": profile.get("voucher_hash", ""),
        "voucher_code": profile.get("voucher_code", ""),
        "course_label": profile.get("course_label", "") or "Course",
        "subject": profile.get("subject", "math"),
        "subject_label": profile.get("subject_label", profile.get("subject", "math")),
        "grading_prompt_extra": profile.get("grading_prompt_extra", ""),
        "created_at": profile.get("created_at", ""),
        "updated_at": profile.get("updated_at", ""),
        "status": "active",
    }
    return [legacy_course]


def _active_course(profile: dict[str, Any]) -> dict[str, Any]:
    courses = _courses_list(profile)
    active_id = str(profile.get("active_course_id") or "").strip()

    if active_id:
        for course in courses:
            if str(course.get("course_id") or "") == active_id:
                return course

    return courses[0] if courses else {}


def _sync_active_course_fields(profile: dict[str, Any]) -> None:
    """
    Keep old code working by mirroring the active course onto the old top-level
    fields: subject, voucher_id, course_label, grading_prompt_extra, etc.
    """
    courses = _courses_list(profile)
    profile["courses"] = courses

    if not courses:
        return

    active = _active_course(profile)
    if not active:
        active = courses[0]

    profile["active_course_id"] = active.get("course_id", "")
    profile["subject"] = active.get("subject", profile.get("subject", "math"))
    profile["subject_label"] = active.get("subject_label", profile.get("subject_label", profile.get("subject", "math")))
    profile["course_label"] = active.get("course_label", profile.get("course_label", ""))
    profile["voucher_id"] = active.get("voucher_id", profile.get("voucher_id", ""))
    profile["voucher_hash"] = active.get("voucher_hash", profile.get("voucher_hash", ""))
    profile["voucher_code"] = active.get("voucher_code", profile.get("voucher_code", ""))
    profile["grading_prompt_extra"] = active.get("grading_prompt_extra", profile.get("grading_prompt_extra", ""))


def _public_teacher(profile: dict[str, Any]) -> dict[str, Any]:
    out = dict(profile)
    _sync_active_course_fields(out)
    out["active_course"] = _active_course(out)
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


def normalize_username(username: str | None) -> str:
    """
    Canonical username used for uniqueness and authentication.

    Whitespace is collapsed and casefold() is used instead of lower() so
    Unicode usernames are compared consistently.
    """
    display = " ".join((username or "").strip().split())
    return display.casefold()


def _validate_username(username: str | None) -> tuple[str, str]:
    """
    Return (display_username, normalized_username) after validation.
    """
    display = " ".join((username or "").strip().split())

    if not display:
        raise HTTPException(status_code=400, detail="Enter a username.")

    if len(display) < _USERNAME_MIN_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Username must contain at least {_USERNAME_MIN_LENGTH} characters.",
        )

    if len(display) > _USERNAME_MAX_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Username may contain at most {_USERNAME_MAX_LENGTH} characters.",
        )

    if not any(ch.isalnum() for ch in display):
        raise HTTPException(
            status_code=400,
            detail="Username must contain at least one letter or number.",
        )

    invalid = [
        ch
        for ch in display
        if not ch.isalnum() and ch not in _USERNAME_EXTRA_CHARACTERS
    ]

    if invalid:
        raise HTTPException(
            status_code=400,
            detail=(
                "Username may contain letters, numbers, spaces, dots, "
                "underscores, and hyphens."
            ),
        )

    normalized = normalize_username(display)
    return display, normalized


def _new_profile_teacher_id(existing: dict[str, Any]) -> str:
    """
    Generate an internal ID independent of the public username.

    This prevents usernames from becoming folder names and allows Unicode
    usernames without affecting filesystem paths.
    """
    for _ in range(100):
        candidate = f"teacher_{secrets.token_hex(8)}"
        if candidate not in existing:
            return candidate

    raise RuntimeError("Could not generate a unique teacher ID.")

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
            "used_by_username": "",

            # Temporary compatibility field. Removed after the old
            # voucher-first teacher registration route is deleted.
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


def create_teacher_account(
    *,
    username: str,
    password: str,
    password_confirm: str,
) -> dict[str, Any]:
    """
    Create a teacher account without creating a course or using a voucher.

    Courses remain empty until the authenticated teacher redeems a voucher.
    """
    display_username, normalized_username = _validate_username(username)
    _validate_password(password, password_confirm)

    with _DATA_LOCK:
        data = _load_teachers()
        teachers = data.setdefault("teachers", {})

        for existing in teachers.values():
            if not isinstance(existing, dict):
                continue

            existing_normalized = str(
                existing.get("username_normalized")
                or normalize_username(existing.get("username"))
                or ""
            )

            if existing_normalized == normalized_username:
                raise HTTPException(
                    status_code=409,
                    detail="This username is already in use.",
                )

        teacher_id = _new_profile_teacher_id(teachers)
        now = _now()

        profile = {
            "teacher_id": teacher_id,
            "username": display_username,
            "username_normalized": normalized_username,

            # Existing course/student code displays currently read "name".
            # Keep it mirrored until those call sites are cleaned in Phase 4.
            "name": display_username,

            "password_hash": hash_password(password),
            "courses": [],
            "active_course_id": "",
            "created_at": now,
            "updated_at": now,
            "status": "active",
        }

        teachers[teacher_id] = profile
        _save_teachers(data)

    return _public_teacher(profile)


def redeem_voucher_for_course(
    *,
    teacher_id: str,
    voucher_code: str,
    course_label: str = "",
    subject: str = "",
    grading_prompt_extra: str = "",
) -> dict[str, Any]:
    """
    Redeem one unused voucher and attach the resulting course to an existing
    authenticated teacher account.
    """
    teacher_id = (teacher_id or "").strip()
    voucher_code = (voucher_code or "").strip()
    course_label = (course_label or "").strip()

    if not teacher_id:
        raise HTTPException(status_code=400, detail="Missing teacher profile.")

    if not voucher_code:
        raise HTTPException(status_code=400, detail="Enter a voucher code.")

    digest = _voucher_hash(voucher_code)

    with _DATA_LOCK:
        teachers = _load_teachers()
        profile = teachers.get("teachers", {}).get(teacher_id)

        if not isinstance(profile, dict):
            raise HTTPException(
                status_code=404,
                detail="Teacher profile not found.",
            )

        if profile.get("status") != "active":
            raise HTTPException(
                status_code=403,
                detail="This teacher profile is not active.",
            )

        vouchers = _load_vouchers()
        voucher = vouchers.get("vouchers", {}).get(digest)

        if not isinstance(voucher, dict):
            raise HTTPException(
                status_code=401,
                detail="Invalid voucher code.",
            )

        if voucher.get("status") != "unused":
            raise HTTPException(
                status_code=409,
                detail="This voucher has already been used.",
            )

        courses = _courses_list(profile)

        if any(
            str(course.get("voucher_hash") or "") == digest
            for course in courses
        ):
            raise HTTPException(
                status_code=409,
                detail="This voucher is already attached to this profile.",
            )

        selected_subject = normalize_subject(
            subject or voucher.get("subject")
        )

        new_course = _course_from_voucher(
            voucher=voucher,
            digest=digest,
            subject=selected_subject,
            course_label=course_label,
            grading_prompt_extra=grading_prompt_extra,
        )

        courses.append(new_course)

        profile["courses"] = courses
        profile["active_course_id"] = new_course["course_id"]
        profile["updated_at"] = _now()
        _sync_active_course_fields(profile)

        voucher["status"] = "used"
        voucher["used_at"] = _now()
        voucher["used_by_teacher_id"] = teacher_id
        voucher["used_by_username"] = profile.get("username", "")
        voucher["used_by_email"] = ""

        _save_teachers(teachers)
        _save_vouchers(vouchers)

    return _public_teacher(profile)


def register_teacher_from_voucher(
    *,
    voucher_code: str,
    teacher_name: str,
    teacher_email: str,
    subject: str,
    password: str,
    password_confirm: str,
    grading_prompt_extra: str = "",
    course_label: str = "",
) -> dict[str, Any]:
    voucher_code = (voucher_code or "").strip()
    teacher_name = (teacher_name or "").strip()
    teacher_email = (teacher_email or "").strip().lower()
    course_label = (course_label or "").strip()

    if not voucher_code:
        raise HTTPException(status_code=400, detail="Missing voucher code.")
    if not teacher_name:
        raise HTTPException(status_code=400, detail="Missing teacher name.")
    if not _EMAIL_RE.match(teacher_email):
        raise HTTPException(status_code=400, detail="Enter a valid teacher email.")

    # For a new teacher this creates the password.
    # For an existing teacher this must match the existing password,
    # because redeeming another voucher now means "add course".
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
        existing_teacher_id = ""
        existing_profile: dict[str, Any] | None = None

        for tid, existing in teachers.get("teachers", {}).items():
            if isinstance(existing, dict) and existing.get("email") == teacher_email:
                existing_teacher_id = tid
                existing_profile = existing
                break

        selected_subject = normalize_subject(subject or voucher.get("subject"))
        new_course = _course_from_voucher(
            voucher=voucher,
            digest=digest,
            subject=selected_subject,
            course_label=course_label,
            grading_prompt_extra=grading_prompt_extra,
        )

        if existing_profile is not None:
            if not verify_password_hash(password, existing_profile.get("password_hash")):
                raise HTTPException(
                    status_code=401,
                    detail="This email already has a teacher account. Enter the existing teacher password to add this course.",
                )

            courses = _courses_list(existing_profile)
            if any(str(c.get("voucher_hash") or "") == digest for c in courses):
                raise HTTPException(status_code=409, detail="This course voucher is already attached to this teacher.")

            courses.append(new_course)
            existing_profile["courses"] = courses
            existing_profile["active_course_id"] = new_course["course_id"]
            existing_profile["name"] = teacher_name or existing_profile.get("name", "")
            existing_profile["updated_at"] = _now()
            _sync_active_course_fields(existing_profile)
            teacher_id = existing_teacher_id
            profile = existing_profile

        else:
            teacher_id = _new_teacher_id(teacher_email, teacher_name, teachers["teachers"])
            profile = {
                "teacher_id": teacher_id,
                "name": teacher_name,
                "email": teacher_email,
                "password_hash": hash_password(password),
                "courses": [new_course],
                "active_course_id": new_course["course_id"],
                "created_at": _now(),
                "updated_at": _now(),
                "status": "active",
            }
            _sync_active_course_fields(profile)
            teachers["teachers"][teacher_id] = profile

        voucher["status"] = "used"
        voucher["used_at"] = _now()
        voucher["used_by_teacher_id"] = teacher_id
        voucher["used_by_email"] = teacher_email

        _save_teachers(teachers)
        _save_vouchers(vouchers)

    return _public_teacher(profile)


def authenticate_teacher(username: str, password: str) -> dict[str, Any] | None:
    normalized = normalize_username(username)

    if not normalized or not password:
        return None

    with _DATA_LOCK:
        data = _load_teachers()
        candidates: list[dict[str, Any]] = []
        seen_teacher_ids: set[str] = set()

        # Internal teacher IDs remain accepted temporarily for development and
        # troubleshooting, although the UI will expose username login only.
        direct = data.get("teachers", {}).get((username or "").strip())
        if isinstance(direct, dict):
            candidates.append(direct)
            seen_teacher_ids.add(str(direct.get("teacher_id") or ""))

        for profile in data.get("teachers", {}).values():
            if not isinstance(profile, dict):
                continue

            profile_username = str(
                profile.get("username_normalized")
                or normalize_username(profile.get("username"))
                or ""
            )

            # Temporary compatibility for profiles created by the old
            # email-based development workflow. Removed after the dev reset.
            legacy_email = str(profile.get("email") or "").strip().casefold()

            if profile_username != normalized and legacy_email != normalized:
                continue

            teacher_id = str(profile.get("teacher_id") or "")

            if teacher_id in seen_teacher_ids:
                continue

            candidates.append(profile)
            seen_teacher_ids.add(teacher_id)

        for profile in candidates:
            if profile.get("status") != "active":
                continue

            if verify_password_hash(password, profile.get("password_hash")):
                return _public_teacher(profile)

    return None


def update_teacher_profile(
    *,
    teacher_id: str,
    username: str | None = None,
    subject: str | None = None,
    grading_prompt_extra: str | None = None,
    name: str | None = None,
    course_label: str | None = None,
) -> dict[str, Any]:
    teacher_id = (teacher_id or "").strip()
    with _DATA_LOCK:
        data = _load_teachers()
        profile = data.get("teachers", {}).get(teacher_id)
        if not isinstance(profile, dict):
            raise HTTPException(status_code=404, detail="Teacher profile not found.")

        if username is not None:
            display_username, normalized_username = _validate_username(username)

            for other_teacher_id, other in data.get("teachers", {}).items():
                if other_teacher_id == teacher_id or not isinstance(other, dict):
                    continue

                other_normalized = str(
                    other.get("username_normalized")
                    or normalize_username(other.get("username"))
                    or ""
                )

                if other_normalized == normalized_username:
                    raise HTTPException(
                        status_code=409,
                        detail="This username is already in use.",
                    )

            profile["username"] = display_username
            profile["username_normalized"] = normalized_username

            # Temporary compatibility mirror.
            profile["name"] = display_username

        courses = _courses_list(profile)
        active_id = str(profile.get("active_course_id") or "").strip()
        if not active_id and courses:
            active_id = str(courses[0].get("course_id") or "")

        active_course = None
        for course in courses:
            if str(course.get("course_id") or "") == active_id:
                active_course = course
                break

        if active_course is None and courses:
            active_course = courses[0]

        if name is not None and name.strip():
            profile["name"] = name.strip()

        # These are course-level settings, not teacher-account settings.
        if active_course is not None:
            if subject is not None:
                active_course["subject"] = normalize_subject(subject)
                active_course["subject_label"] = SUBJECT_LABELS.get(active_course["subject"], active_course["subject"])
            if grading_prompt_extra is not None:
                active_course["grading_prompt_extra"] = (grading_prompt_extra or "").strip()
            if course_label is not None:
                active_course["course_label"] = (course_label or "").strip()
            active_course["updated_at"] = _now()

        profile["courses"] = courses
        profile["updated_at"] = _now()
        _sync_active_course_fields(profile)
        _save_teachers(data)
        return _public_teacher(profile)


def list_teacher_courses(teacher_id: str) -> list[dict[str, Any]]:
    teacher_id = (teacher_id or "").strip()
    if not teacher_id:
        return []

    with _DATA_LOCK:
        data = _load_teachers()
        profile = data.get("teachers", {}).get(teacher_id)
        if not isinstance(profile, dict):
            return []
        courses = _courses_list(profile)

    courses.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
    return courses


def set_active_teacher_course(*, teacher_id: str, course_id: str) -> dict[str, Any]:
    teacher_id = (teacher_id or "").strip()
    course_id = (course_id or "").strip()

    if not course_id:
        raise HTTPException(status_code=400, detail="Missing course_id.")

    with _DATA_LOCK:
        data = _load_teachers()
        profile = data.get("teachers", {}).get(teacher_id)
        if not isinstance(profile, dict):
            raise HTTPException(status_code=404, detail="Teacher profile not found.")

        courses = _courses_list(profile)
        if not any(str(c.get("course_id") or "") == course_id for c in courses):
            raise HTTPException(status_code=404, detail="Course not found for this teacher.")

        profile["courses"] = courses
        profile["active_course_id"] = course_id
        profile["updated_at"] = _now()
        _sync_active_course_fields(profile)
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
