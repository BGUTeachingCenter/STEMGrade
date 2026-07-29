from __future__ import annotations

import json
import re
import secrets
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote
from zoneinfo import (
    ZoneInfo,
    ZoneInfoNotFoundError,
)

from fastapi import HTTPException

from core.config import (
    APP_TIMEZONE,
    PROJECT_ROOT,
    STUDENT_OCR_DAILY_LIMIT,
    STUDENT_OCR_RESERVATION_TTL_SECONDS,
    TEACHERS_ROOT,
)

# Legacy layout, kept readable:
#   data/student_work/<teacher>/<voucher>/<student>/<work_id>/
LEGACY_STUDENT_WORK_ROOT = PROJECT_ROOT / "data" / "student_work"
LEGACY_STUDENT_WORK_ROOT.mkdir(parents=True, exist_ok=True)

# Keep the old constant name for compatibility inside this module.
STUDENT_WORK_ROOT = LEGACY_STUDENT_WORK_ROOT

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,180}$")

# Prevent simultaneous OCR requests in the same application process from
# updating one student's quota file at the same time.
_OCR_USAGE_LOCK = Lock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_name(value: str, fallback: str = "file") -> str:
    value = (value or "").strip()
    if not value:
        return fallback
    cleaned = _SAFE_NAME_RE.sub("_", value).strip("._-")
    return cleaned[:160] or fallback


def _safe_student_code(student_code: str) -> str:
    return _safe_name(student_code or "unknown_student", fallback="unknown_student")


def _safe_teacher_id(teacher_id: str) -> str:
    return _safe_name(teacher_id or "unknown_teacher", fallback="unknown_teacher")


def _safe_voucher_id(voucher_id: str) -> str:
    return _safe_name(voucher_id or "voucher_default", fallback="voucher_default")


def _new_work_id(prefix: str = "ocr") -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{ts}_{secrets.token_hex(4)}"


def _lookup_student_work_context(
    student_code: str,
    *,
    teacher_id: str = "",
    voucher_id: str = "",
) -> dict[str, str]:
    """
    Resolve the folder context for student work.

    Canonical target path:
      data/teachers/<teacher_id>/courses/<voucher_id>/students/
      <student_code>/<work_id>/
    """
    raw_student_code = str(
        student_code or ""
    ).strip()

    raw_teacher_id = str(
        teacher_id or ""
    ).strip()

    raw_voucher_id = str(
        voucher_id or ""
    ).strip()

    record: dict[str, Any] = {}

    if raw_student_code:
        try:
            from services.student_access import (
                get_student_code_record,
            )

            record = (
                get_student_code_record(
                    raw_student_code
                )
                or {}
            )
        except Exception:
            record = {}

    if not raw_teacher_id:
        raw_teacher_id = str(
            record.get("teacher_id")
            or ""
        ).strip()

    if not raw_voucher_id:
        raw_voucher_id = str(
            record.get("voucher_id")
            or record.get("voucher_code")
            or record.get("voucher_hash")
            or ""
        ).strip()

    if not raw_voucher_id and raw_teacher_id:
        try:
            from services.teacher_profiles import (
                get_teacher,
            )

            teacher = (
                get_teacher(raw_teacher_id)
                or {}
            )

            active_course = (
                teacher.get("active_course")
                if isinstance(
                    teacher.get("active_course"),
                    dict,
                )
                else {}
            )

            raw_voucher_id = str(
                active_course.get("voucher_id")
                or active_course.get("course_id")
                or teacher.get("voucher_id")
                or teacher.get("voucher_code")
                or teacher.get("voucher_hash")
                or ""
            ).strip()
        except Exception:
            raw_voucher_id = ""

    return {
        "student_code": raw_student_code,
        "teacher_id": raw_teacher_id,
        "voucher_id": raw_voucher_id,
        "safe_student_code": (
            _safe_student_code(
                raw_student_code
            )
        ),
        "safe_teacher_id": (
            _safe_teacher_id(
                raw_teacher_id
            )
        ),
        "safe_voucher_id": (
            _safe_voucher_id(
                raw_voucher_id
            )
        ),
    }


def _lookup_teacher_test_context(
    teacher_id: str,
    *,
    voucher_id: str = "",
) -> dict[str, str]:
    """
    Resolve the active course used for teacher-test storage.

    Teacher tests are stored separately from real student work:
      data/teachers/<teacher>/courses/<voucher>/teacher_tests/<work_id>/
    """
    raw_teacher_id = str(
        teacher_id or ""
    ).strip()

    if not raw_teacher_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing teacher ID for "
                "teacher-test storage."
            ),
        )

    try:
        from services.teacher_profiles import (
            get_teacher,
        )

        profile = (
            get_teacher(raw_teacher_id)
            or {}
        )
    except Exception:
        profile = {}

    active_course = (
        profile.get("active_course")
        if isinstance(
            profile.get("active_course"),
            dict,
        )
        else {}
    )

    raw_voucher_id = str(
        voucher_id
        or active_course.get("voucher_id")
        or active_course.get("course_id")
        or profile.get("voucher_id")
        or profile.get("active_course_id")
        or ""
    ).strip()

    if not raw_voucher_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Choose an active course before "
                "saving teacher-test work."
            ),
        )

    return {
        "teacher_id": raw_teacher_id,
        "voucher_id": raw_voucher_id,
        "course_label": str(
            active_course.get("course_label")
            or profile.get("course_label")
            or "Teacher test"
        ).strip(),
        "safe_teacher_id": (
            _safe_teacher_id(
                raw_teacher_id
            )
        ),
        "safe_voucher_id": (
            _safe_voucher_id(
                raw_voucher_id
            )
        ),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _course_root_from_ctx(ctx: dict[str, str]) -> Path:
    """
    Canonical course root:

      data/teachers/<teacher>/courses/<voucher>/
    """
    root = (
        TEACHERS_ROOT
        / ctx["safe_teacher_id"]
        / "courses"
        / ctx["safe_voucher_id"]
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _students_root_from_ctx(ctx: dict[str, str]) -> Path:
    root = _course_root_from_ctx(ctx) / "students"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _teacher_tests_root_from_ctx(
    ctx: dict[str, str],
) -> Path:
    root = (
        _course_root_from_ctx(ctx)
        / "teacher_tests"
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return root


def _course_manifest_path(ctx: dict[str, str]) -> Path:
    return _course_root_from_ctx(ctx) / "course_manifest.json"


def _course_upload_log_path_from_meta(meta: dict[str, Any]) -> Path:
    safe_teacher = _safe_teacher_id(str(meta.get("teacher_id") or "unknown_teacher"))
    safe_voucher = _safe_voucher_id(str(meta.get("voucher_id") or "voucher_default"))
    return TEACHERS_ROOT / safe_teacher / "courses" / safe_voucher / "upload_log.json"


def _ensure_course_manifest(ctx: dict[str, str]) -> None:
    path = _course_manifest_path(ctx)
    current = _read_json(path) or {}

    manifest = {
        "schema_version": "course_manifest_v1",
        "teacher_id": ctx.get("teacher_id") or current.get("teacher_id") or "",
        "voucher_id": ctx.get("voucher_id") or current.get("voucher_id") or "",
        "course_label": ctx.get("course_label") or current.get("course_label") or "Course",
        "updated_at": _now(),
        "created_at": current.get("created_at") or _now(),
    }

    _write_json(path, manifest)


def _upload_log_entry(meta: dict[str, Any]) -> dict[str, Any]:
    feedback = meta.get("feedback") if isinstance(meta.get("feedback"), dict) else {}

    return {
        "work_id": str(meta.get("work_id") or ""),
        "student_code": str(meta.get("student_code") or ""),
        "teacher_id": str(meta.get("teacher_id") or ""),
        "voucher_id": str(meta.get("voucher_id") or ""),
        "exam_id": str(meta.get("exam_id") or ""),
        "kind": str(meta.get("kind") or ""),
        "status": str(meta.get("status") or ""),
        "source_filename": str(meta.get("source_filename") or ""),
        "saved_at": str(meta.get("saved_at") or ""),
        "updated_at": str(meta.get("updated_at") or ""),
        "graded_at": str(meta.get("graded_at") or feedback.get("saved_at") or ""),
        "ocr_provider": str(meta.get("ocr_provider") or ""),
        "ocr_model": str(meta.get("ocr_model") or ""),
        "grading_provider": str(meta.get("grading_provider") or feedback.get("provider") or ""),
    }


def _upsert_course_upload_log(meta: dict[str, Any]) -> None:
    """
    Course-level upload index:

      data/teachers/<teacher>/courses/<voucher>/upload_log.json

    This is an index for dashboards/exports. metadata.json inside each work
    folder remains the detailed source of truth.
    """
    teacher_id = str(meta.get("teacher_id") or "").strip()
    voucher_id = str(meta.get("voucher_id") or "").strip()
    work_id = str(meta.get("work_id") or "").strip()

    if not teacher_id or not work_id:
        return

    path = _course_upload_log_path_from_meta(meta)
    current = _read_json(path) or {}

    uploads = current.get("uploads")
    if not isinstance(uploads, list):
        uploads = []

    entry = _upload_log_entry(meta)
    replaced = False
    out: list[dict[str, Any]] = []

    for old in uploads:
        if isinstance(old, dict) and str(old.get("work_id") or "") == work_id:
            out.append(entry)
            replaced = True
        elif isinstance(old, dict):
            out.append(old)

    if not replaced:
        out.append(entry)

    out.sort(
        key=lambda x: str(x.get("updated_at") or x.get("graded_at") or x.get("saved_at") or ""),
        reverse=True,
    )

    payload = {
        "schema_version": "course_upload_log_v1",
        "teacher_id": teacher_id,
        "voucher_id": voucher_id or "voucher_default",
        "updated_at": _now(),
        "uploads": out,
    }

    _write_json(path, payload)


def _student_root(
    student_code: str,
    *,
    teacher_id: str = "",
    voucher_id: str = "",
) -> Path:
    ctx = _lookup_student_work_context(
        student_code,
        teacher_id=teacher_id,
        voucher_id=voucher_id,
    )

    _ensure_course_manifest(ctx)

    root = _students_root_from_ctx(ctx) / ctx["safe_student_code"]
    root.mkdir(parents=True, exist_ok=True)
    return root


def _legacy_teacher_student_root(
    student_code: str,
    *,
    teacher_id: str = "",
    voucher_id: str = "",
) -> Path:
    ctx = _lookup_student_work_context(
        student_code,
        teacher_id=teacher_id,
        voucher_id=voucher_id,
    )
    return (
        LEGACY_STUDENT_WORK_ROOT
        / ctx["safe_teacher_id"]
        / ctx["safe_voucher_id"]
        / ctx["safe_student_code"]
    )


def _legacy_student_root(student_code: str) -> Path:
    """
    Old layout kept for backward compatibility:
      data/student_work/<student_code>/
    """
    return STUDENT_WORK_ROOT / _safe_student_code(student_code)


def _candidate_student_roots(
    student_code: str,
    *,
    teacher_id: str = "",
    voucher_id: str = "",
) -> list[Path]:
    ctx = _lookup_student_work_context(
        student_code,
        teacher_id=teacher_id,
        voucher_id=voucher_id,
    )

    roots: list[Path] = []
    seen: set[str] = set()

    def add_root(path: Path, *, create: bool = False) -> None:
        if create:
            path.mkdir(parents=True, exist_ok=True)
        if not create and not path.exists():
            return
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            return
        seen.add(key)
        roots.append(path)

    # Canonical current root.
    add_root(_student_root(student_code, teacher_id=ctx["teacher_id"], voucher_id=ctx["voucher_id"]), create=True)

    # Canonical current root, but across every voucher for this teacher.
    teacher_courses_root = TEACHERS_ROOT / ctx["safe_teacher_id"] / "courses"
    if teacher_courses_root.exists():
        for p in teacher_courses_root.glob(f"*/students/{ctx['safe_student_code']}"):
            add_root(p)

    # Previous teacher/voucher/student layout.
    add_root(
        _legacy_teacher_student_root(
            student_code,
            teacher_id=ctx["teacher_id"],
            voucher_id=ctx["voucher_id"],
        )
    )

    # Previous teacher/voucher/student layout, across every voucher.
    legacy_teacher_root = LEGACY_STUDENT_WORK_ROOT / ctx["safe_teacher_id"]
    if legacy_teacher_root.exists():
        for p in legacy_teacher_root.glob(f"*/{ctx['safe_student_code']}"):
            add_root(p)

    # Very old student-only layout:
    #   data/student_work/<student_code>/
    add_root(_legacy_student_root(student_code))

    return roots


def _find_work_dir(
    student_code: str,
    work_id: str,
    *,
    teacher_id: str = "",
    voucher_id: str = "",
) -> Path | None:
    work_id = _assert_safe_token(work_id, label="work_id")

    for root in _candidate_student_roots(
        student_code,
        teacher_id=teacher_id,
        voucher_id=voucher_id,
    ):
        work_dir = root / work_id
        if (work_dir / "metadata.json").exists():
            return work_dir

    return None


def _find_teacher_test_work_dir(
    teacher_id: str,
    work_id: str,
    *,
    voucher_id: str = "",
) -> Path | None:
    work_id = _assert_safe_token(
        work_id,
        label="work_id",
    )

    ctx = _lookup_teacher_test_context(
        teacher_id,
        voucher_id=voucher_id,
    )

    courses_root = (
        TEACHERS_ROOT
        / ctx["safe_teacher_id"]
        / "courses"
    )

    candidate_roots = [
        (
            courses_root
            / ctx["safe_voucher_id"]
            / "teacher_tests"
        )
    ]

    # Work IDs are globally random, so searching the teacher's other
    # courses also lets an older test remain accessible after the
    # teacher changes the active course.
    if courses_root.exists():
        candidate_roots.extend(
            courses_root.glob(
                "*/teacher_tests"
            )
        )

    seen: set[str] = set()

    for root in candidate_roots:
        key = (
            str(root.resolve())
            if root.exists()
            else str(root)
        )

        if key in seen:
            continue

        seen.add(key)

        work_dir = root / work_id

        if (
            work_dir
            / "metadata.json"
        ).exists():
            return work_dir

    return None


def _assert_safe_token(value: str, *, label: str) -> str:
    value = (value or "").strip()
    if not value or "/" in value or "\\" in value or ".." in value:
        raise HTTPException(status_code=400, detail=f"Invalid {label}.")
    if not _SAFE_TOKEN_RE.match(value):
        raise HTTPException(status_code=400, detail=f"Invalid {label}.")
    return value


def _ocr_local_now() -> datetime:
    try:
        return datetime.now(
            ZoneInfo(APP_TIMEZONE)
        )
    except (
        ZoneInfoNotFoundError,
        ValueError,
    ):
        # Windows may not have the IANA timezone database installed.
        return datetime.now().astimezone()


def _student_ocr_usage_path(
    student_code: str,
    *,
    teacher_id: str = "",
    voucher_id: str = "",
) -> tuple[Path, dict[str, str]]:
    ctx = _lookup_student_work_context(
        student_code,
        teacher_id=teacher_id,
        voucher_id=voucher_id,
    )

    root = _student_root(
        student_code,
        teacher_id=ctx["teacher_id"],
        voucher_id=ctx["voucher_id"],
    )

    return (
        root / "ocr_usage.json",
        ctx,
    )


def _parse_ocr_usage_time(
    value: Any,
    *,
    fallback_tz,
) -> datetime | None:
    text = str(
        value or ""
    ).strip()

    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(
            text.replace(
                "Z",
                "+00:00",
            )
        )
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=fallback_tz
        )

    return parsed


def _load_student_ocr_usage(
    path: Path,
    ctx: dict[str, str],
    now: datetime,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    current = (
        _read_json(path)
        or {}
    )

    raw_days = (
        current.get("days")
        if isinstance(
            current.get("days"),
            dict,
        )
        else {}
    )

    cleaned_days: dict[
        str,
        dict[str, Any],
    ] = {}

    history_cutoff = (
        now.date()
        - timedelta(days=31)
    )

    for day_key, raw_day in raw_days.items():
        try:
            day_date = datetime.strptime(
                str(day_key),
                "%Y-%m-%d",
            ).date()
        except ValueError:
            continue

        if day_date < history_cutoff:
            continue

        day = (
            raw_day
            if isinstance(
                raw_day,
                dict,
            )
            else {}
        )

        runs = [
            item
            for item in (
                day.get("runs")
                if isinstance(
                    day.get("runs"),
                    list,
                )
                else []
            )
            if isinstance(
                item,
                dict,
            )
        ]

        reservations: list[
            dict[str, Any]
        ] = []

        for item in (
            day.get("reservations")
            if isinstance(
                day.get("reservations"),
                list,
            )
            else []
        ):
            if not isinstance(
                item,
                dict,
            ):
                continue

            expires_at = (
                _parse_ocr_usage_time(
                    item.get("expires_at"),
                    fallback_tz=now.tzinfo,
                )
            )

            if (
                expires_at is not None
                and expires_at > now
            ):
                reservations.append(
                    item
                )

        cleaned_days[
            str(day_key)
        ] = {
            "limit": (
                STUDENT_OCR_DAILY_LIMIT
            ),
            "used": len(runs),
            "runs": runs,
            "reservations": (
                reservations
            ),
        }

    today = (
        now.date().isoformat()
    )

    current_day = (
        cleaned_days.setdefault(
            today,
            {
                "limit": (
                    STUDENT_OCR_DAILY_LIMIT
                ),
                "used": 0,
                "runs": [],
                "reservations": [],
            },
        )
    )

    state = {
        "schema_version": (
            "student_ocr_usage_v1"
        ),
        "student_code": (
            ctx.get("student_code")
            or ""
        ),
        "teacher_id": (
            ctx.get("teacher_id")
            or ""
        ),
        "voucher_id": (
            ctx.get("voucher_id")
            or ""
        ),
        "timezone": APP_TIMEZONE,
        "days": cleaned_days,
        "updated_at": (
            now.isoformat(
                timespec="seconds"
            )
        ),
    }

    return (
        state,
        current_day,
    )


def _student_ocr_quota_snapshot(
    day: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    runs = (
        day.get("runs")
        if isinstance(
            day.get("runs"),
            list,
        )
        else []
    )

    reservations = (
        day.get("reservations")
        if isinstance(
            day.get("reservations"),
            list,
        )
        else []
    )

    used = len(runs)
    reserved = len(reservations)

    remaining = max(
        0,
        STUDENT_OCR_DAILY_LIMIT
        - used
        - reserved,
    )

    next_day = (
        now + timedelta(days=1)
    ).date()

    reset_at = datetime.combine(
        next_day,
        datetime.min.time(),
        tzinfo=now.tzinfo,
    )

    return {
        "date": (
            now.date().isoformat()
        ),
        "timezone": APP_TIMEZONE,
        "limit": (
            STUDENT_OCR_DAILY_LIMIT
        ),
        "used": used,
        "reserved": reserved,
        "remaining": remaining,
        "reached": (
            remaining <= 0
        ),
        "reset_at": (
            reset_at.isoformat(
                timespec="seconds"
            )
        ),
    }


def get_student_ocr_daily_usage(
    student_code: str,
    *,
    teacher_id: str = "",
    voucher_id: str = "",
) -> dict[str, Any]:
    student_code = str(
        student_code or ""
    ).strip()

    if not student_code:
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing student code "
                "for OCR quota lookup."
            ),
        )

    with _OCR_USAGE_LOCK:
        path, ctx = (
            _student_ocr_usage_path(
                student_code,
                teacher_id=teacher_id,
                voucher_id=voucher_id,
            )
        )

        now = _ocr_local_now()

        (
            state,
            current_day,
        ) = _load_student_ocr_usage(
            path,
            ctx,
            now,
        )

        _write_json(
            path,
            state,
        )

        return (
            _student_ocr_quota_snapshot(
                current_day,
                now,
            )
        )


def reserve_student_ocr_slot(
    student_code: str,
    *,
    teacher_id: str = "",
    voucher_id: str = "",
) -> dict[str, Any]:
    student_code = str(
        student_code or ""
    ).strip()

    if not student_code:
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing student code "
                "for OCR quota reservation."
            ),
        )

    with _OCR_USAGE_LOCK:
        path, ctx = (
            _student_ocr_usage_path(
                student_code,
                teacher_id=teacher_id,
                voucher_id=voucher_id,
            )
        )

        now = _ocr_local_now()

        (
            state,
            current_day,
        ) = _load_student_ocr_usage(
            path,
            ctx,
            now,
        )

        quota = (
            _student_ocr_quota_snapshot(
                current_day,
                now,
            )
        )

        if quota["reached"]:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Daily OCR limit reached. "
                    f"Students may complete "
                    f"{STUDENT_OCR_DAILY_LIMIT} "
                    "successful OCR uploads per day. "
                    "Saved OCR work can still be "
                    "opened, edited, and graded."
                ),
            )

        reservation_id = (
            "ocr_res_"
            + secrets.token_hex(12)
        )

        expires_at = (
            now
            + timedelta(
                seconds=(
                    STUDENT_OCR_RESERVATION_TTL_SECONDS
                )
            )
        )

        current_day[
            "reservations"
        ].append(
            {
                "reservation_id": (
                    reservation_id
                ),
                "created_at": (
                    now.isoformat(
                        timespec="seconds"
                    )
                ),
                "expires_at": (
                    expires_at.isoformat(
                        timespec="seconds"
                    )
                ),
            }
        )

        state["updated_at"] = (
            now.isoformat(
                timespec="seconds"
            )
        )

        _write_json(
            path,
            state,
        )

        result = (
            _student_ocr_quota_snapshot(
                current_day,
                now,
            )
        )

        result["reservation_id"] = (
            reservation_id
        )

        return result


def release_student_ocr_slot(
    student_code: str,
    reservation_id: str,
    *,
    teacher_id: str = "",
    voucher_id: str = "",
) -> dict[str, Any]:
    with _OCR_USAGE_LOCK:
        path, ctx = (
            _student_ocr_usage_path(
                student_code,
                teacher_id=teacher_id,
                voucher_id=voucher_id,
            )
        )

        now = _ocr_local_now()

        (
            state,
            current_day,
        ) = _load_student_ocr_usage(
            path,
            ctx,
            now,
        )

        for day in (
            state["days"].values()
        ):
            day["reservations"] = [
                item
                for item in (
                    day["reservations"]
                )
                if str(
                    item.get(
                        "reservation_id"
                    )
                    or ""
                )
                != reservation_id
            ]

        state["updated_at"] = (
            now.isoformat(
                timespec="seconds"
            )
        )

        _write_json(
            path,
            state,
        )

        return (
            _student_ocr_quota_snapshot(
                current_day,
                now,
            )
        )


def finalize_student_ocr_slot(
    student_code: str,
    reservation_id: str,
    *,
    work_id: str,
    provider: str,
    source_filename: str,
    teacher_id: str = "",
    voucher_id: str = "",
) -> dict[str, Any]:
    student_code = str(
        student_code or ""
    ).strip()

    reservation_id = str(
        reservation_id or ""
    ).strip()

    work_id = str(
        work_id or ""
    ).strip()

    if (
        not student_code
        or not reservation_id
        or not work_id
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                "The successful OCR result "
                "could not be recorded in "
                "the daily quota."
            ),
        )

    with _OCR_USAGE_LOCK:
        path, ctx = (
            _student_ocr_usage_path(
                student_code,
                teacher_id=teacher_id,
                voucher_id=voucher_id,
            )
        )

        now = _ocr_local_now()

        (
            state,
            current_day,
        ) = _load_student_ocr_usage(
            path,
            ctx,
            now,
        )

        # Idempotency: finalizing the same work or reservation twice must not
        # consume another slot.
        for day in (
            state["days"].values()
        ):
            for run in day["runs"]:
                if (
                    str(
                        run.get(
                            "reservation_id"
                        )
                        or ""
                    )
                    == reservation_id
                    or str(
                        run.get("work_id")
                        or ""
                    )
                    == work_id
                ):
                    return (
                        _student_ocr_quota_snapshot(
                            current_day,
                            now,
                        )
                    )

        reserved_day = next(
            (
                day
                for day in (
                    state[
                        "days"
                    ].values()
                )
                if any(
                    str(
                        item.get(
                            "reservation_id"
                        )
                        or ""
                    )
                    == reservation_id
                    for item in (
                        day[
                            "reservations"
                        ]
                    )
                )
            ),
            None,
        )

        if reserved_day is None:
            raise HTTPException(
                status_code=500,
                detail=(
                    "The OCR quota reservation "
                    "expired before the successful "
                    "result was saved."
                ),
            )

        reserved_day[
            "reservations"
        ] = [
            item
            for item in (
                reserved_day[
                    "reservations"
                ]
            )
            if str(
                item.get(
                    "reservation_id"
                )
                or ""
            )
            != reservation_id
        ]

        reserved_day[
            "runs"
        ].append(
            {
                "reservation_id": (
                    reservation_id
                ),
                "work_id": work_id,
                "provider": (
                    provider or ""
                ),
                "source_filename": (
                    source_filename or ""
                ),
                "completed_at": (
                    now.isoformat(
                        timespec="seconds"
                    )
                ),
            }
        )

        reserved_day["used"] = len(
            reserved_day["runs"]
        )

        state["updated_at"] = (
            now.isoformat(
                timespec="seconds"
            )
        )

        _write_json(
            path,
            state,
        )

        return (
            _student_ocr_quota_snapshot(
                current_day,
                now,
            )
        )



def save_ocr_student_work(
    *,
    student_code: str = "",
    teacher_id: str = "",
    voucher_id: str = "",
    work_scope: str = "student",
    source_filename: str,
    uploaded_bytes: bytes,
    ocr_provider: str,
    ocr_model: str,
    document_type: str,
    ocr_path: str,
    route_reason: str = "",
    debug_trace_id: str = "",
    debug_trace_dir: str = "",
    ocr_response_path: Path | None = None,
    student_work_result: Any = None,
    student_summary_path: Path | None = None,
) -> dict[str, Any]:
    """
    Persist the student's OCR attempt in a stable, dashboard-visible folder.

    This stores:
    - original uploaded scan/PDF/image
    - provider-neutral OCR response JSON
    - generated OCR TeX, if available
    - student_answer_bundle.json, if available
    - student_summary.json, if available
    """
    normalized_scope = str(
        work_scope or "student"
    ).strip().lower()

    student_code = (
        student_code or ""
    ).strip()

    if normalized_scope == "teacher_test":
        ctx = _lookup_teacher_test_context(
            teacher_id,
            voucher_id=voucher_id,
        )

        work_id = _new_work_id(
            "teacher_test_ocr"
        )

        work_dir = (
            _teacher_tests_root_from_ctx(
                ctx
            )
            / work_id
        )
    else:
        if not student_code:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing student code "
                    "for OCR storage."
                ),
            )

        ctx = _lookup_student_work_context(
            student_code,
            teacher_id=teacher_id,
            voucher_id=voucher_id,
        )

        work_id = _new_work_id("ocr")

        work_dir = (
            _student_root(
                student_code,
                teacher_id=ctx["teacher_id"],
                voucher_id=ctx["voucher_id"],
            )
            / work_id
        )

    work_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    original_name = "original_" + _safe_name(source_filename or "upload.bin", "upload.bin")
    original_path = work_dir / original_name
    original_path.write_bytes(uploaded_bytes or b"")

    files: list[dict[str, str]] = [
        {
            "kind": "original",
            "filename": original_name,
            "label": "Original upload",
        }
    ]

    primary_filename = original_name

    if ocr_response_path and Path(ocr_response_path).exists():
        shutil.copy2(ocr_response_path, work_dir / "ocr_response.json")
        files.append(
            {
                "kind": "ocr_response",
                "filename": "ocr_response.json",
                "label": "OCR response JSON",
            }
        )

    if student_work_result is not None:
        try:
            _write_json(work_dir / "student_work_ocr_result.json", student_work_result.model_dump())
            files.append(
                {
                    "kind": "student_work_ocr_result",
                    "filename": "student_work_ocr_result.json",
                    "label": "Student OCR result JSON",
                }
            )
        except Exception:
            pass

        tex_path_raw = getattr(student_work_result, "student_tex_path", None)
        if tex_path_raw:
            tex_path = Path(tex_path_raw)
            if tex_path.exists():
                shutil.copy2(tex_path, work_dir / "ocr_student_answer.tex")
                primary_filename = "ocr_student_answer.tex"
                files.append(
                    {
                        "kind": "ocr_tex",
                        "filename": "ocr_student_answer.tex",
                        "label": "OCR LaTeX",
                    }
                )

        bundle_path_raw = getattr(student_work_result, "student_answer_bundle_path", None)
        if bundle_path_raw:
            bundle_path = Path(bundle_path_raw)
            if bundle_path.exists():
                shutil.copy2(bundle_path, work_dir / "student_answer_bundle.json")
                files.append(
                    {
                        "kind": "student_answer_bundle",
                        "filename": "student_answer_bundle.json",
                        "label": "Structured answer JSON",
                    }
                )

        raw_text = str(getattr(student_work_result, "raw_ocr_text", "") or "").strip()
        if raw_text:
            (work_dir / "ocr_text.txt").write_text(raw_text, encoding="utf-8")
            files.append(
                {
                    "kind": "ocr_text",
                    "filename": "ocr_text.txt",
                    "label": "OCR plain text",
                }
            )
    if (
        student_summary_path
        and Path(
            student_summary_path
        ).exists()
    ):
        shutil.copy2(
            student_summary_path,
            work_dir / "student_summary.json",
        )

        files.append(
            {
                "kind": "student_summary",
                "filename": "student_summary.json",
                "label": "Student summary JSON",
            }
        )

    if normalized_scope == "teacher_test":
        metadata = {
            "schema_version": (
                "teacher_test_work_v1"
            ),
            "work_scope": "teacher_test",
            "student_code": "",
            "teacher_id": ctx["teacher_id"],
            "voucher_id": ctx["voucher_id"],
            "course_label": (
                ctx.get("course_label")
                or ""
            ),
            "storage_schema": (
                "teacher_course_teacher_tests_v1"
            ),
            "storage_path_parts": {
                "teacher": (
                    ctx["safe_teacher_id"]
                ),
                "courses": "courses",
                "voucher": (
                    ctx["safe_voucher_id"]
                ),
                "teacher_tests": (
                    "teacher_tests"
                ),
            },
        }
    else:
        metadata = {
            "schema_version": (
                "student_work_v1"
            ),
            "work_scope": "student",
            "student_code": student_code,
            "teacher_id": ctx["teacher_id"],
            "voucher_id": ctx["voucher_id"],
            "storage_schema": (
                "teacher_course_voucher_students_v2"
            ),
            "storage_path_parts": {
                "teacher": (
                    ctx["safe_teacher_id"]
                ),
                "courses": "courses",
                "voucher": (
                    ctx["safe_voucher_id"]
                ),
                "students": "students",
                "student": (
                    ctx["safe_student_code"]
                ),
            },
        }

    metadata.update(
        {
            "kind": "ocr",
            "status": "ocr_ready",
            "work_id": work_id,
            "source_filename": (
                source_filename or ""
            ),
            "saved_at": _now(),
            "ocr_provider": (
                ocr_provider or ""
            ),
            "ocr_model": (
                ocr_model or ""
            ),
            "document_type": (
                document_type or ""
            ),
            "ocr_path": ocr_path or "",
            "route_reason": (
                route_reason or ""
            ),
            "debug_trace_id": (
                debug_trace_id or ""
            ),
            "debug_trace_dir": (
                debug_trace_dir or ""
            ),
            "primary_filename": (
                primary_filename
            ),
            "files": files,
        }
    )

    _write_json(
        work_dir / "metadata.json",
        metadata,
    )

    # Teacher tests must not affect real student upload analytics.
    if normalized_scope != "teacher_test":
        _upsert_course_upload_log(
            metadata
        )

    return metadata


def save_ocr_teacher_test_work(
    *,
    teacher_id: str,
    voucher_id: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    return save_ocr_student_work(
        student_code="",
        teacher_id=teacher_id,
        voucher_id=voucher_id,
        work_scope="teacher_test",
        **kwargs,
    )


def create_tex_student_work(
    *,
    student_code: str = "",
    teacher_id: str = "",
    voucher_id: str = "",
    work_scope: str = "student",
    source_filename: str,
    uploaded_bytes: bytes,
    debug_trace_id: str = "",
    debug_trace_dir: str = "",
) -> dict[str, Any]:
    """
    Persist a direct LaTeX/TXT student upload as a student work item.

    This makes direct TeX uploads use the same canonical storage as OCR uploads:
      data/student_work/<teacher>/<voucher>/<student>/<work_id>/
    """
    normalized_scope = str(
        work_scope or "student"
    ).strip().lower()

    student_code = (
        student_code or ""
    ).strip()

    if normalized_scope == "teacher_test":
        ctx = _lookup_teacher_test_context(
            teacher_id,
            voucher_id=voucher_id,
        )

        work_id = _new_work_id(
            "teacher_test_tex"
        )

        work_dir = (
            _teacher_tests_root_from_ctx(
                ctx
            )
            / work_id
        )
    else:
        if not student_code:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing student code for "
                    "student-work storage."
                ),
            )

        ctx = _lookup_student_work_context(
            student_code,
            teacher_id=teacher_id,
            voucher_id=voucher_id,
        )

        work_id = _new_work_id("tex")

        work_dir = (
            _student_root(
                student_code,
                teacher_id=ctx["teacher_id"],
                voucher_id=ctx["voucher_id"],
            )
            / work_id
        )

    work_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    original_name = "original_" + _safe_name(source_filename or "student_answer.tex", "student_answer.tex")
    original_path = work_dir / original_name
    original_path.write_bytes(uploaded_bytes or b"")

    files: list[dict[str, str]] = [
        {
            "kind": "student_tex",
            "filename": original_name,
            "label": "Student LaTeX upload",
        }
    ]

    if normalized_scope == "teacher_test":
        metadata = {
            "schema_version": (
                "teacher_test_work_v1"
            ),
            "work_scope": "teacher_test",
            "student_code": "",
            "teacher_id": ctx["teacher_id"],
            "voucher_id": ctx["voucher_id"],
            "course_label": (
                    ctx.get("course_label")
                    or ""
            ),
            "storage_schema": (
                "teacher_course_teacher_tests_v1"
            ),
            "storage_path_parts": {
                "teacher": (
                    ctx["safe_teacher_id"]
                ),
                "courses": "courses",
                "voucher": (
                    ctx["safe_voucher_id"]
                ),
                "teacher_tests": (
                    "teacher_tests"
                ),
            },
        }
    else:
        metadata = {
            "schema_version": (
                "student_work_v1"
            ),
            "work_scope": "student",
            "student_code": student_code,
            "teacher_id": ctx["teacher_id"],
            "voucher_id": ctx["voucher_id"],
            "storage_schema": (
                "teacher_voucher_student_work_v1"
            ),
            "storage_path_parts": {
                "teacher": (
                    ctx["safe_teacher_id"]
                ),
                "voucher": (
                    ctx["safe_voucher_id"]
                ),
                "student": (
                    ctx["safe_student_code"]
                ),
            },
        }

    metadata.update(
        {
            "kind": "tex",
            "status": "uploaded",
            "work_id": work_id,
            "source_filename": (
                    source_filename or ""
            ),
            "saved_at": _now(),
            "debug_trace_id": (
                    debug_trace_id or ""
            ),
            "debug_trace_dir": (
                    debug_trace_dir or ""
            ),
            "primary_filename": (
                original_name
            ),
            "files": files,
        }
    )

    _write_json(
        work_dir / "metadata.json",
        metadata,
    )

    # Teacher tests must not affect student upload counts.
    if normalized_scope != "teacher_test":
        _upsert_course_upload_log(
            metadata
        )

    return metadata


def create_tex_teacher_test_work(
        *,
        teacher_id: str,
        voucher_id: str = "",
        **kwargs: Any,
) -> dict[str, Any]:
    return create_tex_student_work(
        student_code="",
        teacher_id=teacher_id,
        voucher_id=voucher_id,
        work_scope="teacher_test",
        **kwargs,
    )


def _student_work_download_url(
    work_id: str,
    filename: str,
) -> str:
    return (
        "/routes/student/work_file"
        f"?work_id={quote(work_id)}"
        f"&filename={quote(filename)}"
    )


def _teacher_test_work_download_url(
    work_id: str,
    filename: str,
) -> str:
    return (
        "/routes/student/work_file"
        "?work_scope=teacher_test"
        f"&work_id={quote(work_id)}"
        f"&filename={quote(filename)}"
    )


def _upsert_file_entry(files: list[dict[str, Any]], entry: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Keep one current file per kind. This avoids showing old feedback files as
    separate buttons when the same OCR work is graded again.
    """
    kind = str(entry.get("kind") or "").strip()
    if not kind:
        files.append(entry)
        return files

    out = []
    replaced = False
    for f in files:
        if isinstance(f, dict) and f.get("kind") == kind:
            if not replaced:
                out.append(entry)
                replaced = True
            continue
        out.append(f)

    if not replaced:
        out.append(entry)

    return out


def attach_grading_feedback_to_student_work(
    *,
    student_code: str = "",
    teacher_id: str = "",
    voucher_id: str = "",
    work_scope: str = "student",
    work_id: str,
    exam_id: str,
    provider: str,
    final_path: Path,
    graded_result_json_path: (
        Path | None
    ) = None,
    graded_result_tex_path: (
        Path | None
    ) = None,
    saved_at: str | None = None,
) -> dict[str, Any] | None:
    """
    Attach all current grading artifacts to one student-work item:

      - final PDF or TeX fallback
      - canonical graded_result.json
      - source TeX used to produce the PDF
    """
    normalized_scope = str(
        work_scope or "student"
    ).strip().lower()

    student_code = (
        student_code or ""
    ).strip()

    work_id = _assert_safe_token(
        work_id,
        label="work_id",
    )

    if normalized_scope == "teacher_test":
        work_dir = (
            _find_teacher_test_work_dir(
                teacher_id,
                work_id,
                voucher_id=voucher_id,
            )
        )
    else:
        if not student_code:
            return None

        work_dir = _find_work_dir(
            student_code,
            work_id,
        )

    if work_dir is None:
        return None

    meta_path = (
        work_dir
        / "metadata.json"
    )

    meta = _read_json(meta_path)

    if not meta:
        return None

    if normalized_scope == "teacher_test":
        if (
            str(
                meta.get("teacher_id")
                or ""
            ).strip()
            != str(
                teacher_id or ""
            ).strip()
            or str(
                meta.get("work_scope")
                or ""
            ).strip()
            != "teacher_test"
        ):
            return None
    elif (
        str(
            meta.get("student_code")
            or ""
        ).strip()
        != student_code
    ):
        return None

    final_source = Path(
        final_path
    )

    if (
        not final_source.exists()
        or not final_source.is_file()
    ):
        return None

    exam_part = _safe_name(
        exam_id or "graded_work",
        "graded_work",
    )

    provider_part = _safe_name(
        provider or "ai",
        "ai",
    )

    version_tag = (
        datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
    )

    def copy_artifact(
        source_path: Path | None,
        desired_filename: str,
    ) -> str:
        if not source_path:
            return ""

        source = Path(source_path)

        if (
            not source.exists()
            or not source.is_file()
        ):
            return ""

        target = (
            work_dir
            / desired_filename
        )

        if target.exists():
            desired = Path(
                desired_filename
            )

            target = (
                work_dir
                / (
                    f"{desired.stem}_"
                    f"{version_tag}"
                    f"{desired.suffix}"
                )
            )

        shutil.copy2(
            source,
            target,
        )

        return target.name

    final_suffix = (
        final_source.suffix.lower()
        or ".pdf"
    )

    feedback_filename = (
        copy_artifact(
            final_source,
            (
                f"graded_{exam_part}_"
                f"{provider_part}"
                f"{final_suffix}"
            ),
        )
    )

    if not feedback_filename:
        return None

    structured_json_filename = (
        copy_artifact(
            graded_result_json_path,
            (
                f"graded_{exam_part}_"
                f"{provider_part}"
                "_structured.json"
            ),
        )
    )

    tex_filename = ""

    if graded_result_tex_path:
        tex_source = Path(
            graded_result_tex_path
        )

        try:
            same_as_final = (
                tex_source.resolve()
                == final_source.resolve()
            )
        except Exception:
            same_as_final = False

        if (
            same_as_final
            and final_suffix == ".tex"
        ):
            tex_filename = (
                feedback_filename
            )
        else:
            tex_filename = (
                copy_artifact(
                    tex_source,
                    (
                        f"graded_{exam_part}_"
                        f"{provider_part}"
                        "_source.tex"
                    ),
                )
            )

    feedback_saved_at = (
        saved_at or _now()
    )

    feedback = {
        "exam_id": exam_id or "",
        "provider": provider or "",
        "filename": (
            feedback_filename
        ),
        "file_type": (
            Path(
                feedback_filename
            ).suffix.lower().lstrip(".")
        ),
        "structured_json_filename": (
            structured_json_filename
        ),
        "tex_filename": tex_filename,
        "saved_at": (
            feedback_saved_at
        ),
    }

    files = (
        meta.get("files")
        if isinstance(
            meta.get("files"),
            list,
        )
        else []
    )

    files = _upsert_file_entry(
        files,
        {
            "kind": "graded_feedback",
            "filename": (
                feedback_filename
            ),
            "label": (
                "Graded feedback"
            ),
        },
    )

    if structured_json_filename:
        files = _upsert_file_entry(
            files,
            {
                "kind": (
                    "graded_result_json"
                ),
                "filename": (
                    structured_json_filename
                ),
                "label": (
                    "Structured graded result"
                ),
            },
        )

    if (
        tex_filename
        and tex_filename
        != feedback_filename
    ):
        files = _upsert_file_entry(
            files,
            {
                "kind": (
                    "graded_result_tex"
                ),
                "filename": (
                    tex_filename
                ),
                "label": (
                    "Graded result LaTeX"
                ),
            },
        )

    meta["status"] = "graded"

    meta["exam_id"] = (
        exam_id
        or meta.get("exam_id")
        or ""
    )

    meta["graded_at"] = (
        feedback_saved_at
    )

    meta["grading_provider"] = (
        provider or ""
    )

    meta["feedback"] = feedback

    meta["graded_result"] = {
        "schema_version": (
            "graded_result_v1"
        ),
        "final_filename": (
            feedback_filename
        ),
        "structured_json_filename": (
            structured_json_filename
        ),
        "tex_filename": (
            tex_filename
        ),
    }

    meta["files"] = files
    meta["updated_at"] = _now()

    _write_json(
        meta_path,
        meta,
    )

    if normalized_scope != "teacher_test":
        _upsert_course_upload_log(
            meta
        )

    return meta


def attach_grading_feedback_to_teacher_test_work(
        *,
        teacher_id: str,
        work_id: str,
        exam_id: str,
        provider: str,
        final_path: Path,
        voucher_id: str = "",
        graded_result_json_path: Path | None = None,
        graded_result_tex_path: Path | None = None,
        saved_at: str | None = None,
) -> dict[str, Any] | None:
    return attach_grading_feedback_to_student_work(
        student_code="",
        teacher_id=teacher_id,
        voucher_id=voucher_id,
        work_scope="teacher_test",
        work_id=work_id,
        exam_id=exam_id,
        provider=provider,
        final_path=final_path,
        graded_result_json_path=(
            graded_result_json_path
        ),
        graded_result_tex_path=(
            graded_result_tex_path
        ),
        saved_at=saved_at,
    )


def list_teacher_student_work_metadata(
        teacher_id: str,
) -> list[dict[str, Any]]:
    """
    Return canonical student-work metadata for a teacher, across vouchers/courses.

    Reads both:
      New:
        data/teachers/<teacher>/courses/<voucher>/students/<student>/<work>/metadata.json

      Legacy:
        data/student_work/<teacher>/<voucher>/<student>/<work>/metadata.json

    The function intentionally returns metadata only, not file contents.
    """
    safe_teacher_id = _safe_teacher_id(teacher_id)
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add_meta(meta_path: Path, *, layout: str) -> None:
        meta = _read_json(meta_path)
        if not meta:
            return

        student_code = str(meta.get("student_code") or "").strip()
        voucher_id = str(meta.get("voucher_id") or "voucher_default").strip()
        work_id = str(meta.get("work_id") or meta_path.parent.name).strip()

        # Recover path-derived fields when old metadata is incomplete.
        try:
            if layout == "current":
                # data/teachers/<teacher>/courses/<voucher>/students/<student>/<work>/metadata.json
                parts = meta_path.parts
                voucher_part = meta_path.parent.parent.parent.parent.name
                student_part = meta_path.parent.parent.name
            else:
                # data/student_work/<teacher>/<voucher>/<student>/<work>/metadata.json
                rel = meta_path.relative_to(LEGACY_STUDENT_WORK_ROOT / safe_teacher_id)
                voucher_part = rel.parts[0] if len(rel.parts) >= 4 else ""
                student_part = rel.parts[1] if len(rel.parts) >= 4 else ""
        except Exception:
            voucher_part = ""
            student_part = ""

        if not student_code:
            student_code = student_part
        if not voucher_id:
            voucher_id = voucher_part or "voucher_default"

        key = (voucher_id, student_code, work_id)
        if key in seen:
            return
        seen.add(key)

        item = dict(meta)
        item["teacher_id"] = str(item.get("teacher_id") or teacher_id or "").strip()
        item["voucher_id"] = voucher_id
        item["student_code"] = student_code
        item["work_id"] = work_id
        item["metadata_saved_at"] = str(item.get("saved_at") or "")
        item["metadata_updated_at"] = str(item.get("updated_at") or "")
        item["metadata_path"] = str(meta_path)
        item["storage_layout"] = layout
        items.append(item)

    # New canonical layout.
    current_root = TEACHERS_ROOT / safe_teacher_id / "courses"
    if current_root.exists():
        for meta_path in current_root.glob("*/students/*/*/metadata.json"):
            add_meta(meta_path, layout="current")

    # Legacy teacher/voucher/student layout.
    legacy_teacher_root = LEGACY_STUDENT_WORK_ROOT / safe_teacher_id
    if legacy_teacher_root.exists():
        for meta_path in legacy_teacher_root.glob("*/*/*/metadata.json"):
            add_meta(meta_path, layout="legacy_teacher_voucher_student")

    # Extra safety: scan any old metadata that says it belongs to this teacher.
    if LEGACY_STUDENT_WORK_ROOT.exists():
        for meta_path in LEGACY_STUDENT_WORK_ROOT.glob("*/*/*/metadata.json"):
            meta = _read_json(meta_path)
            if isinstance(meta, dict) and str(meta.get("teacher_id") or "").strip() == teacher_id:
                add_meta(meta_path, layout="legacy_global_scan")

    items.sort(key=lambda x: str(x.get("graded_at") or x.get("updated_at") or x.get("saved_at") or ""), reverse=True)
    return items


def list_teacher_test_work_for_dashboard(
    teacher_id: str,
    *,
    voucher_id: str = "",
) -> list[dict[str, Any]]:
    """
    List saved teacher-test attempts for the teacher's active course.

    Teacher tests are stored separately from student submissions and are
    therefore not read from upload_log.json.
    """
    teacher_id = str(
        teacher_id or ""
    ).strip()

    if not teacher_id:
        return []

    try:
        ctx = _lookup_teacher_test_context(
            teacher_id,
            voucher_id=voucher_id,
        )
    except HTTPException:
        return []

    root = (
        TEACHERS_ROOT
        / ctx["safe_teacher_id"]
        / "courses"
        / ctx["safe_voucher_id"]
        / "teacher_tests"
    )

    if not root.exists():
        return []

    items: list[dict[str, Any]] = []

    for meta_path in root.glob(
        "*/metadata.json"
    ):
        meta = _read_json(meta_path)

        if not meta:
            continue

        if (
            str(
                meta.get("teacher_id")
                or ""
            ).strip()
            != teacher_id
        ):
            continue

        if (
            str(
                meta.get("work_scope")
                or ""
            ).strip()
            != "teacher_test"
        ):
            continue

        work_id = str(
            meta.get("work_id")
            or meta_path.parent.name
        ).strip()

        if not work_id:
            continue

        primary_filename = str(
            meta.get("primary_filename")
            or ""
        ).strip()

        if not primary_filename:
            continue

        primary_path = (
            meta_path.parent
            / primary_filename
        )

        if not primary_path.exists():
            continue

        feedback = (
            meta.get("feedback")
            if isinstance(
                meta.get("feedback"),
                dict,
            )
            else {}
        )

        feedback_filename = str(
            feedback.get("filename")
            or ""
        ).strip()

        feedback_path = (
            meta_path.parent
            / feedback_filename
            if feedback_filename
            else None
        )

        has_feedback = bool(
            feedback_path
            and feedback_path.exists()
        )

        files: list[dict[str, Any]] = []

        raw_files = (
            meta.get("files")
            if isinstance(
                meta.get("files"),
                list,
            )
            else []
        )

        for file_meta in raw_files:
            if not isinstance(
                file_meta,
                dict,
            ):
                continue

            filename = str(
                file_meta.get("filename")
                or ""
            ).strip()

            if not filename:
                continue

            file_path = (
                meta_path.parent
                / filename
            )

            if not file_path.exists():
                continue

            files.append(
                {
                    "kind": str(
                        file_meta.get("kind")
                        or ""
                    ),
                    "label": str(
                        file_meta.get("label")
                        or filename
                    ),
                    "filename": filename,
                    "download_url": (
                        _teacher_test_work_download_url(
                            work_id,
                            filename,
                        )
                    ),
                }
            )

        source_filename = str(
            meta.get("source_filename")
            or "Teacher test upload"
        ).strip()

        work_kind = str(
            meta.get("kind")
            or "tex"
        ).strip().lower()

        item_kind = (
            "ocr"
            if work_kind == "ocr"
            else "tex"
        )

        exam_id = str(
            meta.get("exam_id")
            or ""
        ).strip()

        if not exam_id:
            exam_id = (
                f"OCR review: {source_filename}"
                if item_kind == "ocr"
                else (
                    "Uploaded work: "
                    f"{source_filename}"
                )
            )

        ocr_provider = str(
            meta.get("ocr_provider")
            or ""
        ).strip()

        grading_provider = str(
            feedback.get("provider")
            or meta.get("grading_provider")
            or ""
        ).strip()

        display_provider = (
            grading_provider
            or ocr_provider
            or "teacher test"
        )

        ocr_saved_at = str(
            meta.get("saved_at")
            or ""
        )

        graded_at = str(
            feedback.get("saved_at")
            or meta.get("graded_at")
            or ""
        )

        latest_saved_at = (
            graded_at
            or str(
                meta.get("updated_at")
                or ""
            )
            or ocr_saved_at
        )

        primary_url = (
            _teacher_test_work_download_url(
                work_id,
                primary_filename,
            )
        )

        item: dict[str, Any] = {
            "kind": item_kind,
            "status": str(
                meta.get("status")
                or ""
            ),
            "work_scope": "teacher_test",
            "work_id": work_id,
            "exam_id": exam_id,
            "exam_folder": work_id,
            "provider": display_provider,
            "ocr_provider": ocr_provider,
            "grading_provider": (
                grading_provider
            ),
            "saved_at": latest_saved_at,
            "ocr_saved_at": ocr_saved_at,
            "graded_at": graded_at,
            "source_filename": (
                source_filename
            ),
            "filename": primary_filename,
            "file_type": (
                primary_path.suffix
                .lower()
                .lstrip(".")
                or "file"
            ),
            "download_url": primary_url,
            "teacher_id": teacher_id,
            "voucher_id": str(
                meta.get("voucher_id")
                or ctx["voucher_id"]
            ),
            "storage_path_parts": (
                meta.get(
                    "storage_path_parts"
                )
                or {}
            ),
            "files": files,
            "original_download_url": next(
                (
                    entry["download_url"]
                    for entry in files
                    if entry.get("kind")
                    in {
                        "original",
                        "student_tex",
                    }
                ),
                "",
            ),
            "graded_result_json_download_url": next(
                (
                    entry["download_url"]
                    for entry in files
                    if entry.get("kind")
                    == "graded_result_json"
                ),
                "",
            ),
            "graded_result_tex_download_url": next(
                (
                    entry["download_url"]
                    for entry in files
                    if entry.get("kind")
                    == "graded_result_tex"
                ),
                "",
            ),
        }

        if has_feedback:
            item["feedback_download_url"] = (
                _teacher_test_work_download_url(
                    work_id,
                    feedback_filename,
                )
            )

            item["feedback_filename"] = (
                feedback_filename
            )

            item["feedback_file_type"] = str(
                feedback.get("file_type")
                or Path(
                    feedback_filename
                ).suffix
                .lower()
                .lstrip(".")
            )

        items.append(item)

    items.sort(
        key=lambda item: str(
            item.get("saved_at")
            or ""
        ),
        reverse=True,
    )

    return items


def list_student_work_for_dashboard(
    student_code: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_work_ids: set[str] = set()

    for root in _candidate_student_roots(student_code):
        for meta_path in root.glob("*/metadata.json"):
            meta = _read_json(meta_path)
            if not meta:
                continue

            work_id = str(meta.get("work_id") or meta_path.parent.name)
            if work_id in seen_work_ids:
                continue
            seen_work_ids.add(work_id)

            primary_filename = str(meta.get("primary_filename") or "").strip()
            if not primary_filename:
                continue

            primary_path = meta_path.parent / primary_filename
            if not primary_path.exists():
                continue

            source = str(meta.get("source_filename") or "Uploaded work")
            ocr_provider = str(meta.get("ocr_provider") or "ocr")
            ocr_saved_at = str(meta.get("saved_at") or "")
            status = str(meta.get("status") or "ocr_ready")
            feedback = meta.get("feedback") if isinstance(meta.get("feedback"), dict) else {}

            feedback_filename = str(feedback.get("filename") or "").strip()
            feedback_path = meta_path.parent / feedback_filename if feedback_filename else None
            has_feedback = bool(feedback_path and feedback_path.exists())

            display_exam_id = str(meta.get("exam_id") or "").strip()
            if not display_exam_id:
                display_exam_id = f"OCR review: {source}" if meta.get("kind") == "ocr" else f"Uploaded work: {source}"

            latest_saved_at = str(feedback.get("saved_at") or meta.get("graded_at") or ocr_saved_at)
            display_provider = str(feedback.get("provider") or meta.get("grading_provider") or ocr_provider)
            file_type = primary_path.suffix.lower().lstrip(".") or "file"

            files = []
            for f in meta.get("files") or []:
                if not isinstance(f, dict):
                    continue
                filename = str(f.get("filename") or "")
                if not filename:
                    continue
                files.append(
                    {
                        "kind": f.get("kind") or "",
                        "label": f.get("label") or filename,
                        "filename": filename,
                        "download_url": _student_work_download_url(work_id, filename),
                    }
                )

            item_kind = "ocr" if meta.get("kind") == "ocr" else "tex"

            item = {
                "kind": item_kind,
                "status": status,
                "work_id": work_id,
                "exam_id": display_exam_id,
                "exam_folder": work_id,
                "provider": display_provider,
                "ocr_provider": ocr_provider,
                "grading_provider": str(feedback.get("provider") or meta.get("grading_provider") or ""),
                "saved_at": latest_saved_at,
                "ocr_saved_at": ocr_saved_at,
                "graded_at": str(feedback.get("saved_at") or meta.get("graded_at") or ""),
                "source_filename": source,
                "filename": primary_filename,
                "file_type": file_type,
                "download_url": _student_work_download_url(work_id, primary_filename),
                "teacher_id": str(meta.get("teacher_id") or ""),
                "voucher_id": str(meta.get("voucher_id") or ""),
                "storage_path_parts": meta.get("storage_path_parts") or {},
                "original_download_url": next(
                    (
                        x["download_url"]
                        for x in files
                        if x.get("kind")
                        in {
                            "original",
                            "student_tex",
                        }
                    ),
                    "",
                ),
                "graded_result_json_download_url": next(
                    (
                        x["download_url"]
                        for x in files
                        if x.get("kind")
                        == "graded_result_json"
                    ),
                    "",
                ),
                "graded_result_tex_download_url": next(
                    (
                        x["download_url"]
                        for x in files
                        if x.get("kind")
                        == "graded_result_tex"
                    ),
                    "",
                ),
                "files": files,
            }

            if has_feedback:
                item["feedback_download_url"] = _student_work_download_url(work_id, feedback_filename)
                item["feedback_filename"] = feedback_filename
                item["feedback_file_type"] = str(
                    feedback.get("file_type") or Path(feedback_filename).suffix.lower().lstrip(".")
                )

            items.append(item)

    items.sort(key=lambda x: x.get("saved_at") or "", reverse=True)
    return items


def resolve_teacher_test_work_file(
    teacher_id: str,
    work_id: str,
    filename: str,
    *,
    voucher_id: str = "",
) -> Path:
    filename = _assert_safe_token(
        filename,
        label="filename",
    )

    work_dir = (
        _find_teacher_test_work_dir(
            teacher_id,
            work_id,
            voucher_id=voucher_id,
        )
    )

    if work_dir is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Teacher-test work folder "
                "not found."
            ),
        )

    metadata = _read_json(
        work_dir / "metadata.json"
    )

    if (
        not metadata
        or str(
            metadata.get("teacher_id")
            or ""
        ).strip()
        != str(teacher_id or "").strip()
        or str(
            metadata.get("work_scope")
            or ""
        ).strip()
        != "teacher_test"
    ):
        raise HTTPException(
            status_code=404,
            detail=(
                "Teacher-test work was "
                "not found."
            ),
        )

    root = work_dir.resolve()

    path = (
        work_dir
        / filename
    ).resolve()

    try:
        path.relative_to(root)
    except ValueError as error:
        raise HTTPException(
            status_code=403,
            detail=(
                "Invalid teacher-test "
                "work path."
            ),
        ) from error

    if (
        not path.exists()
        or not path.is_file()
    ):
        raise HTTPException(
            status_code=404,
            detail=(
                "Teacher-test work file "
                "not found."
            ),
        )

    return path


def resolve_student_work_file(student_code: str, work_id: str, filename: str) -> Path:
    work_id = _assert_safe_token(work_id, label="work_id")
    filename = _assert_safe_token(filename, label="filename")

    work_dir = _find_work_dir(student_code, work_id)
    if work_dir is None:
        raise HTTPException(status_code=404, detail="Student work folder not found.")

    root = work_dir.parent.resolve()
    path = (work_dir / filename).resolve()

    try:
        path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid student work path.")

    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Student work file not found.")

    return path