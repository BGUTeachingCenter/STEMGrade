from __future__ import annotations

import json
import re
import secrets
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import HTTPException

from core.config import PROJECT_ROOT

STUDENT_WORK_ROOT = PROJECT_ROOT / "data" / "student_work"
STUDENT_WORK_ROOT.mkdir(parents=True, exist_ok=True)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,180}$")


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

    Target path:
      data/student_work/<teacher_id>/<voucher_id>/<student_code>/<work_id>/

    If the student-code record has voucher_id, we use it.
    Otherwise we fall back to the teacher profile voucher_id.
    Otherwise we use voucher_default.
    """
    raw_student_code = (student_code or "").strip()
    raw_teacher_id = (teacher_id or "").strip()
    raw_voucher_id = (voucher_id or "").strip()

    record: dict[str, Any] = {}
    if raw_student_code:
        try:
            from services.student_access import get_student_code_record

            record = get_student_code_record(raw_student_code) or {}
        except Exception:
            record = {}

    if not raw_teacher_id:
        raw_teacher_id = str(record.get("teacher_id") or "").strip()

    if not raw_voucher_id:
        raw_voucher_id = str(
            record.get("voucher_id")
            or record.get("voucher_code")
            or record.get("voucher_hash")
            or ""
        ).strip()

    if not raw_voucher_id and raw_teacher_id:
        try:
            from services.teacher_profiles import get_teacher

            teacher = get_teacher(raw_teacher_id) or {}
            raw_voucher_id = str(
                teacher.get("voucher_id")
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
        "safe_student_code": _safe_student_code(raw_student_code),
        "safe_teacher_id": _safe_teacher_id(raw_teacher_id),
        "safe_voucher_id": _safe_voucher_id(raw_voucher_id),
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
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
    root = (
        STUDENT_WORK_ROOT
        / ctx["safe_teacher_id"]
        / ctx["safe_voucher_id"]
        / ctx["safe_student_code"]
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


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
    roots = [
        _student_root(
            student_code,
            teacher_id=teacher_id,
            voucher_id=voucher_id,
        )
    ]

    legacy = _legacy_student_root(student_code)
    if legacy not in roots and legacy.exists():
        roots.append(legacy)

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


def _assert_safe_token(value: str, *, label: str) -> str:
    value = (value or "").strip()
    if not value or "/" in value or "\\" in value or ".." in value:
        raise HTTPException(status_code=400, detail=f"Invalid {label}.")
    if not _SAFE_TOKEN_RE.match(value):
        raise HTTPException(status_code=400, detail=f"Invalid {label}.")
    return value


def save_ocr_student_work(
    *,
    student_code: str,
    teacher_id: str = "",
    voucher_id: str = "",
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
    student_code = (student_code or "").strip()
    if not student_code:
        raise HTTPException(status_code=400, detail="Missing student code for OCR storage.")

    ctx = _lookup_student_work_context(
        student_code,
        teacher_id=teacher_id,
        voucher_id=voucher_id,
    )

    work_id = _new_work_id("ocr")
    work_dir = _student_root(
        student_code,
        teacher_id=ctx["teacher_id"],
        voucher_id=ctx["voucher_id"],
    ) / work_id
    work_dir.mkdir(parents=True, exist_ok=False)

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

    if student_summary_path and Path(student_summary_path).exists():
        shutil.copy2(student_summary_path, work_dir / "student_summary.json")
        files.append(
            {
                "kind": "student_summary",
                "filename": "student_summary.json",
                "label": "Student summary JSON",
            }
        )

    metadata = {
        "schema_version": "student_work_v1",
        "kind": "ocr",
        "status": "ocr_ready",
        "work_id": work_id,
        "student_code": student_code,
        "teacher_id": ctx["teacher_id"],
        "voucher_id": ctx["voucher_id"],
        "storage_schema": "teacher_voucher_student_work_v1",
        "storage_path_parts": {
            "teacher": ctx["safe_teacher_id"],
            "voucher": ctx["safe_voucher_id"],
            "student": ctx["safe_student_code"],
        },
        "source_filename": source_filename or "",
        "saved_at": _now(),
        "ocr_provider": ocr_provider or "",
        "ocr_model": ocr_model or "",
        "document_type": document_type or "",
        "ocr_path": ocr_path or "",
        "route_reason": route_reason or "",
        "debug_trace_id": debug_trace_id or "",
        "debug_trace_dir": debug_trace_dir or "",
        "primary_filename": primary_filename,
        "files": files,
    }

    _write_json(work_dir / "metadata.json", metadata)
    return metadata


def create_tex_student_work(
    *,
    student_code: str,
    teacher_id: str = "",
    voucher_id: str = "",
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
    student_code = (student_code or "").strip()
    if not student_code:
        raise HTTPException(status_code=400, detail="Missing student code for student-work storage.")

    ctx = _lookup_student_work_context(
        student_code,
        teacher_id=teacher_id,
        voucher_id=voucher_id,
    )

    work_id = _new_work_id("tex")
    work_dir = _student_root(
        student_code,
        teacher_id=ctx["teacher_id"],
        voucher_id=ctx["voucher_id"],
    ) / work_id
    work_dir.mkdir(parents=True, exist_ok=False)

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

    metadata = {
        "schema_version": "student_work_v1",
        "storage_schema": "teacher_voucher_student_work_v1",
        "kind": "tex",
        "status": "uploaded",
        "work_id": work_id,
        "student_code": student_code,
        "teacher_id": ctx["teacher_id"],
        "voucher_id": ctx["voucher_id"],
        "storage_path_parts": {
            "teacher": ctx["safe_teacher_id"],
            "voucher": ctx["safe_voucher_id"],
            "student": ctx["safe_student_code"],
        },
        "source_filename": source_filename or "",
        "saved_at": _now(),
        "debug_trace_id": debug_trace_id or "",
        "debug_trace_dir": debug_trace_dir or "",
        "primary_filename": original_name,
        "files": files,
    }

    _write_json(work_dir / "metadata.json", metadata)
    return metadata


def _student_work_download_url(work_id: str, filename: str) -> str:
    return (
        "/routes/student/work_file"
        f"?work_id={quote(work_id)}"
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
    student_code: str,
    work_id: str,
    exam_id: str,
    provider: str,
    final_path: Path,
    saved_at: str | None = None,
) -> dict[str, Any] | None:
    """
    Attach a grading result to an existing OCR work item.

    After this, the dashboard can render one card:
      original upload + OCR result + graded feedback
    """
    student_code = (student_code or "").strip()
    if not student_code:
        return None

    work_id = _assert_safe_token(work_id, label="work_id")
    work_dir = _find_work_dir(student_code, work_id)

    if work_dir is None:
        return None

    meta_path = work_dir / "metadata.json"
    meta = _read_json(meta_path)
    if not meta:
        return None

    source = Path(final_path)
    if not source.exists() or not source.is_file():
        return None

    suffix = source.suffix.lower() or ".pdf"
    exam_part = _safe_name(exam_id or "graded_work", "graded_work")
    provider_part = _safe_name(provider or "ai", "ai")
    feedback_filename = f"graded_{exam_part}_{provider_part}{suffix}"
    feedback_path = work_dir / feedback_filename

    # Avoid Windows overwrite conflicts if the same OCR item is graded again.
    if feedback_path.exists():
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        feedback_filename = f"graded_{exam_part}_{provider_part}_{ts}{suffix}"
        feedback_path = work_dir / feedback_filename

    shutil.copy2(source, feedback_path)

    feedback_saved_at = saved_at or _now()
    feedback = {
        "exam_id": exam_id or "",
        "provider": provider or "",
        "filename": feedback_filename,
        "file_type": suffix.lstrip("."),
        "saved_at": feedback_saved_at,
    }

    files = meta.get("files") if isinstance(meta.get("files"), list) else []
    files = _upsert_file_entry(
        files,
        {
            "kind": "graded_feedback",
            "filename": feedback_filename,
            "label": "Graded feedback",
        },
    )

    meta["status"] = "graded"
    meta["exam_id"] = exam_id or meta.get("exam_id") or ""
    meta["graded_at"] = feedback_saved_at
    meta["grading_provider"] = provider or ""
    meta["feedback"] = feedback
    meta["files"] = files
    meta["updated_at"] = _now()

    _write_json(meta_path, meta)
    return meta


def list_teacher_student_work_metadata(teacher_id: str) -> list[dict[str, Any]]:
    """
    Return canonical student-work metadata for a teacher, across vouchers/courses.

    Scans the current layout:
      data/student_work/<teacher>/<voucher>/<student>/<work_id>/metadata.json

    The function intentionally returns metadata only, not file contents, so it is
    safe to use for teacher dashboard statistics and Excel exports.
    """
    safe_teacher_id = _safe_teacher_id(teacher_id)
    teacher_root = STUDENT_WORK_ROOT / safe_teacher_id
    if not teacher_root.exists():
        return []

    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for meta_path in teacher_root.glob("*/*/*/metadata.json"):
        meta = _read_json(meta_path)
        if not meta:
            continue

        try:
            rel = meta_path.relative_to(teacher_root)
            voucher_part = rel.parts[0] if len(rel.parts) >= 4 else ""
            student_part = rel.parts[1] if len(rel.parts) >= 4 else ""
            work_part = rel.parts[2] if len(rel.parts) >= 4 else meta_path.parent.name
        except Exception:
            voucher_part = ""
            student_part = ""
            work_part = meta_path.parent.name

        student_code = str(meta.get("student_code") or student_part or "").strip()
        voucher_id = str(meta.get("voucher_id") or voucher_part or "voucher_default").strip()
        work_id = str(meta.get("work_id") or work_part or "").strip()

        key = (voucher_id, student_code, work_id)
        if key in seen:
            continue
        seen.add(key)

        item = dict(meta)
        item["teacher_id"] = str(item.get("teacher_id") or teacher_id or "").strip()
        item["voucher_id"] = voucher_id
        item["student_code"] = student_code
        item["work_id"] = work_id
        item["metadata_saved_at"] = str(item.get("saved_at") or "")
        item["metadata_updated_at"] = str(item.get("updated_at") or "")
        item["storage_path_parts"] = item.get("storage_path_parts") or {
            "teacher": safe_teacher_id,
            "voucher": voucher_part,
            "student": student_part,
        }
        items.append(item)

    items.sort(key=lambda x: str(x.get("graded_at") or x.get("saved_at") or ""), reverse=True)
    return items


def list_student_work_for_dashboard(student_code: str) -> list[dict[str, Any]]:
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
                    (x["download_url"] for x in files if x.get("kind") in {"original", "student_tex"}),
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