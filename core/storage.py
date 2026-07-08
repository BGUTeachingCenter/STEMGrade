# core/storage.py
from __future__ import annotations

import contextvars
import json
import re
import shutil
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .config import BANK_ROOT, TEACHER_BANK_ROOT, TEACHERS_ROOT
from common.exam_summary import (
    build_reference_summary_from_full_solution_json,
    build_reference_summary_from_tex,
)

# exam_id becomes a folder name on disk. Allow letters, numbers, spaces,
# underscore and hyphen; it must start with a letter or number (no leading
# space) and must not contain path separators or ".." (no path traversal).
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\- ]{0,80}$")


_SAFE_TEACHER_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,80}$")

_ACTIVE_BANK_ROOT: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "mathgrade_active_bank_root",
    default=None,
)


def require_safe_teacher_id(teacher_id: str) -> str:
    """Validate teacher_id before using it as a folder name."""
    teacher_id = (teacher_id or "").strip()
    if not _SAFE_TEACHER_ID_RE.match(teacher_id):
        raise HTTPException(status_code=400, detail="Invalid teacher_id.")
    return teacher_id


def _safe_course_id(value: str) -> str:
    value = (value or "").strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return value[:120] or "voucher_default"


def _default_voucher_for_teacher(teacher_id: str) -> str:
    try:
        from services.teacher_profiles import get_teacher

        teacher = get_teacher(teacher_id) or {}
        return str(
            teacher.get("voucher_id")
            or teacher.get("voucher_hash")
            or teacher.get("voucher_code")
            or "voucher_default"
        ).strip() or "voucher_default"
    except Exception:
        return "voucher_default"


def teacher_course_root(teacher_id: str, voucher_id: str = "") -> Path:
    """
    Canonical course root:

      data/teachers/<teacher_id>/courses/<voucher_id>/
    """
    teacher_id = require_safe_teacher_id(teacher_id)
    safe_voucher = _safe_course_id(voucher_id or _default_voucher_for_teacher(teacher_id))

    d = TEACHERS_ROOT / teacher_id / "courses" / safe_voucher
    d.mkdir(parents=True, exist_ok=True)
    return d


def teacher_bank_root(teacher_id: str, voucher_id: str = "") -> Path:
    """
    Canonical isolated solution bank root for one teacher/course:

      data/teachers/<teacher_id>/courses/<voucher_id>/solution_bank/

    Backward compatibility:
      if data/teacher_banks/<teacher_id>/ exists and the new course bank is empty,
      copy the old bank into the new course bank once.
    """
    teacher_id = require_safe_teacher_id(teacher_id)
    course_root = teacher_course_root(teacher_id, voucher_id=voucher_id)
    d = course_root / "solution_bank"
    d.mkdir(parents=True, exist_ok=True)

    legacy = TEACHER_BANK_ROOT / teacher_id
    try:
        is_empty = not any(d.iterdir())
    except Exception:
        is_empty = False

    if legacy.exists() and legacy.is_dir() and is_empty:
        try:
            shutil.copytree(legacy, d, dirs_exist_ok=True)
        except Exception:
            pass

    return d


def bind_bank_root(bank_root: Path | str | None) -> None:
    """
    Set the active bank root for the current request/task.

    Existing call sites can keep using uploads_dir(exam_id), exam_dir(exam_id),
    write_reference_summary(exam_id), etc.; they will resolve through this
    active context.
    """
    if bank_root is None:
        _ACTIVE_BANK_ROOT.set(None)
        return
    root = Path(bank_root)
    root.mkdir(parents=True, exist_ok=True)
    _ACTIVE_BANK_ROOT.set(root)


def current_bank_root(bank_root: Path | str | None = None) -> Path:
    if bank_root is not None:
        root = Path(bank_root)
        root.mkdir(parents=True, exist_ok=True)
        return root

    root = _ACTIVE_BANK_ROOT.get()
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        return root

    BANK_ROOT.mkdir(parents=True, exist_ok=True)
    return BANK_ROOT


def require_safe_exam_id(exam_id: str) -> str:
    """Validate bank exam_id to avoid path traversal and weird names."""
    exam_id = (exam_id or "").strip()
    if not _SAFE_NAME_RE.match(exam_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid exam_id. Use letters, numbers, spaces, _ or - only.",
        )
    return exam_id


def require_safe_filename(name: str) -> str:
    """Validate filename for safe access within uploads folder."""
    name = (name or "").strip()
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not name.lower().endswith(".tex"):
        raise HTTPException(status_code=400, detail="Filename must end with .tex")
    return name


def exam_dir(exam_id: str, *, bank_root: Path | str | None = None) -> Path:
    return current_bank_root(bank_root) / exam_id


def uploads_dir(exam_id: str, *, bank_root: Path | str | None = None) -> Path:
    d = exam_dir(exam_id, bank_root=bank_root) / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


# -----------------------------
# Per-question reservoir
# -----------------------------
# Alongside the complete full_solution_bundle.json we also save one JSON file
# per question, so downstream operations (regrading a single question, reusing a
# question, editing one solution) can read/write questions individually without
# touching the whole bundle.

def reservoir_dir(exam_id: str, *, bank_root: Path | str | None = None) -> Path:
    d = uploads_dir(exam_id, bank_root=bank_root) / "questions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _question_filename(question_id: Any) -> str:
    """Safe, stable filename for one question, e.g. q1.json."""
    token = re.sub(r"[^A-Za-z0-9_\-]", "_", str(question_id)).strip("_") or "unknown"
    return f"q{token}.json"


def write_question_reservoir(exam_id: str, full_bundle: dict[str, Any]) -> list[Path]:
    """
    Write one JSON file per question from a full_solution_bundle dict, plus an
    index.json, into solution_bank/<exam_id>/uploads/questions/.

    Each per-question file is self-contained: it carries the exam context
    (exam_id, exam_title, source_names) together with the single question, so it
    can be consumed on its own.

    Returns the list of per-question file paths written. Stale question files
    from a previous upload are removed so the reservoir mirrors the current
    bundle exactly.
    """
    exam_id = require_safe_exam_id(exam_id)
    rdir = reservoir_dir(exam_id)

    exam_title = str(full_bundle.get("exam_title") or "")
    source_names = list(full_bundle.get("source_names") or [])
    questions = full_bundle.get("questions") or []

    written: list[Path] = []
    index_entries: list[dict[str, Any]] = []
    keep_names: set[str] = {"index.json"}

    for question in questions:
        if not isinstance(question, dict):
            continue

        question_id = question.get("question_id")
        filename = _question_filename(question_id)

        # Avoid clobbering when two questions share an id (keep first, suffix rest).
        if filename in keep_names:
            base = filename[:-5]
            n = 2
            while f"{base}__{n}.json" in keep_names:
                n += 1
            filename = f"{base}__{n}.json"

        parts = question.get("parts") or []
        payload = {
            "schema_version": "reservoir_question_v1",
            "exam_id": exam_id,
            "exam_title": exam_title,
            "question_id": question_id,
            "parts": parts,
            "source_names": source_names,
        }

        path = rdir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        written.append(path)
        keep_names.add(filename)
        index_entries.append(
            {
                "question_id": question_id,
                "file": filename,
                "part_count": len(parts),
                "part_keys": [
                    str(p.get("part_key") or p.get("part") or "")
                    for p in parts
                    if isinstance(p, dict)
                ],
            }
        )

    index = {
        "schema_version": "reservoir_index_v1",
        "exam_id": exam_id,
        "exam_title": exam_title,
        "question_count": len(index_entries),
        "questions": index_entries,
        "source_names": source_names,
    }
    (rdir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Remove stale question files that are no longer part of the bundle.
    for existing in rdir.glob("q*.json"):
        if existing.name not in keep_names:
            try:
                existing.unlink()
            except OSError:
                pass

    return written


def read_question_reservoir(exam_id: str) -> list[dict[str, Any]]:
    """Read all per-question reservoir files for an exam, ordered by the index."""
    exam_id = require_safe_exam_id(exam_id)
    rdir = uploads_dir(exam_id) / "questions"
    if not rdir.exists():
        return []

    index_path = rdir / "index.json"
    order: list[str] = []
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            order = [e.get("file") for e in (index.get("questions") or []) if e.get("file")]
        except Exception:
            order = []

    files = order or sorted(p.name for p in rdir.glob("q*.json"))

    out: list[dict[str, Any]] = []
    for name in files:
        p = rdir / name
        if not p.exists():
            continue
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return out


# -----------------------------
# Reference summary support
# -----------------------------

def reference_tex_path(exam_id: str, *, bank_root: Path | str | None = None) -> Path:
    """
    Canonical display/export TeX for an exam_id.

    Grading should prefer full_solution_bundle.json. This TeX is kept for
    preview/PDF/export and legacy fallback only.
    """
    return uploads_dir(exam_id, bank_root=bank_root) / "reference_current.tex"


def reference_json_path(exam_id: str, *, bank_root: Path | str | None = None) -> Path:
    """Canonical structured grading reference for an exam_id."""
    return uploads_dir(exam_id, bank_root=bank_root) / "full_solution_bundle.json"


def summary_path(exam_id: str, *, bank_root: Path | str | None = None) -> Path:
    """
    Canonical location of the precomputed summary JSON for an exam_id.
    Matching reads these instead of parsing reference files at runtime.
    """
    return uploads_dir(exam_id, bank_root=bank_root) / "reference_summary.json"


def write_reference_summary(
    exam_id: str,
    *,
    tex_path: Path | None = None,
    json_path: Path | None = None,
    bank_root: Path | str | None = None,
) -> Path:
    """
    Generate + write reference_summary.json for this exam_id.

    Preferred source:
      full_solution_bundle.json  (same structured source used for grading)

    Legacy fallback:
      reference_current.tex      (display/export format only)

    Safe to call multiple times (overwrites summary).
    """
    exam_id = require_safe_exam_id(exam_id)

    if json_path is None:
        json_path = reference_json_path(exam_id, bank_root=bank_root)
    if tex_path is None:
        tex_path = reference_tex_path(exam_id, bank_root=bank_root)

    if json_path.exists():
        payload = build_reference_summary_from_full_solution_json(json_path, exam_id=exam_id)
    elif tex_path.exists():
        payload = build_reference_summary_from_tex(tex_path, exam_id=exam_id)
    else:
        raise FileNotFoundError(
            f"No reference source found for {exam_id}. Expected {json_path} or {tex_path}"
        )

    p = summary_path(exam_id, bank_root=bank_root)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def read_reference_summary(exam_id: str, *, bank_root: Path | str | None = None) -> dict[str, Any] | None:
    """Read summary JSON if present, else None."""
    exam_id = require_safe_exam_id(exam_id)
    p = summary_path(exam_id, bank_root=bank_root)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_exam_summaries(*, bank_root: Path | str | None = None) -> list[dict[str, Any]]:
    """
    Return summaries for all exams that have reference_summary.json.
    Useful for fast matching (no TeX parsing).
    """
    out: list[dict[str, Any]] = []
    root = current_bank_root(bank_root)

    if not root.exists():
        return out

    for d in root.iterdir():
        if not d.is_dir():
            continue

        p = d / "uploads" / "reference_summary.json"
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    out.append(data)
            except Exception:
                pass

    out.sort(key=lambda x: (x.get("exam_id") or ""))
    return out


def ensure_reference_summary(exam_id: str, *, bank_root: Path | str | None = None) -> Path | None:
    """
    Ensure summary exists for exam_id.

    If missing, prefer full_solution_bundle.json. Fall back to reference_current.tex
    for legacy exams.
    """
    exam_id = require_safe_exam_id(exam_id)
    sp = summary_path(exam_id, bank_root=bank_root)
    if sp.exists():
        return sp

    try:
        return write_reference_summary(exam_id, bank_root=bank_root)
    except Exception:
        return None
