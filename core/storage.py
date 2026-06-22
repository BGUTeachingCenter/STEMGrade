# core/storage.py
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .config import BANK_ROOT
from services.file_handling.reference_tex import parse_reference_tex

# exam_id becomes a folder name on disk. Allow letters, numbers, spaces,
# underscore and hyphen; it must start with a letter or number (no leading
# space) and must not contain path separators or ".." (no path traversal).
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\- ]{0,80}$")


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


def exam_dir(exam_id: str) -> Path:
    return BANK_ROOT / exam_id


def uploads_dir(exam_id: str) -> Path:
    d = exam_dir(exam_id) / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


# -----------------------------
# NEW: reference summary support
# -----------------------------

def reference_tex_path(exam_id: str) -> Path:
    """
    Canonical location of the current reference file for an exam_id.
    Your upload endpoint writes reference_current.tex here.
    """
    return uploads_dir(exam_id) / "reference_current.tex"


def summary_path(exam_id: str) -> Path:
    """
    Canonical location of the precomputed summary JSON for an exam_id.
    Matching reads these instead of parsing TeX files at runtime.
    """
    return uploads_dir(exam_id) / "reference_summary.json"


def _extract_qnums_and_preview(tex_path: Path, *, max_items: int = 18) -> dict[str, Any]:
    """
    Parse reference TeX and build a compact summary.
    This is used at UPLOAD time (teacher bank), not during grading.
    """
    parts = parse_reference_tex(tex_path)  # keys might be tuple(q, part) or Key with .qnum/.part
    qnums_set = set()
    preview_items: list[str] = []

    def _key_q_part(k):
        # support both tuple and Key dataclass styles
        if isinstance(k, tuple) and len(k) == 2:
            return k[0], k[1]
        return getattr(k, "qnum", None), getattr(k, "part", "")

    for k, rp in list(parts.items())[:max_items]:
        q, part = _key_q_part(k)
        if q is not None:
            try:
                qnums_set.add(int(q))
            except Exception:
                pass
        title = re.sub(r"\s+", " ", (getattr(rp, "title", "") or "").strip())
        if q is None:
            label = "Q?"
        else:
            label = f"Q{q}{part}"
        preview_items.append(f"{label}: {title[:80]}" if title else label)

    qnums = sorted(qnums_set)

    return {
        "qnums": qnums,
        "q_count": len(set(qnums)),
        "part_count": len(parts),
        "parts_preview": preview_items,
        "filename": tex_path.name,
    }


def write_reference_summary(exam_id: str, *, tex_path: Path | None = None) -> Path:
    """
    Generate + write summary JSON for this exam_id.

    - If tex_path is not provided, uses the canonical reference_current.tex.
    - Safe to call multiple times (overwrites summary).
    """
    exam_id = require_safe_exam_id(exam_id)

    if tex_path is None:
        tex_path = reference_tex_path(exam_id)

    if not tex_path.exists():
        raise FileNotFoundError(f"reference TeX not found: {tex_path}")

    summary = _extract_qnums_and_preview(tex_path)
    payload = {
        "exam_id": exam_id,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **summary,
    }

    p = summary_path(exam_id)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def read_reference_summary(exam_id: str) -> dict[str, Any] | None:
    """Read summary JSON if present, else None."""
    exam_id = require_safe_exam_id(exam_id)
    p = summary_path(exam_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_exam_summaries() -> list[dict[str, Any]]:
    """
    Return summaries for all exams that have reference_summary.json.
    Useful for fast matching (no TeX parsing).
    """
    out: list[dict[str, Any]] = []
    if not BANK_ROOT.exists():
        return out

    for d in BANK_ROOT.iterdir():
        if not d.is_dir():
            continue
        p = d / "uploads" / "reference_summary.json"
        if p.exists():
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                # ignore broken summary files
                pass

    # stable ordering
    out.sort(key=lambda x: (x.get("exam_id") or ""))
    return out


def ensure_reference_summary(exam_id: str) -> Path | None:
    """
    Ensure summary exists for exam_id.
    If missing but reference_current.tex exists, generate it.
    Returns summary path if exists/created, else None.
    """
    exam_id = require_safe_exam_id(exam_id)
    sp = summary_path(exam_id)
    if sp.exists():
        return sp

    refp = reference_tex_path(exam_id)
    if refp.exists():
        try:
            return write_reference_summary(exam_id, tex_path=refp)
        except Exception:
            return None

    return None