# common/exam_summary.py
"""
Shared exam-summary utilities.

This module centralizes the small helpers used to:
  * summarize a student's parsed answers.tex      (write_student_summary)
  * summarize a teacher reference solution         (build_reference_summary_*)
  * compare a student summary against a reference  (compare_summaries)

These were previously inlined in routes/grading_routes.py. They are kept
provider/exam agnostic so they work for any submission, not a specific file.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from common.tex.student_tex import parse_student_tex_answers
from schemas.reference_bundle import FullSolutionBundle


# -------------------------
# Generic helpers
# -------------------------

def _safe_str(x: Any, limit: int = 200) -> str:
    s = str(x) if x is not None else ""
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


def read_json_file(path: Path | str) -> dict | None:
    """Read+parse a JSON file. Return None on any error (missing/invalid)."""
    try:
        p = Path(path)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


# -------------------------
# Student summary
# -------------------------

def _fallback_student_qnums(tex_path: Path) -> list[int]:
    """Best-effort question-number detection from raw TeX text."""
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    found: set[int] = set()

    for m in re.finditer(r"\b(?:Question|Q)\s*([0-9]{1,3})\b", text, flags=re.IGNORECASE):
        found.add(int(m.group(1)))

    for m in re.finditer(r"\bשאלה\s*([0-9]{1,3})\b", text):
        found.add(int(m.group(1)))

    return sorted(found)


def _extract_student_summary(student_tex_path: Path, *, out_dir: Path) -> dict:
    """Parse a student answers.tex into a compact, JSON-serializable summary."""
    summary: dict[str, Any] = {
        "qnums": [],
        "keys_preview": [],
        "preview_text": "",
        "part_count": 0,
        "parse_ok": False,
    }

    try:
        student_answers, _ranges = parse_student_tex_answers(student_tex_path, out_dir)
        if student_answers:
            qnums = sorted({q for (q, _part) in student_answers.keys()})
            keys = []
            preview_lines = []

            for (q, part), body in list(student_answers.items())[:14]:
                part_str = (part or "").strip()
                key = f"Q{q}{part_str}"
                keys.append(key)

                snip = _safe_str(body, limit=140)
                preview_lines.append(f"{key}: {snip if snip else '(empty)'}")

            summary.update(
                {
                    "qnums": qnums,
                    "keys_preview": keys[:60],
                    "preview_text": "\n".join(preview_lines)[:6000],
                    "part_count": len(student_answers),
                    "parse_ok": True,
                }
            )
            return summary

    except Exception as e:
        summary["parse_error"] = _safe_str(e, 500)

    qnums = _fallback_student_qnums(student_tex_path)
    summary.update(
        {
            "qnums": qnums,
            "keys_preview": [f"Q{q}" for q in qnums][:60],
            "part_count": len(qnums),
            "preview_text": "(fallback qnum detection)",
            "parse_ok": False,
        }
    )
    return summary


def write_student_summary(student_tex_path: Path, *, out_dir: Path) -> Path:
    """Write student_summary.json next to the run output. Returns its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "filename": student_tex_path.name,
        **_extract_student_summary(student_tex_path, out_dir=out_dir),
    }
    p = out_dir / "student_summary.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# -------------------------
# Reference summary
# -------------------------

def _part_label(part: str, part_key: str) -> str:
    return str(part_key or "").strip() or str(part or "").strip()


def build_reference_summary_from_full_solution_json(
    path: Path, *, exam_id: str = ""
) -> dict:
    """
    Build a reference summary from a full_solution_bundle.json file.

    Output shape is what solution_bank_matcher consumes:
      exam_id, source, qnums, part_keys, parts_preview, updated_at
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    bundle = FullSolutionBundle.model_validate(raw)

    exam_id = (exam_id or bundle.exam_id or "").strip()

    qnums: list[int] = []
    part_keys: list[str] = []
    parts_preview: list[str] = []

    for q in sorted(bundle.questions, key=lambda x: x.question_id):
        qnums.append(int(q.question_id))
        for p in q.parts:
            label = _part_label(p.part, p.part_key)
            key = f"Q{q.question_id}{label}"
            part_keys.append(key)
            snip = _safe_str(p.question_text or p.required_action, limit=120)
            parts_preview.append(f"{key}: {snip}" if snip else key)

    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "exam_id": exam_id,
        "exam_title": bundle.exam_title or "",
        "source": "full_solution_bundle.json",
        "qnums": sorted(set(qnums)),
        "part_keys": part_keys[:120],
        "parts_preview": parts_preview[:60],
    }


def build_reference_summary_from_tex(path: Path, *, exam_id: str = "") -> dict:
    """
    Legacy fallback: build a reference summary from reference_current.tex.

    Only question numbers can be reliably recovered from display TeX, so
    part-level data is left empty.
    """
    tex_path = Path(path)
    qnums = _fallback_student_qnums(tex_path)

    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "exam_id": (exam_id or "").strip(),
        "exam_title": "",
        "source": "reference_current.tex",
        "qnums": qnums,
        "part_keys": [],
        "parts_preview": [],
    }


# -------------------------
# Comparison
# -------------------------

def _as_int_set(values: Any) -> set[int]:
    out: set[int] = set()
    for v in values or []:
        try:
            out.add(int(v))
        except (TypeError, ValueError):
            continue
    return out


def compare_summaries(
    *,
    student_summary: dict,
    reference_summary: dict,
    exam_id: str,
    reference_path: str,
    match_method: str,
    match_reason: str,
) -> dict:
    """
    Compare a student summary against a reference summary.

    Returns a JSON-serializable dict with overlap metrics and warnings,
    used for debug traces and summary_compare.json.
    """
    student_summary = student_summary or {}
    reference_summary = reference_summary or {}

    student_qnums = _as_int_set(student_summary.get("qnums"))
    reference_qnums = _as_int_set(reference_summary.get("qnums"))

    qnum_overlap = sorted(student_qnums & reference_qnums)
    qnum_missing = sorted(student_qnums - reference_qnums)

    student_parts = {str(k).strip() for k in (student_summary.get("keys_preview") or []) if str(k).strip()}
    reference_parts = {str(k).strip() for k in (reference_summary.get("part_keys") or []) if str(k).strip()}
    part_overlap = sorted(student_parts & reference_parts)

    warnings: list[str] = []
    if not student_qnums:
        warnings.append("No question numbers detected in student submission.")
    if not reference_qnums:
        warnings.append("No question numbers found in reference summary.")
    if student_qnums and not qnum_overlap:
        warnings.append("Student and reference share no question numbers.")
    if qnum_missing:
        warnings.append(
            "Student questions not present in reference: "
            + ", ".join(str(q) for q in qnum_missing)
        )

    status = "ok" if qnum_overlap else "warning"

    return {
        "status": status,
        "exam_id": str(exam_id),
        "reference_path": str(reference_path),
        "match_method": match_method,
        "match_reason": match_reason,
        "overlap": {
            "qnum_overlap": qnum_overlap,
            "qnum_overlap_count": len(qnum_overlap),
            "qnum_missing": qnum_missing,
            "part_overlap": part_overlap,
            "part_overlap_count": len(part_overlap),
        },
        "student_qnums": sorted(student_qnums),
        "reference_qnums": sorted(reference_qnums),
        "warnings": warnings,
    }
