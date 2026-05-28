from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_HEBREW_PARTS = set("אבגדהוזחטיכלמנסעפצקרשת")
_LATIN_PARTS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Handles lines like:
# .1 מצאו...
# 1 מצאו...
# 1. מצאו...
QUESTION_LINE_RE = re.compile(
    r"^\s*\.?\s*([0-9]{1,2})(?:\s+|[.)])",
    re.UNICODE,
)

# Handles part labels like:
# (א)
# (ב(
# (a)
# [א]
PART_PAREN_RE = re.compile(
    r"[\(\[]\s*([א-תa-zA-Z])\s*[\)\]]|[\(\[]\s*([א-תa-zA-Z])\s*$",
    re.UNICODE,
)

# Handles OCR weirdness like:
# א)
# ב.
# ג:
PART_STANDALONE_RE = re.compile(
    r"^\s*([א-תa-zA-Z])\s*[\)\].:]",
    re.UNICODE,
)


def _clean_line(line: str) -> str:
    return (
        (line or "")
        .replace("\u200f", "")
        .replace("\u200e", "")
        .replace("\xa0", " ")
        .strip()
    )


def _valid_part(label: str) -> bool:
    label = (label or "").strip()
    return label in _HEBREW_PARTS or label in _LATIN_PARTS


def _add_question(
    *,
    questions: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    qid: str,
) -> dict[str, Any]:
    if qid not in by_id:
        item = {"question_id": qid, "parts": []}
        by_id[qid] = item
        questions.append(item)
    return by_id[qid]


def extract_exam_structure(reference_text: str) -> dict[str, Any]:
    """
    Extract stable question/part structure from a typed exam form or OCR text.

    This is intentionally based on the exam form, not on student handwriting.
    """
    questions: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    current_q: str | None = None

    for raw in (reference_text or "").splitlines():
        line = _clean_line(raw)
        if not line:
            continue

        q_match = QUESTION_LINE_RE.match(line)
        if q_match:
            current_q = q_match.group(1)
            _add_question(questions=questions, by_id=by_id, qid=current_q)

        if current_q is None:
            continue

        candidate_parts: list[str] = []

        for a, b in PART_PAREN_RE.findall(line):
            part = a or b
            if _valid_part(part):
                candidate_parts.append(part)

        standalone = PART_STANDALONE_RE.match(line)
        if standalone and _valid_part(standalone.group(1)):
            candidate_parts.append(standalone.group(1))

        if candidate_parts:
            q = _add_question(questions=questions, by_id=by_id, qid=current_q)
            for part in candidate_parts:
                if part not in q["parts"]:
                    q["parts"].append(part)

    return {
        "questions": questions,
        "question_count": len(questions),
        "part_count": sum(len(q.get("parts") or []) for q in questions),
    }


def structure_to_student_tex_template(
    structure: dict[str, Any],
    *,
    placeholder: str = "% Paste / review OCR answer here.",
) -> str:
    out: list[str] = []

    for q in structure.get("questions", []):
        qid = str(q.get("question_id", "")).strip()
        if not qid:
            continue

        out.append("")
        out.append(rf"\subsection*{{Question {qid}}}")

        parts = q.get("parts") or []
        if parts:
            for part in parts:
                part = str(part).strip()
                if not part:
                    continue
                out.append("")
                out.append(rf"\textbf{{({part})}}")
                out.append(placeholder)
        else:
            out.append("")
            out.append(placeholder)

    return "\n".join(out).strip()


def save_exam_structure(reference_text: str, out_path: Path) -> dict[str, Any]:
    structure = extract_exam_structure(reference_text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(structure, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return structure


def load_exam_structure(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))