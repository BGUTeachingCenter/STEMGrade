from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_HEBREW_PARTS = list("אבגדהוזחטיכלמנסעפצקרשת")
_LATIN_PARTS = list("abcdefghijklmnopqrstuvwxyz")


QUESTION_WITH_TEXT_RE = re.compile(
    r"^\s*\.?\s*([1-9][0-9]?)(?:[.)]\s*|\s+).+",
    re.UNICODE,
)

BARE_QUESTION_RE = re.compile(
    r"^\s*([1-9][0-9]?)\s*$",
    re.UNICODE,
)

# Handles Hebrew exam extraction patterns:
#   (א)
#   (א(
#   )א
#   )א(
#   |x − 2| ≤ 5)א
HEBREW_PART_NEAR_PAREN_RE = re.compile(
    r"[\(\)]\s*([אבגדהוזחטיכלמנסעפצקרשת])\s*[\(\)]?",
    re.UNICODE,
)

# English fallback:
#   (a)
#   a)
LATIN_PART_RE = re.compile(
    r"(?:[\(\[]\s*([a-zA-Z])\s*[\)\]]|^\s*([a-zA-Z])\s*[\).:])",
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


def _looks_like_question_context(lines: list[str], idx: int) -> bool:
    """
    Bare PDF extraction often gives:
      1
      על ציר המספרים
      |x − 2| ≤ 5)א
      פתרון:

    This decides whether a bare number is probably a question number.
    """
    window = "\n".join(lines[idx + 1 : idx + 9])
    if "פתרון" in window:
        return True
    if HEBREW_PART_NEAR_PAREN_RE.search(window):
        return True
    return False


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


def _add_part(q: dict[str, Any], part: str) -> None:
    part = (part or "").strip()
    if not part:
        return
    if part not in q["parts"]:
        q["parts"].append(part)


def extract_exam_structure(reference_text: str) -> dict[str, Any]:
    """
    Extract a stable structure from a typed exam / OCR text.

    Goal:
      Question 1: א, ב, ג...
      Question 2: א, ב, ג...

    This is intentionally not used as final grading content. It is only the
    stable skeleton that later answers can be inserted into.
    """
    lines = [_clean_line(x) for x in (reference_text or "").splitlines()]
    lines = [x for x in lines if x]

    questions: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    current_q: str | None = None
    last_qnum = 0

    for idx, line in enumerate(lines):
        qid: str | None = None

        q_match = QUESTION_WITH_TEXT_RE.match(line)
        if q_match:
            possible = int(q_match.group(1))
            if possible == last_qnum + 1 or (last_qnum == 0 and possible <= 3):
                qid = str(possible)

        if qid is None:
            bare_match = BARE_QUESTION_RE.match(line)
            if bare_match:
                possible = int(bare_match.group(1))

                # Avoid dates/page numbers like 20. Prefer natural sequence.
                if (
                    (possible == last_qnum + 1 or (last_qnum == 0 and possible <= 3))
                    and _looks_like_question_context(lines, idx)
                ):
                    qid = str(possible)

        if qid is not None:
            current_q = qid
            last_qnum = int(qid)
            _add_question(questions=questions, by_id=by_id, qid=current_q)

        if current_q is None:
            continue

        q = _add_question(questions=questions, by_id=by_id, qid=current_q)

        for part in HEBREW_PART_NEAR_PAREN_RE.findall(line):
            _add_part(q, part)

        for a, b in LATIN_PART_RE.findall(line):
            part = (a or b or "").lower()
            if part in _LATIN_PARTS:
                _add_part(q, part)

    return {
        "questions": questions,
        "question_count": len(questions),
        "part_count": sum(len(q.get("parts") or []) for q in questions),
    }


def structure_to_reference_tex_template(
    structure: dict[str, Any],
    *,
    placeholder: str = "% Answer goes here.",
) -> str:
    """
    Canonical solution-bank/reference format.

    This is the format parse_reference_tex already expects:
      \\section*{Question 1}
      \\subsection*{(א)}
    """
    out: list[str] = []

    for q in structure.get("questions", []):
        qid = str(q.get("question_id", "")).strip()
        if not qid:
            continue

        out.append("")
        out.append(rf"\section*{{Question {qid}}}")

        parts = q.get("parts") or []
        if parts:
            for part in parts:
                part = str(part).strip()
                if not part:
                    continue
                out.append("")
                out.append(rf"\subsection*{{({part})}}")
                out.append(placeholder)
        else:
            out.append("")
            out.append(r"\subsection*{(a)}")
            out.append(placeholder)

    return "\n".join(out).strip()


def structure_to_student_tex_template(
    structure: dict[str, Any],
    *,
    placeholder: str = "% Paste / review OCR answer here.",
) -> str:
    """
    Student-answer format for the OCR review page.
    """
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