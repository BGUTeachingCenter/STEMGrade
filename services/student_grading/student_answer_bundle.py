# services/student_grading/student_answer_bundle.py
"""
Read a student_answer_bundle.json into the (question_id, part) -> answer_text
mapping used by the grading payload builder.

The bundle is produced by the OCR "student_answer_bundle" prompt variant and
has the shape:

    {
      "student_name": "",
      "student_id": "",
      "exam_id": "",
      "answers": [
        {"question_id": 1, "part_key": "a", "answer_text": "...", ...}
      ],
      "unmatched_blocks": [],
      "warnings": []
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from common.tex.reference_ranges import Key


def _coerce_qnum(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def load_answer_bundle(path: Path | str) -> dict:
    """Load and JSON-parse the bundle. Raises on missing/invalid file."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"student_answer_bundle is not a JSON object: {p}")
    return data


def read_student_answers_from_bundle(path: Path | str) -> Dict[Key, str]:
    """
    Return {(question_id, part_key): answer_text} from a student_answer_bundle.json.

    - Entries without a usable integer question_id are skipped.
    - part_key is kept as-is (caller normalizes); missing part -> "".
    - Multiple entries for the same (q, part) are concatenated in order.
    """
    data = load_answer_bundle(path)

    answers: Dict[Key, str] = {}
    for item in data.get("answers") or []:
        if not isinstance(item, dict):
            continue

        qnum = _coerce_qnum(item.get("question_id"))
        if qnum is None:
            continue

        part = str(item.get("part_key") or "").strip()
        text = str(item.get("answer_text") or "").strip()

        key: Key = (qnum, part)
        if key in answers and text:
            answers[key] = f"{answers[key]}\n{text}".strip()
        elif key not in answers:
            answers[key] = text

    return answers
