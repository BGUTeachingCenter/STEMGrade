from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF

from grader.reference_ranges import Key


@dataclass(frozen=True)
class StudentPdfPart:
    """One extracted student answer part from a PDF."""

    key: Key  # (qnum, part)
    qid: str
    page_range_1_based_inclusive: Tuple[int, int]
    text: str


# Matches headers like:
#   "Q1(א — (Answer Student"
#   "Q2(ב"
#   "Q3"
_QID_RE = re.compile(r"\bQ\s*(\d+)\s*(?:\(\s*([^\)]+?)\s*\))?", re.UNICODE)

# Hebrew letters often used for parts
_HEB_PARTS = {"א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט", "י"}


def _qid_from_key(key: Key) -> str:
    qnum, part = key
    if part:
        return f"Q{qnum}({part})"
    return f"Q{qnum}"


def _normalize_part(part: str | None) -> str:
    if not part:
        return ""
    p = str(part).strip()
    # Sometimes extraction yields "א —" or "א-" etc.
    p = re.sub(r"[^\w\u0590-\u05FF]", "", p)
    return p


def parse_student_answer_pdf(
    student_answer_pdf: Path,
    out_dir: Path,
) -> Tuple[Dict[Key, str], Dict[Key, Tuple[int, int]], Path]:
    """Parse a student answer PDF into per-question text.

    Returns:
      - answers: {Key: extracted_text}
      - ranges: {Key: (start_page_1b, end_page_1b)}
      - student_answers_json_path (written to out_dir)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(student_answer_pdf))
    try:
        pages_text: List[str] = []
        for i in range(doc.page_count):
            txt = (doc[i].get_text("text") or "").replace("\r\n", "\n").replace("\r", "\n")
            pages_text.append(txt)

        # Detect headers per page
        page_keys: List[Key | None] = []
        for txt in pages_text:
            m = _QID_RE.search(txt or "")
            if not m:
                page_keys.append(None)
                continue
            qnum = int(m.group(1))
            part = _normalize_part(m.group(2))
            if part and part not in _HEB_PARTS and len(part) > 2:
                part = part[:1]
            key: Key = (qnum, part)
            page_keys.append(key)

        # Forward-fill missing keys until next header (multi-page answers)
        current: Key | None = None
        for i in range(len(page_keys)):
            if page_keys[i] is not None:
                current = page_keys[i]
            else:
                page_keys[i] = current

        # Group pages by key
        grouped: Dict[Key, List[int]] = {}
        for i, key in enumerate(page_keys):
            if key is None:
                continue
            grouped.setdefault(key, []).append(i)

        answers: Dict[Key, str] = {}
        ranges: Dict[Key, Tuple[int, int]] = {}
        items: List[StudentPdfPart] = []

        for key, idxs in sorted(grouped.items()):
            start0, end0 = min(idxs), max(idxs)
            text = "\n\n".join((pages_text[j] or "").strip() for j in range(start0, end0 + 1)).strip()

            qid = _qid_from_key(key)
            start1, end1 = start0 + 1, end0 + 1
            answers[key] = text
            ranges[key] = (start1, end1)
            items.append(StudentPdfPart(key=key, qid=qid, page_range_1_based_inclusive=(start1, end1), text=text))

        out_json = out_dir / "student_answers.json"
        out_json.write_text(
            json.dumps(
                {
                    "version": 1,
                    "source": student_answer_pdf.name,
                    "count": len(items),
                    "items": [
                        {
                            "qid": it.qid,
                            "key": {"qnum": it.key[0], "part": it.key[1]},
                            "page_range_1_based_inclusive": list(it.page_range_1_based_inclusive),
                            "text": it.text,
                        }
                        for it in items
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return answers, ranges, out_json
    finally:
        doc.close()
