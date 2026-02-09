# grader/reference_ranges.py
"""Detect question/part page ranges inside a reference PDF.

The reference PDF is assumed to contain the *questions + official solutions*.
For bundling, we want, for each question (and optionally part א/ב), the
inclusive page range where that content appears.

The detection is text-based (PyMuPDF's `page.get_text('text')`). If your PDF is
scanned (no selectable text), this will fail and you should OCR the PDF first.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

# Public key type: (question_number, part_letter)
# part_letter in {"א", "ב", ""} where "" means "no part detected".
Key = Tuple[int, str]


# --- Regexes for common reference-PDF formats ---

# Style A (common in Technion/HUJI PDFs): "(נקודות35) .1" -> qnum=1
_REF_Q_POINTS_DOT_RE = re.compile(r"\(נקודות\s*\d+\)\s*\.?\s*(\d+)", re.UNICODE)

# Style B: "שאלה 1" or "Question 1"
_REF_Q_WORD_RE = re.compile(r"(?:שאלה|Question)\s*(\d+)", re.IGNORECASE | re.UNICODE)

# Part markers: "(א)" or "()א" (RTL PDFs sometimes reorder parentheses)
_REF_PART_PAREN_RE = re.compile(r"\(\s*([א-תA-Za-z])\s*\)", re.UNICODE)
_REF_PART_EMPTY_PAREN_RE = re.compile(r"\(\s*\)\s*([א-תA-Za-z])", re.UNICODE)


def _normalize_part(part: str) -> str:
    """Normalize common part markers into Hebrew א/ב."""

    p = (part or "").strip()
    if p in ("a", "A"):
        return "א"
    if p in ("b", "B"):
        return "ב"
    return p


def _extract_qnum(text: str) -> Optional[int]:
    """Extract a question number from a page's text, if present."""

    m = _REF_Q_POINTS_DOT_RE.search(text)
    if m:
        return int(m.group(1))
    m = _REF_Q_WORD_RE.search(text)
    if m:
        return int(m.group(1))
    return None


def _extract_parts(text: str) -> List[str]:
    """Extract (deduped) part letters found on a page ("א"/"ב")."""

    parts: List[str] = []
    for m in _REF_PART_EMPTY_PAREN_RE.finditer(text):
        parts.append(_normalize_part(m.group(1)))
    for m in _REF_PART_PAREN_RE.finditer(text):
        parts.append(_normalize_part(m.group(1)))

    parts = [p for p in parts if p in ("א", "ב", "ג")]
    seen = set()
    out: List[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def find_reference_ranges(reference_pdf: Path, out_dir: Path) -> Dict[Key, Tuple[int, int]]:
    """Build page ranges (1-based inclusive) for each (qnum, part).

    This function also writes `debug_reference_pages.txt` to `out_dir` with the
    extracted text per page to make it easy to tune heuristics for new PDFs.

    Args:
        reference_pdf: The (cleaned) reference PDF containing questions + solutions.
        out_dir: Output directory for debug files.

    Returns:
        Dict mapping (qnum, part) -> (start_page_1based, end_page_1based).
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(reference_pdf))
    page_keys: List[Optional[Key]] = []
    current_qnum: Optional[int] = None
    current_part: str = ""

    debug_lines: List[str] = []

    for i in range(doc.page_count):
        text = doc[i].get_text("text") or ""

        debug_lines.append(f"--- PAGE {i + 1} ---")
        debug_lines.append(text[:2500])
        debug_lines.append("")

        q = _extract_qnum(text)
        if q is not None:
            current_qnum = q
            current_part = ""  # reset when a new question begins

        if current_qnum is None:
            page_keys.append(None)
            continue

        parts = _extract_parts(text)
        if parts:
            current_part = parts[0]

        page_keys.append((current_qnum, current_part))

    (out_dir / "debug_reference_pages.txt").write_text("\n".join(debug_lines), encoding="utf-8")

    # First page where each key appears
    first_page: Dict[Key, int] = {}
    for idx, k in enumerate(page_keys):
        if k is None:
            continue
        first_page.setdefault(k, idx)

    if not first_page:
        return {}

    # Convert first-occurrence pages into contiguous ranges
    ordered = sorted(first_page.items(), key=lambda kv: kv[1])
    ranges: Dict[Key, Tuple[int, int]] = {}
    for j, (key, start0) in enumerate(ordered):
        end0 = (ordered[j + 1][1] - 1) if j + 1 < len(ordered) else (doc.page_count - 1)
        ranges[key] = (start0 + 1, end0 + 1)

    return ranges
