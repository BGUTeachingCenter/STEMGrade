from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF


@dataclass
class BundleQuestionText:
    """Text extracted for a single question from the bundle."""
    qid: str                 # e.g. "Q1(א)"
    reference_solution: str  # official question+solution text
    student_answer: str      # student's rendered answer text


_QID_RE = re.compile(r"\bQ\s*(\d+)\s*\(\s*([^)]+)\s*\)", re.IGNORECASE)


def extract_bundle_text(bundle_pdf: Path) -> str:
    """
    Extract all text from the bundle PDF into one string.
    PyMuPDF is usually robust for mixed Hebrew/English PDFs.
    """
    doc = fitz.open(str(bundle_pdf))
    try:
        parts = []
        for page in doc:
            parts.append(page.get_text("text"))
        return "\n".join(parts)
    finally:
        doc.close()


def split_into_questions(bundle_text: str) -> Dict[str, BundleQuestionText]:
    """
    Heuristically split the bundle's extracted text into per-question segments.

    The bundle format usually has:
      - a "Reference (question + official solution)" section
      - then a "(rendered) Student answer" section
      - student answer block often begins with something like: "Q1(א — (Answer Student"

    Returns mapping: qid -> BundleQuestionText
    """
    # Normalize whitespace
    t = bundle_text.replace("\r\n", "\n").replace("\r", "\n")

    # Find markers that start student answers, like "Q1(א"
    matches = list(_QID_RE.finditer(t))
    if not matches:
        return {}

    # We’ll use Q markers to define student-answer blocks.
    # For each Q marker, take text from marker to next marker as "student_answer".
    student_blocks: Dict[str, str] = {}
    for i, m in enumerate(matches):
        qnum = m.group(1)
        part = m.group(2).strip()
        qid = f"Q{qnum}({part})"
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(t)
        student_blocks[qid] = t[start:end].strip()

    # Reference/solution blocks: in your bundle PDF they appear before student blocks.
    # A pragmatic approach: take everything before each student block and carve out the "closest"
    # reference text chunk by searching backward for the "solution) official" marker.
    ref_marker = "solution) official"
    rendered_marker = "(rendered) answer Student"

    out: Dict[str, BundleQuestionText] = {}
    for qid, sblock in student_blocks.items():
        # locate sblock in the full text
        idx = t.find(sblock)
        if idx < 0:
            continue

        # search backwards for reference marker near this student section
        back_window = t[max(0, idx - 20000):idx]
        rm = back_window.rfind(ref_marker)
        sm = back_window.rfind(rendered_marker)

        # If we can locate both, assume reference text is between ref_marker and rendered_marker
        reference_solution = ""
        if rm != -1 and sm != -1 and rm < sm:
            reference_solution = back_window[rm:sm].strip()
        else:
            # fallback: take a chunk before student answer
            reference_solution = back_window[-8000:].strip()

        out[qid] = BundleQuestionText(
            qid=qid,
            reference_solution=reference_solution,
            student_answer=sblock,
        )

    return out
