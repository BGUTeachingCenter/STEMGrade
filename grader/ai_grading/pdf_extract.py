from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import fitz  # PyMuPDF


@dataclass
class BundleQuestionText:
    """Text extracted for a single question part from the bundle."""
    qid: str                 # e.g. "Q2(א)"
    reference_solution: str  # official question+solution text
    student_answer: str      # student's answer text


# Matches text like: "()א2שאלה3"  (i.e., שאלה 2 (א)) in extracted PDF text
# Groups: (letter, question_number)
_HE_QID_RE = re.compile(r"\(\)\s*([אבגדה])\s*(\d+)\s*שאלה\s*\d+", re.UNICODE)

_REF_MARK = "solution) official"
_STU_MARK = "(rendered) answer Student"


def _extract_qid_from_page_text(text: str) -> Optional[str]:
    """
    Extract a normalized qid like 'Q2(א)' from a page.
    Works with the bundle's Hebrew TOC/title format.
    """
    m = _HE_QID_RE.search(text)
    if not m:
        return None
    letter = m.group(1)
    qnum = m.group(2)
    return f"Q{qnum}({letter})"


def split_bundle_pdf_into_questions(bundle_pdf: Path) -> Dict[str, BundleQuestionText]:
    """
    Split the bundle PDF into per-question blocks by scanning pages and using
    the bundle's section markers:
      - reference block starts at pages containing 'solution) official'
      - student block starts after pages containing '(rendered) answer Student'

    Returns: {qid -> BundleQuestionText}
    """
    doc = fitz.open(str(bundle_pdf))
    try:
        out: Dict[str, BundleQuestionText] = {}
        current_qid: Optional[str] = None
        mode: Optional[str] = None  # "ref" or "student"

        for page in doc:
            text = page.get_text("text").replace("\r\n", "\n").replace("\r", "\n").strip()
            if not text:
                continue

            # Reference section start
            if _REF_MARK in text:
                qid = _extract_qid_from_page_text(text)
                if qid is None:
                    # Sometimes the title isn't on the same page; try next pages by keeping mode
                    qid = current_qid

                current_qid = qid
                mode = "ref"

                if current_qid and current_qid not in out:
                    out[current_qid] = BundleQuestionText(current_qid, "", "")

                if current_qid:
                    out[current_qid].reference_solution += text + "\n\n"
                continue

            # Student section marker page (usually just a label page)
            if _STU_MARK in text:
                mode = "student"
                # do NOT append this marker page to student answer
                continue

            # If we don't know the current question yet, try extracting from this page anyway
            if current_qid is None:
                qid_guess = _extract_qid_from_page_text(text)
                if qid_guess:
                    current_qid = qid_guess
                    if current_qid not in out:
                        out[current_qid] = BundleQuestionText(current_qid, "", "")
                else:
                    continue

            # Append content depending on mode
            if current_qid not in out:
                out[current_qid] = BundleQuestionText(current_qid, "", "")

            if mode == "student":
                out[current_qid].student_answer += text + "\n\n"
            else:
                # default to reference if mode unknown (safer)
                out[current_qid].reference_solution += text + "\n\n"

        # Drop entries with no student content (just in case)
        out = {k: v for k, v in out.items() if v.student_answer.strip() or v.reference_solution.strip()}
        return out

    finally:
        doc.close()
