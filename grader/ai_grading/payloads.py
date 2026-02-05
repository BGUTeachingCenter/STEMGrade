from __future__ import annotations

"""Build machine-readable grading payloads.

We separate *data preparation* from *AI grading*.

Payloads are JSON files (one per question/part) plus a manifest.json that
enumerates them. This makes the pipeline reproducible and debuggable:

  reference.pdf + student.tex -> payloads/*.json -> grades.json -> graded PDF

Design goals:
  - Hebrew stays as Unicode text.
  - Math stays as LaTeX (student answer is raw LaTeX).
  - Each question/part is a standalone unit with a stable id.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF

from grader.pdf_cleanse import cleanse_test_pdf
from grader.reference_ranges import Key, find_reference_ranges
from grader.student_tex import parse_student_tex_answers


@dataclass(frozen=True)
class PayloadItem:
    key: Key
    qid: str
    max_points: float
    reference_text: str
    student_latex: str
    payload_path: Path


def qid_from_key(key: Key) -> str:
    qnum, part = key
    if part:
        return f"Q{qnum}({part})"
    return f"Q{qnum}"


def extract_reference_text(reference_pdf: Path, start_page_1b: int, end_page_1b: int) -> str:
    """Extract selectable text from a page range (1-based inclusive)."""
    doc = fitz.open(str(reference_pdf))
    try:
        parts: List[str] = []
        start0 = max(0, start_page_1b - 1)
        end0 = min(doc.page_count - 1, end_page_1b - 1)
        for i in range(start0, end0 + 1):
            txt = (doc[i].get_text("text") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if txt:
                parts.append(txt)
        return "\n\n".join(parts).strip()
    finally:
        doc.close()


def build_payloads(
    *,
    reference_pdf: Path,
    student_tex: Path,
    out_dir: Path,
    default_max_points: float = 0.0,
) -> Tuple[Path, List[PayloadItem]]:
    """Create per-question JSON payloads + a manifest.

    Returns:
      - manifest.json path
      - list of created PayloadItem
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = out_dir / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)

    # 1) Clean reference and detect per-question ranges
    cleanse_report = cleanse_test_pdf(reference_pdf, out_dir)
    reference_clean_pdf = cleanse_report.output_pdf

    ref_ranges = find_reference_ranges(reference_clean_pdf, out_dir)
    if not ref_ranges:
        raise RuntimeError(
            "Reference question detection failed. "
            "Check out/debug_reference_pages.txt. "
            "If empty, the PDF may be scanned (no selectable text)."
        )

    # 2) Parse student's TeX into snippets per question/part
    student_answers, _student_ranges = parse_student_tex_answers(student_tex, out_dir)
    if not student_answers:
        raise RuntimeError(
            "Could not parse any student answers from the TeX. "
            "Check out/debug_student_tex_parts.txt."
        )

    # 3) Create payloads only for overlapping keys
    keys = sorted(set(ref_ranges.keys()) & set(student_answers.keys()))
    if not keys:
        raise RuntimeError(
            "No overlapping questions found between reference PDF ranges and parsed student TeX answers. "
            "(Common causes: question numbering mismatch, or missing part markers in the student template.)"
        )

    items: List[PayloadItem] = []

    for key in keys:
        start_p, end_p = ref_ranges[key]
        ref_text = extract_reference_text(reference_clean_pdf, start_p, end_p)
        student_latex = (student_answers.get(key) or "").strip()

        qid = qid_from_key(key)

        # Payload format optimized for LLM readability:
        #   - Hebrew as Unicode text
        #   - Student math as raw LaTeX
        #   - Stable per-question id
        # We keep backward-compatible keys (qid, reference.text) but add the
        # Option-B prompt keys (question_id, reference.solution_text).
        payload = {
            "question_id": qid,
            "qid": qid,
            "key": {"qnum": key[0], "part": key[1]},
            "max_points": default_max_points,  # model may infer/update; grader clamps
            "rubric": {"score_max": default_max_points, "key_points": []},
            "reference": {
                "source": "reference.pdf",
                "page_range_1_based_inclusive": [start_p, end_p],
                "question_text": "",
                "solution_text": ref_text,
                "text": ref_text,
            },
            "student": {
                "source": "student.tex",
                "latex_raw": student_latex,
                "latex_clean": "",
            },
        }

        payload_path = payload_dir / f"{qid.replace('(', '_').replace(')', '')}.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        items.append(
            PayloadItem(
                key=key,
                qid=qid,
                max_points=float(default_max_points),
                reference_text=ref_text,
                student_latex=student_latex,
                payload_path=payload_path,
            )
        )

    manifest = {
        "version": 1,
        "reference_pdf": str(reference_pdf.name),
        "student_tex": str(student_tex.name),
        "count": len(items),
        "items": [
            {
                "qid": it.qid,
                "payload_file": it.payload_path.name,
            }
            for it in items
        ],
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return manifest_path, items
