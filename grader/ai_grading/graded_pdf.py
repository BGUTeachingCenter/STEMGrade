from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from grader.ai_grading.feedback_reportlab import build_feedback_pdf_from_grades


def build_graded_pdf(
    bundle_pdf: Path,
    grades_json: Path,
    out_dir: Path,
    font_path: Path | None = None,
) -> Path:
    """
    Build final graded PDF by creating feedback.pdf with ReportLab (no LaTeX),
    then merging: [bundle_pdf] + [feedback_pdf].
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    feedback_pdf = out_dir / "graded_feedback.pdf"
    build_feedback_pdf_from_grades(
        grades_json=grades_json,
        out_pdf=feedback_pdf,
        font_path=font_path,
    )

    graded_pdf = out_dir / "graded_test.pdf"
    _merge_pdfs([bundle_pdf, feedback_pdf], graded_pdf)
    return graded_pdf


def _merge_pdfs(inputs: list[Path], output: Path) -> None:
    out = fitz.open()
    try:
        for p in inputs:
            doc = fitz.open(str(p))
            try:
                out.insert_pdf(doc)
            finally:
                doc.close()
        out.save(str(output))
    finally:
        out.close()
