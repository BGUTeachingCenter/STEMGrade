# grader/qa_bundle.py
"""High-level pipeline to create a Q/A bundle PDF.

This is the project's main entry point.

Pipeline:
  1) Clean the reference PDF (drops cover/formula/blank pages).
  2) Detect question/part page ranges inside the reference PDF.
  3) Parse the student's LaTeX into per-question/per-part answer snippets.
  4) Compile each snippet into its own PDF.
  5) Create a bundle .tex that embeds reference pages + student PDFs.
  6) Compile the bundle into a final PDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from .answer_render import compile_student_answer_pdfs
from .bundle_writer import write_bundle_tex
from .compile_tex import compile_tex_to_pdf
from .pdf_cleanse import CleanseReport, cleanse_test_pdf
from .reference_ranges import Key, find_reference_ranges
from .student_tex import parse_student_tex_answers


@dataclass(frozen=True)
class QABundleOutputs:
    """Outputs of `generate_qa_bundle_pdf`.

    Attributes:
        bundle_pdf: Final bundled PDF.
        bundle_tex: LaTeX source used to build the bundle.
        student_clean_pdf: Cleaned/compiled PDF version of the student's full submission.
        reference_clean_pdf: Cleaned reference PDF that was actually used.
        cleanse_report: Report describing which pages were removed from the reference.
        ref_ranges: Detected page ranges per (question, part).
        student_ranges: Character ranges in the student .tex per (question, part).
        student_answer_pdfs: Per-part compiled answer PDFs (None if compilation failed).
    """

    bundle_pdf: Path
    bundle_tex: Path
    student_clean_pdf: Path
    reference_clean_pdf: Path
    cleanse_report: CleanseReport
    ref_ranges: Dict[Key, Tuple[int, int]]
    student_ranges: Dict[Key, Tuple[int, int]]
    student_answer_pdfs: Dict[Key, Optional[Path]]


def generate_qa_bundle_pdf(
    *,
    reference_pdf: Path,
    student_tex: Path,
    out_dir: Path,
    bundle_stem: str = "qa_bundle",
    font_name: str = "Arial",
) -> QABundleOutputs:
    """Generate a single PDF containing reference pages + student answers.

    Args:
        reference_pdf: PDF that contains the questions and official solutions.
        student_tex: Student LaTeX submission.
        out_dir: Output directory.
        bundle_stem: Base filename (without extension) for the bundle outputs.
        font_name: Font to use for Hebrew RTL text. Must exist on the machine.

    Returns:
        `QABundleOutputs` describing all generated artifacts.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # 0) Clean reference/test PDF (remove cover page, formula sheet, trailing blanks)
    cleanse_report = cleanse_test_pdf(reference_pdf, out_dir)
    reference_clean_pdf = cleanse_report.output_pdf

    # 1) Detect reference ranges
    ref_ranges = find_reference_ranges(reference_clean_pdf, out_dir)
    if not ref_ranges:
        raise RuntimeError(
            "Reference question detection failed.\n"
            "Check build/debug_reference_pages.txt for extracted text.\n"
            "If it is empty, your PDF may be scanned (no selectable text)."
        )

    # 2) Parse student TeX answers
    student_answers, student_ranges = parse_student_tex_answers(student_tex, out_dir)
    if not student_answers:
        raise RuntimeError(
            "Could not parse any student answers from the TeX.\n"
            "Check build/debug_student_tex_parts.txt."
        )

    # 3) Compile full student TeX (useful for debugging/validation)
    student_clean_pdf = compile_tex_to_pdf(
        student_tex,
        out_dir,
        clean=True,
        font_name=font_name,
        passes=2,
        texinputs=[student_tex.parent],
    ).pdf

    # 4) Compile each answer snippet into its own PDF
    answer_pdfs = compile_student_answer_pdfs(student_answers, out_dir, font_name)

    # 5) Write bundle TeX
    bundle_tex = out_dir / f"{bundle_stem}.tex"
    write_bundle_tex(
        bundle_tex,
        reference_pdf=reference_clean_pdf,
        ref_ranges=ref_ranges,
        answer_pdfs=answer_pdfs,
        font_name=font_name,
        title="Q/A Bundle (Reference pages + Rendered student answers)",
    )

    # 6) Compile bundle
    bundle_pdf = compile_tex_to_pdf(
        bundle_tex,
        out_dir,
        clean=False,
        font_name=font_name,
        passes=2,
        texinputs=[bundle_tex.parent],
    ).pdf

    # Normalize bundle name if compile_tex produced *_clean.pdf
    normalized = out_dir / f"{bundle_stem}.pdf"
    if bundle_pdf.exists() and bundle_pdf != normalized:
        try:
            bundle_pdf.replace(normalized)
            bundle_pdf = normalized
        except Exception:
            pass

    if not bundle_pdf.exists():
        raise RuntimeError("Bundle PDF was not created. Check LaTeX logs in build/.")

    return QABundleOutputs(
        bundle_pdf=bundle_pdf,
        bundle_tex=bundle_tex,
        student_clean_pdf=student_clean_pdf,
        reference_clean_pdf=reference_clean_pdf,
        cleanse_report=cleanse_report,
        ref_ranges=ref_ranges,
        student_ranges=student_ranges,
        student_answer_pdfs=answer_pdfs,
    )
