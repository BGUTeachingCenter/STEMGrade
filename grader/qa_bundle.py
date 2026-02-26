# grader/qa_bundle.py
"""Backward-compatible wrappers for bundle generation.

Bundling logic lives in ONE module now: `grader.bundler`.

TeX-first refactor:
- Default wrappers generate ONLY the TeX bundle (no XeLaTeX compile here).
- Dedicated *_pdf wrappers exist if you still need the compiled PDF.
"""

from __future__ import annotations

from pathlib import Path

from .bundler import QABundleOutputs, generate_bundle


# ----------------------------
# Legacy: explicit PDF wrapper
# ----------------------------

def generate_qa_bundle_pdf(
    *,
    reference_pdf: Path,
    student_tex: Path,
    out_dir: Path,
    bundle_stem: str = "qa_bundle",
    font_name: str = "Arial",
) -> QABundleOutputs:
    """
    Legacy behavior: generate bundle TEX + compile to PDF.
    Use only if you explicitly need the PDF at this stage.
    """
    return generate_bundle(
        reference=reference_pdf,
        student_tex=student_tex,
        out_dir=out_dir,
        bundle_stem=bundle_stem,
        font_name=font_name,
        compile_pdf=True,  # ✅ compile here
    )


# -------------------------------------------
# TeX-first: used by the new grading pipeline
# -------------------------------------------

def generate_qa_bundle_tex(
    *,
    reference_pdf: Path,
    student_tex: Path,
    out_dir: Path,
    bundle_stem: str = "qa_bundle",
    font_name: str = "Arial",
) -> QABundleOutputs:
    """
    TeX-first: generate ONLY bundle TeX (no PDF compilation).
    """
    return generate_bundle(
        reference=reference_pdf,
        student_tex=student_tex,
        out_dir=out_dir,
        bundle_stem=bundle_stem,
        font_name=font_name,
        compile_pdf=False,  # ✅ TeX only
    )


def generate_qa_bundle_from_reference_tex(
    *,
    reference_tex: Path,
    student_tex: Path,
    out_dir: Path,
    bundle_stem: str = "qa_bundle",
    font_name: str = "Arial",
) -> QABundleOutputs:
    """
    TeX-first: reference is TeX. Generate ONLY bundle TeX (no PDF compilation).
    """
    return generate_bundle(
        reference=reference_tex,
        student_tex=student_tex,
        out_dir=out_dir,
        bundle_stem=bundle_stem,
        font_name=font_name,
        compile_pdf=False,  # ✅ TeX only
    )