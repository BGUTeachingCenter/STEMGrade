# services/qa_bundle.py
"""Backward-compatible wrappers for bundle generation.

Bundling logic lives in ONE module now: `services.bundler`.

TeX-first refactor:
- Default wrappers generate ONLY the TeX bundle (no XeLaTeX compile here).
- Dedicated *_pdf wrappers exist if you still need the compiled PDF.
"""

from __future__ import annotations

from pathlib import Path

from flows.student_grading.bundler import QABundleOutputs, generate_bundle

# -------------------------------------------
# TeX-first: used by the new grading pipeline
# -------------------------------------------

def generate_qa_bundle_tex(
    *,
    reference_tex: Path,
    student_tex: Path,
    out_dir: Path,
    bundle_stem: str = "qa_bundle",
    font_name: str = "Arial",
) -> QABundleOutputs:
    """
    TeX-first: generate ONLY bundle TeX (no PDF compilation).
    """
    return generate_bundle(
        reference_tex=reference_tex,
        student_tex=student_tex,
        out_dir=out_dir,
        bundle_stem=bundle_stem,
        font_name=font_name,
        compile_pdf=False,  # ✅ TeX only
    )