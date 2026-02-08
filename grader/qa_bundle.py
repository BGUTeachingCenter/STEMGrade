# grader/qa_bundle.py
"""Backward-compatible wrappers for bundle generation.

Bundling logic lives in ONE module now: `grader.bundler`.
"""

from __future__ import annotations

from pathlib import Path

from .bundler import QABundleOutputs, generate_bundle


def generate_qa_bundle_pdf(
    *,
    reference_pdf: Path,
    student_tex: Path,
    out_dir: Path,
    bundle_stem: str = "qa_bundle",
    font_name: str = "Arial",
) -> QABundleOutputs:
    return generate_bundle(
        reference=reference_pdf,
        student_tex=student_tex,
        out_dir=out_dir,
        bundle_stem=bundle_stem,
        font_name=font_name,
    )


def generate_qa_bundle_from_reference_tex(
    *,
    reference_tex: Path,
    student_tex: Path,
    out_dir: Path,
    bundle_stem: str = "qa_bundle",
    font_name: str = "Arial",
) -> QABundleOutputs:
    return generate_bundle(
        reference=reference_tex,
        student_tex=student_tex,
        out_dir=out_dir,
        bundle_stem=bundle_stem,
        font_name=font_name,
    )
