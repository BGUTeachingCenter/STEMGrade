from __future__ import annotations

"""AI grading using *source-of-truth* inputs:

Instead of extracting the student's answer from the rendered bundle PDF (which
often loses math structure), we grade using:
  1) the official reference PDF text for the relevant question/part, and
  2) the student's original LaTeX snippet for that same question/part.

This typically improves both accuracy and readability.
"""

import os
from pathlib import Path
from typing import Dict, Tuple

from grader.reference_ranges import Key
from grader.student_tex import parse_student_tex_answers

from .payloads import build_payloads
from .grader_payloads import grade_payload_manifest


def _parse_student_answers_for_debug(student_tex: Path, out_dir: Path) -> Dict[Key, str]:
    student_answers, _student_ranges = parse_student_tex_answers(student_tex, out_dir)
    return student_answers


def grade_reference_and_student_tex(
    *,
    reference_pdf: Path,
    student_tex: Path,
    out_dir: Path,
    ollama_base_url: str | None = None,
    model: str | None = None,
) -> Tuple[Path, Dict[Key, str]]:
    """Grade by pairing reference pages with the student's LaTeX snippet.

    Returns:
      - grades.json path
      - the parsed student answers dict (useful for debugging or future UI)
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build payloads first (per question/part), then grade from those payloads.
    manifest_json, _items = build_payloads(
        reference_pdf=reference_pdf,
        student_tex=student_tex,
        out_dir=out_dir,
    )

    grades_json = grade_payload_manifest(
        manifest_json=manifest_json,
        out_dir=out_dir,
        ollama_base_url=ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=model or os.getenv("OLLAMA_MODEL", "gemma3:4b"),
    )

    # Return student answers as before (useful for UI/debugging)
    student_answers = _parse_student_answers_for_debug(student_tex, out_dir)
    return grades_json, student_answers
