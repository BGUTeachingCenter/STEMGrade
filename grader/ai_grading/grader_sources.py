from __future__ import annotations

"""AI grading using *source-of-truth* inputs:

Instead of extracting the student's answer from the rendered bundle PDF (which
often loses math structure), we grade using:
  1) the official reference PDF text for the relevant question/part, and
  2) the student's original LaTeX snippet for that same question/part.

Pipeline:
  reference.pdf + student.tex
    -> payloads/*.json (+ manifest.json)   [build_payloads]
    -> grades.json                         [grade_payload_manifest]
    -> graded PDF                          [build_graded_pdf elsewhere]

This module ensures the payload-building step runs first, then grading runs
from those payloads. It also enforces using the compact payload.ai_input
structure (to reduce LLM confusion and improve math readability).
"""

import json
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


def _sanity_check_payloads(manifest_json: Path, out_dir: Path) -> None:
    """
    Optional sanity check:
    Ensure each payload has ai_input fields populated (question/solution/student).
    Writes a short debug report if something is missing.
    """
    try:
        manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    except Exception:
        return

    payload_dir = out_dir / "payloads"
    bad: list[str] = []

    for it in manifest.get("items", []):
        fn = it.get("payload_file")
        if not fn:
            continue
        p = payload_dir / fn
        if not p.exists():
            bad.append(f"{fn}: missing file")
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            bad.append(f"{fn}: invalid json")
            continue

        ai = payload.get("ai_input") or {}
        q = (ai.get("question_latex") or "").strip()
        sol = (ai.get("reference_solution_latex") or "").strip()
        stu = (ai.get("student_answer_latex") or "").strip()

        # If question is empty, that's okay sometimes (some refs don't split cleanly),
        # but solution + student should not be empty.
        if not sol or not stu:
            qid = payload.get("qid") or payload.get("question_id") or fn
            bad.append(f"{qid}: empty solution={not bool(sol)} empty student={not bool(stu)}")

    if bad:
        (out_dir / "debug_payload_sanity.txt").write_text(
            "Payload sanity issues:\n" + "\n".join(bad) + "\n",
            encoding="utf-8",
        )


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

    # Sanity check to catch empty/misaligned payloads early
    _sanity_check_payloads(manifest_json, out_dir)

    # Enforce: grading reads payload["ai_input"] (compact, LLM-friendly).
    # grader_payloads.py should read this env var and send only ai_input to the model.
    os.environ["MATHGRADE_USE_AI_INPUT"] = "1"

    grades_json = grade_payload_manifest(
        manifest_json=manifest_json,
        out_dir=out_dir,
        ollama_base_url=ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=model or os.getenv("OLLAMA_MODEL", "gemma3:4b"),
    )

    # Return student answers as before (useful for UI/debugging)
    student_answers = _parse_student_answers_for_debug(student_tex, out_dir)
    return grades_json, student_answers
