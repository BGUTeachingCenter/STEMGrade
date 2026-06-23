from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemas.ocr_tasks import FullSolutionBuildResult
from schemas.reference_bundle import (
    AnswersOnlyBundle,
    FullSolutionBundle,
    QuestionsAnswersBundle,
    QuestionsOnlyBundle,
)
from services.solution_bank.reference_builder import (
    merge_questions_and_answers_to_full_solution,
    promote_questions_answers_to_full_solution,
)


def full_solution_to_tex(bundle: dict[str, Any] | FullSolutionBundle) -> str:
    """
    Input:
      FullSolutionBundle.

    Output:
      Canonical TeX reference file that parse_reference_tex() can read.

    This TeX becomes:
      reference_current.tex
    """
    full = (
        bundle
        if isinstance(bundle, FullSolutionBundle)
        else FullSolutionBundle.model_validate(bundle)
    )

    lines: list[str] = [
        r"\documentclass[12pt]{article}",
        r"\usepackage[a4paper,margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,mathtools}",
        r"\usepackage{fontspec}",
        r"\usepackage{polyglossia}",
        r"\setmainlanguage{hebrew}",
        r"\setotherlanguage{english}",
        r"\newfontfamily\hebrewfont{Arial}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        "",
        r"\begin{document}",
        r"% Canonical MathGrade full solution file generated from structured JSON.",
        "",
    ]

    if full.exam_title:
        lines.append(rf"\textbf{{{full.exam_title}}}")
        lines.append("")

    for q in full.questions:
        lines.append("")
        lines.append(rf"\section*{{Question {q.question_id}}}")

        for p in q.parts:
            display_part = p.part or p.part_key or "a"

            lines.append("")
            lines.append(rf"\subsection*{{({display_part})}}")

            if p.question_text:
                lines.append(p.question_text.strip())
                lines.append("")

            if p.required_action:
                lines.append(r"\paragraph{Required action}")
                lines.append(p.required_action.strip())
                lines.append("")

            if p.official_solution:
                lines.append(r"\paragraph{Official solution}")
                lines.append(p.official_solution.strip())
                lines.append("")

            if p.expected_answer:
                lines.append(r"\paragraph{Expected answer}")
                lines.append(p.expected_answer.strip())
                lines.append("")

            if p.grading_instructions:
                lines.append(r"\paragraph{Grading instructions}")
                lines.append(p.grading_instructions.strip())
                lines.append("")

            if p.review_status and p.review_status != "ok":
                lines.append(r"\paragraph{Review status}")
                lines.append(str(p.review_status))
                lines.append("")

            if p.warnings:
                lines.append(r"\paragraph{Warnings}")
                for w in p.warnings:
                    lines.append(f"- {w}")
                lines.append("")

    lines.extend(["", r"\end{document}", ""])
    return "\n".join(lines)


def full_solution_to_exam_structure(bundle: dict[str, Any] | FullSolutionBundle) -> dict[str, Any]:
    """
    Input:
      FullSolutionBundle.

    Output:
      Lightweight exam structure used by previews and student template generation.
    """
    full = (
        bundle
        if isinstance(bundle, FullSolutionBundle)
        else FullSolutionBundle.model_validate(bundle)
    )

    questions = []

    for q in full.questions:
        questions.append(
            {
                "question_id": str(q.question_id),
                "parts": [
                    p.part
                    for p in q.parts
                    if p.part
                ],
            }
        )

    return {
        "questions": questions,
        "question_count": len(questions),
        "part_count": sum(len(q.get("parts") or []) for q in questions),
    }


def build_full_solution_from_questions_answers(
    *,
    questions_answers_bundle: QuestionsAnswersBundle | dict[str, Any],
    exam_id: str = "",
    source_name: str = "",
    out_dir: Path | None = None,
) -> FullSolutionBuildResult:
    """
    Input:
      QuestionsAnswersBundle from a combined questions+answers upload.

    Output:
      FullSolutionBuildResult containing FullSolutionBundle and canonical TeX.
    """
    full_dict = promote_questions_answers_to_full_solution(
        questions_answers_bundle=questions_answers_bundle,
        exam_id=exam_id,
    )

    full = FullSolutionBundle.model_validate(full_dict)
    canonical_tex = full_solution_to_tex(full)
    structure = full_solution_to_exam_structure(full)

    canonical_tex_path = ""

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        bundle_path = out_dir / "full_solution_bundle.json"
        bundle_path.write_text(
            json.dumps(full.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        tex_path = out_dir / "reference_current.tex"
        tex_path.write_text(canonical_tex, encoding="utf-8")
        canonical_tex_path = str(tex_path)

        structure_path = out_dir / "exam_structure.json"
        structure_path.write_text(
            json.dumps(structure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return FullSolutionBuildResult(
        ok=True,
        exam_id=full.exam_id or exam_id,
        source_name=source_name,
        full_solution_bundle=full,
        canonical_tex=canonical_tex,
        canonical_tex_path=canonical_tex_path,
        q_count=structure.get("question_count"),
        part_count=structure.get("part_count"),
        warnings=list(full.warnings or []),
        needs_teacher_review=bool(full.warnings or full.structure_corrections),
    )


def build_full_solution_from_questions_and_answers(
    *,
    questions_bundle: QuestionsOnlyBundle | dict[str, Any],
    answers_bundle: AnswersOnlyBundle | dict[str, Any],
    exam_id: str = "",
    source_name: str = "",
    out_dir: Path | None = None,
) -> FullSolutionBuildResult:
    """
    Input:
      QuestionsOnlyBundle + AnswersOnlyBundle.

    Output:
      FullSolutionBuildResult containing FullSolutionBundle and canonical TeX.
    """
    full_dict = merge_questions_and_answers_to_full_solution(
        questions_bundle=questions_bundle,
        answers_bundle=answers_bundle,
        exam_id=exam_id,
    )

    full = FullSolutionBundle.model_validate(full_dict)
    canonical_tex = full_solution_to_tex(full)
    structure = full_solution_to_exam_structure(full)

    canonical_tex_path = ""

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        bundle_path = out_dir / "full_solution_bundle.json"
        bundle_path.write_text(
            json.dumps(full.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        tex_path = out_dir / "reference_current.tex"
        tex_path.write_text(canonical_tex, encoding="utf-8")
        canonical_tex_path = str(tex_path)

        structure_path = out_dir / "exam_structure.json"
        structure_path.write_text(
            json.dumps(structure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return FullSolutionBuildResult(
        ok=True,
        exam_id=full.exam_id or exam_id,
        source_name=source_name,
        full_solution_bundle=full,
        canonical_tex=canonical_tex,
        canonical_tex_path=canonical_tex_path,
        q_count=structure.get("question_count"),
        part_count=structure.get("part_count"),
        warnings=list(full.warnings or []),
        needs_teacher_review=bool(full.warnings or full.structure_corrections),
    )