from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .ollama_client import OllamaClient
from .pdf_extract import BundleQuestionText, extract_bundle_text, split_into_questions
from .schema import grading_response_schema


@dataclass
class QuestionGrade:
    qid: str
    max_points: float
    score: float
    summary: str
    what_was_correct: List[str]
    main_mistakes: List[str]
    how_to_improve: List[str]
    confidence: float


@dataclass
class BundleGrades:
    total_score: float
    total_max: float
    question_grades: List[QuestionGrade]

    def to_dict(self) -> dict:
        return {
            "total_score": self.total_score,
            "total_max": self.total_max,
            "questions": [q.__dict__ for q in self.question_grades],
        }


_POINTS_RE = re.compile(r"\(\s*(\d+)\s*נקודות\s*\)")


def infer_max_points(reference_solution_text: str, default_max: float = 0.0) -> float:
    """
    Try to infer max points from the reference/solution block.
    Looks for patterns like '(15 נקודות)'.

    If not found, returns default_max.
    """
    m = _POINTS_RE.search(reference_solution_text)
    if not m:
        return default_max
    return float(m.group(1))


def grade_bundle_pdf(
    bundle_pdf: Path,
    out_dir: Path,
    ollama_base_url: str = "http://localhost:11434",
    model: str = "gemma3:4b",
) -> Tuple[Path, Path]:
    """
    Main entry point:
      - reads the bundle pdf
      - extracts per-question official solution + student answer
      - calls Ollama to grade each question
      - writes grades.json and a graded_feedback.tex (PDF generation handled elsewhere)

    Returns: (grades_json_path, grades_tex_path)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_text = extract_bundle_text(bundle_pdf)
    questions: Dict[str, BundleQuestionText] = split_into_questions(bundle_text)

    if not questions:
        raise RuntimeError("Could not detect questions in bundle PDF text extraction.")

    client = OllamaClient(base_url=ollama_base_url, model=model)

    system = (
        "You are a strict but fair university calculus exam grader. "
        "You MUST compare the student's answer to the official solution. "
        "Grade only based on mathematical correctness and completeness. "
        "If the student's answer is irrelevant or 'I don't know', give 0. "
        "If the student's answer partially matches, award partial credit. "
        "Output ONLY valid JSON matching the provided schema."
    )

    schema = grading_response_schema()

    graded: List[QuestionGrade] = []
    for qid, qt in questions.items():
        max_points = infer_max_points(qt.reference_solution, default_max=0.0)

        user = (
            f"Question ID: {qid}\n"
            f"Max points: {max_points}\n\n"
            "OFFICIAL SOLUTION (and question statement if included):\n"
            f"{qt.reference_solution}\n\n"
            "STUDENT ANSWER:\n"
            f"{qt.student_answer}\n\n"
            "Now produce the JSON grade. "
            "Score must be between 0 and Max points. "
            "Confidence between 0 and 1."
        )

        resp = client.chat_json(system=system, user=user, schema=schema, temperature=0.2)

        # Normalize + validate numeric ranges
        score = float(resp["score"])
        if max_points > 0:
            score = max(0.0, min(score, max_points))
        confidence = float(resp["confidence"])
        confidence = max(0.0, min(confidence, 1.0))

        graded.append(
            QuestionGrade(
                qid=resp["qid"],
                max_points=float(resp["max_points"]),
                score=score,
                summary=resp["summary"],
                what_was_correct=list(resp["what_was_correct"]),
                main_mistakes=list(resp["main_mistakes"]),
                how_to_improve=list(resp["how_to_improve"]),
                confidence=confidence,
            )
        )

    total_score = sum(q.score for q in graded)
    total_max = sum(q.max_points for q in graded if q.max_points)

    bundle = BundleGrades(total_score=total_score, total_max=total_max, question_grades=graded)

    grades_json = out_dir / "grades.json"
    grades_json.write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    grades_tex = out_dir / "graded_feedback.tex"
    grades_tex.write_text(_render_feedback_tex(bundle, bundle_pdf.name), encoding="utf-8")

    return grades_json, grades_tex


def _latex_escape(s: str) -> str:
    """
    Escape plain text for LaTeX reliably.
    - Escapes special characters.
    - Converts backslashes to \\textbackslash{} to avoid accidental commands.
    - Leaves newlines as line breaks.
    """
    if s is None:
        return ""

    # Normalize line endings
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    # Turn newlines into LaTeX line breaks (more stable than raw newlines)
    return "".join(out).replace("\n", r"\\ ")



def _render_feedback_tex(bundle: BundleGrades, original_pdf_filename: str) -> str:
    """
    Produce a LaTeX document that:
      - shows total score
      - includes per-question feedback
      - (optionally) can be combined with the original bundle PDF via pdfpages in a separate step
    """
    lines = []
    lines.append(r"\documentclass[12pt]{article}")
    lines.append(r"\usepackage[a4paper,margin=2cm]{geometry}")
    lines.append(r"\usepackage{fontspec}")  # XeLaTeX
    lines.append(r"\setmainfont{Arial}")
    lines.append(r"\usepackage{hyperref}")
    lines.append(r"\usepackage{enumitem}")
    lines.append(r"\usepackage{microtype}")
    lines.append(r"\usepackage{polyglossia}")
    lines.append(r"\setdefaultlanguage{english}")
    lines.append(r"\setotherlanguage{hebrew}")
    lines.append(r"\begin{document}")

    for q in bundle.question_grades:
        lines.append(r"\hrule\medskip")
        lines.append(rf"\subsection*{{{_latex_escape(q.qid)} \ \ \ ({q.score:.1f}/{q.max_points:.1f})}}")
        lines.append(_latex_escape(q.summary))
        lines.append(r"\medskip")

        lines.append(r"\textbf{What you did well:}")
        lines.append(r"\begin{itemize}[leftmargin=*]")
        for item in q.what_was_correct[:8]:
            lines.append(rf"\item {_latex_escape(item)}")
        lines.append(r"\end{itemize}")

        lines.append(r"\textbf{Main mistakes / missing parts:}")
        lines.append(r"\begin{itemize}[leftmargin=*]")
        for item in q.main_mistakes[:8]:
            lines.append(rf"\item {_latex_escape(item)}")
        lines.append(r"\end{itemize}")

        lines.append(r"\textbf{How to improve (next time):}")
        lines.append(r"\begin{itemize}[leftmargin=*]")
        for item in q.how_to_improve[:8]:
            lines.append(rf"\item {_latex_escape(item)}")
        lines.append(r"\end{itemize}")

        lines.append(rf"\textit{{Confidence: {q.confidence:.2f}}}")

    lines.append(r"\end{document}")
    return "\n".join(lines)
