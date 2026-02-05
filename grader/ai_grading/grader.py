from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .ollama_client import OllamaClient
from .pdf_extract import BundleQuestionText, split_bundle_pdf_into_questions
from .prompting import load_grading_prompt
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
    mismatch: dict
    common_errors_detected: List[str]
    suggested_next_step_he: str
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
    Extract max points from reference/solution text.
    Picks the first '(NN נקודות)' it sees (usually the part's points).
    """
    if not reference_solution_text:
        return default_max
    m = _POINTS_RE.search(reference_solution_text)
    return float(m.group(1)) if m else default_max



def grade_bundle_pdf(
    bundle_pdf: Path,
    out_dir: Path,
    ollama_base_url: str | None = None,
    model: str | None = None,
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

    # Single source of truth: server.py sets these environment variables.
    # If this function is used standalone, environment variables still work.
    if ollama_base_url is None:
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if model is None:
        model = os.getenv("OLLAMA_MODEL", "gemma3:4b")

    questions = split_bundle_pdf_into_questions(bundle_pdf)
    if not questions:
        raise RuntimeError("Could not detect questions in bundle PDF text extraction.")

    client = OllamaClient(base_url=ollama_base_url, model=model)

    # Load human-readable grading instructions from a txt file next to this module.
    system = load_grading_prompt()

    schema = grading_response_schema()

    graded: List[QuestionGrade] = []
    for qid, qt in questions.items():
        max_points = infer_max_points(qt.reference_solution, default_max=0.0)

        payload = {
            "question_id": qid,
            "reference": {
                "question_text": "",
                "solution_text": qt.reference_solution,
            },
            "student": {
                "latex_raw": qt.student_answer,
            },
            "rubric": {
                "score_max": max_points,
                "key_points": [],
            },
        }

        user = json.dumps(payload, ensure_ascii=False, indent=2)

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
                mismatch=dict(resp.get("mismatch") or {}),
                common_errors_detected=list(resp.get("common_errors_detected") or []),
                suggested_next_step_he=str(resp.get("suggested_next_step_he") or ""),
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


import re

_MATH_SPLIT_RE = re.compile(r"(\$\$.*?\$\$|\$.*?\$)", re.DOTALL)

def _escape_text_only(s: str) -> str:
    """Escape plain text for LaTeX (NOT math)."""
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    repl = {
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
    return "".join(out).replace("\n", r"\\ ")

def _escape_math_only(s: str) -> str:
    """
    Escape only characters that commonly break LaTeX *inside math*.
    Keep backslashes intact so commands like \epsilon, \frac, \Delta survive.
    """
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # In math mode, % and # still comment/parameterize; escape them.
    # Curly braces are meaningful in math; keep them.
    s = s.replace("%", r"\%").replace("#", r"\#")
    return s

def _latex_render_mixed(s: str) -> str:
    """
    Render a string that may contain $...$ or $$...$$.
    Text outside math is escaped; math content is preserved.
    """
    if s is None:
        return ""
    parts = _MATH_SPLIT_RE.split(s)
    rendered = []
    for part in parts:
        if not part:
            continue
        if (part.startswith("$") and part.endswith("$")):
            # math segment (either $...$ or $$...$$)
            rendered.append(part[0:2] + _escape_math_only(part[2:-2]) + part[-2:]
                           if part.startswith("$$") else
                           "$" + _escape_math_only(part[1:-1]) + "$")
        else:
            rendered.append(_escape_text_only(part))
    return "".join(rendered)




def _render_feedback_tex(bundle: BundleGrades, original_pdf_filename: str) -> str:
    lines = []
    lines.append(r"\documentclass[12pt]{article}")
    lines.append(r"\usepackage[a4paper,margin=2cm]{geometry}")
    lines.append(r"\usepackage{fontspec}")  # XeLaTeX
    lines.append(r"\setmainfont{Arial}")
    lines.append(r"\usepackage{hyperref}")
    lines.append(r"\usepackage{enumitem}")
    lines.append(r"\usepackage{microtype}")
    lines.append(r"\usepackage{polyglossia}")
    lines.append(r"\setdefaultlanguage{hebrew}")
    lines.append(r"\setotherlanguage{english}")
    lines.append(r"\setlength{\parskip}{6pt}")
    lines.append(r"\setlength{\parindent}{0pt}")
    lines.append(r"\begin{document}")

    # Optional title header
    lines.append(r"\section*{משוב בדיקה}")
    lines.append(_latex_render_mixed(
        f"שם קובץ: {original_pdf_filename}"
    ))

    for q in bundle.question_grades:
        lines.append(r"\hrule\medskip")
        # qid is plain text (usually "Q1(a)" etc.) – escape as text only
        lines.append(rf"\subsection*{{{_escape_text_only(q.qid)} \ \ \ ({q.score:.1f}/{q.max_points:.1f})}}")

        # summary may include math
        lines.append(_latex_render_mixed(q.summary))
        lines.append(r"\medskip")

        # Hebrew headings (readability)
        lines.append(r"\textbf{מה עשית נכון:}")
        lines.append(r"\begin{itemize}[leftmargin=*,itemsep=3pt,topsep=2pt]")
        for item in q.what_was_correct[:8]:
            lines.append(rf"\item {_latex_render_mixed(item)}")
        lines.append(r"\end{itemize}")

        lines.append(r"\textbf{טעויות / חלקים חסרים:}")
        lines.append(r"\begin{itemize}[leftmargin=*,itemsep=3pt,topsep=2pt]")
        for item in q.main_mistakes[:8]:
            lines.append(rf"\item {_latex_render_mixed(item)}")
        lines.append(r"\end{itemize}")

        lines.append(r"\textbf{איך להשתפר לפעם הבאה:}")
        lines.append(r"\begin{itemize}[leftmargin=*,itemsep=3pt,topsep=2pt]")
        for item in q.how_to_improve[:8]:
            lines.append(rf"\item {_latex_render_mixed(item)}")
        lines.append(r"\end{itemize}")

        # confidence line
        lines.append(_escape_text_only(f"רמת ביטחון: {q.confidence:.2f}"))

    lines.append(r"\end{document}")
    return "\n".join(lines)

