# services/ai_grading/feedback_tex.py
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# If you have these already, keep them. Otherwise remove/replace accordingly.
from services.ai_grading.json_sanitizer import sanitize_grades_json  # optional but recommended

FIXED_FONT = os.getenv("MATHGRADE_FONT", "Arial")

def _escape_minimal(s: str) -> str:
    """
    Minimal escaping for prose. We intentionally do NOT try to "fix" math here.
    XeLaTeX/tex_cleanup handles robustness in compilation stage.
    """
    if not s:
        return ""
    s = str(s)
    s = s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    return s


def _preamble(title: str = "משוב בדיקה", font_name: str = FIXED_FONT) -> List[str]:
    return [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,margin=2cm]{geometry}",
        r"\usepackage{fontspec}",
        rf"\setmainfont{{{font_name}}}",
        r"\usepackage{amsmath,amssymb,mathtools}",
        r"\usepackage{polyglossia}",
        r"\usepackage{xcolor}",
        r"\usepackage{enumitem}",
        r"\usepackage{bidi}",
        r"\setdefaultlanguage{hebrew}",
        r"\setotherlanguage{english}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.5em}",
        "",
        r"\begin{document}",
        r"\small",
        "",
        rf"{{\LARGE {_escape_minimal(title)}\par}}",
        r"\vspace{0.6cm}",
        "",
    ]


def render_feedback_body(grades: Dict[str, Any]) -> str:
    """
    Render ONLY the body portion (no preamble/document wrappers).
    """
    lines: List[str] = []

    # adapt this to your grades schema as needed
    questions = grades.get("question_grades") or grades.get("questions") or []
    if not isinstance(questions, list):
        questions = []

    for q in questions:
        qid = str(q.get("qid") or q.get("id") or "Q?")

        lines += [
            r"\hrule\vspace{0.35cm}",
            rf"{{\Large {_escape_minimal(qid)}\par}}",
            r"\vspace{0.2cm}",
        ]

        score = q.get("score")
        maxp = q.get("max_points")
        if score is not None and maxp is not None:
            lines.append(rf"\textbf{{ציון:}} {_escape_minimal(score)} / {_escape_minimal(maxp)}\par")

        summary = q.get("summary") or ""
        if summary:
            lines.append(_escape_minimal(summary) + r"\par")

        good = q.get("what_was_correct") or []
        if isinstance(good, list) and good:
            lines.append(r"\textbf{מה עשית נכון:}")
            lines.append(r"\begin{itemize}")
            for item in good[:10]:
                if item:
                    lines.append(rf"\item {_escape_minimal(item)}")
            lines.append(r"\end{itemize}")

        mistakes = q.get("main_mistakes") or []
        if isinstance(mistakes, list) and mistakes:
            lines.append(r"\textbf{טעויות עיקריות:}")
            lines.append(r"\begin{itemize}")
            for item in mistakes[:10]:
                if item:
                    lines.append(rf"\item {_escape_minimal(item)}")
            lines.append(r"\end{itemize}")

        improve = q.get("how_to_improve") or []
        if isinstance(improve, list) and improve:
            lines.append(r"\textbf{איך להשתפר:}")
            lines.append(r"\begin{itemize}")
            for item in improve[:10]:
                if item:
                    lines.append(rf"\item {_escape_minimal(item)}")
            lines.append(r"\end{itemize}")

        next_step = q.get("suggested_next_step_he") or q.get("suggested_next_step") or ""
        if next_step:
            lines.append(r"\textbf{צעד מומלץ הבא:}")
            lines.append(_escape_minimal(next_step) + r"\par")

        lines.append(r"\vspace{0.4cm}")

    return "\n".join(lines)


def render_feedback_tex(grades: Dict[str, Any], title: str = "משוב בדיקה", font_name: str = "Arial") -> str:
    """
    Render a FULL TeX document for feedback (still TeX-only; compilation happens elsewhere).
    """
    lines = _preamble(title=title, font_name=font_name)
    lines.append(render_feedback_body(grades))
    lines += ["", r"\end{document}"]
    return "\n".join(lines)


def build_feedback_tex(
    *,
    grades_json: Path,
    out_dir: Path,
    title: str = "משוב בדיקה",
) -> Path:
    """
    MAIN OUTPUT: feedback.tex (no PDF).
    Writes:
      - out_dir/feedback.tex
      - out_dir/debug_json_sanitize.txt (if sanitizer exists)
    Returns: Path to feedback.tex
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = json.loads(grades_json.read_text(encoding="utf-8"))
    try:
        sanitized, st = sanitize_grades_json(raw)  # optional but helpful
        (out_dir / "debug_json_sanitize.txt").write_text(str(st), encoding="utf-8")
        grades = sanitized
    except Exception:
        # if sanitizer not present / fails, just proceed
        grades = raw

    tex = render_feedback_tex(grades, title=title)

    tex_path = out_dir / "feedback.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path