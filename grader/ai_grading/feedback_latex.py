from __future__ import annotations

import json
from pathlib import Path

from grader.compile_tex import compile_tex_to_pdf


def _latex_escape(s: str) -> str:
    """Escape plain text for LaTeX."""
    if s is None:
        return ""
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
    return "".join(out).replace("\n", r"\\ ")


def _render_feedback_tex(data: dict, title: str) -> str:
    total_score = float(data.get("total_score", 0.0))
    total_max = float(data.get("total_max", 0.0))

    lines: list[str] = []
    lines.append(r"\documentclass[12pt]{article}")
    lines.append(r"\usepackage[a4paper,margin=2cm]{geometry}")
    lines.append(r"\usepackage{fontspec}")  # XeLaTeX
    lines.append(r"\setmainfont{Arial}")
    lines.append(r"\usepackage{microtype}")
    lines.append(r"\usepackage{hyperref}")
    lines.append(r"\usepackage{enumitem}")
    lines.append(r"\usepackage{polyglossia}")
    lines.append(r"\usepackage{bidi}")  # robust RTL/LTR handling with XeLaTeX
    lines.append(r"\setdefaultlanguage{hebrew}")
    lines.append(r"\setotherlanguage{english}")
    lines.append(r"\setlength{\parindent}{0pt}")
    lines.append(r"\begin{document}")

    # Header
    lines.append(rf"{{\LARGE {_latex_escape(title)}\par}}")
    lines.append(r"\vspace{0.3cm}")
    lines.append(rf"{{\large ציון כולל: {total_score:.1f} / {total_max:.1f}\par}}")
    lines.append(r"\vspace{0.6cm}")

    for q in data.get("questions", []):
        qid = _latex_escape(str(q.get("qid", "Q?")))
        score = float(q.get("score", 0.0))
        max_points = float(q.get("max_points", 0.0))
        summary = _latex_escape(q.get("summary", ""))

        good = q.get("what_was_correct", []) or []
        mistakes = q.get("main_mistakes", []) or []
        improve = q.get("how_to_improve", []) or []
        conf = q.get("confidence", None)

        lines.append(r"\hrule\vspace{0.35cm}")
        lines.append(rf"{{\Large {qid} \ \ \ ({score:.1f}/{max_points:.1f})\par}}")
        if summary:
            lines.append(r"\vspace{0.2cm}")
            lines.append(summary + r"\par")
        lines.append(r"\vspace{0.2cm}")

        if good:
            lines.append(r"\textbf{מה עשית נכון:}")
            lines.append(r"\begin{itemize}[leftmargin=*,itemsep=0.2em]")
            for item in good[:12]:
                lines.append(rf"\item {_latex_escape(str(item))}")
            lines.append(r"\end{itemize}")

        if mistakes:
            lines.append(r"\textbf{טעויות / חלקים חסרים:}")
            lines.append(r"\begin{itemize}[leftmargin=*,itemsep=0.2em]")
            for item in mistakes[:12]:
                lines.append(rf"\item {_latex_escape(str(item))}")
            lines.append(r"\end{itemize}")

        if improve:
            lines.append(r"\textbf{איך להשתפר לפעם הבאה:}")
            lines.append(r"\begin{itemize}[leftmargin=*,itemsep=0.2em]")
            for item in improve[:12]:
                lines.append(rf"\item {_latex_escape(str(item))}")
            lines.append(r"\end{itemize}")

        if conf is not None:
            lines.append(rf"\textit{{רמת ביטחון: {float(conf):.2f}}}\par")

        lines.append(r"\vspace{0.2cm}")

    lines.append(r"\end{document}")
    return "\n".join(lines)


def build_feedback_pdf_from_grades_latex(
    grades_json: Path,
    out_dir: Path,
    title: str = "משוב בדיקה",
    font_name: str = "Arial",
) -> Path:
    """
    Build feedback PDF via LaTeX (XeLaTeX) so Hebrew/RTL is readable.

    Returns path to generated PDF.
    """
    data = json.loads(grades_json.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tex_path = out_dir / "graded_feedback.tex"
    tex_path.write_text(_render_feedback_tex(data, title=title), encoding="utf-8")

    outputs = compile_tex_to_pdf(tex_path, out_dir=out_dir, clean=True, font_name=font_name)
    return outputs.pdf
