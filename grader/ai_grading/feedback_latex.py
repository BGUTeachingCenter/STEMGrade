from __future__ import annotations

import json
from pathlib import Path

from grader.compile_tex import compile_tex_to_pdf


import re
from typing import Optional, List

_MATH_ENV_NAMES = (
    "equation", "equation*", "align", "align*", "gather", "gather*",
    "multline", "multline*", "eqnarray", "eqnarray*",
)

def _build_math_env_regex() -> str:
    # Build: \\begin{align} ... \\end{align}  | \\begin{align*} ... \\end{align*} | ...
    alts = []
    for env in _MATH_ENV_NAMES:
        env_esc = re.escape(env)
        alts.append(rf"\\begin\{{{env_esc}\}}[\s\S]*?\\end\{{{env_esc}\}}")
    return r"(?:%s)" % "|".join(alts)

_MATH_SEGMENT_RE = re.compile(
    r"("
    r"\\\[[\s\S]*?\\\]"          # \[ ... \]
    r"|\\\([\s\S]*?\\\)"         # \( ... \)
    r"|\$\$[\s\S]*?\$\$"         # $$ ... $$
    r"|\$(?:\\.|[^\$\\])+\$"     # $ ... $ (simple)
    r"|" + _build_math_env_regex() +
    r")",
    re.MULTILINE
)

def _escape_text_only(text: str) -> str:
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
    out = [repl.get(ch, ch) for ch in text]
    escaped = "".join(out)
    escaped = escaped.replace("\n\n", r"\par ").replace("\n", r"\newline ")
    return escaped

def _normalize_math(math: str) -> str:
    # Remove accidental newlines markers inside math
    math = math.replace(r"\n", " ").replace("\n", " ")

    # Fix common bogus commands produced by LLMs
    # \intx -> \int x
    math = re.sub(r"\\int(?=[A-Za-z0-9\\])", r"\\int ", math)

    # Remove \nx / \ny etc. that are not LaTeX (often "newline x")
    math = re.sub(r"\\n(?=[A-Za-z])", "", math)

    # Normalize operators
    math = re.sub(r"(?<!\\)\bln\b", r"\\ln", math)
    math = re.sub(r"(?<!\\)\bcos\b", r"\\cos", math)
    math = re.sub(r"(?<!\\)\bsin\b", r"\\sin", math)

    # Add thin space before dx
    math = re.sub(r"(?<!\\)\bdx\b", r"\\,dx", math)

    return math


def latex_escape_preserve_math(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    parts: List[str] = []
    last = 0
    for m in _MATH_SEGMENT_RE.finditer(s):
        before = s[last:m.start()]
        if before:
            parts.append(_escape_text_only(before))
        math_seg = s[m.start():m.end()]
        parts.append(_normalize_math(math_seg))
        last = m.end()

    tail = s[last:]
    if tail:
        parts.append(_escape_text_only(tail))

    return "".join(parts)


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
    lines.append(r"\usepackage{xcolor}")
    lines.append(r"\usepackage{bidi}")  # robust RTL/LTR handling with XeLaTeX
    lines.append(r"\setdefaultlanguage{hebrew}")
    lines.append(r"\setotherlanguage{english}")
    lines.append(r"\setlength{\parindent}{0pt}")
    lines.append(r"\begin{document}")

    # Header
    lines.append(rf"{{\LARGE {latex_escape_preserve_math(title)}\par}}")
    lines.append(r"\vspace{0.3cm}")
    lines.append(rf"{{\large ציון כולל: {total_score:.1f} / {total_max:.1f}\par}}")
    lines.append(r"\vspace{0.6cm}")

    for q in data.get("questions", []):
        qid = latex_escape_preserve_math(str(q.get("qid", "Q?")))
        score = float(q.get("score", 0.0))
        max_points = float(q.get("max_points", 0.0))
        summary = latex_escape_preserve_math(q.get("summary", ""))

        good = q.get("what_was_correct", []) or []
        mistakes = q.get("main_mistakes", []) or []
        improve = q.get("how_to_improve", []) or []
        mismatch = q.get("mismatch", {}) or {}
        tags = q.get("common_errors_detected", []) or []
        next_step = q.get("suggested_next_step_he", "") or ""
        conf = q.get("confidence", None)

        lines.append(r"\hrule\vspace{0.35cm}")
        lines.append(rf"{{\Large {qid} \ \ \ ({score:.1f}/{max_points:.1f})\par}}")
        # Mismatch warning (if student solved a different problem)
        try:
            is_mismatch = bool(mismatch.get("is_mismatch"))
        except Exception:
            is_mismatch = False

        if is_mismatch:
            lines.append(r"\vspace{0.15cm}")
            lines.append(r"\textbf{\Large \color{red}(!) פתירת שאלה אחרת / אי־התאמה:}")
            expl = latex_escape_preserve_math(str(mismatch.get("explanation_he", "") or ""))
            ref_t = latex_escape_preserve_math(str(mismatch.get("reference_target", "") or ""))
            stu_t = latex_escape_preserve_math(str(mismatch.get("student_target", "") or ""))
            if expl:
                lines.append(expl + r"\par")
            if ref_t:
                lines.append(r"\textbf{מה נדרש:} " + ref_t + r"\par")
            if stu_t:
                lines.append(r"\textbf{מה נפתר בפועל:} " + stu_t + r"\par")
            lines.append(r"\vspace{0.2cm}")
        if summary:
            lines.append(r"\vspace{0.2cm}")
            lines.append(summary + r"\par")
        lines.append(r"\vspace{0.2cm}")

        if good:
            lines.append(r"\textbf{מה עשית נכון:}")
            lines.append(r"\begin{itemize}[leftmargin=*,itemsep=0.2em]")
            for item in good[:12]:
                lines.append(rf"\item {latex_escape_preserve_math(str(item))}")
            lines.append(r"\end{itemize}")

        if mistakes:
            lines.append(r"\textbf{טעויות / חלקים חסרים:}")
            lines.append(r"\begin{itemize}[leftmargin=*,itemsep=0.2em]")
            for item in mistakes[:12]:
                lines.append(rf"\item {latex_escape_preserve_math(str(item))}")
            lines.append(r"\end{itemize}")

        if improve:
            lines.append(r"\textbf{איך להשתפר לפעם הבאה:}")
            lines.append(r"\begin{itemize}[leftmargin=*,itemsep=0.2em]")
            for item in improve[:12]:
                lines.append(rf"\item {latex_escape_preserve_math(str(item))}")
            lines.append(r"\end{itemize}")

        if next_step:
            lines.append(r"\textbf{צעד מומלץ עכשיו:}")
            lines.append(latex_escape_preserve_math(str(next_step)) + r"\par")

        if tags:
            # Render tags on one line for quick scanning
            tag_line = ", ".join(str(t) for t in tags[:12])
            lines.append(r"\textit{תגיות: " + latex_escape_preserve_math(tag_line) + r"}\par")

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
