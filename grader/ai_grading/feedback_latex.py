# grader/feedbab_tex.py
from __future__ import annotations

import json
import re
from pathlib import Path

from grader.ai_grading.math_normalize import normalize_math_text
from grader.compile_tex import compile_tex_to_pdf

# -------------------------
# Regex helpers
# -------------------------

# Convert display math \[...\] into $$...$$ before we do mixed escaping
_DISPLAY_MATH_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)

# Tokenize math chunks so we can escape only the surrounding text.
# group(0) is either $$...$$ or $...$ (non-greedy); DOTALL allows newlines inside.
_MATH_TOKEN_RE = re.compile(r"\$\$.*?\$\$|\$.*?\$", re.DOTALL)


def _latex_escape_plain(s: str) -> str:
    """Escape plain text for LaTeX (no LaTeX commands/math expected)."""
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
    escaped = "".join(out)
    escaped = escaped.replace("\n\n", r"\par ").replace("\n", r"\newline ")
    return escaped


def _latex_escape_text_only(s: str) -> str:
    """Escape LaTeX special chars for TEXT MODE only (no math expected)."""
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
    escaped = "".join(out)
    escaped = escaped.replace("\n\n", r"\par ").replace("\n", r"\newline ")
    return escaped


def _sanitize_math_block(m: str) -> str:
    """
    Keep math intact but undo text-escaping artifacts inside math, and prevent comments.
    """
    if not m:
        return m

    is_display = m.startswith("$$")
    inner = m[2:-2] if is_display else m[1:-1]

    # If braces were escaped like text, undo in math
    inner = inner.replace(r"\{", "{").replace(r"\}", "}")

    # If our escaping produced \textbackslash{} inside math, undo it
    inner = inner.replace(r"\textbackslash{}", "\\")

    # Comments inside math can blow compilation
    inner = inner.replace("%", r"\%")

    return ("$$" + inner + "$$") if is_display else ("$" + inner + "$")


def _escape_mixed_keep_math(s: str) -> str:
    """
    Escape text outside math; keep math blocks as-is (with a bit of sanitization).
    """
    if s is None:
        return ""

    # Reduce obvious AI glitch: $$$$ or $$$ -> $$ (the robust cleaner will also handle this later)
    s = re.sub(r"(?<!\\)\${3,}", "$$", s)

    # If odd number of unescaped $ remain, bail to full text-escape (prevents xelatex crashes)
    dollar_count = re.sub(r"\\\$", "", s).count("$")
    if dollar_count % 2 == 1:
        return _latex_escape_text_only(s)

    parts: list[str] = []
    last = 0

    for m in _MATH_TOKEN_RE.finditer(s):
        parts.append(_latex_escape_text_only(s[last:m.start()]))
        parts.append(_sanitize_math_block(m.group(0)))
        last = m.end()

    parts.append(_latex_escape_text_only(s[last:]))
    return "".join(parts)


def _fmt_ai(s: str) -> str:
    """
    Normalize math-like fragments in AI text and keep them renderable.
      - Normalize math-ish text via normalize_math_text
      - Convert \[...\] to $$...$$
      - Escape only outside math ($...$ / $$...$$)
    Note: Final robust safety cleaning is done in compile_tex_to_pdf (tex_cleaner.clean_tex).
    """
    t = normalize_math_text(s or "")

    # Common AI junk: $$$...$$, $$$$...
    t = re.sub(r"(?<!\\)\${3,}", "$$", t)

    # Convert display math \[...\] -> $$...$$
    t = _DISPLAY_MATH_RE.sub(lambda m: "$$" + m.group(1) + "$$", t)

    return _escape_mixed_keep_math(t)


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
    lines.append(r"\usepackage{amsmath,amssymb,mathtools}")
    lines.append(r"\usepackage{bidi}")
    lines.append(r"\setdefaultlanguage{hebrew}")
    lines.append(r"\setotherlanguage{english}")
    lines.append(r"\setlength{\parindent}{0pt}")
    lines.append(r"\begin{document}")

    # Header (plain)
    lines.append(rf"{{\LARGE {_latex_escape_plain(title)}\par}}")
    lines.append(r"\vspace{0.3cm}")
    lines.append(rf"{{\large ציון כולל: {total_score:.1f} / {total_max:.1f}\par}}")
    lines.append(r"\vspace{0.6cm}")

    for q in data.get("questions", []):
        qid = _latex_escape_plain(str(q.get("qid", "Q?")))
        score = float(q.get("score", 0.0))
        max_points = float(q.get("max_points", 0.0))

        summary = _fmt_ai(q.get("summary", ""))

        good = q.get("what_was_correct", []) or []
        mistakes = q.get("main_mistakes", []) or []
        improve = q.get("how_to_improve", []) or []
        mismatch = q.get("mismatch", {}) or {}
        tags = q.get("common_errors_detected", []) or []
        next_step = q.get("suggested_next_step_he", "") or ""
        conf = q.get("confidence", None)

        lines.append(r"\hrule\vspace{0.35cm}")
        lines.append(rf"{{\Large {qid} \ \ \ ({score:.1f}/{max_points:.1f})\par}}")

        # Mismatch warning
        try:
            is_mismatch = bool(mismatch.get("is_mismatch"))
        except Exception:
            is_mismatch = False

        if is_mismatch:
            lines.append(r"\vspace{0.15cm}")
            lines.append(r"\textbf{\Large \color{red}פתירת שאלה אחרת / אי־התאמה:}")

            expl = _fmt_ai(str(mismatch.get("explanation_he", "") or ""))
            ref_t = _fmt_ai(str(mismatch.get("reference_target", "") or ""))
            stu_t = _fmt_ai(str(mismatch.get("student_target", "") or ""))

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
                lines.append(rf"\item {_fmt_ai(str(item))}")
            lines.append(r"\end{itemize}")

        if mistakes:
            lines.append(r"\textbf{טעויות / חלקים חסרים:}")
            lines.append(r"\begin{itemize}[leftmargin=*,itemsep=0.2em]")
            for item in mistakes[:12]:
                lines.append(rf"\item {_fmt_ai(str(item))}")
            lines.append(r"\end{itemize}")

        if improve:
            lines.append(r"\textbf{איך להשתפר לפעם הבאה:}")
            lines.append(r"\begin{itemize}[leftmargin=*,itemsep=0.2em]")
            for item in improve[:12]:
                lines.append(rf"\item {_fmt_ai(str(item))}")
            lines.append(r"\end{itemize}")

        if next_step:
            lines.append(r"\textbf{צעד מומלץ עכשיו:}")
            lines.append(_fmt_ai(str(next_step)) + r"\par")

        if tags:
            tag_line = ", ".join(str(t) for t in tags[:12])
            lines.append(r"\textit{תגיות: " + _latex_escape_plain(tag_line) + r"}\par")

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

    # compile_tex_to_pdf now runs robust cleaning (tex_cleaner.clean_tex) + Windows font stabilization
    outputs = compile_tex_to_pdf(
        tex_path,
        out_dir=out_dir,
        clean=True,
        font_name=font_name,
    )
    return outputs.pdf
