from __future__ import annotations

import re
from typing import List

from .math_normalize import normalize_math_text


_MATH_SPLIT_RE = re.compile(r"(\$\$.*?\$\$|\$.*?\$)", re.DOTALL)


def escape_text_only(s: str) -> str:
    """
    Escape plain text for LaTeX (NOT math).
    Newlines become paragraphs/spaces for readability.
    """
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
    escaped = "".join(out)

    escaped = escaped.replace("\n\n", r"\par ").replace("\n", " ")
    return escaped


def escape_math_only(s: str) -> str:
    """
    Escape only characters that commonly break LaTeX inside math.
    Keep backslashes intact.
    """
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.replace("%", r"\%").replace("#", r"\#")


def latex_render_mixed(s: str) -> str:
    """
    Render a string that may contain $...$ or $$...$$.
    Text outside math is escaped; math content is preserved.
    """
    if s is None:
        return ""

    parts = _MATH_SPLIT_RE.split(s)
    rendered: List[str] = []

    for part in parts:
        if not part:
            continue
        if part.startswith("$$") and part.endswith("$$"):
            rendered.append("$$" + escape_math_only(part[2:-2]) + "$$")
        elif part.startswith("$") and part.endswith("$"):
            rendered.append("$" + escape_math_only(part[1:-1]) + "$")
        else:
            rendered.append(escape_text_only(part))

    return "".join(rendered)


def _render_text(s: str) -> str:
    """
    One place to apply: normalize math-ish text -> render with LaTeX-safe escaping.
    """
    return latex_render_mixed(normalize_math_text(s or ""))


def render_feedback_tex(bundle, original_pdf_filename: str) -> str:
    """
    Produce a LaTeX feedback document (Hebrew RTL) with readable layout.
    """
    def bullets(items: List[str], empty_text: str, limit: int = 8) -> List[str]:
        items = [str(x).strip() for x in (items or []) if str(x).strip()]
        if not items:
            return [rf"\item {escape_text_only(empty_text)}"]
        return [rf"\item {_render_text(x)}" for x in items[:limit]]

    def render_tags(tags: List[str]) -> str:
        tags = [t.strip() for t in (tags or []) if t and t.strip()]
        if not tags:
            return escape_text_only("תגיות: —")
        joined = ", ".join(tags)
        return (
            r"\textbf{תגיות:} "
            + r"\begin{english}\texttt{"
            + escape_text_only(joined)
            + r"}\end{english}"
        )

    def render_mismatch_box(m: dict) -> List[str]:
        if not isinstance(m, dict) or not m.get("is_mismatch"):
            return []
        ref_t = str(m.get("reference_target") or "").strip()
        stu_t = str(m.get("student_target") or "").strip()
        expl = str(m.get("explanation_he") or "").strip()

        # Normalize math-ish text first (so targets can become $...$)
        ref_t = normalize_math_text(ref_t)
        stu_t = normalize_math_text(stu_t)

        box: List[str] = []
        box.append(r"\begin{tcolorbox}[colback=black!3,colframe=black!40,title={אי־התאמה: פתרון שאלה אחרת}]")
        if expl:
            box.append(_render_text(expl) + r"\par")

        # For targets, prefer display math if they look math-heavy, otherwise LTR monospace.
        def render_target(label: str, t: str) -> List[str]:
            if not t:
                return []
            out: List[str] = []
            out.append(rf"\textbf{{{escape_text_only(label)}}}\par")
            # If it contains $...$ after normalization, show as-is (it will render math)
            if "$" in t or r"\int" in t or r"\sum" in t or r"\lim" in t:
                # Put in English block to avoid RTL punctuation issues; keep LaTeX math rendering
                out.append(r"\begin{english}" + _render_text(t) + r"\end{english}\par")
            else:
                out.append(r"\begin{english}\ttfamily " + escape_text_only(t) + r"\end{english}\par")
            return out

        box.extend(render_target("מה נדרש:", ref_t))
        box.extend(render_target("מה נפתר בפועל:", stu_t))

        box.append(r"\end{tcolorbox}")
        return box

    lines: List[str] = []
    lines.append(r"\documentclass[12pt]{article}")
    lines.append(r"\usepackage[a4paper,margin=2cm]{geometry}")
    lines.append(r"\usepackage{fontspec}")  # XeLaTeX
    lines.append(r"\setmainfont{Arial}")
    lines.append(r"\usepackage{microtype}")
    lines.append(r"\usepackage{hyperref}")
    lines.append(r"\usepackage{enumitem}")
    lines.append(r"\usepackage{polyglossia}")
    lines.append(r"\setdefaultlanguage{hebrew}")
    lines.append(r"\setotherlanguage{english}")
    lines.append(r"\usepackage{xcolor}")
    lines.append(r"\usepackage[most]{tcolorbox}")

    lines.append(r"\setlength{\parindent}{0pt}")
    lines.append(r"\setlength{\parskip}{6pt}")
    lines.append(r"\setlist[itemize]{leftmargin=*,itemsep=2pt,topsep=2pt}")

    lines.append(r"\begin{document}")
    lines.append(r"\section*{משוב בדיקה}")
    lines.append(_render_text(f"שם קובץ: {original_pdf_filename}"))

    if getattr(bundle, "total_max", 0) and bundle.total_max > 0:
        lines.append(_render_text(f"ציון כולל: {bundle.total_score:.1f} / {bundle.total_max:.1f}"))

    for q in bundle.question_grades:
        lines.append(r"\hrule\medskip")
        lines.append(rf"\subsection*{{{escape_text_only(q.qid)} \ \ \ ({q.score:.1f}/{q.max_points:.1f})}}")

        if q.summary and str(q.summary).strip():
            lines.append(_render_text(q.summary))

        lines.extend(render_mismatch_box(getattr(q, "mismatch", {}) or {}))

        lines.append(r"\textbf{מה עשית נכון:}")
        lines.append(r"\begin{itemize}")
        lines.extend(bullets(q.what_was_correct, empty_text="לא צוינו נקודות חיוביות."))
        lines.append(r"\end{itemize}")

        lines.append(r"\textbf{טעויות / חלקים חסרים:}")
        lines.append(r"\begin{itemize}")
        lines.extend(bullets(q.main_mistakes, empty_text="לא זוהו טעויות משמעותיות."))
        lines.append(r"\end{itemize}")

        lines.append(r"\textbf{איך להשתפר לפעם הבאה:}")
        lines.append(r"\begin{itemize}")
        lines.extend(bullets(q.how_to_improve, empty_text="המשך/י לתרגל ולכתוב בצורה מסודרת."))
        lines.append(r"\end{itemize}")

        if getattr(q, "suggested_next_step_he", "") and str(q.suggested_next_step_he).strip():
            lines.append(r"\begin{tcolorbox}[colback=black!2,colframe=black!25,title={צעד מומלץ עכשיו}]")
            lines.append(_render_text(q.suggested_next_step_he))
            lines.append(r"\end{tcolorbox}")

        lines.append(render_tags(getattr(q, "common_errors_detected", []) or []))
        lines.append(escape_text_only(f"רמת ביטחון: {q.confidence:.2f}"))
        lines.append(r"\medskip")

    lines.append(r"\end{document}")
    return "\n".join(lines)
