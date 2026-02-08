"""
Simplified feedback PDF generation focused only on feedback rendering.
All LaTeX cleaning is delegated to tex_cleanup.py.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

from grader.compile_tex import compile_tex_to_pdf


def escape_for_latex(text: str) -> str:
    """
    Simple text escaping that preserves Hebrew and other Unicode text.
    Only escapes truly problematic characters for LaTeX.
    """
    if not text:
        return ""

    # Only escape characters that will break LaTeX compilation
    # Do NOT escape Hebrew characters or other Unicode
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",  # Only if not already a LaTeX command
    }

    result = text

    # Handle backslashes carefully - don't escape if it's already a LaTeX command
    if "\\" in result:
        # Only escape backslashes that aren't part of LaTeX commands
        # This is a simple heuristic - tex_cleanup.py will handle complex cases
        result = re.sub(r"\\(?![a-zA-Z])", r"\\textbackslash{}", result)

    # Apply other replacements
    for char, escape in replacements.items():
        if char != "\\":  # Already handled above
            result = result.replace(char, escape)

    # Handle newlines but preserve paragraphs
    result = result.replace("\n\n", r"\par ")
    result = result.replace("\n", r" ")

    return result


def preserve_hebrew_in_latex(text: str) -> str:
    """
    Ensure Hebrew text is properly preserved for XeLaTeX.
    XeLaTeX with fontspec and bidi should handle Hebrew automatically.
    """
    if not text:
        return ""

    # Just escape the essential LaTeX special characters
    # Let XeLaTeX handle Hebrew rendering with fontspec + bidi
    essential_escapes = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
    }

    result = text
    for char, escape in essential_escapes.items():
        result = result.replace(char, escape)

    return result


def render_feedback_tex(data: dict, title: str = "משוב בדיקה") -> str:
    """Generate LaTeX source for feedback PDF with proper Hebrew support."""
    total_score = float(data.get("total_score", 0.0))
    total_max = float(data.get("total_max", 0.0))

    lines = [
        r"\documentclass[12pt]{article}",
        r"\usepackage[a4paper,margin=2cm]{geometry}",
        r"\usepackage{fontspec}",
        r"\setmainfont{Arial}",  # This will be updated by tex_cleanup.py
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
        "",
        # Use Hebrew title as-is - XeLaTeX will handle it
        rf"{{\LARGE {title}\par}}",
        r"\vspace{0.3cm}",
        rf"{{\large ציון כולל: {total_score:.1f} / {total_max:.1f}\par}}",
        r"\vspace{0.6cm}",
        "",
    ]

    for q in data.get("questions", []):
        # Preserve original question ID - don't translate
        qid = str(q.get("qid", "Q?"))
        score = float(q.get("score", 0.0))
        max_points = float(q.get("max_points", 0.0))

        lines.extend([
            r"\hrule\vspace{0.35cm}",
            rf"{{\Large {preserve_hebrew_in_latex(qid)} ({score:.1f}/{max_points:.1f})\par}}",
            r"\vspace{0.2cm}",
        ])

        # Summary - preserve Hebrew
        summary = q.get("summary", "")
        if summary:
            lines.append(preserve_hebrew_in_latex(str(summary)) + r"\par")
            lines.append(r"\vspace{0.2cm}")

        # Mismatch warning (if present)
        mismatch = q.get("mismatch", {}) or {}
        try:
            is_mismatch = bool(mismatch.get("is_mismatch", False))
        except:
            is_mismatch = False

        if is_mismatch:
            lines.extend([
                r"\vspace{0.2cm}",
                r"{\color{red}\textbf{התאמה: פתרת שאלה אחרת}}",
                r"\vspace{0.1cm}",
            ])

            explanation = mismatch.get("explanation_he", "")
            if explanation:
                lines.append(preserve_hebrew_in_latex(str(explanation)) + r"\par")

        # What was correct - preserve Hebrew
        good = q.get("what_was_correct", []) or []
        if good:
            lines.append(r"\textbf{מה עשית נכון:}")
            lines.append(r"\begin{itemize}[rightmargin=2cm]")
            for item in good[:10]:
                if item:  # Skip empty items
                    lines.append(rf"\item {preserve_hebrew_in_latex(str(item))}")
            lines.append(r"\end{itemize}")

        # Mistakes - preserve Hebrew
        mistakes = q.get("main_mistakes", []) or []
        if mistakes:
            lines.append(r"\textbf{טעויות עיקריות:}")
            lines.append(r"\begin{itemize}[rightmargin=2cm]")
            for item in mistakes[:10]:
                if item:  # Skip empty items
                    lines.append(rf"\item {preserve_hebrew_in_latex(str(item))}")
            lines.append(r"\end{itemize}")

        # How to improve - preserve Hebrew
        improve = q.get("how_to_improve", []) or []
        if improve:
            lines.append(r"\textbf{איך להשתפר:}")
            lines.append(r"\begin{itemize}[rightmargin=2cm]")
            for item in improve[:10]:
                if item:  # Skip empty items
                    lines.append(rf"\item {preserve_hebrew_in_latex(str(item))}")
            lines.append(r"\end{itemize}")

        # Next step - preserve Hebrew
        next_step = q.get("suggested_next_step_he", "")
        if next_step:
            lines.append(r"\textbf{צעד מומלץ הבא:}")
            lines.append(preserve_hebrew_in_latex(str(next_step)) + r"\par")

        # Common errors tags
        tags = q.get("common_errors_detected", []) or []
        if tags:
            tag_text = ", ".join(str(t) for t in tags[:5])  # Limit number of tags
            lines.append(rf"\textit{{תגיות: {preserve_hebrew_in_latex(tag_text)}}}\par")

        lines.append(r"\vspace{0.4cm}")

    lines.extend([
        "",
        r"\end{document}"
    ])

    return "\n".join(lines)


def build_feedback_pdf(
    grades_json: Path,
    out_dir: Path,
    title: str = "משוב בדיקה",
    font_name: str = "Arial",
) -> Path:
    """
    Build feedback PDF from grades JSON.
    Uses tex_cleanup.py for robust LaTeX cleaning while preserving Hebrew.
    """
    data = json.loads(grades_json.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tex_content = render_feedback_tex(data, title)
    tex_path = out_dir / "feedback.tex"
    tex_path.write_text(tex_content, encoding="utf-8")

    # Compile with robust cleaning enabled
    outputs = compile_tex_to_pdf(
        tex_path,
        out_dir,
        clean=True,
        font_name=font_name,
    )

    return outputs.pdf


# Backwards compatibility
def build_feedback_pdf_from_grades_latex(*args, **kwargs):
    """Backwards compatible alias."""
    return build_feedback_pdf(*args, **kwargs)