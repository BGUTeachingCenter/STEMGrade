# grader/report_builder.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from grader.subprocess_utils import run
from grader.tex_format import rtl_lr_mix_to_tex, escape_tex_plain


def build_report_tex(
    model: str,
    grade: dict[str, Any],
    total_score: int,
    prompts_txt_path: Path,
    clean_pdf_filename: str = "012_clean.pdf",
) -> str:
    """
    Build the LaTeX report content.

    Uses Calibri for text. Avoids unicode-math (since it caused font lookup errors on MiKTeX).
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    items = grade.get("items", []) or []
    assumptions = rtl_lr_mix_to_tex(grade.get("assumptions", "") or "")

    prompts_path_tex = escape_tex_plain(str(prompts_txt_path).replace("\\", "/"))

    lines: list[str] = []
    lines.append(r"\documentclass[12pt]{article}" "\n")
    lines.append(r"\usepackage{fontspec}" "\n")
    lines.append(r"\usepackage{bidi}" "\n")
    lines.append(r"\usepackage{biditools}" "\n")
    lines.append(r"\usepackage{amsmath}" "\n")
    lines.append(r"\usepackage{amssymb}" "\n")
    lines.append(r"\usepackage[a4paper,margin=2cm]{geometry}" "\n")
    lines.append(r"\usepackage{longtable}" "\n")
    lines.append(r"\usepackage{hyperref}" "\n")
    lines.append(r"\usepackage{pdfpages}" "\n")

    # Calibri (Windows font). If it fails, change to Arial/DejaVu Sans.
    lines.append(r"\setmainfont{Calibri}" "\n")

    lines.append(r"\setRTL" "\n")
    # bidi may already define \LR; use providecommand to avoid errors.
    lines.append(r"\providecommand{\LR}[1]{\beginL #1\endL}" "\n")
    lines.append(r"\begin{document}" "\n\n")

    lines.append(r"{\Large \textbf{דו''ח בדיקה – שיעורי בית}}\par" "\n")
    lines.append(r"\vspace{0.3cm}" "\n")
    lines.append(r"\textbf{תאריך:} " + rtl_lr_mix_to_tex(now) + r"\par" "\n")
    lines.append(r"\textbf{מודל:} " + rtl_lr_mix_to_tex(model) + r"\par" "\n")
    lines.append(r"\textbf{ציון סופי (מחושב):} " + rtl_lr_mix_to_tex(f"{total_score}/100") + r"\par" "\n")
    lines.append(r"\vspace{0.2cm}" "\n")
    lines.append(r"\textbf{קובץ פרומפטים להרצה:} \LR{" + prompts_path_tex + r"}\par" "\n")
    lines.append(r"\vspace{0.3cm}" "\n")

    lines.append(r"\textbf{הנחות:}\par" "\n")
    lines.append(assumptions + r"\par" "\n")
    lines.append(r"\vspace{0.6cm}" "\n")

    lines.append(r"{\Large \textbf{פירוט ציונים ומשוב}}\par" "\n")
    lines.append(r"\vspace{0.3cm}" "\n")
    lines.append(r"\begin{longtable}{|p{0.26\textwidth}|p{0.08\textwidth}|p{0.58\textwidth}|}" "\n")
    lines.append(r"\hline" "\n")
    lines.append(r"\textbf{סעיף} & \textbf{ציון} & \textbf{פירוט} \\" "\n")
    lines.append(r"\hline" "\n")
    lines.append(r"\endfirsthead" "\n")
    lines.append(r"\hline" "\n")
    lines.append(r"\textbf{סעיף} & \textbf{ציון} & \textbf{פירוט} \\" "\n")
    lines.append(r"\hline" "\n")
    lines.append(r"\endhead" "\n")

    for it in items:
        title = rtl_lr_mix_to_tex(it.get("title", "") or "")
        score = str(int(it.get("score", 0) or 0))

        question = rtl_lr_mix_to_tex(it.get("question", "") or "")
        student_answer = rtl_lr_mix_to_tex(it.get("student_answer", "") or "")
        feedback = rtl_lr_mix_to_tex(it.get("feedback_he", "") or "")
        notes = rtl_lr_mix_to_tex(it.get("notes_for_teacher", "") or "")

        errors = it.get("errors", []) or []
        err_block = ""
        if errors:
            err_lines = [rtl_lr_mix_to_tex(str(e)) for e in errors]
            err_block = r"\textbf{נקודות לתיקון:} " + r" \newline ".join(err_lines)

        cell = (
            r"\textbf{השאלה:} " + question + r"\newline "
            r"\textbf{תשובת התלמיד/ה:} " + student_answer + r"\newline "
            r"\textbf{משוב:} " + feedback
        )
        if err_block:
            cell += r"\newline " + err_block
        if notes.strip():
            cell += r"\newline " + r"\textbf{הערות למורה:} " + notes

        lines.append(f"{title} & {score} & {cell} \\\\" "\n")
        lines.append(r"\hline" "\n")

    lines.append(r"\end{longtable}" "\n")
    lines.append(r"\newpage" "\n")

    lines.append(r"{\Large \textbf{הגשה מקורית (PDF כפי שהודפס מ-LaTeX)}}\par" "\n")
    lines.append(r"\vspace{0.3cm}" "\n")
    lines.append(r"\includepdf[pages=-]{" + clean_pdf_filename + r"}" "\n")

    lines.append(r"\end{document}" "\n")
    return "".join(lines)


def compile_tex_to_pdf(tex_path: Path, build_dir: Path) -> None:
    """Compile TeX twice for refs/PageLabels stability."""
    run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={build_dir}", str(tex_path)])
    run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={build_dir}", str(tex_path)])
