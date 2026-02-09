# grader/answer_render.py
"""Render student answer snippets to standalone PDFs.

Each extracted answer snippet is compiled into its own small PDF. Those PDFs are
then embedded into the final bundle using LaTeX's `pdfpages`.

Compiling per-snippet keeps failures isolated: if one answer doesn't compile,
we still produce the rest of the bundle and mark the failing part in red.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import traceback
from datetime import datetime

from .compile_tex import compile_tex_to_pdf
from .reference_ranges import Key


def make_answer_tex(qnum: int, part: str, answer_latex: str, font_name: str) -> str:
    """Create a minimal XeLaTeX document for a single answer snippet.

    We intentionally keep the preamble small and provide lightweight macro
    fallbacks for common student commands (\abs, \norm, etc.).

    Args:
        qnum: Question number.
        part: Part letter ("א"/"ב"/"").
        answer_latex: The raw latex snippet extracted from the student's file.
        font_name: Font used for Hebrew RTL typesetting.

    Returns:
        Full LaTeX source for a compilable document.
    """

    title = f"Student Answer — Q{qnum}" + (f"({part})" if part else "")

    # IMPORTANT: use real newlines ("\n"), not the literal two characters "\\n".
    # If you write literal "\\n" into a .tex file, XeLaTeX will see a command \n
    # and fail with: "Undefined control sequence ... \n".
    return (
        "\\documentclass[12pt]{article}\n"
        "\\usepackage[a4paper,margin=2cm]{geometry}\n"
        "\\usepackage{amsmath,amssymb,mathtools}\n"
        "% ---- Common student macros (robust defaults) ----\n"
        "\\providecommand{\\abs}[1]{\\left\\lvert#1\\right\\rvert}\n"
        "\\providecommand{\\norm}[1]{\\left\\lVert#1\\right\\rVert}\n"
        "\\providecommand{\\ceil}[1]{\\left\\lceil#1\\right\\rceil}\n"
        "\\providecommand{\\floor}[1]{\\left\\lfloor#1\\right\\rfloor}\n"
        "\\providecommand{\\set}[1]{\\left\\{#1\\right\\}}\n"
        "\\providecommand{\\paren}[1]{\\left(#1\\right)}\n"
        "\\providecommand{\\bracks}[1]{\\left[#1\\right]}\n"
        "\\providecommand{\\angles}[1]{\\left\\langle#1\\right\\rangle}\n"
        "\\providecommand{\\RR}{\\mathbb{R}}\n"
        "\\providecommand{\\CC}{\\mathbb{C}}\n"
        "\\providecommand{\\NN}{\\mathbb{N}}\n"
        "\\providecommand{\\ZZ}{\\mathbb{Z}}\n"
        "\\providecommand{\\QQ}{\\mathbb{Q}}\n"
        "\\usepackage{xcolor}\n"
        "\\usepackage{fontspec}\n"
        "\\usepackage{bidi}\n"
        f"\\setmainfont[Script=Hebrew]{{{font_name}}}\n"
        f"\\setmonofont{{{font_name}}}\n"
        "\\setRTL\n"
        "\\setlength{\\parskip}{0.6em}\n"
        "\\setlength{\\parindent}{0pt}\n"
        "\\begin{document}\n"
        f"\\section*{{{title}}}\n"
        "\\begingroup\n"
        + answer_latex.strip()
        + "\n"
        "\\endgroup\n"
        "\\end{document}\n"
    )

def _save_answer_compile_error(ans_dir: Path, stem: str, exc: Exception) -> None:
    dbg = ans_dir / "debug_answer_compile"
    dbg.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (dbg / f"{stem}_{ts}.traceback.txt").write_text(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        encoding="utf-8",
    )


def compile_student_answer_pdfs(
    student_answers: Dict[Key, str],
    out_dir: Path,
    font_name: str,
) -> Dict[Key, Optional[Path]]:
    """Compile each answer snippet into its own PDF.

    Args:
        student_answers: Mapping (qnum, part) -> LaTeX snippet.
        out_dir: Base output directory.
        font_name: Font used for Hebrew RTL typesetting.

    Returns:
        Mapping (qnum, part) -> compiled PDF path, or None if that part failed.
    """

    ans_dir = out_dir / "student_answers"
    ans_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[Key, Optional[Path]] = {}
    for (qnum, part), snippet in student_answers.items():
        stem = f"student_ans_Q{qnum}" + (f"_{part}" if part else "")
        tex_path = ans_dir / f"{stem}.tex"
        pdf_path = ans_dir / f"{stem}.pdf"

        tex_path.write_text(make_answer_tex(qnum, part, snippet, font_name), encoding="utf-8")

        # inside the loop:
        try:
            compiled = compile_tex_to_pdf(tex_path, ans_dir, clean=False, font_name=font_name, passes=2)
            produced = compiled.pdf
            if produced.exists() and produced != pdf_path:
                try:
                    produced.replace(pdf_path)
                    produced = pdf_path
                except Exception:
                    pass
            out[(qnum, part)] = produced if produced.exists() else None
        except Exception as e:
            _save_answer_compile_error(ans_dir, stem, e)
            out[(qnum, part)] = None

    return out
